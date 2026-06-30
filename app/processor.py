from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable

from app.exporters import export_segments
from app.models import FileStatus, JobStatus
from app.storage import JobStorage
from app.transcription import TranscriptionPipeline


class ProgressHeartbeat:
    """在模型长时间无回调时，按阶段上限持续推进近似进度。"""

    def __init__(
        self,
        update_progress: Callable[[str, int], None],
        interval_seconds: float = 2.0,
        step: int = 1,
    ):
        self.update_progress = update_progress
        self.interval_seconds = interval_seconds
        self.step = step
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._stage = ""
        self._progress = 0
        self._max_progress = 0

    def start(self, stage: str, current_progress: int, max_progress: int) -> None:
        with self._lock:
            self._stage = stage
            self._progress = current_progress
            self._max_progress = max_progress
            if self._thread is None or not self._thread.is_alive():
                self._stop_event.clear()
                self._thread = Thread(target=self._run, daemon=True)
                self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 2)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            with self._lock:
                if self._progress >= self._max_progress:
                    continue
                self._progress = min(self._progress + self.step, self._max_progress)
                stage = self._stage
                progress = self._progress
            self.update_progress(stage, progress)


class JobProcessor:
    def __init__(
        self,
        storage: JobStorage,
        pipeline: TranscriptionPipeline,
        heartbeat_interval_seconds: float = 2.0,
    ):
        self.storage = storage
        self.pipeline = pipeline
        self.heartbeat_interval_seconds = heartbeat_interval_seconds

    def process(self, job_id: str) -> None:
        job = self.storage.load_job(job_id)
        job.status = JobStatus.running
        job.stage = "processing"
        job.progress = 0
        job.started_at = datetime.now(timezone.utc).isoformat()
        self.storage.save_job(job)

        try:
            for index, job_file in enumerate(job.files):
                job.current_file = job_file.filename
                job_file.status = FileStatus.running
                job_file.stage = "queued"
                job_file.progress = 0
                job.progress = int((index / job.total_files) * 100)
                self.storage.save_job(job)
                progress_lock = Lock()

                def update_progress(stage: str, file_progress: int) -> None:
                    with progress_lock:
                        # 模型回调和心跳线程可能交错到达，进度只允许前进，避免页面进度条回跳。
                        next_file_progress = max(job_file.progress, file_progress)
                        job.stage = stage
                        job.current_file = job_file.filename
                        job_file.stage = stage
                        job_file.progress = next_file_progress
                        job.progress = int(((index + next_file_progress / 100) / job.total_files) * 100)
                        self.storage.save_job(job)

                heartbeat = ProgressHeartbeat(
                    update_progress=update_progress,
                    interval_seconds=self.heartbeat_interval_seconds,
                )
                stage_caps = {
                    "normalizing_audio": 20,
                    "transcribing_audio": 65,
                    "diarizing_speakers": 84,
                    "assigning_speakers": 88,
                }

                def update_progress_with_heartbeat(stage: str, file_progress: int) -> None:
                    update_progress(stage, file_progress)
                    heartbeat.start(stage, job_file.progress, stage_caps.get(stage, file_progress))

                try:
                    segments = self.pipeline.transcribe(
                        audio_path=Path(job_file.stored_path),
                        progress_callback=update_progress_with_heartbeat,
                    )
                finally:
                    heartbeat.stop()

                job_file.stage = "exporting"
                job_file.progress = 90
                job.progress = int(((index + 0.9) / job.total_files) * 100)
                self.storage.save_job(job)

                result_dir = self.storage.result_dir(job.job_id, job_file.filename)
                job_file.outputs = export_segments(segments, result_dir, job_file.filename)
                job_file.status = FileStatus.completed
                job_file.stage = "completed"
                job_file.progress = 100
                job.progress = int(((index + 1) / job.total_files) * 100)
                self.storage.save_job(job)

            job.status = JobStatus.completed
            job.stage = "completed"
            job.progress = 100
            job.current_file = None
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.elapsed_seconds = self._elapsed_seconds(job.started_at, job.completed_at)
            self.storage.save_job(job)
        except Exception as error:  # noqa: BLE001
            # 任务失败时保留已完成文件的输出，便于用户重跑或人工排查。
            job.status = JobStatus.failed
            job.stage = "failed"
            job.error = str(error)
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.elapsed_seconds = self._elapsed_seconds(job.started_at, job.completed_at)
            if job.current_file:
                for job_file in job.files:
                    if job_file.filename == job.current_file:
                        job_file.status = FileStatus.failed
                        job_file.stage = "failed"
                        job_file.error = str(error)
                        break
            self.storage.save_job(job)
            raise

    @staticmethod
    def _elapsed_seconds(started_at: str | None, completed_at: str | None) -> float | None:
        if started_at is None or completed_at is None:
            return None
        started = datetime.fromisoformat(started_at)
        completed = datetime.fromisoformat(completed_at)
        return round((completed - started).total_seconds(), 2)
