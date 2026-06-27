from pathlib import Path

from app.exporters import export_segments
from app.models import FileStatus, JobStatus
from app.storage import JobStorage
from app.transcription import TranscriptionPipeline


class JobProcessor:
    def __init__(self, storage: JobStorage, pipeline: TranscriptionPipeline):
        self.storage = storage
        self.pipeline = pipeline

    def process(self, job_id: str) -> None:
        job = self.storage.load_job(job_id)
        job.status = JobStatus.running
        job.stage = "processing"
        job.progress = 0
        self.storage.save_job(job)

        try:
            for index, job_file in enumerate(job.files):
                job.current_file = job_file.filename
                job_file.status = FileStatus.running
                job_file.stage = "transcribing"
                job_file.progress = 10
                self.storage.save_job(job)

                segments = self.pipeline.transcribe(audio_path=Path(job_file.stored_path))

                job_file.stage = "exporting"
                job_file.progress = 80
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
            self.storage.save_job(job)
        except Exception as error:  # noqa: BLE001
            # 任务失败时保留已完成文件的输出，便于用户重跑或人工排查。
            job.status = JobStatus.failed
            job.stage = "failed"
            job.error = str(error)
            if job.current_file:
                for job_file in job.files:
                    if job_file.filename == job.current_file:
                        job_file.status = FileStatus.failed
                        job_file.stage = "failed"
                        job_file.error = str(error)
                        break
            self.storage.save_job(job)
            raise
