import time
from pathlib import Path

from app.models import Job, JobFile, TranscriptSegment
from app.processor import JobProcessor, ProgressHeartbeat
from app.storage import JobStorage


class RecordingStorage(JobStorage):
    def __init__(self, data_dir: Path):
        super().__init__(data_dir)
        self.saved_file_progress = []

    def save_job(self, job: Job) -> None:
        if job.files:
            self.saved_file_progress.append(job.files[0].progress)
        super().save_job(job)


class RegressivePipeline:
    def transcribe(self, audio_path: Path, progress_callback):
        progress_callback("transcribing_audio", 50)
        progress_callback("transcribing_audio", 40)
        return [TranscriptSegment(start=0.0, end=1.0, speaker="Speaker 1", text=audio_path.name)]


def test_progress_heartbeat_advances_until_stage_cap():
    events = []

    heartbeat = ProgressHeartbeat(
        update_progress=lambda stage, progress: events.append((stage, progress)),
        interval_seconds=0.01,
        step=5,
    )

    heartbeat.start(stage="transcribing_audio", current_progress=35, max_progress=45)
    time.sleep(0.06)
    heartbeat.stop()

    assert ("transcribing_audio", 40) in events
    assert ("transcribing_audio", 45) in events
    assert all(progress <= 45 for _stage, progress in events)


def test_job_processor_does_not_regress_progress_from_callbacks(tmp_path: Path):
    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"fake-audio")
    storage = RecordingStorage(tmp_path / "data")
    job = Job(
        job_id="job-1",
        total_files=1,
        files=[JobFile(filename="meeting.wav", stored_path=str(audio_path))],
    )
    storage.save_job(job)

    processor = JobProcessor(storage=storage, pipeline=RegressivePipeline(), heartbeat_interval_seconds=60)
    processor.process(job.job_id)

    assert 50 in storage.saved_file_progress
    assert 40 not in storage.saved_file_progress
