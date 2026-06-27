import json
from pathlib import Path
from typing import Iterable, List
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import UploadFile

from app.models import Job, JobFile


SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".wav",
    ".webm",
}


class JobStorage:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.uploads_dir = data_dir / "uploads"
        self.jobs_dir = data_dir / "jobs"
        self.results_dir = data_dir / "results"
        for directory in (self.uploads_dir, self.jobs_dir, self.results_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def create_job(self, files: Iterable[UploadFile]) -> Job:
        job_id = uuid4().hex
        upload_dir = self.uploads_dir / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        job_files: List[JobFile] = []
        for upload in files:
            self._validate_audio_file(upload.filename)
            stored_path = upload_dir / Path(upload.filename).name
            # UploadFile 可能由 SpooledTemporaryFile 支撑，逐块复制可以避免一次性读入大文件。
            with stored_path.open("wb") as output:
                while chunk := upload.file.read(1024 * 1024):
                    output.write(chunk)
            upload.file.seek(0)
            job_files.append(JobFile(filename=stored_path.name, stored_path=str(stored_path)))

        job = Job(job_id=job_id, total_files=len(job_files), files=job_files)
        self.save_job(job)
        return job

    def load_job(self, job_id: str) -> Job:
        path = self._job_path(job_id)
        if not path.exists():
            raise FileNotFoundError(job_id)
        return Job.model_validate_json(path.read_text())

    def save_job(self, job: Job) -> None:
        self._job_path(job.job_id).write_text(job.model_dump_json(indent=2))

    def result_dir(self, job_id: str, filename: str) -> Path:
        stem = Path(filename).stem
        directory = self.results_dir / job_id / stem
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def result_artifacts(self, job_id: str) -> dict:
        result_root = self.results_dir / job_id
        if not result_root.exists():
            return {}
        return {
            path.name: str(path)
            for path in result_root.rglob("*")
            if path.is_file()
        }

    def result_zip(self, job_id: str) -> Path:
        result_root = self.results_dir / job_id
        zip_path = self.results_dir / f"{job_id}.zip"
        with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
            for path in result_root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(result_root))
        return zip_path

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    @staticmethod
    def _validate_audio_file(filename: str) -> None:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio file type: {filename}")
