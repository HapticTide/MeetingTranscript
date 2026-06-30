from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from app.config import build_settings
from app.models import CreateJobResponse, Job, JobResult
from app.processor import JobProcessor
from app.storage import JobStorage
from app.transcription import TranscriptionPipeline


def create_app(
    data_dir: Optional[Path] = None,
    auto_process: Optional[bool] = None,
    asr_backend: Optional[str] = None,
    diarization_backend: Optional[str] = None,
) -> FastAPI:
    settings = build_settings(
        data_dir=data_dir,
        auto_process=auto_process,
        asr_backend=asr_backend,
        diarization_backend=diarization_backend,
    )
    storage = JobStorage(settings.data_dir)
    pipeline = TranscriptionPipeline(
        asr_backend=settings.asr_backend,
        diarization_backend=settings.diarization_backend,
        asr_model=settings.asr_model,
        asr_language=settings.asr_language,
        pyannote_model=settings.pyannote_model,
        min_speakers=settings.min_speakers,
        max_speakers=settings.max_speakers,
    )
    processor = JobProcessor(storage=storage, pipeline=pipeline)

    app = FastAPI(title="Meeting Transcript API", version="0.1.0")
    app.state.settings = settings
    app.state.storage = storage
    app.state.processor = processor

    @app.get("/", response_class=FileResponse)
    def index() -> FileResponse:
        return FileResponse(Path("static/index.html"), media_type="text/html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/transcriptions", response_model=CreateJobResponse, status_code=202)
    def create_transcription(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
    ) -> CreateJobResponse:
        if not files:
            raise HTTPException(status_code=400, detail="At least one audio file is required.")

        try:
            job = storage.create_job(files)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        if settings.auto_process:
            # TestClient 场景中直接同步处理，真实 uvicorn 运行时则交给 BackgroundTasks。
            if data_dir is not None:
                processor.process(job.job_id)
                job = storage.load_job(job.job_id)
            else:
                background_tasks.add_task(processor.process, job.job_id)

        return CreateJobResponse(
            job_id=job.job_id,
            status=job.status,
            total_files=job.total_files,
        )

    @app.get("/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str) -> Job:
        try:
            return storage.load_job(job_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Job not found.") from error

    @app.get("/jobs/{job_id}/files")
    def get_job_files(job_id: str) -> dict:
        try:
            job = storage.load_job(job_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Job not found.") from error
        return {"job_id": job.job_id, "files": job.files}

    @app.get("/jobs/{job_id}/result", response_model=JobResult)
    def get_job_result(job_id: str) -> JobResult:
        try:
            job = storage.load_job(job_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Job not found.") from error

        if job.status.value != "completed":
            raise HTTPException(status_code=409, detail="Job is not completed.")

        return JobResult(
            job_id=job.job_id,
            elapsed_seconds=job.elapsed_seconds,
            files=job.files,
            artifacts=storage.result_artifacts(job_id),
        )

    @app.get("/jobs/{job_id}/result/text", response_class=PlainTextResponse)
    def get_job_text_result(job_id: str) -> str:
        try:
            job = storage.load_job(job_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Job not found.") from error

        if job.status.value != "completed":
            raise HTTPException(status_code=409, detail="Job is not completed.")

        parts = []
        for job_file in job.files:
            if job_file.outputs.text is None:
                continue
            text_path = Path(job_file.outputs.text)
            if text_path.exists():
                parts.append(text_path.read_text())
        return "\n".join(parts)

    @app.get("/jobs/{job_id}/result.zip")
    def download_job_result(job_id: str) -> FileResponse:
        try:
            job = storage.load_job(job_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Job not found.") from error

        if job.status.value != "completed":
            raise HTTPException(status_code=409, detail="Job is not completed.")

        zip_path = storage.result_zip(job_id)
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"{job_id}.zip",
        )

    return app


app = create_app()
