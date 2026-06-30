from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class FileStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class TranscriptSegment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str


class OutputPaths(BaseModel):
    text: Optional[str] = None
    markdown: Optional[str] = None
    json_path: Optional[str] = None
    srt: Optional[str] = None


class JobFile(BaseModel):
    filename: str
    stored_path: str
    status: FileStatus = FileStatus.queued
    stage: str = "queued"
    progress: int = 0
    error: Optional[str] = None
    outputs: OutputPaths = Field(default_factory=OutputPaths)


class Job(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.queued
    stage: str = "queued"
    progress: int = 0
    total_files: int
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    current_file: Optional[str] = None
    error: Optional[str] = None
    files: List[JobFile]


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    total_files: int


class JobResult(BaseModel):
    job_id: str
    elapsed_seconds: Optional[float] = None
    files: List[JobFile]
    artifacts: Dict[str, str]
