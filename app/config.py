from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """服务配置。

    第一版只把运行目录、处理开关和 backend 名称做成配置项，避免把模型路径、
    文件路径写死在业务代码里。后续迁移到 Linux 时主要改这里或 `.env`。
    """

    data_dir: Path = Path("data")
    auto_process: bool = True
    asr_backend: str = "mock"
    diarization_backend: str = "mock"
    asr_model: str = "mlx-community/whisper-large-v3-turbo"
    asr_language: str = "zh"
    pyannote_model: str = "pyannote/speaker-diarization-community-1"
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    max_upload_bytes: int = 1024 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MEETING_TRANSCRIPT_",
        extra="ignore",
    )


def build_settings(
    data_dir: Optional[Path] = None,
    auto_process: Optional[bool] = None,
    asr_backend: Optional[str] = None,
    diarization_backend: Optional[str] = None,
) -> Settings:
    settings = Settings()
    if data_dir is not None:
        settings.data_dir = data_dir
    if auto_process is not None:
        settings.auto_process = auto_process
    if asr_backend is not None:
        settings.asr_backend = asr_backend
    if diarization_backend is not None:
        settings.diarization_backend = diarization_backend
    return settings
