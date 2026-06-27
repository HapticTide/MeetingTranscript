import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from app.models import TranscriptSegment


class SpeakerSegment:
    def __init__(self, start: float, end: float, label: str):
        self.start = start
        self.end = end
        self.label = label


class TranscriptionPipeline:
    """转录流水线入口。

    backend 的输出统一收敛到 `TranscriptSegment`，让 API、任务进度和导出层不用关心
    底层是 `mlx-whisper`、`whisper.cpp` 还是后续 Linux 上的 `faster-whisper`。
    """

    def __init__(
        self,
        asr_backend: str,
        diarization_backend: str,
        asr_model: str = "mlx-community/whisper-large-v3-turbo",
        asr_language: str = "zh",
        pyannote_model: str = "pyannote/speaker-diarization-community-1",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        self.asr_backend = asr_backend
        self.diarization_backend = diarization_backend
        self.asr_model = asr_model
        self.asr_language = asr_language
        self.pyannote_model = pyannote_model
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def transcribe(self, audio_path: Path) -> List[TranscriptSegment]:
        if self.asr_backend == "mock":
            return self._mock_transcribe(audio_path)
        if self.asr_backend == "mlx_whisper":
            with self._normalized_audio(audio_path) as normalized_audio:
                asr_segments = self._transcribe_with_mlx_whisper(normalized_audio)
                speaker_segments = self._diarize(normalized_audio)
            return self._assign_speakers(asr_segments, speaker_segments)
        raise NotImplementedError(f"Unsupported ASR backend: {self.asr_backend}")

    def _mock_transcribe(self, audio_path: Path) -> List[TranscriptSegment]:
        return [
            TranscriptSegment(
                start=0.0,
                end=3.2,
                speaker="Speaker 1",
                text=f"Mock transcript for {audio_path.name}.",
            )
        ]

    def _transcribe_with_mlx_whisper(self, audio_path: Path) -> List[TranscriptSegment]:
        try:
            import mlx_whisper
        except ImportError as error:
            raise RuntimeError(
                "mlx-whisper is not installed. Install the real backend dependencies first."
            ) from error

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self.asr_model,
            language=self.asr_language,
        )

        segments = []
        for raw_segment in result.get("segments", []):
            text = str(raw_segment.get("text", "")).strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    start=float(raw_segment.get("start", 0.0)),
                    end=float(raw_segment.get("end", 0.0)),
                    speaker="Speaker 1",
                    text=text,
                )
            )
        return segments

    def _normalized_audio(self, audio_path: Path):
        if audio_path.suffix.lower() == ".wav":
            return _ExistingAudio(audio_path)
        return _FfmpegNormalizedAudio(audio_path)

    def _diarize(self, audio_path: Path) -> List[SpeakerSegment]:
        if self.diarization_backend in {"none", "mock"}:
            return []
        if self.diarization_backend != "pyannote":
            raise NotImplementedError(f"Unsupported diarization backend: {self.diarization_backend}")

        try:
            from pyannote.audio import Pipeline
        except ImportError as error:
            raise RuntimeError(
                "pyannote.audio is not installed. Install the diarization backend dependencies first."
            ) from error

        pipeline = Pipeline.from_pretrained(self.pyannote_model)
        options = {}
        if self.min_speakers is not None:
            options["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            options["max_speakers"] = self.max_speakers

        diarization = pipeline(str(audio_path), **options)
        speaker_segments = []
        source = getattr(diarization, "exclusive_speaker_diarization", diarization)
        for turn, _track, label in source.itertracks(yield_label=True):
            speaker_segments.append(SpeakerSegment(float(turn.start), float(turn.end), str(label)))
        return speaker_segments

    def _assign_speakers(
        self,
        asr_segments: List[TranscriptSegment],
        speaker_segments: List[SpeakerSegment],
    ) -> List[TranscriptSegment]:
        if not speaker_segments:
            return asr_segments

        speaker_names: Dict[str, str] = {}
        assigned = []
        for segment in asr_segments:
            label = self._best_speaker_label(segment, speaker_segments)
            if label not in speaker_names:
                speaker_names[label] = f"Speaker {len(speaker_names) + 1}"
            assigned.append(
                TranscriptSegment(
                    start=segment.start,
                    end=segment.end,
                    speaker=speaker_names[label],
                    text=segment.text,
                )
            )
        return assigned

    @staticmethod
    def _best_speaker_label(segment: TranscriptSegment, speaker_segments: List[SpeakerSegment]) -> str:
        best_label = speaker_segments[0].label
        best_overlap = -1.0
        for speaker_segment in speaker_segments:
            overlap = min(segment.end, speaker_segment.end) - max(segment.start, speaker_segment.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = speaker_segment.label
        return best_label


class _ExistingAudio:
    def __init__(self, audio_path: Path):
        self.audio_path = audio_path

    def __enter__(self) -> Path:
        return self.audio_path

    def __exit__(self, *_args) -> None:
        return None


class _FfmpegNormalizedAudio:
    def __init__(self, audio_path: Path):
        self.audio_path = audio_path
        self._tmpdir: Optional[tempfile.TemporaryDirectory] = None

    def __enter__(self) -> Path:
        self._tmpdir = tempfile.TemporaryDirectory()
        output_path = Path(self._tmpdir.name) / f"{self.audio_path.stem}.wav"
        # 统一成 16kHz 单声道，减少不同格式、采样率对 ASR 和说话人分离的影响。
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(self.audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(output_path),
            ],
            check=True,
        )
        return output_path

    def __exit__(self, *_args) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
