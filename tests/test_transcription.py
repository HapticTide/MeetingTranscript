import sys
import types
from pathlib import Path

from app.transcription import TranscriptionPipeline


def test_non_wav_audio_is_normalized_with_ffmpeg(tmp_path: Path, monkeypatch):
    audio_path = tmp_path / "meeting.m4a"
    audio_path.write_bytes(b"fake")

    calls = []

    run_options = []

    def fake_run(command, **kwargs):
        calls.append(command)
        run_options.append(kwargs)
        Path(command[-1]).write_bytes(b"wav")

    fake_mlx = types.SimpleNamespace(
        transcribe=lambda audio, **_kwargs: {
            "segments": [{"start": 0.0, "end": 1.0, "text": Path(audio).suffix}]
        }
    )
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx)
    monkeypatch.setattr("app.transcription.subprocess.run", fake_run)

    pipeline = TranscriptionPipeline(
        asr_backend="mlx_whisper",
        diarization_backend="none",
    )
    progress_events = []

    segments = pipeline.transcribe(audio_path, progress_callback=lambda stage, progress: progress_events.append((stage, progress)))

    assert calls[0][:6] == ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    assert calls[0][6:8] == ["-i", str(audio_path)]
    assert calls[0][8:12] == ["-ac", "1", "-ar", "16000"]
    assert run_options[0]["check"] is True
    assert run_options[0]["capture_output"] is True
    assert run_options[0]["text"] is True
    assert progress_events[:2] == [("normalizing_audio", 10), ("transcribing_audio", 35)]
    assert segments[0].text == ".wav"


def test_mlx_whisper_backend_converts_segments_without_diarization(tmp_path: Path, monkeypatch):
    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"fake")

    fake_mlx = types.SimpleNamespace(
        transcribe=lambda *_args, **_kwargs: {
            "segments": [
                {"start": 0.0, "end": 1.5, "text": " 你好。 "},
                {"start": 1.5, "end": 3.0, "text": "继续讨论。"},
            ]
        }
    )
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx)

    pipeline = TranscriptionPipeline(
        asr_backend="mlx_whisper",
        diarization_backend="none",
        asr_model="mlx-community/whisper-large-v3-turbo",
    )

    segments = pipeline.transcribe(audio_path)

    assert [segment.text for segment in segments] == ["你好。", "继续讨论。"]
    assert [segment.speaker for segment in segments] == ["Speaker 1", "Speaker 1"]


def test_mlx_whisper_backend_assigns_speakers_from_pyannote(tmp_path: Path, monkeypatch):
    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"fake")

    fake_mlx = types.SimpleNamespace(
        transcribe=lambda *_args, **_kwargs: {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "第一位发言。"},
                {"start": 1.1, "end": 2.0, "text": "第二位发言。"},
            ]
        }
    )
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx)

    class FakeTurn:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class FakeDiarization:
        def itertracks(self, yield_label: bool):
            assert yield_label is True
            yield FakeTurn(0.0, 1.0), None, "SPEAKER_00"
            yield FakeTurn(1.0, 2.0), None, "SPEAKER_01"

    class FakePyannotePipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, *_args, **_kwargs):
            return FakeDiarization()

    fake_pyannote_audio = types.SimpleNamespace(Pipeline=FakePyannotePipeline)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_audio)

    pipeline = TranscriptionPipeline(
        asr_backend="mlx_whisper",
        diarization_backend="pyannote",
        asr_model="mlx-community/whisper-large-v3-turbo",
        pyannote_model="local-model",
    )

    segments = pipeline.transcribe(audio_path)

    assert [segment.speaker for segment in segments] == ["Speaker 1", "Speaker 2"]
