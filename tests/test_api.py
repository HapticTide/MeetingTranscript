from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_returns_service_status(tmp_path: Path):
    client = TestClient(create_app(data_dir=tmp_path, asr_backend="mock", diarization_backend="mock"))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_serves_upload_page(tmp_path: Path):
    client = TestClient(create_app(data_dir=tmp_path, asr_backend="mock", diarization_backend="mock"))

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Meeting Transcript" in response.text
    assert "历史任务" in response.text
    assert "清空历史" not in response.text
    assert 'class="layout"' in response.text
    assert 'class="sidebar panel"' in response.text
    assert "耗时" in response.text
    assert "elapsedText" in response.text
    assert "当前步骤" in response.text
    assert "currentStepText" in response.text
    assert "pollIntervalMs = 1000" in response.text
    assert "localStorage" in response.text


def test_create_app_passes_diarization_settings_to_pipeline(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEETING_TRANSCRIPT_ASR_BACKEND", "mlx_whisper")
    monkeypatch.setenv("MEETING_TRANSCRIPT_DIARIZATION_BACKEND", "pyannote")
    monkeypatch.setenv("MEETING_TRANSCRIPT_MIN_SPEAKERS", "5")
    monkeypatch.setenv("MEETING_TRANSCRIPT_MAX_SPEAKERS", "8")

    app = create_app(data_dir=tmp_path)
    pipeline = app.state.processor.pipeline

    assert pipeline.diarization_backend == "pyannote"
    assert pipeline.min_speakers == 5
    assert pipeline.max_speakers == 8


def test_upload_multiple_files_creates_job_and_persists_inputs(tmp_path: Path):
    client = TestClient(
        create_app(data_dir=tmp_path, auto_process=False, asr_backend="mock", diarization_backend="mock")
    )

    response = client.post(
        "/transcriptions",
        files=[
            ("files", ("first.wav", b"fake-audio-1", "audio/wav")),
            ("files", ("second.m4a", b"fake-audio-2", "audio/mp4")),
        ],
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["total_files"] == 2

    job = client.get(f"/jobs/{payload['job_id']}").json()
    assert job["status"] == "queued"
    assert [item["filename"] for item in job["files"]] == ["first.wav", "second.m4a"]
    assert (tmp_path / "uploads" / payload["job_id"] / "first.wav").exists()
    assert (tmp_path / "uploads" / payload["job_id"] / "second.m4a").exists()


def test_upload_rejects_unsupported_file_type(tmp_path: Path):
    client = TestClient(
        create_app(data_dir=tmp_path, auto_process=False, asr_backend="mock", diarization_backend="mock")
    )

    response = client.post(
        "/transcriptions",
        files=[("files", ("notes.txt", b"not audio", "text/plain"))],
    )

    assert response.status_code == 400
    assert "Unsupported audio file type" in response.json()["detail"]


def test_auto_process_completes_job_and_exposes_results(tmp_path: Path):
    client = TestClient(
        create_app(data_dir=tmp_path, auto_process=True, asr_backend="mock", diarization_backend="mock")
    )

    response = client.post(
        "/transcriptions",
        files=[("files", ("meeting.wav", b"fake-audio", "audio/wav"))],
    )

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["started_at"] is not None
    assert job["completed_at"] is not None
    assert job["elapsed_seconds"] is not None
    assert job["elapsed_seconds"] >= 0
    assert job["files"][0]["status"] == "completed"

    result = client.get(f"/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.json()["job_id"] == job_id
    assert result.json()["elapsed_seconds"] == job["elapsed_seconds"]
    assert result.json()["files"][0]["outputs"]["markdown"].endswith("meeting.md")

    text_result = client.get(f"/jobs/{job_id}/result/text")
    assert text_result.status_code == 200
    assert "Mock transcript for meeting.wav." in text_result.text

    zip_result = client.get(f"/jobs/{job_id}/result.zip")
    assert zip_result.status_code == 200
    assert zip_result.headers["content-type"] == "application/zip"


def test_unknown_job_returns_404(tmp_path: Path):
    client = TestClient(create_app(data_dir=tmp_path, asr_backend="mock", diarization_backend="mock"))

    response = client.get("/jobs/not-found")

    assert response.status_code == 404
