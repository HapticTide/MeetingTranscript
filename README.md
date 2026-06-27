# Meeting Transcript API

本项目是一个本地会议音频转录 API 服务。当前支持上传、任务状态查询、结果导出和 zip 下载能力；ASR backend 可使用 `mlx-whisper`，说话人分离可选接入 `pyannote.audio`。

## 当前能力

- 上传一个或多个音频文件
- 返回 `job_id`
- 查询任务进度
- 查询每个文件状态
- 导出 `txt`、`md`、`json`、`srt`
- 下载结果 zip 包
- 通过配置预留 ASR 和 speaker diarization backend

## 本地启动

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e '.[dev,asr]'
.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

## API 示例

上传音频：

```bash
curl -X POST http://127.0.0.1:8000/transcriptions \
  -F "files=@/path/to/meeting.wav" \
  -F "files=@/path/to/another.m4a"
```

查询进度：

```bash
curl http://127.0.0.1:8000/jobs/{job_id}
```

查询文件状态：

```bash
curl http://127.0.0.1:8000/jobs/{job_id}/files
```

查询结果 manifest：

```bash
curl http://127.0.0.1:8000/jobs/{job_id}/result
```

下载结果包：

```bash
curl -L http://127.0.0.1:8000/jobs/{job_id}/result.zip -o result.zip
```

## 目录结构

```text
app/
  main.py            FastAPI 路由
  storage.py         上传文件、任务 JSON、结果文件持久化
  processor.py       任务处理状态机
  transcription.py   ASR / diarization backend 入口
  exporters.py       txt / md / json / srt 导出
  models.py          API 和任务模型
data/
  uploads/           原始上传文件
  jobs/              任务状态 JSON
  results/           转录结果
```

## 后续接真实模型

当前 `app/transcription.py` 的 `TranscriptionPipeline` 已支持：

- `ASR_BACKEND=mock`
- `ASR_BACKEND=mlx_whisper`
- `DIARIZATION_BACKEND=none`
- `DIARIZATION_BACKEND=pyannote`

默认建议先使用 `mlx_whisper + none` 跑通真实中文转录；确认速度和质量后再启用 `pyannote` 做 speaker diarization。

## 测试

```bash
.venv/bin/python -m pytest -q
```
