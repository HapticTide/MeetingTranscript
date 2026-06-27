# Meeting Transcript API 部署说明

这是一个本地会议音频转录 API 服务，支持上传一个或多个音频文件，离线生成 `txt`、`md`、`json`、`srt` 结果，并可选使用 `pyannote.audio` 做说话人分离。

## 运行环境

推荐环境：

- macOS + Apple Silicon
- Python 3.12
- Homebrew
- `ffmpeg`
- Hugging Face 账号与 read token

当前 Mac 推荐 backend：

- ASR：`mlx-whisper`
- Speaker diarization：`pyannote.audio`

## 安装系统依赖

```bash
brew install ffmpeg python@3.12
```

确认安装：

```bash
ffmpeg -version
/opt/homebrew/bin/python3.12 --version
```

## 创建 Python 环境

```bash
/opt/homebrew/bin/python3.12 -m venv .venv312
.venv312/bin/python -m pip install --upgrade pip setuptools wheel
.venv312/bin/python -m pip install -e '.[dev,asr,diarization]'
```

如果只需要文字转录、不需要说话人分离，可只安装：

```bash
.venv312/bin/python -m pip install -e '.[dev,asr]'
```

## Hugging Face 登录

`pyannote.audio` 需要访问 Hugging Face gated model。先在网页接受模型条款：

- `pyannote/speaker-diarization-community-1`

然后登录：

```bash
.venv312/bin/hf auth login
.venv312/bin/hf auth whoami
```

不要把 token 写进 Git 仓库。

## 配置服务

复制配置模板：

```bash
cp .env.example .env
```

只做 ASR：

```env
MEETING_TRANSCRIPT_ASR_BACKEND=mlx_whisper
MEETING_TRANSCRIPT_DIARIZATION_BACKEND=none
MEETING_TRANSCRIPT_ASR_MODEL=mlx-community/whisper-large-v3-turbo
MEETING_TRANSCRIPT_ASR_LANGUAGE=zh
```

启用说话人分离：

```env
MEETING_TRANSCRIPT_DIARIZATION_BACKEND=pyannote
MEETING_TRANSCRIPT_PYANNOTE_MODEL=pyannote/speaker-diarization-community-1
MEETING_TRANSCRIPT_MIN_SPEAKERS=2
MEETING_TRANSCRIPT_MAX_SPEAKERS=8
```

## 启动服务

```bash
VENV_DIR=.venv312 ./scripts/dev-start.sh
```

默认监听：

```text
http://127.0.0.1:8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

## API 使用

上传音频：

```bash
curl -X POST http://127.0.0.1:8000/transcriptions \
  -F "files=@/path/to/meeting.m4a"
```

查询任务：

```bash
curl http://127.0.0.1:8000/jobs/{job_id}
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

## 数据目录

运行数据默认写入 `data/`：

```text
data/uploads/   原始上传文件
data/jobs/      任务状态 JSON
data/results/   txt、md、json、srt 结果
```

这些目录是本地运行产物，不应提交。

## 测试

```bash
.venv312/bin/python -m pytest -q
```

单元测试会 mock 大模型路径，避免测试阶段下载模型或依赖 Metal GPU。

## 常见问题

### pyannote 提示 401 Unauthorized

确认已完成：

1. 在 Hugging Face 网页接受模型条款
2. `.venv312/bin/hf auth login`
3. 不要覆盖 `XDG_CACHE_HOME` 导致 SDK 读不到默认 token

### 普通 shell 中 MLX 提示 No Metal device

`mlx-whisper` 需要访问 Metal。若在受限或 headless 环境报错，请在正常 macOS 终端启动服务。

### 说话人数量不符合预期

调整：

```env
MEETING_TRANSCRIPT_MIN_SPEAKERS=5
MEETING_TRANSCRIPT_MAX_SPEAKERS=8
```

然后重启服务并重新转录。
