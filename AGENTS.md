# Repository Guidelines

## Project Structure & Module Organization

本仓库是本地会议音频转录 API 服务。核心代码在 `app/`：

- `app/main.py`：FastAPI 路由与应用创建入口。
- `app/processor.py`：任务状态机与处理流程。
- `app/transcription.py`：ASR 与 speaker diarization backend。
- `app/storage.py`：上传文件、任务 JSON、结果文件持久化。
- `app/exporters.py`：导出 `txt`、`md`、`json`、`srt`。
- `tests/`：pytest 测试。
- `data/`：运行期上传、任务与结果目录，不应提交。

## Build, Test, and Development Commands

创建 Python 3.12 环境并安装依赖：

```bash
/opt/homebrew/bin/python3.12 -m venv .venv312
.venv312/bin/python -m pip install -e '.[dev,asr,diarization]'
```

运行测试：

```bash
.venv312/bin/python -m pytest -q
```

启动本地服务：

```bash
VENV_DIR=.venv312 ./scripts/dev-start.sh
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

常用 API：

```bash
curl -X POST http://127.0.0.1:8000/transcriptions -F "files=@meeting.m4a"
curl http://127.0.0.1:8000/jobs/{job_id}
curl -L http://127.0.0.1:8000/jobs/{job_id}/result.zip -o result.zip
```

## Architecture Overview

处理链路是“上传文件 → 创建任务 → 音频标准化 → ASR → 可选说话人分离 → 导出结果”。`mlx-whisper` 负责中文转录，`pyannote.audio` 负责 speaker diarization。所有 backend 输出都应收敛到 `TranscriptSegment`，避免 API 层依赖具体模型实现。

## Coding Style & Naming Conventions

使用 Python 3.12，遵循现有模块风格。函数、变量、文件名使用英文 `snake_case`；类名使用 `PascalCase`。复杂逻辑需要简短中文注释，尤其是音频预处理、speaker 合并和任务状态转换。避免引入一次性抽象，修改范围应贴近需求。

## Testing Guidelines

测试框架是 pytest。新增 API 行为放在 `tests/test_api.py`，转录和 diarization 逻辑放在 `tests/test_transcription.py`。测试应优先 mock 大模型和外部服务，避免单元测试直接下载模型或访问 Metal GPU。提交前至少运行：

```bash
.venv312/bin/python -m pytest -q
```

## Configuration & Security Tips

本地配置在 `.env`，不要提交真实 token。Hugging Face token 应通过 `hf auth login` 管理。`MEETING_TRANSCRIPT_DIARIZATION_BACKEND=pyannote` 需要已接受 pyannote 模型条款。不要把 `data/`、`.venv*/`、`.cache/` 加入版本控制。

## Commit & Pull Request Guidelines

当前目录没有 Git 历史可参考。提交信息建议使用简体中文，例如：`feat: 接入 pyannote 说话人分离`。PR 应说明变更范围、验证命令、是否影响模型依赖或 `.env` 配置；涉及 API 行为时附示例请求与响应。
