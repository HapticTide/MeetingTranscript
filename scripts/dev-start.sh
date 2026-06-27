#!/bin/sh
set -eu

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
VENV_DIR="${VENV_DIR:-.venv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

exec "$VENV_DIR/bin/uvicorn" app.main:app --reload --host "$HOST" --port "$PORT"
