#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/config_env.sh"

# Conda `base` has onnxruntime-gpu + torch with CUDA. The old project venv
# only has CPU ONNX — it would silently fall back to CPU and run ~10x slower.
CONDA_BASE="${CONDA_BASE:-/home/mq/miniconda3}"
CONDA_ENV="${CONDA_ENV:-base}"
CONDA_ACTIVATED=0
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    set +e
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    CONDA_ACTIVATED=$?
    set -e
fi

# Allow override but default to whichever python is now on PATH.
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ "$CONDA_ACTIVATED" = "0" ]; then
        PYTHON_BIN="$(command -v python)"
    elif [ -x "$CONDA_BASE/bin/python" ]; then
        PYTHON_BIN="$CONDA_BASE/bin/python"
    else
        PYTHON_BIN="$(command -v python || true)"
    fi
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Không tìm thấy Python: $PYTHON_BIN"
    exit 1
fi

echo "QHH web : http://192.168.6.16:${QHH_WEB_PORT}"
echo "Redis   : ${REDIS_HOST}:${REDIS_PORT} DB ${REDIS_DB}"
echo "Record  : ${QHH_RECORD_DIR}"
echo "On AI   : ${QHH_WEB_RECORD_ON_AI}"
echo "Python  : $PYTHON_BIN  (conda env=$CONDA_ENV)"
"$PYTHON_BIN" - <<'PY'
import os
import sys
import onnxruntime as ort
providers = ort.get_available_providers()
print(f"ONNXRuntime providers: {providers}")
if os.getenv("REQUIRE_ONNX_CUDA", "1").strip() in {"1", "true", "yes", "on"}:
    if "CUDAExecutionProvider" not in providers:
        print("CUDAExecutionProvider không khả dụng; dừng để tránh chạy CPU.", file=sys.stderr)
        sys.exit(3)
PY
exec "$PYTHON_BIN" web_server.py --host "$QHH_WEB_HOST" --port "$QHH_WEB_PORT"
