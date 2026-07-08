#!/bin/bash
# Setup & launch the QHH web test surface only.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/config_env.sh"

redis_cmd() {
    local output
    local status
    set +e
    output="$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" "$@" 2>&1)"
    status=$?
    set -e
    if { [ "$status" -ne 0 ] || [[ "$output" == *"NOAUTH"* ]]; } && [ -n "${REDIS_PASSWORD:-}" ]; then
        REDISCLI_AUTH="$REDIS_PASSWORD" redis-cli \
            -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" \
            --no-auth-warning "$@"
        return
    fi
    printf '%s\n' "$output"
    return "$status"
}

if ! redis_cmd ping >/dev/null 2>&1; then
    echo "Không kết nối được Redis $REDIS_HOST:$REDIS_PORT DB $REDIS_DB"
    exit 1
fi

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

REQUIRE_ONNX_CUDA="${REQUIRE_ONNX_CUDA:-1}"
"$PYTHON_BIN" - <<'PY'
import os
import sys
import onnxruntime as ort

providers = ort.get_available_providers()
print(f"ONNXRuntime providers: {providers}", flush=True)
if os.getenv("REQUIRE_ONNX_CUDA", "1").strip() in {"1", "true", "yes", "on"}:
    if "CUDAExecutionProvider" not in providers:
        print("CUDAExecutionProvider không khả dụng; dừng để tránh chạy CPU.", file=sys.stderr)
        sys.exit(3)
PY

export REDIS_HOST REDIS_PORT REDIS_DB REDIS_PASSWORD QHH_REDIS_PREFIX
export QHH_RECORD_DIR QHH_WEB_RECORD_ON_AI QHH_WEB_RECORD_DURATION_SEC
export QHH_WEB_RECORD_INTERVAL_SEC QHH_AI_AUTO_START QHH_VIDEO_READER_MAX_FPS
export FPS_LOG FPS_LOG_PERIOD ARC_LOG QHH_AI_RESULT_VIDEO_ON QHH_AI_RESULT_DIR
export QHH_AI_RESULT_VIDEO_FPS QHH_AI_RESULT_VIDEO_QUEUE QHH_AI_RESULT_VIDEO_CODEC
export QHH_AI_RESULT_VIDEO_EXT QHH_AI_DELETE_PROCESSED_VIDEO QHH_AI_SNAPSHOT_ON
export QHH_WEB_LIVE_PREVIEW_ON QHH_WEB_HOST QHH_WEB_PORT

echo "QHH web : http://${QHH_WEB_HOST}:${QHH_WEB_PORT}"
echo "Redis   : ${REDIS_HOST}:${REDIS_PORT} DB ${REDIS_DB}"
echo "Record  : ${QHH_RECORD_DIR}"
echo "On AI   : ${QHH_WEB_RECORD_ON_AI}"
echo "Python  : ${PYTHON_BIN}  (conda env=${CONDA_ENV})"

exec "$PYTHON_BIN" web_server.py --host "$QHH_WEB_HOST" --port "$QHH_WEB_PORT"
