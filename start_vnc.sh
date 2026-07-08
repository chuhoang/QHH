#!/bin/bash
# ───────────────────────────────────────────────────────────
# Launch Classroom Manager with VNC + Web VNC
# Access: http://<server_ip>:6080/vnc_lite.html
# ───────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/config_env.sh"

NOVNC_DIR="${NOVNC_DIR:-/tmp/novnc-web}"
XVFB_DISPLAY="${XVFB_DISPLAY:-:99}"
VNC_PORT="${VNC_PORT:-5900}"
WEB_PORT="${WEB_PORT:-6080}"
SCREEN_SIZE="${SCREEN_SIZE:-1920x1080x24}"
# Default to conda `base` (has onnxruntime-gpu + torch+CUDA). Override with
# VENV_DIR / VENV_PYTHON if you really need the legacy CPU venv.
CONDA_BASE="${CONDA_BASE:-/home/mq/miniconda3}"
CONDA_ENV="${CONDA_ENV:-base}"
VENV_DIR="${VENV_DIR:-}"
if [ -n "$VENV_DIR" ]; then
    VENV_PYTHON="${VENV_PYTHON:-$VENV_DIR/bin/python}"
else
    VENV_PYTHON="${VENV_PYTHON:-$CONDA_BASE/envs/$CONDA_ENV/bin/python}"
    [ -x "$VENV_PYTHON" ] || VENV_PYTHON="$CONDA_BASE/bin/python"
fi

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

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🎓  Classroom Manager + VNC"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Kill old instances for this display/ports ──────
echo "🧹 Dọn dẹp tiến trình cũ..."
pkill -f "$SCRIPT_DIR/main.py" 2>/dev/null || true
pkill -f "websockify.*$WEB_PORT" 2>/dev/null || true
pkill -f "x11vnc.*$XVFB_DISPLAY" 2>/dev/null || true
pkill -f "Xvfb $XVFB_DISPLAY" 2>/dev/null || true
sleep 2

# ── 1b. Redis (persistent) ─────────────────────────────
if ! redis_cmd ping >/dev/null 2>&1; then
    if [ "$REDIS_HOST" != "localhost" ] && [ "$REDIS_HOST" != "127.0.0.1" ]; then
        echo "❌ Không kết nối được Redis $REDIS_HOST:$REDIS_PORT DB $REDIS_DB"
        exit 1
    fi
    echo "🗄  Khởi động Redis cổng $REDIS_PORT với cấu hình bền vững..."
    if [ -f "$SCRIPT_DIR/redis.conf" ]; then
        redis-server "$SCRIPT_DIR/redis.conf" --port "$REDIS_PORT"
    else
        echo "⚠️  redis.conf không có; chạy redis với mặc định (KHÔNG bền vững!)"
        redis-server --daemonize yes --port "$REDIS_PORT"
    fi
    sleep 1
    if ! redis_cmd ping >/dev/null 2>&1; then
        echo "❌ Redis không khởi động được"; exit 1
    fi
fi
echo "✅ Redis $REDIS_HOST:$REDIS_PORT DB $REDIS_DB sẵn sàng (appendonly=$(redis_cmd CONFIG GET appendonly | tail -1))"
export REDIS_HOST REDIS_PORT REDIS_DB REDIS_PASSWORD
export MIDDLEWARE_STREAMS_KEY MIDDLEWARE_AI_READ_DIR

# ── 2. Xvfb virtual display ────────────────────────────
echo "📺 Khởi động Xvfb $XVFB_DISPLAY..."
Xvfb $XVFB_DISPLAY -screen 0 $SCREEN_SIZE -ac +extension GLX +render &
XVFB_PID=$!
sleep 1
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "❌ Xvfb không khởi động được"
    exit 1
fi
echo "✅ Xvfb PID=$XVFB_PID"

# ── 3. x11vnc (VNC server) ─────────────────────────────
echo "🖥  Khởi động x11vnc trên cổng $VNC_PORT..."
x11vnc -display $XVFB_DISPLAY -forever -nopw -bg -o /tmp/x11vnc.log 2>/dev/null
sleep 1
echo "✅ x11vnc ready on port $VNC_PORT"

# ── 4. Websockify + noVNC (web viewer) ─────────────────
if [ ! -f "$NOVNC_DIR/vnc_lite.html" ] || [ ! -f "$NOVNC_DIR/package.json" ]; then
    echo "📥 Tải noVNC v1.5.0..."
    rm -rf "$NOVNC_DIR"
    mkdir -p "$NOVNC_DIR"
    if ! curl -sL "https://github.com/novnc/noVNC/archive/refs/tags/v1.5.0.tar.gz" \
        | tar xz -C "$NOVNC_DIR" --strip-components=1; then
        echo "⚠️  Không tải được noVNC từ GitHub. Thử dùng /usr/share/novnc..."
        if [ -d /usr/share/novnc ]; then
            NOVNC_DIR=/usr/share/novnc
            echo "   → dùng $NOVNC_DIR (lưu ý: bản apt có thể có lỗi UI; mở vnc_lite.html)"
        else
            echo "❌ Không có noVNC khả dụng"; exit 1
        fi
    fi
fi

echo "🌐 Khởi động web VNC trên cổng $WEB_PORT (web=$NOVNC_DIR)..."
nohup websockify --web "$NOVNC_DIR" $WEB_PORT localhost:$VNC_PORT \
    > /tmp/websockify.log 2>&1 < /dev/null &
WEB_PID=$!
disown
sleep 2
if ! kill -0 $WEB_PID 2>/dev/null; then
    echo "❌ Websockify không khởi động được. Log:"; tail -20 /tmp/websockify.log; exit 1
fi
echo "✅ Websockify PID=$WEB_PID"

# ── 5. Launch app ──────────────────────────────────────
export DISPLAY=$XVFB_DISPLAY

if [ ! -x "$VENV_PYTHON" ]; then
    echo "❌ Không tìm thấy Python: $VENV_PYTHON"
    exit 1
fi

# Prefer activating the conda env if we landed on its python. This keeps
# library search paths (cudnn etc.) consistent with the gpu wheels.
if [ -z "${VENV_DIR:-}" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
elif [ -n "${VENV_DIR:-}" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi
APP_LOG="${APP_LOG:-/tmp/classroom_manager.log}"

echo "✅ Python: $VENV_PYTHON"
"$VENV_PYTHON" - <<'PY' || true
import onnxruntime as ort
print(f"   ONNX providers: {ort.get_available_providers()}")
try:
    import torch
    print(f"   torch CUDA   : {torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
except Exception:
    pass
PY

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🚀  Khởi động Classroom Manager..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
IP="$(hostname -I | awk '{print $1}')"
echo "  🌐 Web VNC (đầy đủ): http://$IP:$WEB_PORT/vnc.html"
echo "  🌐 Web VNC (gọn):    http://$IP:$WEB_PORT/vnc_lite.html"
echo "  📺 VNC viewer:       $IP:$VNC_PORT  (no password)"
echo "  💡 Nếu trình duyệt báo lỗi UI, hãy Ctrl+Shift+R (hard refresh) để xoá cache."
echo ""

nohup "$VENV_PYTHON" -u "$SCRIPT_DIR/main.py" > "$APP_LOG" 2>&1 &
APP_PID=$!
sleep 2
if ! kill -0 "$APP_PID" 2>/dev/null; then
    echo "❌ Classroom Manager thoát ngay sau khi khởi động. Log: $APP_LOG"
    tail -80 "$APP_LOG" 2>/dev/null || true
    exit 1
fi

echo "✅ Classroom Manager PID=$APP_PID"
echo "  📝 App log:  $APP_LOG"
echo ""
echo "Nhấn Ctrl+C để dừng script; VNC và app vẫn chạy nền."
wait "$APP_PID"
