#!/bin/bash
# Load config.json into environment variables. Existing environment wins.

QHH_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QHH_CONFIG_FILE="${QHH_CONFIG_FILE:-${QHH_ROOT_DIR}/config.json}"

eval "$(
python3 - "$QHH_CONFIG_FILE" <<'PY'
import json
import os
import shlex
import sys

config_file = sys.argv[1]
try:
    with open(config_file, "r", encoding="utf-8") as handle:
        config = json.load(handle)
except (OSError, ValueError) as exc:
    raise SystemExit("Không đọc được QHH config %s: %s" % (config_file, exc))

mapping = {
    "REDIS_HOST": ("redis", "host", "192.168.6.170"),
    "REDIS_PORT": ("redis", "port", 6377),
    "REDIS_DB": ("redis", "db", 0),
    "REDIS_PASSWORD": ("redis", "password", ""),
    "QHH_REDIS_PREFIX": ("redis", "prefix", "qhh"),
    "QHH_RECORD_DIR": ("recording", "root_dir", "recordings"),
    "QHH_WEB_RECORD_ON_AI": ("web_record", "on_ai", True),
    "QHH_WEB_RECORD_DURATION_SEC": ("web_record", "duration_sec", 10),
    "QHH_WEB_RECORD_INTERVAL_SEC": ("web_record", "interval_sec", 60),
    "QHH_AI_AUTO_START": ("ai", "auto_start", False),
    "QHH_VIDEO_READER_MAX_FPS": ("ai", "video_reader_max_fps", 25),
    "FPS_LOG": ("ai", "fps_log", True),
    "FPS_LOG_PERIOD": ("ai", "fps_log_period", 2.0),
    "ARC_LOG": ("ai", "arc_log", False),
    "QHH_AI_RESULT_VIDEO_ON": ("ai", "result_video_on", True),
    "QHH_AI_RESULT_DIR": ("ai", "result_dir", "result_test"),
    "QHH_AI_RESULT_VIDEO_FPS": ("ai", "result_video_fps", 25),
    "QHH_AI_RESULT_VIDEO_QUEUE": ("ai", "result_video_queue", 8),
    "QHH_AI_RESULT_VIDEO_CODEC": ("ai", "result_video_codec", "mp4v"),
    "QHH_AI_RESULT_VIDEO_EXT": ("ai", "result_video_ext", ".mp4"),
    "AI_DRAW_ALL_DETECTIONS": ("ai", "draw_all_detections", True),
    "QHH_AI_DELETE_PROCESSED_VIDEO": ("ai", "delete_processed_video", True),
    "QHH_AI_SNAPSHOT_ON": ("ai", "snapshot_on", False),
    "QHH_WEB_LIVE_PREVIEW_ON": ("ai", "live_preview_on", False),
    "QHH_WEB_HOST": ("web", "host", "0.0.0.0"),
    "QHH_WEB_PORT": ("web", "port", 8090),
}

for env_name, (section, key, default) in mapping.items():
    if env_name in os.environ:
        continue
    section_data = config.get(section, {})
    value = section_data.get(key, default) if isinstance(section_data, dict) else default
    print("export %s=%s" % (env_name, shlex.quote(str(value))))
PY
)"
