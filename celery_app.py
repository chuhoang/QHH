"""Celery app — broker + result backend trỏ Redis QHH (192.168.6.16:6378).

Worker chạy: celery -A celery_app worker -Q clip -P solo -c 1 -n clip-worker@%h
"""

from __future__ import annotations

from urllib.parse import quote

from celery import Celery

from config_loader import env_or_config


def _broker_url() -> str:
    host = str(env_or_config("REDIS_HOST", "redis", "host", "127.0.0.1"))
    port = int(env_or_config("REDIS_PORT", "redis", "port", 6379))
    db = int(env_or_config("REDIS_DB", "redis", "db", 0))
    password = env_or_config("REDIS_PASSWORD", "redis", "password", "") or ""
    auth = f":{quote(password, safe='')}@" if password else ""
    return f"redis://{auth}{host}:{port}/{db}"


BROKER_URL = _broker_url()

celery_app = Celery(
    "qhh_ai",
    broker=BROKER_URL,
    backend=BROKER_URL,
    include=["tasks"],
)

celery_app.conf.update(
    task_default_queue="clip",
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    result_expires=7 * 24 * 3600,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

# Celery Beat — chủ động dọn retry queue cho event bắn snapshot bị fail, kể cả
# khi không có video mới. Bật beat: celery -A celery_app beat --loglevel=info
# (xem service `beat` trong docker-compose.yml).
SNAPSHOT_FLUSH_INTERVAL = float(
    env_or_config("SNAPSHOT_FLUSH_INTERVAL", "celery", "snapshot_flush_interval", 60)
)
celery_app.conf.beat_schedule = {
    "flush-snapshot-retry-queue": {
        "task": "flush_snapshot_retry_queue",
        "schedule": SNAPSHOT_FLUSH_INTERVAL,
        "options": {"queue": "clip"},
    },
}
