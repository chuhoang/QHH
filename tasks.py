"""Celery tasks. See CELERY_LOCAL_PLAN_v2.md §4.4 for contract."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from celery_app import celery_app
import clip_queue as cq
import clip_inference as ci
from db import redis_client as db


RESULT_KEY = "qhh:ai:clip:result:{md5}"

# Log file ghi mỗi event POST lên API
_LOG_DIR = Path(os.getenv("QHH_EVENT_LOG_DIR", "/app/detection/event_logs"))
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_EVENT_LOG = _LOG_DIR / "snapshot_events.jsonl"


def _log_event(payload: dict, status: int | None, error: str | None = None) -> None:
    """Ghi 1 event ra file JSONL: mỗi dòng là 1 JSON gồm timestamp + status + payload."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url": SNAPSHOT_INGEST_URL,
        "httpStatus": status,
        "error": error,
        "payload": payload,
    }
    try:
        with open(_EVENT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[tasks] event log write failed: {exc}", flush=True)
INDEX_KEY = "qhh:ai:clip:index:{cam}:{cls}"
CALLBACK_KEY = "qhh:ai:clip:callback:{cam}:{cls}"
RESULT_TTL = 30 * 24 * 3600

# API ingest cố định — bắn classroom-snapshot mỗi khi xử lý xong 1 video.
SNAPSHOT_INGEST_URL = os.getenv(
    "SNAPSHOT_INGEST_URL",
    "https://qhh.test.mqsolutions.vn/api/attendance/ai/classroom-snapshot",
)
SNAPSHOT_INGEST_API_KEY = os.getenv(
    "SNAPSHOT_INGEST_API_KEY", "qhh_ai_ingest_change_me",
)
# Server ingest (test) đôi khi xử lý >10s → đặt read-timeout rộng hơn để
# tránh "read operation timed out" giả. Chỉnh qua SNAPSHOT_POST_TIMEOUT.
SNAPSHOT_POST_TIMEOUT = int(os.getenv("SNAPSHOT_POST_TIMEOUT", "60"))

# ── Redis retry queue cho event bắn API bị fail ───────────────────────────
# Mỗi item giữ payload + đường dẫn video để retry thành công thì mới xóa video.
SNAPSHOT_RETRY_KEY = "qhh:ai:snapshot:retry"        # List — chờ gửi lại (FIFO)
SNAPSHOT_DEADLETTER_KEY = "qhh:ai:snapshot:deadletter"  # List — quá hạn / hỏng
# Tiêu chí "bỏ qua": item sống quá SNAPSHOT_MAX_AGE_SEC (tính từ firstFailedAt)
# thì chuyển sang deadletter — KHÔNG phụ thuộc số lần thử hay lưu lượng video.
# Mặc định 24h; chịu được server ingest sập dài. Chỉnh qua env (giây).
SNAPSHOT_MAX_AGE_SEC = int(os.getenv("SNAPSHOT_MAX_AGE_SEC", str(24 * 3600)))
# Mỗi lần task xong, dọn tối đa N item tồn trong queue (piggyback flush).
SNAPSHOT_FLUSH_BATCH = int(os.getenv("SNAPSHOT_FLUSH_BATCH", "20"))


def _item_age_sec(item: dict) -> float:
    """Số giây kể từ firstFailedAt. Không parse được → coi như 0 (mới)."""
    ts = item.get("firstFailedAt", "")
    try:
        first = datetime.fromisoformat(ts)
        if first.tzinfo is None:
            first = first.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - first).total_seconds()
    except Exception:  # noqa: BLE001
        return 0.0

# Xóa video gốc sau khi xử lý xong (mặc định BẬT). Tắt bằng DELETE_PROCESSED_VIDEO=0.
DELETE_PROCESSED_VIDEO = os.getenv("QHH_AI_DELETE_PROCESSED_VIDEO", "true").strip().lower() in {"1", "true", "yes"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _delete_processed_video(r, video_path: str, md5: str, cam_id: str, cls_id: str) -> None:
    """Xóa file video gốc — CHỈ khi đã chắc chắn xử lý xong.

    Điều kiện an toàn (verify lại Redis trước khi unlink):
      1. result:{md5}        đã tồn tại  (kết quả đã lưu)
      2. index:{cam}:{cls}   đã chứa md5 (đã gắn tag/index)
      3. status:{md5}.status == SUCCESS
    Thiếu bất kỳ điều kiện nào → KHÔNG xóa, giữ video để xử lý lại.
    """
    if not DELETE_PROCESSED_VIDEO:
        return
    try:
        # 1. result đã lưu?
        if not r.exists(RESULT_KEY.format(md5=md5)):
            print(f"[tasks] skip delete — result missing for {md5}", flush=True)
            return
        # 2. index đã chứa md5? (zscore trả None nếu chưa có)
        if r.zscore(INDEX_KEY.format(cam=cam_id, cls=cls_id), md5) is None:
            print(f"[tasks] skip delete — index missing for {md5}", flush=True)
            return
        # 3. status == SUCCESS?
        status = r.hget(f"qhh:ai:clip:status:{md5}", "status")
        status = status.decode() if isinstance(status, bytes) else status
        if status != "SUCCESS":
            print(f"[tasks] skip delete — status={status} for {md5}", flush=True)
            return

        p = Path(video_path)
        if p.exists():
            p.unlink()
            print(f"[tasks] deleted processed video: {video_path}", flush=True)
    except Exception as exc:  # noqa: BLE001 — xóa lỗi không được làm fail task
        print(f"[tasks] delete video failed for {video_path}: {exc}", flush=True)


def _build_snapshot_payload(result: dict, classroom: dict) -> dict:
    """Map clip_inference result → schema `classroom-snapshot` ingest API.

    Mirror nguyên cấu trúc request đầu vào (cameraId, classId, classCode,
    aiEnabled, rtspChannel, rtspPath, regions[], updatedAt) + ghi đè mỗi
    student bằng kết quả AI: attention, absence, yaw, pitch.
    """
    students_in = {s["id"]: s for s in classroom.get("students", [])}
    ai_by_id = {s.get("studentId", ""): s for s in result.get("students", [])}

    # ── Gom ảnh kết quả (distraction snapshot) theo từng student ────────────
    # Mỗi ảnh đã lưu sẵn allDistractedIds = các HS mất tập trung trong frame đó
    # (clip_inference.py). Một HS mất tập trung nhiều frame → nhiều URL ảnh.
    snapshots = result.get("distractionSnapshots", []) or []
    all_image_urls: list[str] = []
    imgs_by_student: dict[str, list[str]] = {}
    for s in snapshots:
        if not isinstance(s, dict):
            continue
        url = str(s.get("url") or "").strip()
        if not url:
            continue
        all_image_urls.append(url)
        for sid in s.get("allDistractedIds", []) or []:
            imgs_by_student.setdefault(str(sid), []).append(url)

    students_out = []
    # Bắn theo đúng danh sách students trong config (kể cả ai không detect được).
    for meta in classroom.get("students", []):
        sid = meta.get("id", "")
        ai = ai_by_id.get(sid, {})
        absence = ai.get("attendanceStatus", "ABSENT") == "ABSENT"
        attention = False if absence else not ai.get("distractionAlert", False)

        # resultImageUrls per-student:
        #   • Có ảnh mất tập trung của HS này → list đúng các ảnh đó.
        #   • HS vắng mặt (không xuất hiện trong vid) → 1 ảnh ngẫu nhiên trong
        #     các ảnh đã lưu (để client vẫn có bằng chứng frame lớp học).
        student_imgs = imgs_by_student.get(sid, [])
        if not student_imgs and absence and all_image_urls:
            student_imgs = [random.choice(all_image_urls)]

        students_out.append({
            "id": sid,
            "studentCode": meta.get("studentCode", ""),
            "fullName": meta.get("fullName", ""),
            "avatarUrl": meta.get("avatarUrl"),
            "attention": attention,
            "absence": absence,
            "yaw": ai.get("avgGazeYawDeg"),
            "pitch": ai.get("avgGazePitchDeg"),
            "resultImageUrls": student_imgs,
        })

    # Chỉ giữ x,y,w,h trong regions — bỏ mapX/mapY/mapW/mapH.
    regions_out = []
    for reg in classroom.get("regions", []):
        regions_out.append({
            "id": reg.get("id"),
            "label": reg.get("label", ""),
            "x": reg.get("x"),
            "y": reg.get("y"),
            "w": reg.get("w"),
            "h": reg.get("h"),
            "studentIds": reg.get("studentIds", []),
        })

    return {
        "cameraId": result.get("cameraId") or classroom.get("cameraId"),
        "classId": result.get("classId") or classroom.get("classId"),
        "classCode": classroom.get("classCode", ""),
        "aiEnabled": classroom.get("aiEnabled", True),
        "rtspChannel": classroom.get("rtspChannel"),
        "rtspPath": classroom.get("rtspPath", ""),
        "regions": regions_out,
        # Mỗi student mang resultImageUrls riêng (ảnh mất tập trung của HS đó,
        # hoặc 1 ảnh ngẫu nhiên nếu HS vắng mặt).
        "students": students_out,
        "updatedAt": result.get("generatedAt") or classroom.get("updatedAt", ""),
    }


def _post_snapshot(payload: dict) -> bool:
    """POST snapshot lên ingest API. Trả True nếu server nhận (2xx), False nếu fail.

    KHÔNG nuốt lỗi âm thầm nữa: caller dựa vào giá trị trả về để quyết định
    có đẩy vào retry queue và có được phép xóa video gốc hay không.
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        SNAPSHOT_INGEST_URL,
        data=body,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "X-Ai-Ingest-Api-Key": SNAPSHOT_INGEST_API_KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=SNAPSHOT_POST_TIMEOUT) as resp:
            ok = 200 <= int(resp.status) < 300
            print(f"[tasks] snapshot POST → {resp.status} ({'ok' if ok else 'fail'})", flush=True)
            _log_event(payload, status=int(resp.status))
            return ok
    except Exception as exc:  # noqa: BLE001 — fail (kể cả timeout) → caller xử lý retry
        print(f"[tasks] snapshot POST failed: {exc}", flush=True)
        _log_event(payload, status=None, error=str(exc))
        return False


def _enqueue_snapshot_retry(r, payload: dict, video_path: str, md5: str,
                            cam_id: str, cls_id: str, last_error: str = "") -> None:
    """Đẩy 1 event fail vào Redis retry queue (giữ kèm video_path để xóa sau)."""
    item = {
        "payload": payload,
        "videoPath": video_path,
        "md5": md5,
        "cameraId": cam_id,
        "classId": cls_id,
        "attempts": 0,
        "firstFailedAt": _now_iso(),
        "lastFailedAt": _now_iso(),
        "lastError": last_error,
    }
    try:
        r.rpush(SNAPSHOT_RETRY_KEY, json.dumps(item, ensure_ascii=False))
        depth = r.llen(SNAPSHOT_RETRY_KEY)
        print(f"[tasks] snapshot queued for retry (depth={depth}) md5={md5}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[tasks] failed to enqueue snapshot retry: {exc}", flush=True)


def _flush_snapshot_retry(r, max_items: int = SNAPSHOT_FLUSH_BATCH) -> dict:
    """Gửi lại các event đang chờ trong retry queue.

    Với mỗi item lấy ra (LPOP):
      • POST lại → 2xx: xóa video gốc (nếu bật) — đây là lúc DUY NHẤT ngoài
        đường thành công trực tiếp mà video được phép xóa.
      • fail: tăng attempts; còn lượt → đẩy lại CUỐI queue, hết lượt → deadletter.
    Chỉ xử lý tối đa max_items để không chiếm worker quá lâu; phần còn lại để
    lượt flush sau (piggyback hoặc Celery Beat) dọn tiếp.
    """
    sent = retried = dead = 0
    # Chỉ duyệt số item ĐANG có lúc bắt đầu: item fail được requeue ở cuối queue
    # sẽ KHÔNG bị xử lý lại trong cùng pass này (tránh đốt hết attempts 1 lúc).
    pending = int(r.llen(SNAPSHOT_RETRY_KEY) or 0)
    budget = min(max(0, int(max_items)), pending)
    for _ in range(budget):
        raw = r.lpop(SNAPSHOT_RETRY_KEY)
        if raw is None:
            break
        try:
            item = json.loads(raw)
        except Exception:  # noqa: BLE001 — item hỏng → bỏ vào deadletter
            r.rpush(SNAPSHOT_DEADLETTER_KEY, raw)
            dead += 1
            continue

        payload = item.get("payload", {})
        ok = _post_snapshot(payload)
        if ok:
            sent += 1
            # Gửi được rồi MỚI xóa video gốc đã giữ lại.
            _delete_processed_video(
                r,
                item.get("videoPath", ""),
                item.get("md5", ""),
                item.get("cameraId", ""),
                item.get("classId", ""),
            )
            continue

        item["attempts"] = int(item.get("attempts", 0)) + 1
        item["lastFailedAt"] = _now_iso()
        item["lastError"] = "post failed"
        age = _item_age_sec(item)
        if age >= SNAPSHOT_MAX_AGE_SEC:
            # Quá hạn sống → bỏ qua (deadletter), không retry nữa.
            r.rpush(SNAPSHOT_DEADLETTER_KEY, json.dumps(item, ensure_ascii=False))
            dead += 1
            print(
                f"[tasks] snapshot → deadletter (age={age/3600:.1f}h ≥ "
                f"{SNAPSHOT_MAX_AGE_SEC/3600:.1f}h, attempts={item['attempts']}) "
                f"md5={item.get('md5')}",
                flush=True,
            )
        else:
            r.rpush(SNAPSHOT_RETRY_KEY, json.dumps(item, ensure_ascii=False))
            retried += 1

    if sent or retried or dead:
        print(
            f"[tasks] snapshot flush: sent={sent} requeued={retried} dead={dead} "
            f"remaining={r.llen(SNAPSHOT_RETRY_KEY)}",
            flush=True,
        )
    return {"sent": sent, "requeued": retried, "dead": dead,
            "remaining": r.llen(SNAPSHOT_RETRY_KEY)}


@celery_app.task(name="flush_snapshot_retry_queue", queue="clip")
def flush_snapshot_retry_queue(max_items: int = 200):
    """Task gọi tay / Celery Beat để chủ động dọn retry queue.

    Gọi tay:  celery -A celery_app call flush_snapshot_retry_queue
    hoặc:     flush_snapshot_retry_queue.apply_async()
    """
    r = db.get_client()
    return _flush_snapshot_retry(r, max_items=max_items)


@celery_app.task(
    bind=True,
    name="process_clip_task",
    max_retries=3,
    default_retry_delay=10,
    acks_late=True,
)
def process_clip_task(self, video_path: str, camera_id: str, class_id: str):
    cq.set_status(
        video_path, "PROCESSING",
        taskId=self.request.id,
        cameraId=camera_id, classId=class_id,
        attempts=self.request.retries,
    )
    try:
        result = ci.run_clip(video_path, camera_id, class_id)

        r = db.get_client()
        md5 = cq.md5_of(video_path)
        r.set(RESULT_KEY.format(md5=md5), json.dumps(result), ex=RESULT_TTL)

        meta = cq.parse_clip_name(Path(video_path))
        epoch_ms = meta["epochMs"] if meta else None
        if epoch_ms is None:
            # Fallback: parse epoch (s hoặc ms) cuối cùng trong stem filename.
            import re
            m = re.findall(r"(\d{10,13})", Path(video_path).stem)
            if m:
                v = int(m[-1])
                epoch_ms = v if v > 10**12 else v * 1000
        if epoch_ms is None:
            # Vẫn không có → dùng thời điểm hoàn thành.
            import time as _t
            epoch_ms = int(_t.time() * 1000)
        r.zadd(
            INDEX_KEY.format(cam=camera_id, cls=class_id),
            {md5: float(epoch_ms)},
        )

        cq.set_status(
            video_path, "SUCCESS",
            taskId=self.request.id,
            cameraId=camera_id, classId=class_id,
        )

        # Bắn classroom-snapshot lên ingest API cố định (luôn gửi mỗi video xong).
        raw_cfg = r.get(f"qhh:attendance:camera-class:{camera_id}:{class_id}")
        classroom = json.loads(raw_cfg) if raw_cfg else {}
        snapshot = _build_snapshot_payload(result, classroom)
        posted = _post_snapshot(snapshot)

        if posted:
            # Gửi được → xóa video gốc (result + index + status=SUCCESS đã ghi xong).
            _delete_processed_video(r, video_path, md5, camera_id, class_id)
        else:
            # Gửi FAIL → KHÔNG xóa video. Đưa event vào Redis retry queue, giữ
            # nguyên video để chỉ xóa khi retry queue gửi lại thành công.
            _enqueue_snapshot_retry(
                r, snapshot, video_path, md5, camera_id, class_id,
                last_error="initial post failed",
            )

        # Piggyback: tranh thủ dọn bớt event tồn trong retry queue từ các video trước.
        _flush_snapshot_retry(r, max_items=SNAPSHOT_FLUSH_BATCH)

        return {
            "md5": md5,
            "students": len(result.get("students", [])),
            "framesProcessed": result.get("framesProcessed", 0),
        }

    except Exception as exc:  # noqa: BLE001
        if self.request.retries < self.max_retries:
            cq.set_status(
                video_path, "RETRY",
                taskId=self.request.id,
                cameraId=camera_id, classId=class_id,
                error=str(exc), attempts=self.request.retries + 1,
            )
            raise self.retry(exc=exc)
        cq.set_status(
            video_path, "FAILED",
            taskId=self.request.id,
            cameraId=camera_id, classId=class_id,
            error=str(exc), attempts=self.request.retries + 1,
        )
        raise
