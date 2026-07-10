"""POST /api/students/{id}/register-face

Flow tích hợp:
    1. Quality check ảnh khuôn mặt (reuse check_face_quality)
    2. Nếu qualified → lưu ảnh xuống /home/mq/.classroom_manager/faces/{id}.{ext}
    3. Build ArcFace embedding 512-d
    4. HSET vào qhh:user:{id}  (profile fields + avatar URL + embedding bytes)
    5. SADD qhh:users + qhh:users:students | qhh:users:teachers
    6. Trả response gồm cả kết quả quality check + trạng thái registration
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from db import redis_client as db
from web.api_face_quality import (
    check_face_quality,
    load_image,
)

FACE_DIR = Path(os.getenv(
    "FACE_REGISTRY_DIR",
    "/home/mq/.classroom_manager/faces",
)).resolve()
FACE_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_str(v) -> str:
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    return "" if v is None else str(v)


# Index code ↔ id để map nhanh ở 2 chiều.
_STUDENT_CODE_TO_ID = "qhh:face:student-code:{code}"  # studentCode → studentId


def _find_existing_user_id(r, student_code: str) -> str:
    """Tìm user qhh-server đã ghi sẵn có username/studentCode = student_code.

    Server ghi hồ sơ user vào `qhh:user:{id}` (id, username, fullName, ...)
    TRƯỚC khi AI đăng ký face. Nếu tìm thấy → reuse id đó để embedding gắn
    vào đúng hồ sơ gốc, không sinh UUID song song.
    """
    for k in r.scan_iter(match="qhh:user:*", count=500):
        key = _ensure_str(k)
        username = _ensure_str(r.hget(key, "username"))
        code = _ensure_str(r.hget(key, "studentCode"))
        if student_code in (username, code):
            sid = _ensure_str(r.hget(key, "id")) or key.rsplit(":", 1)[-1]
            return sid
    return ""


def get_or_create_student_id(student_code: str) -> str:
    """Trả về studentId stable cho studentCode.

    Thứ tự resolve:
    1. Index `qhh:face:student-code:{code}` đã có → trả id cũ (cập nhật
       embedding cùng id, không tạo mới).
    2. Chưa có index → quét `qhh:user:*` tìm hồ sơ server có
       username/studentCode trùng → reuse id đó (ghi index để lần sau khỏi quét).
    3. Không tìm thấy → sinh UUID v4 mới, ghi index, trả về.
    """
    if not student_code:
        return ""
    r = db.get_client()
    key = _STUDENT_CODE_TO_ID.format(code=student_code)

    cached = r.get(key)
    if cached:
        # Luôn reuse UUID cũ — kể cả khi qhh:user:{sid} mất (Redis restart).
        # Embedding sẽ được ghi lại vào đúng sid này, không tạo ID mới.
        return _ensure_str(cached)

    existing = _find_existing_user_id(r, student_code)
    if existing:
        r.set(key, existing)
        print(f"[face-register] studentCode={student_code} → reuse server user id={existing}", flush=True)
        return existing

    import uuid
    sid = str(uuid.uuid4())
    r.set(key, sid)
    return sid


def _save_face_image(student_id: str, image: np.ndarray) -> tuple[Path, int]:
    """Lưu ảnh chuẩn hoá JPEG vào registry, trả (path, size_bytes)."""
    dest = FACE_DIR / f"{student_id}.jpg"
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Encode ảnh thất bại")
    dest.write_bytes(encoded.tobytes())
    return dest, dest.stat().st_size


def _extract_embedding(
    image: np.ndarray,
    *,
    face_detector,
    arc_extractor,
) -> np.ndarray:
    """RetinaFace → align → ArcFace → L2-normalized 512-d vector."""
    dets = face_detector.detect_all(image)
    if not dets:
        raise RuntimeError("Không tìm thấy khuôn mặt khi extract embedding")
    # detect_all đã sort theo score giảm dần; lấy mặt to nhất.
    best = max(dets, key=lambda d: float(d.get("conf", 0.0)))
    aligned = best.get("aligned_face")
    if aligned is None:
        raise RuntimeError("Không align được khuôn mặt")
    emb = arc_extractor.extract([aligned])
    if emb is None or len(emb) == 0:
        raise RuntimeError("ArcFace không trả embedding")
    vec = np.asarray(emb[0], dtype=np.float32)
    n = float(np.linalg.norm(vec)) + 1e-9
    return vec / n


def register_user_with_face(
    student_id: str,
    *,
    file_bytes: bytes | None = None,
    data_url: str | None = None,
    avatar_url: str | None = None,
    student_code: str = "",
    full_name: str = "",
    username: str = "",
    user_type: str = "student",
    face_detector,
    arc_extractor,
) -> dict[str, Any]:
    """Quality check → lưu ảnh → embedding → HSET qhh:user:{id}.

    AI service tự quản trị toàn bộ field trong `qhh:user:{id}` (vì server
    có thể không có sẵn record này). Field face: avatar + embedding*.
    Field profile: studentCode, fullName, username, userType.
    """
    # 1. Quality check (re-use)
    qa = check_face_quality(
        student_id,
        file_bytes=file_bytes,
        data_url=data_url,
        avatar_url=avatar_url,
        face_detector=face_detector,
    )

    if not qa["qualified"]:
        # Trả về sớm, KHÔNG ghi Redis.
        qa["registered"] = False
        qa["reason"] = "QUALITY_CHECK_FAILED"
        return qa

    if not student_id:
        # Defensive: endpoint hiện đã yêu cầu studentId là path param.
        qa["registered"] = False
        qa["reason"] = "MISSING_STUDENT_ID"
        return qa

    # 2. Reload image lại — qa không expose image
    image = load_image(
        file_bytes=file_bytes,
        data_url=data_url,
        http_url=avatar_url,
    )
    if image is None:
        qa["registered"] = False
        qa["reason"] = "IMAGE_LOST"
        return qa

    # 3. Lưu ảnh xuống đĩa
    try:
        face_path, size = _save_face_image(student_id, image)
    except Exception as exc:  # noqa: BLE001
        qa["registered"] = False
        qa["reason"] = f"SAVE_IMAGE_FAILED: {exc}"
        return qa

    # 4. Extract embedding
    try:
        embedding = _extract_embedding(
            image,
            face_detector=face_detector,
            arc_extractor=arc_extractor,
        )
    except Exception as exc:  # noqa: BLE001
        qa["registered"] = False
        qa["reason"] = f"EMBEDDING_FAILED: {exc}"
        return qa

    # 5. HSET qhh:user:{sid} — profile + face.
    r = db.get_client()
    mtime_ns = face_path.stat().st_mtime_ns
    avatar_uri = f"file://{face_path}"
    mapping = {
        "id": student_id,
        "avatar": avatar_uri,
        "embedding": base64.b64encode(embedding.tobytes()).decode(),
        "embeddingDim": str(embedding.shape[0]),
        "embeddingMtimeNs": str(mtime_ns),
        "embeddingSize": str(size),
        "embeddingFlip": "false",
    }
    if student_code:
        mapping["studentCode"] = student_code
    if full_name:
        mapping["fullName"] = full_name
    if username:
        mapping["username"] = username
    if user_type:
        mapping["userType"] = user_type

    r.hset(f"qhh:user:{student_id}", mapping=mapping)
    r.sadd("qhh:users", student_id)
    if (user_type or "student") == "student":
        r.sadd("qhh:users:students", student_id)
    elif user_type == "teacher":
        r.sadd("qhh:users:teachers", student_id)

    qa["registered"] = True
    qa["reason"] = ""
    qa["avatarUrl"] = avatar_uri
    qa["studentId"] = student_id
    return qa
