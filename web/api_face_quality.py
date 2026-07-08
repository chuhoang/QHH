"""API kiểm tra chất lượng ảnh khuôn mặt khi đăng ký người học.

Endpoint:
    POST /api/students/{studentId}/face-quality-check

Body chấp nhận một trong các dạng:
    • multipart/form-data với field file ảnh ("avatar"/"file"/"image") hoặc
      field "avatarUrl" trỏ tới ảnh trên mạng;
    • JSON: { "avatarUrl": "https://..." } hoặc
            { "face_data": "data:image/jpeg;base64,..." }.

Response JSON:
{
  "id": "<student-guid>",
  "studentCode": "HS001",
  "fullName": "Nguyễn Văn A",
  "avatarUrl": "https://...",
  "passed": false,
  "faceCount": 1,
  "qualified": false,
  "issues": [
    { "code": "BLURRY_FACE",       "message": "Khuôn mặt bị mờ." },
    { "code": "NOSE_NOT_VISIBLE",  "message": "Không thấy rõ mũi." }
  ]
}
"""

from __future__ import annotations

import base64
import io
import urllib.request
from typing import Any

import cv2
import numpy as np

from db import redis_client as db


# ---------------------------------------------------------------------------
# Ngưỡng đánh giá. Cố tình tách hằng ra để dễ tinh chỉnh theo dataset.
# ---------------------------------------------------------------------------
MIN_FACE_SIZE_PX = 80          # mặt nhỏ hơn => FACE_TOO_SMALL
MIN_LAPLACIAN_VAR = 100.0       # Laplacian variance sau khi denoise (NL-Means) < => BLURRY_FACE
                                # Denoise loại grain/noise phim trước khi đo → tách đúng motion-blur
                                # khỏi ảnh chụp hợp lệ nhưng có noise.
                                # Calibration: blurr-stock=42, AVA=357, Hiep=530 → threshold=100.
MIN_BRIGHTNESS = 40             # Trung bình kênh V < => LOW_LIGHT
MAX_BRIGHTNESS = 235            # > => OVERBRIGHT
MAX_FACE_RATIO_LOW = 0.02       # bbox/ảnh < => FACE_TOO_SMALL
MAX_DET_CONF_MIN = 0.7          # confidence < => LOW_CONFIDENCE (phải khớp conf_thres RetinaFace)
MAX_NOSE_OFFSET_RATIO = 0.30    # nose lệch ra ngoài [25%, 75%] giữa 2 mắt
MAX_EYE_TILT_DEG = 25.0         # nghiêng đầu quá lớn


ISSUE_MESSAGES = {
    "NO_FACE": "Không tìm thấy khuôn mặt trong ảnh.",
    "MULTIPLE_FACES": "Có nhiều hơn một khuôn mặt trong ảnh.",
    "FACE_TOO_SMALL": "Khuôn mặt quá nhỏ so với khung hình.",
    "BLURRY_FACE": "Khuôn mặt bị mờ.",
    "LOW_LIGHT": "Ảnh quá tối.",
    "OVERBRIGHT": "Ảnh quá sáng / cháy sáng.",
    "LOW_CONFIDENCE": "Độ tin cậy nhận diện khuôn mặt thấp.",
    "NOSE_NOT_VISIBLE": "Không thấy rõ mũi (mặt nghiêng hoặc bị che).",
    "EYES_NOT_VISIBLE": "Không thấy rõ hai mắt.",
    "MOUTH_NOT_VISIBLE": "Không thấy rõ miệng.",
    "HEAD_TILTED": "Đầu nghiêng quá nhiều.",
    "INVALID_IMAGE": "Không đọc được dữ liệu ảnh đầu vào.",
}


# ---------------------------------------------------------------------------
# Load ảnh
# ---------------------------------------------------------------------------
def _decode_image_bytes(raw: bytes) -> np.ndarray | None:
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size == 0:
        return None
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def load_image(*, file_bytes: bytes | None = None,
               data_url: str | None = None,
               http_url: str | None = None) -> np.ndarray | None:
    """Hỗ trợ 3 đường vào: file upload, data URL base64, hoặc URL HTTP."""
    if file_bytes:
        return _decode_image_bytes(file_bytes)
    if data_url:
        payload = data_url.split(",", 1)[-1] if "," in data_url else data_url
        try:
            return _decode_image_bytes(base64.b64decode(payload))
        except Exception:
            return None
    if http_url:
        try:
            with urllib.request.urlopen(http_url, timeout=8) as resp:
                if int(resp.headers.get("Content-Length", "0") or 0) > 12 * 1024 * 1024:
                    return None
                return _decode_image_bytes(resp.read(12 * 1024 * 1024))
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Đánh giá chất lượng
# ---------------------------------------------------------------------------
def _add_issue(issues: list[dict], code: str, override: str | None = None) -> None:
    issues.append({"code": code, "message": override or ISSUE_MESSAGES.get(code, code)})


def _face_quality_issues(image: np.ndarray, detections: list[dict]) -> list[dict]:
    """Check 3 điều kiện song song — có thể trả NHIỀU mã đồng thời:
        1. Không có khuôn mặt → NO_FACE
        2. Nhiều hơn 1 khuôn mặt → MULTIPLE_FACES (vẫn check blur trên mặt chính)
        3. Khuôn mặt bị mờ → BLURRY_FACE

    Ví dụ: ảnh 3 mặt + tất cả mờ → trả [MULTIPLE_FACES, BLURRY_FACE].
    """
    issues: list[dict] = []
    h, w = image.shape[:2]

    if not detections:
        _add_issue(issues, "NO_FACE")
        return issues

    if len(detections) > 1:
        _add_issue(issues, "MULTIPLE_FACES")
        # KHÔNG return — vẫn check blur tiếp trên mặt confidence cao nhất.

    # Chọn mặt có conf cao nhất để đo các chỉ số tiếp theo.
    det = max(detections, key=lambda d: float(d.get("conf", 0.0)))

    loc = det.get("loc")
    if loc is None or len(loc) < 4:
        if not any(i["code"] == "NO_FACE" for i in issues):
            _add_issue(issues, "NO_FACE")
        return issues

    x1, y1, x2, y2 = (float(v) for v in loc[:4])
    cx1 = max(0, int(x1)); cy1 = max(0, int(y1))
    cx2 = min(w, int(x2)); cy2 = min(h, int(y2))
    if cx2 <= cx1 or cy2 <= cy1:
        if not any(i["code"] == "NO_FACE" for i in issues):
            _add_issue(issues, "NO_FACE")
        return issues

    # Kiểm tra kích thước mặt tối thiểu.
    face_w = cx2 - cx1
    face_h = cy2 - cy1
    face_area_ratio = (face_w * face_h) / (w * h)
    if face_w < MIN_FACE_SIZE_PX or face_h < MIN_FACE_SIZE_PX:
        _add_issue(issues, "FACE_TOO_SMALL")
    elif face_area_ratio < MAX_FACE_RATIO_LOW:
        _add_issue(issues, "FACE_TOO_SMALL")

    crop = image[cy1:cy2, cx1:cx2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Denoise trước để loại grain/film noise, sau đó đo Laplacian variance.
    # Grain làm tăng giả tạo variance khiến ảnh motion-blur vẫn pass nếu dùng raw/GaussianBlur.
    face_resized = cv2.resize(gray, (112, 112))
    denoised = cv2.fastNlMeansDenoising(face_resized, h=10)
    if float(cv2.Laplacian(denoised, cv2.CV_64F).var()) < MIN_LAPLACIAN_VAR:
        _add_issue(issues, "BLURRY_FACE")

    return issues


# ---------------------------------------------------------------------------
# Entry chính dùng cho web_server
# ---------------------------------------------------------------------------
def check_face_quality(
    student_id: str,
    *,
    file_bytes: bytes | None = None,
    data_url: str | None = None,
    avatar_url: str | None = None,
    face_detector,
) -> dict[str, Any]:
    """Chạy các bước check chất lượng và đóng gói payload trả ra HTTP."""
    student = db.get_student(student_id) if student_id else {}

    image = load_image(
        file_bytes=file_bytes,
        data_url=data_url,
        http_url=avatar_url,
    )

    issues: list[dict] = []
    face_count = 0

    if image is None:
        _add_issue(issues, "INVALID_IMAGE")
    else:
        try:
            detections = face_detector.detect_all(image)
        except Exception as exc:  # noqa: BLE001 — đẩy lỗi runtime ra issues
            detections = []
            issues.append({"code": "DETECTOR_ERROR", "message": str(exc)})
        face_count = len(detections)
        issues.extend(_face_quality_issues(image, detections))

    qualified = (face_count == 1) and not issues
    passed = qualified  # Hiện 2 cờ trùng nghĩa; tách để client dễ mở rộng sau.

    return {
        "id": student_id,
        "studentCode": student.get("student_code") or student.get("studentCode") or "",
        "fullName": student.get("name") or student.get("fullName") or "",
        "avatarUrl": avatar_url or "",
        "passed": bool(passed),
        "faceCount": int(face_count),
        "qualified": bool(qualified),
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Parse multipart đơn giản (đủ dùng cho image + 1 vài field text)
# ---------------------------------------------------------------------------
def parse_multipart(body: bytes, content_type: str) -> dict[str, Any]:
    """Trả về {"files": {name: bytes}, "fields": {name: str}}."""
    if "boundary=" not in content_type:
        return {"files": {}, "fields": {}}
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
    sep = ("--" + boundary).encode()
    parts = body.split(sep)
    files: dict[str, bytes] = {}
    fields: dict[str, str] = {}
    for part in parts:
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if b"\r\n\r\n" not in part:
            continue
        header_blob, data = part.split(b"\r\n\r\n", 1)
        data = data.rstrip(b"\r\n")
        headers = header_blob.decode("utf-8", errors="replace")
        name = ""
        filename = ""
        for line in headers.split("\r\n"):
            if line.lower().startswith("content-disposition"):
                for kv in line.split(";"):
                    kv = kv.strip()
                    if kv.startswith("name="):
                        name = kv[5:].strip('"')
                    elif kv.startswith("filename="):
                        filename = kv[9:].strip('"')
        if not name:
            continue
        if filename:
            files[name] = data
        else:
            fields[name] = data.decode("utf-8", errors="replace")
    return {"files": files, "fields": fields}
