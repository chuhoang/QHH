"""Seed full E2E data — chạy trong container qhh-ai-worker."""

import json
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, "/app")
from db import redis_client as db

r = db.get_client()

CAM = "CAM-E2E-001"
CLS = "CLS-E2E-001"
STU_AVA = "stu-e2e-ava"
STU_SON = "stu-e2e-son"
STU_HIEP = "stu-e2e-hiep"

# Cleanup
for k in [
    f"qhh:user:{STU_AVA}", f"qhh:user:{STU_SON}", f"qhh:user:{STU_HIEP}",
    f"qhh:attendance:camera-class:{CAM}:{CLS}",
    f"qhh:attendance:camera-class:index:{CAM}",
]:
    r.delete(k)

users = [
    {"id": STU_AVA, "username": "ava_e2e", "fullName": "AVA E2E",
     "avatar": "file:///home/mq/.classroom_manager/faces/stu-ava-0001.jpg",
     "userType": "student", "studentCode": "HS-AVA"},
    {"id": STU_SON, "username": "son_e2e", "fullName": "Son E2E",
     "avatar": "file:///home/mq/.classroom_manager/faces/stu-son-0002.jpg",
     "userType": "student", "studentCode": "HS-SON"},
    {"id": STU_HIEP, "username": "hiep_e2e", "fullName": "Hiep E2E",
     "avatar": "file:///home/mq/.classroom_manager/faces/stu-hiep-003.jpg",
     "userType": "student", "studentCode": "HS-HIEP"},
]
for u in users:
    sid = u["id"]
    r.hset(f"qhh:user:{sid}", mapping=u)
    r.sadd("qhh:users", sid)
    r.sadd("qhh:users:students", sid)

vn = timezone(timedelta(hours=7))
now = datetime.now(vn)
iso_y = now.isocalendar().year
iso_w = now.isocalendar().week
iso = f"{iso_y}-W{iso_w:02d}"
ss = (now - timedelta(minutes=20)).strftime("%H:%M")
se = (now + timedelta(minutes=40)).strftime("%H:%M")

tt = {
    "courseId": CLS,
    "isoWeek": iso,
    "slots": [{
        "dayOfWeek": now.isoweekday(),
        "periodNumber": 1,
        "subjectId": "subj-toan",
        "teacherId": "teach-99",
        "roomId": "room-E2E",
        "startTime": ss,
        "endTime": se,
        "status": "Active",
    }],
}
r.set(f"qhh:timetable:week:{iso}:course:{CLS}", json.dumps(tt, ensure_ascii=False))
r.sadd(f"qhh:attendance:camera-class:index:{CAM}", CLS)

cfg = {
    "cameraId": CAM, "classId": CLS, "classCode": "10A1", "aiEnabled": True,
    "rtspChannel": 1, "rtspPath": "/cam/realmonitor?channel=1&subtype=1",
    "regions": [
        {"id": "desk-1", "label": "Bàn 1", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "studentIds": [STU_AVA]},
        {"id": "desk-2", "label": "Bàn 2", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "studentIds": [STU_SON]},
        {"id": "desk-3", "label": "Bàn 3", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "studentIds": [STU_HIEP]},
    ],
    "students": [
        {"id": STU_AVA, "studentCode": "HS-AVA", "fullName": "AVA E2E"},
        {"id": STU_SON, "studentCode": "HS-SON", "fullName": "Son E2E"},
        {"id": STU_HIEP, "studentCode": "HS-HIEP", "fullName": "Hiep E2E"},
    ],
    "updatedAt": now.isoformat(),
}
r.set(f"qhh:attendance:camera-class:{CAM}:{CLS}", json.dumps(cfg, ensure_ascii=False))

print("SEEDED OK")
print("  iso_week =", iso, " slot:", ss, "->", se, "dow =", now.isoweekday())
print("  cameraId =", CAM)
print("  classId  =", CLS)
print("  students =", [u["id"] for u in users])
