# Redis — Tổng hợp key cho bên thứ 3

Tài liệu tham chiếu các key Redis do QHH SIS ghi ra, phục vụ hệ thống bên thứ 3 (AI điểm danh, dashboard, tích hợp nội bộ).

Prefix chung: **`qhh:`**. Payload JSON dùng **camelCase** trừ khi ghi chú khác.

---

## Kết nối

| Môi trường | Host | Mật khẩu |
|------------|------|----------|
| Docker local (host) | `localhost:6377` | `qhh_redis_change_me` hoặc biến `REDIS_PASSWORD` |
| Trong Docker network | `qhh-redis:6379` | cùng mật khẩu |

```bash
redis-cli -a <password> -h localhost -p 6377
```

Cấu hình server: `Redis__ConnectionString` trong `docker-compose.yml`.

---

## 1. Người dùng (hồ sơ tối giản)

Ghi khi tạo/sửa/xóa user, đồng bộ SoGDDT, và backfill lúc khởi động `qhh-server`.

| Key | Kiểu Redis | Mô tả |
|-----|------------|-------|
| `qhh:users` | Set | Toàn bộ `userId` (GUID) |
| `qhh:users:teachers` | Set | `userId` giáo viên |
| `qhh:users:students` | Set | `userId` học sinh |
| `qhh:user:{userId}` | Hash | Hồ sơ từng user |

### Hash fields — `HGETALL qhh:user:{userId}`

| Field | Kiểu / giá trị | Mô tả |
|-------|----------------|-------|
| `id` | GUID | ID user |
| `username` | string | Tên đăng nhập |
| `fullName` | string | Họ tên |
| `avatar` | string | URL ảnh (có thể rỗng) |
| `userType` | string | `teacher` \| `student` \| `other` |

### Truy vấn mẫu

```bash
SMEMBERS qhh:users
SMEMBERS qhh:users:teachers
SMEMBERS qhh:users:students
HGETALL qhh:user:aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa
HGET qhh:user:{userId} userType
```

### Mã nguồn

- `backend/src/QHH.Server/Services/Caching/UserRedisCache.cs`
- `backend/shared/QHH.Common/Caching/UserCacheType.cs`

---

## 2. Tài khoản camera

Ghi khi CRUD tại **Quản trị Điểm danh → Tài khoản camera**, và backfill lúc khởi động `qhh-server`.

> **Lưu ý quan trọng:** Khác với user (có seed ~1400+ bản ghi), bảng `CameraAccounts` **không được seed**. Nếu chưa tạo camera qua UI/API, Redis **không có** `qhh:cameras` / `qhh:camera:*`. Log khởi động sẽ ghi `đã đồng bộ 0 camera` — đây là hành vi đúng, không phải lỗi.

| Key | Kiểu Redis | Mô tả |
|-----|------------|-------|
| `qhh:cameras` | Set | Toàn bộ `cameraId` (GUID); key chỉ xuất hiện sau lần `SADD` đầu tiên |
| `qhh:camera:{cameraId}` | Hash | Hồ sơ camera (kèm mật khẩu RTSP) |

### Hash fields — `HGETALL qhh:camera:{cameraId}`

| Field | Mô tả |
|-------|--------|
| `id` | GUID camera |
| `name` | Tên camera |
| `roomId` | GUID phòng (rỗng nếu không gán) |
| `roomName` | Tên phòng |
| `location` | Vị trí / mô tả |
| `ipAddress` | IP camera |
| `port` | Cổng RTSP (thường `554`) |
| `onvifPort` | Cổng ONVIF (thường `80`) |
| `username` | Tài khoản RTSP |
| `password` | Mật khẩu RTSP (plaintext) |
| `wsPort` | Cổng WebSocket proxy |
| `streamUrl` | URL stream (vd. `/cctv-ws/9998`) |
| `brand` | Hãng (Dahua, Hikvision, …) |
| `model` | Model |
| `isActive` | `1` hoạt động, `0` tắt |
| `notes` | RTSP path mặc định / ghi chú |
| `createdAt` | ISO 8601 |
| `updatedAt` | ISO 8601 (rỗng nếu chưa sửa) |

### Truy vấn mẫu

Thay `{cameraId}` bằng GUID thật lấy từ bước 1. **Bắt buộc** dùng `-a` và port `6377` khi kết nối từ host.

```bash
# 1. Có bao nhiêu camera? (0 = chưa tạo camera trong DB)
docker exec qhh-redis redis-cli -a qhh_redis_change_me SCARD qhh:cameras

# 2. Liệt kê ID camera
docker exec qhh-redis redis-cli -a qhh_redis_change_me SMEMBERS qhh:cameras

# 3. Đọc hồ sơ (thay GUID thật, không giữ {cameraId})
docker exec qhh-redis redis-cli -a qhh_redis_change_me HGETALL qhh:camera:ee0d2e94-5704-4ee8-b036-6cb0f1794e90

# Hoặc từng field
docker exec qhh-redis redis-cli -a qhh_redis_change_me HGET qhh:camera:<cameraId> ipAddress
docker exec qhh-redis redis-cli -a qhh_redis_change_me HGET qhh:camera:<cameraId> password

# Tìm key camera (tránh KEYS qhh:* — có hàng nghìn key user)
docker exec qhh-redis redis-cli -a qhh_redis_change_me KEYS "qhh:camera*"
```

### Mã nguồn

- `backend/src/QHH.Course/Services/CameraRedisCache.cs`
- `backend/src/QHH.Course/Controllers/CameraAccountController.cs`

> **Bảo mật:** Key `password` lưu plaintext trong Redis. Giới hạn quyền truy cập Redis cho consumer tin cậy.

---

## 3. Cấu hình camera theo lớp (AI điểm danh)

Ghi khi cấu hình **camera + lớp** (AI, kênh RTSP, vùng bàn, gán học sinh). **Không** tự tạo khi chỉ thêm tài khoản camera.

| Key | Kiểu Redis | Mô tả |
|-----|------------|-------|
| `qhh:attendance:camera-class:{cameraId}:{classId}` | String JSON | Cấu hình đầy đủ |
| `qhh:attendance:camera-class:index:{cameraId}` | Set | Danh sách `classId` (courseId) |
| `qhh:attendance:class-cameras:index:{classId}` | Set | Danh sách `cameraId` |

`classId` = `CourseId` trong hệ thống (một lớp chỉ gán tối đa một camera).

### JSON payload — `GET qhh:attendance:camera-class:{cameraId}:{classId}`

```json
{
  "cameraId": "guid",
  "classId": "guid",
  "classCode": "10A1",
  "aiEnabled": true,
  "rtspChannel": 1,
  "rtspPath": "/cam/realmonitor?channel=1&subtype=1",
  "regions": [
    {
      "id": "desk-1",
      "label": "Bàn 1",
      "x": 0.1,
      "y": 0.2,
      "w": 0.15,
      "h": 0.1,
      "mapX": null,
      "mapY": null,
      "mapW": null,
      "mapH": null,
      "studentIds": ["guid-hoc-sinh"]
    }
  ],
  "students": [
    {
      "id": "guid",
      "studentCode": "HS001",
      "fullName": "Nguyễn Văn A",
      "avatarUrl": "https://..."
    }
  ],
  "updatedAt": "2026-06-16T07:00:00.0000000Z"
}
```

| Field | Mô tả |
|-------|--------|
| `regions[].x,y,w,h` | Tọa độ ROI trên frame camera, chuẩn hóa `0..1` |
| `regions[].mapX,mapY,mapW,mapH` | Vị trí bàn trên bản đồ 2D (tùy chọn) |
| `regions[].studentIds` | Học sinh gán vào bàn |
| `students` | Thông tin HS được gán trong các vùng |

### Truy vấn mẫu

```bash
# Camera đang gán những lớp nào?
SMEMBERS qhh:attendance:camera-class:index:{cameraId}

# Cấu hình AI + ROI của một lớp
GET qhh:attendance:camera-class:{cameraId}:{classId}

# Lớp đang dùng camera nào?
SMEMBERS qhh:attendance:class-cameras:index:{classId}
```

### Mã nguồn

- `backend/src/QHH.Course/Services/CameraClassConfigCache.cs`
- `backend/src/QHH.Course/Services/ICameraClassConfigCache.cs`

---

## 4. Preview stream camera (tạm thời)

| Key | Kiểu | TTL |
|-----|------|-----|
| `qhh:cctv:preview:{cameraId}` | String JSON | 10 phút |

```json
{
  "rtspPath": "/cam/realmonitor?channel=1&subtype=1",
  "channel": 1,
  "expiresAt": "2026-06-16T07:10:00.0000000Z"
}
```

Dùng khi preview trong UI và bởi `qhh-cctv-proxy`. Bên thứ 3 thường dùng `rtspPath` từ key camera-class hoặc hash camera thay vì key preview.

---

## 5. Thời khóa biểu tuần

Ghi bởi `SystemBackgroundWorker` (định kỳ) và sau `POST /api/integration/timetable/sync`. API đọc TKB vẫn query DB.

| Key | Kiểu | TTL |
|-----|------|-----|
| `qhh:timetable:week:{isoWeek}:course:{courseId}` | String JSON | 14 ngày (cấu hình `TimetableCache:TtlDays`) |
| `qhh:timetable:week:{isoWeek}:course:{courseId}:meta` | String | Fingerprint — chỉ ghi lại payload khi dữ liệu đổi |

`isoWeek`: định dạng `2026-W25` (tuần thứ Hai–Chủ nhật, múi giờ Việt Nam).

### JSON payload TKB

```json
{
  "courseId": "guid",
  "isoWeek": "2026-W25",
  "weekStart": "2026-06-16",
  "weekEnd": "2026-06-22",
  "fingerprint": "12:1234567890:45",
  "generatedAt": "2026-06-16T00:05:00.0000000Z",
  "slots": [
    {
      "dayOfWeek": 1,
      "periodNumber": 1,
      "subjectId": "guid",
      "teacherId": "guid",
      "roomId": "guid",
      "startTime": "07:00",
      "endTime": "07:45",
      "status": "Active"
    }
  ]
}
```

```bash
GET qhh:timetable:week:2026-W25:course:{courseId}
GET qhh:timetable:week:2026-W25:course:{courseId}:meta
```

### Mã nguồn

- `backend/src/QHH.Server/Services/Caching/TimetableWeekCache.cs`

---

## Sơ đồ quan hệ (điểm danh AI)

```
qhh:cameras ──► qhh:camera:{cameraId}
                      │
                      ▼
        qhh:attendance:camera-class:index:{cameraId}
                      │
                      ▼
   qhh:attendance:camera-class:{cameraId}:{classId}
                      │
                      ▼
        qhh:attendance:class-cameras:index:{classId}

qhh:users ──► qhh:user:{userId}  (userType: teacher|student|other)
```

**Luồng tích hợp gợi ý:**

1. `SMEMBERS qhh:cameras` → lấy danh sách camera.
2. `HGETALL qhh:camera:{id}` → IP, user, password, RTSP.
3. `SMEMBERS qhh:attendance:camera-class:index:{cameraId}` → lớp gắn camera.
4. `GET qhh:attendance:camera-class:{cameraId}:{classId}` → ROI + học sinh.
5. (Tùy chọn) `HGET qhh:user:{userId}` để map user hệ thống; `studentCode` nằm trong payload `students` của camera-class.

---

## Key không dành cho bên thứ 3

| Key / pattern | Mục đích |
|---------------|----------|
| `qhh:idempotency:*` | Idempotency middleware (API nội bộ) |
| Key `IDistributedCache` (prefix `qhh:`) | Cache ASP.NET Core |
| SignalR Redis backplane | Realtime notification nội bộ |

---

## Đồng bộ & vận hành

| Sự kiện | Hành vi |
|---------|---------|
| Khởi động `qhh-server` | Backfill toàn bộ user + camera từ DB (upsert + gỡ bản ghi đã xóa) |
| CRUD user | Cập nhật `qhh:user:*` và set index ngay |
| CRUD tài khoản camera | Cập nhật `qhh:camera:*`; sửa camera cũng refresh `camera-class` nếu đã gán lớp |
| Cấu hình camera-lớp | Cập nhật `qhh:attendance:camera-class:*` |
| `docker compose up --build` | Sau rebuild `qhh-server`, frontend nginx resolve DNS động — không cần restart frontend vì IP đổi |

### Kiểm tra nhanh

```bash
# Log backfill lúc khởi động (camera = 0 nếu DB chưa có camera)
docker logs qhh-server 2>&1 | grep -E "Redis (user|camera) cache"

# Đếm nhanh — không dump hết 1400+ user
docker exec qhh-redis redis-cli -a qhh_redis_change_me SCARD qhh:users
docker exec qhh-redis redis-cli -a qhh_redis_change_me SCARD qhh:users:teachers
docker exec qhh-redis redis-cli -a qhh_redis_change_me SCARD qhh:users:students
docker exec qhh-redis redis-cli -a qhh_redis_change_me SCARD qhh:cameras
docker exec qhh-redis redis-cli -a qhh_redis_change_me KEYS "qhh:attendance:*"
```

---

## Rà soát trạng thái (audit)

Kết quả kiểm tra trên môi trường Docker local (`qhh-redis`, `qhh-server`):

| Nhóm key | Trạng thái | Ghi chú |
|----------|------------|---------|
| `qhh:users`, `qhh:user:*` | Hoạt động | Backfill ~1434 user; `qhh:users:teachers` / `students` có phân loại |
| `qhh:cameras`, `qhh:camera:*` | Hoạt động **khi DB có camera** | Lúc khởi động đầu tiên = 0 camera; sau CRUD tạo camera → key xuất hiện ngay |
| `qhh:attendance:camera-class:*` | Hoạt động **khi gán camera–lớp** | Không tự sinh khi chỉ thêm tài khoản camera |
| `qhh:cctv:preview:*` | Hoạt động | TTL 10 phút; chỉ có sau gọi preview stream |
| `qhh:timetable:week:*` | Hoạt động | Background worker / API sync TKB |

**Không phải lỗi code** nếu không thấy key camera: thường do (1) chưa tạo camera, (2) kết nối sai port/mật khẩu, (3) dùng `KEYS qhh:*` bị trôi trong hàng nghìn key user.

### Xử lý khi không thấy key camera

1. Xác nhận DB có bản ghi: UI **Quản trị Điểm danh → Tài khoản camera**, hoặc log `Redis camera cache saved for {CameraId}` sau khi tạo/sửa.
2. Kết nối đúng: `localhost:6377`, mật khẩu `qhh_redis_change_me` (hoặc `REDIS_PASSWORD` trong `.env`).
3. Dùng `SCARD qhh:cameras` thay vì `KEYS qhh:*`.
4. Restart `qhh-server` để chạy lại backfill sau khi đã có camera trong DB:
   `docker compose restart qhh-server`
5. Nếu vẫn 0: kiểm tra Redis có được cấu hình không (`Redis__ConnectionString` rỗng → cache bị bỏ qua im lặng).

---

## Liên quan

- [attendance-devices.md](./attendance-devices.md) — thiết bị điểm danh, camera, RFID
- [runtime.md](../01-architecture/runtime.md) — kiến trúc runtime
