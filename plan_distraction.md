# Gaze-based Classroom Attention Detection using Vector Inside Wedge (v2 — đã review)

## 0. Tóm tắt review v1

| # | Vấn đề trong v1 | Mức độ | Fix trong v2 |
|---|---|---|---|
| 1 | Nhìn thẳng camera → `g = (-sin(yaw)cos(pitch), -sin(pitch)) ≈ (0,0)` → normalize vector ~0 cho hướng toàn nhiễu, đúng lúc học sinh ĐANG nhìn bảng | **Chết người** | Thêm "central cone": `‖g‖ = sin(góc lệch trục camera)`; nếu `‖g‖ < sin(10°)` → Focused ngay, không cần wedge (§5) |
| 2 | Điều kiện `cross(v_left,g)>=0 AND cross(g,v_right)>=0` chỉ đúng với 1 chiều quay; ảnh dùng y-hướng-xuống nên dấu cross ngược với toán thường | Cao | Test theo dấu tham chiếu: cùng dấu với `cross(v_left, v_right)` (§6) |
| 3 | Wedge mở vô hạn — nhìn lên trần (vượt qua đoạn L-R) vẫn "inside" | Cao | Thay wedge bằng **giao điểm tia gaze với đường thẳng L-R**: bounded tự nhiên (§6b) |
| 4 | `O` = center bbox YOLO (cả thân người) — gaze xuất phát từ mặt, center thân lệch xuống ~nửa người | Trung bình | Dùng center **face box RetinaFace** (`det["loc"]`); fallback = điểm 1/6 trên của person bbox (§2) |
| 5 | Giả định ngầm không được nêu: phương pháp CHỈ đúng khi camera đặt gần bảng (cùng phía học sinh nhìn) | Cao | Nêu rõ thành điều kiện tiên quyết (§1) |
| 6 | Không nói yaw/pitch là radian hay độ — repo (`gaze_estimator.estimate_batch`) trả **radian** | Trung bình | Ghi rõ: đưa thẳng radian vào `sin()`, KHÔNG convert độ (§3) |
| 7 | Nhìn xuống vở ghi chép → pitch âm lớn → Distracted oan | Trung bình | Thêm trạng thái "writing" tùy chọn (§7) |
| 8 | Không nói L, R lưu ở đâu, integrate vào code nào | Trung bình | §9: lưu trong camera-class config Redis, tích hợp tại `clip_inference` (thay check `abs(yaw)>th`) |
| 9 | Không có khử nhiễu theo thời gian — 1 frame nhiễu = 1 frame distracted | Thấp | Đã có sẵn `distracted_ratio` aggregate; thêm yêu cầu N frame liên tiếp cho snapshot (§8) |

Những phần v1 **ổn, giữ nguyên**: ý tưởng vùng quan sát động theo vị trí từng học sinh; không cần calibration 3D/vị trí ghế; dùng đúng vector 2D mà repo vẽ gaze; margin an toàn; O(1) mỗi học sinh; cảnh báo camera nghiêng mạnh cần homography.

---

## 1. Objective & điều kiện tiên quyết

Thay vì ngưỡng `yaw`/`pitch` cố định cho toàn lớp, xây **vùng quan sát động** cho từng học sinh dựa trên vị trí của học sinh trong ảnh: học sinh Focused nếu gaze hướng về đoạn bảng L–R.

**Điều kiện tiên quyết (bắt buộc, v1 thiếu):**

- Camera **cố định** sau calibration.
- Camera **đặt gần bảng, hướng xuống lớp** (học sinh nhìn bảng ≈ nhìn về phía camera). Nếu camera đặt cuối lớp, phương pháp này sai hoàn toàn — gaze về bảng khi đó là quay LƯNG lại camera.
- Camera nghiêng quá mạnh (nhìn gần thẳng từ trên xuống) → cần homography, ngoài phạm vi v2.

---

## 2. Input

### Face box (KHÔNG dùng center person bbox)

Gaze xuất phát từ mặt. Dùng RetinaFace box (`det["loc"] = (x1, y1, x2, y2)`):

```python
O = ((x1 + x2) / 2, (y1 + y2) / 2)   # center face box
```

Fallback khi frame chỉ có person bbox (track fill): lấy điểm giữa theo x, và y tại 1/6 chiều cao tính từ đỉnh bbox (vùng đầu). Không dùng center thân người — lệch xuống ~nửa người làm wedge sai hẳn.

### Gaze Estimation

`GazeEstimator.estimate_batch(aligned_faces)` của repo trả `(yaw, pitch)` **RADIAN**.

### ROI bảng — hai điểm L, R

Người dùng chọn 1 lần trên frame camera:

```python
L = (350, 120)     # mép trái bảng (pixel)
R = (1550, 120)    # mép phải bảng
```

Lưu vào camera-class config (§9). Yêu cầu `L.x < R.x`.

---

## 3. Yaw/pitch → gaze vector 2D

Dùng đúng công thức repo đang vẽ mũi tên (camera_worker/visualize_clip):

```python
# yaw, pitch: RADIAN — đưa thẳng vào sin/cos, KHÔNG math.radians() lần nữa
dx = -sin(yaw) * cos(pitch)
dy = -sin(pitch)
g  = (dx, dy)
```

**Quan trọng (v1 sai):** KHÔNG normalize ngay. Độ dài `‖g‖ = sin(θ)` với `θ` = góc lệch giữa gaze và trục quang camera — chính độ dài này là tín hiệu "đang nhìn về camera", dùng ở §5.

---

## 4. Vùng quan sát của từng học sinh

```python
v_left  = L - O          # (Lx - Ox, Ly - Oy)
v_right = R - O
```

Mỗi học sinh một wedge riêng theo vị trí trong ảnh. Không cần vị trí ghế. (Giữ nguyên từ v1.)

---

## 5. Central cone — xử lý ca suy biến (MỚI, fix lỗi #1)

Khi học sinh nhìn thẳng camera: `yaw≈0, pitch≈0` → `g ≈ (0,0)`. Normalize vector ~0 → hướng ngẫu nhiên → kết quả nhiễu đúng lúc dễ nhất phải đúng.

Vì camera đặt gần bảng (§1), nhìn về camera ≈ nhìn về bảng:

```python
CENTRAL_CONE_RAD = radians(10)          # tune 8–15°

if hypot(dx, dy) < sin(CENTRAL_CONE_RAD):
    return "Focused"                    # nhìn gần trục camera → nhìn bảng
```

Chỉ khi `‖g‖` đủ lớn (hướng 2D đáng tin) mới đi tiếp bước wedge.

---

## 6. Kiểm tra gaze trong vùng

### 6a. Wedge test — sửa dấu cross (fix lỗi #2)

```python
def cross(a, b):
    return a[0]*b[1] - a[1]*b[0]
```

Ảnh dùng hệ y-hướng-xuống nên dấu cross ngược với hệ toán thường; ngoài ra thứ tự L/R so với O đổi chiều quay tùy vị trí. Test đúng mọi trường hợp: `g` nằm giữa `v_left` và `v_right` khi cả hai cross **cùng dấu với cross tham chiếu**:

```python
ref = cross(v_left, v_right)            # chiều quay từ v_left sang v_right
inside = (cross(v_left, g) * ref >= 0) and (cross(g, v_right) * ref >= 0)
```

(Điều kiện `>= 0` cứng của v1 chỉ đúng khi ref > 0.)

### 6b. Ray–line intersection — khuyến nghị dùng thay 6a (fix lỗi #3)

Wedge 6a mở vô hạn: nhìn lên trần (qua khỏi đoạn L-R) vẫn inside. Test tương đương nhưng bounded — bắn tia gaze từ O, tìm giao với đường ngang y = Ly:

```python
def gaze_hits_board(O, g, L, R, margin_px=80):
    dx, dy = g
    t = (L[1] - O[1]) / dy if abs(dy) > 1e-6 else None
    if t is None or t <= 0:
        return False                     # gaze song song/ngược hướng bảng
    x_hit = O[0] + dx * t
    return (L[0] - margin_px) <= x_hit <= (R[0] + margin_px)
```

- `t > 0` bảo đảm nhìn VỀ PHÍA bảng (loại nhìn xuống đất/ra sau).
- `x_hit ∈ [Lx, Rx]` bounded tự nhiên — nhìn trần hoặc quá mép trái/phải đều out.
- Margin tính bằng pixel trên đường bảng, trực quan hơn margin góc của v1.

Pipeline chuẩn: **§5 central cone → 6b ray hit**. 6a giữ làm tài liệu tham khảo.

---

## 7. Trạng thái "writing" (tùy chọn, fix lỗi #7)

Nhìn xuống vở (pitch âm lớn, gaze rơi trước mặt) là hành vi học tập, không nên tính Distracted:

```python
WRITING_PITCH_RAD = radians(-30)
if pitch < WRITING_PITCH_RAD and abs(yaw) < radians(20):
    return "Writing"        # đếm riêng, không cộng distracted_frames
```

Bật/tắt bằng config `writing_detection_on` (default off để giữ hành vi hiện tại).

---

## 8. Runtime pipeline

```
YOLO person bbox ─► RetinaFace face box ─► O = center face box
                                        └► aligned face ─► GazeEstimator ─► (yaw, pitch) RADIAN
(yaw, pitch) ─► g = (-sin·cos, -sin)  [KHÔNG normalize]
‖g‖ < sin(10°)?          ──► Focused (central cone)
pitch < -30° & |yaw|<20°? ──► Writing (nếu bật)
gaze_hits_board(O, g, L, R)? ──► Focused / Distracted
```

Khử nhiễu thời gian: giữ nguyên aggregate `distracted_frames / frames_present` như hiện tại; riêng **snapshot cảnh báo** yêu cầu distracted ≥ N frame liên tiếp (N=5, ~0.2s @25fps) mới chụp — tránh chụp theo 1 frame gaze nhiễu.

## Pseudo code tổng

```python
def attention_state(face_box, yaw, pitch, L, R):
    O  = ((face_box[0]+face_box[2])/2, (face_box[1]+face_box[3])/2)
    dx = -sin(yaw)*cos(pitch)
    dy = -sin(pitch)

    if hypot(dx, dy) < sin(CENTRAL_CONE_RAD):
        return "Focused"
    if WRITING_ON and pitch < WRITING_PITCH_RAD and abs(yaw) < radians(20):
        return "Writing"
    return "Focused" if gaze_hits_board(O, (dx, dy), L, R) else "Distracted"
```

---

## 9. Tích hợp vào hệ thống (MỚI, fix lỗi #8)

- **Lưu L, R**: thêm field vào camera-class config JSON (`qhh:attendance:camera-class:{cam}:{cls}`):
  ```json
  "boardLine": { "L": [350, 120], "R": [1550, 120] }
  ```
  Mỗi camera một boardLine (camera cố định). UI region-editor hiện có thể mở rộng để pick 2 điểm.
- **Điểm tích hợp**: `clip_inference.py`, vòng aggregate per-frame — thay check hiện tại
  `abs(yaw_f) > yaw_th or abs(pitch_f) > pitch_th` bằng `attention_state(...) == "Distracted"`.
  Cần `_detect` trả thêm face box pixel (đã có trong result) cùng yaw/pitch.
- **Fallback**: config KHÔNG có `boardLine` → giữ nguyên chế độ ngưỡng yaw/pitch cũ. Rollout từng camera, không breaking change.
- **Đơn vị**: `_detect` hiện trả `gaze_yaw_deg`/`gaze_pitch_deg` (độ) — convert về radian trước khi vào công thức, hoặc lấy radian gốc từ estimator. Chốt MỘT chỗ convert, tránh lỗi double-convert đã từng gặp.
- **Visualize**: `visualize_clip.py` vẽ thêm đoạn L-R + điểm `x_hit` để tune `CENTRAL_CONE_RAD` và `margin_px` bằng mắt.

---

## 10. Ưu điểm (giữ từ v1)

- Không cần calibration 3D, ExpectedYaw/PitchMap, vị trí ghế.
- Vùng quan sát tự thay đổi theo vị trí từng học sinh; O(1)/học sinh.
- v2 thêm: không còn ca suy biến nhìn-thẳng-camera; bounded theo đoạn bảng; rollout an toàn nhờ fallback.

## 11. Hạn chế còn lại

- Chỉ đúng khi camera gần bảng (§1) — camera cuối lớp cần phương pháp khác.
- Gaze vector và L-R phải cùng hệ tọa độ ảnh; dùng đúng vector 2D repo đang vẽ.
- Camera nghiêng mạnh (top-down) → cần homography/calibration bổ sung.
- Heuristic 2D: học sinh ngồi lệch mép trái nhìn chéo sang mép phải bảng có sai số hình chiếu — chấp nhận được với margin, nhưng nên verify bằng video thật từng vị trí ngồi.
