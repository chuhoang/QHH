# §6c. Pitch Limit Line (v2 — đã sửa theo review) — giới hạn cao độ bằng 2 điểm trên

## Review v1 của phần bổ sung này

| # | Vấn đề v1 | Fix v2 |
|---|---|---|
| 1 | "Tia cắt TL-TR **trước** L-R" không bao giờ xảy ra: TL-TR nằm trên L-R, tia đi lên từ O luôn cắt L-R trước | Bỏ hoàn toàn cơ chế so thứ tự giao điểm |
| 2 | Tia 2D không có "điểm dừng" — nhìn giữa bảng và nhìn trần ngay trên bảng cho CÙNG một tia → `if hit_top: LookingUp` gán oan cho người đang nhìn bảng | Không dùng intersection cho cao độ; dùng **ngưỡng góc pitch** suy từ vị trí đường TL-TR (pitch là thông tin 3D từ gaze model, không mất khi chiếu 2D) |
| 3 | TL-TR hẹp hơn L-R → ngước trần lệch trái thì Focused, ngước giữa thì LookingUp | Đường giới hạn dùng theo **cao độ y**, không phụ thuộc x_hit trong đoạn hẹp |

Giữ từ v1: ý tưởng 2 điểm trên do người dùng chọn, lưu config per-camera, trạng thái `LookingUp` riêng, opt-in không phá logic cũ.

---

## Vai trò 4 điểm

```text
TL ------------------ TR      ← giới hạn CAO ĐỘ (pitch): ngước quá đường này = LookingUp
        (y = T_y)

L -------------------- R      ← giới hạn YAW (trái/phải): ray-hit §6b như cũ
        (y = L_y)
```

- **2 điểm dưới (L, R)**: giữ nguyên §6b — tia gaze giao đường L-R, `x_hit ∈ [Lx, Rx] ± margin` → đạt yaw.
- **2 điểm trên (TL, TR)**: định nghĩa **pitch tối đa cho từng học sinh** theo vị trí của họ trong ảnh.

## Cơ chế pitch limit (thay ray-intersection)

Xấp xỉ pinhole: 1 pixel dọc ≈ `VFOV / H` radian. Học sinh ở `O` muốn nhìn tới cao độ
đường TL-TR phải ngước một góc xấp xỉ:

```python
PITCH_PER_PIX = VFOV_RAD / frame_H          # VFOV camera, config per-camera (default 55°)
alpha_top     = (O.y - T_y) * PITCH_PER_PIX # góc ngước tối đa cho phép của học sinh này

if pitch > alpha_top:                       # pitch RADIAN, dương = ngước lên
    return "LookingUp"
```

Tính chất đúng với trực giác:
- Học sinh ngồi **xa phía dưới** đường TL-TR (`O.y - T_y` lớn) → được ngước nhiều hơn.
- Học sinh có mặt **gần sát** đường giới hạn → `alpha_top` nhỏ → chặn chặt.
- Mỗi người một ngưỡng, tự thay đổi theo vị trí — cùng triết lý wedge yaw, không cần calibration 3D đầy đủ (chỉ cần VFOV gần đúng của camera).

`x` của TL/TR chỉ dùng để **vẽ** đường giới hạn trên UI/visualize; phần logic dùng `T_y` (đường ngang).

## Thứ tự trạng thái (cập nhật §8)

```python
def attention_state(O, yaw, pitch, L, R, T_y):
    dx, dy = -sin(yaw)*cos(pitch), -sin(pitch)
    if hypot(dx, dy) < sin(CENTRAL_CONE_RAD):
        return "Focused"                                    # §5 central cone
    if WRITING_ON and pitch < WRITING_PITCH_RAD and abs(yaw) < radians(20):
        return "Writing"                                    # §7
    if pitch > (O[1] - T_y) * PITCH_PER_PIX:
        return "LookingUp"                                  # §6c — cao độ
    return "Focused" if gaze_hits_board(O, (dx, dy), L, R) else "Distracted"  # §6b — yaw
```

| Điều kiện | Kết quả |
|-----------|---------|
| Central cone | Focused |
| Writing (opt-in) | Writing |
| pitch > alpha_top | LookingUp |
| Ray hit đoạn L-R | Focused |
| Còn lại | Distracted |

`LookingUp` đếm riêng (như Writing); có cộng vào distracted_frames hay không là quyết định config (`lookup_counts_distracted`, default true — ngước trần thường là mất tập trung thật).

## Config

```json
{
  "boardLine":  { "L": [350, 120], "R": [1550, 120] },
  "pitchLimit": { "TL": [820, 60], "TR": [1100, 60] },
  "cameraVfovDeg": 55
}
```

- Thiếu `pitchLimit` → bỏ qua check LookingUp (backward compatible).
- `cameraVfovDeg` sai lệch ±10° chỉ làm ngưỡng lệch tỉ lệ — tune bằng mắt qua visualize là đủ.

## Hạn chế

- Xấp xỉ tuyến tính pixel→góc chỉ đúng tốt ở vùng giữa frame với lens thường; lens fisheye
  (như camera lớp đang test) lệch nhiều ở mép — cần tune `cameraVfovDeg` theo camera, hoặc
  undistort trước nếu đòi chính xác cao.
- `pitch` từ gaze model có nhiễu ±5-10° với mặt nhỏ — LookingUp nên đi kèm khử nhiễu
  N-frame liên tiếp như snapshot (§8).
