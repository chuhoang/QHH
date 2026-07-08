import sys, os
sys.path.insert(0, '/app')

# Bật ghi video annotated với face box + gaze arrow
os.environ["QHH_AI_RESULT_VIDEO_ON"] = "true"
os.environ["QHH_AI_RESULT_DIR"]       = "/app/detection"

import clip_inference as ci

VIDEO   = '/app/detection/video_lop_qhh.mp4'
CAM_ID  = '989e86b1-9133-49a2-ad37-a215b70c083c'
CLS_ID  = '10495f0c-9a46-4446-a8c4-ec283b6512b7'

# print(VIDEO)
# exit()
result = ci.run_clip(VIDEO, CAM_ID, CLS_ID)

import json
print("\n=== RESULT ===")
for s in result.get('students', []):
    print(f"  {s.get('studentCode')}: presence={s.get('presenceRatio'):.2f} "
          f"absence={s.get('absence')} attention={s.get('attention')}")

annotated = result.get('annotated_video') or result.get('annotatedVideo')
if annotated:
    print(f"\nAnnotated video: {annotated}")
else:
    # Tìm file mp4 vừa tạo trong detection/
    import hashlib
    md5 = hashlib.md5(VIDEO.encode()).hexdigest()
    print(f"\nAnnotated video: /app/detection/{md5}.mp4")
