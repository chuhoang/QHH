import sys; sys.path.insert(0, '/app')
import clip_inference as ci

result = ci.run_clip(
    '/app/detection/test_sample.avi',
    '989e86b1-9133-49a2-ad37-a215b70c083c',  # cameraId
    '10495f0c-9a46-4446-a8c4-ec283b6512b7',  # classId
)
import json; print(json.dumps(result, indent=2, ensure_ascii=False))
