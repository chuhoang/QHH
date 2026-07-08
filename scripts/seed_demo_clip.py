"""Copy AVA_face.avi into videos/ with the cam{cid}_class{clsid}_{ts} naming
convention so the scheduler can pick it up. Used for end-to-end smoke tests.

Usage:
    python scripts/seed_demo_clip.py                       # default GUIDs
    python scripts/seed_demo_clip.py --cam <guid> --cls <guid>
    python scripts/seed_demo_clip.py --n 3                 # seed 3 clips
"""

from __future__ import annotations

import argparse
import shutil
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "AVA_face.avi"
VIDEOS_DIR = ROOT / "videos"

DEFAULT_CAM = "9b1c0000-0000-0000-0000-000000000001"
DEFAULT_CLS = "3f7e0000-0000-0000-0000-000000000002"


def seed_one(cam: str, cls: str, epoch_ms: int) -> Path:
    if not SAMPLE.exists():
        raise FileNotFoundError(f"sample clip missing: {SAMPLE}")
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"cam{cam}_class{cls}_{epoch_ms}{SAMPLE.suffix}"
    final = VIDEOS_DIR / name
    tmp = VIDEOS_DIR / (name + ".part")
    shutil.copyfile(SAMPLE, tmp)
    tmp.rename(final)  # atomic — scheduler only sees the final name
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default=DEFAULT_CAM)
    ap.add_argument("--cls", default=DEFAULT_CLS)
    ap.add_argument("--n", type=int, default=1, help="number of clips to seed")
    ap.add_argument("--random-cam", action="store_true",
                    help="generate a random cameraId for each clip")
    args = ap.parse_args()

    base_ms = int(time.time() * 1000)
    for i in range(args.n):
        cam = str(uuid.uuid4()) if args.random_cam else args.cam
        path = seed_one(cam, args.cls, base_ms + i)
        print(f"seeded: {path}")


if __name__ == "__main__":
    main()
