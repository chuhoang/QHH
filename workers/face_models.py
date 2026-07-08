"""
Deep face detection (RetinaFace) + recognition (ArcFace R100) wrappers.

RetinaFace: self-contained preprocess/postprocess (copied from Face_recogition_v2
    src/service_ai/retinanet_det.py → preProcess_batch / postProcess_batch).
ArcFace: thin wrapper over src/service_ai/arcface_r100_onnx.py → get_feature_without_det.
Landmark alignment: src/libs/face_preprocess.py → face_preprocess.

Weights:
    weights/detectFace_model_op16.onnx  (RetinaFace)
    weights/arcface_r100.onnx           (ArcFace R100, 512-dim)
"""

from __future__ import annotations

import os
import sys
import threading
import warnings
from collections import OrderedDict
from math import ceil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as _ort

# Suppress ONNX Runtime warnings
_ort.set_default_logger_severity(3)

# ── Paths ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WORKERS_DIR = Path(__file__).resolve().parent
for _p in (str(_WORKERS_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

warnings.filterwarnings("ignore", category=FutureWarning)

from libs.face_preprocess import preprocess as _face_align
from sklearn.preprocessing import normalize as _l2_normalize

# ── Model paths ─────────────────────────────────────────────────────────
_WEIGHTS = _PROJECT_ROOT / "weights"
RETINAFACE_MODEL = str(_WEIGHTS / "detectFace_model_op16.onnx")
ARCFACE_MODEL = str(_WEIGHTS / "arcface_r100.onnx")

# Registration embeddings are immutable until their source image changes.
# Keep them across AI worker restarts to avoid re-running both face models.
_GALLERY_CACHE_MAX = 2048
_GALLERY_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_GALLERY_CACHE_LOCK = threading.Lock()


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _batch_enabled(session, env_name: str) -> bool:
    mode = os.getenv(env_name, "auto").strip().lower()
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False
    return "CUDAExecutionProvider" in session.get_providers()


# ── Configs ─────────────────────────────────────────────────────────────
# RETINAFACE_IMAGE_SIZE: network input (letterbox canvas). Default 640.
# Tăng lên 1280 giúp bắt mặt nhỏ/xa trên camera độ phân giải cao
# (PriorBox sinh anchor theo image_size nên chỉ cần đổi ở đây),
# đổi lại compute ~4x.
_RETINA_INPUT = int(os.getenv("RETINAFACE_IMAGE_SIZE", "640"))

CONFIG_RETINAFACE = {
    # RETINAFACE_MODEL_PATH: override sang bản ONNX dynamic-shape khi cần
    # input khác 640 (model gốc fix cứng 640x640).
    "model_path": os.getenv("RETINAFACE_MODEL_PATH", RETINAFACE_MODEL),
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "conf_thres": 0.7,
    "iou_thres": 0.4,
    "image_size": [_RETINA_INPUT, _RETINA_INPUT],
}

CONFIG_ARCFACE = {
    "model_path": ARCFACE_MODEL,
    "imgsz": [112, 112],
    "conf_thres": 0.75,
    "device": "cpu",
}


# ═══════════════════════════════════════════════════════════════════════════
# RetinaFaceDetector — self-contained preprocess / postprocess
# ═══════════════════════════════════════════════════════════════════════════

class PriorBox:
    """Generate anchor priors for RetinaFace (same as retinanet_det.PriorBox)."""

    def __init__(self, min_sizes, steps, clip, image_size):
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip
        self.image_size = image_size
        self.feature_maps = [
            [ceil(image_size[0] / s), ceil(image_size[1] / s)] for s in steps
        ]

    def forward(self):
        from itertools import product as _product
        anchors = []
        for k, f in enumerate(self.feature_maps):
            msz = self.min_sizes[k]
            for i, j in _product(range(f[0]), range(f[1])):
                for s in msz:
                    cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                    cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                    s_w = s / self.image_size[1]
                    s_h = s / self.image_size[0]
                    anchors += [cx, cy, s_w, s_h]
        out = np.array(anchors, dtype=np.float32).reshape(-1, 4)
        if self.clip:
            out = np.clip(out, 0, 1)
        return out


class RetinaFaceDetector:
    """RetinaFace ONNX face detector.

    Preprocess / postprocess copied from Face_recogition_v2:
      src/service_ai/retinanet_det.py
    (preProcess_batch / postProcess_batch / decode_cpu_batch / py_cpu_nms).
    """

    def __init__(self, config: dict | None = None):
        cfg = dict(CONFIG_RETINAFACE)
        if config:
            cfg.update(config)
        self.conf_thres = cfg["conf_thres"]
        self.iou_thres = cfg["iou_thres"]
        self.variance = cfg["variance"]
        self.image_size = cfg["image_size"]
        self.align_size = [112, 112]

        # ONNX session
        opts = _ort.SessionOptions()
        opts.log_severity_level = 3
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in _ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.sess = _ort.InferenceSession(
            cfg["model_path"], sess_options=opts, providers=providers
        )
        self._batch_supported: bool | None = (
            None
            if _batch_enabled(self.sess, "RETINAFACE_BATCH")
            else False
        )
        self._max_batch_size = _positive_int_env(
            "RETINAFACE_BATCH_SIZE", 8
        )

        # Prior boxes (from retinanet_det.PriorBox.forward)
        pb = PriorBox(
            cfg["min_sizes"], cfg["steps"], cfg["clip"], self.image_size
        )
        self.priors = pb.forward()

    # ── preProcess_batch (retinanet_det.py:264-272) ────────────────────
    @staticmethod
    def _preprocess_batch(images: list[np.ndarray]) -> np.ndarray:
        ims = np.array(images, dtype=np.float32)
        ims -= (104, 117, 123)                   # mean subtract
        ims = ims.transpose(0, 3, 1, 2)          # NWHC → NCHW
        return np.ascontiguousarray(ims)

    @staticmethod
    def _letterbox_resize(
        frame: np.ndarray, target_shape: tuple[int, int]
    ) -> tuple[np.ndarray, float]:
        """Aspect-ratio-preserving resize + zero-pad canvas.

        Mirror of ``uniface.common.resize_image``. RetinaFace was trained on
        square images; squashing 1920x1080 into 640x640 distorts faces enough
        to shift bbox decoding by tens of pixels and flip the gaze model's
        yaw sign for off-centre faces.

        Returns the padded canvas plus the scale factor (`new_size /
        original_size`) so the postprocessor can undo it.
        """
        width, height = target_shape
        im_ratio = float(frame.shape[0]) / frame.shape[1]
        model_ratio = float(height) / float(width)
        if im_ratio > model_ratio:
            new_height = height
            new_width = int(new_height / im_ratio)
        else:
            new_width = width
            new_height = int(new_width * im_ratio)
        resize_factor = float(new_height) / frame.shape[0]
        resized = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR,
        )
        canvas = np.zeros((height, width, 3), dtype=frame.dtype)
        canvas[:new_height, :new_width, :] = resized
        return canvas, resize_factor

    # ── decode_cpu_batch (retinanet_det.py:350-355) ────────────────────
    @staticmethod
    def _decode_boxes_batch(locs, priors, variance):
        boxes = np.concatenate((
            priors[:, :, :2] + locs[:, :, :2] * variance[0] * priors[:, :, 2:],
            priors[:, :, 2:] * np.exp(locs[:, :, 2:] * variance[1]),
        ), axis=2)
        boxes[:, :, :2] -= boxes[:, :, 2:] / 2
        boxes[:, :, 2:] += boxes[:, :, :2]
        return boxes

    # ── decode_landm_cpu_batch (retinanet_det.py:357-364) ─────────────
    @staticmethod
    def _decode_landmarks_batch(pre, priors, variance):
        landms = np.concatenate((
            priors[:, :, :2] + pre[:, :, 0:2]  * variance[0] * priors[:, :, 2:],
            priors[:, :, :2] + pre[:, :, 2:4]  * variance[0] * priors[:, :, 2:],
            priors[:, :, :2] + pre[:, :, 4:6]  * variance[0] * priors[:, :, 2:],
            priors[:, :, :2] + pre[:, :, 6:8]  * variance[0] * priors[:, :, 2:],
            priors[:, :, :2] + pre[:, :, 8:10] * variance[0] * priors[:, :, 2:],
        ), axis=2)
        return landms

    # ── py_cpu_nms (retinanet_det.py:157-179) ─────────────────────────
    @staticmethod
    def _nms(dets, thresh):
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    # ── postProcess_batch (retinanet_det.py:274-348) ───────────────────
    def _postprocess_batch(self, ims, resize_factors, locs, confs, landms):
        """Map decoded boxes back to original image coords.

        ``resize_factors`` is a per-image scalar — the same value returned by
        :meth:`_letterbox_resize`. To convert normalized model-space
        predictions back to pixels we first scale by the network input size
        (640 x 640), then divide by the resize factor to undo letterboxing.
        """
        prior_datas = np.tile(self.priors, (len(resize_factors), 1, 1))

        # Decode boxes + landmarks (still in normalized [0,1] space).
        boxes = self._decode_boxes_batch(locs, prior_datas, self.variance)
        # All images go through the same network input — scale to pixels of
        # the letterbox canvas (input_size), then undo the resize factor per
        # image so coordinates land on the original frame.
        net_w, net_h = float(self.image_size[0]), float(self.image_size[1])
        scale_boxes = np.array([[[net_w, net_h, net_w, net_h]]], dtype=np.float32)
        boxes = boxes * scale_boxes
        factors = np.asarray(resize_factors, dtype=np.float32).reshape(-1, 1, 1)
        boxes = boxes / factors

        scores = confs[:, :, 1]

        landms = self._decode_landmarks_batch(landms, prior_datas, self.variance)
        scale_landms = np.array(
            [[[net_w, net_h] * 5]], dtype=np.float32,
        )
        landms = landms * scale_landms / factors

        results = []
        miss_det = []
        croped_images = []

        for i, (im, b, s, l) in enumerate(zip(ims, boxes, scores, landms)):
            orig_w, orig_h = im.shape[1], im.shape[0]
            inds = np.where(s > self.conf_thres)[0]
            b, l, s = b[inds], l[inds], s[inds]
            order = s.argsort()[::-1][:5000]
            b, l, s = b[order], l[order], s[order]

            dets = np.hstack((b, s[:, np.newaxis])).astype(np.float32, copy=False)
            keep = self._nms(dets, self.iou_thres)
            dets, l = dets[keep, :], l[keep]
            dets, l = dets[:750, :], l[:750, :]
            dets = np.concatenate((dets, l), axis=1)

            if len(dets) != 0:
                # Pick the highest-confidence face. The previous "biggest face"
                # + 2%-of-frame area gate was tuned for the legacy single-face
                # crop pipeline and silently filtered out every face on a full
                # classroom frame. The remaining hard guard is a tiny pixel-size
                # check to drop sub-detector-resolution false positives.
                dets_max = max(dets.tolist(), key=lambda x: x[4])
                dets_max = np.array(dets_max)
                dets_max[:4:2] = np.clip(dets_max[:4:2], 0, orig_w)
                dets_max[1:4:2] = np.clip(dets_max[1:4:2], 0, orig_h)
                dets_max[5::2] = np.clip(dets_max[5::2], 0, orig_w)
                dets_max[6::2] = np.clip(dets_max[6::2], 0, orig_h)

                bbox = dets_max[:4]
                if (bbox[2] - bbox[0]) < 8 or (bbox[3] - bbox[1]) < 8:
                    miss_det.append(i)
                    continue

                result = dict(loc=bbox, conf=dets_max[4], landms=dets_max[5:])
                results.append(result)

                # Landmark alignment (face_preprocess from Face_recogition_v2)
                lm = dets_max[5:]
                landmarks = np.array([
                    lm[0], lm[2], lm[4], lm[6], lm[8],
                    lm[1], lm[3], lm[5], lm[7], lm[9],
                ], dtype=np.float32).reshape(2, 5).T
                aligned = _face_align(im, bbox.copy(), landmarks, image_size=self.align_size)
                aligned = cv2.resize(aligned, tuple(self.align_size), interpolation=cv2.INTER_AREA)
                croped_images.append(aligned)
            else:
                miss_det.append(i)

        return np.array(results), np.array(miss_det), np.array(croped_images)

    def _infer(self, images: list[np.ndarray]):
        # Aspect-ratio-preserving letterbox keeps faces undistorted. Without
        # it a 1920x1080 frame gets squashed into 640x640, mis-locating bbox
        # centres and flipping gaze yaw for faces away from the centre.
        resized: list[np.ndarray] = []
        resize_factors: list[float] = []
        for image in images:
            canvas, factor = self._letterbox_resize(image, tuple(self.image_size))
            resized.append(canvas)
            resize_factors.append(factor)
        inputs = self._preprocess_batch(resized)
        ort_inputs = {self.sess.get_inputs()[0].name: inputs}
        locs, confs, landms = self.sess.run(None, ort_inputs)
        return self._postprocess_batch(
            images, resize_factors, locs, confs, landms
        )

    def _detect_single(self, image: np.ndarray) -> list[dict]:
        results, _miss_det, cropped_images = self._infer([image])
        crops = list(cropped_images) if len(cropped_images) > 0 else []
        return [
            {
                "loc": result["loc"],
                "landm": result.get("landms", result.get("landm", [])),
                "conf": float(result["conf"]),
                "aligned_face": crops[i] if i < len(crops) else None,
            }
            for i, result in enumerate(results)
        ]

    def detect_batch(self, images: list[np.ndarray]) -> list[list[dict]]:
        """Detect one best face per image using the fastest provider strategy."""
        if not images:
            return []
        if len(images) == 1 or self._batch_supported is False:
            return [self._detect_single(image) for image in images]

        output: list[list[dict]] = []
        try:
            for start in range(0, len(images), self._max_batch_size):
                chunk = images[start:start + self._max_batch_size]
                results, missed, cropped_images = self._infer(chunk)
                missed_indexes = {
                    int(i) for i in np.asarray(missed).reshape(-1)
                }
                result_iter = iter(results)
                crop_iter = iter(cropped_images)
                for index in range(len(chunk)):
                    if index in missed_indexes:
                        output.append([])
                        continue
                    result = next(result_iter)
                    aligned = next(crop_iter, None)
                    output.append([{
                        "loc": result["loc"],
                        "landm": result.get(
                            "landms", result.get("landm", [])
                        ),
                        "conf": float(result["conf"]),
                        "aligned_face": aligned,
                    }])
            self._batch_supported = True
        except Exception:
            # Some exported ONNX files have a fixed batch dimension. Retain
            # full compatibility and remember the limitation for later frames.
            self._batch_supported = False
            return [self._detect_single(image) for image in images]
        return output

    # ── detect (inference_batch from retinanet_det.py:366-379) ────────
    def detect(self, image: np.ndarray) -> tuple[list[dict], np.ndarray]:
        """Detect the biggest face in one BGR image."""
        dets = self.detect_batch([image])[0]
        cropped_images = np.array(
            [det["aligned_face"] for det in dets if det.get("aligned_face") is not None]
        )
        return dets, cropped_images

    def detect_all(self, image: np.ndarray) -> list[dict]:
        """Detect EVERY face in one frame in a single forward pass.

        Existing :meth:`detect_batch` is tuned for the legacy "one face per
        person-crop" pipeline — postprocess picks the highest-confidence box
        per image. Running it N times (once per YOLO person crop) means N
        forward passes through a 640x640 network and dominates GPU time on
        any classroom-sized frame (~200 ms for 8 people on a T4). One pass
        on the full frame is ~10 ms and finds the same faces.

        Returns a list of dicts with the same shape as the per-crop detector
        (``loc``, ``landm``, ``conf``, ``aligned_face``) so callers can swap
        them in.
        """
        h, w = image.shape[:2]
        canvas, factor = self._letterbox_resize(image, tuple(self.image_size))
        inputs = self._preprocess_batch([canvas])
        locs, confs, landms = self.sess.run(
            None, {self.sess.get_inputs()[0].name: inputs},
        )

        priors = self.priors[None]
        boxes = self._decode_boxes_batch(locs, priors, self.variance)[0]
        scores = confs[0, :, 1]
        lms = self._decode_landmarks_batch(landms, priors, self.variance)[0]

        net_w, net_h = float(self.image_size[0]), float(self.image_size[1])
        boxes[:, [0, 2]] *= net_w
        boxes[:, [1, 3]] *= net_h
        lms[:, 0::2] *= net_w
        lms[:, 1::2] *= net_h
        boxes /= factor
        lms /= factor

        inds = np.where(scores > self.conf_thres)[0]
        if inds.size == 0:
            return []
        boxes, scores, lms = boxes[inds], scores[inds], lms[inds]
        order = scores.argsort()[::-1]
        boxes, scores, lms = boxes[order], scores[order], lms[order]

        dets = np.hstack((boxes, scores[:, None])).astype(np.float32, copy=False)
        keep = self._nms(dets, self.iou_thres)
        dets = dets[keep]
        lms = lms[keep]

        out: list[dict] = []
        for det, lm in zip(dets, lms):
            x1 = float(np.clip(det[0], 0, w))
            y1 = float(np.clip(det[1], 0, h))
            x2 = float(np.clip(det[2], 0, w))
            y2 = float(np.clip(det[3], 0, h))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
            lm = np.asarray(lm, dtype=np.float32)
            landmarks = np.array([
                lm[0], lm[2], lm[4], lm[6], lm[8],
                lm[1], lm[3], lm[5], lm[7], lm[9],
            ], dtype=np.float32).reshape(2, 5).T
            aligned = _face_align(
                image, bbox.copy(), landmarks, image_size=self.align_size,
            )
            aligned = cv2.resize(
                aligned, tuple(self.align_size), interpolation=cv2.INTER_AREA,
            )
            out.append({
                "loc": bbox,
                "landm": lm,
                "conf": float(det[4]),
                "aligned_face": aligned,
            })
        return out


# ═══════════════════════════════════════════════════════════════════════════
# ArcFaceExtractor — self-contained (copied from arcface_r100_onnx.py)
# ═══════════════════════════════════════════════════════════════════════════

class ArcFaceExtractor:
    """ArcFace R100 ONNX face embedding extractor.

    Preprocess / postprocess copied from Face_recogition_v2:
      src/service_ai/arcface_r100_onnx.py
    (get_feature_without_det / compare_face_1_n_1).

    Pipeline per face:
      resize 112×112 → BGR→RGB → HWC→CHW transpose → (img-127.5)*0.0078125
      → NCHW → flip augmentation → ONNX infer → sum original+flip → L2 normalize.
    """

    EMBEDDING_DIM = 512

    def __init__(self, config: dict | None = None):
        cfg = dict(CONFIG_ARCFACE)
        if config:
            cfg.update(config)

        # Flip-augmentation doubles every ONNX call: for N faces we send
        # 2N images and average. It improves matching by ~1% but costs
        # exactly 2x. Disable by default (env ARCFACE_FLIP=0). Gallery and
        # runtime use the same setting so embeddings stay comparable.
        self._use_flip = os.getenv("ARCFACE_FLIP", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }

        # ONNX session (same as arcface_r100_onnx.ArcfaceRunnable.__init__)
        opts = _ort.SessionOptions()
        opts.log_severity_level = 3
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in _ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.sess = _ort.InferenceSession(
            cfg["model_path"], sess_options=opts, providers=providers
        )
        self._batch_supported: bool | None = (
            None if _batch_enabled(self.sess, "ARCFACE_BATCH") else False
        )
        self._max_batch_size = _positive_int_env("ARCFACE_BATCH_SIZE", 32)

    # ── get_feature (arcface_r100_onnx.py:29-78) ──────────────────────
    def get_feature(self, ims: list[np.ndarray],
                    dets: list[dict]) -> np.ndarray | None:
        """Landmark-aligned face embedding from detection results.

        For each detection: finds biggest face box → landmark alignment via
        face_preprocess → resize 112×112 → BGR→RGB → CHW transpose → ONNX → L2 norm.
        """
        if len(dets[0].get("loc", [])) == 0:
            return None

        outputs = []
        for i, det in enumerate(dets):
            im = ims[i]
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            landms = det.get("landms", det.get("landm", []))
            bboxes = det.get("loc", [])

            biggest_box = None
            max_area = 0
            for j, bboxe in enumerate(bboxes):
                x1, y1, x2, y2 = bboxe
                area = (x2 - x1) * (y2 - y1)
                if area > max_area and area > 400:
                    max_area = area
                    biggest_box = bboxe
                    landmarks = landms[j]

            if biggest_box is not None:
                bbox = np.array(biggest_box)
                lm = np.array([
                    landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
                    landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9],
                ]).reshape(2, 5).T

                nimg = _face_align(im, bbox, lm, image_size=[112, 112])
                nimg = cv2.resize(nimg, (112, 112), interpolation=cv2.INTER_AREA)
                nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                nimg = np.transpose(nimg, (2, 0, 1))  # HWC → CHW

                input_blob = np.expand_dims(nimg, axis=0).astype(np.float32)
                input_name = self.sess.get_inputs()[0].name
                embedding = self.sess.run(None, {input_name: input_blob})
                embedding = _l2_normalize(embedding[0]).flatten()

                fp = embedding.reshape(1, -1)
                outputs = fp if len(outputs) == 0 else np.concatenate((outputs, fp), axis=0)
        if len(outputs) == 0:
            return None
        return np.array(outputs) if isinstance(outputs, list) else outputs

    # ── get_feature_without_det (arcface_r100_onnx.py:80-108) ─────────
    def _get_feature_without_det(self, ims: list[np.ndarray]) -> np.ndarray | None:
        """Embedding from pre-cropped face images (no landmark detection needed).

        Preprocess:
          resize 112×112 → BGR→RGB → HWC→CHW → (img-127.5)*0.0078125
          → NCHW → [flip augmentation if ARCFACE_FLIP=1] → concat
        Postprocess:
          ONNX infer → reshape → [sum flip pair if flip enabled] → L2 normalize.

        Disabling flip halves the ONNX workload (1 input/face instead of 2)
        and is bit-compatible with the gallery as long as ARCFACE_FLIP is the
        same value when registration and recognition run.
        """
        # Build a single (N or 2N, 3, 112, 112) tensor for ONE batched call.
        per_face = 2 if self._use_flip else 1
        prepared = np.empty(
            (len(ims) * per_face, 3, 112, 112), dtype=np.float32,
        )
        for i, im in enumerate(ims):
            im = cv2.resize(im, (112, 112), interpolation=cv2.INTER_AREA)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            chw = np.transpose(im, (2, 0, 1)).astype(np.float32)
            chw = (chw - 127.5) * 0.0078125
            if self._use_flip:
                prepared[i * 2] = chw
                # Mirror across width axis of the CHW tensor.
                prepared[i * 2 + 1] = chw[:, :, ::-1]
            else:
                prepared[i] = chw
        if prepared.size == 0:
            return None

        input_name = self.sess.get_inputs()[0].name
        chunks = []
        if self._batch_supported is False:
            for inp in prepared:
                raw = self.sess.run(
                    None, {input_name: inp[None].astype(np.float32)},
                )
                chunks.append(np.asarray(raw[0]))
            embedding = np.concatenate(chunks, axis=0)
        else:
            try:
                chunk_size = self._max_batch_size
                for start in range(0, len(prepared), chunk_size):
                    inp = prepared[start:start + chunk_size]
                    raw = self.sess.run(None, {input_name: inp})
                    chunks.append(np.asarray(raw[0]))
                embedding = np.concatenate(chunks, axis=0)
                self._batch_supported = True
            except Exception:
                # Fall back to one-at-a-time if the runtime rejects batching for
                # some reason. Slow but always correct.
                self._batch_supported = False
                chunks = []
                for inp in prepared:
                    raw = self.sess.run(
                        None, {input_name: inp[None].astype(np.float32)},
                    )
                    chunks.append(np.asarray(raw[0]))
                embedding = np.concatenate(chunks, axis=0)

        if self._use_flip:
            # (2N, 512) → (N, 2, 512) → sum pair → (N, 512)
            embedding = embedding.reshape(len(ims), 2, -1).sum(axis=1)
        return _l2_normalize(embedding, axis=1).astype(np.float32, copy=False)

    # ── Public API (same signature as before) ──────────────────────────
    def extract(self, aligned_faces: list[np.ndarray]) -> np.ndarray | None:
        """Extract 512-dim embeddings (N, 512) or None."""
        if not aligned_faces:
            return None
        valid = [f for f in aligned_faces if f is not None and f.size > 0]
        if not valid:
            return None
        emb = self._get_feature_without_det(valid)
        return np.array(emb, dtype=np.float32) if emb is not None else None

    def extract_single(self, face: np.ndarray) -> np.ndarray | None:
        """Extract single 512-dim embedding."""
        emb = self._get_feature_without_det([face])
        return emb[0] if emb is not None and len(emb) > 0 else None

    # ── compare_face_1_n_1 (arcface_r100_onnx.py:109-117) ─────────────
    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Tanh-calibrated similarity (arcface_r100_onnx.compare_face_1_n_1)."""
        dist = np.linalg.norm(a - b)
        return float((np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2)


# ── Helper: build face gallery from student records ─────────────────────

# ── Redis feature cache helpers ──────────────────────────────────────────

_USER_KEY = "qhh:user:{student_id}"  # Embedding sống cùng hồ sơ user.


def _decode_hash(raw: dict) -> dict:
    """Hash từ Redis có thể là bytes hoặc str (tuỳ decode_responses). Chuẩn hoá str."""
    out = {}
    for k, v in (raw or {}).items():
        k = k.decode() if isinstance(k, (bytes, bytearray)) else k
        v = v.decode() if isinstance(v, (bytes, bytearray)) else v
        out[k] = v
    return out


def _redis_feature_get(r, sid: str, mtime_ns: int | None = None,
                        size: int | None = None, flip: bool | None = None):
    """Đọc embedding gắn trong `qhh:user:{sid}`. Trả (embedding, record) hoặc (None, ...).

    Stale (mtime/size/flip không khớp) → trả (None, rec) để caller quyết định.
    """
    try:
        import base64 as _b64
        raw = r.hgetall(_USER_KEY.format(student_id=sid))
        rec = _decode_hash(raw)
        if not rec or not rec.get("embedding"):
            return None, rec or None

        # Cast các metadata số về int/bool để so sánh.
        try:
            rec_mtime = int(rec.get("embeddingMtimeNs", 0) or 0) or None
            rec_size = int(rec.get("embeddingSize", 0) or 0) or None
        except (TypeError, ValueError):
            rec_mtime = rec_size = None
        rec_flip = str(rec.get("embeddingFlip", "")).lower() == "true"
        rec["avatarMtimeNs"] = rec_mtime
        rec["avatarSize"] = rec_size
        rec["flipUsed"] = rec_flip

        if mtime_ns is not None and (
            rec_mtime != mtime_ns or rec_size != size or rec_flip != flip
        ):
            return None, rec

        emb = np.frombuffer(_b64.b64decode(rec["embedding"]), dtype=np.float32).copy()
        return emb, rec
    except Exception:
        pass
    return None, None


def _redis_feature_set(r, sid: str, student: dict, mtime_ns: int, size: int,
                        flip: bool, embedding: np.ndarray) -> bool:
    """Ghi embedding lên hash `qhh:user:{sid}` (không expire).

    Các field hồ sơ user khác (username, userType, fullName...) do qhh-server
    ghi sẵn — chỉ HSET những field mình quản lý để khỏi đè lên.
    """
    try:
        import base64 as _b64
        mapping = {
            "id": sid,
            "embedding": _b64.b64encode(embedding.astype(np.float32).tobytes()).decode(),
            "embeddingDim": str(embedding.shape[0]),
            "embeddingMtimeNs": str(mtime_ns),
            "embeddingSize": str(size),
            "embeddingFlip": "true" if flip else "false",
        }
        avatar_url = student.get("avatarUrl") or student.get("face_image") or ""
        if avatar_url:
            mapping["avatar"] = avatar_url
        full_name = student.get("name") or student.get("fullName") or ""
        if full_name:
            mapping.setdefault("fullName", full_name)
        r.hset(_USER_KEY.format(student_id=sid), mapping=mapping)
        return True
    except Exception as exc:
        print(f"[face-gallery] Redis write failed for {sid}: {exc}", flush=True)
        return False


def _get_redis():
    """Return a Redis client reusing db.redis_client settings, or None if unavailable."""
    try:
        import sys as _sys
        _root = str(Path(__file__).resolve().parents[1])
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from db.redis_client import get_client
        return get_client()
    except Exception:
        return None


def build_face_gallery(
    students: list[dict],
    arcface: ArcFaceExtractor,
    retinaface: RetinaFaceDetector | None = None,
) -> list[dict]:
    """Build ArcFace gallery for a class.  Cache priority (fastest first):

      1. In-process LRU  — zero I/O, per-process lifetime
      2. Redis  qhh:face:feature:{student_id}  — survives worker restarts;
         contains student metadata (code, name, avatarUrl) + embedding.
         Redis is checked by student_id FIRST (no file access needed).
         If the stored mtime/size differ from the actual file the entry is
         stale → fall through to re-extract.
      3. RetinaFace + ArcFace on the avatar image → write result to both
         Redis and the in-process LRU.

    This means a worker restart loads the entire gallery from Redis without
    touching a single avatar file, and the vectorised numpy cosine search in
    _detect() runs against these pre-computed embeddings every frame.
    """
    r = _get_redis()
    flip = bool(getattr(arcface, "_use_flip", False))
    gallery = []
    pending: list[tuple[dict, tuple[str, int, int, bool], np.ndarray]] = []

    for student in students:
        sid = str(student.get("id", "") or "")
        if not sid:
            continue
        name = student.get("name") or student.get("fullName") or ""
        code = student.get("student_code") or student.get("studentCode") or ""

        # ── 1. Try Redis first (no file I/O required) ──────────────────
        if r is not None:
            emb, rec = _redis_feature_get(r, sid)          # unconditional load
            if emb is not None:
                # Validate freshness against avatar file if it exists
                path = str(student.get("face_image", "") or "")
                source = Path(path) if path else None
                stale = False
                if source is not None and source.exists():
                    stat = source.stat()
                    mtime_ns = int(stat.st_mtime_ns)
                    size = int(stat.st_size)
                    if (
                        rec.get("avatarMtimeNs") != mtime_ns
                        or rec.get("avatarSize") != size
                        or rec.get("flipUsed") != flip
                    ):
                        stale = True

                if not stale:
                    # Populate name/code from Redis record if student dict is sparse
                    name = name or rec.get("fullName", "")
                    code = code or rec.get("studentCode", "")
                    cache_key = (rec.get("avatarUrl", sid), rec.get("avatarMtimeNs", 0),
                                 rec.get("avatarSize", 0), flip)
                    with _GALLERY_CACHE_LOCK:
                        _GALLERY_CACHE[cache_key] = emb
                        _GALLERY_CACHE.move_to_end(cache_key)
                        while len(_GALLERY_CACHE) > _GALLERY_CACHE_MAX:
                            _GALLERY_CACHE.popitem(last=False)
                    gallery.append({
                        "student_id": sid,
                        "name": name,
                        "student_code": code,
                        "embedding": emb,
                    })
                    print(f"[face-gallery] Redis hit: {name or sid}", flush=True)
                    continue
                # stale → fall through to re-extract below

        # ── 2. In-process LRU (file must exist for cache_key) ──────────
        path = str(student.get("face_image", "") or "")
        source = Path(path)
        if not path or not source.exists():
            continue
        stat = source.stat()
        mtime_ns = int(stat.st_mtime_ns)
        size = int(stat.st_size)
        cache_key = (str(source.resolve()), mtime_ns, size, flip)

        with _GALLERY_CACHE_LOCK:
            cached = _GALLERY_CACHE.get(cache_key)
            if cached is not None:
                _GALLERY_CACHE.move_to_end(cache_key)
        if cached is not None:
            gallery.append({
                "student_id": sid,
                "name": name,
                "student_code": code,
                "embedding": cached,
            })
            continue

        # ── 3. Extract via RetinaFace + ArcFace ────────────────────────
        image = cv2.imread(path)
        if image is None:
            continue
        pending.append((student, cache_key, image))

    if pending:
        source_images = [item[2] for item in pending]
        face_images = list(source_images)
        if retinaface is not None:
            detections = retinaface.detect_batch(source_images)
            for i, dets in enumerate(detections):
                if dets and dets[0].get("aligned_face") is not None:
                    face_images[i] = dets[0]["aligned_face"]
        embeddings = arcface.extract(face_images)
        if embeddings is not None:
            for (student, cache_key, _), embedding in zip(pending, embeddings):
                embedding = np.asarray(embedding, dtype=np.float32)
                sid = str(student.get("id", "") or "")
                _, mtime_ns, size, flip_ = cache_key
                name = student.get("name") or student.get("fullName") or ""
                code = student.get("student_code") or student.get("studentCode") or ""
                with _GALLERY_CACHE_LOCK:
                    _GALLERY_CACHE[cache_key] = embedding
                    _GALLERY_CACHE.move_to_end(cache_key)
                    while len(_GALLERY_CACHE) > _GALLERY_CACHE_MAX:
                        _GALLERY_CACHE.popitem(last=False)
                if r is not None:
                    if _redis_feature_set(r, sid, student, mtime_ns, size, flip_, embedding):
                        print(f"[face-gallery] Extracted → saved to Redis: {name or sid}", flush=True)
                    else:
                        print(f"[face-gallery] Extracted → cached in process: {name or sid}", flush=True)
                gallery.append({
                    "student_id": sid,
                    "name": name,
                    "student_code": code,
                    "embedding": embedding,
                })
    return gallery
