"""Gaze estimation wrapper around the MobileGaze ResNet-50 ONNX model.

Preprocess and decode mirror the reference implementation in
``gaze-estimation/onnx_inference.py:GazeEstimationONNX`` byte-for-byte:

* preprocess: BGR→RGB → cv2.resize to ``input_size`` (warp, no aspect-ratio
  preservation) → /255 → ImageNet mean/std → HWC→CHW → batch dim.
* decode: softmax on each head → expectation across 90 bins of 4° offset by
  180° → np.radians (the reference function returns radians, so we do too).
* estimate / estimate_batch take a face crop and return ``(yaw, pitch)`` in
  radians. Callers that want degrees apply ``np.degrees`` themselves.

This module deliberately keeps only what's in the reference. No padding, no
letterboxing, no smoothing, no degree conversion — those belong outside.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as _ort

_WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
GAZE_MODEL_PATH = str(_WEIGHTS / "resnet50_gaze.onnx")


class GazeEstimator:
    """ONNX gaze estimator. Faithful port of GazeEstimationONNX.

    Returns ``(yaw, pitch)`` in **radians** — same units the reference returns.
    """

    def __init__(self, model_path: str | None = None,
                 session: _ort.InferenceSession | None = None):
        # ── Identical to GazeEstimationONNX.__init__ ─────────────────────
        self.session = session
        if self.session is None:
            path = model_path or GAZE_MODEL_PATH
            if not os.path.exists(path):
                raise FileNotFoundError(f"Gaze model not found: {path}")
            opts = _ort.SessionOptions()
            opts.log_severity_level = 3
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in _ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )
            self.session = _ort.InferenceSession(
                path, sess_options=opts, providers=providers
            )

        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)

        self.input_shape = (448, 448)
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        self.input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:][::-1])

        outputs = self.session.get_outputs()
        output_names = [output.name for output in outputs]
        self.output_names = output_names
        assert len(output_names) == 2, "Expected 2 output nodes, got {}".format(len(output_names))

        # The shipped resnet50_gaze.onnx has fixed batch=1 — when callers
        # send N>1 crops via estimate_batch we chunk them. The reference
        # `estimate()` only handles N=1 so this never matters for it.
        batch_dim = input_shape[0] if input_shape else 1
        self._max_batch = (
            int(batch_dim) if isinstance(batch_dim, int) and batch_dim > 0 else 0
        )

    def _preprocess_one(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size)
        image = image.astype(np.float32) / 255.0
        mean = np.array(self.input_mean, dtype=np.float32)
        std = np.array(self.input_std, dtype=np.float32)
        image = (image - mean) / std
        return np.transpose(image, (2, 0, 1)).astype(np.float32)  # CHW

    # Back-compat: single-image preprocess that adds the batch dim.
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return np.expand_dims(self._preprocess_one(image), axis=0)

    # ── EXACT copy of GazeEstimationONNX.softmax ─────────────────────────
    def softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def _decode_batch(
        self, yaw_logits: np.ndarray, pitch_logits: np.ndarray
    ) -> list[tuple[float, float]]:
        yaw_probs = self.softmax(yaw_logits)
        pitch_probs = self.softmax(pitch_logits)
        yaw = np.sum(yaw_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        pitch = np.sum(pitch_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        return [(float(yaw_rad[i]), float(pitch_rad[i])) for i in range(yaw_rad.shape[0])]

    # Back-compat single-sample decode that returns a tuple of scalars.
    def decode(self, yaw_logits: np.ndarray, pitch_logits: np.ndarray) -> tuple[float, float]:
        return self._decode_batch(yaw_logits, pitch_logits)[0]

    def estimate(self, face_image: np.ndarray) -> tuple[float, float]:
        """Estimate gaze for one face crop. Returns (yaw, pitch) in radians."""
        return self.estimate_batch([face_image])[0]

    def estimate_batch(self, crops: list[np.ndarray]) -> list[tuple[float, float]]:
        """Run gaze on N face crops in a single ``session.run``.

        Returns ``(yaw, pitch)`` in **radians** for each crop, in input order.
        Empty/None crops yield ``(0.0, 0.0)`` and are skipped from the ONNX
        call to avoid wasting GPU on degenerate inputs.
        """
        if not crops:
            return []

        valid_indices: list[int] = []
        tensors: list[np.ndarray] = []
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            valid_indices.append(i)
            tensors.append(self._preprocess_one(crop))

        out: list[tuple[float, float]] = [(0.0, 0.0)] * len(crops)
        if not tensors:
            return out

        batch = np.stack(tensors, axis=0)  # [N, 3, 448, 448]
        yaw_logits, pitch_logits = self.session.run(
            self.output_names, {self.input_name: batch}
        )
        decoded = self._decode_batch(yaw_logits, pitch_logits)
        for slot, value in zip(valid_indices, decoded):
            out[slot] = value
        return out

    # Backwards-compatible aliases kept for callers in camera_worker/web_server.
    # The reference uses lowercase attribute names; map our previous capitals.
    @property
    def sess(self):
        return self.session

    @property
    def _input_name(self):
        return self.input_name

    @property
    def _output_names(self):
        return self.output_names

    @property
    def INPUT_SIZE(self):
        return self.input_size


class DistractionTracker:
    """Latches a per-student alert when yaw exceeds a threshold for too long.

    Use one tracker per worker; call :meth:`update` once per detection.

    Raw gaze readings jitter ±10° because RetinaFace's face_box wobbles by a
    few pixels each frame and the binned-expectation decoder is sensitive to
    that. We exponentially smooth per-student before threshold comparison.
    """

    def __init__(
        self,
        yaw_threshold_deg: float = 60.0,
        pitch_threshold_deg: float = 9999.0,   # pitch disabled by default
        alert_after_sec: float = 2.5,
        clear_after_sec: float = 1.0,
        ema_alpha: float = 0.35,
    ):
        self.yaw_th = float(yaw_threshold_deg)
        self.pitch_th = float(pitch_threshold_deg)
        self.alert_after = float(alert_after_sec)
        self.clear_after = float(clear_after_sec)
        # ema_alpha=1.0 disables smoothing; 0.2 is heavy smoothing.
        self.ema_alpha = float(max(0.05, min(1.0, ema_alpha)))
        # student_id → {distracted_since, focused_since, alert, yaw_ema, pitch_ema}
        self._state: dict[str, dict] = {}

    @staticmethod
    def _is_distracted(yaw_deg: float, pitch_deg: float,
                       yaw_th: float, pitch_th: float) -> bool:
        return abs(yaw_deg) >= yaw_th or abs(pitch_deg) >= pitch_th

    def update(self, key: str, yaw_deg: float, pitch_deg: float, now: float) -> dict:
        """Return dict with focused/alert flags. ``key`` is e.g. student_id or bbox-id."""
        st = self._state.get(key)
        if st is None:
            st = {
                "distracted_since": None,
                "focused_since": None,
                "alert": False,
                "yaw_ema": float(yaw_deg),
                "pitch_ema": float(pitch_deg),
            }
            self._state[key] = st
        else:
            a = self.ema_alpha
            st["yaw_ema"] = a * float(yaw_deg) + (1 - a) * st["yaw_ema"]
            st["pitch_ema"] = a * float(pitch_deg) + (1 - a) * st["pitch_ema"]

        # Threshold against the RAW signal — EMA was masking real distraction
        # turns by pulling the smoothed value back toward the historical mean.
        # If the student is biased toward looking +yaw, a sudden -50° swing
        # never reaches the EMA's threshold before they look back. The
        # alert_after / clear_after windows still debounce single-frame jitter.
        # EMA is kept ONLY for the displayed/smoothed values, not for gating.
        yaw_smooth = st["yaw_ema"]
        pitch_smooth = st["pitch_ema"]
        distracted = self._is_distracted(
            float(yaw_deg), float(pitch_deg), self.yaw_th, self.pitch_th
        )

        if distracted:
            if st["distracted_since"] is None:
                st["distracted_since"] = now
            st["focused_since"] = None
            if (now - st["distracted_since"]) >= self.alert_after:
                st["alert"] = True
        else:
            if st["focused_since"] is None:
                st["focused_since"] = now
            st["distracted_since"] = None
            if (now - st["focused_since"]) >= self.clear_after:
                st["alert"] = False

        return {
            "gaze_focused": not distracted,
            "gaze_alert": st["alert"],
            "gaze_distracted_for": (
                now - st["distracted_since"] if st["distracted_since"] else 0.0
            ),
            "gaze_yaw_smooth": round(yaw_smooth, 1),
            "gaze_pitch_smooth": round(pitch_smooth, 1),
        }

    def forget(self, key: str):
        self._state.pop(key, None)

    def prune(self, keep_keys: set[str]):
        for k in list(self._state.keys()):
            if k not in keep_keys:
                self._state.pop(k, None)
