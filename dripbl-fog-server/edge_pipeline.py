"""Motion gate -> CNN latent -> PCA angles, plus lightweight gait metrics
(speed, stride frequency, jitter) used downstream as nervousness/crime signals."""
from __future__ import annotations
import os
import time
from collections import deque

import numpy as np
import cv2
import torch
from torch import nn

from config import Settings

torch.set_num_threads(1)  # single-thread CPU: predictable, low edge overhead


class LightweightCNN(nn.Module):
    """(Conv-BN-ReLU-Pool) x3 -> AdaptiveAvgPool(2) -> Linear(128).

    `in_channels` is 3 for a plain RGB frame, or 6 for RGB + inter-frame
    difference (motion), which is what the temporal variant feeds it.

    Still trivial for a Pi, but unlike a 1x1 global-average
    bottleneck it keeps coarse spatial layout, which is what makes the 128-d
    latent (and the 4 PCA angles derived from it) actually discriminative.
    """

    def __init__(self, latent_dim: int = 128, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(2),
        )
        self.fc = nn.Linear(128 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.fc(x)


class MotionDetector:
    """MOG2 background subtraction restricted to the configured ROI. Returns
    the largest contour's centroid too, so callers can track gait over time."""

    def __init__(self, settings: Settings) -> None:
        self._roi = settings.roi
        self._settings = settings
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=settings.mog2_history,
            varThreshold=settings.mog2_var_threshold,
            detectShadows=False,
        )

    def crop_roi(self, frame: np.ndarray) -> np.ndarray:
        """ROI clamped to the frame; the whole frame if the ROI misses it (e.g. a camera
        smaller than the configured ROI), so a bad ROI degrades instead of crashing."""
        x, y, w, h = self._roi
        crop = frame[y:y + h, x:x + w]
        return crop if crop.shape[0] >= 8 and crop.shape[1] >= 8 else frame

    def detect(self, roi_frame: np.ndarray) -> tuple[bool, tuple[float, float] | None]:
        mask = self._bg.apply(roi_frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = max(contours, key=cv2.contourArea, default=None)
        if best is None or cv2.contourArea(best) < self._settings.area_threshold:
            return False, None
        m = cv2.moments(best)
        centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else None
        return True, centroid


class GaitAnalyzer:
    """Tracks a rolling window of (time, x, y) centroids to derive coarse gait
    proxies: walking speed, stride oscillation frequency, and direction jitter
    (erratic movement, used downstream as a nervousness signal)."""

    def __init__(self, window: int = 20) -> None:
        self._track: deque[tuple[float, float, float]] = deque(maxlen=window)

    def update(self, centroid: tuple[float, float] | None) -> dict[str, float]:
        if centroid is None:
            self._track.clear()
            return {"speed_px_s": 0.0, "stride_freq_hz": 0.0, "jitter_score": 0.0}
        self._track.append((time.monotonic(), *centroid))
        if len(self._track) < 3:
            return {"speed_px_s": 0.0, "stride_freq_hz": 0.0, "jitter_score": 0.0}

        t = np.array([p[0] for p in self._track])
        x = np.array([p[1] for p in self._track])
        y = np.array([p[2] for p in self._track])
        dt = np.diff(t)
        dt[dt <= 0] = 1e-3
        dx, dy = np.diff(x), np.diff(y)
        dist = np.hypot(dx, dy)

        speed = float(np.mean(dist / dt))

        # Stride proxy: lateral sway oscillation frequency via zero-crossings of x.
        xc = x - x.mean()
        crossings = int(np.sum(np.diff(np.sign(xc)) != 0))
        duration = max(t[-1] - t[0], 1e-3)
        stride_freq = crossings / (2.0 * duration)

        # Jitter proxy: variance of frame-to-frame heading angle (erratic motion).
        headings = np.arctan2(dy, dx)
        jitter = float(np.std(np.diff(headings))) if len(headings) > 1 else 0.0
        jitter_score = float(min(jitter / np.pi, 1.0))

        return {"speed_px_s": speed, "stride_freq_hz": stride_freq, "jitter_score": jitter_score}


class PCAProjector:
    """Vectorized (1x128) @ (128x4) projection, scaled to [-pi, pi]."""

    def __init__(self, settings: Settings) -> None:
        path = settings.pca_components_path
        if os.path.isfile(path):
            components = np.load(path).astype(np.float32)
        else:
            # ponytail: no pretrained PCA available, seeded orthonormal stand-in;
            # swap in real fitted components via config.pca_components_path.
            rng = np.random.default_rng(42)
            components, _ = np.linalg.qr(rng.standard_normal((settings.latent_dim, settings.pca_dim)))
            components = components.astype(np.float32)
        assert components.shape == (settings.latent_dim, settings.pca_dim)
        self._components = components

    def project(self, latent: np.ndarray) -> np.ndarray:
        proj = latent @ self._components               # (1, pca_dim)
        span = np.abs(proj).max() + 1e-8
        return (proj / span * np.pi).astype(np.float32) # scale to [-pi, pi]


class EdgePipeline:
    """Stage 1 of the pipeline: motion gate -> CNN latent -> PCA angles + gait."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._motion = MotionDetector(settings)
        self._gait = GaitAnalyzer()
        self._pca = PCAProjector(settings)
        self._cnn = LightweightCNN(settings.latent_dim).eval()
        if os.path.isfile(settings.cnn_weights_path):
            self._cnn.load_state_dict(torch.load(settings.cnn_weights_path, map_location="cpu"))
        self._size = settings.cnn_input_size
        self._buf = np.empty((self._size, self._size, 3), dtype=np.float32)  # reused each call

    def observe(self, frame: np.ndarray) -> tuple[bool, dict[str, float]]:
        """Runs motion detection + gait tracking; call every frame regardless
        of trigger state so the gait window stays continuous."""
        triggered, centroid = self._motion.detect(self._motion.crop_roi(frame))
        gait = self._gait.update(centroid)
        return triggered, gait

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        roi = self._motion.crop_roi(frame)
        resized = cv2.resize(roi, (self._size, self._size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        np.divide(rgb, 255.0, out=self._buf, casting="unsafe")  # normalize into reused buffer
        tensor = torch.from_numpy(self._buf).permute(2, 0, 1).unsqueeze(0)
        with torch.inference_mode():
            latent = self._cnn(tensor).numpy()
        return self._pca.project(latent)[0]  # (pca_dim,)
