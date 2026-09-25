"""Stage 3 (deployed): the TRAINED quantum-kernel SVM, in the shape AIJudge expects.

Loads models/qsvm_model.joblib (any format LiveTheftClassifier reads, including the
cross-dataset motion ensembles from src/train_xd.py) and returns the same dict shape
as the prototype QuantumClassifier plus `trained: True`.

Two things make the live decision match how the model was measured:

* **Motion input.** `observe()` is fed every frame and keeps ~3 s of 64x64 frames, so
  a motion model gets the frame `motion_gap_sec` (0.33 s) earlier, exactly as in training.
* **Clip-level judgement.** Every accuracy number is clip-level: the MEAN frame margin
  over a clip vs the artifact's clip threshold. So `score()` is called several times a
  second while there is motion, and `classify()` judges the mean margin of the last
  `judge_window_sec`, not one frame.

`crime_signal` = sigmoid(2 * (window_mean - clip_threshold) + ln(0.625/0.375)): at the
model's own threshold it is 0.625, which is exactly what makes the default judge
(w_crime 0.8, sensitivity 0.5 -> flag at 0.5) flag at the validated operating point.
"""
from __future__ import annotations

import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from config import Settings

ROOT = Path(__file__).resolve().parent
# The artifact pickles PCAAngleProjector by module name, so src/ must be importable.
sys.path.insert(0, str(ROOT / "src"))

_CALIBRATION_BIAS = math.log(0.625 / 0.375)
_MARGIN_GAIN = 2.0
_HISTORY_SEC = 3.0


class TheftDetector:
    """Wraps LiveTheftClassifier; `classify(frame_bgr)` mirrors QuantumClassifier."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        from inference import LiveTheftClassifier   # imported late: heavy, and optional

        path = self.model_path_for(settings)
        if not path.is_file():
            raise FileNotFoundError(f"no trained artifact at {path}")
        self.path, self.mtime = path, path.stat().st_mtime
        self._clf = LiveTheftClassifier(path)
        meta = self._clf.metadata
        self._thresholds = {k: float(v) for k, v in meta.get("clip_thresholds", {}).items()}
        self._default_threshold = float(meta.get("clip_threshold", 0.0))
        self.motion = self._clf.in_channels == 6
        self._history: deque[tuple[float, np.ndarray]] = deque()   # (t, 64x64 BGR)
        self._window: deque[tuple[float, float]] = deque()         # (t, margin)
        acc = meta.get("clip_test_accuracy", float("nan"))
        self.description = (f"{meta.get('ensemble_size', 1)}-member {meta.get('experiment', 'qsvm')} "
                            f"(clip {acc:.3f}{', motion' if self.motion else ''})")

    @staticmethod
    def model_path_for(settings: Settings) -> Path:
        """The expert for this site: the retail model when configured and present, else the default."""
        retail = ROOT / settings.retail_model_path
        if settings.site_profile == "retail" and retail.is_file():
            return retail
        return ROOT / settings.model_path

    @property
    def threshold(self) -> float:
        """Clip threshold for the configured site profile (falls back to the model default)."""
        return self._thresholds.get(self._settings.site_profile, self._default_threshold)

    def observe(self, frame_bgr: np.ndarray, now: float | None = None) -> None:
        """Call on every frame: keeps the short history motion models need (cheap: a 64x64 resize)."""
        if not self.motion:
            return
        now = time.monotonic() if now is None else now
        self._history.append((now, cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA)))
        while self._history and now - self._history[0][0] > _HISTORY_SEC:
            self._history.popleft()

    def _previous(self, now: float) -> np.ndarray | None:
        """The frame closest to `motion_gap_sec` ago (None until enough history exists)."""
        target = now - self._clf.motion_gap_sec
        best = None
        for t, small in self._history:
            if t <= target:
                best = small
            else:
                break
        return best

    def score(self, frame_bgr: np.ndarray, now: float | None = None) -> dict:
        """Score one frame and add its margin to the judgement window."""
        now = time.monotonic() if now is None else now
        r = self._clf.predict_frame(frame_bgr, self._previous(now) if self.motion else None)
        self._window.append((now, float(r["margin"])))
        return r

    def _prune(self, now: float) -> None:
        while self._window and now - self._window[0][0] > self._settings.judge_window_sec:
            self._window.popleft()

    def window_frames(self, now: float) -> int:
        """Scored frames inside the judgement window (the node waits for enough of them)."""
        self._prune(now)
        return len(self._window)

    def classify(self, frame_bgr: np.ndarray, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        r = self.score(frame_bgr, now)
        self._prune(now)
        window_sec = self._settings.judge_window_sec
        margins = [m for _, m in self._window]
        mean = float(np.mean(margins))
        crime = 1.0 / (1.0 + math.exp(-(_MARGIN_GAIN * (mean - self.threshold) + _CALIBRATION_BIAS)))
        return {
            "trained": True,                      # AIJudge weights this signal higher
            "object_class": "theft" if mean > self.threshold else "normal",
            "class_scores": {"suspicious_signature": crime, "background": 1.0 - crime},
            "margin": float(r["margin"]),         # this frame
            "window": {"frames": len(margins), "seconds": window_sec, "mean_margin": mean,
                       "clip_threshold": self.threshold, "site_profile": self._settings.site_profile},
            "features": list(r["features_4d"]),   # the 4 ZZFeatureMap angles
            "member_margins": r["member_margins"],
            "latent": r["latent"],
            "latency_ms": r["latency_ms"],
            "backend": "trained_qsvm",
        }


if __name__ == "__main__":
    det = TheftDetector(Settings())
    print("loaded:", det.description)
    rng = np.random.default_rng(0)
    t = 100.0
    for _ in range(11):  # 2.75 s of frames scored at 4 Hz, then one judgement
        f = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        det.observe(f, t)
        det.score(f, t)
        t += 0.25
    det.observe(f, t)
    out = det.classify(f, t)
    assert out["object_class"] in ("theft", "normal")
    assert 0.0 <= out["class_scores"]["suspicious_signature"] <= 1.0
    assert len(out["features"]) == 4 and out["window"]["frames"] == 9  # 2 s window at 4 Hz, inclusive
    assert all(-np.pi - 1e-6 <= v <= np.pi + 1e-6 for v in out["features"])
    print("ok:", out["object_class"], round(out["class_scores"]["suspicious_signature"], 3), out["window"])
