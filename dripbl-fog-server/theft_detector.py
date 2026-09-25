"""Stage 3 (deployed): the TRAINED quantum-kernel SVM, in the shape AIJudge expects.

Without this the node never touches the offline training work: `main.py` used
`QuantumClassifier`, which scores fidelity against hand-seeded prototypes with a
stand-in CNN, while `models/qsvm_model.joblib` -- a 5-fold ensemble measured at
0.871 clip / 0.845 frame accuracy on video-disjoint test data -- sat unused.

This adapter loads that artifact and returns the same dict shape the prototype
classifier returns, so the rest of the pipeline is unchanged. If the artifact is
missing (fresh checkout, no training run yet) construction raises and `main.py`
falls back to the prototype path, so the node still runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from config import Settings

ROOT = Path(__file__).resolve().parent
# The artifact pickles PCAAngleProjector by module name, so src/ must be importable.
sys.path.insert(0, str(ROOT / "src"))


class TheftDetector:
    """Wraps LiveTheftClassifier; `classify(frame_bgr)` mirrors QuantumClassifier."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        from inference import LiveTheftClassifier   # imported late: heavy, and optional

        path = ROOT / settings.model_path
        if not path.is_file():
            raise FileNotFoundError(f"no trained artifact at {path}")
        self._clf = LiveTheftClassifier(path)
        meta = self._clf.metadata
        self.description = (
            f"{meta.get('ensemble_size', 1)}-member {meta.get('experiment', 'qsvm')} "
            f"(clip {meta.get('clip_test_accuracy', float('nan')):.3f})")

    def classify(self, frame_bgr: np.ndarray) -> dict:
        r = self._clf.predict_frame(frame_bgr)
        anomaly = float(r["confidence"]) if r["prediction"] == "Anomaly" else 1.0 - float(r["confidence"])
        anomaly = min(max(anomaly, 0.0), 1.0)
        return {
            "trained": True,                      # AIJudge weights this signal higher
            "object_class": "theft" if r["prediction"] == "Anomaly" else "normal",
            "class_scores": {
                "suspicious_signature": anomaly,
                "background": 1.0 - anomaly,
            },
            "margin": float(r.get("margin", 0.0)),
            "features": list(r["features_4d"]),   # the 4 ZZFeatureMap angles
            "member_margins": r["member_margins"],
            "latent": r["latent"],
            "latency_ms": r["latency_ms"],
            "backend": "trained_qsvm",
        }


if __name__ == "__main__":
    det = TheftDetector(Settings())
    print("loaded:", det.description)
    out = det.classify(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8))
    assert out["object_class"] in ("theft", "normal")
    assert 0.0 <= out["class_scores"]["suspicious_signature"] <= 1.0
    assert len(out["features"]) == 4
    assert all(-np.pi - 1e-6 <= v <= np.pi + 1e-6 for v in out["features"])
    print("ok:", out["object_class"], round(out["class_scores"]["suspicious_signature"], 3))
