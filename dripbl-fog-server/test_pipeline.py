"""Smoke test: full pipeline math and shapes, no camera/network required."""
import numpy as np

from config import DEFAULT
from edge_pipeline import EdgePipeline
from quantum_classifier import QuantumClassifier
from ai_judge import AIJudge
from network_alerter import NetworkAlerter


def _dummy_frame() -> np.ndarray:
    x, y, w, h = DEFAULT.roi
    return np.random.default_rng(0).integers(0, 255, (y + h + 10, x + w + 10, 3), dtype=np.uint8)


def test_extract_features_shape_and_range() -> None:
    pipeline = EdgePipeline(DEFAULT)
    features = pipeline.extract_features(_dummy_frame())
    assert features.shape == (DEFAULT.pca_dim,)
    assert np.all(np.abs(features) <= np.pi + 1e-5)


def test_observe_returns_gait_keys() -> None:
    pipeline = EdgePipeline(DEFAULT)
    _, gait = pipeline.observe(_dummy_frame())
    assert set(gait) == {"speed_px_s", "stride_freq_hz", "jitter_score"}


def test_quantum_classifier_scores_all_prototypes() -> None:
    quantum = QuantumClassifier(DEFAULT)
    result = quantum.classify(np.array([0.1, -0.2, 3.1, -3.1], dtype=np.float32))
    assert result["object_class"] in result["class_scores"]
    assert result["backend"] in ("aer", "classical_fallback")


def test_ai_judge_sensitivity_gates_flag() -> None:
    judge = AIJudge(DEFAULT)
    quantum_result = {"object_class": "person", "class_scores": {"suspicious_signature": 0.9}}
    gait = {"speed_px_s": 300.0, "stride_freq_hz": 1.0, "jitter_score": 0.8}
    verdict = judge.judge(quantum_result, gait)
    assert verdict["suspicion_score"] > 0.5
    assert verdict["flagged"] is (verdict["suspicion_score"] >= DEFAULT.flag_threshold)


def test_payload_roundtrip() -> None:
    alerter = NetworkAlerter(DEFAULT)
    features = np.array([0.1, -0.2, 3.1, -3.1])
    quantum_result = {"object_class": "person", "class_scores": {}, "confidence": 0.5}
    verdict = {"suspicion_score": 0.9, "threshold": 0.5, "flagged": True, "gait": {}}
    payload = alerter.build_payload("t1", features, quantum_result, verdict, None)
    assert payload["trial_id"] == "t1"
    assert len(payload["features"]) == 4
    assert payload["verdict"]["flagged"] is True
    alerter.close()


if __name__ == "__main__":
    test_extract_features_shape_and_range()
    test_observe_returns_gait_keys()
    test_quantum_classifier_scores_all_prototypes()
    test_ai_judge_sensitivity_gates_flag()
    test_payload_roundtrip()
    print("ok")
