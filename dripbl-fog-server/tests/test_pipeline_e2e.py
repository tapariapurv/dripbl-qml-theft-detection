"""Assert-based smoke test for the QSVM pipeline: data sampling, artifact
round-trip, live-inference latency/shape, and a real accuracy floor on Test/.
Run: .venv/bin/python3 tests/test_pipeline_e2e.py (after training once).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "qsvm_model.joblib"
# Floors are regression guards, set just under the measured values so a real
# regression trips them while ordinary sampling noise does not.
MIN_FRAME_ACCURACY = 0.78   # measured 0.845 on the held-out Test split
MIN_CLIP_ACCURACY = 0.83    # measured 0.897 (CCTV expert; 0.871 for the original model)
MIN_RETAIL_CLIP_ACCURACY = 0.75  # retail expert on the dataset-1 hold-out (36 clips: 1 clip = 2.8 pts)


def test_sampling_is_split_clean() -> None:
    from train_qsvm import DATASET_DIR, sample_split

    rng = np.random.default_rng(0)
    train_paths, y_train, _ = sample_split("Train", 20, rng)
    test_paths, y_test, groups = sample_split("Test", 20, rng)

    assert len(y_train) == 40 and len(y_test) == 40
    assert sorted(np.bincount(y_train)) == [20, 20], "train sample must be class balanced"
    assert sorted(np.bincount(y_test)) == [20, 20], "test sample must be class balanced"
    assert all(str(DATASET_DIR / "Train") in str(p) for p in train_paths)
    assert all(str(DATASET_DIR / "Test") in str(p) for p in test_paths), "test must come from Test/"
    assert not ({p.name for p in train_paths} & {p.name for p in test_paths})
    assert set(groups) <= {"NormalVideos", "Burglary", "Robbery", "Shoplifting", "Stealing"}
    print("ok: sampling_is_split_clean")


def test_artifact_round_trip() -> None:
    assert MODEL_PATH.is_file(), f"missing {MODEL_PATH} — run src/train_qsvm.py first"
    artifact = joblib.load(MODEL_PATH)
    assert "metadata" in artifact
    # format 2 (train_xd.py) keeps everything per ensemble member; format 1 also at top level
    members = artifact["ensemble"] if artifact.get("format") == 2 else [artifact]
    encoder = "backbone_state" if artifact.get("format") == 2 else "cnn_state_dict"
    for m in members:
        for key in ("svm_classifier", "pca_transformer", "support_vectors", encoder):
            assert key in m, f"ensemble member missing key: {key}"
        assert m["support_vectors"].shape[1] == 4
    print(f"ok: artifact_round_trip (format {artifact.get('format', 1)}, {len(members)} members)")


def test_live_predict_frame() -> None:
    from inference import LiveTheftClassifier

    clf = LiveTheftClassifier(MODEL_PATH)
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3)).astype(np.uint8)
    clf.predict_frame(frame)  # warm up

    t0 = time.perf_counter()
    result = clf.predict_frame(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    assert result["prediction"] in ("Normal", "Anomaly")
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["features_4d"]) == 4
    assert all(-np.pi - 1e-6 <= v <= np.pi + 1e-6 for v in result["features_4d"])
    assert elapsed_ms < 150.0, f"predict_frame took {elapsed_ms:.1f}ms, budget is 150ms"
    print(f"ok: live_predict_frame ({elapsed_ms:.1f}ms)")


def test_accuracy_on_held_out_test_split() -> None:
    """Frame- and clip-level accuracy on dataset-3/Test, which no fitting step saw."""
    from inference import LiveTheftClassifier
    from train_qsvm import _load_images, sample_by_video

    clf = LiveTheftClassifier(MODEL_PATH)
    paths, y_true, _, videos = sample_by_video("Test", 20, np.random.default_rng(123))
    images = _load_images(paths)
    results = clf.predict_batch(images[:, :, :, ::-1])
    margins = np.array([r["margin"] for r in results])

    frame_acc = float(((margins > 0).astype(int) == y_true).mean())
    assert frame_acc >= MIN_FRAME_ACCURACY, f"frame accuracy {frame_acc:.3f} < {MIN_FRAME_ACCURACY}"

    # clip verdicts: aggregate frame margins per source video, as deployment does
    threshold = float(clf.metadata.get("clip_threshold", 0.0))
    labels, preds = [], []
    for video in dict.fromkeys(videos):
        idx = [i for i, v in enumerate(videos) if v == video]
        labels.append(int(y_true[idx][0]))
        preds.append(int(margins[idx].mean() > threshold))
    clip_acc = float((np.array(preds) == np.array(labels)).mean())
    assert clip_acc >= MIN_CLIP_ACCURACY, f"clip accuracy {clip_acc:.3f} < {MIN_CLIP_ACCURACY}"
    print(f"ok: accuracy_on_held_out_test_split (frame {frame_acc:.3f}, clip {clip_acc:.3f} over {len(labels)} clips)")


def test_retail_holdout_accuracy() -> None:
    """The retail expert on the fixed dataset-1 hold-out (36 clips no model trained on)."""
    retail = MODEL_PATH.parent / "retail_model.joblib"
    if not retail.is_file():
        print("skip: retail_holdout_accuracy (no models/retail_model.joblib)")
        return
    import cross_data as C
    fn, thr = C.artifact_margin_fn(retail)
    clf_meta = joblib.load(retail)["metadata"]
    d = C.dataset1_part("holdout")
    labels, scores = C.clip_scores(fn(d["rgb"], d["motion"], d["vid"]), d["vid"], d["y"])
    r = C.report(labels, scores, clf_meta.get("clip_thresholds", {}).get("retail", thr))
    assert r["accuracy"] >= MIN_RETAIL_CLIP_ACCURACY, f"retail hold-out accuracy {r['accuracy']:.3f}"
    print(f"ok: retail_holdout_accuracy ({r['accuracy']:.3f}, AUC {r['auc']:.3f}, {r['clips']} clips)")


def test_quantum_encoder_matches_qiskit() -> None:
    """The fast analytic encoder must stay identical to qiskit's simulator."""
    from quantum_kernel import verify_fast_encode

    delta = verify_fast_encode()
    print(f"ok: quantum_encoder_matches_qiskit (max kernel delta {delta:.1e})")


if __name__ == "__main__":
    test_sampling_is_split_clean()
    test_artifact_round_trip()
    test_live_predict_frame()
    test_accuracy_on_held_out_test_split()
    test_retail_holdout_accuracy()
    test_quantum_encoder_matches_qiskit()
    print("ALL TESTS PASSED")
