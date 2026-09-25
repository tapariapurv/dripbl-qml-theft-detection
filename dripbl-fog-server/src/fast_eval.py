"""Fast evaluation + operating-point tuning on cached features.

Re-decoding ~23k PNGs and re-running the ensemble CNNs takes minutes, which is
far too slow for something run repeatedly. The expensive half (decode + CNN
forward) depends only on the model weights, so it is computed once and cached;
everything after it -- angles, quantum kernel, SVM margins, aggregation,
threshold -- is milliseconds.

    .venv/bin/python3 src/fast_eval.py --rebuild   # refresh cache (~2 min, after retraining)
    .venv/bin/python3 src/fast_eval.py             # full report (~2 s)
    .venv/bin/python3 src/fast_eval.py --tune      # also search the operating point
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiments as E  # noqa: E402
import train_qsvm as T  # noqa: E402
from cnn_extractor import embed_batch  # noqa: E402
from inference import LiveTheftClassifier  # noqa: E402
from quantum_kernel import encode, kernel_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "reports" / "feature_cache.npz"
VAL_FRAMES_PER_VIDEO = 20
TEST_FRAMES_PER_VIDEO = 120


def build_cache(clf: LiveTheftClassifier) -> dict:
    """Decode frames once and store each ensemble member's 128-d latents."""
    t0 = time.time()
    out: dict[str, np.ndarray] = {}
    for split, per_video, subset in (("val", VAL_FRAMES_PER_VIDEO, "val"),
                                     ("test", TEST_FRAMES_PER_VIDEO, "all")):
        src = "Train" if split == "val" else "Test"
        paths, y, _, videos = T.sample_by_video(src, per_video, np.random.default_rng(T.SEED),
                                                video_subset=subset)
        images = E._load(paths, False)
        batch = (images.astype(np.float32) / 255.0)
        for m, member in enumerate(clf._members):
            lat = np.concatenate([embed_batch(batch[s:s + 512], member["cnn"])
                                  for s in range(0, len(batch), 512)])
            out[f"{split}_lat{m}"] = lat.astype(np.float32)
        out[f"{split}_y"] = y
        out[f"{split}_vid"] = np.array(videos)
        print(f"  cached {split}: {len(y)} frames x {len(clf._members)} members "
              f"({time.time() - t0:.0f}s)", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f"  wrote {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")
    return out


def member_margins(clf: LiveTheftClassifier, cache: dict, split: str) -> np.ndarray:
    """(n_members, n_frames) SVM margins, straight from cached latents."""
    rows = []
    for m, member in enumerate(clf._members):
        angles = member["projector"].transform(cache[f"{split}_lat{m}"])
        K = np.zeros((len(angles), member["n_train"]), dtype=np.float64)
        K[:, member["clf"].support_] = kernel_matrix(encode(angles), member["states"])
        rows.append(member["clf"].decision_function(K))
    return np.array(rows)


def evaluate(labels: np.ndarray, videos: np.ndarray, margins: np.ndarray,
             how: str, threshold: float) -> dict:
    lab, scores = E.aggregate(list(videos), labels, margins, how)
    report = E.clip_report(lab, scores, threshold)
    report["oracle_threshold_accuracy"] = E.pick_threshold(lab, scores)[1]
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="recompute the cache (after retraining)")
    ap.add_argument("--tune", action="store_true",
                    help="search aggregation/weighting/threshold on val -- ONLY valid for a model "
                         "whose validation videos it never trained on (see warning below)")
    ap.add_argument("--save", action="store_true", help="write the tuned operating point into the model")
    args = ap.parse_args()

    t0 = time.time()
    clf = LiveTheftClassifier()
    cache = dict(np.load(CACHE, allow_pickle=True)) if CACHE.exists() and not args.rebuild \
        else build_cache(clf)
    print(f"[load] {len(clf._members)} members, cache ready ({time.time() - t0:.1f}s)", flush=True)

    val_m = member_margins(clf, cache, "val")
    test_m = member_margins(clf, cache, "test")
    y_val, vid_val = cache["val_y"], cache["val_vid"]
    y_test, vid_test = cache["test_y"], cache["test_vid"]

    how = clf.metadata.get("clip_aggregation", "mean")
    threshold = float(clf.metadata.get("clip_threshold", 0.0))
    weights = np.ones(len(clf._members)) / len(clf._members)

    if args.tune:
        # WARNING: the "val" split here is the 15% video holdout used by
        # train_qsvm.py. A k-fold ensemble (train_final.py) trains its members on
        # *all* Train videos, so for that artifact these clips are training data
        # for 4 of 5 members and the score is inflated (~0.94 val vs 0.85 test).
        # Its threshold is already fitted correctly on each fold's own held-out
        # videos; leave it alone.
        if clf.metadata.get("experiment") == "final_fold_ensemble":
            print("  REFUSING to tune: this k-fold artifact has no unseen validation split left;\n"
                  "  its threshold was fitted on each fold's own held-out videos.", flush=True)
            args.tune = False

    if args.tune:
        # Everything here is chosen on the VALIDATION clips only.
        best = (-1.0, how, threshold, weights)
        val_acc_per_member = []
        for m in range(len(clf._members)):
            lab, sc = E.aggregate(list(vid_val), y_val, val_m[m], "mean")
            val_acc_per_member.append(E.pick_threshold(lab, sc)[1])
        schemes = {
            "uniform": np.ones(len(clf._members)),
            "val_weighted": np.array(val_acc_per_member) ** 4,   # favour stronger folds
            "best_member": np.eye(len(clf._members))[int(np.argmax(val_acc_per_member))],
        }
        for wname, w in schemes.items():
            w = w / w.sum()
            vm = w @ val_m
            for agg in E.AGGREGATIONS:
                lab, sc = E.aggregate(list(vid_val), y_val, vm, agg)
                thr, acc = E.pick_threshold(lab, sc)
                print(f"  {wname:13s} {agg:7s} val_clip_acc={acc:.4f}", flush=True)
                if acc > best[0]:
                    best = (acc, agg, thr, w)
        val_acc, how, threshold, weights = best
        print(f"  chosen: {how} @ {threshold:+.4f}, weights={np.round(weights, 3).tolist()} "
              f"(val {val_acc:.4f})", flush=True)

    report = evaluate(y_test, vid_test, weights @ test_m, how, threshold)
    frame_acc = float((((weights @ test_m) > 0).astype(int) == y_test).mean())
    out = {
        "clip": report, "frame_accuracy": frame_acc, "aggregation": how,
        "threshold": threshold, "members": len(clf._members),
        "n_test_frames": int(len(y_test)), "eval_seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=2))
    (ROOT / "reports" / "fast_eval.json").write_text(json.dumps(out, indent=2))

    if args.save:
        import joblib
        path = ROOT / "models" / "qsvm_model.joblib"
        artifact = joblib.load(path)
        artifact["metadata"].update({
            "clip_aggregation": how, "clip_threshold": threshold,
            "ensemble_weights": weights.tolist(),
            "clip_test_accuracy": report["accuracy"], "frame_test_accuracy": frame_acc,
        })
        joblib.dump(artifact, path)
        print(f"saved operating point -> {path}")


if __name__ == "__main__":
    main()
