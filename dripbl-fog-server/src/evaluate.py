"""Test the trained QSVM theft detector in several different ways.

    .venv/bin/python3 src/evaluate.py testset            # metrics on dataset-3/Test
    .venv/bin/python3 src/evaluate.py testset -n 5000    # bigger test sample
    .venv/bin/python3 src/evaluate.py trainset           # same, on Train/ (overfit check)
    .venv/bin/python3 src/evaluate.py image path/to.png  # one image
    .venv/bin/python3 src/evaluate.py dir path/to/folder # every image in a folder
    .venv/bin/python3 src/evaluate.py video clip.mp4     # sample frames from a video
    .venv/bin/python3 src/evaluate.py bench              # latency benchmark
    .venv/bin/python3 src/evaluate.py camera             # live webcam / synthetic feed

Every mode prints a compact report; `testset`/`trainset` also write a confusion
matrix PNG under reports/figures/.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import LiveTheftClassifier  # noqa: E402
from train_qsvm import (CLASS_DIRS, DATASET_DIR, REPORTS_DIR, _load_images, best_threshold,  # noqa: E402
                        clip_metrics, clip_scores, sample_by_video, sample_split)

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def _report(y_true: np.ndarray, y_pred: np.ndarray, margins: np.ndarray,
            groups: list[str], split: str, out_png: Path | None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "split": split,
        "n": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_negative_rate": float(cm[1, 0] / max(cm[1].sum(), 1)),
        "false_positive_rate": float(cm[0, 1] / max(cm[0].sum(), 1)),
        "roc_auc": float(roc_auc_score(y_true, margins)) if len(set(y_true)) > 1 else float("nan"),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "per_source_accuracy": {
            g: float((y_pred[idx] == y_true[idx]).mean())
            for g in dict.fromkeys(groups)
            if (idx := [i for i, gg in enumerate(groups) if gg == g])
        },
    }
    print(json.dumps(metrics, indent=2))
    if out_png is not None:
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"]).plot(cmap="Blues")
        plt.title(f"{split} — accuracy {metrics['accuracy']:.1%} (n={len(y_true)})")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"confusion matrix -> {out_png}")
    return metrics


def eval_split(clf: LiveTheftClassifier, split: str, per_label: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    paths, y_true, groups = sample_split(split, per_label, rng)
    images = _load_images(paths)
    t0 = time.time()
    results = clf.predict_batch(images[:, :, :, ::-1])  # RGB store -> BGR for predict
    elapsed = time.time() - t0
    y_pred = np.array([1 if r["prediction"] == "Anomaly" else 0 for r in results])
    margins = np.array([r["margin"] for r in results])
    print(f"scored {len(y_true)} frames in {elapsed:.1f}s ({1000 * elapsed / len(y_true):.2f} ms/frame batched)")
    _report(y_true, y_pred, margins, groups, f"dataset-3/{split}",
            REPORTS_DIR / "figures" / f"cm_eval_{split.lower()}.png")


def _clip_scores_for(clf: LiveTheftClassifier, split: str, frames: int, seed: int,
                     video_subset: str = "all"):
    rng = np.random.default_rng(seed)
    paths, y, _, videos = sample_by_video(split, frames, rng, video_subset=video_subset)
    images = _load_images(paths)
    margins = np.array([r["margin"] for r in clf.predict_batch(images[:, :, :, ::-1])])
    return (*clip_scores(videos, y, margins), videos, y, margins)


def eval_clips(clf: LiveTheftClassifier, frames: int, retune: str | None, model_path: str | None) -> None:
    """Clip-level verdicts: average the frame margins over each source clip, the
    way the live node aggregates a stream instead of judging one frame."""
    threshold = float(clf.metadata.get("clip_threshold", 0.0))
    if retune:
        vl, vs, *_ = _clip_scores_for(clf, "Train", frames, 5, video_subset="val")
        if retune == "accuracy":
            grid = np.unique(np.round(vs, 4))
            cand = np.concatenate([[vs.min() - 1e-3], (grid[:-1] + grid[1:]) / 2, [vs.max() + 1e-3]])
            threshold = float(max(cand, key=lambda t: ((vs > t).astype(int) == vl).mean()))
        else:
            threshold = best_threshold(vl, vs)
        print(f"retuned clip threshold on {len(vl)} held-out Train clips ({retune}): {threshold:+.4f}")

    _, _, videos, y, margins = _clip_scores_for(clf, "Test", frames, 7)
    report = {
        "frames_per_clip": frames,
        "at_tuned_threshold": clip_metrics(videos, y, margins, threshold),
        "at_zero_threshold": clip_metrics(videos, y, margins, 0.0),
    }
    print(json.dumps(report, indent=2))

    if retune and model_path is not False:
        import joblib
        path = Path(model_path) if model_path else Path(__file__).resolve().parent.parent / "models" / "qsvm_model.joblib"
        artifact = joblib.load(path)
        artifact["metadata"]["clip_threshold"] = threshold
        joblib.dump(artifact, path)
        print(f"saved clip_threshold={threshold:+.4f} -> {path}")


def eval_image(clf: LiveTheftClassifier, path: Path) -> None:
    frame = cv2.imread(str(path))
    if frame is None:
        sys.exit(f"could not read image: {path}")
    result = clf.predict_frame(frame)
    print(json.dumps({"file": path.name, **result}, indent=2))


def eval_dir(clf: LiveTheftClassifier, path: Path, limit: int) -> None:
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXT)[:limit]
    if not files:
        sys.exit(f"no images found in {path}")
    frames = [cv2.imread(str(p)) for p in files]
    results = clf.predict_batch(frames)
    anomalies = [f.name for f, r in zip(files, results) if r["prediction"] == "Anomaly"]
    print(json.dumps({
        "directory": str(path),
        "n_images": len(files),
        "anomaly_count": len(anomalies),
        "anomaly_rate": len(anomalies) / len(files),
        "mean_confidence": float(np.mean([r["confidence"] for r in results])),
        "first_anomalies": anomalies[:10],
        # folder-name label is a convenience for the dataset-3 layout
        "accuracy_if_folder_labeled": (
            float(np.mean([(r["prediction"] == "Anomaly") == bool(CLASS_DIRS[path.name])
                           for r in results])) if path.name in CLASS_DIRS else None
        ),
    }, indent=2))


def eval_video(clf: LiveTheftClassifier, path: Path, stride: int, limit: int) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"could not open video: {path}")
    frames, idx = [], 0
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        sys.exit("no frames decoded")
    results = clf.predict_batch(frames)
    flags = [r["prediction"] == "Anomaly" for r in results]
    print(json.dumps({
        "video": str(path),
        "frames_scored": len(frames),
        "anomaly_frames": int(sum(flags)),
        "anomaly_rate": float(np.mean(flags)),
        "verdict": "Anomaly" if np.mean(flags) > 0.5 else "Normal",
        "peak_confidence": float(max(r["confidence"] for r in results)),
    }, indent=2))


def bench(clf: LiveTheftClassifier, n: int) -> None:
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(n)]
    clf.predict_frame(frames[0])  # warm up
    times = [clf.predict_frame(f)["latency_ms"] for f in frames]
    print(json.dumps({
        "frames": n,
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "max_ms": float(np.max(times)),
        "budget_ms": 150,
        "within_budget": bool(np.percentile(times, 95) < 150),
    }, indent=2))


def camera(clf: LiveTheftClassifier, source: str, seconds: float) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from camera import FrameGrabber
    from config import Settings

    src: str | int = int(source) if source.isdigit() else source
    grabber = FrameGrabber(Settings(camera_source=src))
    deadline, counts, n = time.time() + seconds, {"Normal": 0, "Anomaly": 0}, 0
    try:
        while time.time() < deadline:
            frame = grabber.read()
            if frame is None:
                continue
            r = clf.predict_frame(frame)
            counts[r["prediction"]] += 1
            n += 1
            print(f"\r{r['prediction']:>8}  conf={r['confidence']:.3f}  {r['latency_ms']:6.1f}ms", end="")
    except KeyboardInterrupt:
        pass
    finally:
        grabber.release()
        print(f"\nscored {n} frames: {counts}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["testset", "trainset", "clips", "image", "dir", "video", "bench", "camera"])
    ap.add_argument("target", nargs="?", help="path (image/dir/video) or camera source")
    ap.add_argument("-n", type=int, default=None, help="sample size per label / frame count")
    ap.add_argument("--model", default=None, help="path to qsvm_model.joblib")
    ap.add_argument("--seed", type=int, default=7, help="sampling seed for testset/trainset")
    ap.add_argument("--stride", type=int, default=10, help="video: keep every Nth frame")
    ap.add_argument("--seconds", type=float, default=10.0, help="camera: run duration")
    ap.add_argument("--frames-per-clip", type=int, default=60, help="clips: frames averaged per clip")
    ap.add_argument("--retune", choices=["accuracy", "balanced"], default=None,
                    help="clips: re-pick the clip threshold on held-out Train clips and save it")
    args = ap.parse_args()

    clf = LiveTheftClassifier(args.model) if args.model else LiveTheftClassifier()
    if args.mode == "testset":
        eval_split(clf, "Test", args.n or 1500, args.seed)
    elif args.mode == "trainset":
        eval_split(clf, "Train", args.n or 1500, args.seed)
    elif args.mode == "clips":
        eval_clips(clf, args.frames_per_clip, args.retune, args.model)
    elif args.mode == "image":
        eval_image(clf, Path(args.target))
    elif args.mode == "dir":
        eval_dir(clf, Path(args.target), args.n or 2000)
    elif args.mode == "video":
        eval_video(clf, Path(args.target), args.stride, args.n or 200)
    elif args.mode == "bench":
        bench(clf, args.n or 50)
    else:
        camera(clf, args.target or "0", args.seconds)


if __name__ == "__main__":
    main()
