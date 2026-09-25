"""Cross-dataset data + evaluation: one fixed, video-disjoint protocol for every model.

The deployed node must catch theft on footage it was not trained on, so every
candidate model is scored on three held-out sets, each clip-level (mean frame margin
vs the artifact's own clip threshold, exactly what the live node does):

  * dataset-3/Test   UCF-Crime, 194 clips, zero source-video overlap with Train
  * dataset-1 hold   retail shoplifting, a fixed stratified 20% of videos (36 clips),
                     split stored in reports/dataset1_split.json so it never drifts
  * dataset-2 hold   cross-camera, 2 clips (not_shoplifting4, shoplifting4)

Frames are decoded straight from the full-resolution video to 64x64, the same way the
live node resizes a 640x480 camera frame, plus a motion-difference image (frame t minus
frame t-10, 0.33 s at 30 fps: the same spacing as dataset-3's frame dumps, so a motion
channel means the same thing on every dataset and on the live camera).

    .venv/bin/python3 -u src/cross_data.py models/qsvm_model.joblib   # cross-dataset report
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "training-data"
SIZE = 64
D1_CACHE = ROOT / "reports" / "dataset1_64.npz"
D1_SPLIT = ROOT / "reports" / "dataset1_split.json"
D1_STRIDE = 5          # 6 frames/s at 30 fps
D1_MAX = 60            # frames per video (clips are 5-21 s)
MOTION_GAP = 10     # = dataset-3 dump spacing (0.33 s), so motion means the same everywhere
D2_HOLDOUT = ("not_shoplifting4", "shoplifting4")
SEED = 42


def _small(frame_bgr: np.ndarray, size: int = SIZE) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def motion_image(cur_rgb: np.ndarray, prev_rgb: np.ndarray) -> np.ndarray:
    """Signed frame difference mapped to uint8 around 128 (same encoding as training)."""
    return np.clip((cur_rgb.astype(np.int16) - prev_rgb.astype(np.int16)) // 2 + 128, 0, 255).astype(np.uint8)


def extract_video(path: Path, max_frames: int = D1_MAX, size: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    """One video -> (rgb, motion) uint8 arrays of size x size frames spread over the whole clip."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(D1_STRIDE, (n - MOTION_GAP) // max_frames)
    buf, idx, rgb, mot = [], 0, [], []
    while len(rgb) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(_small(frame, size))
        buf = buf[-(MOTION_GAP + 1):]
        if idx >= MOTION_GAP and (idx - MOTION_GAP) % stride == 0:
            rgb.append(buf[-1])
            mot.append(motion_image(buf[-1], buf[0]))
        idx += 1
    cap.release()
    return np.array(rgb, np.uint8).reshape(-1, size, size, 3), np.array(mot, np.uint8).reshape(-1, size, size, 3)


def _extract_folder(pairs: list[tuple[Path, int, str]], cache: Path, size: int = SIZE) -> dict[str, np.ndarray]:
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    rgb, mot, ys, vids = [], [], [], []
    for path, label, vid in pairs:
        r, m = extract_video(path, size=size)
        rgb.append(r)
        mot.append(m)
        ys += [label] * len(r)
        vids += [vid] * len(r)
    out = {"rgb": np.concatenate(rgb), "motion": np.concatenate(mot), "y": np.array(ys), "vid": np.array(vids)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    print(f"  {cache.name}: {len(ys)} frames from {len(set(vids))} videos ({time.time() - t0:.0f}s)", flush=True)
    return out


def load_dataset1(size: int = SIZE) -> dict[str, np.ndarray]:
    """All dataset-1 videos -> rgb (N,s,s,3), motion (N,s,s,3), y, vid. Cached per size."""
    cache = D1_CACHE if size == SIZE else D1_CACHE.with_name(f"dataset1_{size}.npz")
    return _extract_folder([(p, label, f"d1:{p.stem}") for folder, label in (("normal", 0), ("shoplifting", 1))
                            for p in sorted((DATA / "dataset-1" / folder).glob("*.mp4"))], cache, size)


def dataset1_split() -> dict[str, list[str]]:
    """Fixed stratified 80/20 split by video; created once, then read from disk."""
    if D1_SPLIT.exists():
        return json.loads(D1_SPLIT.read_text())
    rng = np.random.default_rng(SEED)
    split: dict[str, list[str]] = {"train": [], "holdout": []}
    for folder in ("normal", "shoplifting"):
        names = sorted(f"d1:{p.stem}" for p in (DATA / "dataset-1" / folder).glob("*.mp4"))
        order = rng.permutation(len(names))
        n_hold = round(len(names) * 0.2)
        split["holdout"] += [names[i] for i in order[:n_hold]]
        split["train"] += [names[i] for i in order[n_hold:]]
    D1_SPLIT.write_text(json.dumps(split, indent=1))
    return split


def dataset1_part(part: str, size: int = SIZE) -> dict[str, np.ndarray]:
    d = load_dataset1(size)
    keep = np.isin(d["vid"], dataset1_split()[part])
    return {k: v[keep] for k, v in d.items()}


def load_dataset2(size: int = SIZE) -> dict[str, np.ndarray]:
    """dataset-2 decoded from its source videos, same extractor as dataset-1."""
    return _extract_folder([(p, 0 if p.stem.startswith("not_") else 1, f"d2:{p.stem}")
                            for p in sorted((DATA / "dataset-2" / "videos").glob("*.mp4"))],
                           ROOT / "reports" / f"dataset2_{size}.npz", size)


def dataset2_part(part: str, size: int = SIZE) -> dict[str, np.ndarray]:
    d = load_dataset2(size)
    hold = np.isin(d["vid"], [f"d2:{v}" for v in D2_HOLDOUT])
    keep = hold if part == "holdout" else ~hold
    return {k: v[keep] for k, v in d.items()}


def load_dataset3_test(frames_per_video: int = 60) -> dict[str, np.ndarray]:
    import experiments as E
    import train_qsvm as T
    paths, y, _, vids = T.sample_by_video("Test", frames_per_video, np.random.default_rng(7))
    x = E._load(paths, True)  # RGB + motion vs the previous dump (0.33 s earlier)
    return {"rgb": x[..., :3].copy(), "motion": x[..., 3:].copy(), "y": np.asarray(y),
            "vid": np.array([f"d3:{v}" for v in vids])}


# ------------------------------------------------------------------ evaluation

def clip_scores(margins: np.ndarray, vids: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, scores = [], []
    for v in dict.fromkeys(vids.tolist()):
        idx = vids == v
        labels.append(int(y[idx][0]))
        scores.append(float(margins[idx].mean()))
    return np.array(labels), np.array(scores)


def report(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import roc_auc_score
    pred = (scores > threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    return {"clips": int(len(labels)), "accuracy": float((pred == labels).mean()),
            "recall": tp / max(int((labels == 1).sum()), 1),
            "fp": int(((pred == 1) & (labels == 0)).sum()), "fn": int(((pred == 0) & (labels == 1)).sum()),
            "auc": float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) > 1 else float("nan")}


def holdout_clip_scores(margin_fn, d3_frames: int = 60) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Score every hold-out clip once: {set: (labels, clip scores)}.
    margin_fn(rgb_uint8 (N,64,64,3), motion or None, vids) -> (N,) frame margins."""
    sets = {
        "dataset3_test": load_dataset3_test(d3_frames),
        "dataset1_holdout": dataset1_part("holdout"),
        "dataset2_holdout": dataset2_part("holdout"),
    }
    return {name: clip_scores(margin_fn(d["rgb"], d.get("motion"), d["vid"]), d["vid"], d["y"])
            for name, d in sets.items()}


def evaluate(margin_fn, threshold: float, d3_frames: int = 60, thresholds: dict[str, float] | None = None,
             scores: dict | None = None) -> dict[str, dict]:
    """Clip-level report per hold-out set (pass `scores` to reuse holdout_clip_scores)."""
    scores = scores or holdout_clip_scores(margin_fn, d3_frames)
    return {name: report(lab, sc, (thresholds or {}).get(name, threshold)) for name, (lab, sc) in scores.items()}


def artifact_margin_fn(model_path: str | Path):
    import os
    import torch
    from inference import LiveTheftClassifier
    clf = LiveTheftClassifier(model_path)
    torch.set_num_threads(os.cpu_count() or 4)  # offline evaluation: use every core (the node pins 1)

    def fn(rgb: np.ndarray, motion, vids) -> np.ndarray:
        x = np.concatenate([rgb, motion], -1) if clf.in_channels == 6 else rgb
        return clf.margins_from_arrays(x)
    return fn, float(clf.metadata.get("clip_threshold", 0.0))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "models" / "qsvm_model.joblib")
    fn, thr = artifact_margin_fn(path)
    res = evaluate(fn, thr)
    for k, v in res.items():
        print(f"{k:18s} acc={v['accuracy']:.3f} auc={v['auc']:.3f} recall={v['recall']:.3f} "
              f"fp={v['fp']} fn={v['fn']} clips={v['clips']}")
    print(json.dumps(res))
