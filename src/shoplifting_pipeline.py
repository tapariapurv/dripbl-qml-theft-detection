"""Higher-resolution, genuinely temporal pipeline on dataset-1 (retail shoplifting).

dataset-3 capped accuracy at ~0.87 for two structural reasons: frames are 64x64,
and only every 10th frame was kept so no real motion is recoverable. dataset-1
fixes both -- 182 balanced videos (90 normal / 92 shoplifting) at 640x480, 30 fps,
all frames present -- and its task (retail theft) is what this project is for.

Pipeline is unchanged where it matters: frames -> LightweightCNN (128-d latent)
-> PLS(4) angles in [-pi, pi] -> ZZFeatureMap fidelity kernel -> SVC(precomputed),
with the clip verdict aggregated over frames.

Evaluation is 5-fold cross-validation BY VIDEO: every clip is tested exactly once
by a model that never saw it, and the threshold for each fold comes from an inner
split of that fold's own training videos. With only 182 clips a single holdout
would be ~36 clips (one clip = 2.8 accuracy points), far too noisy to steer on.

    .venv/bin/python3 -u src/shoplifting_pipeline.py            # extract (cached) + CV
    .venv/bin/python3 -u src/shoplifting_pipeline.py --folds 3  # quicker
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiments as E  # noqa: E402
import train_qsvm as T  # noqa: E402
from pca_projector import PCAAngleProjector  # noqa: E402
from quantum_kernel import FEATURE_DIM, encode, kernel_matrix  # noqa: E402
from edge_pipeline import LightweightCNN  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "training-data" / "dataset-1"
CACHE = ROOT / "reports" / "dataset1_frames.npz"
SIZE = 96              # 2.25x the pixels of the 64x64 dump
FRAME_STRIDE = 5       # ~6 frames/second kept
MOTION_GAP = 2         # frames back for the difference channels (~0.07 s: real motion)
MAX_PER_VIDEO = 60
SEED = 42


def extract() -> dict:
    """Decode every video once into (N, 96, 96, 6): RGB + motion difference."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}

    t0 = time.time()
    frames, labels, videos = [], [], []
    for folder, label in (("normal", 0), ("shoplifting", 1)):
        for path in sorted((DATA / folder).glob("*.mp4")):
            cap = cv2.VideoCapture(str(path))
            buf, kept, idx = [], 0, 0
            while kept < MAX_PER_VIDEO:
                ok, frame = cap.read()
                if not ok:
                    break
                small = cv2.cvtColor(cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA),
                                     cv2.COLOR_BGR2RGB)
                buf.append(small)
                if idx % FRAME_STRIDE == 0 and len(buf) > MOTION_GAP:
                    cur = buf[-1].astype(np.int16)
                    prev = buf[-1 - MOTION_GAP].astype(np.int16)
                    motion = np.clip((cur - prev) // 2 + 128, 0, 255).astype(np.uint8)
                    frames.append(np.dstack([buf[-1], motion]))
                    labels.append(label)
                    videos.append(path.stem)
                    kept += 1
                buf = buf[-(MOTION_GAP + 1):]
                idx += 1
            cap.release()
        print(f"  {folder}: {labels.count(label)} frames ({time.time() - t0:.0f}s)", flush=True)

    data = {"x": np.array(frames, dtype=np.uint8), "y": np.array(labels),
            "vid": np.array(videos)}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **data)
    print(f"  cached {data['x'].shape} -> {CACHE} ({CACHE.stat().st_size / 1e6:.0f} MB, "
          f"{time.time() - t0:.0f}s)", flush=True)
    return data


class Net(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.backbone = LightweightCNN(latent_dim=128, in_channels=in_channels)
        self.head = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def _tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).permute(0, 3, 1, 2).float().div_(255.0)


def _augment(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pad = 8
    padded = np.pad(batch, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    ys, xs = rng.integers(0, 2 * pad, len(batch)), rng.integers(0, 2 * pad, len(batch))
    flips = rng.random(len(batch)) < 0.5
    out = np.empty_like(batch)
    for i in range(len(batch)):
        crop = padded[i, ys[i]:ys[i] + SIZE, xs[i]:xs[i] + SIZE]
        out[i] = crop[:, ::-1] if flips[i] else crop
    gain = rng.uniform(0.8, 1.25, (len(batch), 1, 1, 1))
    bias = rng.uniform(-20, 20, (len(batch), 1, 1, 1))
    out[..., :3] = np.clip(out[..., :3] * gain + bias, 0, 255).astype(np.uint8)
    return out


@torch.inference_mode()
def _infer(model: nn.Module, x: np.ndarray, backbone: bool = False) -> np.ndarray:
    model.eval()
    net = model.backbone if backbone else model
    return np.concatenate([net(_tensor(x[s:s + 256])).numpy() for s in range(0, len(x), 256)])


def train_fold(x_fit: np.ndarray, y_fit: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
               epochs: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Net(x_fit.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=5e-4)
    steps = epochs * int(np.ceil(len(y_fit) / 128))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=steps)
    counts = np.bincount(y_fit, minlength=2).astype(np.float64)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32))
    yt = torch.from_numpy(y_fit).long()

    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(y_fit))
        for s in range(0, len(order), 128):
            idx = order[s:s + 128]
            loss = loss_fn(model(_tensor(np.ascontiguousarray(_augment(x_fit[idx], rng)))), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        acc = float((_infer(model, x_val).argmax(1) == y_val).mean())
        if acc >= best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"        epoch {ep + 1}/{epochs} inner_val={acc:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model


def qsvm(lat_fit, y_fit, lat_val, y_val, lat_test):
    best = (-1.0, None, None, None)
    projectors = {m: PCAAngleProjector(n_components=FEATURE_DIM, mode=m).fit(lat_fit, y_fit)
                  for m in ("pca", "pls")}
    for mode, proj in projectors.items():
        for scale in T.ANGLE_SCALES:
            proj.angle_scale = scale
            e_fit, e_val = encode(proj.transform(lat_fit)), encode(proj.transform(lat_val))
            k_fit, k_val = kernel_matrix(e_fit), kernel_matrix(e_val, e_fit)
            for c in T.C_GRID:
                svc = SVC(kernel="precomputed", C=c, class_weight="balanced").fit(k_fit, y_fit)
                acc = balanced_accuracy_score(y_val, svc.predict(k_val))
                if acc > best[0]:
                    best = (float(acc), mode, scale, c)
    val_acc, mode, scale, c = best
    proj = projectors[mode]
    proj.angle_scale = scale
    ang = proj.transform(lat_fit)
    e_fit = encode(ang)
    clf = SVC(kernel="precomputed", C=c, class_weight="balanced").fit(kernel_matrix(e_fit), y_fit)
    val_m = clf.decision_function(kernel_matrix(encode(proj.transform(lat_val)), e_fit))
    test_m = clf.decision_function(kernel_matrix(encode(proj.transform(lat_test)), e_fit))
    return {"proj": proj, "clf": clf, "support": ang[clf.support_], "support_y": y_fit[clf.support_],
            "val_m": val_m, "test_m": test_m, "val_acc": val_acc, "mode": mode,
            "scale": scale, "C": c}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    t0 = time.time()

    print("[1/3] frames ...", flush=True)
    data = extract()
    x, y, vid = data["x"], data["y"], data["vid"]
    videos = np.array(sorted(set(vid.tolist())))
    vlabel = np.array([y[vid == v][0] for v in videos])
    print(f"  {len(y)} frames, {len(videos)} videos "
          f"({int((vlabel == 0).sum())} normal / {int((vlabel == 1).sum())} shoplifting)", flush=True)

    order = np.random.default_rng(SEED).permutation(len(videos))
    clip_true, clip_pred, clip_score, fold_reports, fits = [], [], [], [], []

    for fold in range(args.folds):
        test_v = set(videos[order[fold::args.folds]].tolist())
        inner = [v for v in videos if v not in test_v]
        val_v = set(inner[::6])                       # inner split for threshold + selection
        m_test = np.isin(vid, list(test_v))
        m_val = np.isin(vid, list(val_v))
        m_fit = ~m_test & ~m_val
        print(f"[2/3] fold {fold + 1}/{args.folds}: fit={m_fit.sum()} val={m_val.sum()} "
              f"test={m_test.sum()} frames / {len(test_v)} test clips", flush=True)

        model = train_fold(x[m_fit], y[m_fit], x[m_val], y[m_val], args.epochs, SEED + fold)
        r = qsvm(_infer(model, x[m_fit], True), y[m_fit], _infer(model, x[m_val], True), y[m_val],
                 _infer(model, x[m_test], True))
        r["model"] = model
        fits.append(r)

        lab_v, sc_v = E.aggregate(list(vid[m_val]), y[m_val], r["val_m"], "mean")
        thr, inner_acc = E.pick_threshold(lab_v, sc_v)
        lab_t, sc_t = E.aggregate(list(vid[m_test]), y[m_test], r["test_m"], "mean")
        pred_t = (sc_t > thr).astype(int)
        acc = float((pred_t == lab_t).mean())
        clip_true += lab_t.tolist()
        clip_pred += pred_t.tolist()
        clip_score += sc_t.tolist()
        fold_reports.append({"fold": fold, "clips": int(len(lab_t)), "accuracy": acc,
                             "inner_val": inner_acc, "threshold": thr, "svm_val": r["val_acc"],
                             "projector": r["mode"], "angle_scale": r["scale"], "C": r["C"]})
        print(f"      fold clip accuracy={acc:.4f} ({int((pred_t == lab_t).sum())}/{len(lab_t)} clips) "
              f"proj={r['mode']} scale={r['scale']}", flush=True)

    ct, cp, cs = np.array(clip_true), np.array(clip_pred), np.array(clip_score)
    cm = confusion_matrix(ct, cp, labels=[0, 1])
    metrics = {
        "dataset": "dataset-1 (retail shoplifting, 640x480 source, 96x96 + motion)",
        "protocol": f"{args.folds}-fold cross-validation by video; every clip tested once by a model that never saw it",
        "n_clips": int(len(ct)), "n_frames": int(len(y)),
        "clip_accuracy": float((cp == ct).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(ct, cp)),
        "recall": float(cm[1, 1] / max(cm[1].sum(), 1)),
        "precision": float(cm[1, 1] / max(cm[:, 1].sum(), 1)),
        "false_negative_rate": float(cm[1, 0] / max(cm[1].sum(), 1)),
        "roc_auc": float(roc_auc_score(ct, cs)),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "per_fold": fold_reports,
        "runtime_sec": round(time.time() - t0, 1),
    }
    print("[3/3] " + json.dumps({k: v for k, v in metrics.items() if k != "per_fold"}, indent=2), flush=True)
    (ROOT / "reports" / "shoplifting_metrics.json").write_text(json.dumps(metrics, indent=2))

    members = [{"svm_classifier": f["clf"], "pca_transformer": f["proj"],
                "support_vectors": f["support"], "support_vector_labels": f["support_y"],
                "cnn_state_dict": f["model"].backbone.state_dict()} for f in fits]
    joblib.dump({**members[0], "ensemble": members, "metadata": {
        "feature_map": {"type": "ZZFeatureMap", "feature_dimension": FEATURE_DIM,
                        "reps": 2, "entanglement": "linear"},
        "svm_hyperparams": {"kernel": "precomputed", "C": fits[0]["C"]},
        "angle_scale": fits[0]["scale"], "projector_mode": fits[0]["mode"],
        "cnn_latent_dim": 128, "in_channels": 6, "frame_size": SIZE, "temporal": True,
        "motion_gap": MOTION_GAP, "dataset": "dataset-1",
        "classes": {"0": "Normal", "1": "Shoplifting"},
        "clip_aggregation": "mean",
        "clip_threshold": float(np.mean([f["threshold"] for f in fold_reports])),
        "ensemble_size": len(members), "experiment": "shoplifting_cv",
        "clip_test_accuracy": metrics["clip_accuracy"], "timestamp": time.time(),
    }}, ROOT / "models" / "shoplifting_model.joblib")
    print(f"saved -> {ROOT / 'models' / 'shoplifting_model.joblib'}  ({metrics['runtime_sec']:.0f}s)")


if __name__ == "__main__":
    main()
