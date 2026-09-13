"""End-to-end QSVM training for DRIBPL theft detection.

Pipeline: 64x64 RGB frames -> trained LightweightCNN (128-d latent) -> PCA(4)
-> min-max scale to [-pi, pi] -> ZZFeatureMap fidelity quantum kernel -> SVC
(precomputed kernel).

Stages:
  A. Supervised CNN training on dataset-3/Train ONLY (with a Train-internal
     validation holdout). Without this the latent space is random noise and the
     downstream QSVM cannot do better than chance.
  B. PCA + angle scaling + quantum kernel + SVC, fit on Train features; C is
     picked on the Train-internal validation split.
  C. Evaluation strictly on dataset-3/Test, which is never seen during any
     fitting step.

Run: .venv/bin/python3 src/train_qsvm.py
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.svm import SVC
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_extractor import embed_batch  # noqa: E402
from pca_projector import PCAAngleProjector  # noqa: E402
from quantum_kernel import FEATURE_DIM, encode, kernel_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edge_pipeline import LightweightCNN  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT.parent / "training-data" / "dataset-3"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

CLASS_DIRS = {  # folder -> binary label (0 = Normal, 1 = Anomaly)
    "NormalVideos": 0,
    "Burglary": 1,
    "Robbery": 1,
    "Shoplifting": 1,
    "Stealing": 1,
}
CRIME_DIRS = [d for d, lab in CLASS_DIRS.items() if lab == 1]

FRAME_SIZE = 64
SEED = 42
FRAMES_PER_VIDEO = 45        # cap per source clip -> scene diversity over clip length
VAL_FRAMES_PER_VIDEO = 20
TEST_FRAMES_PER_VIDEO = 45
SVM_FIT_PER_LABEL = 1800     # samples whose Gram matrix the SVC is fit on
EPOCHS = 12
MIXUP_ALPHA = 0.0    # mixup measurably hurt the downstream kernel geometry
LABEL_SMOOTHING = 0.0
BATCH_SIZE = 128
LR = 1.5e-3
DROPOUT = 0.5
WEIGHT_DECAY = 5e-4
C_GRID = (1.0, 4.0, 16.0, 64.0, 256.0)
ANGLE_SCALES = (0.15, 0.3, 0.6, 1.2, np.pi)  # encoding half-range, always <= pi


def _load_images(paths: list[Path]) -> np.ndarray:
    """-> (N, 64, 64, 3) uint8 RGB. Frames already ship at 64x64; resize is a no-op guard."""
    out = np.empty((len(paths), FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p))
        if bgr.shape[:2] != (FRAME_SIZE, FRAME_SIZE):
            bgr = cv2.resize(bgr, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        out[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out


def _quotas(per_label: int) -> list[tuple[str, int]]:
    share = per_label // len(CRIME_DIRS)
    return [("NormalVideos", per_label)] + [
        (d, share + (1 if i < per_label - share * len(CRIME_DIRS) else 0))
        for i, d in enumerate(CRIME_DIRS)
    ]


def _video_id(filename: str) -> str:
    """'Burglary001_x264_137.png' -> 'Burglary001_x264' (the source clip)."""
    return filename.rsplit("_", 1)[0]


def sample_split(
    split: str, per_label: int, rng: np.random.Generator, video_subset: str = "all",
    video_holdout: float = 0.15,
) -> tuple[list[Path], np.ndarray, list[str]]:
    """Balanced sample: `per_label` normal frames and `per_label` anomaly frames
    (the anomaly half split evenly across the four crime folders).

    `video_subset` splits by SOURCE CLIP, not by frame: "fit" draws from 85% of
    each folder's videos and "val" from the disjoint 15%. Frames from one clip
    are near-duplicates, so a frame-level holdout would let the CNN memorise
    scenes and report a val score that does not transfer to unseen clips.

    Returns (paths, labels, source folder per sample).
    """
    split_dir = DATASET_DIR / split
    paths: list[Path] = []
    labels: list[int] = []
    groups: list[str] = []

    for folder, quota in _quotas(per_label):
        by_video: dict[str, list[str]] = {}
        for name in sorted(os.listdir(split_dir / folder)):
            by_video.setdefault(_video_id(name), []).append(name)

        videos = sorted(by_video)
        if video_subset != "all":
            order = np.random.default_rng(SEED).permutation(len(videos))
            cut = max(1, int(len(videos) * (1.0 - video_holdout)))
            keep = order[:cut] if video_subset == "fit" else order[cut:]
            videos = [videos[i] for i in sorted(keep)]

        pool = [n for v in videos for n in by_video[v]]
        chosen = rng.choice(len(pool), size=min(quota, len(pool)), replace=False)
        for i in chosen:
            paths.append(split_dir / folder / pool[i])
            labels.append(CLASS_DIRS[folder])
            groups.append(folder)
    return paths, np.array(labels), groups


def sample_by_video(
    split: str, frames_per_video: int, rng: np.random.Generator, video_subset: str = "all",
    video_holdout: float = 0.15, max_videos: int | None = None,
) -> tuple[list[Path], np.ndarray, list[str], list[str]]:
    """Sample up to `frames_per_video` frames from EVERY video of the split.

    Capping per video is what buys scene diversity: a flat random draw over all
    frames is dominated by whichever clips happen to be longest, so the CNN sees
    few distinct scenes and memorises them. Returns (paths, labels, folders,
    video ids) — the video ids are what clip-level evaluation aggregates over.
    """
    split_dir = DATASET_DIR / split
    paths: list[Path] = []
    labels: list[int] = []
    folders: list[str] = []
    videos_out: list[str] = []

    for folder in CLASS_DIRS:
        by_video: dict[str, list[str]] = {}
        for name in sorted(os.listdir(split_dir / folder)):
            by_video.setdefault(_video_id(name), []).append(name)

        videos = sorted(by_video)
        if video_subset != "all":
            order = np.random.default_rng(SEED).permutation(len(videos))
            cut = max(1, int(len(videos) * (1.0 - video_holdout)))
            keep = order[:cut] if video_subset == "fit" else order[cut:]
            videos = [videos[i] for i in sorted(keep)]
        if max_videos is not None and len(videos) > max_videos:
            videos = [videos[i] for i in sorted(rng.choice(len(videos), max_videos, replace=False))]

        for video in videos:
            frames = by_video[video]
            take = rng.choice(len(frames), size=min(frames_per_video, len(frames)), replace=False)
            for i in sorted(take):
                paths.append(split_dir / folder / frames[i])
                labels.append(CLASS_DIRS[folder])
                folders.append(folder)
                videos_out.append(video)
    return paths, np.array(labels), folders, videos_out


def clip_scores(video_ids: list[str], y_true: np.ndarray, margins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-clip (label, mean frame margin)."""
    labels, scores = [], []
    for video in dict.fromkeys(video_ids):
        idx = [i for i, v in enumerate(video_ids) if v == video]
        labels.append(int(y_true[idx][0]))
        scores.append(float(np.mean(margins[idx])))
    return np.array(labels), np.array(scores)


def best_threshold(labels: np.ndarray, scores: np.ndarray, objective: str = "accuracy") -> float:
    """Clip decision threshold maximising BALANCED accuracy (raw accuracy on an
    imbalanced clip set is maximised by simply calling everything Normal)."""
    candidates = np.unique(np.round(scores, 4))
    grid = np.concatenate([[scores.min() - 1e-3], (candidates[:-1] + candidates[1:]) / 2, [scores.max() + 1e-3]])
    def score(t: float) -> tuple[float, float]:
        pred = (scores > t).astype(int)
        acc = float((pred == labels).mean())
        bal = float(balanced_accuracy_score(labels, pred))
        return (acc, bal) if objective == "accuracy" else (bal, acc)

    return float(max(grid, key=score))


def clip_metrics(video_ids: list[str], y_true: np.ndarray, margins: np.ndarray,
                 threshold: float = 0.0) -> dict:
    """Aggregate frame margins into one decision per source clip — what the live
    node effectively does when it watches a stream instead of a single frame."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    clip_true_arr, clip_score = clip_scores(video_ids, y_true, margins)
    clip_true = clip_true_arr.tolist()
    clip_pred = (clip_score > threshold).astype(int)
    cm = confusion_matrix(clip_true_arr, clip_pred, labels=[0, 1])
    return {
        "n_clips": len(clip_true),
        "threshold": float(threshold),
        "accuracy": float((clip_pred == clip_true_arr).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(clip_true_arr, clip_pred)),
        "precision": float(precision_score(clip_true_arr, clip_pred, zero_division=0)),
        "recall": float(recall_score(clip_true_arr, clip_pred, zero_division=0)),
        "f1_score": float(f1_score(clip_true_arr, clip_pred, zero_division=0)),
        "false_negative_rate": float(cm[1, 0] / max(cm[1].sum(), 1)),
        "roc_auc": float(roc_auc_score(clip_true_arr, clip_score)) if len(set(clip_true)) > 1 else float("nan"),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
    }


def augment(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flip + shift + brightness/contrast jitter, so the CNN keys on appearance
    rather than on the exact pixels of a memorised scene."""
    out = np.empty_like(batch)
    pad = 6
    padded = np.pad(batch, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    ys = rng.integers(0, 2 * pad, len(batch))
    xs = rng.integers(0, 2 * pad, len(batch))
    flips = rng.random(len(batch)) < 0.5
    for i in range(len(batch)):
        crop = padded[i, ys[i]:ys[i] + FRAME_SIZE, xs[i]:xs[i] + FRAME_SIZE]
        out[i] = crop[:, ::-1] if flips[i] else crop
    gain = rng.uniform(0.75, 1.3, (len(batch), 1, 1, 1))
    bias = rng.uniform(-25, 25, (len(batch), 1, 1, 1))
    return np.clip(out * gain + bias, 0, 255).astype(np.uint8)


class CNNClassifier(nn.Module):
    """LightweightCNN backbone + linear head; only the backbone is kept afterwards."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = LightweightCNN(latent_dim=128)
        self.head = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(DROPOUT), nn.Linear(128, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def _to_tensor(images: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float32 in [0, 1]."""
    return torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)


def train_cnn(
    train_img: np.ndarray, train_y: np.ndarray, val_img: np.ndarray, val_y: np.ndarray
) -> tuple[LightweightCNN, float]:
    torch.manual_seed(SEED)
    model = CNNClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = EPOCHS * int(np.ceil(len(train_y) / BATCH_SIZE))
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=steps)
    counts = np.bincount(train_y, minlength=2).astype(np.float64)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32),
                                  label_smoothing=LABEL_SMOOTHING)
    y_train_t = torch.from_numpy(train_y).long()
    rng = np.random.default_rng(SEED)

    best_val, best_state = 0.0, None
    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(len(train_y))
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = _to_tensor(np.ascontiguousarray(augment(train_img[idx], rng)))
            yb = y_train_t[idx]
            # mixup: convex blends of two samples, a strong regulariser against
            # memorising individual scenes
            lam = float(rng.beta(MIXUP_ALPHA, MIXUP_ALPHA)) if MIXUP_ALPHA > 0 else 1.0
            perm = torch.randperm(len(idx))
            logits = model(lam * xb + (1.0 - lam) * xb[perm])
            loss = lam * loss_fn(logits, yb) + (1.0 - lam) * loss_fn(logits, yb[perm])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sched.step()
            total_loss += float(loss) * len(idx)

        val_acc = _cnn_accuracy(model, val_img, val_y)
        print(f"      epoch {epoch + 1}/{EPOCHS} loss={total_loss / len(order):.4f} val_acc={val_acc:.4f}")
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model.backbone.eval(), best_val


@torch.inference_mode()
def _cnn_accuracy(model: nn.Module, images: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    correct = 0
    for start in range(0, len(y), 512):
        logits = model(_to_tensor(images[start:start + 512]))
        correct += int((logits.argmax(1).numpy() == y[start:start + 512]).sum())
    return correct / len(y)


def _embed(cnn: LightweightCNN, images: np.ndarray) -> np.ndarray:
    chunks = [embed_batch(images[s:s + 512].astype(np.float32) / 255.0, cnn)
              for s in range(0, len(images), 512)]
    return np.concatenate(chunks)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))  # training may use all cores
    rng = np.random.default_rng(SEED)
    t_start = time.time()

    print("[1/6] sampling frames (Train/ for fitting, Test/ for evaluation only) ...")
    train_paths, y_cnn_train, _, _ = sample_by_video("Train", FRAMES_PER_VIDEO, rng, video_subset="fit")
    val_paths, y_cnn_val, _, val_videos = sample_by_video("Train", VAL_FRAMES_PER_VIDEO, rng, video_subset="val")
    test_paths, y_test, test_groups, test_videos = sample_by_video("Test", TEST_FRAMES_PER_VIDEO, rng)
    img_train = _load_images(train_paths)
    img_val = _load_images(val_paths)
    img_test = _load_images(test_paths)
    print(f"      cnn_train={img_train.shape[0]} val={img_val.shape[0]} (video-disjoint) test={img_test.shape[0]}")

    print(f"[2/6] training LightweightCNN ({EPOCHS} epochs, CPU) ...")
    cnn, cnn_val_acc = train_cnn(img_train, y_cnn_train, img_val, y_cnn_val)
    torch.save(cnn.state_dict(), ROOT / "cnn_weights.pt")

    print("[3/6] extracting 128-d latents + class-ranked PCA(4) ...")
    fit_idx = rng.permutation(len(y_cnn_train))[:SVM_FIT_PER_LABEL * 2]
    latent_fit = _embed(cnn, img_train[fit_idx])
    y_fit = y_cnn_train[fit_idx]
    latent_val = _embed(cnn, img_val)
    latent_test = _embed(cnn, img_test)
    projector = PCAAngleProjector(n_components=FEATURE_DIM).fit(latent_fit, y_fit)
    print(f"      kept PCA components {projector.components_kept.tolist()} of {projector.pca.n_components_}")

    print("[4/6] tuning angle scale x C on the Train-internal validation split ...")
    t0 = time.time()
    best = (-1.0, None, None)
    for scale in ANGLE_SCALES:
        projector.angle_scale = scale
        E_fit = encode(projector.transform(latent_fit))
        E_val = encode(projector.transform(latent_val))
        K_fit_s, K_val_s = kernel_matrix(E_fit), kernel_matrix(E_val, E_fit)
        for c in C_GRID:
            acc = float(balanced_accuracy_score(y_cnn_val, SVC(kernel="precomputed", C=c, class_weight="balanced").fit(K_fit_s, y_fit).predict(K_val_s)))
            if acc > best[0]:
                best = (acc, scale, c)
        print(f"      angle_scale={scale:.3f} best val_acc so far={best[0]:.4f}")
    val_acc, best_scale, best_c = best
    tune_sec = time.time() - t0

    print(f"[5/6] refitting SVC with angle_scale={best_scale:.3f}, C={best_c} ...")
    projector.angle_scale = best_scale
    X_fit = projector.transform(latent_fit)
    X_val = projector.transform(latent_val)
    X_test = projector.transform(latent_test)
    t0 = time.time()
    E_fit, E_test = encode(X_fit), encode(X_test)
    K_fit, K_test = kernel_matrix(E_fit), kernel_matrix(E_test, E_fit)
    kernel_sec = time.time() - t0
    clf = SVC(kernel="precomputed", C=best_c, class_weight="balanced").fit(K_fit, y_fit)
    print(f"      val_acc={val_acc:.4f} support_vectors={len(clf.support_)} "
          f"(tuning {tune_sec:.0f}s, final kernels {kernel_sec:.1f}s)")

    y_pred = clf.predict(K_test)
    test_margins = clf.decision_function(K_test)

    K_val = kernel_matrix(encode(X_val), E_fit)
    val_labels, val_scores = clip_scores(val_videos, y_cnn_val, clf.decision_function(K_val))
    clip_threshold = best_threshold(val_labels, val_scores, objective="accuracy")
    clip_threshold_balanced = best_threshold(val_labels, val_scores, objective="balanced")
    print(f"      clip threshold {clip_threshold:+.3f} (tuned on {len(val_labels)} held-out val clips)")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    per_class = {
        g: float((y_pred[[i for i, gg in enumerate(test_groups) if gg == g]] ==
                  y_test[[i for i, gg in enumerate(test_groups) if gg == g]]).mean())
        for g in dict.fromkeys(test_groups)
    }
    metrics = {
        "accuracy": float((y_pred == y_test).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "clip_level": clip_metrics(test_videos, y_test, test_margins, clip_threshold),
        "clip_level_recall_tuned": clip_metrics(test_videos, y_test, test_margins, clip_threshold_balanced),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "false_negative_rate": float(cm[1, 0] / max(cm[1].sum(), 1)),
        "train_accuracy": float(clf.score(K_fit, y_fit)),
        "cnn_val_accuracy": cnn_val_acc,
        "svm_val_accuracy": val_acc,
        "per_source_accuracy": per_class,
        "n_cnn_train": int(len(y_cnn_train)),
        "n_train_videos": len(set(_video_id(p.name) for p in train_paths)),
        "n_test_videos": len(set(test_videos)),
        "n_svm_fit": int(len(y_fit)),
        "n_test": int(len(y_test)),
        "n_support_vectors": int(len(clf.support_)),
        "best_C": best_c,
        "best_angle_scale": float(best_scale),
        "pca_components_kept": projector.components_kept.tolist(),
        "kernel_compute_sec": kernel_sec,
        "total_runtime_sec": time.time() - t_start,
        "test_split": "dataset-3/Test (never used for fitting)",
        "validation": "Train/ clips held out by video id",
    }
    print("      frame-level: " + json.dumps({k: v for k, v in metrics.items() if k not in ("per_source_accuracy", "clip_level")}))
    print("      CLIP-level:  " + json.dumps(metrics["clip_level"]))
    print("      per-source accuracy: " + json.dumps(per_class))

    print("[6/6] serializing artifact + reports ...")
    joblib.dump({
        "svm_classifier": clf,
        "pca_transformer": projector,
        "support_vectors": X_fit[clf.support_],
        "support_vector_labels": y_fit[clf.support_],
        "cnn_state_dict": cnn.state_dict(),
        "metadata": {
            "feature_map": {"type": "ZZFeatureMap", "feature_dimension": FEATURE_DIM,
                            "reps": 2, "entanglement": "linear"},
            "svm_hyperparams": {"kernel": "precomputed", "C": best_c},
            "angle_scale": float(best_scale),
            "clip_threshold": clip_threshold,
            "clip_threshold_recall_tuned": clip_threshold_balanced,
            "cnn_latent_dim": 128,
            "frame_size": FRAME_SIZE,
            "classes": {"0": "Normal", "1": "Anomaly"},
            "train_accuracy": metrics["train_accuracy"],
            "test_accuracy": metrics["accuracy"],
            "clip_test_accuracy": metrics["clip_level"]["accuracy"],
            "timestamp": time.time(),
        },
    }, MODELS_DIR / "qsvm_model.joblib")

    with open(REPORTS_DIR / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"]).plot(cmap="Blues")
    import matplotlib.pyplot as plt
    plt.title(f"QSVM on dataset-3/Test — acc {metrics['accuracy']:.1%}")
    plt.savefig(REPORTS_DIR / "figures" / "cm_qsvm_trained.png", dpi=150, bbox_inches="tight")

    print(f"model saved -> {MODELS_DIR / 'qsvm_model.joblib'}")
    print(f"FRAME-LEVEL TEST ACCURACY: {metrics['accuracy']:.4f}")
    print(f"CLIP-LEVEL  TEST ACCURACY: {metrics['clip_level']['accuracy']:.4f} "
          f"(balanced {metrics['clip_level']['balanced_accuracy']:.4f}, "
          f"AUC {metrics['clip_level']['roc_auc']:.4f}, n={metrics['clip_level']['n_clips']} clips)")
    print(f"runtime {metrics['total_runtime_sec']:.0f}s")


if __name__ == "__main__":
    main()
