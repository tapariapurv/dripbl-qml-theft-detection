"""Cross-dataset experiments: which representation detects theft on EVERY dataset?

The deployed model scores 0.871 on UCF-Crime (dataset-3) but chance on retail
footage (dataset-1) and a second camera (dataset-2): it learned one dataset's look.
This harness trains on the training portions of all three and scores the fixed,
untouched hold-outs from src/cross_data.py, one config per run:

  data    dataset-3 Train (fit/val by video), dataset-1 train split (fit/val by video),
          dataset-2 minus its 2 hold-out clips; sampled per step by dataset share
  model   backbone -> 128-d latent -> [head for training]
          then the project's quantum head: PLS/PCA(4) -> angles -> ZZFeatureMap
          fidelity kernel -> SVC, exactly as deployed
  select  epoch, projector, angle scale, C and the clip threshold on VALIDATION clips
          only, maximising the mean of per-dataset scores (so the 170 dataset-3 clips
          cannot drown out the 22 retail ones)
  report  hold-out clip accuracy/AUC per dataset -> reports/xd_experiments.jsonl

    .venv/bin/python3 -u src/xd_experiments.py tiny_rgb
    .venv/bin/python3 -u src/xd_experiments.py --list
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cross_data as C  # noqa: E402
import experiments as E  # noqa: E402
import train_qsvm as T  # noqa: E402
from edge_pipeline import LightweightCNN  # noqa: E402
from pca_projector import PCAAngleProjector  # noqa: E402
from quantum_kernel import FEATURE_DIM, encode, kernel_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "reports" / "xd_experiments.jsonl"
CANDIDATES = ROOT / "models" / "candidates"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SOURCES = ("d3", "d1", "d2")
SELECT = ("d3", "d1")   # dataset-2 has 2 val clips: AUC is 0 or 1, pure noise for selection


@dataclasses.dataclass
class Cfg:
    name: str
    backbone: str = "tiny"            # tiny | resnet18 | mobilenet
    motion: bool = False              # add the 0.33 s motion-difference channels
    input_size: int = 64              # pretrained backbones: frames upsampled to this
    epochs: int = 10
    steps_per_epoch: int = 300
    batch: int = 128
    lr: float = 1.5e-3
    weight_decay: float = 5e-4
    dropout: float = 0.5
    share: tuple[float, float, float] = (0.5, 0.4, 0.1)   # d3, d1, d2 share of each batch
    d3_frames: int = 20
    fit_per_label: int = 1500         # QSVM training frames per class (Gram matrix is O(n^2))
    seed: int = 42


CONFIGS = {c.name: c for c in [
    Cfg("tiny_rgb"),
    Cfg("tiny_motion", motion=True),
    Cfg("resnet18_rgb", backbone="resnet18", input_size=112, lr=3e-4, epochs=8),
    Cfg("resnet18_motion", backbone="resnet18", motion=True, input_size=112, lr=3e-4, epochs=8),
    Cfg("mobilenet_rgb", backbone="mobilenet", input_size=112, lr=5e-4, epochs=8),
    Cfg("mobilenet_motion", backbone="mobilenet", motion=True, input_size=112, lr=5e-4, epochs=8),
]}


# ----------------------------------------------------------------------------- data

def _split_videos(vids: np.ndarray, y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Boolean mask of validation rows: `frac` of each class's videos, by video."""
    val = np.zeros(len(vids), bool)
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        names = sorted(set(vids[y == label].tolist()))
        pick = {names[i] for i in rng.permutation(len(names))[:max(1, round(len(names) * frac))]}
        val |= np.isin(vids, list(pick))
    return val


def _d3(part: str, frames: int, motion: bool) -> dict[str, np.ndarray]:
    cache = ROOT / "reports" / f"d3_{part}_{frames}{'_m' if motion else ''}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    paths, y, _, vids = T.sample_by_video("Train", frames, np.random.default_rng(T.SEED), video_subset=part)
    x = E._load(paths, motion)
    out = {"x": x, "y": np.asarray(y), "vid": np.array([f"d3:{v}" for v in vids])}
    np.savez(cache, **out)
    return out


def load_train(cfg: Cfg) -> tuple[dict, dict]:
    parts = {"fit": [], "val": []}

    def add(part: str, x, y, vid, src: str) -> None:
        parts[part].append({"x": x, "y": y, "vid": vid, "src": np.full(len(y), src)})

    for part, frames in (("fit", cfg.d3_frames), ("val", 10)):
        d = _d3(part, frames, cfg.motion)
        add(part, d["x"], d["y"], d["vid"], "d3")
    for src, d in (("d1", C.dataset1_part("train")), ("d2", C.dataset2_part("train"))):
        x = np.concatenate([d["rgb"], d["motion"]], -1) if cfg.motion else d["rgb"]
        val = _split_videos(d["vid"], d["y"], 0.15 if src == "d1" else 0.34, cfg.seed)
        add("fit", x[~val], d["y"][~val], d["vid"][~val], src)
        add("val", x[val], d["y"][val], d["vid"][val], src)
    cat = lambda rows: {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}  # noqa: E731
    return cat(parts["fit"]), cat(parts["val"])


def load_holdouts(motion: bool) -> dict[str, dict]:
    sets = {"d3": C.load_dataset3_test(60), "d1": C.dataset1_part("holdout"), "d2": C.dataset2_part("holdout")}
    for d in sets.values():
        d["x"] = np.concatenate([d["rgb"], d["motion"]], -1) if motion else d["rgb"]
    return sets


# ---------------------------------------------------------------------------- model

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Backbone(nn.Module):
    """uint8-scaled (B, C, 64, 64) in [0, 1] -> (B, 128) latent."""

    def __init__(self, kind: str, channels: int, input_size: int) -> None:
        super().__init__()
        self.kind, self.channels, self.input_size = kind, channels, input_size
        if kind == "tiny":
            self.net = LightweightCNN(128, channels)
            return
        from torchvision import models
        if kind == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            stem, feat_dim = m.conv1, 512
            m.fc = nn.Identity()
        else:
            m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            stem, feat_dim = m.features[0][0], 576
            m.classifier = nn.Identity()
        if channels != 3:  # motion channels start from the RGB filters, scaled down
            new = nn.Conv2d(channels, stem.out_channels, stem.kernel_size, stem.stride, stem.padding, bias=False)
            with torch.no_grad():
                new.weight[:, :3] = stem.weight
                new.weight[:, 3:] = stem.weight * 0.5
            if kind == "resnet18":
                m.conv1 = new
            else:
                m.features[0][0] = new
        self.net = nn.Sequential(m, nn.Linear(feat_dim, 128))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "tiny":
            return self.net(x)
        rgb = (x[:, :3] - _MEAN.to(x.device)) / _STD.to(x.device)
        x = torch.cat([rgb, (x[:, 3:] - 0.5) / 0.25], 1) if self.channels > 3 else rgb
        if self.input_size != x.shape[-1]:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return self.net(x)


class Net(nn.Module):
    def __init__(self, cfg: Cfg, channels: int) -> None:
        super().__init__()
        self.backbone = Backbone(cfg.backbone, channels, cfg.input_size)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(128, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float().div_(255.0)


@torch.inference_mode()
def forward(model: nn.Module, x: np.ndarray, latent: bool) -> np.ndarray:
    model.eval()
    net = model.backbone if latent else model
    return np.concatenate([net(_t(x[s:s + 512]).to(DEVICE)).float().cpu().numpy()
                           for s in range(0, len(x), 512)])


def per_source_clip_auc(scores: np.ndarray, d: dict) -> dict[str, float]:
    out = {}
    for src in SOURCES:
        m = d["src"] == src
        if m.sum() == 0:
            continue
        lab, sc = C.clip_scores(scores[m], d["vid"][m], d["y"][m])
        if len(set(lab.tolist())) > 1:
            out[src] = float(roc_auc_score(lab, sc))
    return out


def train(cfg: Cfg, fit: dict, val: dict) -> tuple[nn.Module, dict]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = Net(cfg, fit["x"].shape[-1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * cfg.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total)
    # sampling weights: each dataset gets its share of every batch, classes balanced within it
    w = np.zeros(len(fit["y"]))
    for s, share in zip(SOURCES, cfg.share):
        for label in (0, 1):
            m = (fit["src"] == s) & (fit["y"] == label)
            if m.any():
                w[m] = share / 2 / m.sum()
    w /= w.sum()
    y_t = torch.from_numpy(fit["y"]).long()
    best, best_state, hist = -1.0, None, []
    for epoch in range(cfg.epochs):
        model.train()
        t0, loss_sum = time.time(), 0.0
        for _ in range(cfg.steps_per_epoch):
            idx = rng.choice(len(w), cfg.batch, p=w)
            xb = _t(E._augment(fit["x"][idx], rng)).to(DEVICE)
            loss = F.cross_entropy(model(xb), y_t[idx].to(DEVICE))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += float(loss)
        logits = forward(model, val["x"], latent=False)
        aucs = per_source_clip_auc(logits[:, 1] - logits[:, 0], val)
        score = float(np.mean([aucs[s] for s in SELECT]))
        hist.append({"epoch": epoch + 1, "loss": loss_sum / cfg.steps_per_epoch, "val_clip_auc": aucs})
        mark = ""
        if score >= best:
            best, best_state, mark = score, {k: v.detach().clone() for k, v in model.state_dict().items()}, " <- best"
        print(f"    epoch {epoch + 1}/{cfg.epochs} {time.time() - t0:4.0f}s loss={loss_sum / cfg.steps_per_epoch:.4f} "
              f"val clip AUC " + " ".join(f"{k}={v:.3f}" for k, v in aucs.items()) + mark, flush=True)
    model.load_state_dict(best_state)
    return model, {"history": hist, "best_val_mean_auc": best}


# ------------------------------------------------------------------- quantum head

def balanced_threshold(val_scores: np.ndarray, val: dict) -> tuple[float, dict]:
    """Clip threshold maximising the MEAN of per-dataset clip accuracies (val only)."""
    per = {}
    for src in SELECT:
        m = val["src"] == src
        if m.any():
            per[src] = C.clip_scores(val_scores[m], val["vid"][m], val["y"][m])
    allsc = np.concatenate([sc for _, sc in per.values()])
    grid = np.quantile(allsc, np.linspace(0, 1, 401))
    accs = np.array([np.mean([((sc > t).astype(int) == lab).mean() for lab, sc in per.values()]) for t in grid])
    good = np.flatnonzero(accs >= accs.max() - 0.005)
    # midpoint of the widest near-optimal run: robust to small score shifts
    runs = np.split(good, np.flatnonzero(np.diff(good) != 1) + 1)
    run = max(runs, key=lambda r: grid[r[-1]] - grid[r[0]])
    t = float((grid[run[0]] + grid[run[-1]]) / 2)
    return t, {s: float(((sc > t).astype(int) == lab).mean()) for s, (lab, sc) in per.items()}


def fit_quantum_head(cfg: Cfg, model: nn.Module, fit: dict, val: dict) -> dict:
    rng = np.random.default_rng(cfg.seed)
    pick = []
    for s, share in zip(SOURCES, cfg.share):  # QSVM fit frames follow the same dataset shares
        for label in (0, 1):
            idx = np.flatnonzero((fit["src"] == s) & (fit["y"] == label))
            n = min(len(idx), max(20, int(cfg.fit_per_label * share)))
            pick += rng.choice(idx, n, replace=False).tolist()
    pick = np.array(pick)
    lat_fit, y_fit = forward(model, fit["x"][pick], latent=True), fit["y"][pick]
    lat_val = forward(model, val["x"], latent=True)
    best = (-1.0, None)
    for mode in ("pls", "pca"):
        proj = PCAAngleProjector(n_components=FEATURE_DIM, mode=mode).fit(lat_fit, y_fit)
        for scale in T.ANGLE_SCALES:
            proj.angle_scale = scale
            e_fit = encode(proj.transform(lat_fit))
            k_fit, k_val = kernel_matrix(e_fit), kernel_matrix(encode(proj.transform(lat_val)), e_fit)
            for c in T.C_GRID:
                svc = SVC(kernel="precomputed", C=c, class_weight="balanced").fit(k_fit, y_fit)
                aucs = per_source_clip_auc(svc.decision_function(k_val), val)
                score = float(np.mean([aucs[s] for s in SELECT]))
                if score > best[0]:
                    best = (score, (mode, scale, c, aucs))
    score, (mode, scale, c, aucs) = best
    proj = PCAAngleProjector(n_components=FEATURE_DIM, mode=mode).fit(lat_fit, y_fit)
    proj.angle_scale = scale
    ang_fit = proj.transform(lat_fit)
    e_fit = encode(ang_fit)
    clf = SVC(kernel="precomputed", C=c, class_weight="balanced").fit(kernel_matrix(e_fit), y_fit)
    val_m = clf.decision_function(kernel_matrix(encode(proj.transform(lat_val)), e_fit))
    thr, val_acc = balanced_threshold(val_m, val)
    print(f"    quantum head: {mode} scale={scale} C={c} support={len(clf.support_)} "
          f"val clip AUC {aucs} threshold={thr:+.3f} val acc {val_acc}", flush=True)
    return {"projector": proj, "clf": clf, "support": ang_fit[clf.support_], "states": e_fit[clf.support_],
            "mode": mode, "scale": scale, "C": c, "threshold": thr, "val_auc": aucs, "val_acc": val_acc}


def quantum_margins(model: nn.Module, head: dict, x: np.ndarray) -> np.ndarray:
    ang = head["projector"].transform(forward(model, x, latent=True))
    K = np.zeros((len(ang), int(head["clf"].shape_fit_[0])))
    K[:, head["clf"].support_] = kernel_matrix(encode(ang), head["states"])
    return head["clf"].decision_function(K)


def run(cfg: Cfg) -> dict:
    t0 = time.time()
    print(f"\n=== {cfg.name} on {DEVICE} ===", flush=True)
    fit, val = load_train(cfg)
    print(f"  fit {len(fit['y'])} frames " + " ".join(f"{s}={int((fit['src'] == s).sum())}" for s in SOURCES)
          + f" | val {len(val['y'])} frames", flush=True)
    model, info = train(cfg, fit, val)
    head = fit_quantum_head(cfg, model, fit, val)
    hold = load_holdouts(cfg.motion)
    results, head_only = {}, {}
    for s, d in hold.items():
        results[s] = C.report(*C.clip_scores(quantum_margins(model, head, d["x"]), d["vid"], d["y"]), head["threshold"])
        logits = forward(model, d["x"], latent=False)
        lab, sc = C.clip_scores(logits[:, 1] - logits[:, 0], d["vid"], d["y"])
        head_only[s] = float(roc_auc_score(lab, sc))
    mean_acc = float(np.mean([results[s]["accuracy"] for s in ("d3", "d1")]))
    row = {"config": dataclasses.asdict(cfg), "holdout": results, "cnn_head_holdout_auc": head_only,
           "mean_d3_d1_acc": mean_acc, "quantum_head": {k: head[k] for k in ("mode", "scale", "C", "threshold",
                                                                             "val_auc", "val_acc")},
           "train": info, "minutes": round((time.time() - t0) / 60, 1), "ts": time.time()}
    with LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    joblib.dump({"cfg": dataclasses.asdict(cfg), "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                 "head": {k: head[k] for k in ("projector", "clf", "support", "mode", "scale", "C", "threshold")},
                 "results": results}, CANDIDATES / f"{cfg.name}.joblib")
    print(f"  HOLDOUT " + " | ".join(f"{s}: acc={r['accuracy']:.3f} auc={r['auc']:.3f} fp={r['fp']} fn={r['fn']}"
                                    for s, r in results.items())
          + f"\n  cnn-head AUC {head_only}  ({row['minutes']} min)", flush=True)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--steps", type=int)
    args = ap.parse_args()
    if args.list or not args.names:
        print("\n".join(CONFIGS))
        sys.exit()
    for name in args.names:
        cfg = CONFIGS[name]
        if args.epochs:
            cfg = dataclasses.replace(cfg, epochs=args.epochs)
        if args.steps:
            cfg = dataclasses.replace(cfg, steps_per_epoch=args.steps)
        run(cfg)
