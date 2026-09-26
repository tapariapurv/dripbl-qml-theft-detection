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
import cv2
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cross_data as C  # noqa: E402
import experiments as E  # noqa: E402
import train_qsvm as T  # noqa: E402
from backbones import Backbone  # noqa: E402
from pca_projector import PCAAngleProjector  # noqa: E402
from quantum_kernel import FEATURE_DIM, encode, kernel_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "reports" / "xd_experiments.jsonl"
CANDIDATES = ROOT / "models" / "candidates"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SOURCES = ("d3", "d1", "d2")
SELECT = ("d3", "d1")   # dataset-2 has 2 val clips: AUC is 0 or 1, pure noise for selection


def sources_in(d: dict) -> list[str]:
    """Datasets present in `d`, built-ins first in their fixed order, then your own sites
    (x:<name>) sorted. The order matters: fold assignment draws random numbers per dataset."""
    present = set(d["src"].tolist())
    return [s for s in SOURCES if s in present] + sorted(present - set(SOURCES))


def select_in(d: dict) -> list[str]:
    """Datasets used to choose epochs/thresholds: the reliable built-ins plus your own sites."""
    return [s for s in sources_in(d) if s in SELECT or s.startswith("x:")]


def mix(cfg: "Cfg") -> list[tuple[str, float]]:
    """(dataset, share of each batch). Your own sites take `extra_share`, split equally."""
    base = [(s, sh) for s, sh in zip(SOURCES, cfg.share)]
    if not cfg.extra:
        return base
    scale = 1.0 - cfg.extra_share
    return [(s, sh * scale) for s, sh in base] + [(f"x:{n}", cfg.extra_share / len(cfg.extra)) for n in cfg.extra]


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
    size: int = 64                    # frame resolution; dataset-3 (64 px only) is upsampled to it
    sources: tuple[str, ...] = ("d3", "d1", "d2")   # datasets to train on (single-dataset studies)
    init_from: str = ""               # start the encoder from this model's weights (transfer learning)
    extra: tuple[str, ...] = ()       # your own sites: training-data/<name>/{normal,theft}/*.mp4
    extra_share: float = 0.5          # share of each batch drawn from `extra`


CONFIGS = {c.name: c for c in [
    Cfg("tiny_rgb"),
    Cfg("tiny_motion", motion=True),
    Cfg("resnet18_rgb", backbone="resnet18", input_size=112, lr=3e-4, epochs=8),
    Cfg("resnet18_motion", backbone="resnet18", motion=True, input_size=112, lr=3e-4, epochs=8),
    Cfg("mobilenet_rgb", backbone="mobilenet", input_size=112, lr=5e-4, epochs=8),
    Cfg("mobilenet_motion", backbone="mobilenet", motion=True, input_size=112, lr=5e-4, epochs=8),
    # resolution study on retail alone: identical except frame size
    Cfg("d1_r18m_64", backbone="resnet18", motion=True, input_size=128, lr=3e-4, epochs=6, steps_per_epoch=200,
        sources=("d1",), share=(0.0, 1.0, 0.0)),
    Cfg("d1_r18m_128", backbone="resnet18", motion=True, input_size=128, lr=3e-4, epochs=6, steps_per_epoch=200,
        sources=("d1",), share=(0.0, 1.0, 0.0), size=128),
    # per-domain experts (joint training let dataset-3's 20k frames pull the encoder away from retail)
    Cfg("retail_r18m", backbone="resnet18", motion=True, input_size=128, lr=3e-4, epochs=4, steps_per_epoch=200,
        sources=("d1", "d2"), share=(0.0, 0.85, 0.15)),
    Cfg("cctv_r18", backbone="resnet18", input_size=112, lr=3e-4, epochs=4, steps_per_epoch=300,
        sources=("d3",), share=(1.0, 0.0, 0.0)),
    # a cheap, differently-biased partner for ensembling with retail_r18m (same seed -> same folds)
    Cfg("retail_tiny_m", motion=True, epochs=10, steps_per_epoch=200, sources=("d1", "d2"), share=(0.0, 0.85, 0.15)),
    # transfer: learn "crime" on UCF-Crime first (CCTV experts, no retail data), then specialise on retail
    Cfg("retail_r18m_ft", backbone="resnet18", motion=True, input_size=128, lr=1.5e-4, epochs=4, steps_per_epoch=200,
        sources=("d1", "d2"), share=(0.0, 0.85, 0.15), init_from="models/candidates/cctv_r18.joblib"),
    Cfg("retail_tiny_m_ft", motion=True, epochs=10, steps_per_epoch=200, lr=8e-4, sources=("d1", "d2"),
        share=(0.0, 0.85, 0.15), init_from="models/candidates/ucf_tiny_original.joblib"),
    # zero-shot camera test: never sees dataset-2, then scored on all 8 of its videos
    Cfg("retail_tiny_m_ft_d1only", motion=True, epochs=10, steps_per_epoch=200, lr=8e-4, sources=("d1",),
        share=(0.0, 1.0, 0.0), init_from="models/candidates/ucf_tiny_original.joblib"),
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

    if "d3" in cfg.sources:
        for part, frames in (("fit", cfg.d3_frames), ("val", 10)):
            d = _d3(part, frames, cfg.motion)
            add(part, _resize(d["x"], cfg.size), d["y"], d["vid"], "d3")
    for src, d in (("d1", C.dataset1_part("train", cfg.size)), ("d2", C.dataset2_part("train", cfg.size))):
        if src not in cfg.sources:
            continue
        x = np.concatenate([d["rgb"], d["motion"]], -1) if cfg.motion else d["rgb"]
        val = _split_videos(d["vid"], d["y"], 0.15 if src == "d1" else 0.34, cfg.seed)
        add("fit", x[~val], d["y"][~val], d["vid"][~val], src)
        add("val", x[val], d["y"][val], d["vid"][val], src)
    for name in cfg.extra:  # your own sites: their train split (the hold-out is never touched here)
        d = C.custom_part(name, "train", cfg.size)
        x = np.concatenate([d["rgb"], d["motion"]], -1) if cfg.motion else d["rgb"]
        val = _split_videos(d["vid"], d["y"], 0.2, cfg.seed)
        add("fit", x[~val], d["y"][~val], d["vid"][~val], f"x:{name}")
        add("val", x[val], d["y"][val], d["vid"][val], f"x:{name}")
    cat = lambda rows: {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}  # noqa: E731
    return cat(parts["fit"]), cat(parts["val"])


def _resize(x: np.ndarray, size: int) -> np.ndarray:
    """Upsample 64 px frames (dataset-3) to the experiment's resolution; no-op at 64."""
    if x.shape[1] == size:
        return x
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR) for f in x])


def load_holdouts(motion: bool, size: int = 64) -> dict[str, dict]:
    sets = {"d3": C.load_dataset3_test(60), "d1": C.dataset1_part("holdout", size),
            "d2": C.dataset2_part("holdout", size)}
    for d in sets.values():
        d["x"] = _resize(np.concatenate([d["rgb"], d["motion"]], -1) if motion else d["rgb"], size)
    return sets


# ---------------------------------------------------------------------------- model

class Net(nn.Module):
    def __init__(self, cfg: Cfg, channels: int) -> None:
        super().__init__()
        self.backbone = Backbone(cfg.backbone, channels, cfg.input_size, pretrained=True)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(128, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def augment(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random shift + mirror on all channels, brightness/contrast on RGB only (scaling the
    motion channels would fake motion that never happened). Any frame size."""
    n, size = len(batch), batch.shape[1]
    pad = max(2, size // 10)
    padded = np.pad(batch, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    ys, xs, flips = rng.integers(0, 2 * pad, n), rng.integers(0, 2 * pad, n), rng.random(n) < 0.5
    out = np.empty_like(batch)
    for i in range(n):
        crop = padded[i, ys[i]:ys[i] + size, xs[i]:xs[i] + size]
        out[i] = crop[:, ::-1] if flips[i] else crop
    gain, bias = rng.uniform(0.75, 1.3, (n, 1, 1, 1)), rng.uniform(-25, 25, (n, 1, 1, 1))
    out[..., :3] = np.clip(out[..., :3] * gain + bias, 0, 255).astype(np.uint8)
    return out


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
    for src in sources_in(d):
        m = d["src"] == src
        if m.sum() == 0:
            continue
        lab, sc = C.clip_scores(scores[m], d["vid"][m], d["y"][m])
        if len(set(lab.tolist())) > 1:
            out[src] = float(roc_auc_score(lab, sc))
    return out


def init_backbone(model: nn.Module, path: str) -> int:
    """Copy encoder weights from a trained artifact (member 0) into `model.backbone`.
    A 3-channel stem feeding a 6-channel model gets its RGB filters copied to the motion
    channels at half strength (same rule as ImageNet init). Returns tensors copied."""
    art = joblib.load(ROOT / path)
    member = (art.get("ensemble") or [art])[0]
    src = member.get("backbone_state") or {f"net.{k}": v for k, v in member["cnn_state_dict"].items()}
    dst = model.backbone.state_dict()
    copied = 0
    for k, v in dst.items():
        w = src.get(k)
        if w is None:
            continue
        if w.shape == v.shape:
            dst[k] = w.clone()
        elif w.dim() == 4 and w.shape[1] == 3 and v.shape[1] == 6 and w.shape[0] == v.shape[0]:
            dst[k] = torch.cat([w, w * 0.5], 1)
        else:
            continue
        copied += 1
    model.backbone.load_state_dict(dst)
    return copied


def train(cfg: Cfg, fit: dict, val: dict) -> tuple[nn.Module, dict]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = Net(cfg, fit["x"].shape[-1])
    if cfg.init_from:
        print(f"    encoder initialised from {cfg.init_from}: {init_backbone(model, cfg.init_from)} tensors", flush=True)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * cfg.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total)
    # sampling weights: each dataset gets its share of every batch, classes balanced within it
    w = np.zeros(len(fit["y"]))
    for s, share in mix(cfg):
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
            xb = _t(augment(fit["x"][idx], rng)).to(DEVICE)
            loss = F.cross_entropy(model(xb), y_t[idx].to(DEVICE))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += float(loss)
        logits = forward(model, val["x"], latent=False)
        aucs = per_source_clip_auc(logits[:, 1] - logits[:, 0], val)
        score = float(np.mean([aucs[s] for s in select_in(val) if s in aucs]))
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
    for src in select_in(val):
        m = val["src"] == src
        if m.sum() and len(set(val["y"][m].tolist())) > 1:
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
    for s, share in mix(cfg):  # QSVM fit frames follow the same dataset shares
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
                score = float(np.mean([aucs[s] for s in select_in(val) if s in aucs]))
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
    print(f"  fit {len(fit['y'])} frames " + " ".join(f"{s}={int((fit['src'] == s).sum())}" for s in sources_in(fit))
          + f" | val {len(val['y'])} frames", flush=True)
    model, info = train(cfg, fit, val)
    head = fit_quantum_head(cfg, model, fit, val)
    hold = load_holdouts(cfg.motion, cfg.size)
    results, head_only = {}, {}
    for s, d in hold.items():
        results[s] = C.report(*C.clip_scores(quantum_margins(model, head, d["x"]), d["vid"], d["y"]), head["threshold"])
        logits = forward(model, d["x"], latent=False)
        lab, sc = C.clip_scores(logits[:, 1] - logits[:, 0], d["vid"], d["y"])
        head_only[s] = float(roc_auc_score(lab, sc))
    mean_acc = float(np.mean([results[s]["accuracy"] for s in ("d3", "d1") if s in cfg.sources] or [0.0]))
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
