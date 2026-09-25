"""Probe: does a Kinetics-400 video model (R3D-18, frozen) give a better representation
for theft than per-frame CNNs? Features -> the project's quantum head, same splits.

Each clip window is 16 frames at 112x112:
  dataset-1/2  video decoded at 30 fps, every 2nd frame  (~1.07 s of motion)
  dataset-3    16 consecutive frame dumps, 0.33 s apart   (~5.3 s, upsampled from 64 px)
A video's score is the mean over its windows (clip-level, as everywhere else).

    .venv/bin/python3 -u src/video_probe.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cross_data as C  # noqa: E402
import train_qsvm as T  # noqa: E402
import xd_experiments as X  # noqa: E402
from pca_projector import PCAAngleProjector  # noqa: E402
from quantum_kernel import FEATURE_DIM, encode, kernel_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRAMES, SIZE = 16, 112
MEAN = np.array([0.43216, 0.394666, 0.37645], np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], np.float32)
DEVICE = X.DEVICE


def _windows(n_frames: int, span: int, k: int) -> list[int]:
    if n_frames < span:
        return [0]
    return sorted(set(np.linspace(0, n_frames - span, k).astype(int).tolist()))


def video_clips(path: Path, k: int, stride: int = 2) -> np.ndarray:
    """(k, 16, 112, 112, 3) uint8 RGB windows spread over the video."""
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    cap.release()
    span = FRAMES * stride
    out = []
    for s in _windows(len(frames), span, k):
        idx = [min(s + i * stride, len(frames) - 1) for i in range(FRAMES)]
        out.append(np.stack([frames[i] for i in idx]))
    return np.stack(out)


_LISTING: dict[Path, dict[str, list[str]]] = {}


def _frames_by_video(d: Path) -> dict[str, list[str]]:
    """One directory listing per class folder (they hold hundreds of thousands of files)."""
    if d not in _LISTING:
        by: dict[str, list[str]] = {}
        for n in os.listdir(d):
            by.setdefault(n.rsplit("_", 1)[0], []).append(n)
        for v in by.values():
            v.sort(key=lambda n: int(n.rsplit("_", 1)[1].split(".")[0]))
        _LISTING[d] = by
    return _LISTING[d]


def d3_clips(split: str, video: str, folder: str, k: int) -> np.ndarray:
    d = T.DATASET_DIR / split / folder
    names = _frames_by_video(d)[video]
    windows = [[min(s + i, len(names) - 1) for i in range(FRAMES)] for s in _windows(len(names), FRAMES, k)]
    needed = sorted({i for w in windows for i in w})  # decode only the frames the windows use
    imgs = {i: cv2.cvtColor(cv2.resize(cv2.imread(str(d / names[i])), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR),
                            cv2.COLOR_BGR2RGB) for i in needed}
    return np.stack([np.stack([imgs[i] for i in w]) for w in windows])


@torch.inference_mode()
def embed(model, clips: np.ndarray) -> np.ndarray:
    x = (clips.astype(np.float32) / 255.0 - MEAN) / STD                 # (N, T, H, W, 3)
    t = torch.from_numpy(x).permute(0, 4, 1, 2, 3).contiguous()        # (N, 3, T, H, W)
    return np.concatenate([model(t[i:i + 16].to(DEVICE)).float().cpu().numpy() for i in range(0, len(t), 16)])


def build_model():
    from torchvision.models.video import R3D_18_Weights, r3d_18
    m = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    m.fc = torch.nn.Identity()
    return m.eval().to(DEVICE)


def features() -> dict[str, np.ndarray]:
    cache = ROOT / "reports" / "video_feats_r3d18.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    model, rows, t0 = build_model(), {"f": [], "vid": [], "y": [], "src": [], "part": []}, time.time()

    def add(clips, vid, y, src, part):
        rows["f"].append(embed(model, clips))
        n = len(clips)
        rows["vid"] += [vid] * n
        rows["y"] += [y] * n
        rows["src"] += [src] * n
        rows["part"] += [part] * n

    val3 = set(T.sample_by_video("Train", 1, np.random.default_rng(0), video_subset="val")[3])
    for split in ("Train", "Test"):
        for folder, label in T.CLASS_DIRS.items():
            videos = sorted(_frames_by_video(T.DATASET_DIR / split / folder))
            for v in videos:
                part = "test" if split == "Test" else ("val" if v in val3 else "fit")
                add(d3_clips(split, v, folder, 6 if split == "Test" else 4), f"d3:{v}", label, "d3", part)
        print(f"  d3 {split} done ({time.time() - t0:.0f}s)", flush=True)
    split1 = C.dataset1_split()
    rng = np.random.default_rng(42)
    d1_val = set()
    for label, folder in ((0, "normal"), (1, "shoplifting")):
        names = sorted(f"d1:{p.stem}" for p in (C.DATA / "dataset-1" / folder).glob("*.mp4") if f"d1:{p.stem}" in split1["train"])
        d1_val |= {names[i] for i in rng.permutation(len(names))[:round(len(names) * 0.15)]}
    for label, folder in ((0, "normal"), (1, "shoplifting")):
        for p in sorted((C.DATA / "dataset-1" / folder).glob("*.mp4")):
            vid = f"d1:{p.stem}"
            part = "test" if vid in split1["holdout"] else ("val" if vid in d1_val else "fit")
            add(video_clips(p, 6), vid, label, "d1", part)
    print(f"  d1 done ({time.time() - t0:.0f}s)", flush=True)
    for p in sorted((C.DATA / "dataset-2" / "videos").glob("*.mp4")):
        vid = f"d2:{p.stem}"
        add(video_clips(p, 4, stride=1), vid, 0 if p.stem.startswith("not_") else 1, "d2",
            "test" if p.stem in C.D2_HOLDOUT else "fit")
    out = {"f": np.concatenate(rows["f"]), **{k: np.array(v) for k, v in rows.items() if k != "f"}}
    np.savez(cache, **out)
    print(f"  features {out['f'].shape} ({time.time() - t0:.0f}s)", flush=True)
    return out


def main() -> None:
    d = features()
    part = lambda p: {k: v[d["part"] == p] for k, v in d.items()}  # noqa: E731
    fit, val, test = part("fit"), part("val"), part("test")
    sc = StandardScaler().fit(fit["f"])
    Xf, Xv, Xt = sc.transform(fit["f"]), sc.transform(val["f"]), sc.transform(test["f"])

    def evaluate(name: str, val_m: np.ndarray, test_m: np.ndarray) -> None:
        import experiments as E
        glob, _ = X.balanced_threshold(val_m, val)
        line = []
        for s in ("d3", "d1", "d2"):
            m = test["src"] == s
            lab, scs = C.clip_scores(test_m[m], test["vid"][m], test["y"][m])
            if s in X.SELECT:
                vm = val["src"] == s
                t_site = E.pick_threshold(*C.clip_scores(val_m[vm], val["vid"][vm], val["y"][vm]))[0]
            else:
                t_site = glob
            r, rs = C.report(lab, scs, glob), C.report(lab, scs, t_site)
            line.append(f"{s}: auc={r['auc']:.3f} acc global={r['accuracy']:.3f} site={rs['accuracy']:.3f}")
        print(f"{name:26s} " + " | ".join(line), flush=True)

    # sample weights: equalise datasets (d3 has ~8x more clips than d1)
    w = np.zeros(len(fit["y"]))
    for s, share in zip(X.SOURCES, (0.5, 0.4, 0.1)):
        for label in (0, 1):
            m = (fit["src"] == s) & (fit["y"] == label)
            w[m] = share / 2 / max(m.sum(), 1)
    for c in (0.01, 0.1):
        lr = LogisticRegression(C=c, max_iter=3000).fit(Xf, fit["y"], sample_weight=w * len(w))
        evaluate(f"logistic C={c}", lr.decision_function(Xv), lr.decision_function(Xt))

    best = (-1, None)
    for mode in ("pls", "pca"):
        proj = PCAAngleProjector(n_components=FEATURE_DIM, mode=mode).fit(Xf, fit["y"])
        for scale in T.ANGLE_SCALES:
            proj.angle_scale = scale
            ef, ev = encode(proj.transform(Xf)), encode(proj.transform(Xv))
            kf, kv = kernel_matrix(ef), kernel_matrix(ev, ef)
            for c in T.C_GRID:
                svc = SVC(kernel="precomputed", C=c, class_weight="balanced").fit(kf, fit["y"], sample_weight=w * len(w))
                aucs = X.per_source_clip_auc(svc.decision_function(kv), val)
                score = float(np.mean([aucs[s] for s in X.SELECT]))
                if score > best[0]:
                    best = (score, (mode, scale, c))
    mode, scale, c = best[1]
    proj = PCAAngleProjector(n_components=FEATURE_DIM, mode=mode).fit(Xf, fit["y"])
    proj.angle_scale = scale
    ef = encode(proj.transform(Xf))
    svc = SVC(kernel="precomputed", C=c, class_weight="balanced").fit(kernel_matrix(ef), fit["y"], sample_weight=w * len(w))
    evaluate(f"QSVM {mode} s={scale:.2f} C={c}", svc.decision_function(kernel_matrix(encode(proj.transform(Xv)), ef)),
             svc.decision_function(kernel_matrix(encode(proj.transform(Xt)), ef)))


if __name__ == "__main__":
    main()
