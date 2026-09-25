"""Re-score saved xd_experiments candidates without retraining.

For each candidate, per dataset and per clip-aggregation rule (mean / top25 / top10 / max
of the frame margins), with thresholds always chosen on VALIDATION clips:
  global  one threshold for every dataset
  site    each dataset's own threshold from its own validation clips (a node
          calibrated to its camera domain / site profile)
Clip scores are saved to reports/xd_clip_scores/<name>.npz for ensembling
(`--ensemble a b c`: z-score each member on its validation clips, then average).

    .venv/bin/python3 -u src/xd_analyze.py tiny_motion resnet18_rgb
    .venv/bin/python3 -u src/xd_analyze.py --ensemble tiny_motion resnet18_rgb resnet18_motion
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cross_data as C  # noqa: E402
import experiments as E  # noqa: E402
import xd_experiments as X  # noqa: E402
from quantum_kernel import encode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "reports" / "xd_clip_scores"
AGGS = ("mean", "top25", "top10", "max")


def clip_table(margins: np.ndarray, d: dict, how: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(clip ids, labels, scores) per video with aggregation `how`."""
    vids = list(dict.fromkeys(d["vid"].tolist()))
    lab, sc = E.aggregate(d["vid"].tolist(), d["y"], margins, how)
    return np.array(vids), lab, sc


def score_candidate(name: str) -> None:
    cand = joblib.load(X.CANDIDATES / f"{name}.joblib")
    cfg = X.Cfg(**{k: tuple(v) if k == "share" else v for k, v in cand["cfg"].items()})
    _, val = X.load_train(cfg)
    model = X.Net(cfg, 6 if cfg.motion else 3).to(X.DEVICE)
    model.load_state_dict(cand["state_dict"])
    head = {**cand["head"], "states": encode(cand["head"]["support"])}
    sets = {"val": val, **{f"test_{s}": d for s, d in X.load_holdouts(cfg.motion).items()}}
    out = {}
    for tag, d in sets.items():
        m = X.quantum_margins(model, head, d["x"])
        src = d["src"] if "src" in d else np.full(len(d["y"]), tag[-2:])
        for s in X.SOURCES:
            keep = src == s
            if not keep.any():
                continue
            sub = {k: v[keep] for k, v in d.items() if isinstance(v, np.ndarray) and len(v) == len(keep)}
            for how in AGGS:
                vids, lab, sc = clip_table(m[keep], sub, how)
                split = "val" if tag == "val" else "test"
                out[f"{split}_{s}_{how}_vid"], out[f"{split}_{s}_{how}_y"], out[f"{split}_{s}_{how}_s"] = vids, lab, sc
    SCORES.mkdir(parents=True, exist_ok=True)
    np.savez(SCORES / f"{name}.npz", **out)
    report(name, out)


def report(name: str, z: dict) -> None:
    for how in AGGS:
        line = []
        for s in X.SELECT:
            vy, vs = z[f"val_{s}_{how}_y"], z[f"val_{s}_{how}_s"]
            ty, ts = z[f"test_{s}_{how}_y"], z[f"test_{s}_{how}_s"]
            t_site = E.pick_threshold(vy, vs)[0]
            r = C.report(ty, ts, t_site)
            line.append(f"{s}: auc={r['auc']:.3f} site-acc={r['accuracy']:.3f} (val auc "
                        f"{C.report(vy, vs, t_site)['auc']:.3f})")
        print(f"{name:22s} {how:6s} " + " | ".join(line), flush=True)


def ensemble(names: list[str]) -> None:
    zs = [dict(np.load(SCORES / f"{n}.npz")) for n in names]
    out = {}
    for s in X.SELECT:
        for how in AGGS:
            for split in ("val", "test"):
                base = zs[0]
                vids = base[f"{split}_{s}_{how}_vid"]
                total = np.zeros(len(vids))
                for z in zs:
                    order = {v: i for i, v in enumerate(z[f"{split}_{s}_{how}_vid"])}
                    sc = z[f"{split}_{s}_{how}_s"][[order[v] for v in vids]]
                    ref = z[f"val_{s}_{how}_s"]            # normalise on validation clips only
                    total += (sc - ref.mean()) / (ref.std() + 1e-9)
                out[f"{split}_{s}_{how}_vid"], out[f"{split}_{s}_{how}_y"] = vids, base[f"{split}_{s}_{how}_y"]
                out[f"{split}_{s}_{how}_s"] = total / len(zs)
    report("ENSEMBLE(" + "+".join(names) + ")", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--ensemble", action="store_true")
    args = ap.parse_args()
    if args.ensemble:
        ensemble(args.names)
    else:
        for n in args.names:
            score_candidate(n)
