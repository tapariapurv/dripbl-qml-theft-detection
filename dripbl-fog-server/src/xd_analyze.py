"""Re-score saved xd_experiments candidates without retraining.

Variants per candidate (thresholds always chosen on VALIDATION clips, hold-outs scored once):
  global    one clip threshold for every dataset (as trained)
  per-site  each dataset's own threshold from its own validation clips: what a deployed
            node gets by calibrating to its camera domain (site profile)
  +flip     test-time augmentation: mean margin of the frame and its mirror image

    .venv/bin/python3 -u src/xd_analyze.py tiny_motion resnet18_rgb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cross_data as C  # noqa: E402
import xd_experiments as X  # noqa: E402
from quantum_kernel import encode, kernel_matrix  # noqa: E402


def site_thresholds(val_m: np.ndarray, val: dict) -> dict[str, float]:
    out = {}
    for s in X.SELECT:
        m = val["src"] == s
        lab, sc = C.clip_scores(val_m[m], val["vid"][m], val["y"][m])
        import experiments as E
        out[s] = E.pick_threshold(lab, sc)[0]
    return out


def main() -> None:
    for name in sys.argv[1:]:
        cand = joblib.load(X.CANDIDATES / f"{name}.joblib")
        cfg = X.Cfg(**{k: tuple(v) if k == "share" else v for k, v in cand["cfg"].items()})
        _, val = X.load_train(cfg)
        model = X.Net(cfg, 6 if cfg.motion else 3).to(X.DEVICE)
        model.load_state_dict(cand["state_dict"])
        head = cand["head"]
        head["states"] = encode(head["support"])
        hold = X.load_holdouts(cfg.motion)

        def margins(x: np.ndarray, flip: bool) -> np.ndarray:
            m = X.quantum_margins(model, head, x)
            return (m + X.quantum_margins(model, head, np.ascontiguousarray(x[:, :, ::-1]))) / 2 if flip else m

        for flip in (False, True):
            val_m = margins(val["x"], flip)
            glob, _ = X.balanced_threshold(val_m, val)
            per = site_thresholds(val_m, val)
            row = {}
            for s, d in hold.items():
                lab, sc = C.clip_scores(margins(d["x"], flip), d["vid"], d["y"])
                row[s] = {"global": C.report(lab, sc, glob)["accuracy"],
                          "site": C.report(lab, sc, per.get(s, glob))["accuracy"],
                          "auc": C.report(lab, sc, glob)["auc"]}
            print(f"{name:18s} flip={flip!s:5s} " + " | ".join(
                f"{s}: auc={r['auc']:.3f} acc global={r['global']:.3f} site={r['site']:.3f}" for s, r in row.items()),
                flush=True)


if __name__ == "__main__":
    main()
