"""Combine k-fold ensembles trained on the SAME folds into one artifact.

    .venv/bin/python3 -u src/combine_experts.py retail_r18m retail_tiny_m --out models/candidates/retail_combo.joblib

Each input came from train_xd.py with the same seed and data, so member k of every
input never saw fold k. For each input we rebuild those out-of-fold (OOF) margins,
standardise them (the member's margin_shift / margin_scale = median / std of its OOF
margins), and the combined OOF score of a frame is the mean over inputs. Thresholds
(global + per site) are chosen on combined OOF clips only; the hold-outs are then
scored once through LiveTheftClassifier, like every other model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cross_data as C  # noqa: E402
import experiments as E  # noqa: E402
import train_xd as TX  # noqa: E402
import xd_experiments as X  # noqa: E402
from inference import LiveTheftClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def oof_margins(artifact: Path, pool: dict, fold: np.ndarray) -> np.ndarray:
    clf = LiveTheftClassifier(artifact)
    import os
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    out = np.zeros(len(pool["y"]))
    for k, member in enumerate(clf._members):
        idx = np.flatnonzero(fold == k)
        for s in range(0, len(idx), 512):
            chunk = pool["x"][idx[s:s + 512], ..., :clf.in_channels].astype(np.float32) / 255.0
            out[idx[s:s + 512]] = clf._member_margins(chunk, member)[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="xd_experiments config names, trained by train_xd.py")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    cfgs = [X.CONFIGS[c] for c in args.configs]
    assert len({(c.seed, c.sources, c.motion, c.size) for c in cfgs}) == 1, "inputs must share folds and inputs"
    pool = TX.training_pool(cfgs[0])
    fold = TX.assign_folds(pool, args.folds, cfgs[0].seed)
    members, oof_sum, parts = [], np.zeros(len(pool["y"])), []
    for c in cfgs:
        path = ROOT / "models" / "candidates" / f"{c.name}.joblib"
        art = joblib.load(path)
        oof = oof_margins(path, pool, fold)
        shift, scale = float(np.median(oof)), float(oof.std() + 1e-9)
        oof_sum += (oof - shift) / scale
        auc = X.per_source_clip_auc(oof, pool)
        parts.append({"config": c.name, "oof_clip_auc": auc})
        print(f"  {c.name}: OOF clip AUC {auc}", flush=True)
        for m in art["ensemble"]:
            members.append({**m, "backbone": art["backbone"], "margin_shift": shift, "margin_scale": scale})
    oof = oof_sum / len(cfgs)
    threshold, oof_acc = X.balanced_threshold(oof, pool)
    site = {"default": threshold}
    for name, src in (("cctv", "d3"), ("retail", "d1")):
        m = pool["src"] == src
        if m.any():
            site[name] = E.pick_threshold(*C.clip_scores(oof[m], pool["vid"][m], pool["y"][m]))[0]
    oof_auc = X.per_source_clip_auc(oof, pool)
    print(f"  COMBINED OOF clip AUC {oof_auc} acc {oof_acc} thresholds {site}", flush=True)

    first = joblib.load(ROOT / "models" / "candidates" / f"{cfgs[0].name}.joblib")
    artifact = {"format": 2, "backbone": first["backbone"], "ensemble": members,
                "metadata": {**first["metadata"], "experiment": "combo_" + "+".join(args.configs),
                             "ensemble_size": len(members), "clip_threshold": threshold, "clip_thresholds": site,
                             "training": {"combined": parts, "oof_clip_auc": oof_auc, "oof_clip_acc": oof_acc},
                             "timestamp": time.time()}}
    out = Path(args.out)
    joblib.dump(artifact, out)
    fn, thr = C.artifact_margin_fn(out)
    scores = C.holdout_clip_scores(fn)
    results = C.evaluate(fn, thr, scores=scores)
    site_results = C.evaluate(fn, thr, scores=scores, thresholds={"dataset3_test": site.get("cctv", thr),
                                                                  "dataset1_holdout": site.get("retail", thr)})
    artifact["metadata"].update({"holdout": results, "holdout_site_calibrated": site_results,
                                 "clip_test_accuracy": results["dataset3_test"]["accuracy"]})
    joblib.dump(artifact, out)
    (ROOT / "reports" / f"xd_metrics_{out.stem}.json").write_text(json.dumps(
        {"configs": args.configs, "oof_clip_auc": oof_auc, "oof_clip_acc": oof_acc, "site_thresholds": site,
         "holdout": results, "holdout_site_calibrated": site_results}, indent=2))
    print("  HOLD-OUT (site-calibrated):")
    for name, r in site_results.items():
        print(f"    {name:18s} acc={r['accuracy']:.3f} auc={r['auc']:.3f} recall={r['recall']:.3f} fp={r['fp']} fn={r['fn']}")
    print(f"saved -> {out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
