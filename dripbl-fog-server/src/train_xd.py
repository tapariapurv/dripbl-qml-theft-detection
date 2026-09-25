"""Train the deployable cross-dataset model: a k-fold ensemble of one xd_experiments config.

    .venv/bin/python3 -u src/train_xd.py resnet18_motion [--folds 5] [--out models/qsvm_model.joblib]

Why folds: members train on different 80% slices of the videos (of every dataset),
so their errors decorrelate, and each video is held out by exactly one member. That
member's margin on its held-out clips is an out-of-fold (OOF) prediction; pooling the
OOF clips of all members gives ~1,200 clips for choosing the clip threshold, instead
of one small validation split.

Protocol (unchanged strictness): dataset-3/Test, the dataset-1 hold-out and the
dataset-2 hold-out are never used for training or for any choice; they are scored
once, through LiveTheftClassifier (the live node's code path), after saving.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cross_data as C  # noqa: E402
import xd_experiments as X  # noqa: E402
from quantum_kernel import FEATURE_DIM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--steps", type=int, help="steps per epoch (smoke tests)")
    ap.add_argument("--out", default=str(ROOT / "models" / "candidates" / "xd_ensemble.joblib"))
    args = ap.parse_args()
    cfg = X.CONFIGS[args.config]
    if args.epochs:
        cfg = dataclasses.replace(cfg, epochs=args.epochs)
    if args.steps:
        cfg = dataclasses.replace(cfg, steps_per_epoch=args.steps)
    t0 = time.time()

    fit, val = X.load_train(cfg)
    pool = {k: np.concatenate([fit[k], val[k]]) for k in fit}
    print(f"[train_xd] {cfg.name}: {len(pool['y'])} frames, "
          + " ".join(f"{s}={len(set(pool['vid'][pool['src'] == s].tolist()))} videos" for s in X.SOURCES), flush=True)

    # fold assignment by VIDEO, stratified per (dataset, class)
    fold_of: dict[str, int] = {}
    rng = np.random.default_rng(cfg.seed)
    for s in X.SOURCES:
        for label in (0, 1):
            names = sorted(set(pool["vid"][(pool["src"] == s) & (pool["y"] == label)].tolist()))
            for i, j in enumerate(rng.permutation(len(names))):
                fold_of[names[j]] = i % args.folds
    fold = np.array([fold_of[v] for v in pool["vid"]])

    members, oof = [], np.zeros(len(pool["y"]))
    for k in range(args.folds):
        tr = {key: v[fold != k] for key, v in pool.items()}
        va = {key: v[fold == k] for key, v in pool.items()}
        print(f"\n[train_xd] fold {k + 1}/{args.folds}: fit {len(tr['y'])} / val {len(va['y'])} frames", flush=True)
        kcfg = dataclasses.replace(cfg, seed=cfg.seed + k)
        model, _ = X.train(kcfg, tr, va)
        head = X.fit_quantum_head(kcfg, model, tr, va)
        oof[fold == k] = X.quantum_margins(model, head, va["x"])
        model = model.cpu()
        members.append({
            "backbone_state": {key.removeprefix("backbone."): v for key, v in model.state_dict().items()
                               if key.startswith("backbone.")},
            "svm_classifier": head["clf"], "pca_transformer": head["projector"],
            "support_vectors": head["support"], "angle_scale": head["scale"], "C": head["C"],
        })

    threshold, oof_acc = X.balanced_threshold(oof, pool)
    import experiments as E
    site_thresholds = {"default": threshold}
    for site, src in (("cctv", "d3"), ("retail", "d1")):  # a node calibrated to its camera domain
        m = pool["src"] == src
        if not m.any():
            continue
        site_thresholds[site] = E.pick_threshold(*C.clip_scores(oof[m], pool["vid"][m], pool["y"][m]))[0]
    oof_auc = X.per_source_clip_auc(oof, pool)
    print(f"\n[train_xd] OOF clip AUC {oof_auc}  threshold {threshold:+.4f}  OOF clip acc {oof_acc}", flush=True)

    artifact = {
        "format": 2,
        "backbone": {"kind": cfg.backbone, "channels": 6 if cfg.motion else 3, "input_size": cfg.input_size},
        "ensemble": members,
        "metadata": {
            "experiment": f"xd_{cfg.name}_{args.folds}fold", "ensemble_size": len(members),
            "feature_map": {"type": "ZZFeatureMap", "feature_dimension": FEATURE_DIM, "reps": 2,
                            "entanglement": "linear"},
            "in_channels": 6 if cfg.motion else 3, "frame_size": C.SIZE, "motion_gap_sec": C.MOTION_GAP / 30,
            "clip_aggregation": "mean", "clip_threshold": threshold, "clip_thresholds": site_thresholds,
            "training": {"datasets": ["dataset-3/Train", "dataset-1 train split", "dataset-2 minus holdout"],
                         "config": dataclasses.asdict(cfg), "oof_clip_auc": oof_auc, "oof_clip_acc": oof_acc},
            "classes": {"0": "Normal", "1": "Anomaly"}, "timestamp": time.time(),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out)

    # score the SAVED artifact through the live node's classifier: what you deploy is what is measured
    fn, thr = C.artifact_margin_fn(out)
    results = C.evaluate(fn, thr)
    site_results = C.evaluate(fn, thr, thresholds={"dataset3_test": site_thresholds.get("cctv", thr),
                                                   "dataset1_holdout": site_thresholds.get("retail", thr)})
    art = joblib.load(out)
    art["metadata"].update({
        "holdout": results, "holdout_site_calibrated": site_results,
        "clip_test_accuracy": results["dataset3_test"]["accuracy"],
        "dataset1_holdout_accuracy": results["dataset1_holdout"]["accuracy"],
    })
    joblib.dump(art, out)
    metrics = {"artifact": str(out), "config": dataclasses.asdict(cfg), "folds": args.folds,
               "threshold": threshold, "oof_clip_auc": oof_auc, "oof_clip_acc": oof_acc,
               "site_thresholds": site_thresholds, "holdout": results, "holdout_site_calibrated": site_results,
               "minutes": round((time.time() - t0) / 60, 1),
               "protocol": "k-fold by video over dataset-3 Train + dataset-1 train split + dataset-2 train clips; "
                           "threshold from pooled out-of-fold clips; hold-outs scored once via LiveTheftClassifier"}
    (ROOT / "reports" / f"xd_metrics_{out.stem}.json").write_text(json.dumps(metrics, indent=2))
    print("\n[train_xd] HOLD-OUT (scored once, live code path)")
    for name, r in results.items():
        print(f"    {name:18s} acc={r['accuracy']:.3f} auc={r['auc']:.3f} recall={r['recall']:.3f} "
              f"fp={r['fp']} fn={r['fn']} clips={r['clips']}")
    print("  site-calibrated (threshold from that domain's own out-of-fold clips):")
    for name, r in site_results.items():
        print(f"    {name:18s} acc={r['accuracy']:.3f} recall={r['recall']:.3f} fp={r['fp']} fn={r['fn']}")
    print(f"saved -> {out}  ({metrics['minutes']} min)")


if __name__ == "__main__":
    main()
