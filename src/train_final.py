"""Final champion: k-fold-by-video ensemble, the deployable artifact.

Why folds rather than more seeds: seed-ensemble members see identical data and
differ only by initialisation, so their errors correlate. Fold members each
train on a different 80% of the source videos, which decorrelates them, and
every training video is used by some member. It also fixes the calibration
weakness measured earlier -- each fold's held-out videos pool into ~1000
validation clips for choosing the clip aggregation and threshold, instead of
the 176 clips a single 15% split provides.

Evaluation protocol is unchanged and strict: fit on dataset-3/Train only,
select everything on held-out-by-video Train clips, score dataset-3/Test once.

    .venv/bin/python3 -u src/train_final.py [--folds 5] [--epochs 10]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiments as E  # noqa: E402
import train_qsvm as T  # noqa: E402
from quantum_kernel import FEATURE_DIM, encode, kernel_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRAMES_PER_VIDEO = 30
TEST_FRAMES_PER_VIDEO = 120   # more frames per clip -> lower-variance clip score
SEED = 42


def load_all(frames_per_video: int, test_frames: int) -> dict:
    rng = np.random.default_rng(SEED)
    tr_p, y_tr, _, vid_tr = T.sample_by_video("Train", frames_per_video, rng, video_subset="all")
    te_p, y_te, grp_te, vid_te = T.sample_by_video("Test", test_frames, rng)
    print(f"  train frames={len(y_tr)} videos={len(set(vid_tr))} | "
          f"test frames={len(y_te)} clips={len(set(vid_te))}", flush=True)
    return {
        "train": (E._load(tr_p, False), y_tr, vid_tr),
        "test": (E._load(te_p, False), y_te, vid_te, grp_te),
    }


def load_dataset2(holdout: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, list[str]]:
    """dataset-2 frames at 64x64, split by source video into (train, holdout).

    Only 8 videos, but from a different camera/source than dataset-3 -- a model
    trained on dataset-3 alone scores AUC 0.56 here (chance), i.e. it learned
    that dataset's appearance rather than theft. Mixing it in is about
    cross-source robustness, not about the dataset-3 headline number.
    """
    import cv2
    root = ROOT.parent / "training-data" / "dataset-2" / "images"
    xs, ys, vids = [], [], []
    hx, hy, hv = [], [], []
    for f in sorted(root.glob("*.png")):
        video = f.name.rsplit("_f", 1)[0]
        label = 0 if video.startswith("not_") else 1
        img = cv2.cvtColor(cv2.resize(cv2.imread(str(f)), (T.FRAME_SIZE, T.FRAME_SIZE),
                                      interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        if video in holdout:
            hx.append(img); hy.append(label); hv.append(video)
        else:
            xs.append(img); ys.append(label); vids.append(video)
    return (np.array(xs, dtype=np.uint8), np.array(ys), vids,
            np.array(hx, dtype=np.uint8), np.array(hy), hv)


def fold_split(videos: list[str], fold: int, n_folds: int) -> tuple[np.ndarray, np.ndarray]:
    """Row indices for (fit, val), partitioned by SOURCE VIDEO, not by frame."""
    unique = sorted(set(videos))
    order = np.random.default_rng(SEED).permutation(len(unique))
    val_videos = {unique[i] for i in order[fold::n_folds]}
    val = np.array([i for i, v in enumerate(videos) if v in val_videos])
    fit = np.array([i for i, v in enumerate(videos) if v not in val_videos])
    return fit, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5, help="split geometry (1/folds held out per member)")
    ap.add_argument("--run-folds", type=int, default=None, help="train only this many members")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--frames", type=int, default=FRAMES_PER_VIDEO)
    ap.add_argument("--with-dataset2", action="store_true",
                    help="mix dataset-2 into training and report cross-source transfer")
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    t0 = time.time()
    print("[1/4] loading frames ...", flush=True)
    data_all = load_all(args.frames, TEST_FRAMES_PER_VIDEO)
    x_all, y_all, vid_all = data_all["train"]
    x_te, y_te, vid_te, grp_te = data_all["test"]

    d2_hold = ("not_shoplifting4", "shoplifting4")   # one clip per class, never trained on
    if args.with_dataset2:
        d2x, d2y, d2v, hx, hy, hv = load_dataset2(d2_hold)
        x_all = np.concatenate([x_all, d2x])
        y_all = np.concatenate([y_all, d2y])
        vid_all = list(vid_all) + d2v
        print(f"  + dataset-2: {len(d2y)} frames from {len(set(d2v))} videos "
              f"(holdout {d2_hold}: {len(hy)} frames)", flush=True)

    val_pool: list[tuple[list[str], np.ndarray, np.ndarray]] = []
    test_margins, fits = [], []
    t_fold0 = time.time()

    for fold in range(min(args.run_folds or args.folds, args.folds)):
        cfg = E.Config(f"fold{fold}", frames_per_video=args.frames, epochs=args.epochs,
                       seeds=(SEED + fold,), test_frames_per_video=TEST_FRAMES_PER_VIDEO)
        fit_idx, val_idx = fold_split(vid_all, fold, args.folds)
        data = {
            "train": (x_all[fit_idx], y_all[fit_idx], [vid_all[i] for i in fit_idx]),
            "val": (x_all[val_idx], y_all[val_idx], [vid_all[i] for i in val_idx]),
            "test": (x_te, y_te, vid_te, grp_te),
        }
        n_run = min(args.run_folds or args.folds, args.folds)
        eta = (time.time() - t_fold0) / fold * (n_run - fold) if fold else None
        print(f"\n{'=' * 74}\n[2/4] FOLD {fold + 1}/{n_run}   fit={len(fit_idx)} frames "
              f"({len(set(data_fit_vids := [vid_all[i] for i in fit_idx]))} videos)  "
              f"val={len(val_idx)} frames ({len(set(vid_all[i] for i in val_idx))} videos)"
              + (f"   eta {eta / 60:.1f} min" if eta else "") + f"\n{'=' * 74}", flush=True)
        del data_fit_vids
        model, cnn_acc = E.train_cnn(cfg, SEED + fold, data)
        fit = E.fit_qsvm(cfg, model, data, np.random.default_rng(SEED + fold))
        fit["model"] = model
        fits.append(fit)
        test_margins.append(fit["test_margins"])
        val_pool.append(([vid_all[i] for i in val_idx], y_all[val_idx], fit["val_margins"]))
        solo_lab, solo_sc = E.aggregate(vid_te, y_te, fit["test_margins"], "mean")
        ens_lab, ens_sc = E.aggregate(vid_te, y_te, np.mean(test_margins, axis=0), "mean")
        print(f"      cnn_val={cnn_acc:.4f} svm_val={fit['svm_val_acc']:.4f} "
              f"proj={fit['projector_mode']} scale={fit['angle_scale']} C={fit['C']} "
              f"support_vectors={len(fit['support'])}\n"
              f"      this fold alone on Test: clip AUC={E.clip_report(solo_lab, solo_sc, 0.0)['roc_auc']:.4f} "
              f"| ensemble so far ({len(test_margins)} members): "
              f"clip AUC={E.clip_report(ens_lab, ens_sc, 0.0)['roc_auc']:.4f}", flush=True)

    print("[3/4] selecting aggregation + threshold on held-out clips ...", flush=True)
    # Each fold's SVM has its own margin scale, so thresholds are averaged rather
    # than pooled after z-scoring: standardising against the TEST split's own mean
    # and variance would leak the test distribution into the decision, which is
    # not something a deployed node could do anyway.
    best = (-1.0, None, None)
    for how in E.AGGREGATIONS:
        accs, thrs = [], []
        for vids, yv, margins in val_pool:
            lab, sc = E.aggregate(vids, yv, margins, how)
            thr, acc = E.pick_threshold(lab, sc)
            accs.append(acc)
            thrs.append(thr)
        acc, thr = float(np.mean(accs)), float(np.mean(thrs))
        print(f"      {how:7s} mean held-out clip acc={acc:.4f} threshold={thr:+.4f}", flush=True)
        if acc > best[0]:
            best = (acc, how, thr)
    val_acc, how, threshold = best
    print(f"      chosen: {how} @ {threshold:+.4f} (val {val_acc:.4f})", flush=True)

    ens_test = np.mean(test_margins, axis=0)
    t_lab, t_scores = E.aggregate(vid_te, y_te, ens_test, how)
    clip = E.clip_report(t_lab, t_scores, threshold)
    clip["oracle_threshold_accuracy"] = E.pick_threshold(t_lab, t_scores)[1]
    clip_zero = E.clip_report(t_lab, t_scores, 0.0)
    frame_acc = float(((ens_test > 0).astype(int) == y_te).mean())

    d2_report = None
    if args.with_dataset2:
        def _d2_margin(f):
            lat = np.concatenate([E.embed_batch_compat(f["model"], hx[i:i + 256])
                                  for i in range(0, len(hx), 256)])
            ang = f["projector"].transform(lat)
            K = np.zeros((len(ang), int(f["clf"].shape_fit_[0])), dtype=np.float64)
            K[:, f["clf"].support_] = kernel_matrix(encode(ang), encode(f["support"]))
            return f["clf"].decision_function(K)

        d2_margins = np.mean([_d2_margin(f) for f in fits], axis=0)
        lab, sc = E.aggregate(hv, hy, d2_margins, how)
        d2_report = {"clips": len(lab), "frame_accuracy": float(((d2_margins > 0).astype(int) == hy).mean()),
                     "clip_scores": {v: float(s_) for v, s_ in zip(dict.fromkeys(hv), sc)},
                     "clip_labels": lab.tolist(),
                     "clip_correct": int(((sc > threshold).astype(int) == lab).sum())}
        print("      dataset-2 holdout: " + json.dumps(d2_report), flush=True)

    metrics = {
        "dataset2_holdout": d2_report,
        "model": f"{args.folds}-fold video ensemble, {args.epochs} epochs",
        "aggregation": how, "val_clip_accuracy": val_acc,
        "frame_accuracy": frame_acc, "clip": clip,
        "clip_zero_threshold": clip_zero, "threshold": threshold,
        "n_test_clips": int(len(t_lab)), "n_test_frames": int(len(y_te)),
        "runtime_sec": round(time.time() - t0, 1),
        "protocol": "fit on Train videos; aggregation+threshold on pooled held-out Train clips; Test scored once",
    }
    cm = clip["confusion_matrix"]
    print(f"\n{'=' * 74}\n[4/4] FINAL — dataset-3/Test, {len(t_lab)} clips never seen by any fold\n"
          f"{'=' * 74}", flush=True)
    for name, value in (("clip accuracy", clip["accuracy"]),
                        ("balanced accuracy", clip["balanced_accuracy"]),
                        ("recall (thefts caught)", clip["recall"]),
                        ("precision", clip["precision"]),
                        ("ROC-AUC (threshold-free)", clip["roc_auc"]),
                        ("oracle-threshold accuracy", clip["oracle_threshold_accuracy"]),
                        ("frame accuracy", frame_acc)):
        print(f"      {name:<28} {value:.4f}", flush=True)
    print(f"      confusion (clips)            tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}\n"
          f"      aggregation={how} threshold={threshold:+.4f} members={len(fits)} "
          f"runtime={metrics['runtime_sec'] / 60:.1f} min", flush=True)

    members = [{
        "svm_classifier": f["clf"], "pca_transformer": f["projector"],
        "support_vectors": f["support"], "support_vector_labels": f["support_labels"],
        "cnn_state_dict": f["model"].backbone.state_dict(),
    } for f in fits]
    artifact = {
        **members[0], "ensemble": members,
        "metadata": {
            "feature_map": {"type": "ZZFeatureMap", "feature_dimension": FEATURE_DIM,
                            "reps": 2, "entanglement": "linear"},
            "svm_hyperparams": {"kernel": "precomputed", "C": fits[0]["C"]},
            "angle_scale": fits[0]["angle_scale"], "projector_mode": fits[0]["projector_mode"],
            "cnn_latent_dim": 128, "in_channels": 3, "frame_size": T.FRAME_SIZE,
            "classes": {"0": "Normal", "1": "Anomaly"},
            "clip_aggregation": how, "clip_threshold": threshold,
            "ensemble_size": len(members), "experiment": "final_fold_ensemble",
            "clip_test_accuracy": clip["accuracy"], "frame_test_accuracy": frame_acc,
            "timestamp": time.time(),
        },
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(artifact, ROOT / "models" / "qsvm_model.joblib")
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved -> {ROOT / 'models' / 'qsvm_model.joblib'}  ({metrics['runtime_sec']:.0f}s)")


if __name__ == "__main__":
    main()
