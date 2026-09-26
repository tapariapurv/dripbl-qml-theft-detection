"""Promote a trained candidate to the model the fog node runs.

    .venv/bin/python3 src/deploy_model.py models/candidates/cctv_r18.joblib              # default/CCTV model
    .venv/bin/python3 src/deploy_model.py models/candidates/retail_r18m.joblib --as retail  # retail expert

1. loads it through LiveTheftClassifier and scores a real frame (refuses a broken artifact)
2. backs up the model it replaces to models/backup/<timestamp>.joblib
3. installs it (models/qsvm_model.joblib, or models/retail_model.joblib with --as retail) and
   writes reports/deployed_<slot>.json (what the Research page shows)

A running node hot-swaps to the new file within about a second (no restart).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import LiveTheftClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SLOTS = {"default": "qsvm_model.joblib", "retail": "retail_model.joblib"}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--as", dest="slot", choices=sorted(SLOTS), default="default")
    args = ap.parse_args()
    cand = Path(args.candidate).resolve()
    live = ROOT / "models" / SLOTS[args.slot]
    clf = LiveTheftClassifier(cand)
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    r = clf.predict_frame(frame, frame if clf.in_channels == 6 else None)
    assert np.isfinite(r["margin"]) and len(r["features_4d"]) == 4, "candidate produced invalid output"
    art = joblib.load(cand)
    meta = art["metadata"]

    # store encoder weights as float16 (half the file; inference still runs in float32) and
    # refuse the conversion if it moves any margin noticeably
    for m in art["ensemble"]:
        for key in ("backbone_state", "cnn_state_dict"):
            if key in m:
                m[key] = {k: v.half() if v.is_floating_point() else v for k, v in m[key].items()}
    half = live.with_suffix(".half.tmp")
    joblib.dump(art, half, compress=3)
    probe = np.random.default_rng(1).integers(0, 255, (128, 64, 64, clf.in_channels), dtype=np.uint8)
    full, halved = clf.margins_from_arrays(probe), LiveTheftClassifier(half).margins_from_arrays(probe)
    drift, spread = float(np.abs(full - halved).max()), float(full.std()) or 1.0
    thr = float(meta.get("clip_threshold", 0.0))
    flips = int(((full > thr) != (halved > thr)).sum())
    # scale-aware: ensembles of standardised members have margins in std units
    if drift < 0.05 * spread and flips == 0:
        print(f"float16 storage: max margin change {drift:.4f} ({drift / spread:.2%} of spread), 0 decisions flip; "
              f"{cand.stat().st_size / 1e6:.0f} MB -> {half.stat().st_size / 1e6:.0f} MB")
    else:  # not provably harmless: ship full precision (releases take files up to 2 GB)
        print(f"float16 would change margins by {drift / spread:.1%} of spread ({flips} decisions flip): "
              f"keeping full precision")
        half.unlink()
        shutil.copy2(cand, half)
    if live.exists():
        backup = ROOT / "models" / "backup" / f"{live.stem}_{time.strftime('%Y%m%d_%H%M%S')}.joblib"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live, backup)
        print(f"backed up current model -> {backup.relative_to(ROOT)}")
    half.replace(live)
    summary = {k: meta.get(k) for k in ("experiment", "ensemble_size", "in_channels", "clip_threshold",
                                         "clip_thresholds", "holdout", "holdout_site_calibrated", "training")}
    summary["deployed_at"] = time.time()
    summary["source"] = str(cand.relative_to(ROOT)) if cand.is_relative_to(ROOT) else str(cand)
    summary["slot"] = args.slot
    (ROOT / "reports" / f"deployed_{args.slot}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"deployed {meta.get('experiment')} ({clf.in_channels}-channel, {len(clf._members)} members, "
          f"{r['latency_ms']:.1f} ms/frame). A running node switches to it within ~1 s.")


if __name__ == "__main__":
    main()
