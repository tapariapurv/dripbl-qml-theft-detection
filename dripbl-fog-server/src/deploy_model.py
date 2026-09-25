"""Promote a trained candidate to the model the fog node runs.

    .venv/bin/python3 src/deploy_model.py models/candidates/xd_resnet18_motion.joblib

1. loads it through LiveTheftClassifier and scores a real frame (refuses a broken artifact)
2. backs up the current models/qsvm_model.joblib to models/backup/<timestamp>.joblib
3. installs the candidate and writes reports/deployed_model.json (what the Research page shows)

The running node picks the new model up on its next restart.
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
LIVE = ROOT / "models" / "qsvm_model.joblib"


def main() -> None:
    cand = Path(sys.argv[1]).resolve()
    clf = LiveTheftClassifier(cand)
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    r = clf.predict_frame(frame, frame if clf.in_channels == 6 else None)
    assert np.isfinite(r["margin"]) and len(r["features_4d"]) == 4, "candidate produced invalid output"
    meta = joblib.load(cand)["metadata"]
    if LIVE.exists():
        backup = ROOT / "models" / "backup" / f"qsvm_model_{time.strftime('%Y%m%d_%H%M%S')}.joblib"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LIVE, backup)
        print(f"backed up current model -> {backup.relative_to(ROOT)}")
    shutil.copy2(cand, LIVE.with_suffix(".tmp"))
    LIVE.with_suffix(".tmp").replace(LIVE)
    summary = {k: meta.get(k) for k in ("experiment", "ensemble_size", "in_channels", "clip_threshold",
                                         "clip_thresholds", "holdout", "holdout_site_calibrated", "training")}
    summary["deployed_at"] = time.time()
    summary["source"] = str(cand.relative_to(ROOT)) if cand.is_relative_to(ROOT) else str(cand)
    (ROOT / "reports" / "deployed_model.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"deployed {meta.get('experiment')} ({clf.in_channels}-channel, {len(clf._members)} members, "
          f"{r['latency_ms']:.1f} ms/frame). Restart the node to load it.")


if __name__ == "__main__":
    main()
