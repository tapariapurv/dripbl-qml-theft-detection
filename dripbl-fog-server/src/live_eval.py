"""End-to-end accuracy of the LIVE SYSTEM, not just the model.

Replays each hold-out video frame by frame through exactly what the fog node runs
(camera frame -> motion gate + gait -> windowed TheftDetector -> AI judge, with the
node's scoring rate, judgement window and debounce), on the video's own clock, and
records whether an ALERT fires during the clip. Real-world metric: an alarm on a
theft clip is a catch; an alarm on a normal clip is a false alarm.

    .venv/bin/python3 -u src/live_eval.py                       # retail hold-out, retail profile
    .venv/bin/python3 -u src/live_eval.py --profile default --set dataset2
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import cross_data as C  # noqa: E402
from ai_judge import AIJudge  # noqa: E402
from camera import to_bgr8  # noqa: E402
from config import ConfigStore, Settings  # noqa: E402
from edge_pipeline import EdgePipeline  # noqa: E402
from theft_detector import TheftDetector  # noqa: E402


def clips(which: str) -> list[tuple[Path, int]]:
    if which == "dataset1":
        hold = set(C.dataset1_split()["holdout"])
        return [(p, label) for folder, label in (("normal", 0), ("shoplifting", 1))
                for p in sorted((C.DATA / "dataset-1" / folder).glob("*.mp4")) if f"d1:{p.stem}" in hold]
    return [(p, 0 if p.stem.startswith("not_") else 1) for p in sorted((C.DATA / "dataset-2" / "videos").glob("*.mp4"))
            if which == "dataset2all" or p.stem in C.D2_HOLDOUT]


def calibration_windows(settings: Settings, det: TheftDetector) -> list[float]:
    """Window means seen while replaying dataset-2's TRAINING normal videos (never the hold-out),
    one per scored frame: what POST /api/calibrate learns from on a real camera."""
    means = []
    for p in sorted((C.DATA / "dataset-2" / "videos").glob("not_*.mp4")):
        if p.stem in C.D2_HOLDOUT:
            continue
        cap, i, fps = cv2.VideoCapture(str(p)), 0, 30.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        pipe, last = EdgePipeline(settings), -1e9
        det._history.clear()
        det._window.clear()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now, frame = i / fps, to_bgr8(frame)
            i += 1
            triggered, _ = pipe.observe(frame, now)
            det.observe(frame, now)
            if triggered and now - last >= 1.0 / settings.score_fps:
                det.score(frame, now)
                last = now
                if det.window_frames(now) >= 2:
                    means.append(float(np.mean([m for _, m in det._window])))
        cap.release()
    return means


def replay(path: Path, settings: Settings, det: TheftDetector, judge: AIJudge) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipe = EdgePipeline(settings)                      # fresh background model per clip, like the node
    det._history.clear()
    det._window.clear()
    last_trigger = last_score = -1e9
    trials, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now, frame = i / fps, to_bgr8(frame)
        i += 1
        triggered, gait = pipe.observe(frame, now)
        det.observe(frame, now)
        if triggered and now - last_score >= 1.0 / settings.score_fps:
            det.score(frame, now)
            last_score = now
        ready = det.window_ready(now)
        if triggered and ready and now - last_trigger >= settings.debounce_sec:
            q = det.classify(frame, now)
            last_score = now
            v = judge.judge(q, gait)
            trials.append({"t": round(now, 2), "suspicion": v["suspicion_score"], "flagged": v["flagged"],
                           "window_mean": q["window"]["mean_margin"]})
            last_trigger = now
    cap.release()
    return {"trials": len(trials), "alert": any(t["flagged"] for t in trials),
            "max_suspicion": max((t["suspicion"] for t in trials), default=0.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="retail", choices=["default", "cctv", "retail"])
    ap.add_argument("--set", default="dataset1", choices=["dataset1", "dataset2", "dataset2all"],
                    help="dataset2all = every dataset-2 video (only for models that never trained on it)")
    ap.add_argument("--sensitivity", type=float, default=None)
    ap.add_argument("--model", help="evaluate this artifact instead of the installed one")
    ap.add_argument("--calibrate", action="store_true",
                    help="dataset2: calibrate the threshold on its training normal videos first")
    args = ap.parse_args()
    base = ConfigStore().settings
    over = {"sensitivity": args.sensitivity} if args.sensitivity is not None else {}
    if args.model:
        over["retail_model_path" if args.profile == "retail" else "model_path"] = str(Path(args.model).resolve())
    settings = dataclasses.replace(base, site_profile=args.profile, roi=(0, 0, 10000, 10000), **over)  # whole frame
    det, judge = TheftDetector(settings), AIJudge(settings)
    if args.calibrate:
        means = np.array(calibration_windows(settings, det))
        spread = float(np.subtract(*np.percentile(means, [75, 25]))) or 1e-3
        cal = float(np.quantile(means, 0.98) + 0.1 * spread)  # same rule as POST /api/calibrate
        settings = dataclasses.replace(settings, threshold_mode="calibrated", calibrated_threshold=cal)
        det._settings, judge._settings = settings, settings
        print(f"calibrated on {len(means)} normal windows from training videos -> threshold {cal:+.3f}")
    print(f"model: {det.description} · profile {args.profile} · threshold {det.threshold:+.3f} · "
          f"sensitivity {settings.sensitivity}", flush=True)
    rows = []
    for path, label in clips(args.set):
        r = replay(path, settings, det, judge)
        rows.append({"clip": path.stem, "label": label, **r})
        print(f"  {path.stem:18s} {'THEFT ' if label else 'normal'} trials={r['trials']:2d} "
              f"alert={'YES' if r['alert'] else 'no ':3s} max_suspicion={r['max_suspicion']:.2f}", flush=True)
    y = np.array([r["label"] for r in rows])
    alert = np.array([r["alert"] for r in rows])
    summary = {"set": args.set, "profile": args.profile, "model": det.description, "clips": len(rows),
               "accuracy": float((alert == y).mean()), "thefts_caught": int((alert & (y == 1)).sum()),
               "thefts": int((y == 1).sum()), "false_alarms": int((alert & (y == 0)).sum()),
               "normals": int((y == 0).sum()),
               "auc_max_suspicion": float(roc_auc_score(y, [r["max_suspicion"] for r in rows])) if len(set(y)) > 1 else None}
    print(json.dumps(summary))
    (ROOT / "reports" / f"live_eval_{args.set}_{args.profile}.json").write_text(json.dumps({"summary": summary, "clips": rows}, indent=2))


if __name__ == "__main__":
    main()
