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
            if p.stem in C.D2_HOLDOUT]


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
        ready = det.window_frames(now) >= max(2, int(settings.score_fps * settings.judge_window_sec / 2))
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
    ap.add_argument("--set", default="dataset1", choices=["dataset1", "dataset2"])
    ap.add_argument("--sensitivity", type=float, default=None)
    args = ap.parse_args()
    base = ConfigStore().settings
    settings = dataclasses.replace(base, site_profile=args.profile, roi=(0, 0, 10000, 10000),  # whole frame
                                   **({"sensitivity": args.sensitivity} if args.sensitivity is not None else {}))
    det, judge = TheftDetector(settings), AIJudge(settings)
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
