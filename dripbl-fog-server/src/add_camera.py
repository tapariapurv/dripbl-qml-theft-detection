"""Teach the system a new camera from labelled clips recorded on it.

Why: a model trained on one store's footage learns that store's look; on a camera it
has never seen it scores scenes, not acts (measured: zero-shot on dataset-2 every clip
alerts). A few labelled clips from the new camera fix that.

    training-data/<site>/normal/*.mp4      everyday activity on this camera
    training-data/<site>/theft/*.mp4       thefts (or folder name "shoplifting")
    .venv/bin/python3 -u src/add_camera.py <site> [--base retail_tiny_m_ft] [--folds 5]

It holds out 20% of the site's videos (by video, per class) and never trains on them,
fine-tunes a k-fold ensemble starting from the deployed retail expert (which has never
seen this camera) on the site's other videos plus the retail training set (so it does not
forget), picks this site's alert threshold on out-of-fold clips, then reports accuracy on
the site's hold-out AND on the original retail hold-out. Deploy with the printed command.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cross_data as C  # noqa: E402
import train_xd as TX  # noqa: E402
import xd_experiments as X  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site", help="folder name under training-data/")
    ap.add_argument("--base", default="retail_tiny_m_ft", help="xd_experiments config to fine-tune")
    ap.add_argument("--init", default="models/retail_model.joblib", help="start from this model's weights")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, help="override (smoke tests)")
    ap.add_argument("--steps", type=int, help="override steps per epoch (smoke tests)")
    args = ap.parse_args()

    videos = C.custom_videos(args.site)
    split = C.custom_split(args.site)
    n_theft = sum(label for _, label, _ in videos)
    print(f"[add_camera] {args.site}: {len(videos) - n_theft} normal + {n_theft} theft videos; "
          f"{len(split['holdout'])} held out, {len(split['train'])} for training", flush=True)
    if not (ROOT / args.init).is_file():
        sys.exit(f"--init {args.init} not found (deploy a retail model first, or pass another)")

    base = X.CONFIGS[args.base]
    cfg = dataclasses.replace(base, name=f"site_{args.site}", extra=(args.site,), init_from=args.init,
                              **({"epochs": args.epochs} if args.epochs else {}),
                              **({"steps_per_epoch": args.steps} if args.steps else {}))
    out = ROOT / "models" / "candidates" / f"site_{args.site}.joblib"
    result = TX.run(cfg, args.folds, out)

    # the deployed node will use this model as its "retail" expert: give it this site's threshold
    site_thr = result["site_thresholds"].get(f"x:{args.site}", result["site_thresholds"]["default"])
    art = joblib.load(out)
    art["metadata"]["clip_thresholds"] = {**art["metadata"]["clip_thresholds"], "retail": site_thr}
    fn, _ = C.artifact_margin_fn(out)
    d = C.custom_part(args.site, "holdout")
    labels, scores = C.clip_scores(fn(d["rgb"], d["motion"], d["vid"]), d["vid"], d["y"])
    site_report = C.report(labels, scores, site_thr)
    art["metadata"]["site_holdout"] = {args.site: site_report}
    joblib.dump(art, out)
    (ROOT / "reports" / f"site_{args.site}.json").write_text(json.dumps(
        {"site": args.site, "site_threshold": site_thr, "site_holdout": site_report,
         "retail_holdout": result["holdout_site_calibrated"]["dataset1_holdout"],
         "oof_clip_auc": result["oof_clip_auc"]}, indent=2))
    r1 = result["holdout_site_calibrated"]["dataset1_holdout"]
    print(f"\n[add_camera] {args.site} hold-out ({site_report['clips']} clips): accuracy {site_report['accuracy']:.3f}, "
          f"AUC {site_report['auc']:.3f}, theft recall {site_report['recall']:.3f}, false alarms {site_report['fp']}\n"
          f"[add_camera] original retail hold-out: accuracy {r1['accuracy']:.3f} (checks it did not forget)\n"
          f"[add_camera] deploy on this camera:\n"
          f"    .venv/bin/python3 src/deploy_model.py {out.relative_to(ROOT)} --as retail\n"
          f"    then Settings -> Detection -> Site profile: Retail", flush=True)


if __name__ == "__main__":
    main()
