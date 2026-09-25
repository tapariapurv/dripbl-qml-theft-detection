"""Per-frame anomaly labels for dataset-3 crime frames, from dataset-4's masks.

dataset-3 labels every frame of a crime video as Anomaly, but the crime occupies
a fraction of the clip -- for Shoplifting only ~17% of frames. Training on that
teaches the CNN that "a shop aisle" is anomalous, which is exactly the failure
mode measured earlier (it transferred at chance to dataset-2).

dataset-4/Labels/<Class>.csv carries UCF-Crime's temporal annotation as 32
uniform bins per video: `<video>_<bin>,<Class>,<0|1>`. dataset-3 filenames are
`<video>_<frame>.png` where <frame> is the original video frame number at
step 10, so bin = floor(frame / (max_frame + 10) * 32) lines the two up.

Applied to TRAIN ONLY. Test keeps its clip-level protocol so the numbers stay
comparable with every earlier run.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

CLASSES = ("Burglary", "Robbery", "Shoplifting", "Stealing")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "training-data"
EXTENTS = ROOT / "reports" / "video_extents.json"
N_BINS = 32


def _masks() -> dict[str, dict[int, int]]:
    out: dict[str, dict[int, int]] = {}
    for name in CLASSES:
        path = DATA / "dataset-4" / "Labels" / f"{name}.csv"
        if not path.is_file():
            continue
        for row in csv.reader(path.open()):
            if len(row) < 3 or row[2].strip() not in ("0", "1"):
                continue
            m = re.fullmatch(r"(.+)_(\d+)", row[0].strip())
            if m:
                out.setdefault(m.group(1), {})[int(m.group(2))] = int(row[2])
    return out


def _extents() -> dict[str, int]:
    """Highest frame number present per video -- cached, the glob is ~1.2M files."""
    if EXTENTS.is_file():
        return json.loads(EXTENTS.read_text())
    ext: dict[str, int] = {}
    for split in ("Train", "Test"):
        for name in CLASSES:
            for p in (DATA / "dataset-3" / split / name).glob("*.png"):
                video, frame = p.stem.rsplit("_", 1)
                i = int(frame)
                if i > ext.get(video, -1):
                    ext[video] = i
    EXTENTS.parent.mkdir(parents=True, exist_ok=True)
    EXTENTS.write_text(json.dumps(ext))
    return ext


_CACHE: dict[str, dict[str, int]] | None = None


def frame_labels() -> dict[str, int]:
    """{'Shoplifting001_x264_120': 0|1} for every crime frame a mask covers."""
    global _CACHE
    if _CACHE is None:
        masks, ext = _masks(), _extents()
        labels: dict[str, int] = {}
        for video, bins in masks.items():
            if video not in ext or not bins:
                continue
            span = ext[video] + 10          # frames are sampled at step 10
            n_bins = max(bins) + 1
            for frame in range(0, ext[video] + 1, 10):
                b = min(int(frame / span * n_bins), n_bins - 1)
                if b in bins:
                    labels[f"{video}_{frame}"] = bins[b]
        _CACHE = labels
    return _CACHE


def refine(paths, y):
    """Replace clip-level labels with true frame labels where a mask exists.

    Normal frames and videos without a mask are left exactly as they were, so
    this can only ever correct a wrong 1 to a 0 (or confirm it).
    """
    import numpy as np

    lut = frame_labels()
    y = np.asarray(y).copy()
    flipped = 0
    for i, p in enumerate(paths):
        v = lut.get(Path(p).stem)
        if v is not None and v != y[i]:
            y[i] = v
            flipped += 1
    return y, flipped


if __name__ == "__main__":
    lut = frame_labels()
    anom = sum(lut.values())
    print(f"masked crime frames: {len(lut)}  true-anomaly {anom} ({anom / len(lut):.1%})")
    assert len(lut) > 50_000, "expected ~60k covered frames"
    assert 0 < anom < len(lut), "mask must contain both classes"
    sample = next(k for k in lut if k.startswith("Shoplifting"))
    print("sample:", sample, "->", lut[sample])
    print("ok")
