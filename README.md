# DRIPBL: Quantum-Kernel Theft Detection on a Cloud–Edge–Fog Pipeline

A research system that detects theft on camera. A camera feeds a fog node (Raspberry Pi or laptop), which
gates on motion inside a region of interest and tracks gait. It compresses what it sees with a CNN and
supervised PLS into **4 qubit angles**, classifies them with a **5-fold quantum-kernel SVM ensemble**
(ZZFeatureMap fidelity kernel), and judges a 2-second window rather than a single frame. An AI judge
combines that with the gait signals, and alerts reach a human through the dashboard, notifications and
GPIO alarms.

DRIBPL Research Program 2026 · Researcher: Purv Taparia · Coach: Mrs. Rachna Nath · Quantum advisor: Mr. Anirban Bhattacharya

## Results

Every number is measured once, on videos no model, threshold or design choice ever saw. Each camera
domain runs its own expert, selected by the node's **site profile**.

| Camera domain | Model | Accuracy | Notes |
|---|---|---|---|
| CCTV (UCF-Crime Test, 194 clips) | CCTV expert, ResNet-18 × 5 | **89.7%** | AUC 0.890, 6 false alarms (the original model: 87.1%, 10) |
| Retail (dataset-1 hold-out, 36 clips) | retail expert, 10-member ensemble | **80.6%** | AUC 0.880; 87.0% out-of-fold on 146 training clips, **85.7%** over all 182 |
| Retail, the live system end to end | same, replayed through the node's real logic | **14 of 18 thefts caught** | 3 false alarms. The original model caught 0 of 18 |
| Retail, lightweight option (Pi) | tiny CNN × 5 | 15 of 18 caught, 2 false alarms | 10× cheaper per frame |
| Camera never seen before (dataset-2) | any | not solved | scores follow the scene, not the act. Fix: `src/add_camera.py` |

**What made the difference:** one expert per camera domain (training jointly let UCF-Crime's 20,000 frames
drown the retail signal); transfer learning (learn "crime" on UCF-Crime, then fine-tune on retail: out-of-fold
AUC 0.879 → 0.914); judging clips instead of single frames, which is how accuracy is measured. **What didn't
work** (so it isn't tried again): a Kinetics video model, higher resolution, and top-k aggregation. See
`dripbl-fog-server/CLAUDE.md` for the full table.

**Quantum hardware:** exact statevector simulation on the fog node is the production path. At realistic
shot budgets real QPUs lose most of the accuracy (52% at 1,024 shots, 76% at 16,384), and the
noisy IBM Brisbane digital twin shifts kernel values while keeping their ranking (correlation ~0.9).
Details are in the dashboard's Research and Quantum Lab pages.

## Repository layout

```
├── DRIPBL Project Master Context.md   full research context, decisions and milestones
└── dripbl-fog-server/                 the system
    ├── main.py … ai_judge.py          fog node: camera → motion/gait → encoder → PLS → QSVM → judge
    ├── api/                           FastAPI server (docs at /docs), notifications, calibration, Quantum Lab
    ├── dashboard/                     Streamlit command center
    ├── src/                           training, cross-dataset evaluation, deployment tools
    ├── models/MANIFEST.json           which trained models to download (they live in GitHub Releases)
    ├── reports/                       metrics and experiment logs behind every number above
    ├── tests/                         run_all.py runs every suite
    ├── deploy/                        Raspberry Pi systemd services + install script
    └── CLAUDE.md                      engineering guide: architecture, findings, gotchas
```

## Quick start

```bash
cd dripbl-fog-server
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python3 src/fetch_models.py    # downloads + verifies the trained models (needs `gh auth login`)
.venv/bin/python3 start.py --mock        # node + API + dashboard, replaying held-out retail clips
```

Open http://localhost:8501 for the dashboard and http://localhost:8000/docs for the API (add
`--api-port`/`--dash-port` if those ports are taken). Run `start.py` without `--mock` for the live camera,
and set the camera's **site profile** (CCTV or Retail) in Settings → Detection. On a Raspberry Pi,
`sudo -E bash deploy/install.sh` installs everything as services that start on boot.

**A new camera:** in Settings → Detection → *Calibrate to this camera*, learn its alert threshold from
normal activity. For the best accuracy, record labelled clips into `training-data/<site>/{normal,theft}/`
and run `.venv/bin/python3 -u src/add_camera.py <site>`.

## Tests

```bash
.venv/bin/python3 tests/run_all.py          # 7 suites, ~2 min, no datasets needed
.venv/bin/python3 tests/run_all.py --data   # + model accuracy on the held-out datasets
.venv/bin/python3 tests/run_all.py --soak   # + 30-minute leak/stall soak
```

They cover hostile camera frames, a camera that drops and reconnects, broken config files, concurrent
database access, ~1,000 junk API requests, sabotaging a running node (dead cameras, a deleted data folder,
corrupt model files, live model swaps), every dashboard page with the API up, down and empty, and 500
simultaneous live-video viewers.

## Data

The datasets (~13 GB) are not in this repository. Place them in `training-data/` next to
`dripbl-fog-server/`:

| Folder | Content |
|---|---|
| `dataset-1/{normal,shoplifting}/*.mp4` | retail footage; 36 clips are a fixed hold-out (`reports/dataset1_split.json`) |
| `dataset-2/videos/*.mp4` | 8 clips from a second store camera |
| `dataset-3/{Train,Test}/<class>/*.png` | UCF-Crime 64×64 frames, the CCTV benchmark |
| `dataset-4/` | UCF-Crime source videos and temporal labels |
| `<your-site>/{normal,theft}/*.mp4` | your own camera's clips, for `src/add_camera.py` |

## Working with git

Runtime files the system rewrites constantly (the trials DB, the live frame, logs, settings that may
hold secrets) and model binaries are ignored, so a running system never shows up in `git status`.
Models are published with `src/publish_models.py <tag>`. Commit when a piece of work is done:

```bash
git add -A && git commit -m "Describe the change" && git push
```
