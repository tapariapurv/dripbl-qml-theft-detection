# DRIPBL: Quantum-Kernel Theft Detection on a Cloud–Edge–Fog Pipeline

A research system that detects procedural theft in cash-handling environments. A camera feeds a fog
node (Raspberry Pi or laptop). The node gates on motion inside a region of interest and tracks gait. It
then compresses each frame with a CNN and supervised PLS into **4 qubit angles** and classifies them with
a **5-fold quantum-kernel SVM ensemble** (ZZFeatureMap fidelity kernel). An AI judge combines that with
the gait signals, and flagged events reach a human through the dashboard, notifications and GPIO alarms.

DRIBPL Research Program 2026 · Researcher: Purv Taparia · Coach: Mrs. Rachna Nath · Quantum advisor: Mr. Anirban Bhattacharya

## Results (unseen test videos)

| Metric | Value |
|---|---|
| Clip accuracy (headline, 194 test clips, zero overlap with training videos) | **87.1%** |
| Clip ROC AUC | 0.908 |
| Frame accuracy (20,473 frames) | 84.5% |
| Retail shoplifting dataset (5-fold CV by video) | 80.2% clip accuracy |
| Inference on the fog node | ~15 ms / frame |
| Quantum circuit on IBM Brisbane topology | 4 qubits, transpiled depth 108 |

Exact statevector simulation on the fog node is the production path. At realistic shot budgets, real
QPUs lose most of the accuracy (52% at 1,024 shots, 76% at 16,384). The full study is in
`dripbl-fog-server/reports/hardware_report.json` and on the dashboard's Research page.

## Repository layout

```
├── DRIPBL Project Master Context.md   full research context, decisions and milestones
└── dripbl-fog-server/                 the system
    ├── main.py … ai_judge.py          fog node: camera → motion/gait → CNN → PLS → QSVM → judge
    ├── store.py                       SQLite trial store shared by node and API
    ├── api/                           FastAPI server (docs at /docs), notifications, Quantum Lab
    ├── dashboard/                     Streamlit command center
    ├── src/                           offline training, evaluation and hardware studies
    ├── models/qsvm_model.joblib       the deployed 5-fold ensemble
    ├── reports/                       metrics and experiment logs behind every number above
    ├── deploy/                        Raspberry Pi systemd services + install script
    └── CLAUDE.md                      engineering guide: architecture, gotchas, commands
```

## Quick start

```bash
cd dripbl-fog-server
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python3 start.py --mock        # node + API + dashboard, replaying labelled clips
```

Open http://localhost:8501 for the dashboard and http://localhost:8000/docs for the API. Use
`start.py` without `--mock` for the live camera. On a Raspberry Pi, `sudo -E bash deploy/install.sh`
installs everything as services that start on boot.

Tests: `.venv/bin/python3 test_pipeline.py`, `tests/test_api.py`, and `tests/test_pipeline_e2e.py`
(the last one needs the datasets).

## Data

The datasets (~13 GB) are not in this repository. Place them in `training-data/` next to
`dripbl-fog-server/`:

| Folder | Content |
|---|---|
| `dataset-1/{normal,shoplifting}/*.mp4` | retail shoplifting footage, used by mock mode |
| `dataset-2/` | cross-source frames from a different camera |
| `dataset-3/{Train,Test}/<class>/*.png` | UCF-Crime 64×64 frames, the main benchmark |
| `dataset-4/` | UCF-Crime source videos and temporal labels |

## Working with git

Runtime files the system rewrites constantly (the trials DB, the live frame, logs, and settings JSON
that may hold secrets) are ignored, so a running system never shows up in `git status`. Commit when a
piece of work is done:

```bash
git add -A && git commit -m "Describe the change" && git push
```
