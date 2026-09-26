# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The fog/edge node of the DRIBPL Quantum Machine Learning (QML) theft-detection project. Runs on a Raspberry Pi/ARM edge device (single-threaded CPU torch, no GPU assumed). Not a package — a flat set of modules imported directly by `main.py`.

## Setup & commands

```bash
python3.11 -m venv .venv        # use 3.11 — pinned torch/opencv wheels don't support 3.14+
.venv/bin/pip install -r requirements.txt

.venv/bin/python3 test_pipeline.py   # smoke test (assert-based, no pytest) — run after any pipeline change
.venv/bin/python3 -u main.py         # run the node; -u avoids stdout buffering delays
```

There is no camera/network dependency required to test: no real camera source falls back to a synthetic generated feed, and the quantum classifier and AI judge run fully locally (no cloud/QPU required) unless `quantum_endpoint`/`alert_endpoint` are set in `runtime_config.json`.

For the dashboard and API, see **Command center** below.

## Architecture

Four-stage pipeline, one stage per module, orchestrated by `main.py`'s loop:

1. **`camera.py`** (`FrameGrabber`) — background thread keeps a single-slot "freshest frame" buffer so the main loop never reads stale/queued frames. `camera_source` in config can be a camera index, an RTSP/HTTP(S) URL, or a local file path — `cv2.VideoCapture` handles all three; file sources are additionally paced to their native FPS and looped. Falls back to a synthetic moving-frame generator if the source fails to open. `reconfigure()` lets the source be swapped live without restarting the process.

2. **`edge_pipeline.py`** (`EdgePipeline`) — motion + gait gate, then feature extraction:
   - `MotionDetector`: MOG2 background subtraction restricted to the configured ROI; returns whether a contour cleared the sensitivity-scaled area threshold, plus its centroid.
   - `GaitAnalyzer`: rolling window of centroids → walking speed, stride-oscillation frequency (zero-crossings of lateral sway), and a jitter score used downstream as a nervousness proxy. Call `EdgePipeline.observe()` every frame (not just on trigger) so this window stays continuous.
   - `LightweightCNN`: 3-block conv net (32/64/128 ch, BN, `AdaptiveAvgPool(2)`) → 128-dim latent, under `torch.inference_mode()`, single-threaded (`torch.set_num_threads(1)`). `in_channels` is 3 normally, 6 for the (rejected) motion-difference variant. Trained weights come from the offline pipeline below.
   - `PCAProjector`: vectorized NumPy `(1×128) @ (128×4)` projection scaled to `[-π, π]` — the 4 angles feed the ZZFeatureMap. This is the *untrained live* path; the deployed model uses the fitted `PCAAngleProjector` inside `models/qsvm_model.joblib` (see below). Falls back to seeded stand-ins (`pca_components.npy` / `cnn_weights.pt`) when no trained artifact is present.

3. **`theft_detector.py`** (`TheftDetector`) — **the deployed classifier.** Adapter around `src/inference.py`'s `LiveTheftClassifier`, loading the trained 5-fold QSVM ensemble from `models/qsvm_model.joblib` (0.871 clip / 0.845 frame accuracy, ~15 ms/frame in the node). It returns the same dict shape as `QuantumClassifier` plus `trained: True`, so the rest of the pipeline is unchanged. `main.py` prefers it and falls back to the prototype classifier below when no artifact is present. Note `AIJudge` weights a `trained` signal at 0.80 and only 0.50 otherwise — gait should not get a near-equal vote against a measured detector.

3b. **`quantum_classifier.py`** (`QuantumClassifier`) — *fallback only, not the deployed path.* Encodes the 4 PCA angles with a Qiskit `ZZFeatureMap` and scores class membership by Aer statevector fidelity against reference class prototypes (`person`, `money_object`, `background`, `suspicious_signature`, stored in `class_prototypes.json`, auto-created with seeded defaults). Falls back to a classical RBF-kernel stand-in if qiskit/qiskit-aer aren't installed, or delegates to a remote endpoint if `quantum_endpoint` is set (falling back locally on failure).

4. **`ai_judge.py`** (`AIJudge`) — combines the quantum classifier's crime-signal score with gait signals (nervousness, speed anomaly) into one `suspicion_score`, flagged against `Settings.flag_threshold` (sensitivity-derived).

5. **`network_alerter.py`** (`NetworkAlerter`) — every observed trial is logged to `trials.log` (JSON lines, audit trail) regardless of verdict; only flagged trials are dispatched to `alert_endpoint` (async via a thread pool, non-blocking) and trigger the GPIO buzzer/LED. GPIO is real (`RPi.GPIO`) on a Pi and auto-mocked (logs pin state) elsewhere.

### Configuration is the single edit surface

`config.py`'s `Settings` dataclass holds every tunable (video source, ROI, sensitivity, thresholds, model/prototype paths, endpoints, GPIO pins). `ConfigStore` persists it to `runtime_config.json`, created with defaults on first run, and `main.py` polls `reload_if_changed()` (by mtime) every second — editing that JSON file while the node is running hot-reloads settings into every stage, including swapping the video source, with no restart. `sensitivity` (0–1) is the single knob users are expected to tune; it derives both `area_threshold` (motion gate) and `flag_threshold` (AI judge) via properties on `Settings` — don't hardcode thresholds elsewhere, add new sensitivity-derived values as properties there.

### Payload shape

`NetworkAlerter.build_payload` assembles the full per-trial record: `trial_id`, `timestamp`, `features` (4 PCA angles), `quantum` (classifier output), `verdict` (AI judge output incl. gait), optional base64 JPEG thumbnail. This is both what's logged and what's POSTed on a flagged alert — extend it here if a new stage's output needs to reach the log or the human endpoint.

## Command center (API server + dashboard)

```bash
.venv/bin/python3 start.py                  # node + API (:8000) + dashboard (:8501); Ctrl-C stops all
.venv/bin/python3 start.py --mock           # research mode: replays labelled dataset-1 clips
.venv/bin/python3 start.py --api-port 8010 --dash-port 8600   # if the default ports are taken
.venv/bin/python3 tests/test_api.py         # API check (temp DB/config; no node needed)
sudo -E bash deploy/install.sh              # Pi / Linux: systemd services, start on boot
```

Three processes that share nothing but files in `data/`, so any one can crash or restart
without taking down the others:

- **Node (`main.py`)** writes every trial to `data/dripbl.db` (`store.py`, SQLite in WAL mode),
  the annotated live frame to `data/live.jpg` (written atomically at `live_frame_fps`), a JPEG per flagged
  trial to `data/snapshots/`, and a heartbeat row every `heartbeat_sec`. `payload["stage"]` plus the judge's
  `signals`/`breakdown` and the detector's `member_margins`/`latent` are what the Pipeline Inspector draws.
  Mock mode (`mode="mock"`) rotates shuffled `mock_dir/{normal,shoplifting}/*.mp4` clips through
  `FrameGrabber.reconfigure` and tags each trial with `ground_truth`.
- **API (`api/server.py`, FastAPI)** is the only data surface: the dashboard never reads files itself.
  Writes need `X-API-Key` (env `DRIPBL_API_KEY`, else generated into `data/api_key`). `PUT /api/config`
  validates, then writes `runtime_config.json` atomically, and the node hot-reloads it. `api/app_settings.py`
  owns `app_settings.json` (dashboard look, notifications, AI judge, quantum, API, retention); its
  `DEFAULTS` dict *is* the schema, so to add a setting, add it there. Secrets are masked on read.
  `api/worker.py` runs in the API process: optional Claude second opinion on flagged trials, then
  ntfy/webhook/email, then retention. It lives in the API rather than the node so network calls can
  never stall frame processing. `api/quantum_check.py` runs the fidelity kernel for recent trials on
  FakeBrisbane / IBM hardware in a background job.
- **Dashboard (`dashboard/app.py` + `dashboard/views/*`, Streamlit)**: live video is the API's
  MJPEG stream in a plain `<img>`, so frames never pass through Streamlit, and only
  `st.fragment(run_every=refresh_sec)` panels poll. That keeps it light enough for a Pi.
  All HTTP goes through `dashboard/lib.py`. Set `DRIPBL_DASHBOARD_PASSWORD` before exposing it on a network.

Live judgement and experts:
- The node judges **clips, not frames**: while there is motion it scores frames at `score_fps` and a
  trial is judged on the mean margin of the last `judge_window_sec` (`TheftDetector.classify`), which
  is how every accuracy number is measured. Motion models (6-channel) get the frame 0.33 s earlier
  from `TheftDetector.observe`'s history.
- `site_profile` picks the expert: `retail` -> `retail_model_path`, otherwise `model_path`; each
  artifact also carries per-site clip thresholds. Changing the profile or deploying a new model file
  (`src/deploy_model.py [--as retail]`) hot-swaps it; a corrupt file is rejected and the old model stays.
- `crime_signal` is calibrated so that at default sensitivity the judge flags exactly at the model's
  validated clip threshold (0.625 at the threshold x w_crime 0.8 = flag threshold 0.5).

Tests (`tests/run_all.py` runs them all; `--data` adds accuracy on the datasets, `--soak` a 30-min soak):
smoke, model self-check, API contract, robustness (odd frames, camera drop/reconnect, broken config,
concurrent DB, retention, ~1000 junk API requests), node sabotage in a live process (bad config, dead
cameras, deleted data dir, model hot-swap, SIGTERM), every dashboard page x {empty, populated, API down},
and 500 concurrent live-video viewers.

Gotchas:
- A dead camera must never create trials: the grabber shows a static NO SIGNAL frame (zero motion) and
  reconnects; the synthetic moving-square feed only runs when `camera_source` is "" or "synthetic".
  Streams are opened on the capture thread with 5 s FFmpeg timeouts (OpenCV serialises opens).
- `runtime_config.json` is the whole truth on reload: missing keys take defaults, invalid values keep the
  previous value (`config.apply_changes`), shared by the node and `PUT /api/config`.
- `/api/stream` must stay async: a sync generator holds a worker thread per viewer.
- Settings hot-reload rebuilds `MotionDetector` (it copies ROI/MOG2/sensitivity at init; without the rebuild,
  ROI and sensitivity edits never reached the motion gate).
- `FrameGrabber.read()` returns `None` when there is no new frame. Don't "fix" that: returning the last
  frame again made the loop re-classify duplicates at 100% CPU.
- Mock mode replays all of dataset-1, including the clips the retail expert trained on. For an honest
  retail number use `src/cross_data.py <model>` (fixed hold-out) rather than the mock scoreboard.

## Cross-dataset models (`src/cross_data.py`, `xd_experiments.py`, `train_xd.py`)

The original model scored 0.871 on UCF-Crime but chance on retail (0.472, recall 0). Findings, so they
are not re-litigated (all on fixed video-disjoint hold-outs; see `reports/xd_experiments.jsonl`):

| change | result | verdict |
|---|---|---|
| one model trained jointly on all datasets | UCF 0.82-0.87, retail AUC 0.82-0.86 | rejected: UCF's 20k frames pull the encoder away from retail |
| retail trained alone | retail AUC 0.895 vs 0.843 jointly | **kept**: one expert per camera domain (`site_profile`) |
| frozen Kinetics R3D-18 video model | retail AUC 0.82, 80x the CPU | rejected |
| 128 px instead of 64 px retail frames | no gain | rejected |
| top-k clip aggregation, flip TTA | within noise | rejected |
| ResNet-18 CCTV expert (UCF only, 5-fold) | UCF Test **0.897** (was 0.871), 6 FP vs 10 | **deployed** as default |
| transfer: UCF-trained encoder -> fine-tune on retail | retail OOF AUC 0.914 vs 0.879 from scratch | **kept** |
| ensemble of retail transfer models (combine_experts.py) | retail OOF AUC 0.9245 | superseded below |
| + 855 Kaggle retail CCTV videos (`training-data/omar`, omarelg/shoplifting-videos-dataset) as an extra site | tiny: d1 hold-out 0.806 (unchanged), omar hold-out **1.000** (old model 0.678, recall 0); r18 and r18+tiny combo no better | **deployed** `retail_tiny_m_ft_omar`: 2 MB vs 230 MB |
| zero-shot on an unseen camera (dataset-2, all 8 videos) | every clip alerts; normal/theft pairs score alike | not solved: scores track the scene. Use `add_camera.py` |

Rules that keep these numbers honest: hold-outs are scored once, after all choices; models are chosen by
out-of-fold AUC (within 0.01, the cheaper one); per-site thresholds come from that site's out-of-fold clips.

New camera: put labelled clips in `training-data/<site>/{normal,theft}/*.mp4` and run
`src/add_camera.py <site>`: it holds out 20% of the site's videos, fine-tunes from the deployed retail expert
with retail replay (no forgetting), sets the site threshold on out-of-fold clips and reports both hold-outs.

Models are too large for git: `src/deploy_model.py` installs (float16, drift-checked, backup kept),
`src/publish_models.py <tag>` uploads to a GitHub Release + writes `models/MANIFEST.json`,
`src/fetch_models.py` downloads and verifies (install.sh runs it).

## Offline QSVM training pipeline (`src/`, `models/`, `reports/`)

Trains the quantum-kernel SVM that the live node deploys. Separate from the
real-time loop above; everything here is offline except `src/inference.py`.

```bash
.venv/bin/python3 -u src/train_qsvm.py           # single end-to-end training run
.venv/bin/python3 -u src/experiments.py          # autonomous config search (hours, detached)
.venv/bin/python3 src/evaluate.py testset        # metrics on dataset-3/Test
.venv/bin/python3 src/evaluate.py clips          # clip-level verdicts (the headline metric)
.venv/bin/python3 src/evaluate.py bench          # inference latency
.venv/bin/python3 tests/test_pipeline_e2e.py     # smoke + accuracy floor, run after training
```

### Data

`../training-data/dataset-3/{Train,Test}`, one folder per class (`NormalVideos`,
`Burglary`, `Robbery`, `Shoplifting`, `Stealing`), collapsed to binary
Normal/Anomaly. Frames are pre-extracted 64x64 PNGs named
`<video>_<frameindex>.png` — the video id is the filename minus the last `_N`.

**Train and Test share zero source videos** (verified). That makes this a real
generalisation benchmark and it is why published ~99% frame-level numbers on
this Kaggle dump are not comparable: those evaluate on a random split *of
Train*, where near-duplicate frames of the same clip sit on both sides.
Any validation split made here must therefore be **grouped by video id**
(`sample_by_video(..., video_subset="fit"|"val")`); a frame-level holdout
reports ~0.96 that collapses to ~0.70 on unseen clips.

### Stages

1. **`src/cnn_extractor.py`** — batch-embeds frames through `LightweightCNN`
   (shared with `edge_pipeline.py`) to a 128-d latent.
2. **`src/pca_projector.py`** (`PCAAngleProjector`) — 128-d -> 4 angles in
   `[-pi, pi]`. Two decisions here matter more than anything downstream:
   - `mode="pls"` (supervised) beats `mode="pca"`: PCA keeps high-variance
     directions, which for a CNN latent are brightness/texture, not the label.
   - `angle_scale` bounds the encoding range. Spreading features over the full
     `[-pi, pi]` makes the periodic ZZFeatureMap oscillate until the Gram matrix
     is nearly the identity and the SVM cannot generalise. It is tuned on
     validation; small values (0.15-0.3) win.
3. **`src/quantum_kernel.py`** — `ZZFeatureMap(4, reps=2, linear)` fidelity
   kernel. `encode()` uses a closed-form vectorised statevector
   (`D(x)H D(x)H|0>`), **227x faster than per-sample circuit simulation and
   equal to it within 4e-15** — `verify_fast_encode()` asserts that against
   qiskit, and `encode_qiskit()`/`build_noisy_kernel()` keep the reference and
   the FakeBrisbane-noise path available.
4. **`src/train_qsvm.py`** — trains the CNN, fits PLS+SVC, tunes angle scale, C
   and the clip threshold on video-disjoint validation, evaluates on Test.
5. **`src/inference.py`** (`LiveTheftClassifier`) — `predict_frame()` for one
   frame, `predict_clip()` for a sequence (applies the trained aggregation and
   threshold). Handles ensemble artifacts (mean margin over members).

### Frame-level vs clip-level

A crime clip is mostly ordinary footage around a short criminal act, so
per-frame labels are noisy and single-frame accuracy is capped. The headline
metric is **clip-level**: aggregate frame margins per video (aggregation and
threshold both chosen on validation only). This also matches deployment — the
node watches a stream, not one still.

### What has been tried (`reports/experiments.jsonl`)

`src/experiments.py` runs configs unattended, logs one JSON line each, and saves
the best artifact. Findings, so results are not re-litigated:

| change | clip AUC | verdict |
|---|---|---|
| PCA -> PLS for the 4-d bottleneck | 0.887 | **kept** (+2.6 pts accuracy) |
| 3-seed ensemble (mean margin) | **0.912** | **kept** |
| plateau-midpoint threshold | same model | **kept** (+1.6 pts) |
| RGB + inter-frame difference (motion) | 0.875 | rejected — frames are 10 apart (~0.33s), so the difference is scene drift |
| 45 frames/clip, 16 epochs | 0.869 | rejected — extra capacity memorises scenes |
| MIL ranking (Sultani-style) | 0.646 | rejected — ~5x fewer optimiser steps from scratch; needs precomputed features |
| clip-descriptor QSVM (`clip_qsvm`) | 0.875 | rejected for deployment — cannot score a single frame |
| frozen ImageNet (MobileNetV3/ResNet18) | 0.711 | rejected — built for 224x224, collapses to a 2x2 map at 64x64 |
| mix dataset-2 in (`--with-dataset2`) | **0.871** vs 0.811 | **kept** — measured at matched cost; also fixed cross-source transfer (2/2 holdout clips vs 0/2 before) |
| dataset-4 temporal masks (`--temporal-labels`) | 0.841 vs 0.871 | rejected — see below |

### The other datasets

- **dataset-2** — 8 clips from a different camera/source. Small, but the only
  cross-source signal available; a dataset-3-only model scores AUC 0.56 (chance)
  on it, proving it learned UCF-Crime scene appearance rather than theft.
  Mixing it in helps both numbers. Kept.
- **dataset-4** — UCF-Crime **source video**, not new footage: all 244 clips are
  already in dataset-3 (220 Train, **24 Test** — training on its `.mp4`s would
  leak the test set). Its value is `Labels/*.csv`: 32 uniform temporal bins per
  video with a binary anomaly flag, mapped onto dataset-3 frame names by
  `src/temporal_labels.py` (60,714 crime frames covered).
  The masks are genuine — 1.43 contiguous anomalous blocks per video vs 5.99
  when shuffled, 70% a single interval — but using them as Train labels *lowered*
  AUC to 0.841. `src/check_mask_alignment.py` explains why: the model gives every
  frame of a clip a near-identical margin (within-clip agreement 47%, chance), so
  it has no temporal resolution to exploit them. Dropping 29% of positives just
  shrinks the class it actually relies on. Kept behind a flag, off by default.
  **This is the clearest statement yet of the ceiling: 64x64 single frames carry
  scene, not act.** Temporal modelling (frame stacks/optical flow over the real
  video in dataset-4) is the only untried lever with real headroom.

Five structurally different methods converge on AUC 0.87-0.91, and the
oracle-threshold diagnostic (best achievable accuracy if the cut were placed
perfectly) sits ~1 point above what is achieved, so calibration is close to
exhausted: **only a better representation moves the number**, not more tuning.

### Gotchas worth keeping

- The frame-level Gram matrix is O(n^2): fitting on all ~29k frames needs ~7 GB.
  Subsample (`svm_fit_per_label`) — that bug OOM-killed a run.
- Artifacts pickle `PCAAngleProjector` by module name, so `src/` must be on
  `sys.path` to `joblib.load` them (`inference.py` handles this).
- `experiments.py` seeds its best-so-far from `experiments.jsonl`; without that
  each new process overwrote a better model saved by an earlier wave.
- An ensemble artifact must store **every** member, or the deployed model is not
  the model that was measured.

## Ground rule: never read the training data yourself

**Do not open image or video files from `../training-data/` with Read/cat/etc.**
It burns tokens for no benefit. List filenames and directories to plan, and
grep filenames for labels; pixels are only ever read inside the scripts above,
at their own runtime.
