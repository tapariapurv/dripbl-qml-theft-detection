# DRIBPL Research Project — Master Context & Technical Blueprint

**Project Title:** Multi-Stage Cloud-Edge-Fog Quantum Machine Learning (QML) Pipeline for Procedural Theft and Anomaly Detection in Cash-Handling Environments  
**Primary Researcher:** Purv Taparia (Grade 5, Chatrabhuj Narsee School, Kandivali, Mumbai, India)  
**Program:** DRIBPL Research Program (DRIPBL STEAM Analytics LLC, June 12 – November 30, 2026\)  
**Academic / Research Coach:** Mrs. Rachna Nath (Founder & CEO, DRIPBL STEAM Analytics LLC | [rachnanath@gmail.com](mailto:rachnanath@gmail.com) / [rachna.nath@dripbl.com](mailto:rachna.nath@dripbl.com))  
**Industry Mentor & Quantum Advisor:** Mr. Anirban Bhattacharya (Quantum Computing Expert | [anirbanbhattacharya1234@yahoo.co.in](mailto:anirbanbhattacharya1234@yahoo.co.in))  
**Target Completion / Submission Milestone:** September 15, 2026  
**Primary Repository on Fog Node:** `dripbl-fog-server`  
**GitHub Repositories:** [tapariapurv/dribpl\_qml\_pipeline\_v2](https://github.com/tapariapurv/dribpl_qml_pipeline_v2) | [Purv Taparia Portfolio](https://tapariapurv.github.io)

---

## 1\. Executive Summary & Purpose for Claude Code

This master document is designed as the definitive, zero-loss context repository for Claude Code (and engineering contributors) operating on the **DRIBPL Quantum Machine Learning (QML) Procedural Theft Detection Project**.

The goal of this research is to solve procedural theft, cash register skimming, sweethearting, and internal trust violations in cash-intensive retail and banking environments. The system operates as an end-to-end, multi-stage **Cloud-Edge-Fog pipeline**:

1. **Edge/Sensor Layer:** High-definition video camera captures point-of-sale (POS) interactions.  
2. **Fog Server Layer (`dripbl-fog-server`):** A local edge node (Raspberry Pi 4 / ARM device or local workstation) performs Region of Interest (ROI) motion gating, computer vision feature extraction via a single-threaded PyTorch Convolutional Neural Network (CNN), gait/nervousness tracking, dimensionality reduction (supervised PLS / PCA down to 4 angles), a deployed 5-fold ensemble Quantum Support Vector Machine (QSVM), multi-modal suspicion arbitration via an AI Judge, and hardware/network alert dispatch.  
3. **Cloud / Quantum Simulation Layer:** High-performance quantum kernel evaluations and noisy digital twin simulations using Qiskit Aer (`AerSimulator.from_backend(FakeBrisbane)`), reproducing the 127-qubit heavy-hex topology, native basis gates (`CZ`, `SX`, `X`, `RZ`), and full decoherence/gate error profiles of real IBM Quantum hardware.

---

## 2\. Research Hypothesis & Problem Statement

### 2.1 Formal Hypothesis (Log Book Formulation)

> *"If Partial Least Squares (PLS) / Principal Component Analysis (PCA) dimensionality reduction and a lightweight Convolutional Neural Network (CNN) pre-filtering isolate cash-drawer and procedural events on an edge/fog server, then a Quantum Support Vector Machine (QSVM) operating on cloud quantum backends/simulators will detect procedural theft anomalies with high fidelity and low false alarm rates, because dimensionality reduction down to 4 qubits mitigates NISQ hardware noise and exponential concentration."*

### 2.2 The Real-World Problem

Internal retail theft and cashier procedural violations account for tens of billions of dollars in annual shrinkage worldwide:

- **Cash Drawer Anomalies:** Unrecorded drawer openings, physical tampering, unauthorized skimming during routine shifts.  
- **Procedural Evasions (Sweethearting):** Passing high-value merchandise across checkout scanners without ringing up transactions, or faking manual cash transactions.  
- **Physical & Behavioral Indicators:** Erratic physical movement, pacing, abnormal gait jitter, and rapid pacing during criminal acts.

### 2.3 Why Quantum Machine Learning (QML)?

- **Feature Space Mapping:** High-dimensional quantum Hilbert spaces allow non-linear classification boundaries via `ZZFeatureMap` fidelity kernels: \$\$K(x\_i, x\_j) \= |\\langle\\phi(x\_i)|\\phi(x\_j)\\rangle|^2\$\$  
- **NISQ Constraint Management:** Current Noisy Intermediate-Scale Quantum (NISQ) devices suffer from high gate error rates and decoherence. Mentored by Mr. Anirban Bhattacharya, the project established that deep quantum circuits (like Quantum Graph Neural Networks \- QGNN) accumulate fatal circuit noise. By aggressively compressing 128-dimensional latent representations down to 4 qubits using supervised PLS and scaled angle projections, the quantum circuit depth is bounded to 3 (transpiled depth 108), making quantum classification practical on NISQ hardware.

---

## 3\. System Architecture: Cloud-Edge-Fog

\+-----------------------------------------------------------------------------------------+

|                                    EDGE / CAMERA LAYER                                  |

|   \+-------------------+      RTSP / USB Video Stream (640x480 @ 15-30 FPS)              |

|   |  HD Camera (POS)  | \===================================================+           |

\+---+-------------------+----------------------------------------------------|------------+

                                                                             |

\+----------------------------------------------------------------------------v------------+

|                          FOG SERVER NODE (dripbl-fog-server / Raspberry Pi)             |

|                                                                                         |

|   \+---------------------------------------------------------------------------------+   |

|   | 1\. FrameGrabber (camera.py)                                                     |   |

|   |    Threaded capture; 1-slot freshest frame buffer (zero frame queuing lag).     |   |

|   |    Paced replay for local test files; auto-fallback to synthetic frame generator.|  |

|   \+----------------------------------------+----------------------------------------+   |

|                                            |                                            |

|   \+----------------------------------------v----------------------------------------+   |

|   | 2\. Edge Pipeline & Feature Gating (edge\_pipeline.py)                            |   |

|   |    a. MotionDetector: MOG2 background subtractor on drawer ROI \[140,60,360,360\] |   |

|   |    b. GaitAnalyzer: Rolling window of centroids \-\> speed\_px\_s, stride\_freq\_hz,   |   |

|   |       and jitter\_score (direction variance proxy for nervousness).             |   |

|   |    c. LightweightCNN: 3-block Conv-BN-ReLU-Pool (128-d latent vector).          |   |

|   |    d. PCAProjector: (1x128) @ (128x4) \-\> \[-pi, pi\] angle projection.            |   |

|   \+----------------------------------------+----------------------------------------+   |

|                                            |                                            |

|   \+----------------------------------------v----------------------------------------+   |

|   | 3\. Deployed Classifier Stage (theft\_detector.py & inference.py)                 |   |

|   |    LiveTheftClassifier: Loads 5-fold ensemble QSVM artifact (qsvm\_model.joblib). |   |

|   |    \- Evaluates exact statevector fidelity kernel against support vectors.       |   |

|   |    \- Inference latency: \~15 ms / frame on CPU.                                  |   |

|   |    \- Fallback: quantum\_classifier.py (Aer statevector vs class prototypes).     |   |

|   \+----------------------------------------+----------------------------------------+   |

|                                            |                                            |

|   \+----------------------------------------v----------------------------------------+   |

|   | 4\. AI Judge (ai\_judge.py)                                                       |   |

|   |    Multi-modal arbitration:                                                     |   |

|   |    suspicion \= 0.80 \* crime\_signal \+ 0.13 \* nervousness \+ 0.07 \* speed\_anomaly     |   |

|   |    Gated against dynamic threshold: suspicion \>= flag\_threshold                 |   |

|   \+----------------------------------------+----------------------------------------+   |

|                                            |                                            |

|   \+----------------------------------------v----------------------------------------+   |

|   | 5\. Network Alerter & Hardware Dispatch (network\_alerter.py)                      |   |

|   |    \- Audit Logging: All trials written to trials.log (JSONL).                   |   |

|   |    \- Async HTTP POST: Flagged events sent to alert\_endpoint with b64 thumbnail. |   |

|   |    \- GPIO Hardware Alerts: Pin 17 (Buzzer) & Pin 27 (LED) (RPi.GPIO or mock).   |   |

|   \+---------------------------------------------------------------------------------+   |

|                                                                                         |

|   \+---------------------------------------------------------------------------------+   |

|   | Configuration Store (config.py & runtime\_config.json)                           |   |

|   | Hot-reloads all settings every 1.0s without process restart.                    |   |

|   | Single master tuning knob: sensitivity (0.0 to 1.0).                            |   |

|   \+---------------------------------------------------------------------------------+   |

\+-----------------------------------------------------------------------------------------+

                                             |

\+--------------------------------------------v--------------------------------------------+

|                          CLOUD / QUANTUM BACKEND LAYER                                  |

|   \+---------------------------------------------------------------------------------+   |

|   | Qiskit Aer Simulator (Digital Twin of IBM Brisbane \- 127 Qubits)                |   |

|   | \- ZZFeatureMap(4 qubits, reps=2, linear entanglement)                           |   |

|   | \- Fast vectorised analytic encoder (fast\_encode): 227x speedup over circuits.   |   |

|   | \- Real hardware backend pass manager transpilation & shot noise modeling.       |   |

|   | \- Cloud QML Endpoint (AWS BraKet / IBM Quantum Runtime API).                    |   |

|   \+---------------------------------------------------------------------------------+   |

\+-----------------------------------------------------------------------------------------+

---

## 4\. Codebase Ecosystem & Repository Breakdown

The DRIBPL project has evolved across three key workspaces and repositories located in Google Drive:

### 4.1 Primary Repository: `dripbl-fog-server`

*Location in Drive:* `actual-dripbl-project/dripbl-fog-server/` (Folder ID: `1jL4pvADwIn8youswqiwLGvtq-kS6b500`)  
This is the **operational production repository** running the live edge/fog node and hosting the offline training, benchmarking, and experiment framework.

#### Core Root Modules:

- **`main.py`**: The live loop of the fog node. Initializes all components, reads frames from `FrameGrabber`, evaluates `EdgePipeline.observe()`, runs `TheftDetector` or fallback `QuantumClassifier`, queries `AIJudge.judge()`, logs audit trials, dispatches alerts via `NetworkAlerter`, and hot-reloads configuration changes every second.  
- **`config.py`**: Declares the `Settings` dataclass and `ConfigStore` class. `Settings` contains all tunable hyperparameters (video sources, ROI, debounce, sensitivity, model paths, alert endpoints, GPIO pins). Defines derived properties:  
  - `area_threshold`: Motion contour threshold scaled by sensitivity (\$500 \\times (1.5 \- sensitivity)\$).  
  - `flag_threshold`: AI judge suspicion cutoff scaled by sensitivity (\$0.85 \- 0.45 \\times sensitivity\$).  
- **`runtime_config.json`**: JSON-serialized settings file polled every 1.0 second by `ConfigStore.reload_if_changed()`. Modifying this file on disk hot-reloads the live node without dropping frames or restarting.  
- **`camera.py`**: Implements `FrameGrabber`. Spawns a daemon thread continuously reading frames into a 1-slot buffer (`self._frame`), eliminating OpenCV frame queue latency. Supports local video files (paced to original FPS), RTSP/HTTP streams, USB webcams, and synthetic test frames.  
- **`edge_pipeline.py`**:  
  - `MotionDetector`: Background subtraction using OpenCV `createBackgroundSubtractorMOG2(history=300, varThreshold=32.0, detectShadows=False)`. Crops to configured ROI.  
  - `GaitAnalyzer`: Deque tracking \$(t, x, y)\$ centroids over a 20-frame rolling window. Calculates:  
    - `speed_px_s`: Average pixel velocity.  
    - `stride_freq_hz`: Frequency of lateral sway sign-crossings.  
    - `jitter_score`: Standard deviation of frame-to-frame heading angles normalized in \$\[0, 1\]\$ (erratic nervousness proxy).  
  - `LightweightCNN`: 3-stage PyTorch CNN (`latent_dim=128`, `in_channels=3`). Single-threaded CPU execution.  
  - `PCAProjector`: Vectorized linear projection \$(1 \\times 128\) @ (128 \\times 4)\$ scaled to \$\[-\\pi, \\pi\]\$.  
- **`theft_detector.py`**: The deployed production adapter. Wraps `src/inference.py`'s `LiveTheftClassifier`. Loads `models/qsvm_model.joblib`. Mirrors `QuantumClassifier` output structure while setting `trained: True`.  
- **`quantum_classifier.py`**: Fallback prototype classifier. Encodes 4 PCA angles using Qiskit `ZZFeatureMap` and measures statevector fidelity against reference prototypes in `class_prototypes.json` (`person`, `money_object`, `background`, `suspicious_signature`).  
- **`ai_judge.py`**: Combines QSVM crime signals with gait metrics. Employs asymmetric weighting:  
  - Trained model: \$0.80 \\times crime\_signal \+ 0.13 \\times nervousness \+ 0.07 \\times speed\_anomaly\$.  
  - Prototype fallback: \$0.50 \\times crime\_signal \+ 0.25 \\times nervousness \+ 0.15 \\times speed\_anomaly \+ 0.10 \\times person\_nervous\_bonus\$.  
- **`network_alerter.py`**: Asynchronous worker thread pool (2 workers). Logs all trials to `trials.log`. Dispatches HTTP POST requests to `alert_endpoint`. Toggles physical GPIO buzzer (BCM 17\) and LED (BCM 27\) via `RPi.GPIO` or fallback `_MockGPIO`.  
- **`test_pipeline.py`**: Smoke test validating all modules, fallback flows, synthetic video generation, and hot-reloading without external hardware dependencies.

#### Offline Training & Experiment Suite (`src/`):

- **`src/train_qsvm.py`**: End-to-end training pipeline.  
  - Supervised training of `LightweightCNN` on video-disjoint Train split.  
  - Fitting `PCAAngleProjector` (PLS regression or PCA).  
  - Grid search over SVM parameter \$C \\in {1.0, 4.0, 16.0, 64.0, 256.0}\$ and `angle_scale` \$\\in {0.15, 0.3, 0.6, 1.2, \\pi}\$.  
  - Serializes model artifact to `models/qsvm_model.joblib`.  
- **`src/quantum_kernel.py`**:  
  - `FEATURE_MAP`: 4-qubit `ZZFeatureMap(reps=2, linear)`.  
  - `fast_encode(X)`: Vectorized analytic statevector formulation (\$D(x) H D(x) H |0\\rangle\$). Executes **227x faster** than circuit simulation with identical precision (\$\< 4 \\times 10^{-15}\$ max delta vs Qiskit).  
  - `kernel_matrix(A, B)`: Batch matrix multiplication \$|A B^\\dagger|^2\$.  
  - `build_noisy_kernel()`: Real hardware-faithful shots kernel backed by `FakeBrisbane`.  
- **`src/pca_projector.py`**: Implements `PCAAngleProjector`. Supports supervised `mode="pls"` (Partial Least Squares) and unsupervised `mode="pca"`. Supervised PLS aligns the 4 dimensions with label covariance, yielding significant gains over plain PCA. Applies robust 1st–99th percentile scaling.  
- **`src/cnn_extractor.py`**: Efficient batch inference for frame embedding.  
- **`src/inference.py`**: `LiveTheftClassifier`. Implements `predict_frame()` (\$\<15\$ ms single-frame inference) and `predict_clip()` (aggregates frame margins across full video sequences).  
- **`src/experiments.py`**: Autonomous experimental search runner. Evaluates architectures unattended, appends JSON lines to `reports/experiments.jsonl`, and updates the champion model artifact.  
- **`src/hardware_report.py`**: Benchmarks quantum circuit transpilation and finite shot-noise degradation against `FakeBrisbane`.  
- **`src/shoplifting_pipeline.py`**: 5-fold cross-validation pipeline on `dataset-1` (real 640x480 retail video footage).  
- **`src/temporal_labels.py` & `src/check_mask_alignment.py`**: Analyzes 32-bin temporal annotations from `dataset-4`. Confirms single 64x64 frames capture static scene appearance rather than temporal criminal acts.

---

### 4.2 Legacy & Dashboard Architecture: `dripbl-dashboard-architecture`

*Location in Drive:* `DRIBPL Research/dripbl-dashboard-architecture/` (Folder ID: `11JB9pdS-O0tf3IPzY-ZCtocVoOUi6iS5`)  
Contains the Streamlit dark-mode operations dashboard and the initial modular prototype of the fog node:

- **`dripbl_fog/dashboard/dashboard.py`**: Streamlit web dashboard running on port 8501:  
  - Real-time video monitor with overlaid motion ROI bounding box.  
  - Live incident audit feed with color-coded classification tags (Normal vs Anomaly).  
  - Dynamic charts displaying PCA explained variance ratios and feature distributions.  
  - Cloud QML bridge health status and ping monitor.  
- **`dripbl_fog/orchestrator.py`**: Threaded orchestrator connecting `CameraEventDetector`, `FeaturePCAPipeline`, and `CloudQMLBridge`.  
- **`dripbl_fog/cloud/cloud_bridge.py`**: Client for dispatching feature vectors to remote cloud endpoints with local fallback simulation.

---

### 4.3 Original Benchmark Pipeline: `dribpl_qml_pipeline`

*Location in Drive:* `DRIBPL Research/dribpl_qml_pipeline/` (Folder ID: `1vxxrUqpnwp_K1RH2UoKMBAY9Tl-ILMnW`)  
*GitHub:* [tapariapurv/dribpl\_qml\_pipeline\_v2](https://github.com/tapariapurv/dribpl_qml_pipeline_v2)  
The foundational research codebase developed in July/August 2026\. Contains the three-way benchmark:

1. **Ideal Quantum Kernel:** Noiseless statevector simulation.  
2. **Noisy Quantum Kernel:** Qiskit Aer simulation using `FakeBrisbane` noise models.  
3. **Classical RBF SVM:** Baseline Scikit-Learn radial basis function support vector classifier. Key module: `src/noise_builder.py` implementing `AerSimulator.from_backend(FakeBrisbane)` to capture complete hardware topology, basis gates, and error models.

---

## 5\. Datasets, Benchmarking & Experimental Results

### 5.1 Dataset Specifications

| Dataset | Source / Content | Resolution / Format | Size / Split | Purpose & Key Finding |
| :---- | :---- | :---- | :---- | :---- |
| **`dataset-3`** | UCF-Crime subset: Burglary, Robbery, Shoplifting, Stealing, Normal | 64x64 PNG stills (sampled every 10 frames) | **Train:** 34,635 frames (1,156 clips) **Test:** 20,473 frames (194 clips) | **Primary Benchmark.** Train and Test share **ZERO** source videos. Crucial: Random frame splitting causes scene leakage (false 96% accuracy that drops to 70% on unseen clips). |
| **`dataset-1`** | Real retail shoplifting surveillance footage | 640x480 MP4 videos @ 30 FPS | 182 videos (90 normal, 92 theft), 10,837 frames | Genuinely temporal dataset. Evaluated via 5-fold CV by video; achieves **80.22% clip accuracy** (AUC 0.837). |
| **`dataset-2`** | Distinct camera/store footage | Varied surveillance stills | 8 clips (326 frames) | Cross-source transfer benchmark. Training on dataset-3 alone scored chance (0.56 AUC); mixing dataset-2 in raised holdout accuracy to 100% (2/2 clips). |
| **`dataset-4`** | UCF-Crime full source videos \+ temporal CSV masks | Full MP4 clips \+ 32-bin CSV annotations | 244 videos (220 train, 24 test), 60,714 crime frames | Evaluated in `check_mask_alignment.py`. Model gave identical margins across all frames of a clip (47.1% agreement, chance), proving 64x64 single frames capture scene appearance, not the criminal act. |

---

### 5.2 Benchmark Performance Matrix (`metrics.json` & `reports/`)

The deployed champion model is a **5-fold video-disjoint ensemble trained for 10 epochs** using supervised PLS projection, angle scale \$0.15–0.3\$, \$C=1.0–4.0\$, and mean clip margin aggregation.

#### Test Set Performance (194 Unseen Test Clips, 20,473 Frames):

- **Clip-Level Accuracy (Headline Metric):** **87.11%** (169 / 194 clips correct)  
- **Clip ROC AUC:** **0.9076**  
- **Balanced Accuracy:** **79.62%**  
- **Theft Detection Recall:** **65.91%** (29 / 44 crime clips flagged)  
- **Theft Detection Precision:** **74.36%** (29 / 39 alerts true positive)  
- **False Negative Rate:** **34.09%** (15 / 44 crime clips missed)  
- **False Positive Count:** **10 / 150** normal clips (FP rate: 6.67%)  
- **Frame-Level Accuracy:** **84.53%** across all 20,473 frames  
- **Oracle-Threshold Ceiling:** **88.14%** (indicates threshold calibration is near-optimal)  
- **Single-Frame Inference Latency:** **\~15 ms** on edge CPU

#### Confusion Matrix (Test Clips):

                  Predicted Normal    Predicted Anomaly

Actual Normal           140                  10

Actual Anomaly           15                  29

---

### 5.3 Architectural Decisions & Ablation Study (`reports/experiments.jsonl`)

| Technique / Variation | Clip AUC | Status | Key Engineering Takeaway |
| :---- | :---- | :---- | :---- |
| **Supervised PLS (4-d bottleneck)** | **0.887** | **Adopted** | Supervised PLS preserves class covariance; plain PCA discards class signal for scene brightness (+2.6% accuracy). |
| **3-to-5 Seed Ensemble** | **0.912** | **Adopted** | Ensembling decision margins across seeds reduces variance without increasing circuit depth. |
| **Plateau-Midpoint Thresholding** | — | **Adopted** | Calibrating the decision threshold at the midpoint of validation accuracy plateau adds \+1.6% accuracy. |
| **Dataset-2 Mixing (`--with-dataset2`)** | **0.871 vs 0.811** | **Adopted** | Resolves cross-source overfitting; holdout clip accuracy improved from 0/2 to 2/2. |
| **Inter-Frame Difference (RGB \+ Motion)** | 0.875 | *Rejected* | Frames were sampled 10 apart (\~0.33s); motion difference captured background drift rather than action. |
| **Longer Training (45 frames, 16 epochs)** | 0.869 | *Rejected* | Extra network capacity led to scene memorization on 64x64 inputs. |
| **Sultani Multiple Instance Learning (MIL)** | 0.646 | *Rejected* | Failed when trained from scratch; requires pre-extracted 3D CNN features. |
| **Pretrained ImageNet (MobileNetV3 / ResNet)** | 0.711 / 0.732 | *Rejected* | Designed for 224x224; feature representations collapsed to 2x2 maps at 64x64 resolution. |
| **Temporal Labels (`dataset-4` masks)** | 0.841 | *Rejected* | Model lacks within-clip temporal resolution; filtering quiet frames merely reduced training volume. |

---

## 6\. Quantum Hardware Feasibility & Noise Study (`hardware_report.json`)

To evaluate deployment feasibility on physical quantum hardware, the 4-qubit `ZZFeatureMap` kernel was analyzed against IBM Quantum's 127-qubit `FakeBrisbane` device:

### 6.1 Circuit Complexity on Physical Topology

- **Logical Qubits:** 4  
- **Abstract Circuit Depth:** 3  
- **Target Backend:** `FakeBrisbane` (127 qubits, heavy-hex coupling map)  
- **Transpiled Circuit Depth:** **108**  
- **Two-Qubit Gates (`CZ`, `ECR`):** **22**  
- **Total Physical Gates:** **249**

### 6.2 Computational Cost per Frame

- **Ensemble Members:** 5  
- **Support Vectors per Member:** `[565, 789, 618, 879, 1037]`  
- **Total Quantum Circuits per Classified Frame:** **3,888 circuits**

### 6.3 Shot-Noise & Finite Sampling Degradation

In physical quantum execution, kernel matrix elements are estimated via finite measurement shots (\$s\$ shots \$\\rightarrow \\text{Binomial}(s, K)/s\$). The impact on clip classification accuracy is severe:

| Measurement Sampling | Kernel Mean Absolute Error | Clip Accuracy | ROC AUC | Practical Status |
| :---- | :---- | :---- | :---- | :---- |
| **Exact Analytic / Statevector** | **0.0000** | **86.60%** | **0.9048** | **Production Deployed on Fog Node** |
| **16,384 Shots** | 0.0022 | 75.77% | 0.7585 | Acceptable NISQ accuracy floor |
| **4,096 Shots** | 0.0046 | 64.43% | 0.6712 | Noticeable classification degradation |
| **1,024 Shots** | 0.0093 | 52.06% | 0.5208 | Borderline random guessing |
| **256 Shots** | 0.0178 | 50.00% | 0.5002 | Complete collapse to random chance |

**Scientific Takeaway:** Running 3,888 circuits per frame at \$\\ge 16,384\$ shots on physical QPUs is economically and latency-prohibitive for 30 FPS edge processing. Therefore, **vectorized statevector simulation on the fog node is the optimal production deployment**, while cloud QPU execution serves as periodic batch verification.

---

## 7\. Operational Guidelines for Claude Code

When developing, refactoring, or running code in this project, Claude Code MUST adhere to the following rules:

### 7.1 Python Environment & Tooling

- **Python Version:** **Python 3.11** (`python3.11 -m venv .venv`).  
  *Warning:* Do NOT use Python 3.14+; pre-compiled wheels for PyTorch and OpenCV headless do not support it.  
- **Dependency Installation:**  
    
  source .venv/bin/activate  
    
  pip install \-r requirements.txt  
    
- **Key Pinned Dependencies:**  
  - `torch==2.3.1` (Single-threaded CPU mode)  
  - `opencv-python-headless==4.10.0.84`  
  - `qiskit==2.5.2`  
  - `qiskit-aer==0.17.2`  
  - `qiskit-machine-learning==0.9.1`  
  - `qiskit-ibm-runtime==0.45.1`  
  - `scikit-learn==1.9.1`  
  - `joblib==1.6.0`  
  - `numpy==2.4.6`

### 7.2 Standard Execution & Testing Commands

\# Smoke test (assert-based, zero external hardware or network dependencies)

.venv/bin/python3 test\_pipeline.py

\# End-to-end pipeline & regression guard test

.venv/bin/python3 tests/test\_pipeline\_e2e.py

\# Run live fog node (use \-u to prevent stdout buffering)

.venv/bin/python3 \-u main.py

\# Run single QSVM offline training run

.venv/bin/python3 \-u src/train\_qsvm.py

\# Evaluate clip accuracy and metrics

.venv/bin/python3 src/evaluate.py clips

\# Run autonomous experiment parameter search

.venv/bin/python3 \-u src/experiments.py

\# Run hardware feasibility & transpilation report

.venv/bin/python3 \-u src/hardware\_report.py

### 7.3 Critical Development Gotchas

1. **Never Read Raw Dataset Files in Context:** Do NOT open raw `.png` or `.mp4` files from `../training-data/` with read tools. It exhausts token windows. Read structured metrics from `reports/metrics.json` or inspect metadata.  
2. **Module Pathing for Joblib Artifacts:** The trained artifact `models/qsvm_model.joblib` pickles `PCAAngleProjector` by module name. Any script loading the model must ensure `src/` is on `sys.path`:  
     
   import sys  
     
   from pathlib import Path  
     
   ROOT \= Path(\_\_file\_\_).resolve().parent  
     
   sys.path.insert(0, str(ROOT / "src"))  
     
3. **Single Surface Configuration:** All operational settings live in `config.py` and `runtime_config.json`. Do NOT hardcode thresholds in downstream modules; expose them as properties on `Settings`.  
4. **Grouped Validation Splitting:** Always group train/validation splits by source video ID (`sample_by_video(..., video_subset="fit"|"val")`). Frame-level splitting leaks scene background and invalidates experimental validity.  
5. **Ensemble Integrity:** When serializing an ensemble model, every constituent member (CNN weights, PCA projector, support vectors) must be persisted in the artifact dictionary.

---

## 8\. Mentor Guidance, Logs & Milestone Chronology

### 8.1 Key Mentorship Interactions (Mr. Anirban Bhattacharya)

- **July 15, 2026:** Initial review. Confirmed technical feasibility of multi-stage QML pipeline. Recommended dimensionality reduction (PCA) after CNN stage to protect delicate quantum circuits from NISQ noise. Introduced QSVM vs QGNN trade-offs.  
- **July 20, 2026:** Validated the Cloud-Edge-Fog architecture. Formally designated the Raspberry Pi as the "Fog server". Emphasized circuit depth constraints in IBM Quantum NISQ backends.  
- **August 14, 2026:** Provided critical guidance on quantum simulation: creating an exact digital twin of hardware via `AerSimulator.from_backend(backend)` rather than just extracting noise models, ensuring qubit count, topology, native gates (`CZ`, `SX`, `X`, `RZ`), and \$T\_1/T\_2\$ times are captured. Discussed selection of IBM Brisbane.  
- **September 11 & September 25, 2026:** Addressed multi-modal behavioral signals (gait speed, excessive sweating, nervousness). Evaluated whether quantum computing or classical AI is better suited for psychological suspicion arbitration, leading to the hybrid AI Judge implementation.

### 8.2 Research Program Milestones (Mrs. Rachna Nath)

- **May 9, 2026:** Enrolled in DRIPBL STEAM Analytics Program (6-month contract: June 12 – November 30, 2026).  
- **June 12, 2026:** Program commencement; setup of DRIBPL Log Book.  
- **July 17, 2026:** Formulated formal hypothesis and created research summary document for mentor reviews.  
- **July 24, 2026:** Completed Cloud-Edge-Fog architecture framework; published [YouTube Progress Reflection Video](https://youtu.be/RUMPlAKLcc8) ([Script](https://docs.google.com/document/d/1hVZpkMmeWMBaR4PnI4nY_4ToXYk9eJBXI_brrlrtBpQ/edit)).  
- **August 7, 2026:** Published full QML Aer pipeline on GitHub; established Pre-Session Playbooks.  
- **August 21, 2026:** Implemented prototype monitoring dashboard and accounted for hardware digital twin in Aer simulation.  
- **September 15, 2026:** Project feature freeze, hardware benchmarking, and master documentation.

---

## 9\. Comprehensive Workspace & Resource Index

### 9.1 Google Drive Folders

- **Main DRIBPL Research Root:** [DRIBPL Research](https://drive.google.com/drive/folders/1cEeirqqEEzzdKHvcaWspJPL3dx2FQBPt)  
- **Actual Project Directory:** [actual-dripbl-project](https://drive.google.com/drive/folders/11sutAPL_P82_OpfJh2WKP_-gVvkJB1az)  
- **Fog Server Codebase:** [dripbl-fog-server](https://drive.google.com/drive/folders/1jL4pvADwIn8youswqiwLGvtq-kS6b500)  
- **Dashboard Architecture:** [dripbl-dashboard-architecture](https://drive.google.com/drive/folders/11JB9pdS-O0tf3IPzY-ZCtocVoOUi6iS5)  
- **QML Pipeline Benchmark:** [dribpl\_qml\_pipeline](https://drive.google.com/drive/folders/1vxxrUqpnwp_K1RH2UoKMBAY9Tl-ILMnW)  
- **Year 1 Academic Folder:** [DRIBPL\_PurvTaparia\_Year1](https://drive.google.com/drive/folders/1DDTAKpx4eC_5Sv3LaGkx0pid0d8RLy07)  
- **Research Program Folder:** [\[07\] DRIBPL Research Program](https://drive.google.com/drive/folders/19Fwfun8MFl409vRx3c76r99bQOuS4U4g)

### 9.2 Key Google Docs, Spreadsheets & Assets

- **Project Log Book:** [Log Book (Google Doc)](https://docs.google.com/document/d/1-gnwi2PkDK-d1UqYnE8JlZt5w7K1BTwAyYaBedgpuyQ/edit)  
- **Mentor Review Notes:** [Summaries with Mr. Bhattacharya (Google Doc)](https://docs.google.com/document/d/1lq6w8elIfEvPwMileCz3VOCYQ2YmBdR9-Wehzqn9ixY/edit)  
- **Executive Report:** [2026-08-03 DRIBPL Weekly Executive Report (Google Doc)](https://docs.google.com/document/d/1KEdLehEubgNhzlgkiQAxCJO4dSmBr_alOvCxTuYcsB8/edit)  
- **Pre-Session Playbook:** [2026-08-07 DRIBPL Pre-Session Briefing (Google Doc)](https://docs.google.com/document/d/19gj7ab9yvPZWfhoIuiekJl_tTJtqgmTluQZ2Uzb83sw/edit)  
- **Data Collection Sheet:** [Data Collection Table \- DRIBPL (Google Sheet)](https://docs.google.com/spreadsheets/d/1XnmcjIecdS90GV7nKMOhfUO3QBbpPO-6TYUBUAg14vE/edit)  
- **Presentation Script:** [DRIBPL Program YT Script Reflection (Google Doc)](https://docs.google.com/document/d/1hVZpkMmeWMBaR4PnI4nY_4ToXYk9eJBXI_brrlrtBpQ/edit)  
- **YouTube Milestone Reflection:** [YouTube Progress Video](https://youtu.be/RUMPlAKLcc8)

### 9.3 Email Threads

- **Mentor Communications:** [Gmail Thread — Purv Taparia \- DRIBPL Student](https://mail.google.com/mail/?extsrc=sync&client=h&plid=ACUX6DM8qBMe5rlBRdgqW50yOcJR2b1VdUM3P4c&mid=1a000d4cdee900e0)  
- **Pre-Session Briefing Dispatch:** [Gmail Thread — DRIBPL Pre-Session Briefing August 7, 2026](https://mail.google.com/mail/?extsrc=sync&client=h&plid=ACUX6DPRXy16IAqrUaR6dBtZmjlsmhpjBUwxcUk&mid=19fc81b6dfcba4dc)  
- **Registration & Contract:** [Gmail Thread — DRIPBL Registration Form](https://mail.google.com/mail/?extsrc=sync&client=h&plid=ACUX6DPlnMMt_r0YALTxTHJdJDX8eK78Pjp01-E&mid=19e0cebf437e9821)  
- **Information Session:** [Gmail Thread — DRIPBL Information Session](https://mail.google.com/mail/?extsrc=sync&client=h&plid=ACUX6DNuq_VC7o9PeGMLTI6WIC34AqQwt8v9itY&mid=19de204dacd07f0f)

---

*End of Master Context File — Generated for Claude Code Autonomous Operations.*