"""Settings + a hot-reloadable JSON config store: the single end-to-end edit point.

Edit runtime_config.json (source, ROI, sensitivity, thresholds, endpoints, GPIO pins)
while the system runs; main.py polls ConfigStore.reload_if_changed() each loop and
picks up changes without a restart.
"""
from __future__ import annotations
import dataclasses
import json
import os
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent


@dataclass(slots=True)
class Settings:
    # Video source: camera index (0,1,...), RTSP/HTTP(S) URL, or a local file path.
    camera_source: str | int = 0
    source_kind: str = "auto"      # "auto" | "camera" | "stream" | "file"
    loop_file: bool = True         # replay local video files when they end
    frame_width: int = 640
    frame_height: int = 480


    roi: Tuple[int, int, int, int] = (140, 60, 360, 360)

    # Detection sensitivity, 0 (least sensitive) .. 1 (most sensitive). Scales
    # motion area threshold, debounce, and the AI judge's flag threshold.
    sensitivity: float = 0.5
    min_contour_area: float = 500.0
    debounce_sec: float = 3.0
    mog2_history: int = 300
    mog2_var_threshold: float = 32.0

    # Model I/O
    cnn_input_size: int = 64
    latent_dim: int = 128
    pca_dim: int = 4
    pca_components_path: str = "pca_components.npy"
    cnn_weights_path: str = "cnn_weights.pt"

    # Quantum classification stage
    class_prototypes_path: str = "class_prototypes.json"
    quantum_endpoint: str = ""     # remote Aer/QPU service; blank = local simulation only
    use_qiskit: bool = True        # use real ZZFeatureMap+Aer kernel if qiskit installed

    # Human-facing alert dispatch (only called when the AI judge flags a trial)
    alert_endpoint: str = ""   # optional extra POST per flagged trial; the API reads the DB directly
    request_timeout_sec: float = 2.0
    thumbnail_jpeg_quality: int = 60
    send_thumbnail: bool = False

    buzzer_pin: int = 17
    led_pin: int = 27

    log_path: str = "trials.log"

    # AI judge weights. Trained model: a measured detector dominates, gait modifies.
    w_crime: float = 0.80
    w_nervous: float = 0.13
    w_speed: float = 0.07
    # Prototype-fallback weights: gait gets a near-equal vote when no trained model.
    fb_w_crime: float = 0.50
    fb_w_nervous: float = 0.25
    fb_w_speed: float = 0.15
    fb_w_person_bonus: float = 0.10
    nervous_speed_px_s: float = 250.0   # speed that counts as fully anomalous; camera-scale dependent
    # Clip-level judgement: frames are scored this often while there is motion, and the
    # judge sees the mean margin of the last judge_window_sec (how the model was measured).
    # Camera domain of this node. "retail" runs the retail expert model (store/counter cameras);
    # "cctv" and "default" run the general model; each also picks its calibrated clip threshold.
    site_profile: str = "default"
    score_fps: float = 4.0
    judge_window_sec: float = 2.0

    model_path: str = "models/qsvm_model.joblib"          # default / CCTV expert
    retail_model_path: str = "models/retail_model.joblib"  # retail expert, used when site_profile = "retail"

    # "live" = camera_source; "mock" = replay labelled dataset-1 clips (ground truth recorded).
    mode: str = "live"
    mock_dir: str = "../training-data/dataset-1"
    mock_clip_sec: float = 20.0
    # replay only the dataset-1 hold-out clips (reports/dataset1_split.json): the retail expert
    # trained on the others, so replaying them would inflate the live scoreboard
    mock_holdout_only: bool = True

    # Command center: the node publishes to data_dir (SQLite + live frame + snapshots).
    data_dir: str = "data"
    live_frame_fps: float = 10.0
    live_jpeg_quality: int = 70
    heartbeat_sec: float = 1.0

    @property
    def area_threshold(self) -> float:
        """Sensitivity-scaled motion area gate: higher sensitivity -> lower area needed."""
        return self.min_contour_area * (1.6 - 1.2 * self.sensitivity)

    @property
    def flag_threshold(self) -> float:
        """AI judge suspicion threshold: higher sensitivity -> flags more readily."""
        return 1.0 - self.sensitivity


# Validation shared by the node's loader and the API: a hand-edited or API-sent value
# that fails these never reaches the running pipeline.
RANGES: dict[str, tuple[float, float]] = {
    "sensitivity": (0.0, 1.0), "debounce_sec": (0.0, 3600.0), "min_contour_area": (1.0, 1e6),
    "mog2_history": (1, 10000), "mog2_var_threshold": (1.0, 1000.0), "live_frame_fps": (0.0, 30.0),
    "live_jpeg_quality": (10, 100), "mock_clip_sec": (1.0, 3600.0), "nervous_speed_px_s": (1.0, 1e5),
    "heartbeat_sec": (0.2, 60.0), "thumbnail_jpeg_quality": (10, 100), "request_timeout_sec": (0.1, 60.0),
    "score_fps": (0.5, 30.0), "judge_window_sec": (0.25, 30.0),
    "frame_width": (16, 7680), "frame_height": (16, 4320), "buzzer_pin": (0, 40), "led_pin": (0, 40),
    **{w: (0.0, 1.0) for w in ("w_crime", "w_nervous", "w_speed", "fb_w_crime", "fb_w_nervous",
                               "fb_w_speed", "fb_w_person_bonus")},
}
CHOICES: dict[str, tuple[str, ...]] = {
    "mode": ("live", "mock"), "source_kind": ("auto", "camera", "stream", "file"),
    "site_profile": ("default", "cctv", "retail"),
}
_FIELDS = {f.name for f in fields(Settings)}


def _coerce(key: str, val: object, current: object) -> object:
    """Validated value for `key`, or ValueError with a message a person can act on."""
    if key == "camera_source":
        if isinstance(val, bool) or not isinstance(val, (int, str)):
            raise ValueError("camera_source must be a camera index or a URL/path string")
        return val
    if key == "roi":
        if not (isinstance(val, (list, tuple)) and len(val) == 4
                and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in val)
                and val[2] > 0 and val[3] > 0):
            raise ValueError("roi must be [x, y, width, height] with non-negative integers")
        return tuple(val)
    if key in CHOICES and val not in CHOICES[key]:
        raise ValueError(f"{key} must be one of {'|'.join(CHOICES[key])}")
    if isinstance(current, bool):
        if not isinstance(val, bool):
            raise ValueError(f"{key} must be true/false")
        return val
    if isinstance(current, (int, float)):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or val != val:  # rejects NaN too
            raise ValueError(f"{key} must be a number")
        if isinstance(current, int) and not float(val).is_integer():
            raise ValueError(f"{key} must be a whole number")
        lo, hi = RANGES.get(key, (-1e12, 1e12))
        if not lo <= val <= hi:
            raise ValueError(f"{key} must be between {lo:g} and {hi:g}")
        return type(current)(val)
    if isinstance(current, str) and not isinstance(val, str):
        raise ValueError(f"{key} must be a string")
    return val


def apply_changes(base: Settings, patch: dict, *, strict: bool,
                  fallback: Settings | None = None) -> tuple[Settings, list[str]]:
    """New Settings = base + patch. strict: raise on the first problem (API);
    otherwise skip bad keys, keeping `fallback`'s value for them (else base's), and
    return their messages (hand-edited file)."""
    if not isinstance(patch, dict):
        raise ValueError("settings must be a JSON object")
    values, problems = dataclasses.asdict(base), []
    for key, val in patch.items():
        try:
            if key not in _FIELDS:
                raise ValueError(f"unknown setting: {key}")
            values[key] = _coerce(key, val, values[key])
        except ValueError as exc:
            if strict:
                raise
            problems.append(str(exc))
            if fallback is not None and key in _FIELDS:
                values[key] = getattr(fallback, key)
    values["roi"] = tuple(values["roi"])
    return Settings(**values), problems


class ConfigStore:
    """Loads Settings from JSON, creating the file with defaults if absent, and
    detects external edits by mtime so the running system can hot-reload them.

    A broken edit never stops the node: invalid JSON keeps the current settings,
    invalid values fall back to their previous/default value. `problems` says why."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = path or os.environ.get("DRIPBL_CONFIG") or ROOT / "runtime_config.json"
        self._mtime = 0.0
        self.problems: list[str] = []
        self.settings = self._load_or_init()

    def _load_or_init(self) -> Settings:
        if not os.path.isfile(self._path):
            settings = Settings()
            self.save(settings)
            return settings
        try:
            return self._read(Settings())
        except ValueError as exc:  # unreadable at startup: run on defaults, leave the file for a human
            self.problems = [str(exc)]
            return Settings()

    def _read(self, previous: Settings) -> Settings:
        """The file is the whole truth: missing keys take their defaults; only a key with
        an invalid value keeps its previous value (so a typo never changes behaviour)."""
        self._mtime = os.path.getmtime(self._path)
        try:
            with open(self._path) as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{os.path.basename(str(self._path))} is not valid JSON ({exc})") from exc
        settings, self.problems = apply_changes(Settings(), raw, strict=False, fallback=previous)
        return settings

    def save(self, settings: Settings) -> None:
        # temp + os.replace: another process (node <-> API) never reads a half-written file
        tmp = f"{self._path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(dataclasses.asdict(settings), fh, indent=2)
        os.replace(tmp, self._path)
        self._mtime = os.path.getmtime(self._path)
        self.settings = settings

    def reload_if_changed(self) -> Settings | None:
        """Returns the fresh Settings if the file changed on disk, else None
        (also None when the edit is unreadable; see `problems`)."""
        try:
            mtime = os.path.getmtime(self._path)
        except OSError:
            return None
        if mtime <= self._mtime:
            return None
        try:
            self.settings = self._read(self.settings)
        except ValueError as exc:
            self.problems = [str(exc)]
            return None
        return self.settings


DEFAULT = Settings()
