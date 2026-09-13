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
from typing import Tuple


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
    alert_endpoint: str = "http://localhost:8000/api/alerts"
    request_timeout_sec: float = 2.0
    thumbnail_jpeg_quality: int = 60
    send_thumbnail: bool = False

    buzzer_pin: int = 17
    led_pin: int = 27

    log_path: str = "trials.log"

    @property
    def area_threshold(self) -> float:
        """Sensitivity-scaled motion area gate: higher sensitivity -> lower area needed."""
        return self.min_contour_area * (1.6 - 1.2 * self.sensitivity)

    @property
    def flag_threshold(self) -> float:
        """AI judge suspicion threshold: higher sensitivity -> flags more readily."""
        return 1.0 - self.sensitivity


class ConfigStore:
    """Loads Settings from JSON, creating the file with defaults if absent, and
    detects external edits by mtime so the running system can hot-reload them."""

    def __init__(self, path: str = "runtime_config.json") -> None:
        self._path = path
        self._mtime = 0.0
        self.settings = self._load_or_init()

    def _load_or_init(self) -> Settings:
        if not os.path.isfile(self._path):
            settings = Settings()
            self.save(settings)
            return settings
        return self._read()

    def _read(self) -> Settings:
        with open(self._path) as fh:
            raw = json.load(fh)
        valid = {f.name for f in fields(Settings)}
        raw = {k: v for k, v in raw.items() if k in valid}
        if "roi" in raw:
            raw["roi"] = tuple(raw["roi"])
        self._mtime = os.path.getmtime(self._path)
        return Settings(**raw)

    def save(self, settings: Settings) -> None:
        with open(self._path, "w") as fh:
            json.dump(dataclasses.asdict(settings), fh, indent=2)
        self._mtime = os.path.getmtime(self._path)
        self.settings = settings

    def reload_if_changed(self) -> Settings | None:
        """Returns the fresh Settings if the file changed on disk, else None."""
        try:
            mtime = os.path.getmtime(self._path)
        except OSError:
            return None
        if mtime <= self._mtime:
            return None
        self.settings = self._read()
        return self.settings


DEFAULT = Settings()
