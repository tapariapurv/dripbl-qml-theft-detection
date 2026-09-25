"""Stage 4: human-facing dispatch, GPIO alert trigger, structured trial logging."""
from __future__ import annotations
import base64
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
import requests

from config import Settings

try:
    import RPi.GPIO as GPIO  # type: ignore
    _HAS_GPIO = True
except ImportError:
    _HAS_GPIO = False


class _MockGPIO:
    """Logs pin state instead of driving real hardware (non-Pi hosts)."""
    BCM = OUT = HIGH = LOW = None

    def setmode(self, *_: Any) -> None: pass
    def setup(self, pin: int, *_: Any) -> None: logging.info("GPIO(mock) setup pin=%d", pin)
    def output(self, pin: int, state: Any) -> None: logging.info("GPIO(mock) pin=%d -> %s", pin, state)
    def cleanup(self) -> None: pass


class NetworkAlerter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._gpio = GPIO if _HAS_GPIO else _MockGPIO()
        self._gpio.setmode(self._gpio.BCM)
        for pin in (settings.buzzer_pin, settings.led_pin):
            self._gpio.setup(pin, self._gpio.OUT)

        self._logger = logging.getLogger("dripbl.fog")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False  # keep console clean; file handler below is sole sink
        handler = logging.FileHandler(settings.log_path)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    def build_payload(
        self,
        trial_id: str,
        features: np.ndarray,
        quantum_result: dict,
        verdict: dict,
        frame: np.ndarray | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "trial_id": trial_id,
            "timestamp": time.time(),
            # asarray, not .tolist(): the feature vector arrives as an ndarray from
            # EdgePipeline but as a plain list from TheftDetector.
            "features": np.asarray(features, dtype=float).tolist(),
            "quantum": quantum_result,
            "verdict": verdict,
        }
        if self._settings.send_thumbnail and frame is not None:
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._settings.thumbnail_jpeg_quality]
            )
            if ok:
                payload["thumbnail_jpeg_b64"] = base64.b64encode(buf).decode("ascii")
        return payload

    def log_trial(self, payload: dict[str, Any]) -> None:
        """Every observed trial is logged, flagged or not (audit trail)."""
        self._logger.info(json.dumps({k: v for k, v in payload.items() if k != "thumbnail_jpeg_b64"}))

    def dispatch_alert(self, payload: dict[str, Any]) -> None:
        """Only flagged trials are pushed to the human-facing endpoint + hardware."""
        if self._settings.alert_endpoint:  # blank = no external POST (the API reads the DB instead)
            self._pool.submit(self._post, payload)
        self.trigger_alert(True)

    def _post(self, payload: dict[str, Any]) -> None:
        try:
            requests.post(
                self._settings.alert_endpoint, json=payload, timeout=self._settings.request_timeout_sec
            )
        except requests.RequestException as exc:
            self._logger.warning(json.dumps({"error": str(exc), "trial_id": payload["trial_id"]}))

    def trigger_alert(self, on: bool) -> None:
        state = self._gpio.HIGH if on else self._gpio.LOW
        self._gpio.output(self._settings.buzzer_pin, state)
        self._gpio.output(self._settings.led_pin, state)

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        self._gpio.cleanup()
