"""Threaded, always-freshest-frame capture from a camera index, network stream
URL, or local video file — with synthetic fallback and live source swap."""
from __future__ import annotations
import threading
import time
import numpy as np
import cv2

from config import Settings


def _detect_kind(source: str | int, kind: str) -> str:
    if kind != "auto":
        return kind
    if isinstance(source, int):
        return "camera"
    if isinstance(source, str) and source.split("://", 1)[0] in ("rtsp", "http", "https"):
        return "stream"
    return "file"


class FrameGrabber:
    """Background thread continuously overwrites a single-slot buffer so the
    consumer never reads stale, queued frames (cv2's internal buffer lags).
    Camera/stream sources are read as fast as they deliver; file sources are
    paced to their native FPS and looped so a clip behaves like a live feed."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._kind = "synthetic"
        self.reconfigure(settings.camera_source, settings.source_kind)

    def reconfigure(self, source: str | int, kind: str = "auto") -> None:
        """Swap the video source live (config edit -> new camera/URL/file)."""
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            if self._cap is not None:
                self._cap.release()
        self._stop = threading.Event()
        self._kind = _detect_kind(source, kind)

        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.frame_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap = cap
        else:
            cap.release()
            self._cap = None
            self._kind = "synthetic"

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _synthetic_frame(self) -> np.ndarray:
        """Moving bright square on dark noise, for camera-less dev/testing."""
        w, h = self._settings.frame_width, self._settings.frame_height
        frame = (np.random.default_rng().integers(0, 20, (h, w, 3))).astype(np.uint8)
        t = time.monotonic() - self._t0
        cx = int((np.sin(t) * 0.4 + 0.5) * w)
        cy = int((np.cos(t * 0.7) * 0.4 + 0.5) * h)
        cv2.rectangle(frame, (cx - 20, cy - 20), (cx + 20, cy + 20), (200, 200, 200), -1)
        return frame

    def _loop(self) -> None:
        fps = 30.0
        if self._kind == "file" and self._cap is not None:
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        period = 1.0 / max(fps, 1.0)

        while not self._stop.is_set():
            if self._kind == "synthetic":
                frame = self._synthetic_frame()
                time.sleep(period)
            else:
                ok, frame = self._cap.read()
                if not ok:
                    if self._kind == "file" and self._settings.loop_file:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self._kind = "synthetic"
                    continue
                if self._kind == "file":
                    time.sleep(period)  # pace file playback to its native rate
            with self._lock:
                self._frame = frame

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
