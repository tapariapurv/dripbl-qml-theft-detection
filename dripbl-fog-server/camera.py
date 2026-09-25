"""Threaded, always-freshest-frame capture from a camera index, network stream
URL, or local video file — with reconnect-on-failure and live source swap."""
from __future__ import annotations
import random
import threading
import time
from pathlib import Path

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


def to_bgr8(frame: np.ndarray) -> np.ndarray:
    """Any camera frame -> 3-channel uint8 BGR (grayscale, BGRA and 16-bit sensors exist)."""
    if frame.dtype != np.uint8:
        f = frame.astype(np.float32)
        hi = float(f.max()) or 1.0
        frame = np.clip(f * (255.0 / (65535.0 if hi > 255 else 255.0)), 0, 255).astype(np.uint8)
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


STREAM_TIMEOUT_MS = 5000


class FrameGrabber:
    """Background thread continuously overwrites a single-slot buffer so the
    consumer never reads stale, queued frames (cv2's internal buffer lags).
    Camera/stream sources are read as fast as they deliver; file sources are
    paced to their native FPS and looped so a clip behaves like a live feed.

    Failure behaviour matters on a real deployment:
      * the source is opened on the capture thread, so an unreachable IP camera
        can never freeze the main loop (FFmpeg open timeouts are ~30 s);
      * a source that fails to open, or drops mid-stream, shows a static NO SIGNAL
        frame (no motion, so it can never produce a trial) and is retried with
        backoff until it comes back; `kind` reports "connecting" / "offline";
      * the moving-square synthetic feed is only used when asked for explicitly
        (camera_source "" / "synthetic"), because on a real node fake motion
        would mean fake alerts.
    """

    RETRY_SEC = (1, 2, 5, 10, 30)
    NO_SIGNAL_EVERY_SEC = 1.0

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._frame: np.ndarray | None = None
        self._seq = 0        # bumped per captured frame
        self._read_seq = 0   # last seq handed to read()
        self._gen = 0        # source generation: frames from a replaced source are dropped
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread: threading.Thread | None = None
        self._kind = "connecting"
        self.reconfigure(settings.camera_source, settings.source_kind)

    def reconfigure(self, source: str | int, kind: str = "auto") -> None:
        """Swap the video source live (config edit -> new camera/URL/file). Never blocks:
        the old capture thread is told to stop and releases its own capture."""
        self._stop.set()
        with self._lock:
            self._gen += 1
            gen = self._gen
        self._stop = threading.Event()
        synthetic = kind == "synthetic" or source in ("", "synthetic")
        self._kind = "synthetic" if synthetic else "connecting"
        self._thread = threading.Thread(target=self._run, args=(source, kind, synthetic, gen, self._stop),
                                        daemon=True, name="frame-grabber")
        self._thread.start()

    def _publish(self, frame: np.ndarray, gen: int) -> None:
        frame = to_bgr8(frame)
        with self._lock:
            if gen != self._gen:
                return
            self._frame = frame
            self._seq += 1

    def _synthetic_frame(self) -> np.ndarray:
        """Moving bright square on dark noise, for camera-less dev/testing."""
        w, h = self._settings.frame_width, self._settings.frame_height
        frame = (np.random.default_rng().integers(0, 20, (h, w, 3))).astype(np.uint8)
        t = time.monotonic() - self._t0
        cx = int((np.sin(t) * 0.4 + 0.5) * w)
        cy = int((np.cos(t * 0.7) * 0.4 + 0.5) * h)
        cv2.rectangle(frame, (cx - 20, cy - 20), (cx + 20, cy + 20), (200, 200, 200), -1)
        return frame

    def _no_signal_frame(self, label: str) -> np.ndarray:
        w, h = self._settings.frame_width, self._settings.frame_height
        frame = np.full((h, w, 3), 18, np.uint8)
        cv2.putText(frame, label, (max(10, w // 2 - 150), h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (90, 90, 230), 2, cv2.LINE_AA)
        return frame

    def _wait_offline(self, source: str | int, attempt: int, gen: int, stop: threading.Event) -> None:
        """Show NO SIGNAL (identical frames: zero motion) until the next retry."""
        self._kind = "offline"
        frame = self._no_signal_frame("NO SIGNAL - reconnecting")
        deadline = time.monotonic() + self.RETRY_SEC[min(attempt, len(self.RETRY_SEC) - 1)]
        while not stop.is_set() and time.monotonic() < deadline:
            self._publish(frame, gen)
            stop.wait(self.NO_SIGNAL_EVERY_SEC)

    def _run(self, source: str | int, kind: str, synthetic: bool, gen: int, stop: threading.Event) -> None:
        if synthetic:
            while not stop.is_set():
                self._publish(self._synthetic_frame(), gen)
                stop.wait(1 / 30)
            return
        detected = _detect_kind(source, kind)
        attempt = 0
        while not stop.is_set():
            if detected == "stream":
                # OpenCV's FFmpeg opens are serialised process-wide and default to a 30 s timeout,
                # so one dead IP camera would stall every other source; bound open and reads.
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_TIMEOUT_MS,
                                                                cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_TIMEOUT_MS])
            else:
                cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                cap.release()
                self._wait_offline(source, attempt, gen, stop)
                attempt += 1
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.frame_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            period = 1.0 / max(cap.get(cv2.CAP_PROP_FPS) or 30.0, 1.0)
            self._kind, attempt, failures = detected, 0, 0
            try:
                while not stop.is_set():
                    ok, frame = cap.read()
                    if ok and (frame is None or frame.size == 0):
                        ok = False  # some drivers report success with an empty buffer
                    if not ok:
                        if detected == "file" and self._settings.loop_file and failures < 3:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            failures += 1
                            continue
                        break  # dropped stream / unplugged camera / finished file: reconnect
                    failures = 0
                    self._publish(frame, gen)
                    if detected == "file":
                        stop.wait(period)  # pace file playback to its native rate
            finally:
                cap.release()
            if not stop.is_set():
                self._wait_offline(source, attempt, gen, stop)
                attempt += 1

    def read(self) -> np.ndarray | None:
        """The freshest frame, or None if nothing new arrived since the last read
        (so the consumer never re-processes a duplicate frame at full CPU)."""
        with self._lock:
            if self._frame is None or self._seq == self._read_seq:
                return None
            self._read_seq = self._seq
            return self._frame.copy()

    @property
    def kind(self) -> str:
        """camera | stream | file | synthetic | connecting | offline"""
        return self._kind

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class MockClips:
    """Research mode: labelled clips from <mock_dir>/{normal,shoplifting}/*.mp4
    (dataset-1: real 640x480 retail footage), shuffled and replayed one after another
    so every trial the node records carries a ground-truth label."""

    LABELS = {"normal": "normal", "shoplifting": "theft"}

    def __init__(self, mock_dir: str | Path, seed: int | None = None) -> None:
        root = Path(mock_dir)
        self.clips = [(p, label) for folder, label in self.LABELS.items()
                      for p in sorted((root / folder).glob("*.mp4"))]
        if not self.clips:
            raise FileNotFoundError(f"no mock clips under {root}/{{normal,shoplifting}}/*.mp4")
        self._rng = random.Random(seed)
        self._order: list[tuple[Path, str]] = []

    def next(self) -> tuple[Path, str, float]:
        """(path, ground-truth label, duration in seconds)."""
        if not self._order:
            self._order = self.clips[:]
            self._rng.shuffle(self._order)
        path, label = self._order.pop()
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
        return path, label, duration
