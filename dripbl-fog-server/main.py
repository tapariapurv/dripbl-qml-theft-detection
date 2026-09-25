"""DRIBPL fog/edge node.

Pipeline: capture -> motion+gait gate -> CNN/PCA features -> quantum classifier
(object/scene) -> AI judge (suspicion score, sensitivity-gated) -> human alert.

Video source, ROI, sensitivity, and every other setting live in
runtime_config.json and hot-reload while this runs (see config.ConfigStore).

For the command center (api/, dashboard/) the node also publishes to data_dir:
every trial into SQLite (store.py), the latest annotated frame as live.jpg, a
snapshot per flagged trial, and a heartbeat row with live node status.
"""
from __future__ import annotations
import dataclasses
import os
import signal
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

import store
from camera import FrameGrabber, MockClips
from config import ROOT, ConfigStore, Settings
from edge_pipeline import EdgePipeline, MotionDetector
from quantum_classifier import QuantumClassifier
from ai_judge import AIJudge
from network_alerter import NetworkAlerter

_CONFIG_POLL_SEC = 1.0
_FLAG_HIGHLIGHT_SEC = 2.0  # live frame border stays red this long after a flag


def _annotate(frame: np.ndarray, settings: Settings, alert: bool) -> np.ndarray:
    out = frame.copy()
    x, y, w, h = settings.roi
    colour = (60, 60, 240) if alert else (120, 200, 80)  # BGR: red on alert, green otherwise
    cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
    if alert:
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), colour, 6)
    return out


def _write_jpeg(path: Path, frame: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)  # survives data/ being deleted while running
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(buf.tobytes())
        os.replace(tmp, path)  # readers (the API's MJPEG stream) never see a partial file


def _display_name(source: object) -> str:
    """Camera source for logs/DB/dashboard with any user:password@ removed from URLs."""
    text = str(source)
    if "://" in text:
        scheme, rest = text.split("://", 1)
        netloc, _, path = rest.partition("/")
        text = f"{scheme}://{netloc.rsplit('@', 1)[-1]}" + (f"/{path}" if path else "")
    return text


class _Source:
    """Owns what the grabber is playing: the configured camera/file, or rotating mock clips."""

    def __init__(self, grabber: FrameGrabber) -> None:
        self._grabber = grabber
        self._mock: MockClips | None = None
        self.name, self.ground_truth, self._clip_end = "", None, 0.0

    def apply(self, settings: Settings) -> None:
        if settings.mode == "mock":
            try:
                self._mock = MockClips(ROOT / settings.mock_dir)
            except FileNotFoundError as exc:
                print(f"[dripbl-fog] mock mode unavailable ({exc}); using camera_source")
                self._mock = None
        else:
            self._mock = None
        if self._mock is None:
            self._grabber.reconfigure(settings.camera_source, settings.source_kind)
            self.name, self.ground_truth = _display_name(settings.camera_source), None
        else:
            self._clip_end = 0.0  # rotate immediately

    def tick(self, settings: Settings) -> bool:
        """Advance to the next mock clip when the current one is done; True if switched."""
        if self._mock is None or time.monotonic() < self._clip_end:
            return False
        path, label, duration = self._mock.next()
        self._grabber.reconfigure(str(path), "file")
        self.name, self.ground_truth = path.name, label
        self._clip_end = time.monotonic() + min(settings.mock_clip_sec, duration or settings.mock_clip_sec)
        print(f"[dripbl-fog] mock clip {path.name} (ground truth: {label})")
        return True


def run() -> None:
    config = ConfigStore()
    settings = config.settings
    data_dir = ROOT / settings.data_dir
    snap_dir = data_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    db = data_dir / "dripbl.db"
    live_path = data_dir / "live.jpg"

    # start on the synthetic feed; _Source.apply opens the real camera/file/mock clip exactly once
    grabber = FrameGrabber(dataclasses.replace(settings, camera_source="", source_kind="synthetic"))
    source = _Source(grabber)
    source.apply(settings)
    pipeline = EdgePipeline(settings)
    quantum = QuantumClassifier(settings)
    judge = AIJudge(settings)
    alerter = NetworkAlerter(settings)

    # Prefer the trained QSVM ensemble; fall back to the prototype classifier so a
    # checkout with no models/qsvm_model.joblib still runs.
    try:
        from theft_detector import TheftDetector
        theft = TheftDetector(settings)
        model_desc = theft.description
        print(f"[dripbl-fog] theft model: {model_desc}")
    except Exception as exc:  # noqa: BLE001
        theft = None
        model_desc = "prototype fallback (no trained model)"
        print(f"[dripbl-fog] no trained model ({exc}); using seeded prototype classifier")

    started = time.time()
    last_trigger = last_config_check = last_frame_pub = last_heartbeat = last_score = 0.0
    last_flag_at = -1e9
    frames_since_hb, n_trials, n_flagged = 0, 0, 0
    last_infer_ms, last_gait, last_triggered = 0.0, {}, False
    errors, last_error_print = 0, 0.0
    last_problems = config.problems
    last_kind = grabber.kind
    if config.problems:
        print(f"[dripbl-fog] config problems (using defaults for these): {config.problems}")
    print(f"[dripbl-fog] node online, mode={settings.mode} source={_display_name(settings.camera_source)} "
          f"sensitivity={settings.sensitivity}")

    try:
        while True:
            try:
                now = time.monotonic()
                if now - last_config_check >= _CONFIG_POLL_SEC:
                    last_config_check = now
                    fresh = config.reload_if_changed()
                    if config.problems != last_problems:
                        last_problems = config.problems
                        if last_problems:
                            print(f"[dripbl-fog] config problem (kept the previous values): {last_problems}")
                    if fresh is not None:
                        if (fresh.mode, fresh.camera_source, fresh.source_kind, fresh.mock_dir) != \
                                (settings.mode, settings.camera_source, settings.source_kind, settings.mock_dir):
                            source.apply(fresh)
                        settings = fresh
                        pipeline._settings = settings          # ponytail: direct swap, no per-stage reconfigure API yet
                        # MotionDetector copies ROI/MOG2 params at init, so rebuild it or ROI and
                        # sensitivity edits never reach the motion gate (resets the background model).
                        pipeline._motion = MotionDetector(settings)
                        quantum._settings = settings
                        judge._settings = settings
                        alerter._settings = settings
                        if theft is not None:
                            theft._settings = settings
                        print(f"[dripbl-fog] config reloaded: sensitivity={settings.sensitivity}")

                if source.tick(settings) or grabber.kind != last_kind:
                    # new scene, camera lost or regained: the cut itself is not motion
                    last_kind = grabber.kind
                    pipeline._motion = MotionDetector(settings)

                if now - last_heartbeat >= settings.heartbeat_sec:
                    elapsed = now - last_heartbeat if last_heartbeat else settings.heartbeat_sec
                    store.put_heartbeat(db, {
                        "started": started, "mode": settings.mode, "source": source.name,
                        "source_kind": grabber.kind, "ground_truth": source.ground_truth,
                        "model": model_desc, "trained_model": theft is not None,
                        "fps": frames_since_hb / elapsed, "last_infer_ms": last_infer_ms,
                        "trials": n_trials, "flagged": n_flagged,
                        "motion_triggered": last_triggered, "gait": last_gait,
                        "sensitivity": settings.sensitivity, "flag_threshold": settings.flag_threshold,
                        "area_threshold": settings.area_threshold, "roi": list(settings.roi),
                        "errors": errors, "config_problems": config.problems,
                    })
                    last_heartbeat, frames_since_hb = now, 0

                frame = grabber.read()
                if frame is None:
                    time.sleep(0.005)
                    continue
                frames_since_hb += 1

                triggered, gait = pipeline.observe(frame)
                last_triggered, last_gait = triggered, gait
                if theft is not None:
                    theft.observe(frame, now)
                    # while there is motion, keep the judgement window filled (clip-level decision)
                    if triggered and now - last_score >= 1.0 / settings.score_fps \
                            and last_kind not in ("offline", "connecting"):
                        theft.score(frame, now)
                        last_score = now
                alert_on = now - last_flag_at < _FLAG_HIGHLIGHT_SEC
                if settings.live_frame_fps > 0 and now - last_frame_pub >= 1.0 / settings.live_frame_fps:
                    _write_jpeg(live_path, _annotate(frame, settings, alert_on), settings.live_jpeg_quality)
                    last_frame_pub = now

                window_ready = theft is None or theft.window_frames(now) >= max(
                    2, int(settings.score_fps * settings.judge_window_sec / 2))  # judge a clip, not one frame
                if not (triggered and window_ready and now - last_trigger >= settings.debounce_sec) \
                        or last_kind in ("offline", "connecting"):
                    if not alert_on:
                        alerter.trigger_alert(False)
                    continue

                t0 = time.perf_counter()
                if theft is not None:
                    quantum_result = theft.classify(frame, now)
                    last_score = now
                    features = quantum_result["features"]
                else:
                    features = pipeline.extract_features(frame)
                    quantum_result = quantum.classify(features)
                verdict = judge.judge(quantum_result, gait)
                last_infer_ms = (time.perf_counter() - t0) * 1e3

                trial_id = uuid.uuid4().hex
                payload = alerter.build_payload(trial_id, features, quantum_result, verdict, frame)
                payload["stage"] = {
                    "motion_triggered": True, "roi": list(settings.roi),
                    "area_threshold": settings.area_threshold, "sensitivity": settings.sensitivity,
                    "latency_ms": last_infer_ms, "model": model_desc,
                    "mode": settings.mode, "source": source.name,
                }
                alerter.log_trial(payload)

                snapshot = None
                if verdict["flagged"]:
                    snapshot = f"{trial_id}.jpg"
                    _write_jpeg(snap_dir / snapshot, _annotate(frame, settings, True), 85)
                    alerter.dispatch_alert(payload)
                    last_flag_at, n_flagged = now, n_flagged + 1
                store.insert_trial(db, {k: v for k, v in payload.items() if k != "thumbnail_jpeg_b64"},
                                   source=source.name, ground_truth=source.ground_truth, snapshot=snapshot)
                n_trials += 1

                last_trigger = now
                print(
                    f"[dripbl-fog] trial={trial_id} infer={last_infer_ms:.2f}ms "
                    f"class={quantum_result['object_class']} suspicion={verdict['suspicion_score']:.2f} "
                    f"flagged={verdict['flagged']}" + (f" truth={source.ground_truth}" if source.ground_truth else "")
                )
            except Exception as exc:  # noqa: BLE001 - one bad frame/disk hiccup must not stop the node
                errors += 1
                if time.monotonic() - last_error_print > 10:  # rate-limited: a stuck fault can't flood the log
                    print(f"[dripbl-fog] error ({errors} so far), continuing: {type(exc).__name__}: {exc}", flush=True)
                    last_error_print = time.monotonic()
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[dripbl-fog] shutting down")
    finally:
        alerter.trigger_alert(False)
        alerter.close()
        grabber.release()


def _sigterm(*_: object) -> None:
    raise KeyboardInterrupt  # systemd / start.py stop: run the same clean shutdown as Ctrl-C


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    run()
