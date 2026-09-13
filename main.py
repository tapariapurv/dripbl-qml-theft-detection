"""DRIBPL fog/edge node.

Pipeline: capture -> motion+gait gate -> CNN/PCA features -> quantum classifier
(object/scene) -> AI judge (suspicion score, sensitivity-gated) -> human alert.

Video source, ROI, sensitivity, and every other setting live in
runtime_config.json and hot-reload while this runs (see config.ConfigStore).
"""
from __future__ import annotations
import time
import uuid

from camera import FrameGrabber
from config import ConfigStore
from edge_pipeline import EdgePipeline
from quantum_classifier import QuantumClassifier
from ai_judge import AIJudge
from network_alerter import NetworkAlerter

_CONFIG_POLL_SEC = 1.0


def run() -> None:
    store = ConfigStore()
    settings = store.settings
    grabber = FrameGrabber(settings)
    pipeline = EdgePipeline(settings)
    quantum = QuantumClassifier(settings)
    judge = AIJudge(settings)
    alerter = NetworkAlerter(settings)

    last_trigger = 0.0
    last_config_check = 0.0
    print(f"[dripbl-fog] node online, source={settings.camera_source} sensitivity={settings.sensitivity}")

    try:
        while True:
            now = time.monotonic()
            if now - last_config_check >= _CONFIG_POLL_SEC:
                last_config_check = now
                fresh = store.reload_if_changed()
                if fresh is not None:
                    if fresh.camera_source != settings.camera_source or fresh.source_kind != settings.source_kind:
                        grabber.reconfigure(fresh.camera_source, fresh.source_kind)
                    settings = fresh
                    pipeline._settings = settings          # ponytail: direct swap, no per-stage reconfigure API yet
                    quantum._settings = settings
                    judge._settings = settings
                    alerter._settings = settings
                    print(f"[dripbl-fog] config reloaded: sensitivity={settings.sensitivity}")

            frame = grabber.read()
            if frame is None:
                continue

            triggered, gait = pipeline.observe(frame)
            if not (triggered and now - last_trigger >= settings.debounce_sec):
                alerter.trigger_alert(False)
                continue

            t0 = time.perf_counter()
            features = pipeline.extract_features(frame)
            quantum_result = quantum.classify(features)
            verdict = judge.judge(quantum_result, gait)
            elapsed_ms = (time.perf_counter() - t0) * 1e3

            trial_id = uuid.uuid4().hex
            payload = alerter.build_payload(trial_id, features, quantum_result, verdict, frame)
            alerter.log_trial(payload)
            if verdict["flagged"]:
                alerter.dispatch_alert(payload)
            else:
                alerter.trigger_alert(False)

            last_trigger = now
            print(
                f"[dripbl-fog] trial={trial_id} infer={elapsed_ms:.2f}ms "
                f"class={quantum_result['object_class']} suspicion={verdict['suspicion_score']:.2f} "
                f"flagged={verdict['flagged']}"
            )
    except KeyboardInterrupt:
        print("\n[dripbl-fog] shutting down")
    finally:
        alerter.trigger_alert(False)
        alerter.close()
        grabber.release()


if __name__ == "__main__":
    run()
