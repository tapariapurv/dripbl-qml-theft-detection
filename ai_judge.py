"""Stage 3: AI judge — combines the quantum classifier's verdict with gait
signals (speed, stride, jitter/nervousness) into one suspicion score, gated
by the user-set sensitivity, before a trial is flagged for a human."""
from __future__ import annotations
from config import Settings

_NERVOUS_SPEED_PX_S = 250.0  # brisk/erratic movement proxy, tune to camera scale


class AIJudge:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def judge(self, quantum_result: dict, gait: dict[str, float]) -> dict:
        class_scores = quantum_result.get("class_scores", {})
        crime_signal = class_scores.get("suspicious_signature", 0.0)
        is_person = quantum_result.get("object_class") == "person"

        speed_anomaly = min(gait["speed_px_s"] / _NERVOUS_SPEED_PX_S, 1.0)
        nervousness = gait["jitter_score"]

        suspicion = (
            0.5 * crime_signal
            + 0.25 * nervousness
            + 0.15 * speed_anomaly
            + 0.10 * (1.0 if is_person and nervousness > 0.5 else 0.0)
        )
        suspicion = min(max(suspicion, 0.0), 1.0)
        threshold = self._settings.flag_threshold
        return {
            "suspicion_score": suspicion,
            "threshold": threshold,
            "flagged": suspicion >= threshold,
            "gait": gait,
        }
