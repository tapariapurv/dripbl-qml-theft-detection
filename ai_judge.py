"""Stage 3: AI judge — combines the quantum classifier's verdict with gait
signals (speed, stride, jitter/nervousness) into one suspicion score, gated
by the user-set sensitivity, before a trial is flagged for a human.

Weights live in Settings (editable from the dashboard); `breakdown` records each
weighted term so the Pipeline Inspector can show exactly why a trial was flagged."""
from __future__ import annotations
from config import Settings


class AIJudge:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def judge(self, quantum_result: dict, gait: dict[str, float]) -> dict:
        s = self._settings
        class_scores = quantum_result.get("class_scores", {})
        crime_signal = class_scores.get("suspicious_signature", 0.0)
        is_person = quantum_result.get("object_class") == "person"

        speed_anomaly = min(gait["speed_px_s"] / s.nervous_speed_px_s, 1.0)
        nervousness = gait["jitter_score"]

        if quantum_result.get("trained"):
            # A measured detector (0.871 clip accuracy) deserves to dominate the
            # score; gait stays as a modifier rather than a near-equal vote, which
            # is only appropriate for the seeded prototype fallback below.
            breakdown = {
                "crime_signal": s.w_crime * crime_signal,
                "nervousness": s.w_nervous * nervousness,
                "speed_anomaly": s.w_speed * speed_anomaly,
            }
        else:
            breakdown = {
                "crime_signal": s.fb_w_crime * crime_signal,
                "nervousness": s.fb_w_nervous * nervousness,
                "speed_anomaly": s.fb_w_speed * speed_anomaly,
                "person_nervous_bonus": s.fb_w_person_bonus * (1.0 if is_person and nervousness > 0.5 else 0.0),
            }
        suspicion = min(max(sum(breakdown.values()), 0.0), 1.0)
        threshold = s.flag_threshold
        return {
            "suspicion_score": suspicion,
            "threshold": threshold,
            "flagged": suspicion >= threshold,
            "gait": gait,
            "signals": {"crime_signal": crime_signal, "nervousness": nervousness, "speed_anomaly": speed_anomaly},
            "breakdown": breakdown,
        }
