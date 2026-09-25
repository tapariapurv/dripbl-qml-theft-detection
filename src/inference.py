"""Real-time theft classification for a single OpenCV frame, <150ms budget.

Loads the joblib QSVM artifact once (trained CNN weights included), then per
frame: preprocess -> CNN(128) -> PCA+scale(4) -> exact statevector fidelity
kernel against only the stored support vectors -> SVC decision.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn_extractor import embed_batch, load_cnn  # noqa: E402
from quantum_kernel import encode, kernel_matrix  # noqa: E402
from pca_projector import PCAAngleProjector  # noqa: E402  (needed to unpickle the artifact)

_LABELS = {0: "Normal", 1: "Anomaly"}
FRAME_SIZE = 64
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "qsvm_model.joblib"


class LiveTheftClassifier:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL) -> None:
        artifact = joblib.load(model_path)
        self.metadata = artifact.get("metadata", {})
        in_ch = int(self.metadata.get("in_channels", 3))
        # An ensemble artifact carries one member per seed; a single model is
        # just an ensemble of one, so the scoring path below is shared.
        # A clip-descriptor model scores pooled [mean|std|max] statistics of a whole
        # clip, so it has no meaning for one isolated frame. Say so plainly instead
        # of failing later inside sklearn with a feature-count mismatch.
        self.clip_only = bool(self.metadata.get("clip_qsvm", False))
        members = artifact.get("ensemble") or [artifact]
        self._members = []
        for member in members:
            cnn = load_cnn(in_channels=in_ch)
            if "cnn_state_dict" in member:
                cnn.load_state_dict(member["cnn_state_dict"])
            cnn.eval()
            self._members.append({
                "cnn": cnn,
                "clf": member["svm_classifier"],
                "projector": member["pca_transformer"],
                "states": encode(member["support_vectors"]),
                "n_train": int(member["svm_classifier"].shape_fit_[0]),
            })
        torch.set_num_threads(1)  # edge node: predictable single-thread latency

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.shape[:2] != (FRAME_SIZE, FRAME_SIZE):
            frame_bgr = cv2.resize(frame_bgr, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return (rgb.astype(np.float32) / 255.0)[None]  # (1, 64, 64, 3)

    def features(self, frames_bgr: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """Batch path: raw BGR frames -> (N, 4) quantum angle features (first member)."""
        batch = np.concatenate([self._preprocess(f) for f in frames_bgr])
        member = self._members[0]
        return member["projector"].transform(embed_batch(batch, member["cnn"]))

    def _member_margins(self, batch: np.ndarray, member: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        latents = embed_batch(batch, member["cnn"])
        angles = member["projector"].transform(latents)
        K = np.zeros((len(angles), member["n_train"]), dtype=np.float64)
        K[:, member["clf"].support_] = kernel_matrix(encode(angles), member["states"])
        return member["clf"].decision_function(K), angles, latents

    def predict_batch(self, frames_bgr: list[np.ndarray] | np.ndarray, detail: bool = False) -> list[dict]:
        """`detail=True` adds per-member margins and the 128-d latent (live node only;
        skipped for batch evaluation, where 20k latents as lists would waste memory)."""
        if self.clip_only:
            raise RuntimeError(
                "this artifact is a clip-descriptor model (metadata.clip_qsvm=True): it scores a "
                "whole clip, not single frames — use predict_clip(), or train a frame-level model"
            )
        batch = np.concatenate([self._preprocess(f) for f in frames_bgr])
        per_member = [self._member_margins(batch, m) for m in self._members]
        member_margins = np.array([m for m, _, _ in per_member])  # (members, N)
        margins = member_margins.mean(axis=0)  # ensemble = mean margin
        angles, latents = per_member[0][1], per_member[0][2]
        preds = (margins > 0).astype(int)
        out = [
            {
                "prediction": _LABELS[int(p)],
                "confidence": float(1.0 / (1.0 + np.exp(-abs(m)))),
                "margin": float(m),
                "features_4d": a.tolist(),
            }
            for p, m, a in zip(preds, margins, angles)
        ]
        if detail:
            for i, r in enumerate(out):
                r["member_margins"] = member_margins[:, i].tolist()
                r["latent"] = latents[i].tolist()  # first member's 128-d CNN latent
        return out

    def predict_clip(self, frames_bgr: list[np.ndarray]) -> dict:
        """One verdict for a sequence of frames from the same scene — how the live
        node should decide, using the clip aggregation + threshold chosen at training."""
        if self.clip_only:
            batch = np.concatenate([self._preprocess(f) for f in frames_bgr])
            member = self._members[0]
            latents = embed_batch(batch, member["cnn"])
            desc = np.concatenate([latents.mean(0), latents.std(0), latents.max(0)])[None]
            angles = member["projector"].transform(desc)
            K = np.zeros((1, member["n_train"]), dtype=np.float64)
            K[:, member["clf"].support_] = kernel_matrix(encode(angles), member["states"])
            score = float(member["clf"].decision_function(K)[0])
            threshold = float(self.metadata.get("clip_threshold", 0.0))
            return {"prediction": "Anomaly" if score > threshold else "Normal", "score": score,
                    "threshold": threshold, "aggregation": "clip_descriptor",
                    "n_frames": len(frames_bgr), "anomaly_frame_rate": float("nan")}

        margins = np.array([r["margin"] for r in self.predict_batch(frames_bgr)])
        how = self.metadata.get("clip_aggregation", "mean")
        ranked = np.sort(margins)[::-1]
        if how == "max":
            score = float(ranked[0])
        elif how == "median":
            score = float(np.median(ranked))
        elif how in ("top10", "top25"):
            k = max(1, int(len(ranked) * (0.10 if how == "top10" else 0.25)))
            score = float(ranked[:k].mean())
        else:
            score = float(ranked.mean())
        threshold = float(self.metadata.get("clip_threshold", 0.0))
        return {
            "prediction": "Anomaly" if score > threshold else "Normal",
            "score": score, "threshold": threshold, "aggregation": how,
            "n_frames": len(margins),
            "anomaly_frame_rate": float((margins > 0).mean()),
        }

    def predict_frame(self, frame_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()
        result = self.predict_batch([frame_bgr], detail=True)[0]
        result["latency_ms"] = (time.perf_counter() - t0) * 1e3
        return result
