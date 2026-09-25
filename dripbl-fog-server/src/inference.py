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
        # format 2 (train_xd.py): any backbone from src/backbones.py, optionally with the
        # 3 motion channels (frame minus the frame `motion_gap_sec` earlier)
        spec = artifact.get("backbone") if artifact.get("format") == 2 else None
        self.in_channels = int(spec["channels"]) if spec else in_ch
        self.motion_gap_sec = float(self.metadata.get("motion_gap_sec", 1 / 3))
        self._members = []
        for member in members:
            mspec = member.get("backbone", spec)  # combined ensembles mix backbones per member
            if mspec:
                from backbones import Backbone
                cnn = Backbone(mspec["kind"], int(mspec["channels"]), int(mspec["input_size"]))
                cnn.load_state_dict(member["backbone_state"])
            else:
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
                # members from different models are put on one scale before averaging
                "shift": float(member.get("margin_shift", 0.0)), "scale": float(member.get("margin_scale", 1.0)),
            })
        torch.set_num_threads(1)  # edge node: predictable single-thread latency

    @staticmethod
    def small_rgb(frame_bgr: np.ndarray) -> np.ndarray:
        """Camera frame -> 64x64 uint8 RGB, the resolution every model sees."""
        if frame_bgr.shape[:2] != (FRAME_SIZE, FRAME_SIZE):
            frame_bgr = cv2.resize(frame_bgr, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _preprocess(self, frame_bgr: np.ndarray, prev_bgr: np.ndarray | None = None) -> np.ndarray:
        """-> (1, 64, 64, C) float in [0, 1]. Motion models without a previous frame get a
        neutral (no-motion) motion image rather than failing."""
        rgb = self.small_rgb(frame_bgr)
        if self.in_channels == 6:
            if prev_bgr is None:
                motion = np.full_like(rgb, 128)
            else:
                prev = self.small_rgb(prev_bgr).astype(np.int16)
                motion = np.clip((rgb.astype(np.int16) - prev) // 2 + 128, 0, 255).astype(np.uint8)
            rgb = np.concatenate([rgb, motion], -1)
        return (rgb.astype(np.float32) / 255.0)[None]

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
        return (member["clf"].decision_function(K) - member["shift"]) / member["scale"], angles, latents

    def margins_from_arrays(self, x_uint8: np.ndarray, batch: int = 512) -> np.ndarray:
        """Ensemble frame margins for ready-made 64x64 inputs (N, 64, 64, C) uint8 RGB[+motion]:
        the evaluation path, identical maths to predict_batch without the camera-frame resize."""
        out = []
        for s in range(0, len(x_uint8), batch):
            chunk = x_uint8[s:s + batch, ..., :self.in_channels].astype(np.float32) / 255.0
            out.append(np.mean([self._member_margins(chunk, m)[0] for m in self._members], axis=0))
        return np.concatenate(out)

    def predict_batch(self, frames_bgr: list[np.ndarray] | np.ndarray, detail: bool = False,
                      prev_bgr: list[np.ndarray | None] | None = None) -> list[dict]:
        """`detail=True` adds per-member margins and the 128-d latent (live node only;
        skipped for batch evaluation, where 20k latents as lists would waste memory)."""
        if self.clip_only:
            raise RuntimeError(
                "this artifact is a clip-descriptor model (metadata.clip_qsvm=True): it scores a "
                "whole clip, not single frames — use predict_clip(), or train a frame-level model"
            )
        prevs = prev_bgr if prev_bgr is not None else [None] * len(frames_bgr)
        batch = np.concatenate([self._preprocess(f, p) for f, p in zip(frames_bgr, prevs)])
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

    def predict_frame(self, frame_bgr: np.ndarray, prev_bgr: np.ndarray | None = None) -> dict:
        t0 = time.perf_counter()
        result = self.predict_batch([frame_bgr], detail=True, prev_bgr=[prev_bgr])[0]
        result["latency_ms"] = (time.perf_counter() - t0) * 1e3
        return result
