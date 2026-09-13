"""Stage 2: quantum object/scene classification on the 4-angle PCA features.

Encodes features with a Qiskit ZZFeatureMap and scores class membership by
statevector fidelity against reference class prototypes (a quantum-kernel
classifier) run locally on Aer. Falls back to a classical RBF-kernel stand-in
when qiskit/qiskit-aer isn't installed or a remote QPU endpoint is unset, so
the full pipeline stays testable with zero extra dependencies.
"""
from __future__ import annotations
import json
import os
import numpy as np
import requests

from config import Settings

try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False

_DEFAULT_PROTOTYPES = {
    "person": [0.8, -0.6, 0.3, -0.2],
    "money_object": [-0.9, 0.7, -0.4, 0.5],
    "background": [0.05, 0.05, -0.05, 0.02],
    "suspicious_signature": [2.6, -2.4, 2.1, -1.9],
}


def _load_prototypes(path: str) -> dict[str, np.ndarray]:
    if os.path.isfile(path):
        with open(path) as fh:
            raw = json.load(fh)
    else:
        raw = _DEFAULT_PROTOTYPES
        with open(path, "w") as fh:
            json.dump(raw, fh, indent=2)
    return {name: np.asarray(vec, dtype=np.float32) for name, vec in raw.items()}


class QuantumClassifier:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._prototypes = _load_prototypes(settings.class_prototypes_path)
        self._use_qiskit = settings.use_qiskit and _HAS_QISKIT
        if self._use_qiskit:
            self._feature_map = ZZFeatureMap(feature_dimension=settings.pca_dim, reps=2)
            self._backend = AerSimulator(method="statevector")

    def classify(self, angles: np.ndarray) -> dict:
        if self._settings.quantum_endpoint:
            remote = self._classify_remote(angles)
            if remote is not None:
                return remote
        scores = {name: self._kernel(angles, proto) for name, proto in self._prototypes.items()}
        best = max(scores, key=scores.get)
        return {
            "object_class": best,
            "class_scores": scores,
            "confidence": scores[best],
            "backend": "aer" if self._use_qiskit else "classical_fallback",
        }

    def _kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        if self._use_qiskit:
            sv_x = Statevector.from_instruction(self._feature_map.assign_parameters(x))
            sv_y = Statevector.from_instruction(self._feature_map.assign_parameters(y))
            return float(np.abs(sv_x.inner(sv_y)) ** 2)  # quantum kernel fidelity
        return float(np.exp(-np.sum((x - y) ** 2)))       # classical RBF stand-in

    def _classify_remote(self, angles: np.ndarray) -> dict | None:
        try:
            resp = requests.post(
                self._settings.quantum_endpoint,
                json={"features": angles.tolist()},
                timeout=self._settings.request_timeout_sec,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None  # remote QPU/service unreachable, caller falls back locally
