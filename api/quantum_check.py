"""Quantum Lab: re-evaluate the ZZFeatureMap fidelity kernel for recent live trials
on a real-circuit backend and compare it with the exact kernel the node uses.

Backends:
  * "exact"         closed-form statevector (what the fog node deploys)
  * "fake_brisbane" shots-based circuits on AerSimulator.from_backend(FakeBrisbane):
                    127-qubit heavy-hex topology, native gates, T1/T2 + gate errors
  * "ibm"           a real IBM Quantum device via qiskit-ibm-runtime (needs a token;
                    jobs queue on IBM's side, so this can take minutes to hours)

This checks the *kernel*, which is where hardware noise enters. Reclassifying a
frame on hardware would cost ~3,900 circuits (one per support vector) - see
reports/hardware_report.json for why that is not the production path.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import numpy as np

from quantum_kernel import FEATURE_MAP, encode, kernel_matrix

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()
MAX_TRIALS = 12  # n(n-1)/2 circuits: 12 trials = 66 circuits


def _backend(name: str, qcfg: dict[str, Any]):
    from qiskit_aer import AerSimulator

    if name == "fake_brisbane":
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        return AerSimulator.from_backend(FakeBrisbane()), "sim"
    if name == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService
        if not qcfg["ibm_token"]:
            raise ValueError("set quantum.ibm_token in Settings to use real IBM hardware")
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=qcfg["ibm_token"],
                                       instance=qcfg["ibm_instance"] or None)
        dev = service.backend(qcfg["ibm_backend"]) if qcfg["ibm_backend"] else \
            service.least_busy(operational=True, simulator=False)
        return dev, "ibm"
    raise ValueError(f"unknown backend {name!r}")


def _circuit_kernel(X: np.ndarray, backend_name: str, shots: int, qcfg: dict[str, Any]) -> tuple[np.ndarray, dict]:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.state_fidelities import ComputeUncompute

    backend, kind = _backend(backend_name, qcfg)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    if kind == "sim":
        from qiskit_aer.primitives import SamplerV2
        sampler = SamplerV2.from_backend(backend, default_shots=shots)
    else:
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = shots
    fidelity = ComputeUncompute(sampler=sampler, pass_manager=pm)
    kernel = FidelityQuantumKernel(feature_map=FEATURE_MAP, fidelity=fidelity)
    circuit = pm.run(FEATURE_MAP.compose(FEATURE_MAP.inverse()).measure_all(inplace=False))
    info = {"backend": getattr(backend, "name", str(backend)), "transpiled_depth": circuit.depth(),
            "two_qubit_gates": sum(v for k, v in circuit.count_ops().items() if k in ("cz", "cx", "ecr"))}
    return kernel.evaluate(X), info


def _run(job_id: str, X: np.ndarray, trial_ids: list[str], backend: str, shots: int, qcfg: dict[str, Any]) -> None:
    job = _JOBS[job_id]
    try:
        t0 = time.time()
        exact = kernel_matrix(encode(X))
        job["exact"] = exact.round(4).tolist()
        job["exact_seconds"] = time.time() - t0
        if backend == "exact":
            job.update(status="done", finished=time.time())
            return
        job["status"] = "running circuits"
        t0 = time.time()
        noisy, info = _circuit_kernel(X, backend, shots, qcfg)
        off = ~np.eye(len(X), dtype=bool)  # the diagonal is 1 by definition on both sides
        err = np.abs(noisy - exact)[off]
        job.update(
            status="done", finished=time.time(), circuit_seconds=time.time() - t0,
            circuit=noisy.round(4).tolist(), backend_info=info,
            mean_abs_error=float(err.mean()), max_abs_error=float(err.max()),
            correlation=float(np.corrcoef(noisy[off], exact[off])[0, 1]) if len(X) > 2 else None,
            circuits_run=int(len(X) * (len(X) - 1) / 2),
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI as the job's error
        job.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500], finished=time.time())


def start(features: list[list[float]], trial_ids: list[str], backend: str, shots: int,
          qcfg: dict[str, Any]) -> str:
    X = np.asarray(features, dtype=float)
    if X.ndim != 2 or X.shape[1] != 4 or not 2 <= len(X) <= MAX_TRIALS:
        raise ValueError(f"need 2..{MAX_TRIALS} trials with 4-d features")
    job_id = uuid.uuid4().hex[:12]
    with _LOCK:
        _JOBS[job_id] = {"id": job_id, "status": "queued", "backend": backend, "shots": shots,
                         "trial_ids": trial_ids, "features": X.round(4).tolist(), "started": time.time()}
    threading.Thread(target=_run, args=(job_id, X, trial_ids, backend, shots, qcfg), daemon=True).start()
    return job_id


def get(job_id: str) -> dict[str, Any] | None:
    return _JOBS.get(job_id)


def recent(limit: int = 10) -> list[dict[str, Any]]:
    jobs = sorted(_JOBS.values(), key=lambda j: j["started"], reverse=True)[:limit]
    return [{k: j.get(k) for k in ("id", "status", "backend", "shots", "started", "finished",
                                   "mean_abs_error", "error")} for j in jobs]
