"""Can this model run on real quantum hardware? Measure, don't guess.

The circuit is genuine: ZZFeatureMap(4 qubits, reps=2, linear) with a
compute-uncompute fidelity kernel, which is exactly what you would submit to an
IBM backend. What decides feasibility is not whether it *can* run but three
numbers: transpiled depth on a real coupling map, how many circuits one
classification costs, and how badly finite shots corrupt the kernel.

    .venv/bin/python3 -u src/hardware_report.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quantum_kernel import FEATURE_MAP, encode, kernel_matrix  # noqa: E402

SHOT_GRID = (256, 1024, 4096, 16384)
N_PAIRS = 400          # kernel entries sampled for the shot-noise study
SEED = 0


def transpile_report() -> dict:
    """Depth and gate counts after mapping onto a real device's topology."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    circuit = FEATURE_MAP.compose(FEATURE_MAP.inverse())   # the fidelity circuit
    circuit.measure_all()
    out = {"logical_qubits": FEATURE_MAP.num_qubits,
           "abstract_depth": circuit.depth()}
    try:
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        backend = FakeBrisbane()
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        t = pm.run(circuit)
        ops = dict(t.count_ops())
        out.update({
            "backend": backend.name,
            "device_qubits": backend.num_qubits,
            "transpiled_depth": t.depth(),
            "two_qubit_gates": sum(v for k, v in ops.items() if k in ("cz", "cx", "ecr")),
            "total_gates": sum(ops.values()),
        })
    except Exception as exc:  # noqa: BLE001
        out["backend_error"] = str(exc)[:160]
    return out


def circuits_per_classification() -> dict:
    artifact = joblib.load(ROOT / "models" / "qsvm_model.joblib")
    members = artifact.get("ensemble", [artifact])
    per_member = [len(m["support_vectors"]) for m in members]
    return {"ensemble_members": len(members),
            "support_vectors_per_member": per_member,
            "circuits_per_frame": int(sum(per_member))}


def shot_noise_study() -> dict:
    """Exact fidelity vs what finite sampling actually returns.

    A fidelity kernel entry is the probability of measuring |0000>, so an
    s-shot estimate is Binomial(s, K)/s: unbiased, with standard error
    sqrt(K(1-K)/s). Sampling that directly is equivalent to running the circuit
    on a noiseless device and far cheaper than simulating 400 circuits four times.
    """
    rng = np.random.default_rng(SEED)
    artifact = joblib.load(ROOT / "models" / "qsvm_model.joblib")
    member = artifact.get("ensemble", [artifact])[0]
    sv = member["support_vectors"]
    idx = rng.choice(len(sv), size=int(np.sqrt(N_PAIRS)), replace=False)
    exact = kernel_matrix(encode(sv[idx]), encode(sv[idx]))

    out = {}
    for shots in SHOT_GRID:
        est = rng.binomial(shots, np.clip(exact, 0, 1)) / shots
        out[str(shots)] = {
            "mean_abs_error": float(np.abs(est - exact).mean()),
            "max_abs_error": float(np.abs(est - exact).max()),
        }
    return out


def end_to_end_shot_impact() -> dict:
    """Does that kernel error actually change the verdicts? Re-score Test clips."""
    import experiments as E
    import train_qsvm as T
    from cnn_extractor import embed_batch

    sys.path.insert(0, str(ROOT))
    from inference import LiveTheftClassifier

    clf = LiveTheftClassifier()
    paths, y, _, videos = T.sample_by_video("Test", 12, np.random.default_rng(5))
    images = E._load(paths, False).astype(np.float32) / 255.0
    rng = np.random.default_rng(SEED)

    how = clf.metadata.get("clip_aggregation", "mean")
    thr = float(clf.metadata.get("clip_threshold", 0.0))
    results = {}

    for shots in (None,) + SHOT_GRID:
        margins = []
        for m in clf._members:
            lat = np.concatenate([embed_batch(images[i:i + 512], m["cnn"])
                                  for i in range(0, len(images), 512)])
            ang = m["projector"].transform(lat)
            K = kernel_matrix(encode(ang), m["states"])
            if shots is not None:                       # simulate finite sampling
                K = rng.binomial(shots, np.clip(K, 0, 1)) / shots
            full = np.zeros((len(ang), m["n_train"]))
            full[:, m["clf"].support_] = K
            margins.append(m["clf"].decision_function(full))
        ens = np.mean(margins, axis=0)
        lab, sc = E.aggregate(list(videos), y, ens, how)
        rep = E.clip_report(lab, sc, thr)
        results["exact" if shots is None else str(shots)] = {
            "clip_accuracy": round(rep["accuracy"], 4),
            "roc_auc": round(rep["roc_auc"], 4),
        }
    return results


def main() -> None:
    t0 = time.time()
    report = {
        "circuit": transpile_report(),
        "cost": circuits_per_classification(),
        "kernel_shot_error": shot_noise_study(),
    }
    print(json.dumps(report, indent=2), flush=True)

    print("\nre-scoring Test clips under finite sampling ...", flush=True)
    report["accuracy_vs_shots"] = end_to_end_shot_impact()
    print(json.dumps(report["accuracy_vs_shots"], indent=2))

    c = report["cost"]
    circuits = c["circuits_per_frame"]
    for shots in (1024, 4096):
        total = circuits * shots
        print(f"\n{shots} shots -> {total:,} circuit executions per single frame")
    (ROOT / "reports" / "hardware_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote reports/hardware_report.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
