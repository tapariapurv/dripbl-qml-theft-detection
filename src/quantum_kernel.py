"""4-qubit fidelity quantum kernel on a ZZFeatureMap(reps=2, linear entanglement).

K[i, j] = |<phi(x_i)|phi(x_j)>|^2.

Two ways to get it, same definition:

* `encode(X)` + `kernel_matrix(...)` — encode each sample ONCE into its 16-amplitude
  statevector, then take all pairwise overlaps as a single complex matmul. Exact
  (no shot noise) and O(N) circuit simulations instead of O(N^2) circuit
  executions, which is what makes training on thousands of samples possible.
* `build_noisy_kernel()` — the shots-based FidelityQuantumKernel backed by
  FakeBrisbane's noise model. Hardware-faithful but ~1000x slower and adds
  sampling noise to every entry; kept for hardware-realism experiments.
"""
from __future__ import annotations
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

FEATURE_DIM = 4
FEATURE_MAP = ZZFeatureMap(feature_dimension=FEATURE_DIM, reps=2, entanglement="linear")


def encode_qiskit(X: np.ndarray) -> np.ndarray:
    """(N, 4) angles -> (N, 16) amplitudes, simulated one circuit at a time.
    Reference implementation; `encode` uses the verified analytic equivalent."""
    X = np.atleast_2d(X)
    return np.array(
        [Statevector.from_instruction(FEATURE_MAP.assign_parameters(x)).data for x in X],
        dtype=np.complex128,
    )


def encode(X: np.ndarray) -> np.ndarray:
    """(N, 4) angles -> (N, 16) complex statevector amplitudes (vectorised)."""
    return fast_encode(X)


def kernel_matrix(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Fidelity Gram matrix from encoded statevectors: |A B^dagger|^2."""
    B = A if B is None else B
    return np.abs(A @ B.conj().T) ** 2


def quantum_kernel(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Gram matrix straight from angle features (encodes, then overlaps)."""
    A = encode(X)
    return kernel_matrix(A, A if Y is None else encode(Y))


# --- backwards-compatible aliases used by src/inference.py -------------------
encode_statevectors = encode


def fast_kernel_vector(x: np.ndarray, ref_statevectors: np.ndarray) -> np.ndarray:
    """Kernel row between one sample's angles and precomputed reference states."""
    return kernel_matrix(encode(x), ref_statevectors)[0]


def build_noisy_kernel():
    """Shots-based FidelityQuantumKernel on AerSimulator.from_backend(FakeBrisbane()),
    falling back to a clean Aer simulator offline. Hardware-faithful, very slow."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.state_fidelities import ComputeUncompute

    try:
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        backend = AerSimulator.from_backend(FakeBrisbane())
    except Exception:
        backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    fidelity = ComputeUncompute(sampler=SamplerV2.from_backend(backend), pass_manager=pm)
    return FidelityQuantumKernel(feature_map=FEATURE_MAP, fidelity=fidelity)


# --- analytic, vectorised encoding ------------------------------------------
# A ZZFeatureMap layer is H^(x)n followed by a DIAGONAL phase unitary, so for
# reps=2 the statevector is exactly  D(x) H D(x) H |0>.  Writing that out lets us
# encode a whole batch with two small matmuls instead of simulating one circuit
# per sample -- the per-sample loop above is what made large-scale evaluation
# slow.  `verify_fast_encode()` checks it against qiskit.

_N_QUBITS = FEATURE_DIM
_DIM = 2 ** _N_QUBITS
_BITS = ((np.arange(_DIM)[:, None] >> np.arange(_N_QUBITS)[None, :]) & 1).astype(np.float64)  # (16, 4)
_PAIRS = [(i, i + 1) for i in range(_N_QUBITS - 1)]  # linear entanglement
_XOR = np.stack([np.abs(_BITS[:, i] - _BITS[:, j]) for i, j in _PAIRS], axis=1)  # (16, 3)
_H = np.array([[(-1.0) ** bin(a & b).count("1") for b in range(_DIM)] for a in range(_DIM)]) / np.sqrt(_DIM)


def fast_encode(X: np.ndarray) -> np.ndarray:
    """(N, 4) angles -> (N, 16) statevectors, vectorised over the batch."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    single = X @ _BITS.T                                              # (N, 16)
    pair_coeff = np.stack([(np.pi - X[:, i]) * (np.pi - X[:, j]) for i, j in _PAIRS], axis=1)
    phases = 2.0 * (single + pair_coeff @ _XOR.T)
    d = np.exp(1j * phases)                                           # diagonal layer
    v = d / np.sqrt(_DIM)                                             # D . (H|0>)
    return d * (v @ _H.T)                                             # D . H . that


def verify_fast_encode(n: int = 8, tol: float = 1e-9) -> float:
    """Max kernel discrepancy between the analytic encoder and qiskit's simulator."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-np.pi, np.pi, (n, FEATURE_DIM))
    delta = float(np.max(np.abs(kernel_matrix(fast_encode(X)) - kernel_matrix(encode_qiskit(X)))))
    assert delta < tol, f"analytic encoder disagrees with qiskit by {delta:.2e}"
    return delta
