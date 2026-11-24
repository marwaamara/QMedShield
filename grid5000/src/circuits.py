"""
qmed.circuits — parameterized quantum circuits with composable ablation flags.

Every circuit variant used in the package is produced by ONE builder, so the
ablation variants cannot drift away from the headline architecture.

Entanglement scales (section 3.3 of the paper):
    l = 1  ring CNOT      -- all layers
    l = 2  skip-one CZ    -- even layers
    l = 4  cross-half CNOT-- odd layers

The ablation varies the ACTIVE SCALE SET to give a per-scale attribution of
the entanglement structure.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pennylane as qml

from config import CFG


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def select_backend(n_qubits: int, shots: int | None = None) -> Tuple[str, str]:
    """Return (backend_name, diff_method).

    Backend choice is MEASURED, not assumed -- see bench_backend.py (P0.7) and
    the note in qmed.config.QuantumConfig. default.qubit + backprop is 8.3x
    faster than lightning.qubit + adjoint at n_q = 8, because the batch stays
    in one broadcast torch tensor instead of being executed sample by sample.

    Whatever this resolves to is recorded in every result JSON and must match
    what section 4.1 of the paper reports.

    Finite shots are inference-only; see SHOTS_TRAINING_SUPPORTED.
    """
    if shots is not None:
        # Sampling device for the finite-shot experiment. Gradients are not required here.
        for backend in CFG.quantum.BACKEND_PREFERENCE + (CFG.quantum.BACKEND_FALLBACK,):
            try:
                qml.device(backend, wires=n_qubits, shots=shots)
                return backend, "parameter-shift"
            except Exception:
                continue
        raise RuntimeError(f"no shot-capable backend for n_qubits={n_qubits}")

    for backend in CFG.quantum.BACKEND_PREFERENCE:
        try:
            qml.device(backend, wires=n_qubits)
            return backend, CFG.quantum.DIFF_METHOD_PREFERRED
        except Exception:
            continue
    return CFG.quantum.BACKEND_FALLBACK, CFG.quantum.DIFF_METHOD_FALLBACK


def make_device(n_qubits: int, shots: int | None = None):
    backend, diff = select_backend(n_qubits, shots)
    dev = qml.device(backend, wires=n_qubits, shots=shots)
    return dev, backend, diff


# ---------------------------------------------------------------------------
# Entanglement blocks
# ---------------------------------------------------------------------------
def _apply_entanglement(layer: int, n_qubits: int, scales: Sequence[int]) -> None:
    """Apply the composite multi-scale entangler for the requested scales.

    `scales` is a subset of {1, 2, 4}. The full architecture uses all three;
    The ring_only / ring_skip / ring_cross ablation variants use subsets, which is what
    makes the per-scale contribution measurable.
    """
    if 1 in scales:
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

    # Scales 2 and 4 alternate by layer parity so that consecutive layers do
    # not repeat the same connectivity (section 3.3 rationale in the paper).
    if 2 in scales and layer % 2 == 0:
        for i in range(0, n_qubits, 2):
            qml.CZ(wires=[i, (i + 2) % n_qubits])

    if 4 in scales and layer % 2 == 1:
        for i in range(n_qubits // 2):
            qml.CNOT(wires=[i, i + n_qubits // 2])


# ---------------------------------------------------------------------------
# The single circuit builder
# ---------------------------------------------------------------------------
def build_qnode(
    dev,
    n_qubits: int,
    n_layers: int,
    diff_method: str,
    use_reuploading: bool = True,
    entangle_scales: Sequence[int] = (1, 2, 4),
):
    """Return a batch-broadcasting PennyLane QNode.

    Parameters
    ----------
    use_reuploading
        True  -> angle embedding at EVERY layer (full architecture)
        False -> angle embedding at layer 0 only (`no_reupload` variant)
    entangle_scales
        ()        -> no entanglement           (`no_entangle`)
        (1,)      -> ring only                 (`ring_only`)
        (1,2)     -> ring + skip-one           (`ring_skip`)
        (1,4)     -> ring + cross-half         (`ring_cross`)
        (1,2,4)   -> full composite entangler  (headline)

    Inputs may be shaped (n_qubits,) or (B, n_qubits); PennyLane parameter
    broadcasting vectorises the batch inside the simulator, which is the
    single largest speedup available on CPU.
    """
    scales = tuple(entangle_scales)

    # --- ENCODING CONVENTION (verified against the archived weights) -------
    # The paper, eq. (3.3), defines  AngleEmbed(z) = (x) R_Y(z_j)  with NO
    # pi factor, and the shared_config.py that actually trained the archived
    # Grid'5000 models uses `qml.AngleEmbedding(inputs, rotation='Y')`, i.e.
    # R_Y(z_j) exactly.
    #
    # `smote_correction_optimized_V2/shared_config.py` silently introduced a
    # pi scaling, `qml.RY(inputs[..., i] * np.pi)`, which is a DIFFERENT model:
    # loading the archived weights into it gives 2.9% binary accuracy against
    # the published 99.63%. Do not reintroduce the pi factor -- ANGLE_SCALE
    # exists only to make the convention explicit and testable.
    ANGLE_SCALE = 1.0

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        for layer in range(n_layers):
            if use_reuploading or layer == 0:
                for i in range(n_qubits):
                    qml.RY(inputs[..., i] * ANGLE_SCALE, wires=i)

            for i in range(n_qubits):
                qml.Rot(
                    weights[layer, i, 0],
                    weights[layer, i, 1],
                    weights[layer, i, 2],
                    wires=i,
                )

            if scales:
                _apply_entanglement(layer, n_qubits, scales)

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def n_quantum_params(n_qubits: int, n_layers: int) -> int:
    """|theta_q| = 3 * L * n_q. Used by the parameter-budget table."""
    return 3 * n_layers * n_qubits


def circuit_evals_per_forward(batch_size: int = 1) -> int:
    """Circuit evaluations per forward pass -- feeds the cost model."""
    return batch_size


def circuit_evals_per_gradient(n_qubits: int, n_layers: int) -> int:
    """Parameter-shift gradient cost: 2 evaluations per trainable quantum
    parameter. This is the number that makes 'parameter count is not
    deployment cost' concrete."""
    return 2 * n_quantum_params(n_qubits, n_layers)


# ---------------------------------------------------------------------------
# Variant registry used by the ablation
# ---------------------------------------------------------------------------
CIRCUIT_VARIANTS = {
    "full":         dict(use_reuploading=True,  entangle_scales=(1, 2, 4)),
    "no_reupload":  dict(use_reuploading=False, entangle_scales=(1, 2, 4)),
    "no_entangle":  dict(use_reuploading=True,  entangle_scales=()),
    "ring_only":    dict(use_reuploading=True,  entangle_scales=(1,)),
    "ring_skip":    dict(use_reuploading=True,  entangle_scales=(1, 2)),
    "ring_cross":   dict(use_reuploading=True,  entangle_scales=(1, 4)),
}
