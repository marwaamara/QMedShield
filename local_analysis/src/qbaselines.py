"""
qbaselines.py -- quantum and related-work comparators.

Compares the hybrid model against quantum baselines and against a related
method (Al-Hawawreh & Hossain (2023), Information Fusion 99:101889), in
addition to the classical machine learning baselines.

Four comparator families, all trained and evaluated on the SAME splits, the
same preprocessing and the same schedule as the headline model, so the rows are
directly comparable:

    qsvm_angle      quantum kernel SVM, product-state angle embedding
    qsvm_iqp        quantum kernel SVM, IQP embedding (genuinely entangling)
    vqc             variational classifier: no re-uploading, NO classical
                    decoder -- the measurement is the head
    amp_qnn         amplitude encoding of all 16 PCA components into 4 qubits
    alhawawreh_qdl  re-implementation of the fusion + quantum-deep-learning
                    framework of Al-Hawawreh & Hossain (2023), under the
                    documented deviations in ALHAWAWREH_DEVIATIONS

Why the kernels are computed from state vectors
-----------------------------------------------
For an embedding |phi(x)>, the fidelity kernel is k(x,y) = |<phi(x)|phi(y)>|^2.
Evaluating that pair by pair on a simulator costs N^2 circuit executions and is
what makes quantum-kernel baselines infeasible at realistic N. But the states
themselves only need N executions: simulate once per sample, stack into
S (N x 2^n), and the whole Gram matrix is |S S^H|^2 -- one matmul.

This is exact, not an approximation, and `verify_kernel_matrix` checks it
against pair-by-pair circuit evaluation on a random sample of pairs before any
result is reported. It is what allows a 3,000-sample QSVM here instead of the
few hundred a pairwise implementation would afford.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from config import (
    ALHAWAWREH_FUSION_DIM,
    ALHAWAWREH_GROUPS,
    ALHAWAWREH_GROUP_DIM,
    ALHAWAWREH_MODALITIES,
    AMP_QNN_LAYERS,
    AMP_QNN_QUBITS,
    CFG,
    LITERATURE_ROWS,
    N_LAYERS,
    N_QUBITS,
    QSVM_C_GRID,
    QSVM_TRAIN_SUBSAMPLE,
    VQC_LAYERS,
    n_classes_for,
)


# =============================================================================
# Feature preparation shared by every quantum comparator
# =============================================================================
def to_angles(X: np.ndarray, n_qubits: int = N_QUBITS,
              bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
    """Map PCA components onto rotation angles in [-pi, pi].

    Angle embedding is only meaningful on a bounded input, and PCA components
    are not bounded. The min/max are fitted on the TRAINING rows and reused
    verbatim for validation and test -- fitting them on the test set would leak,
    which is exactly the class of mistake that evaluation protocols guard against.

    Returns (angles, bounds) so the caller can thread the training bounds
    through to the test split.
    """
    Z = np.asarray(X, dtype=np.float64)[:, :n_qubits]
    if bounds is None:
        lo, hi = Z.min(axis=0), Z.max(axis=0)
        bounds = (lo, hi)
    lo, hi = bounds
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return (2.0 * (np.clip(Z, lo, hi) - lo) / span - 1.0) * np.pi, bounds


def stratified_subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int = 42):
    """Seeded stratified subsample; returns (X, y, idx). Used by the QSVM,
    whose Gram matrix is quadratic in the training-set size."""
    if n is None or len(X) <= n:
        return X, y, np.arange(len(X))
    rng = np.random.default_rng(seed)
    idx: List[int] = []
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum(1, np.round(n * counts / counts.sum()).astype(int))
    for c, q in zip(classes, quota):
        pool = np.flatnonzero(y == c)
        idx.extend(rng.choice(pool, size=min(q, len(pool)), replace=False).tolist())
    idx = np.array(sorted(idx))[:n]
    return X[idx], y[idx], idx


# =============================================================================
# Quantum kernels
# =============================================================================
def _state_qnode(embedding: str, n_qubits: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="numpy")
    def circuit(x):
        if embedding == "angle":
            # Product state: sum_j R_Y(x_j). Same encoding family as the
            # paper's eq. (3), so the kernel row is the kernelised
            # counterpart of our own embedding.
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        elif embedding == "iqp":
            # IQP embedding entangles: its Gram matrix is not a product of
            # single-qubit overlaps, so this row is the one that cannot be
            # written down in closed form.
            qml.IQPEmbedding(x, wires=range(n_qubits), n_repeats=2)
        else:
            raise ValueError(f"unknown embedding: {embedding}")
        return qml.state()

    return circuit


def embed_states(X: np.ndarray, embedding: str = "angle",
                 n_qubits: int = N_QUBITS, batch: int = 512) -> np.ndarray:
    """Simulate |phi(x)> for every row. N executions, not N^2."""
    circuit = _state_qnode(embedding, n_qubits)
    out = np.empty((len(X), 2 ** n_qubits), dtype=np.complex128)
    for s in range(0, len(X), batch):
        chunk = np.asarray(X[s:s + batch], dtype=np.float64)
        out[s:s + batch] = np.asarray(circuit(chunk)).reshape(len(chunk), -1)
    return out


def fidelity_kernel(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """k(x,y) = |<phi(x)|phi(y)>|^2 for pre-computed state matrices."""
    return np.abs(S1 @ S2.conj().T) ** 2


def verify_kernel_matrix(X: np.ndarray, K: np.ndarray, embedding: str,
                         n_qubits: int = N_QUBITS, n_pairs: int = 100,
                         seed: int = 42, tol: float = 1e-9) -> Dict:
    """Check the state-vector Gram matrix against pair-by-pair evaluation.

    The fast path is a mathematical identity, but an identity that is asserted
    and never tested can hide a bug. This tests it.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    def _embed(x):
        if embedding == "angle":
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        else:
            qml.IQPEmbedding(x, wires=range(n_qubits), n_repeats=2)

    @qml.qnode(dev, interface="numpy")
    def overlap(x1, x2):
        _embed(x1)
        qml.adjoint(_embed)(x2)
        return qml.probs(wires=range(n_qubits))

    rng = np.random.default_rng(seed)
    ii = rng.integers(0, len(X), n_pairs)
    jj = rng.integers(0, len(X), n_pairs)
    err = [abs(float(overlap(X[i], X[j])[0]) - float(K[i, j]))
           for i, j in zip(ii, jj)]
    max_err = float(np.max(err))
    return {"embedding": embedding, "n_pairs": int(n_pairs),
            "max_abs_error": max_err, "tolerance": tol, "passed": max_err < tol}


def fit_qsvm(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
             embedding: str = "angle", n_qubits: int = N_QUBITS,
             C_grid: Sequence[float] = tuple(QSVM_C_GRID),
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             seed: int = 42) -> Tuple[object, np.ndarray, np.ndarray, Dict]:
    """Quantum-kernel SVM with a precomputed Gram matrix.

    C is selected on the VALIDATION split when one is supplied -- never on the
    test split (gate G3 in spirit: no test data touches model selection).

    Returns (estimator, K_test, K_train, meta).
    """
    from sklearn.svm import SVC

    t0 = time.time()
    A_tr, bounds = to_angles(X_tr, n_qubits)
    A_te, _ = to_angles(X_te, n_qubits, bounds)
    S_tr = embed_states(A_tr, embedding, n_qubits)
    S_te = embed_states(A_te, embedding, n_qubits)
    K_tr = fidelity_kernel(S_tr, S_tr)
    K_te = fidelity_kernel(S_te, S_tr)
    t_kernel = time.time() - t0

    best_C, best_score = C_grid[0], -1.0
    if X_val is not None and y_val is not None and len(C_grid) > 1:
        A_va, _ = to_angles(X_val, n_qubits, bounds)
        K_va = fidelity_kernel(embed_states(A_va, embedding, n_qubits), S_tr)
        for C in C_grid:
            est = SVC(C=C, kernel="precomputed", probability=False,
                      random_state=seed).fit(K_tr, y_tr)
            score = float((est.predict(K_va) == y_val).mean())
            if score > best_score:
                best_C, best_score = C, score

    t1 = time.time()
    est = SVC(C=best_C, kernel="precomputed", probability=True,
              random_state=seed).fit(K_tr, y_tr)
    meta = {"embedding": embedding, "n_qubits": n_qubits, "C": float(best_C),
            "val_accuracy": best_score if best_score >= 0 else None,
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "kernel_wall_s": t_kernel, "fit_wall_s": time.time() - t1,
            "circuit_evals": int(len(X_tr) + len(X_te)),
            "circuit_evals_pairwise_equivalent":
                int(len(X_tr) ** 2 + len(X_te) * len(X_tr)),
            "n_support_vectors": int(est.support_.size),
            "angle_bounds_fitted_on": "train"}
    return est, K_te, K_tr, meta


# =============================================================================
# Variational classifier -- no classical decoder
# =============================================================================
class VQC(nn.Module):
    """Angle-encoded variational classifier, no re-uploading, no decoder.

    This is the comparator that answers "is the classical encoder/decoder doing
    the work?" from the opposite direction to NB04's parameter-matched controls:
    there we kept the scaffolding and removed the quantum layer, here we keep
    the quantum layer and remove the scaffolding.

    Head: one measured qubit per class. A per-class affine readout (2*C
    parameters) is kept because cross-entropy over raw expectation values in
    [-1, 1] is badly scaled; it is counted and disclosed, not hidden.
    """

    def __init__(self, task: str, n_qubits: int = N_QUBITS,
                 n_layers: int = VQC_LAYERS, n_classes: Optional[int] = None):
        super().__init__()
        self.task = task
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.n_classes = n_classes or n_classes_for(task)
        if self.n_classes > n_qubits:
            raise ValueError("VQC head needs one qubit per class")
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.readout_scale = nn.Parameter(torch.ones(self.n_classes))
        self.readout_bias = nn.Parameter(torch.zeros(self.n_classes))
        self.n_circuit_evals = 0

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1],
                            weights[layer, i, 2], wires=i)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.n_circuit_evals += x.shape[0]
        z = torch.tanh(x[:, :self.n_qubits]) * np.pi   # bounded angle map
        q = torch.stack(list(self.qnode(z, self.q_weights)), dim=-1).to(x.dtype)
        return q[:, :self.n_classes] * self.readout_scale + self.readout_bias

    def param_breakdown(self) -> Dict[str, int]:
        n = lambda t: int(sum(p.numel() for p in t))
        return {"encoder": 0,
                "bottleneck": n([self.q_weights]),
                "decoder": n([self.readout_scale, self.readout_bias]),
                "total": n(list(self.parameters()))}

    def footprint_kb(self, bits: int = 32) -> float:
        return self.param_breakdown()["total"] * (bits / 8) / 1024.0


class AmplitudeQNN(nn.Module):
    """Amplitude encoding: all 16 PCA components into log2(16) = 4 qubits.

    Angle encoding needs one qubit per feature and therefore discards 8 of the
    16 components; amplitude encoding keeps all of them at a quarter of the
    width. The head is `qml.probs` over the first ceil(log2 C) qubits, so this
    comparator too has no classical decoder.
    """

    def __init__(self, task: str, n_features: int = 16,
                 n_qubits: int = AMP_QNN_QUBITS, n_layers: int = AMP_QNN_LAYERS,
                 n_classes: Optional[int] = None):
        super().__init__()
        self.n_classes = n_classes or n_classes_for(task)
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.n_features = n_features
        self.head_wires = int(np.ceil(np.log2(self.n_classes)))
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.readout_scale = nn.Parameter(torch.ones(self.n_classes))
        self.n_circuit_evals = 0

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits),
                                   normalize=True, pad_with=0.0)
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1],
                            weights[layer, i, 2], wires=i)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            return qml.probs(wires=range(self.head_wires))

        self.qnode = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.n_circuit_evals += x.shape[0]
        probs = self.qnode(x, self.q_weights).to(x.dtype)
        # log-probabilities are the natural logits for a probability head
        return torch.log(probs[:, :self.n_classes] + 1e-12) * self.readout_scale

    def param_breakdown(self) -> Dict[str, int]:
        n = lambda t: int(sum(p.numel() for p in t))
        return {"encoder": 0, "bottleneck": n([self.q_weights]),
                "decoder": n([self.readout_scale]),
                "total": n(list(self.parameters()))}

    def footprint_kb(self, bits: int = 32) -> float:
        return self.param_breakdown()["total"] * (bits / 8) / 1024.0


# =============================================================================
# Al-Hawawreh & Hossain (2023) -- re-implementation under stated deviations
# =============================================================================
ALHAWAWREH_DEVIATIONS = {
    "reference": "Al-Hawawreh, M., & Hossain, M. S. (2023). A privacy-aware "
                 "framework for detecting cyber attacks on internet of medical "
                 "things systems using data fusion and quantum deep learning. "
                 "Information Fusion, 99, 101889.",
    "fusion_modalities": (
        "The source framework fuses heterogeneous modalities (patient "
        "biometric telemetry alongside network telemetry). The CICIoMT2024 "
        "WiFi/MQTT partition carries no biometric channel, so the "
        "re-implementation fuses the three FEATURE GROUPS present in the flow "
        "records (volume, timing, protocol) through per-group encoders. This "
        "preserves the architectural pattern -- per-modality encoders into a "
        "shared fused representation -- on the data that exists."),
    "privacy_mechanism": (
        "The privacy-preserving component of the source framework is "
        "orthogonal to detection accuracy and is NOT re-implemented. The row "
        "therefore measures the detection architecture only, and the "
        "results must be reported with this caveat."),
    "hyperparameters": (
        "Layer widths not fixed by the source text are set to this paper's "
        "budget (per-group width 8, fused width 16) so the comparison is at "
        "matched capacity rather than matched-to-an-unknown."),
    "training_protocol": (
        "Identical to every other row: same splits, same optimiser, same "
        "schedule, validation-only checkpoint selection."),
    "status": "faithful re-implementation under the deviations above; NOT the "
              "authors' own reported numbers",
}


class AlHawawrehQDL(nn.Module):
    """Per-group encoders -> fusion -> variational quantum layer -> classifier.

    Deliberately NOT built from `HybridIDS`: it is a different architecture from
    a different paper, and reusing our scaffolding would quietly turn it into a
    variant of our own model. `ALHAWAWREH_DEVIATIONS` travels with every result
    this class produces.
    """

    def __init__(self, task: str, group_sizes: Sequence[int],
                 n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS,
                 n_classes: Optional[int] = None):
        super().__init__()
        self.task = task
        self.group_sizes = list(group_sizes)
        self.n_classes = n_classes or n_classes_for(task)
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.n_circuit_evals = 0

        self.group_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(g, ALHAWAWREH_GROUP_DIM), nn.ReLU(),
                          nn.Dropout(0.2))
            for g in self.group_sizes
        ])
        self.fusion = nn.Sequential(
            nn.Linear(ALHAWAWREH_GROUP_DIM * len(self.group_sizes),
                      ALHAWAWREH_FUSION_DIM), nn.ReLU(),
            nn.Linear(ALHAWAWREH_FUSION_DIM, n_qubits), nn.Tanh(),
        )
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.head = nn.Linear(n_qubits, self.n_classes)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1],
                            weights[layer, i, 2], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts, s = [], 0
        for enc, g in zip(self.group_encoders, self.group_sizes):
            parts.append(enc(x[:, s:s + g]))
            s += g
        z = self.fusion(torch.cat(parts, dim=1))
        self.n_circuit_evals += x.shape[0]
        q = torch.stack(list(self.qnode(z, self.q_weights)), dim=-1).to(x.dtype)
        return self.head(q)

    def param_breakdown(self) -> Dict[str, int]:
        n = lambda m: int(sum(p.numel() for p in m.parameters()
                              if p.requires_grad))
        enc = n(self.group_encoders) + n(self.fusion)
        return {"encoder": enc, "bottleneck": int(self.q_weights.numel()),
                "decoder": n(self.head),
                "total": enc + int(self.q_weights.numel()) + n(self.head)}

    def footprint_kb(self, bits: int = 32) -> float:
        return self.param_breakdown()["total"] * (bits / 8) / 1024.0


def group_feature_matrix(X_raw: np.ndarray, feature_names: Sequence[str],
                         modalities: Optional[Dict[str, Sequence[str]]] = None):
    """Re-order raw features into contiguous per-modality blocks.

    Returns (X_grouped, group_sizes, group_columns, names_by_group).

    The fusion architecture needs a PARTITION: every feature in exactly one
    modality, no modality empty. Both conditions are asserted here rather than
    assumed -- an empty modality would silently create a `Linear(0, 8)` encoder
    that trains to a constant, which is the kind of bug that only shows up
    as an inexplicably weak comparator row.
    """
    mods = modalities or ALHAWAWREH_MODALITIES
    lower = [str(n).lower() for n in feature_names]
    assigned: Dict[int, str] = {}
    cols: Dict[str, List[int]] = {g: [] for g in mods}

    for i, name in enumerate(lower):
        for g, keys in mods.items():          # priority = declaration order
            if any(k in name for k in keys):
                cols[g].append(i)
                assigned[i] = g
                break

    unassigned = [i for i in range(len(feature_names)) if i not in assigned]
    if unassigned:                            # nothing may be silently dropped
        cols.setdefault("other", []).extend(unassigned)

    empty = [g for g, c in cols.items() if not c]
    for g in empty:
        cols.pop(g)
    if empty:
        print(f"[qbaselines] modalities with no matching feature, dropped: {empty}")

    flat = [i for g in cols for i in cols[g]]
    assert sorted(flat) == list(range(len(feature_names))),         "modality assignment is not a partition of the feature set"

    sizes = [len(cols[g]) for g in cols]
    names_by_group = {g: [str(feature_names[i]) for i in c]
                      for g, c in cols.items()}
    return np.asarray(X_raw)[:, flat], sizes, cols, names_by_group


# =============================================================================
# Literature rows
# =============================================================================
def literature_table(rows: Sequence[Dict] = tuple(LITERATURE_ROWS)) -> Dict:
    """Split the related-work rows into transcribed and outstanding.

    A reported number that has not been copied from the cited article is
    reported as OUTSTANDING. It is never guessed, never interpolated from an
    abstract, and never left as a blank table cell.
    """
    done = [r for r in rows if r.get("reported") is not None]
    todo = [r for r in rows if r.get("reported") is None]
    return {"transcribed": done, "outstanding": todo,
            "note": "Reported numbers are on the authors' own datasets and "
                    "splits; they are NOT directly comparable with the "
                    "CICIoMT2024 rows measured here. The comparable row is the "
                    "re-implementation, trained on the same split as every "
                    "other model in this paper."}


def build_quantum_baseline(name: str, task: str, n_features: int = 16,
                           group_sizes: Optional[Sequence[int]] = None):
    """Factory for the trainable quantum comparators (the QSVMs are fitted by
    `fit_qsvm`, which is not a torch module)."""
    if name == "vqc":
        return VQC(task)
    if name == "amp_qnn":
        return AmplitudeQNN(task, n_features=n_features)
    if name == "alhawawreh_qdl":
        if group_sizes is None:
            raise ValueError("alhawawreh_qdl needs group_sizes from "
                             "group_feature_matrix()")
        return AlHawawrehQDL(task, group_sizes)
    raise ValueError(f"unknown quantum baseline: {name}")
