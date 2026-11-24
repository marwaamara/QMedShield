"""
qmed.models — ONE model factory for the headline models and every ablation.

Every model variant (headline models and ablations) is built by this single
factory, so parameter counts differ only where the bottleneck differs.

EVERY variant is `HybridIDS` with identical encoder and decoder; only
`bottleneck` changes. `assert_scaffolding_identical()` enforces this and is
called before any ablation training (validation gate G4), so a scaffolding
mismatch cannot be introduced silently.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from circuits import (
    CIRCUIT_VARIANTS,
    build_qnode,
    make_device,
    n_quantum_params,
)
from config import CFG, n_classes_for


# ---------------------------------------------------------------------------
# Bottleneck modules -- all map R^{n_q} -> R^{n_q}
# ---------------------------------------------------------------------------
class QuantumBottleneck(nn.Module):
    """The 8-qubit PQC. 3*L*n_q trainable parameters (96 at L=4, n_q=8)."""

    def __init__(self, n_qubits: int, n_layers: int, use_reuploading: bool = True,
                 entangle_scales: Sequence[int] = (1, 2, 4), shots: Optional[int] = None):
        super().__init__()
        self.n_qubits, self.n_layers, self.shots = n_qubits, n_layers, shots
        self.dev, self.backend, self.diff_method = make_device(n_qubits, shots)
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.qnode = build_qnode(
            self.dev, n_qubits, n_layers, self.diff_method,
            use_reuploading=use_reuploading, entangle_scales=entangle_scales,
        )
        self.n_circuit_evals = 0  # incremented per forward; feeds the cost model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.n_circuit_evals += x.shape[0]
        q = self.qnode(x, self.q_weights)
        # PennyLane returns float64 expectation values; cast back so the
        # classical decoder (float32) and torch.compile stay happy.
        return torch.stack(list(q), dim=-1).to(x.dtype)


class LinearBottleneck(nn.Module):
    """Dense n_q -> n_q. 72 params at n_q=8. The tightest classical control
    for the PQC (matched-control Family A)."""

    def __init__(self, n_qubits: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_qubits, n_qubits), nn.Tanh())

    def forward(self, x):
        return self.net(x)


class FourierBottleneck(nn.Module):
    """Random Fourier features: matches the PQC's smooth trigonometric map
    without any quantum mechanics. Trainable frequencies + phases give 96
    parameters at n_q=8, D=... (chosen to match |theta_q| exactly).

    This is the control that distinguishes 'the quantum circuit helps' from
    'a smooth periodic bottleneck helps'.
    """

    def __init__(self, n_qubits: int, n_params_target: int = 96):
        super().__init__()
        # freq: (D, n_q), phase: (D,) -> D*(n_q+1) params; pick D to match.
        D = max(1, n_params_target // (n_qubits + 1))
        self.freq = nn.Parameter(torch.randn(D, n_qubits) * 0.1)
        self.phase = nn.Parameter(torch.zeros(D))
        self.proj = nn.Linear(D, n_qubits, bias=False)
        self.proj.weight.requires_grad_(False)  # fixed random readout
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(D))

    def forward(self, x):
        h = torch.cos(x @ self.freq.T * math.pi + self.phase)
        return torch.tanh(self.proj(h))


class MLP96Bottleneck(nn.Module):
    """Tiny MLP tuned to ~96 trainable parameters (matched-control Family A)."""

    def __init__(self, n_qubits: int, n_params_target: int = 96):
        super().__init__()
        # n_q*h + h + h*n_q + n_q = 2*n_q*h + h + n_q  ->  solve for h
        h = max(1, (n_params_target - n_qubits) // (2 * n_qubits + 1))
        self.net = nn.Sequential(
            nn.Linear(n_qubits, h), nn.GELU(), nn.Linear(h, n_qubits), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class IdentityBottleneck(nn.Module):
    """Encoder -> decoder directly. 0 parameters (matched-control Family A)."""

    def forward(self, x):
        return x


def build_bottleneck(kind: str, n_qubits: int, n_layers: int,
                     use_reuploading: bool = True,
                     entangle_scales: Sequence[int] = (1, 2, 4),
                     shots: Optional[int] = None) -> nn.Module:
    if kind == "pqc":
        return QuantumBottleneck(n_qubits, n_layers, use_reuploading,
                                 entangle_scales, shots)
    if kind == "linear":
        return LinearBottleneck(n_qubits)
    if kind == "fourier":
        return FourierBottleneck(n_qubits, n_quantum_params(n_qubits, n_layers))
    if kind == "mlp96":
        return MLP96Bottleneck(n_qubits, n_quantum_params(n_qubits, n_layers))
    if kind == "identity":
        return IdentityBottleneck()
    raise ValueError(f"unknown bottleneck: {kind}")


# ---------------------------------------------------------------------------
# Encoder / decoder builders
# ---------------------------------------------------------------------------
# Extracted verbatim from HybridIDS.__init__ so that the hierarchical and
# feature-path models of NB15 reuse the SAME scaffolding instead of
# copying it. The module names and layer order are unchanged, so archived
# state_dicts still load bit-exactly (NB00 verifies this on every run).
def build_encoder(task: str, n_features: int, n_qubits: int) -> nn.Sequential:
    """Classical encoder. Determined ONLY by `task` -- never by a variant."""
    a = CFG.arch
    if task == "binary":
        (h1,) = a.ENC_BINARY
        return nn.Sequential(
            nn.Linear(n_features, h1), nn.LayerNorm(h1), nn.GELU(),
            nn.Dropout(a.DROPOUT_ENC), nn.Linear(h1, n_qubits), nn.Tanh(),
        )
    h1, h2 = a.ENC_MULTICLASS
    return nn.Sequential(
        nn.Linear(n_features, h1), nn.LayerNorm(h1), nn.GELU(),
        nn.Dropout(a.DROPOUT_ENC),
        nn.Linear(h1, h2), nn.LayerNorm(h2), nn.GELU(),
        nn.Linear(h2, n_qubits), nn.Tanh(),
    )


def build_decoder(task: str, n_in: int, n_classes: int) -> nn.Sequential:
    """Classical decoder. `n_in` is the bottleneck width; it is the only thing
    the feature-path model of NB15 changes, and it changes it explicitly."""
    a = CFG.arch
    if task == "binary":
        (d1,) = a.DEC_BINARY
        return nn.Sequential(
            nn.Linear(n_in, d1), nn.LayerNorm(d1), nn.GELU(),
            nn.Dropout(a.DROPOUT_DEC_BINARY), nn.Linear(d1, n_classes),
        )
    d1, d2 = a.DEC_MULTICLASS
    p1, p2 = a.DROPOUT_DEC_MULTI
    return nn.Sequential(
        nn.Linear(n_in, d1), nn.LayerNorm(d1), nn.GELU(),
        nn.Dropout(p1), nn.Linear(d1, d2), nn.GELU(),
        nn.Dropout(p2), nn.Linear(d2, n_classes),
    )


# ---------------------------------------------------------------------------
# The single hybrid model
# ---------------------------------------------------------------------------
class HybridIDS(nn.Module):
    """Encoder -> bottleneck -> decoder.

    The encoder and decoder are determined ONLY by `task`. No variant, no
    ablation, and no experiment may change them -- that invariant is enforced
    by `assert_scaffolding_identical`.
    """

    def __init__(self, task: str, n_features: int = 16, n_qubits: int = 8,
                 n_layers: int = 4, bottleneck: str = "pqc",
                 use_reuploading: bool = True,
                 entangle_scales: Sequence[int] = (1, 2, 4),
                 shots: Optional[int] = None,
                 n_classes: Optional[int] = None):
        super().__init__()
        self.task = task
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.bottleneck_kind = bottleneck
        n_classes = n_classes if n_classes is not None else n_classes_for(task)
        self.n_classes = n_classes

        # ---- encoder (task-determined, variant-invariant) -----------------
        self.classical_encoder = build_encoder(task, n_features, n_qubits)

        # ---- bottleneck (the ONLY thing a variant may change) -------------
        self.bottleneck = build_bottleneck(
            bottleneck, n_qubits, n_layers, use_reuploading, entangle_scales, shots
        )

        # ---- decoder (task-determined, variant-invariant) -----------------
        self.classical_decoder = build_decoder(task, n_qubits, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.classical_encoder(x)
        q = self.bottleneck(z)
        return self.classical_decoder(q)

    # -- introspection used by the result registry --------------------------
    def param_breakdown(self) -> Dict[str, int]:
        """Per-component trainable parameter counts, computed from the LIVE
        model. Table 8's parameter columns are generated from this, never
        transcribed."""
        def n(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        enc, bot, dec = (n(self.classical_encoder), n(self.bottleneck),
                         n(self.classical_decoder))
        return {"encoder": enc, "bottleneck": bot, "decoder": dec,
                "total": enc + bot + dec}

    def footprint_kb(self, bits: int = 32) -> float:
        """Deployable size. Reported alongside -- never instead of -- the
        operational cost measured by the cost model."""
        return self.param_breakdown()["total"] * (bits / 8) / 1024.0


# ---------------------------------------------------------------------------
# Variant construction for the matched controls and the ablation
# ---------------------------------------------------------------------------
def build_model(task: str, variant: str = "full", n_features: int = 16,
                n_qubits: int = 8, n_layers: int = 4,
                shots: Optional[int] = None,
                n_classes: Optional[int] = None) -> HybridIDS:
    """Build any ablation variant or matched-control bottleneck variant by name.

    `n_classes` overrides the head width only. NB15's stage-1 model needs a
    7-way head (DoS and DDoS merged) on otherwise identical scaffolding; every
    other caller leaves it at None and gets the task default, so no existing
    result changes.
    """
    if variant in CIRCUIT_VARIANTS:
        return HybridIDS(task, n_features, n_qubits, n_layers,
                         bottleneck="pqc", shots=shots, n_classes=n_classes,
                         **CIRCUIT_VARIANTS[variant])

    # Matched-control Family A: same scaffolding, non-quantum bottleneck.
    if variant in ("linear", "fourier", "mlp96", "identity"):
        return HybridIDS(task, n_features, n_qubits, n_layers,
                         bottleneck=variant, n_classes=n_classes)

    if variant == "classical_matched":
        return HybridIDS(task, n_features, n_qubits, n_layers,
                         bottleneck="mlp96", n_classes=n_classes)

    raise ValueError(f"unknown variant: {variant}")


class SimpleMLP(nn.Module):
    """Minimal 3-layer reference network (`simple_mlp`). Deliberately does
    NOT share the scaffolding -- it is a different architecture by design, and
    it must be described as such."""

    def __init__(self, task: str, n_features: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes_for(task)),
        )

    def forward(self, x):
        return self.net(x)

    def param_breakdown(self) -> Dict[str, int]:
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"encoder": 0, "bottleneck": 0, "decoder": 0, "total": t}

    def footprint_kb(self, bits: int = 32) -> float:
        return self.param_breakdown()["total"] * (bits / 8) / 1024.0


# ---------------------------------------------------------------------------
# Validation gate G4 -- called before any ablation training
# ---------------------------------------------------------------------------
def assert_scaffolding_identical(models: Dict[str, nn.Module],
                                 exempt: Sequence[str] = ("simple_mlp",)) -> None:
    """Raise if any two ablation variants differ in encoder or decoder size.

    It runs
    BEFORE training, so a scaffolding mismatch costs seconds, not a full
    ablation sweep.
    """
    ref_name, ref = None, None
    for name, m in models.items():
        if name in exempt:
            continue
        b = m.param_breakdown()
        if ref is None:
            ref_name, ref = name, b
            continue
        for part in ("encoder", "decoder"):
            if b[part] != ref[part]:
                raise AssertionError(
                    f"Scaffolding mismatch: "
                    f"variant '{name}' has {part}={b[part]} but '{ref_name}' has "
                    f"{part}={ref[part]}. Every ablation variant must share the "
                    f"identical classical encoder and decoder; only the "
                    f"bottleneck may differ."
                )
