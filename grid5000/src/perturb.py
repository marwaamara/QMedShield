"""
perturb.py -- realistic perturbations and adversarial attacks.

Gaussian perturbations in PCA space do not necessarily correspond to realistic
variations encountered in IoMT network traffic. This module therefore
evaluates robustness under more representative network perturbations (packet
loss, timing jitter, missing features, protocol variation) and under
adversarial attacks.

What matters is WHERE the perturbation is applied. Everything in this module
operates on the ORIGINAL 25-feature space, before the scaler and before PCA;
the pipeline is then re-applied end to end. The PCA-space Gaussian sweep of the
paper is retained as `pca_gaussian` and should be labelled as a sanity
baseline, not as the robustness result.

Feature names come from the archived `kept_names`:
    Header_Length, Duration, Rate, fin_flag_number, syn_flag_number,
    rst_flag_number, ...
so the operators below select columns by semantic role rather than by index.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# =============================================================================
# Semantic column groups
# =============================================================================
def column_groups(feature_names: Sequence[str]) -> Dict[str, List[int]]:
    """Map each perturbation to the feature columns it may legitimately touch.

    Getting this mapping right is what separates a realistic perturbation from
    noise: packet loss must deflate counts and rates, not flag bits.
    """
    lower = [n.lower() for n in feature_names]

    def match(*keys):
        return [i for i, n in enumerate(lower) if any(k in n for k in keys)]

    return {
        # counters and rates shrink when packets are dropped
        "volume": match("rate", "number", "count", "tot", "sum", "length", "size"),
        # timing-derived features distort under jitter
        "timing": match("duration", "rate", "iat", "time", "interval"),
        # header / payload shape shifts with protocol variation
        "protocol": match("header", "length", "size", "protocol", "proto"),
        # TCP/IP flags an attacker can set freely
        "flags": match("flag", "fin", "syn", "rst", "psh", "ack", "urg", "ece", "cwr"),
        # features an attacker can plausibly control end-to-end
        "attacker_controllable": sorted(set(
            match("rate", "length", "size", "duration", "iat", "interval",
                  "flag", "fin", "syn", "rst", "psh", "ack"))),
    }


# =============================================================================
# Realistic perturbations, in RAW feature units
# =============================================================================
def apply_perturbation(X_raw: np.ndarray, kind: str, severity: float,
                       feature_names: Sequence[str],
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Return a perturbed copy of the raw 25-feature matrix.

    severity is a fraction in [0, 1] whose meaning is per-perturbation and is
    documented below, so a report can state exactly what was simulated.
    """
    rng = rng or np.random.default_rng(42)
    if severity == 0:
        return X_raw.copy()

    X = X_raw.copy()
    g = column_groups(feature_names)

    if kind == "missing":
        # MCAR sensor/feature dropout at rate `severity`, mean-imputed --
        # the standard failure mode when an IoMT gateway loses a telemetry field.
        mask = rng.random(X.shape) < severity
        col_mean = X.mean(axis=0)
        rows, cols = np.where(mask)
        X[rows, cols] = col_mean[cols]

    elif kind == "jitter":
        # Timing jitter: multiplicative Gaussian noise of relative scale
        # `severity` on duration / rate / inter-arrival features.
        c = g["timing"]
        if c:
            X[:, c] *= (1.0 + rng.normal(0.0, severity, size=(len(X), len(c))))

    elif kind == "packet_loss":
        # A fraction `severity` of packets is lost, so counters and rates
        # deflate proportionally; the loss is itself slightly variable per flow.
        c = g["volume"]
        if c:
            keep = 1.0 - severity
            per_flow = np.clip(
                rng.normal(keep, severity * 0.25, size=(len(X), 1)), 0.0, 1.0)
            X[:, c] *= per_flow

    elif kind == "protocol_var":
        # Protocol variation: header and payload size distributions shift
        # (different client stack, MQTT version, or MTU).
        c = g["protocol"]
        if c:
            X[:, c] *= (1.0 + rng.normal(0.0, severity, size=(len(X), len(c))))

    elif kind == "pca_gaussian":
        raise ValueError("pca_gaussian operates in PCA space -- "
                         "use apply_pca_gaussian(), and label it a sanity baseline")
    else:
        raise ValueError(f"unknown perturbation: {kind}")

    return X


def apply_pca_gaussian(X_pca: np.ndarray, sigma: float,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """The perturbation used in Table 11 of the paper.

    Retained ONLY as a labelled sanity baseline: additive Gaussian noise in PCA
    space does not correspond to any physical process in IoMT traffic.
    """
    rng = rng or np.random.default_rng(42)
    if sigma == 0:
        return X_pca.copy()
    return X_pca + rng.normal(0.0, sigma, size=X_pca.shape)


# =============================================================================
# Adversarial attacks
# =============================================================================
def fgsm(model, X_pca: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    """Single-step white-box attack on the differentiable hybrid model."""
    return pgd(model, X_pca, y, eps, steps=1)


def pgd(model, X_pca: np.ndarray, y: np.ndarray, eps: float,
        steps: int = 10, alpha: Optional[float] = None) -> np.ndarray:
    """L-infinity PGD in PCA space, where the model is differentiable.

    Operating in PCA space is a stated limitation: it measures the model's
    local sensitivity, not an attack a network adversary could physically
    mount. `constrained_attack` is the operationally meaningful variant.
    """
    import torch

    alpha = alpha or (eps / max(1, steps)) * 2.5
    lossf = torch.nn.CrossEntropyLoss()
    x0 = torch.tensor(X_pca, dtype=torch.float32)
    x = x0.clone()
    yt = torch.tensor(y, dtype=torch.long)

    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        lossf(model(x), yt).backward()
        x = x + alpha * x.grad.sign()
        x = x0 + (x - x0).clamp(-eps, eps)

    return x.detach().numpy()


def constrained_attack(model, X_raw: np.ndarray, y: np.ndarray,
                       feature_names: Sequence[str], scaler, pca,
                       eps_rel: float, steps: int = 10) -> np.ndarray:
    """Adversarial perturbation restricted to attacker-controllable features.

    This is the only adversarial variant that is operationally meaningful for a
    network IDS: an attacker can change packet sizes, rates, timings and TCP
    flags, but cannot rewrite features the defender derives independently.
    Perturbations are bounded RELATIVE to each feature's scale, and applied in
    raw units before the scaler, so the result stays a realisable flow.

    Uses a gradient-free coordinate search on the restricted feature set,
    because the raw-space path (scaler + PCA) is not differentiated end to end.
    """
    import torch

    g = column_groups(feature_names)
    cols = g["attacker_controllable"]
    if not cols:
        return X_raw.copy()

    X = X_raw.copy()
    col_scale = np.abs(X[:, cols]).mean(axis=0) + 1e-12

    def loss_of(Xr):
        Xp = pca.transform(scaler.transform(Xr))
        with torch.no_grad():
            logits = model(torch.tensor(Xp, dtype=torch.float32))
            return torch.nn.functional.cross_entropy(
                logits, torch.tensor(y, dtype=torch.long), reduction="none").numpy()

    best = loss_of(X)
    for _ in range(steps):
        for j, c in enumerate(cols):
            for sign in (+1.0, -1.0):
                trial = X.copy()
                trial[:, c] += sign * (eps_rel / steps) * col_scale[j]
                l = loss_of(trial)
                better = l > best
                if better.any():
                    X[better] = trial[better]
                    best[better] = l[better]
    return X


def transfer_attack(surrogate, target_model, X_pca: np.ndarray, y: np.ndarray,
                    eps: float, steps: int = 10) -> np.ndarray:
    """Black-box transfer: craft on a surrogate, evaluate on the target.

    The realistic threat model for a deployed IDS, where the attacker cannot
    query gradients of the fielded detector.
    """
    return pgd(surrogate, X_pca, y, eps, steps=steps)
