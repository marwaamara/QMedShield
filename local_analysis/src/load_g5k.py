"""
load_g5k.py -- reuse the trained Grid'5000 models instead of retraining.

The headline binary and multiclass models from the Grid'5000 run were saved as
state_dicts. Their tensor shapes are IDENTICAL to the rebuilt `HybridIDS`
factory in `qmed/models.py`; only the module NAMES differ:

    Grid'5000                    rebuilt factory
    -----------------------      -----------------------------
    encoder.*                    classical_encoder.*
    quantum_layer.weights        bottleneck.q_weights
    decoder.*                    classical_decoder.*

So a key remap is all that stands between the archived weights and a working
model. This matters more than it looks: every INFERENCE-ONLY experiment
can then be produced on a laptop, with no cluster time:

    realistic + adversarial robustness
    finite-shot / noisy simulation
    inference latency + energy
    SHAP/LIME fidelity + stability
    PCA inverse-mapping validation
    per-class tables at any granularity

Only experiments that require NEW TRAINING (multi-seed, matched controls,
ablation, qubit/depth sweep, quantum baselines, split protocols, hierarchical
multiclass) actually need Grid'5000.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# Key rewrites, applied in order.
_KEY_MAP = (
    (r"^encoder\.",            "classical_encoder."),
    (r"^decoder\.",            "classical_decoder."),
    (r"^quantum_layer\.weights$", "bottleneck.q_weights"),
)


def remap_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rewrite Grid'5000 module names to the rebuilt factory's names."""
    out = {}
    for k, v in sd.items():
        nk = k
        for pat, rep in _KEY_MAP:
            nk = re.sub(pat, rep, nk)
        out[nk] = v
    return out


def load_hybrid(task: str, weights_path: str, model=None, strict: bool = True):
    """Load an archived Grid'5000 hybrid model into the rebuilt factory.

    Parameters
    ----------
    task : 'binary' | 'multiclass'
    weights_path : path to `hybrid_qcnn_{binary,multiclass}*.pt`
    model : optional pre-built HybridIDS; built from the factory if omitted.
    """
    if model is None:
        from qmed.models import build_model
        model = build_model(task, "full")

    raw = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]

    mapped = remap_state_dict(raw)
    missing, unexpected = model.load_state_dict(mapped, strict=False)

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"state_dict mismatch loading {weights_path}\n"
            f"  missing:    {list(missing)}\n"
            f"  unexpected: {list(unexpected)}"
        )
    model.eval()
    return model, {"missing": list(missing), "unexpected": list(unexpected)}


def load_preprocessors(artifact_dir: str) -> Dict:
    """Load the fitted scaler / PCA / label encoder from the archive.

    These are what make the archived model usable on new inputs, and what let
    the robustness perturbations be applied in the ORIGINAL 25-feature space: the archive
    stores X_scaled, and `scaler.inverse_transform` recovers raw feature units.
    """
    import joblib
    out = {}
    for key, fname in (("scaler", "scaler.joblib"),
                       ("pca_binary", "pca_binary.joblib"),
                       ("pca_multi", "pca_multi.joblib"),
                       ("label_encoder", "multiclass_encoder.joblib")):
        p = os.path.join(artifact_dir, fname)
        if os.path.exists(p):
            out[key] = joblib.load(p)
    return out


def load_prepared_data(artifact_dir: str) -> Dict[str, np.ndarray]:
    """Load the archived prepared_data.npz as a plain dict."""
    d = np.load(os.path.join(artifact_dir, "prepared_data.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def recover_raw_features(artifact_dir: str) -> Tuple[np.ndarray, list]:
    """Invert the StandardScaler to recover the ORIGINAL 25-feature values.

    Required by the robustness experiment: Gaussian noise in PCA space is not
    realistic, whereas packet loss, timing jitter, missing features and
    protocol variation are only meaningful in raw feature units. The archive makes this possible without the source CSVs.

    Returns (X_raw [50000, 25], feature_names).
    """
    data = load_prepared_data(artifact_dir)
    pre = load_preprocessors(artifact_dir)
    if "scaler" not in pre:
        raise FileNotFoundError("scaler.joblib absent; cannot invert scaling")
    X_raw = pre["scaler"].inverse_transform(data["X_scaled"])
    names = [str(s) for s in data["kept_names"]]
    return X_raw, names


# ---------------------------------------------------------------------------
# What the archive can and cannot answer
# ---------------------------------------------------------------------------
RECOVERABLE_WITHOUT_CLUSTER = {
    "realistic + adversarial robustness": "trained model + raw features",
    "finite-shot / noisy simulation":     "trained model, inference only",
    "inference latency + energy":         "trained model, inference only",
    "SHAP/LIME fidelity + stability":     "trained model + test data",
    "PCA inverse-map validation":         "trained model + PCA + raw names",
    "per-class tables (any granularity)":     "saved predictions + labels",
    "noise robustness, FULL detail":          "re-run inference over sigmas",
    "baseline per-sample predictions":        "refit from saved best_params",
}

NEEDS_CLUSTER = {
    "multi-seed headline":       "new training, 5 seeds",
    "parameter-matched controls": "new training",
    "ablation":             "new training (ablation weights were never saved)",
    "qubit/depth sweep":         "new training",
    "quantum baselines":         "new training",
    "session/device/chrono splits": "new training + source CSVs (not archived)",
    "hierarchical multiclass":   "new training",
}

NOT_RECOVERABLE_AT_ALL = {
    "flow provenance (source capture file, timestamps, device id)":
        "never saved; the split protocols REQUIRE re-downloading the CICIoMT2024 CSVs",
    "ablation model weights / curves / predictions":
        "never saved; these must be retrained",
    "CV per-fold models and predictions": "never saved",
    "SHAP / LIME values from the original run": "only PNGs were kept",
}


def print_recovery_map() -> None:
    for title, d in (("RECOVERABLE ON A LAPTOP", RECOVERABLE_WITHOUT_CLUSTER),
                     ("NEEDS GRID'5000", NEEDS_CLUSTER),
                     ("GONE FOR GOOD", NOT_RECOVERABLE_AT_ALL)):
        print(f"\n{title}\n{'-'*len(title)}")
        for k, v in d.items():
            print(f"  {k:52s} {v}")
