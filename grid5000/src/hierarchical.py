"""
hierarchical.py -- diagnosing and attacking the multiclass gap.

The binary detector achieves excellent performance, whereas multiclass
accuracy is relatively lower (79.19%).

The first thing to establish is WHERE the 20.8 points go. On the archived
predictions the answer is unusually clean:

    1,163 DoS  -> DDoS   74.5% of all error
      351 DDoS -> DoS    22.5% of all error
    ---------------------------------------
                         97.0% of all error is one confusion between two
                         volumetric flooding families

Everything else -- all six remaining classes, all other off-diagonal cells --
accounts for 47 of 7,500 test samples. Merge DoS and DDoS into one volumetric
superclass and the same, untouched model scores 99.37%.

That reframes the gap: the multiclass task is not broadly harder, one binary
decision inside it is. This module provides the three things needed to say that
with evidence, and to test whether the decision can be recovered:

    error_decomposition()   where the error actually is          (no training)
    separability_probe()    can ANY classifier separate DoS from DDoS on these
                            features? -- the representation ceiling (minutes)
    two-stage / feature-path models                              (training)

`separability_probe` is the control experiment. If tuned classical specialists,
given only the volumetric rows and the raw 25 features, also plateau, then the
ceiling belongs to the CICIoMT2024 WiFi/MQTT feature set -- which records no
source-address cardinality or dispersion, the very thing that distinguishes a
distributed flood from a single-source one -- and not to the quantum model. The
paper can then state a measured limit.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import (
    CLASS_NAMES,
    COARSE_NAMES,
    FEATURE_PATH_K,
    N_CLASSES,
    VOLUMETRIC_CLASSES,
)
from models import HybridIDS, build_decoder, build_encoder, build_model


# =============================================================================
# Label taxonomy
# =============================================================================
VOL_LO, VOL_HI = min(VOLUMETRIC_CLASSES), max(VOLUMETRIC_CLASSES)
COARSE_VOLUMETRIC = len(COARSE_NAMES) - 1        # index of Volumetric_Flood


def to_coarse(y: np.ndarray) -> np.ndarray:
    """8 classes -> 7, with DoS and DDoS merged into one volumetric class."""
    y = np.asarray(y)
    out = y.copy()
    out[np.isin(y, VOLUMETRIC_CLASSES)] = COARSE_VOLUMETRIC
    return out


def to_fine(y_coarse: np.ndarray, y_specialist: np.ndarray) -> np.ndarray:
    """Compose stage-1 (7-way) and stage-2 (DoS vs DDoS) into 8-class labels.

    `y_specialist` is 0 for the lower volumetric class (DoS) and 1 for the
    higher (DDoS); it is only read where stage 1 predicted volumetric.
    """
    out = np.asarray(y_coarse).copy()
    vol = out == COARSE_VOLUMETRIC
    out[vol] = np.where(np.asarray(y_specialist)[vol] == 1, VOL_HI, VOL_LO)
    return out


def volumetric_mask(y: np.ndarray) -> np.ndarray:
    return np.isin(np.asarray(y), VOLUMETRIC_CLASSES)


def specialist_labels(y: np.ndarray) -> np.ndarray:
    """DoS -> 0, DDoS -> 1 (defined only on volumetric rows)."""
    return (np.asarray(y) == VOL_HI).astype(np.int64)


# =============================================================================
# Where does the error go? -- no training required
# =============================================================================
def error_decomposition(y_true, y_pred,
                        class_names: Sequence[str] = tuple(CLASS_NAMES),
                        top: int = 10) -> Dict:
    """Rank every off-diagonal confusion by its share of total error.

    This is the table that turns "multiclass accuracy is lower" into a located,
    quantified statement, and it costs nothing: it reads the predictions the
    paper already reports.
    """
    from sklearn.metrics import confusion_matrix

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    n_err = int((y_true != y_pred).sum())

    pairs = [{"true": class_names[i], "pred": class_names[j],
              "true_idx": i, "pred_idx": j, "count": int(cm[i, j]),
              "share_of_error": float(cm[i, j] / n_err) if n_err else 0.0,
              "share_of_total": float(cm[i, j] / n)}
             for i in range(len(class_names)) for j in range(len(class_names))
             if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda r: -r["count"])

    vol_conf = int(cm[VOL_LO, VOL_HI] + cm[VOL_HI, VOL_LO])
    coarse_acc = float((to_coarse(y_true) == to_coarse(y_pred)).mean())

    return {
        "n_test": n, "n_errors": n_err,
        "accuracy": float((y_true == y_pred).mean()),
        "top_confusions": pairs[:top],
        "volumetric_confusion_count": vol_conf,
        "volumetric_share_of_error": float(vol_conf / n_err) if n_err else 0.0,
        "accuracy_merging_volumetric": coarse_acc,
        "accuracy_headroom_if_volumetric_perfect": coarse_acc - float(
            (y_true == y_pred).mean()),
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }


def ceiling_analysis(y_true, y_pred, specialist_accuracies: Sequence[float]
                     ) -> List[Dict]:
    """End-to-end 8-class accuracy implied by a given DoS-vs-DDoS accuracy.

    Ties the specialist's subproblem accuracy to the headline number, so a
    2-point gain on the subproblem is not quietly reported as a 2-point gain
    overall. Assumes stage 1 keeps its measured coarse accuracy.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    coarse_acc = float((to_coarse(y_true) == to_coarse(y_pred)).mean())
    vol_frac = float(volumetric_mask(y_true).mean())
    return [{"specialist_accuracy": float(a),
             "implied_overall_accuracy": coarse_acc - vol_frac * (1.0 - float(a)),
             "volumetric_fraction_of_test": vol_frac,
             "coarse_accuracy": coarse_acc}
            for a in specialist_accuracies]


# =============================================================================
# Representation ceiling -- is the information even present?
# =============================================================================
def separability_probe(X_tr, y_tr, X_te, y_te, space: str = "pca16",
                       models: Sequence[str] = ("Random Forest",
                                                "Gradient Boosting", "MLP"),
                       seed: int = 42) -> List[Dict]:
    """Fit classical specialists on the volumetric rows only.

    A specialist that sees nothing but DoS and DDoS, with the full training
    budget of that subproblem, is the strongest evidence available about what
    the features can support. Run it in both the PCA-16 and the raw-25 space:
    the difference is what PCA costs, the level is what the feature set allows.
    """
    from baselines import evaluate_sklearn, fit_baseline

    m_tr, m_te = volumetric_mask(y_tr), volumetric_mask(y_te)
    Xa, ya = np.asarray(X_tr)[m_tr], specialist_labels(y_tr)[m_tr]
    Xb, yb = np.asarray(X_te)[m_te], specialist_labels(y_te)[m_te]

    rows = []
    for name in models:
        est, meta = fit_baseline(name, Xa, ya, seed=seed)
        r = evaluate_sklearn(est, Xb, yb, 2)
        rows.append({"probe": name, "space": space, "n_train": int(m_tr.sum()),
                     "n_test": int(m_te.sum()), "accuracy": r["accuracy"],
                     "f1": r["f1"], "auc": r["auc"],
                     "fit_wall_s": meta.get("fit_time_s")})
    # Majority-class floor: the number every row above must beat to mean anything
    floor = float(max(np.bincount(yb, minlength=2)) / len(yb))
    rows.append({"probe": "majority-class floor", "space": space,
                 "n_train": int(m_tr.sum()), "n_test": int(m_te.sum()),
                 "accuracy": floor, "f1": None, "auc": None, "fit_wall_s": 0.0})
    return rows


def mutual_information_ranking(X, y, feature_names: Sequence[str],
                               seed: int = 42) -> List[Dict]:
    """Rank raw features by MI with the DoS/DDoS label, on TRAIN rows only.

    Feeds the feature-path variant: which raw features are worth routing around
    PCA. Selecting them on the training split is the whole point -- selecting on
    the test split would manufacture the improvement it claims to measure.
    """
    from sklearn.feature_selection import mutual_info_classif

    m = volumetric_mask(y)
    mi = mutual_info_classif(np.asarray(X)[m], specialist_labels(y)[m],
                             random_state=seed)
    order = np.argsort(mi)[::-1]
    return [{"rank": int(r + 1), "feature": str(feature_names[i]),
             "column": int(i), "mutual_information": float(mi[i])}
            for r, i in enumerate(order)]


# =============================================================================
# Approach 1 -- two-stage hierarchical detector
# =============================================================================
class TwoStageIDS(nn.Module):
    """Stage 1: 7-way coarse detector. Stage 2: DoS-vs-DDoS specialist.

    Both stages are `HybridIDS` from the same factory as the headline model --
    same encoder, same 8-qubit PQC, same decoder shape -- so any difference is
    attributable to the decomposition, not to a new architecture. Stage 2 is
    trained on volumetric rows only, which is what gives it a class balance the
    flat 8-way model never sees (32,195 rows split 76/24 instead of 5,012
    against 1,887 inside a 35,000-row 8-class problem).

    Cost is disclosed, not hidden: two models, two forward passes for the
    volumetric fraction of traffic, and roughly twice the parameters of the flat
    model. `param_breakdown` reports the sum.
    """

    def __init__(self, n_features: int = 16, n_qubits: int = 8,
                 n_layers: int = 4):
        super().__init__()
        self.stage1 = build_model("multiclass", "full", n_features, n_qubits,
                                  n_layers, n_classes=len(COARSE_NAMES))
        self.stage2 = build_model("binary", "full", n_features, n_qubits,
                                  n_layers)

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch: int = 4096) -> Dict[str, np.ndarray]:
        """Compose the two stages into 8-class predictions."""
        self.eval()
        xs = torch.tensor(np.asarray(X), dtype=torch.float32)
        coarse, spec = [], []
        for s in range(0, len(xs), batch):
            xb = xs[s:s + batch]
            coarse.append(self.stage1(xb).argmax(1).numpy())
            spec.append(self.stage2(xb).argmax(1).numpy())
        coarse = np.concatenate(coarse)
        spec = np.concatenate(spec)
        return {"y_pred": to_fine(coarse, spec), "coarse": coarse,
                "specialist": spec}

    def param_breakdown(self) -> Dict[str, int]:
        a, b = self.stage1.param_breakdown(), self.stage2.param_breakdown()
        return {"stage1_total": a["total"], "stage2_total": b["total"],
                "encoder": a["encoder"] + b["encoder"],
                "bottleneck": a["bottleneck"] + b["bottleneck"],
                "decoder": a["decoder"] + b["decoder"],
                "total": a["total"] + b["total"]}

    def footprint_kb(self, bits: int = 32) -> float:
        return self.param_breakdown()["total"] * (bits / 8) / 1024.0


# =============================================================================
# Approach 2 -- raw features routed around PCA
# =============================================================================
class FeaturePathIDS(nn.Module):
    """Flat 8-class model with k raw features bypassing PCA into the decoder.

    Hypothesis under test: PCA, fitted to maximise variance over all eight
    classes, discards the directions that separate DoS from DDoS. If so,
    delivering a handful of raw features straight to the decoder should recover
    part of the gap at a cost of k*(d1) extra weights.

    The encoder and the quantum bottleneck are untouched; only the decoder is
    wider by k inputs, and `param_breakdown` reports exactly how much wider.
    """

    def __init__(self, task: str = "multiclass", n_features: int = 16,
                 n_qubits: int = 8, n_layers: int = 4,
                 k_raw: int = FEATURE_PATH_K, raw_columns: Optional[Sequence[int]] = None,
                 n_classes: Optional[int] = None):
        super().__init__()
        self.task = task
        self.k_raw = k_raw
        self.raw_columns = list(raw_columns) if raw_columns is not None else []
        n_classes = n_classes or (2 if task == "binary" else N_CLASSES)
        self.classical_encoder = build_encoder(task, n_features, n_qubits)
        base = HybridIDS(task, n_features, n_qubits, n_layers, bottleneck="pqc")
        self.bottleneck = base.bottleneck
        self.classical_decoder = build_decoder(task, n_qubits + k_raw, n_classes)

    def forward(self, x_pca: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        q = self.bottleneck(self.classical_encoder(x_pca))
        return self.classical_decoder(torch.cat([q, x_raw], dim=1))

    def param_breakdown(self) -> Dict[str, int]:
        def n(m):
            return int(sum(p.numel() for p in m.parameters() if p.requires_grad))
        enc, bot, dec = (n(self.classical_encoder), n(self.bottleneck),
                         n(self.classical_decoder))
        flat = HybridIDS(self.task).param_breakdown()["total"]
        return {"encoder": enc, "bottleneck": bot, "decoder": dec,
                "total": enc + bot + dec, "k_raw": self.k_raw,
                "extra_vs_flat": enc + bot + dec - flat}

    def footprint_kb(self, bits: int = 32) -> float:
        return self.param_breakdown()["total"] * (bits / 8) / 1024.0


class PairedDataset(torch.utils.data.Dataset):
    """(PCA features, routed raw features, label) for FeaturePathIDS."""

    def __init__(self, X_pca, X_raw, y):
        self.a = torch.tensor(np.asarray(X_pca), dtype=torch.float32)
        self.b = torch.tensor(np.asarray(X_raw), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.a[i], self.b[i], self.y[i]


def train_feature_path(model: FeaturePathIDS, tr: Tuple, va: Tuple,
                       task: str = "multiclass",
                       class_weights: Optional[torch.Tensor] = None,
                       batch_size: int = None, verbose: bool = True) -> Dict:
    """Train FeaturePathIDS. Validation-only checkpoint selection, epochs and
    learning rates taken from config exactly as `train.train_model` does -- the
    two-input signature is the only reason this cannot call it directly."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    from config import CFG, epochs_for, lr_for
    from sklearn.metrics import f1_score

    epochs = epochs_for(task)
    lr, lr_min = lr_for(task)
    bs = batch_size or CFG.train.BATCH_SIZE

    dl_tr = DataLoader(PairedDataset(*tr), batch_size=bs, shuffle=True)
    dl_va = DataLoader(PairedDataset(*va), batch_size=bs, shuffle=False)

    crit = nn.CrossEntropyLoss(weight=class_weights)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=CFG.train.WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr_min)

    hist = {"train_loss": [], "train_acc": [], "val_f1": []}
    best_f1, best_state, since = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        tot = corr = 0
        loss_sum = 0.0
        for xa, xb, yb in dl_tr:
            opt.zero_grad()
            out = model(xa, xb)
            loss = crit(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG.train.GRAD_CLIP)
            opt.step()
            loss_sum += float(loss) * len(yb)
            corr += int((out.argmax(1) == yb).sum())
            tot += len(yb)
        sched.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xa, xb, yb in dl_va:
                preds.append(model(xa, xb).argmax(1).numpy())
                labels.append(yb.numpy())
        vf1 = float(f1_score(np.concatenate(labels), np.concatenate(preds),
                             average="weighted", zero_division=0))

        hist["train_loss"].append(loss_sum / tot)
        hist["train_acc"].append(corr / tot)
        hist["val_f1"].append(vf1)

        if vf1 > best_f1:
            best_f1, since = vf1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"  epoch {epoch:3d}/{epochs} train_acc={corr/tot:.4f} "
                  f"val_f1={vf1:.4f}", flush=True)
        if epoch >= CFG.train.EARLY_STOP_MIN_EPOCHS \
                and since >= CFG.train.EARLY_STOP_PATIENCE:
            if verbose:
                print(f"  early stop at epoch {epoch} (best val_f1={best_f1:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    hist["epochs_run"] = epoch
    hist["best_val_f1"] = best_f1
    return hist


@torch.no_grad()
def evaluate_feature_path(model: FeaturePathIDS, X_pca, X_raw, y,
                          n_classes: int = N_CLASSES,
                          batch: int = 4096) -> Dict:
    """Same metric dictionary shape as `train.evaluate`, so the persistence
    layer and every downstream aggregation treat this run like any other."""
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)

    model.eval()
    xa = torch.tensor(np.asarray(X_pca), dtype=torch.float32)
    xb = torch.tensor(np.asarray(X_raw), dtype=torch.float32)
    probs = []
    for s in range(0, len(xa), batch):
        probs.append(torch.softmax(model(xa[s:s + batch], xb[s:s + batch]),
                                   dim=1).numpy())
    probs = np.concatenate(probs)
    pred = probs.argmax(1)
    y = np.asarray(y)

    avg = "binary" if n_classes == 2 else "weighted"
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average=avg,
                                                  zero_division=0)
    try:
        auc = (roc_auc_score(y, probs[:, 1]) if n_classes == 2
               else roc_auc_score(y, probs, multi_class="ovr", average="weighted"))
    except Exception:
        auc = float("nan")
    pc_p, pc_r, pc_f1, pc_sup = precision_recall_fscore_support(
        y, pred, average=None, labels=list(range(n_classes)), zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)), "precision": float(p),
            "recall": float(r), "f1": float(f1), "auc": float(auc),
            "per_class": {"precision": pc_p.tolist(), "recall": pc_r.tolist(),
                          "f1": pc_f1.tolist(), "support": pc_sup.tolist()},
            "confusion_matrix": confusion_matrix(
                y, pred, labels=list(range(n_classes))).tolist(),
            "_predictions": pred, "_labels": y, "_probabilities": probs}


# =============================================================================
# Locating pool rows -- shared with NB07
# =============================================================================
def locate_pool_rows(X_pool_pca: np.ndarray, X_split_pca: np.ndarray,
                     tol: float = 1e-9) -> np.ndarray:
    """Index of each split row inside the 50k pool, by exact PCA match.

    The archive stores the splits in PCA space and the pool in both spaces, so
    this is how a raw-space feature is attached to a training row without the
    source CSVs. NB07 does the same thing for its raw-space perturbations; the
    assertion below is what makes it safe.
    """
    from scipy.spatial import cKDTree

    d, idx = cKDTree(np.asarray(X_pool_pca)).query(np.asarray(X_split_pca), k=1)
    if not (d < tol).all():
        raise AssertionError(
            f"{int((d >= tol).sum())} split rows could not be located in the "
            f"pool (max distance {d.max():.3e}); the archive and the split do "
            f"not come from the same preprocessing run.")
    return idx
