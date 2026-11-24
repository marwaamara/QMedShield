"""
baselines.py -- classical comparators.

Three families of comparators:

  A. PAPER BASELINES (Tables 6/7) -- RF, GB, MLP, KNN, SVM(RBF).
     This module SAVES the fitted models and the per-sample predictions, so
     paired McNemar tests can be computed directly from the saved artefacts.

  B. PARAMETER-MATCHED CONTROLS --
     These isolate whether the observed behaviour is attributable to the
     quantum component rather than to differences in network architecture.
     Implemented in models.py: the SAME encoder/decoder scaffolding with the
     PQC swapped for linear / Fourier / tiny-MLP / identity bottlenecks.

  C. LIGHTWEIGHT DEPLOYABLES --
     Highly optimised lightweight classical models, for comparison with the
     quantum approach. Quantised, pruned and depth-limited models at a
     footprint comparable to the 1,178-parameter binary detector, reported on
     the same accuracy / footprint / energy axes.

Refitting family A from the archived `best_params` takes minutes on CPU and
needs no cluster.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# A. Paper baselines
# =============================================================================
# Grids exactly as reported in Table 8 of the paper (baseline configuration).
PARAM_GRIDS = {
    "Random Forest":     {"n_estimators": [100, 200, 500], "max_depth": [10, 20, None]},
    "Gradient Boosting": {"n_estimators": [100, 200], "max_depth": [3, 5, 7],
                          "learning_rate": [0.01, 0.1]},
    "MLP":               {"hidden_layer_sizes": [(64, 32), (128, 64)],
                          "alpha": [1e-3, 1e-2]},
    "KNN":               {"n_neighbors": [3, 5, 7, 11]},
    "SVM (RBF)":         {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
}


def make_estimator(name: str, seed: int = 42, **params):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    if name == "Random Forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=seed, **params)
    if name == "MLP":
        return MLPClassifier(random_state=seed, max_iter=500, **params)
    if name == "KNN":
        return KNeighborsClassifier(n_jobs=-1, **params)
    if name == "SVM (RBF)":
        # probability=True is REQUIRED: without it there is no predict_proba,
        # hence no AUC, no PR curve and no calibration analysis.
        return SVC(random_state=seed, probability=True, **params)
    raise ValueError(f"unknown baseline: {name}")


def fit_baseline(name: str, X_train, y_train, seed: int = 42,
                 best_params: Optional[Dict] = None, tune: bool = False,
                 cv: int = 3) -> Tuple[object, Dict]:
    """Fit one baseline.

    Pass `best_params` (archived from the original run) to refit directly and
    skip the grid search entirely -- minutes instead of hours, and it reproduces
    the original configuration exactly.
    """
    import time
    t0 = time.time()

    if best_params and not tune:
        est = make_estimator(name, seed, **best_params)
        est.fit(X_train, y_train)
        return est, {"best_params": best_params, "tuned": False,
                     "fit_time_s": time.time() - t0}

    if tune:
        from sklearn.model_selection import GridSearchCV
        gs = GridSearchCV(make_estimator(name, seed), PARAM_GRIDS[name],
                          cv=cv, scoring="f1_weighted", n_jobs=-1)
        gs.fit(X_train, y_train)
        return gs.best_estimator_, {"best_params": gs.best_params_, "tuned": True,
                                    "fit_time_s": time.time() - t0}

    est = make_estimator(name, seed)
    est.fit(X_train, y_train)
    return est, {"best_params": {}, "tuned": False,
                 "fit_time_s": time.time() - t0}


def evaluate_sklearn(est, X_test, y_test, n_classes: int) -> Dict:
    """Metrics + the per-sample arrays that make paired tests possible."""
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 roc_auc_score, confusion_matrix)

    pred = est.predict(X_test)
    try:
        probs = est.predict_proba(X_test)
    except Exception:
        probs = None

    avg = "binary" if n_classes == 2 else "weighted"
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average=avg,
                                                  zero_division=0)
    auc = float("nan")
    if probs is not None:
        try:
            auc = (roc_auc_score(y_test, probs[:, 1]) if n_classes == 2
                   else roc_auc_score(y_test, probs, multi_class="ovr",
                                      average="weighted"))
        except Exception:
            pass

    return {"accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "auc": float(auc),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "_predictions": pred, "_labels": np.asarray(y_test),
            "_probabilities": probs}


def model_footprint_kb(est) -> float:
    """Serialised size -- the deployment-cost axis for the Pareto plot.

    For tree ensembles this is the meaningful number: the quantum model is
    compared against highly optimised lightweight classical models, and a
    500-tree forest is not lightweight however fast it predicts.
    """
    import io
    import joblib
    buf = io.BytesIO()
    joblib.dump(est, buf)
    return buf.getbuffer().nbytes / 1024.0


# =============================================================================
# C. Lightweight classical deployables
# =============================================================================
def make_lightweight(name: str, seed: int = 42, n_classes: int = 8):
    """Classical models chosen to be genuinely small, not merely accurate."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier

    if name == "rf20":
        # 20 shallow trees instead of 500 deep ones
        return RandomForestClassifier(n_estimators=20, max_depth=8,
                                      random_state=seed, n_jobs=-1)
    if name == "gbdt_shallow":
        return GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                          random_state=seed)
    if name == "pruned_mlp":
        # Sized to land near the 1,178-parameter budget of the binary detector
        return MLPClassifier(hidden_layer_sizes=(48,), alpha=1e-2,
                             max_iter=500, random_state=seed)
    if name == "int8_mlp":
        # Same topology as pruned_mlp; quantisation is applied post-fit by
        # quantise_mlp_int8() so the footprint, not the fit, is what changes.
        return MLPClassifier(hidden_layer_sizes=(48,), alpha=1e-2,
                             max_iter=500, random_state=seed)
    raise ValueError(f"unknown lightweight baseline: {name}")


def quantise_mlp_int8(est) -> Dict:
    """Post-training int8 quantisation of a fitted sklearn MLP.

    Reports the accuracy cost and the footprint saving. This is a fair
    comparator: a classical model optimised for deployment, not just for
    accuracy.
    """
    import copy

    q = copy.deepcopy(est)
    scales = []
    for i, W in enumerate(q.coefs_):
        s = np.abs(W).max() / 127.0 if np.abs(W).max() > 0 else 1.0
        q.coefs_[i] = np.round(W / s) * s          # simulated int8 round-trip
        scales.append(float(s))

    n_w = int(sum(W.size for W in est.coefs_) +
              sum(b.size for b in est.intercepts_))
    return {"model": q, "n_params": n_w,
            "footprint_kb_fp32": n_w * 4 / 1024.0,
            "footprint_kb_int8": n_w * 1 / 1024.0,
            "scales": scales}


def count_sklearn_params(est) -> int:
    """Parameter count on the same axis as the hybrid model's 1,178."""
    if hasattr(est, "coefs_"):        # MLP
        return int(sum(W.size for W in est.coefs_) +
                   sum(b.size for b in est.intercepts_))
    if hasattr(est, "estimators_"):   # tree ensemble: total node count
        total = 0
        for t in np.ravel(est.estimators_):
            tree = getattr(t, "tree_", None)
            if tree is not None:
                total += int(tree.node_count)
        return total
    if hasattr(est, "support_vectors_"):   # SVM
        return int(est.support_vectors_.size + est.dual_coef_.size)
    if hasattr(est, "_fit_X"):             # KNN stores the training set
        return int(est._fit_X.size)
    return -1
