"""
xai.py -- explainability, with quantitative validation.

The benefit of combining SHAP and LIME needs to be demonstrated against using a
single explainability method, and the transformation of SHAP and LIME
attributions from PCA space back to the original feature space is a heuristic
that should be presented as such, with experiments on explanation stability
and fidelity.

So this module does three things:

  1. FIDELITY -- deletion and insertion curves, comprehensiveness and
     sufficiency, computed identically for SHAP-only, LIME-only and a fused
     ranking. That is what turns "we used both" into a testable claim.
  2. STABILITY -- repeat-run variance (LIME is stochastic, so this is where a
     combination could genuinely pay off) and a local Lipschitz estimate.
  3. INVERSE-MAP VALIDATION -- the PCA back-projection is treated as a
     heuristic and checked against two references computed directly in the
     original feature space: permutation importance on the full pipeline, and
     KernelSHAP over the whole scaler+PCA+model chain as one black box.

If the fused ranking turns out to be no better than the better single method,
that is a publishable answer and should be reported as such -- under the
a negative result here costs the paper nothing.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Prediction wrappers
# =============================================================================
def torch_predict_proba(model) -> Callable[[np.ndarray], np.ndarray]:
    """PCA-space probability function for SHAP/LIME."""
    import torch

    def f(X):
        with torch.no_grad():
            logits = model(torch.tensor(np.asarray(X), dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()
    return f


def pipeline_predict_proba(model, scaler, pca) -> Callable[[np.ndarray], np.ndarray]:
    """RAW-feature-space probability function.

    Treating scaler + PCA + model as ONE black box is what lets SHAP attribute
    directly to the original 25 features, with no inverse mapping and therefore
    no heuristic -- the reference against which the heuristic is validated.
    """
    import torch

    def f(X_raw):
        Xp = pca.transform(scaler.transform(np.asarray(X_raw)))
        with torch.no_grad():
            logits = model(torch.tensor(Xp, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()
    return f


# =============================================================================
# Attributions
# =============================================================================
def kernel_shap(predict_fn, X_explain: np.ndarray, X_background: np.ndarray,
                nsamples: int = 100, seed: int = 42) -> np.ndarray:
    """KernelSHAP. Returns (n_samples, n_features) for the predicted class."""
    import shap

    bg = shap.kmeans(X_background, min(50, len(X_background)))
    expl = shap.KernelExplainer(predict_fn, bg)
    vals = expl.shap_values(X_explain, nsamples=nsamples, silent=True)

    if isinstance(vals, list):                    # one array per class
        pred = predict_fn(X_explain).argmax(1)
        return np.stack([vals[c][i] for i, c in enumerate(pred)])
    if np.asarray(vals).ndim == 3:                # (n, f, classes)
        vals = np.asarray(vals)
        pred = predict_fn(X_explain).argmax(1)
        return np.stack([vals[i, :, c] for i, c in enumerate(pred)])
    return np.asarray(vals)


def lime_attributions(predict_fn, X_explain: np.ndarray, X_train: np.ndarray,
                      feature_names: Sequence[str], num_samples: int = 1000,
                      seed: int = 42) -> np.ndarray:
    """LIME local attributions, aligned to the same (n_samples, n_features)."""
    from lime.lime_tabular import LimeTabularExplainer

    expl = LimeTabularExplainer(X_train, feature_names=list(feature_names),
                                discretize_continuous=False,
                                random_state=seed, mode="classification")
    n_f = X_explain.shape[1]
    out = np.zeros((len(X_explain), n_f))
    for i, x in enumerate(X_explain):
        label = int(predict_fn(x.reshape(1, -1)).argmax())
        e = expl.explain_instance(x, predict_fn, num_features=n_f,
                                  labels=(label,), num_samples=num_samples)
        for j, w in e.as_map()[label]:
            out[i, j] = w
    return out


def global_importance(attr: np.ndarray) -> np.ndarray:
    """Mean |attribution| per feature -- the global ranking."""
    return np.abs(np.asarray(attr)).mean(axis=0)


def fuse_rankings(*importances: np.ndarray, method: str = "rank_mean") -> np.ndarray:
    """Combine SHAP and LIME into one ranking.

    `rank_mean` averages the per-method ranks, which is scale-free -- SHAP and
    LIME attributions are not on a common scale, so averaging raw magnitudes
    would silently let whichever method has larger values dominate.
    """
    if method == "rank_mean":
        ranks = [np.argsort(np.argsort(-np.abs(v))) for v in importances]
        return -np.mean(ranks, axis=0)
    if method == "zscore_mean":
        zs = [(np.abs(v) - np.abs(v).mean()) / (np.abs(v).std() + 1e-12)
              for v in importances]
        return np.mean(zs, axis=0)
    raise ValueError(method)


# =============================================================================
# 1. FIDELITY
# =============================================================================
def deletion_curve(predict_fn, X: np.ndarray, importance: np.ndarray,
                   baseline: Optional[np.ndarray] = None,
                   steps: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Remove features most-important-first; a faithful ranking drops fastest.

    Returns the full curve, not just its AUC, so it can be plotted.
    """
    X = np.asarray(X, float)
    base = baseline if baseline is not None else X.mean(0)
    order = np.argsort(-np.abs(importance))
    steps = steps or len(order)

    p0 = predict_fn(X)
    cls = p0.argmax(1)
    conf = [p0[np.arange(len(X)), cls].mean()]

    Xw = X.copy()
    for k in range(steps):
        Xw[:, order[k]] = base[order[k]]
        p = predict_fn(Xw)
        conf.append(p[np.arange(len(X)), cls].mean())

    conf = np.asarray(conf)
    return {"n_removed": np.arange(len(conf)), "confidence": conf,
            "auc": np.array([float(np.trapezoid(conf) / (len(conf) - 1))])}


def insertion_curve(predict_fn, X: np.ndarray, importance: np.ndarray,
                    baseline: Optional[np.ndarray] = None,
                    steps: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Add features most-important-first, starting from the baseline."""
    X = np.asarray(X, float)
    base = baseline if baseline is not None else X.mean(0)
    order = np.argsort(-np.abs(importance))
    steps = steps or len(order)

    cls = predict_fn(X).argmax(1)
    Xw = np.tile(base, (len(X), 1))
    conf = [predict_fn(Xw)[np.arange(len(X)), cls].mean()]
    for k in range(steps):
        Xw[:, order[k]] = X[:, order[k]]
        conf.append(predict_fn(Xw)[np.arange(len(X)), cls].mean())

    conf = np.asarray(conf)
    return {"n_inserted": np.arange(len(conf)), "confidence": conf,
            "auc": np.array([float(np.trapezoid(conf) / (len(conf) - 1))])}


def comprehensiveness_sufficiency(predict_fn, X: np.ndarray,
                                  importance: np.ndarray,
                                  k: int = 5) -> Dict[str, float]:
    """Comprehensiveness: confidence drop when the top-k are removed.
    Sufficiency: confidence retained when ONLY the top-k are kept.
    Higher comprehensiveness and lower sufficiency-gap mean a more faithful
    explanation."""
    X = np.asarray(X, float)
    base = X.mean(0)
    order = np.argsort(-np.abs(importance))[:k]

    p0 = predict_fn(X)
    cls = p0.argmax(1)
    c0 = p0[np.arange(len(X)), cls].mean()

    Xr = X.copy()
    Xr[:, order] = base[order]
    c_removed = predict_fn(Xr)[np.arange(len(X)), cls].mean()

    Xk = np.tile(base, (len(X), 1))
    Xk[:, order] = X[:, order]
    c_kept = predict_fn(Xk)[np.arange(len(X)), cls].mean()

    return {"base_confidence": float(c0),
            "comprehensiveness": float(c0 - c_removed),
            "sufficiency": float(c0 - c_kept), "k": k}


# =============================================================================
# 2. STABILITY
# =============================================================================
def stability_repeat(attr_fn: Callable[[int], np.ndarray],
                     repeats: int = 10) -> Dict[str, float]:
    """Variance of the explanation across repeated runs at fixed input.

    LIME samples its neighbourhood stochastically, so this is non-zero for LIME
    and ~zero for KernelSHAP at fixed background -- the most plausible concrete
    benefit of reporting both.
    """
    from scipy.stats import spearmanr

    runs = [global_importance(attr_fn(s)) for s in range(repeats)]
    R = np.stack(runs)

    rhos = [spearmanr(R[i], R[j]).statistic
            for i in range(len(R)) for j in range(i + 1, len(R))]
    top5 = [set(np.argsort(-r)[:5]) for r in R]
    jac = [len(top5[i] & top5[j]) / len(top5[i] | top5[j])
           for i in range(len(top5)) for j in range(i + 1, len(top5))]

    return {"mean_pairwise_spearman": float(np.mean(rhos)),
            "std_pairwise_spearman": float(np.std(rhos)),
            "mean_top5_jaccard": float(np.mean(jac)),
            "mean_cv": float(np.mean(R.std(0) / (np.abs(R).mean(0) + 1e-12))),
            "repeats": repeats}


def local_lipschitz(attr_fn: Callable[[np.ndarray], np.ndarray],
                    X: np.ndarray, eps: float = 0.01,
                    n_probe: int = 5, seed: int = 42) -> Dict[str, float]:
    """Sensitivity of the explanation to a small input perturbation.

    An explanation that swings under a perturbation the MODEL is insensitive to
    is not describing the model. This is the explanation-side analogue of the
    robustness analysis in NB07.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    a0 = attr_fn(X)

    ratios = []
    for _ in range(n_probe):
        d = rng.normal(0, eps, X.shape)
        a1 = attr_fn(X + d)
        num = np.linalg.norm(a1 - a0, axis=1)
        den = np.linalg.norm(d, axis=1) + 1e-12
        ratios.append(num / den)

    r = np.concatenate(ratios)
    return {"mean_lipschitz": float(r.mean()), "max_lipschitz": float(r.max()),
            "p95_lipschitz": float(np.percentile(r, 95)), "eps": eps}


# =============================================================================
# 3. AGREEMENT and INVERSE-MAP VALIDATION
# =============================================================================
def agreement(imp_a: np.ndarray, imp_b: np.ndarray, k: int = 5) -> Dict[str, float]:
    from scipy.stats import spearmanr, kendalltau

    a, b = np.abs(imp_a), np.abs(imp_b)
    ta, tb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return {"spearman": float(spearmanr(a, b).statistic),
            "kendall_tau": float(kendalltau(a, b).statistic),
            f"jaccard_top{k}": len(ta & tb) / len(ta | tb),
            f"overlap_top{k}": len(ta & tb)}


def pca_inverse_map(importance_pca: np.ndarray, pca) -> np.ndarray:
    """The paper's heuristic: project PCA-space importance back through
    the loading matrix onto the original features.

    It is a heuristic because |importance| is not a signed quantity that the
    linear map is entitled to carry, and because component importances are not
    independent. `validate_inverse_map` measures how much that costs.
    """
    W = np.abs(np.asarray(pca.components_))          # (n_comp, n_features)
    return np.abs(np.asarray(importance_pca)) @ W


def permutation_importance_pipeline(predict_fn, X_raw: np.ndarray, y: np.ndarray,
                                    n_repeats: int = 5, seed: int = 42
                                    ) -> np.ndarray:
    """Model-agnostic importance in the ORIGINAL feature space.

    Requires no PCA algebra at all, so it is a clean reference for validating
    the inverse map.
    """
    rng = np.random.default_rng(seed)
    X_raw = np.asarray(X_raw, float)
    base = (predict_fn(X_raw).argmax(1) == y).mean()

    imp = np.zeros(X_raw.shape[1])
    for j in range(X_raw.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X_raw.copy()
            rng.shuffle(Xp[:, j])
            drops.append(base - (predict_fn(Xp).argmax(1) == y).mean())
        imp[j] = float(np.mean(drops))
    return imp


def validate_inverse_map(mapped: np.ndarray, reference: np.ndarray,
                         feature_names: Sequence[str], k: int = 5) -> Dict:
    """Compare the heuristic inverse map against a directly-computed reference.

    High agreement validates the heuristic; low agreement means the direct
    rankings should be reported instead and the inverse map dropped. Either
    outcome is informative -- what is not acceptable is leaving it unchecked.
    """
    a = agreement(mapped, reference, k=k)
    top_m = [feature_names[i] for i in np.argsort(-np.abs(mapped))[:k]]
    top_r = [feature_names[i] for i in np.argsort(-np.abs(reference))[:k]]
    verdict = ("validated" if a["spearman"] >= 0.7 else
               "weak -- prefer the direct ranking" if a["spearman"] >= 0.4 else
               "not supported -- report direct rankings and drop the inverse map")
    return {**a, "top_mapped": top_m, "top_reference": top_r, "verdict": verdict}
