"""
plots.py -- every curve the paper needs, quantitative and qualitative.

Design rule, and the reason this module exists:

    A FIGURE IS NOT DATA. Every plotting function here ALSO returns the arrays
    it drew, and every caller passes them to `save_curve()`, so changing an
    axis, adding a confidence band, or re-plotting at a different DPI never
    requires recomputing the experiment.

Coverage:

  Quantitative (model performance)
    training_curves          loss / accuracy / F1 / AUC per epoch, +/- seed band
    roc_curves               one-vs-rest ROC, per class and micro/macro
    pr_curves                precision-recall (more informative under 66.8% DDoS)
    calibration_curve        reliability diagram + Brier score
    confusion_matrix         counts and row-normalised
    per_class_bars           precision/recall/F1 per class with support
    metric_vs_sweep          any metric against any swept parameter, +/- CI
    seed_boxplot             distribution across seeds
    pareto_scatter           accuracy vs footprint vs energy

  Qualitative (data and explanation)
    class_distribution       class balance, train vs test
    pca_variance             explained variance and cumulative curve
    tsne_projection          2-D embedding coloured by class
    correlation_heatmap      feature correlation structure
    shap_beeswarm_data       instance-level SHAP distributions
    feature_importance_bars  SHAP vs LIME side by side
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Matplotlib is configured headless by the notebook bootstrap (Agg on OAR).
import matplotlib.pyplot as plt

DPI = 300
FIGSIZE = (7.0, 4.2)


def _finish(fig, path: Optional[str], show: bool = False):
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"[plot] {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# =============================================================================
# QUANTITATIVE
# =============================================================================
def training_curves(histories: Dict[str, Dict], path: Optional[str] = None,
                    metrics: Sequence[str] = ("train_loss", "val_f1"),
                    ) -> Dict[str, np.ndarray]:
    """Learning curves, optionally averaged over seeds with a +/-1 std band.

    `histories` maps a label -> history dict (or label -> list of history dicts
    for a multi-seed run). Returns every array drawn, for `save_curve`.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.0))
    axes = np.atleast_1d(axes)
    out: Dict[str, np.ndarray] = {}

    for ax, metric in zip(axes, metrics):
        for label, hist in histories.items():
            runs = hist if isinstance(hist, list) else [hist]
            series = [np.asarray(h[metric], float) for h in runs if metric in h]
            if not series:
                continue
            n = min(len(s) for s in series)
            arr = np.stack([s[:n] for s in series])
            mean = arr.mean(0)
            epochs = np.arange(1, n + 1)

            ax.plot(epochs, mean, label=label, linewidth=1.8)
            if arr.shape[0] > 1:
                sd = arr.std(0, ddof=1)
                ax.fill_between(epochs, mean - sd, mean + sd, alpha=0.18)
                out[f"{label}__{metric}__std"] = sd
            out[f"{label}__{metric}__mean"] = mean
            out[f"{label}__{metric}__epochs"] = epochs

        ax.set_xlabel("epoch")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    _finish(fig, path)
    return out


def roc_curves(y_true, probs, class_names: Optional[Sequence[str]] = None,
               path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """One-vs-rest ROC with per-class AUC, plus macro average."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    probs = np.asarray(probs)
    n_cls = probs.shape[1]
    classes = list(range(n_cls))
    names = list(class_names) if class_names else [str(c) for c in classes]

    Y = label_binarize(y_true, classes=classes) if n_cls > 2 else \
        np.c_[1 - np.asarray(y_true), np.asarray(y_true)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    out, aucs = {}, []
    for i in range(n_cls):
        if Y[:, i].sum() == 0:
            continue
        fpr, tpr, thr = roc_curve(Y[:, i], probs[:, i])
        a = auc(fpr, tpr)
        aucs.append(a)
        ax.plot(fpr, tpr, linewidth=1.4, label=f"{names[i]} (AUC={a:.3f})")
        out[f"fpr__{names[i]}"] = fpr
        out[f"tpr__{names[i]}"] = tpr
        out[f"thresholds__{names[i]}"] = thr

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"one-vs-rest ROC (macro AUC = {np.mean(aucs):.3f})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    _finish(fig, path)
    out["macro_auc"] = np.array([np.mean(aucs)])
    return out


def pr_curves(y_true, probs, class_names: Optional[Sequence[str]] = None,
              path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Precision-recall curves.

    Under the 66.8% DDoS prevalence of this test set, PR is more discriminating
    between competing models than ROC -- which is why the paper reports it.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    probs = np.asarray(probs)
    n_cls = probs.shape[1]
    classes = list(range(n_cls))
    names = list(class_names) if class_names else [str(c) for c in classes]
    Y = label_binarize(y_true, classes=classes) if n_cls > 2 else \
        np.c_[1 - np.asarray(y_true), np.asarray(y_true)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    out = {}
    for i in range(n_cls):
        if Y[:, i].sum() == 0:
            continue
        p, r, thr = precision_recall_curve(Y[:, i], probs[:, i])
        ap = average_precision_score(Y[:, i], probs[:, i])
        ax.plot(r, p, linewidth=1.4, label=f"{names[i]} (AP={ap:.3f})")
        out[f"precision__{names[i]}"] = p
        out[f"recall__{names[i]}"] = r

    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("precision-recall (one-vs-rest)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    _finish(fig, path)
    return out


def calibration_curve(y_true, probs, n_bins: int = 10,
                      path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Reliability diagram + Brier score.

    Not part of the original paper's figures, but cheap and directly relevant to the
    deployment story: an IDS that is over-confident on rare attack classes is
    operationally worse than its accuracy suggests.
    """
    probs = np.asarray(probs)
    y = np.asarray(y_true)
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    acc_bin = np.full(n_bins, np.nan)
    conf_bin = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = m.sum()
        if m.any():
            acc_bin[b] = correct[m].mean()
            conf_bin[b] = conf[m].mean()

    valid = ~np.isnan(acc_bin)
    ece = float(np.sum(cnt[valid] / cnt.sum() * np.abs(acc_bin[valid] - conf_bin[valid])))
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
    ax.plot(conf_bin[valid], acc_bin[valid], "o-", linewidth=1.6, label="model")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(f"calibration (ECE={ece:.4f}, Brier={brier:.4f})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _finish(fig, path)

    return {"bin_confidence": conf_bin, "bin_accuracy": acc_bin,
            "bin_count": cnt, "ece": np.array([ece]), "brier": np.array([brier])}


def confusion_matrix_plot(cm, class_names: Optional[Sequence[str]] = None,
                          normalise: bool = True,
                          path: Optional[str] = None) -> Dict[str, np.ndarray]:
    cm = np.asarray(cm, float)
    names = list(class_names) if class_names else [str(i) for i in range(len(cm))]
    show = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalise else cm

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(show, cmap="Blues", vmin=0, vmax=show.max())
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(names)):
        for j in range(len(names)):
            v = show[i, j]
            ax.text(j, i, f"{v:.2f}" if normalise else f"{int(v)}",
                    ha="center", va="center", fontsize=7,
                    color="white" if v > show.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("confusion matrix" + (" (row-normalised)" if normalise else ""))
    _finish(fig, path)
    return {"confusion_matrix": np.asarray(cm),
            "confusion_matrix_normalised": show}


def per_class_bars(report: Dict, path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Per-class precision/recall/F1 with support annotated.

    Support matters: the paper reports F1 on classes with n=4 and n=15,
    which is why the enlarged test set was introduced.
    """
    names = report["class_names"]
    p = np.asarray(report["precision"], float)
    r = np.asarray(report["recall"], float)
    f = np.asarray(report["f1"], float)
    s = np.asarray(report["support"], int)

    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(7.0, len(names) * 1.05), 4.4))
    ax.bar(x - w, p, w, label="precision")
    ax.bar(x, r, w, label="recall")
    ax.bar(x + w, f, w, label="F1")
    for i, n in enumerate(s):
        ax.text(x[i], 1.02, f"n={n}", ha="center", fontsize=7,
                color="crimson" if n < 30 else "dimgray")
    ax.set_xticks(x, names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    ax.set_title("per-class performance (red support = statistically thin)")
    _finish(fig, path)
    return {"precision": p, "recall": r, "f1": f, "support": s}


def metric_vs_sweep(records: Sequence[Dict], x_key: str, y_key: str,
                    group_key: Optional[str] = None,
                    path: Optional[str] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Any metric against any swept parameter, with a +/-1 std band over seeds."""
    import collections

    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    dropped = 0
    for rec in records:
        x, y = rec.get(x_key), rec.get(y_key)
        if x is None or y is None:
            # A run saved before a field existed, or one whose metric could not
            # be computed. Averaging it raises `NoneType + NoneType`; dropping
            # it plots the runs that ARE complete and says how many went.
            dropped += 1
            continue
        g = rec.get(group_key, "all") if group_key else "all"
        groups[g][x].append(y)

    if dropped:
        print(f"[plot] {dropped} record(s) skipped: missing '{x_key}' or "
              f"'{y_key}'", flush=True)
    if not groups:                      # nothing usable for this slice yet
        return {}

    fig, ax = plt.subplots(figsize=FIGSIZE)
    out = {}
    for g, by_x in groups.items():
        xs = sorted(by_x, key=lambda v: (v is None, v))
        mean = np.array([np.mean(by_x[x]) for x in xs], float)
        sd = np.array([np.std(by_x[x], ddof=1) if len(by_x[x]) > 1 else 0.0
                       for x in xs], float)
        pos = np.arange(len(xs))
        ax.plot(pos, mean, "o-", linewidth=1.7, label=str(g))
        if (sd > 0).any():
            ax.fill_between(pos, mean - sd, mean + sd, alpha=0.18)
        ax.set_xticks(pos, [str(x) for x in xs])
        out[f"{g}__x"] = np.array([str(x) for x in xs])
        out[f"{g}__mean"] = mean
        out[f"{g}__std"] = sd

    ax.set_xlabel(xlabel or x_key)
    ax.set_ylabel(ylabel or y_key)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _finish(fig, path)
    return out


def seed_boxplot(values_by_group: Dict[str, Sequence[float]],
                 ylabel: str = "score",
                 path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Across-seed distribution of a metric."""
    labels = list(values_by_group)
    data = [np.asarray(values_by_group[k], float) for k in labels]

    fig, ax = plt.subplots(figsize=(max(6.0, len(labels) * 1.2), 4.2))
    ax.boxplot(data, labels=labels, showmeans=True)
    for i, d in enumerate(data, start=1):
        ax.plot(np.random.default_rng(0).normal(i, 0.04, len(d)), d,
                "o", alpha=0.6, markersize=4)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    _finish(fig, path)
    return {k: np.asarray(v, float) for k, v in values_by_group.items()}


def pareto_scatter(records: Sequence[Dict], x_key: str = "footprint_kb",
                   y_key: str = "accuracy", label_key: str = "model",
                   path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Accuracy against deployment cost.

    The paper's compactness claim is a Pareto claim, so it should be drawn
    as one rather than asserted in prose.
    """
    xs = np.array([r[x_key] for r in records], float)
    ys = np.array([r[y_key] for r in records], float)
    labels = [str(r[label_key]) for r in records]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(xs, ys, s=60)
    for x, y, l in zip(xs, ys, labels):
        ax.annotate(l, (x, y), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")

    order = np.argsort(xs)
    best, fx, fy = -np.inf, [], []
    for i in order:
        if ys[i] > best:
            best = ys[i]
            fx.append(xs[i])
            fy.append(ys[i])
    ax.step(fx, fy, where="post", linestyle="--", alpha=0.6,
            label="Pareto frontier")

    ax.set_xscale("log")
    ax.set_xlabel(f"{x_key} (log)")
    ax.set_ylabel(y_key)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _finish(fig, path)
    return {"x": xs, "y": ys, "labels": np.array(labels),
            "frontier_x": np.array(fx), "frontier_y": np.array(fy)}


# =============================================================================
# QUALITATIVE
# =============================================================================
def class_distribution(y_by_split: Dict[str, np.ndarray],
                       class_names: Optional[Sequence[str]] = None,
                       path: Optional[str] = None) -> Dict[str, np.ndarray]:
    names = list(class_names) if class_names else None
    n_cls = int(max(arr.max() for arr in y_by_split.values())) + 1
    names = names or [str(i) for i in range(n_cls)]

    x = np.arange(n_cls)
    w = 0.8 / len(y_by_split)
    fig, ax = plt.subplots(figsize=(max(7.0, n_cls * 1.05), 4.2))
    out = {}
    for k, (split, y) in enumerate(y_by_split.items()):
        counts = np.bincount(np.asarray(y), minlength=n_cls)
        ax.bar(x + k * w - 0.4 + w / 2, counts, w, label=split)
        out[f"counts__{split}"] = counts

    ax.set_yscale("log")
    ax.set_xticks(x, names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("count (log)")
    ax.set_title("class distribution by split")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    _finish(fig, path)
    return out


def pca_variance(explained_ratio, n_keep: Optional[int] = None,
                 path: Optional[str] = None) -> Dict[str, np.ndarray]:
    ev = np.asarray(explained_ratio, float)
    cum = np.cumsum(ev)

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax[0].bar(np.arange(1, len(ev) + 1), ev)
    ax[0].set_xlabel("component")
    ax[0].set_ylabel("explained variance ratio")
    ax[0].grid(alpha=0.3, axis="y")

    ax[1].plot(np.arange(1, len(cum) + 1), cum, "o-")
    if n_keep:
        ax[1].axvline(n_keep, linestyle="--", color="crimson")
        ax[1].annotate(f"{n_keep} comps = {cum[n_keep-1]*100:.1f}%",
                       (n_keep, cum[n_keep - 1]), fontsize=8,
                       xytext=(6, -14), textcoords="offset points")
    ax[1].set_xlabel("components")
    ax[1].set_ylabel("cumulative explained variance")
    ax[1].grid(alpha=0.3)
    _finish(fig, path)
    return {"explained_ratio": ev, "cumulative": cum}


def tsne_projection(X, y, class_names: Optional[Sequence[str]] = None,
                    n_sample: int = 3000, seed: int = 42,
                    path: Optional[str] = None) -> Dict[str, np.ndarray]:
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(seed)
    X, y = np.asarray(X), np.asarray(y)
    if len(X) > n_sample:
        idx = rng.choice(len(X), n_sample, replace=False)
        X, y = X[idx], y[idx]

    emb = TSNE(n_components=2, random_state=seed, init="pca",
               perplexity=min(30, max(5, len(X) // 100))).fit_transform(X)

    names = list(class_names) if class_names else \
        [str(c) for c in range(int(y.max()) + 1)]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for c in np.unique(y):
        m = y == c
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.6,
                   label=f"{names[int(c)]} (n={m.sum()})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title("t-SNE of the PCA representation")
    _finish(fig, path)
    return {"embedding": emb, "labels": y}


def correlation_heatmap(X, feature_names: Optional[Sequence[str]] = None,
                        path: Optional[str] = None) -> Dict[str, np.ndarray]:
    C = np.corrcoef(np.asarray(X, float).T)
    names = list(feature_names) if feature_names else \
        [str(i) for i in range(C.shape[0])]

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(names)), names, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("feature correlation")
    _finish(fig, path)
    return {"correlation": C, "feature_names": np.array(names)}


def feature_importance_bars(importances: Dict[str, Sequence[float]],
                            feature_names: Sequence[str], top_k: int = 15,
                            path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """SHAP vs LIME side by side -- whether the pair adds anything over
    either alone, so they must be drawn comparably."""
    ref = np.asarray(list(importances.values())[0], float)
    order = np.argsort(-np.abs(ref))[:top_k]
    names = [feature_names[i] for i in order]

    x = np.arange(len(order))
    w = 0.8 / len(importances)
    fig, ax = plt.subplots(figsize=(7.6, max(4.0, len(order) * 0.32)))
    out = {}
    for k, (label, vals) in enumerate(importances.items()):
        v = np.asarray(vals, float)[order]
        ax.barh(x + k * w - 0.4 + w / 2, v, w, label=label)
        out[f"importance__{label}"] = v

    ax.set_yticks(x, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("importance")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8)
    _finish(fig, path)
    out["feature_names"] = np.array(names)
    return out


def shap_beeswarm_data(shap_values, X, feature_names: Sequence[str],
                       top_k: int = 12,
                       path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Beeswarm drawn from raw arrays, so the values are saved alongside."""
    S = np.asarray(shap_values, float)
    Xv = np.asarray(X, float)
    order = np.argsort(-np.abs(S).mean(0))[:top_k]

    fig, ax = plt.subplots(figsize=(7.2, max(4.0, len(order) * 0.34)))
    rng = np.random.default_rng(0)
    for row, j in enumerate(order):
        v = S[:, j]
        c = Xv[:, j]
        c = (c - c.min()) / (np.ptp(c) + 1e-12)
        ax.scatter(v, row + rng.normal(0, 0.07, len(v)), c=c, cmap="coolwarm",
                   s=7, alpha=0.7)
    ax.set_yticks(range(len(order)), [feature_names[j] for j in order], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.7)
    ax.set_xlabel("SHAP value")
    _finish(fig, path)
    return {"shap_values": S[:, order],
            "feature_values": Xv[:, order],
            "feature_names": np.array([feature_names[j] for j in order])}
