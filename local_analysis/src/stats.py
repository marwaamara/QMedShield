"""
stats.py -- multi-seed aggregation and paired significance testing.

Repeating experiments over multiple random seeds and reporting mean
performance together with standard deviations or confidence intervals
strengthens the empirical evidence over single runs. The reported spread of a 2-fold CV
(+/- 0.0002) is not a variance estimate. This module provides
multi-seed aggregation, and the paired tests that the saved per-sample
predictions make possible.

Test selection:
  * McNemar          -- paired, per-sample, for comparing two classifiers on the
                        SAME test set. The correct test for binary agreement;
                        used with the exact binomial when discordant pairs are few.
  * Wilcoxon         -- paired across seeds, when comparing two methods over a
                        common seed set (n = 5 here, so exact and low-powered:
                        report the effect size alongside).
  * Friedman         -- omnibus across >2 methods over a common seed set.
  * Bootstrap CI     -- for any metric, from the saved per-sample predictions.
  * Holm-Bonferroni  -- family-wise correction across the baseline family.

A note worth reporting alongside the results: with 5 seeds, Wilcoxon's minimum
attainable p-value is 0.0625, so no pairwise seed-level comparison can reach
p < 0.05. Paired McNemar on per-sample predictions is therefore the test that
carries the statistical weight; the seed-level tests describe stability.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Aggregation
# =============================================================================
def summarise(values: Sequence[float], confidence: float = 0.95) -> Dict:
    """mean, std, and a t-based CI appropriate for a small seed count."""
    from scipy import stats as sps

    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "ci_low": np.nan,
                "ci_high": np.nan}
    mean, std = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = sps.t.ppf(0.5 + confidence / 2, n - 1) * std / np.sqrt(n)
    else:
        half = np.nan
    return {"n": n, "mean": mean, "std": std,
            "ci_low": mean - half, "ci_high": mean + half}


def summarise_runs(records: Iterable[Dict], metric: str,
                   group_keys: Sequence[str] = ("variant", "task")) -> List[Dict]:
    """Group saved run manifests and summarise one metric across seeds."""
    groups: Dict[Tuple, List[float]] = {}
    for r in records:
        key = tuple(r.get(k) for k in group_keys)
        val = (r.get("metrics") or {}).get(metric)
        if val is not None:
            groups.setdefault(key, []).append(float(val))

    out = []
    for key, vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row = dict(zip(group_keys, key))
        row["metric"] = metric
        row.update(summarise(vals))
        row["values"] = vals
        out.append(row)
    return out


def format_mean_std(row: Dict, dp: int = 4) -> str:
    """Render 'mean +/- std' for a LaTeX table cell."""
    if row["n"] <= 1:
        return f"{row['mean']:.{dp}f}"
    return f"{row['mean']:.{dp}f} ± {row['std']:.{dp}f}"


# =============================================================================
# Paired tests
# =============================================================================
def mcnemar(y_true, pred_a, pred_b, exact: Optional[bool] = None) -> Dict:
    """Paired McNemar test on per-sample predictions.

    b = A right / B wrong, c = A wrong / B right. Uses the exact binomial when
    b + c < 25, otherwise the continuity-corrected chi-square.
    """
    from scipy import stats as sps

    y = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y
    b_ok = np.asarray(pred_b) == y
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n_disc = b + c

    if exact is None:
        exact = n_disc < 25

    if n_disc == 0:
        return {"b": b, "c": c, "n_discordant": 0, "statistic": 0.0,
                "p_value": 1.0, "test": "degenerate (identical predictions)"}

    if exact:
        p = float(sps.binomtest(min(b, c), n_disc, 0.5).pvalue)
        stat, test = float(min(b, c)), "exact binomial"
    else:
        stat = (abs(b - c) - 1.0) ** 2 / n_disc
        p = float(sps.chi2.sf(stat, df=1))
        test = "chi-square (continuity-corrected)"

    return {"b": b, "c": c, "n_discordant": n_disc, "statistic": float(stat),
            "p_value": p, "test": test,
            "acc_a": float(a_ok.mean()), "acc_b": float(b_ok.mean())}


def wilcoxon_seeds(a: Sequence[float], b: Sequence[float],
                   alternative: str = "two-sided") -> Dict:
    """Paired Wilcoxon across seeds, with the power caveat made explicit."""
    from scipy import stats as sps

    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < 3 or np.allclose(a, b):
        return {"n": n, "statistic": np.nan, "p_value": 1.0,
                "note": "too few seeds or identical values"}
    stat, p = sps.wilcoxon(a, b, zero_method="wilcox", alternative=alternative)
    return {"n": n, "statistic": float(stat), "p_value": float(p),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "min_attainable_p": 2.0 ** -(n - 1) * 2,
            "note": (f"with n={n} seeds the smallest attainable two-sided "
                     f"p-value is {2.0 ** -(n - 1) * 2:.4f}")}


def friedman(*groups: Sequence[float]) -> Dict:
    """Omnibus test across >2 methods measured on a common seed set."""
    from scipy import stats as sps

    if len(groups) < 3:
        return {"statistic": np.nan, "p_value": np.nan,
                "note": "Friedman needs at least 3 groups"}
    stat, p = sps.friedmanchisquare(*groups)
    return {"statistic": float(stat), "p_value": float(p), "k": len(groups),
            "n": len(groups[0])}


def holm_bonferroni(pvalues: Sequence[float],
                    labels: Optional[Sequence[str]] = None,
                    alpha: float = 0.05) -> List[Dict]:
    """Holm-Bonferroni step-down correction across a family of comparisons."""
    p = np.asarray(list(pvalues), float)
    m = len(p)
    labels = list(labels) if labels is not None else [str(i) for i in range(m)]

    order = np.argsort(p)
    adjusted = np.empty(m, float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)

    return [{"label": labels[i], "p_raw": float(p[i]),
             "p_holm": float(adjusted[i]), "reject": bool(adjusted[i] < alpha)}
            for i in range(m)]


# =============================================================================
# Bootstrap
# =============================================================================
def bootstrap_metric(y_true, y_pred, metric_fn, n_boot: int = 2000,
                     seed: int = 42, confidence: float = 0.95) -> Dict:
    """Percentile bootstrap CI for any metric, from saved predictions.

    Because per-sample predictions are persisted for every run, this can be
    added to a report at any time without re-running an experiment.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = metric_fn(y_true[idx], y_pred[idx])

    lo = float(np.percentile(stats, (1 - confidence) / 2 * 100))
    hi = float(np.percentile(stats, (1 + confidence) / 2 * 100))
    return {"point": float(metric_fn(y_true, y_pred)),
            "ci_low": lo, "ci_high": hi, "n_boot": n_boot}


# =============================================================================
# Convenience: full pairwise comparison of a method against a baseline family
# =============================================================================
def compare_against_family(y_true, pred_ours, family: Dict[str, np.ndarray],
                           alpha: float = 0.05) -> List[Dict]:
    """Paired McNemar of one method against each baseline, Holm-corrected."""
    rows, pvals, labels = [], [], []
    for name, pred in family.items():
        r = mcnemar(y_true, pred_ours, pred)
        r["baseline"] = name
        rows.append(r)
        pvals.append(r["p_value"])
        labels.append(name)

    for row, adj in zip(rows, holm_bonferroni(pvals, labels, alpha)):
        row["p_holm"] = adj["p_holm"]
        row["significant_holm"] = adj["reject"]
    return rows
