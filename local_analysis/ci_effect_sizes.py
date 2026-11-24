"""
ci_effect_sizes.py -- confidence intervals and effect sizes.

(A) Bootstrap 95% CIs (percentile, 10,000 resamples of the 7,500 test flows) for the
    seed-42 headline model, and PAIRED bootstrap CIs for the accuracy difference
    against each classical baseline (same flows resampled for both models).
    Also Cohen's g-style paired effect (b-c)/(b+c) and the odds ratio b/c from the
    McNemar discordant counts.
(B) Seed-level effect sizes for the parameter-matched controls, from the five-seed
    means/SDs reported in Table 'matched': Hedges' g with a 95% CI (Hedges & Olkin SE),
    n=5 per arm, plus the raw accuracy difference in percentage points.
Everything is computed from stored artefacts; no model is retrained.
"""
import json, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
H = Path(__file__).parent; OUT = H / "results"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0); B = 10000

# ---------------- (A) per-sample bootstrap -----------------------------------
a = np.load(H / "archive" / "multiclass_predictions.npz"); y = a["labels"]; p_q = a["predictions"]
n = len(y); idx = rng.integers(0, n, (B, n))
def ci(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
corr_q = (p_q == y).astype(float)
acc_b = corr_q[idx].mean(1)
f1_b = np.array([f1_score(y[i], p_q[i], average="weighted") for i in idx[:1000]])  # 1000 resamples (cost)
out = {"headline_multiclass_seed42": {"acc": float(corr_q.mean()), "acc_ci95": ci(acc_b),
       "f1w": float(f1_score(y, p_q, average="weighted")), "f1w_ci95_1000boot": ci(f1_b)}}
ab = np.load(H / "archive" / "binary_predictions.npz"); yb, pb = ab["labels"], ab["predictions"]
cb = (pb == yb).astype(float); ib = rng.integers(0, len(yb), (B, len(yb)))
out["headline_binary_seed42"] = {"acc": float(cb.mean()), "acc_ci95": ci(cb[ib].mean(1))}

paired = {}
for d in sorted((H / "inputs" / "baseline_predictions").iterdir()):
    z = np.load(d / "predictions.npz"); assert (z["y_true"] == y).all()
    c_o = (z["y_pred"] == y).astype(float)
    diff = (corr_q[idx].mean(1) - c_o[idx].mean(1)) * 100          # pp, quantum minus baseline
    b_ = int(((corr_q == 1) & (c_o == 0)).sum()); c_ = int(((corr_q == 0) & (c_o == 1)).sum())
    paired[d.name.split("__")[0]] = {
        "acc_baseline": float(c_o.mean()), "diff_pp": float((corr_q.mean() - c_o.mean()) * 100),
        "diff_pp_ci95": ci(diff), "discordant_ours_only": b_, "discordant_baseline_only": c_,
        "odds_ratio": (b_ / c_) if c_ else None, "g_discordance": (b_ - c_) / (b_ + c_)}
out["paired_vs_baselines_multiclass_seed42"] = paired

# ---------------- (B) seed-level Hedges g from reported mean/SD ---------------
def hedges(m1, s1, m2, s2, n1=5, n2=5):
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (m1 - m2) / sp; J = 1 - 3 / (4 * (n1 + n2) - 9); g = J * d
    se = np.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2)))
    return float(g), [float(g - 1.96 * se), float(g + 1.96 * se)]
pqc = (0.7839, 0.0069)
rows = {"Random Fourier map": (0.7913, 0.0012), "Tiny MLP": (0.7897, 0.0023),
        "Dense linear map": (0.7884, 0.0054), "Identity (no bottleneck)": (0.7857, 0.0050)}
seedlvl = {}
for k, (m, s) in rows.items():
    g, c = hedges(m, s, *pqc)
    seedlvl[k] = {"diff_pp_vs_pqc": round((m - pqc[0]) * 100, 2), "hedges_g": round(g, 2), "g_ci95": [round(x, 2) for x in c]}
out["matched_controls_vs_pqc_hedges_g"] = seedlvl
# classical twin trained in this folder (if present)
import glob
tw = sorted(glob.glob(str(H / "results" / "twin_multiclass_seed*.json")))
if tw:
    v = np.array([list(json.load(open(f))["seeds"].values())[0]["acc"] for f in tw])
    out["twin_rerun_for_timing_and_robustness_5seed"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
    g, c = hedges(v.mean(), v.std(ddof=1), *pqc); out["twin_rerun_for_timing_and_robustness_5seed"].update({"hedges_g_vs_pqc": round(g, 2), "g_ci95": [round(x, 2) for x in c]})
json.dump(out, open(OUT / "ci_effect_sizes.json", "w"), indent=1)
print(json.dumps(out, indent=1))
