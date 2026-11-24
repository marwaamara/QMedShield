"""
Regenerates four figures from the stored artefacts (no model is re-run):
  roc_curves.png, precision_recall_curves.png  <- binary/multiclass_predictions.npz
  kfold_cv_results.png                         <- cv_results.json   (2 folds, not 5)
  sample_efficiency.png                        <- sample_efficiency_results.json
Why: the earlier ROC / PR plots drew several classes in the same colour, and the cross-validation
figure was titled '5-Fold' although the stored results hold two folds.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score

H = Path(__file__).parent
ART = H / "archive"
OUT = H / "figures"
CLASSES = ["Benign", "ARP Spoofing", "Reconnaissance", "MQTT Malformed", "MQTT DoS", "MQTT DDoS", "DoS", "DDoS"]
# Okabe-Ito palette (colour-blind safe) + line styles so that no two classes look alike
OI = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
LS = ["-", "-", "-", "-", "-", "--", "--", "--"]
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

b = np.load(ART / "binary_predictions.npz")
m = np.load(ART / "multiclass_predictions.npz")
yb, pb = b["labels"], b["probabilities"][:, 1]
ym, Pm = m["labels"], m["probabilities"]

# sanity: the weighted one-vs-rest AUC must equal the value reported in the paper (0.8366)
w_auc = roc_auc_score(ym, Pm, multi_class="ovr", average="weighted")
print("weighted OVR AUC (multiclass):", round(w_auc, 4), " binary AUC:", round(roc_auc_score(yb, pb), 4))

# ---------------- ROC
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
fpr, tpr, _ = roc_curve(yb, pb)
ax[0].plot(fpr, tpr, color="#0072B2", lw=2, label=f"Attack vs benign (AUC = {auc(fpr, tpr):.4f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1)
ax[0].set(title="Binary ROC curve", xlabel="False positive rate", ylabel="True positive rate"); ax[0].legend(loc="lower right")
for k, name in enumerate(CLASSES):
    yk = (ym == k).astype(int)
    if yk.sum() == 0: continue
    f, t, _ = roc_curve(yk, Pm[:, k])
    ax[1].plot(f, t, color=OI[k], ls=LS[k], lw=1.8, label=f"{name} (AUC {auc(f, t):.3f}, n = {int(yk.sum())})")
ax[1].plot([0, 1], [0, 1], "k--", lw=1)
ax[1].set(title="Multiclass ROC curves (one-vs-rest)", xlabel="False positive rate", ylabel="True positive rate")
ax[1].legend(loc="lower right", fontsize=8, frameon=True)
fig.tight_layout(); fig.savefig(OUT / "roc_curves.png", dpi=300); plt.close(fig)

# ---------------- PR
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
p, r, _ = precision_recall_curve(yb, pb)
ax[0].plot(r, p, color="#0072B2", lw=2, label=f"Attack vs benign (AP = {average_precision_score(yb, pb):.4f})")
ax[0].set(title="Binary precision-recall curve", xlabel="Recall", ylabel="Precision", ylim=(0.97, 1.001)); ax[0].legend(loc="lower left")
for k, name in enumerate(CLASSES):
    yk = (ym == k).astype(int)
    if yk.sum() == 0: continue
    p, r, _ = precision_recall_curve(yk, Pm[:, k])
    ax[1].plot(r, p, color=OI[k], ls=LS[k], lw=1.8, label=f"{name} (AP {average_precision_score(yk, Pm[:, k]):.3f})")
ax[1].set(title="Multiclass precision-recall curves (one-vs-rest)", xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
ax[1].legend(loc="lower left", fontsize=8, frameon=True)
fig.tight_layout(); fig.savefig(OUT / "precision_recall_curves.png", dpi=300); plt.close(fig)

# ---------------- cross-validation (two folds)
cv = json.load(open(ART / "cv_results.json"))
names = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("auc", "AUC")]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for a, task, title in ((ax[0], "binary", "Binary, stratified 2-fold CV"), (ax[1], "multiclass", "Multiclass, stratified 2-fold CV")):
    for j, (key, lab) in enumerate(names):
        v = np.array(cv[task][key])
        a.scatter(np.full(len(v), j), v, color="#0072B2", s=45, zorder=3, label="Fold value" if j == 0 else None)
        a.hlines(v.mean(), j - 0.25, j + 0.25, color="#D55E00", lw=2, zorder=2, label="Mean" if j == 0 else None)
    a.set_xticks(range(len(names))); a.set_xticklabels([n[1] for n in names]); a.set_title(title)
    lo = min(min(cv[task][k]) for k, _ in names); a.set_ylim(max(0, lo - 0.02), 1.005 if task == "binary" else min(1.0, max(max(cv[task][k]) for k, _ in names) + 0.03))
    a.set_ylabel("Score"); a.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "kfold_cv_results.png", dpi=300); plt.close(fig)

# ---------------- sample efficiency (absolute training-set sizes used by the experiment)
se = json.load(open(ART / "sample_efficiency_results.json"))
sizes = [1000, 2000, 5000, 10000, 20000]
lab = {"QCNN": "QCNN (hybrid)", "Random Forest": "Random Forest", "Gradient Boosting": "Gradient Boosting", "MLP_classical": "MLP"}
sty = {"QCNN": ("#D55E00", "o", "-"), "Random Forest": ("#0072B2", "s", "--"), "Gradient Boosting": ("#009E73", "^", "--"), "MLP_classical": ("#CC79A7", "D", "--")}
fig, ax = plt.subplots(figsize=(8, 5))
for k, vals in se.items():
    c, mk, ls = sty[k]
    ax.plot(sizes[:len(vals)], vals, color=c, marker=mk, ls=ls, lw=2, ms=7, label=lab[k])
ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels([f"{s:,}" for s in sizes]); ax.minorticks_off()
ax.set(xlabel="Number of training samples", ylabel="Test weighted F1 (multiclass)", title="Sample efficiency (single run per point)")
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "sample_efficiency.png", dpi=300); plt.close(fig)
print("figures written")
