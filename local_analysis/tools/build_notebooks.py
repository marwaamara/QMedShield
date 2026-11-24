"""Generates the notebooks in laptop/notebooks/ (run from anywhere: python tools/build_notebooks.py)."""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "notebooks"
OUT.mkdir(exist_ok=True)

SETUP = '''# --- setup: work from the laptop/ folder so that relative paths resolve ---
import os, sys, json
from pathlib import Path
ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
pd.set_option("display.precision", 4)
print("working folder:", ROOT)'''


def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(n, OUT / name)
    print("wrote", name)


# ---------------------------------------------------------------- 00
nb("00_start_here.ipynb", [
 ("md", """# QMedShield: laptop experiments

These notebooks reproduce the **analysis, statistics, cost and robustness comparisons** of the paper from stored artefacts, on an ordinary laptop (no cluster).
The long training runs of the quantum model (about 8.7 h per multiclass run under simulation) are in `../../grid5000/`.

| Notebook | What it does | Needs | Runtime |
|---|---|---|---|
| 01 data and preprocessing | prepared data, class distribution, PCA, feature names | stored arrays | seconds |
| 02 models and parameter counts | builds the hybrid model, the classical bottlenecks and the twin; parameter and footprint tables | PyTorch, PennyLane | seconds |
| 03 reference models: results | metrics, bootstrap intervals, confusion matrix, ROC and precision-recall curves | stored predictions | 1 min |
| 04 baselines and paired statistics | paired bootstrap differences, odds ratios, McNemar, Hedges' g | stored predictions | 2 min |
| 05 classical twin: training and cost | twin accuracy over seeds, same-machine timing, explanation cost (optional retraining) | stored results (or retrain) | seconds (hours if retraining) |
| 06 robustness comparison | twin under raw-space perturbations and attacks | stored results (or rerun) | seconds (10 min if rerun) |
| 07 audit log and threat model | hash chain, tamper harness, anchored vs internal verification | none | 1 min |
| 08 DoS/DDoS windowed aggregation | windowed-aggregation test and feature diagnosis | stored results (raw CSVs to rerun) | seconds (10 min if rerun) |
| 09 figures and tables | regenerates the figures from stored results; table-to-file map | stored results | 1 min |

**Conventions.** Every notebook starts from the same setup cell, reads only from `archive/`, `inputs/` and `results/`, and writes to `results/` or `figures/`.
Seeds used throughout: 42, 1, 7, 13, 2024. Everything quantum is simulated on a state-vector backend; nothing is claimed about quantum hardware."""),
 ("code", SETUP),
 ("code", '''# environment check
import importlib
for m in ["numpy", "scipy", "pandas", "sklearn", "torch", "pennylane", "matplotlib"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"{m:12s} MISSING ({e.__class__.__name__})")

# data check
need = ["archive/prepared_data.npz", "archive/multiclass_predictions.npz", "archive/binary_predictions.npz",
        "archive/scaler.joblib", "archive/pca_multi.joblib", "inputs/runs_index_nb04.csv"]
for f in need:
    print(("ok      " if Path(f).exists() else "MISSING ") + f)'''),
])

# ---------------------------------------------------------------- 01
nb("01_data_and_preprocessing.ipynb", [
 ("md", """# 01 Data and preprocessing

**Purpose.** Inspect the prepared data used by the main experiments: a 50,000-flow stratified sample of the CICIoMT2024 WiFi/MQTT partition, 25 selected flow features, 16 PCA components, eight operational classes.

**Inputs.** `archive/prepared_data.npz`, `archive/scaler.joblib`, `archive/pca_multi.joblib`.

**Important note on preprocessing.** In the main experiments the constant and correlation filters, the ANOVA selector (25 of 35 features), the scaler and the PCA were fitted once on the 50,000-flow sample before splitting into training, validation and test partitions.
The split-protocol experiments (cluster notebook 16) re-fit every step inside each training partition. The paper discusses the consequence."""),
 ("code", SETUP),
 ("code", '''d = np.load("archive/prepared_data.npz", allow_pickle=True)
print({k: d[k].shape for k in d.files})
CLASSES = ["Benign", "ARP Spoofing", "Reconnaissance", "MQTT Malformed", "MQTT DoS", "MQTT DDoS", "DoS", "DDoS"]
tab = pd.DataFrame({"train": np.bincount(d["y_train_m"], minlength=8),
                    "validation": np.bincount(d["y_val_m"], minlength=8),
                    "test": np.bincount(d["y_test_m"], minlength=8)}, index=CLASSES)
tab["test %"] = 100 * tab["test"] / tab["test"].sum()
tab'''),
 ("code", '''import joblib
pca = joblib.load("archive/pca_multi.joblib")
print("PCA components:", pca.n_components_, " explained variance: %.1f%%" % (100 * pca.explained_variance_ratio_.sum()))
print("25 selected features:")
print([str(x) for x in d["kept_names"]])'''),
 ("code", '''import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
tab[["train", "validation", "test"]].plot.bar(ax=ax[0], log=True); ax[0].set_title("Class counts per partition"); ax[0].set_ylabel("flows (log)")
ax[1].plot(np.cumsum(pca.explained_variance_ratio_), marker="o"); ax[1].set_title("PCA cumulative explained variance"); ax[1].set_xlabel("component")
plt.tight_layout(); plt.show()'''),
])

# ---------------------------------------------------------------- 02
nb("02_models_and_parameter_counts.ipynb", [
 ("md", """# 02 Models and parameter counts

**Purpose.** Build the hybrid model (8-qubit circuit, 4 layers, data re-uploading), the parameter-matched classical bottlenecks and the classical twin from one factory, and show that the classical encoder and decoder are identical in every variant.

**Inputs.** `src/models.py`, `src/circuits.py`. **Outputs.** Parameter and footprint tables (paper: parameter-budget and ablation tables)."""),
 ("code", SETUP),
 ("code", '''from models import build_model
rows = []
for task in ["binary", "multiclass"]:
    for variant, label in [("full", "Hybrid (quantum circuit)"), ("linear", "Classical twin (dense 8x8 + tanh)"),
                           ("mlp96", "Tiny MLP (93 params)"), ("fourier", "Random Fourier map"), ("identity", "Identity (no bottleneck)")]:
        m = build_model(task, variant)
        b = m.param_breakdown()
        rows.append({"task": task, "model": label, **b, "footprint KB (32 bit)": round(m.footprint_kb(), 1)})
pd.DataFrame(rows)'''),
 ("code", '''# the encoder and decoder must be identical across variants (only the bottleneck changes)
for task in ["binary", "multiclass"]:
    counts = {v: build_model(task, v).param_breakdown() for v in ["full", "linear", "mlp96", "fourier", "identity"]}
    enc = {c["encoder"] for c in counts.values()}; dec = {c["decoder"] for c in counts.values()}
    print(task, "encoder sizes:", enc, " decoder sizes:", dec)
    assert len(enc) == 1 and len(dec) == 1'''),
])

# ---------------------------------------------------------------- 03
nb("03_reference_models_results.ipynb", [
 ("md", """# 03 Reference models: results

**Purpose.** Metrics of the reference (seed-42) binary and multiclass models on the 7,500-flow test partition, with percentile-bootstrap 95% intervals, the confusion matrix and the ROC and precision-recall curves.

**Inputs.** `archive/binary_predictions.npz`, `archive/multiclass_predictions.npz`, `archive/cv_results.json`. **Outputs.** `figures/roc_curves.png`, `figures/precision_recall_curves.png`."""),
 ("code", SETUP),
 ("code", '''from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, classification_report
CLASSES = ["Benign", "ARP Spoofing", "Reconnaissance", "MQTT Malformed", "MQTT DoS", "MQTT DDoS", "DoS", "DDoS"]
b = np.load("archive/binary_predictions.npz"); m = np.load("archive/multiclass_predictions.npz")
rng = np.random.default_rng(0); B = 2000
def boot(y, p, fn):
    idx = rng.integers(0, len(y), (B, len(y)))
    v = np.array([fn(y[i], p[i]) for i in idx]); return np.percentile(v, [2.5, 97.5])
rows = []
for name, d, avg in [("binary", b, "weighted"), ("multiclass", m, "weighted")]:
    y, p = d["labels"], d["predictions"]
    lo, hi = boot(y, p, accuracy_score)
    rows.append({"task": name, "accuracy": accuracy_score(y, p), "95% CI low": lo, "95% CI high": hi,
                 "weighted F1": f1_score(y, p, average=avg), "weighted OVR AUC": roc_auc_score(y, d["probabilities"][:, 1] if name == "binary" else d["probabilities"], **({} if name == "binary" else dict(multi_class="ovr", average="weighted")))})
pd.DataFrame(rows)'''),
 ("code", '''print(classification_report(m["labels"], m["predictions"], target_names=CLASSES, digits=3, zero_division=0))
cm = confusion_matrix(m["labels"], m["predictions"])
pd.DataFrame(cm, index=CLASSES, columns=CLASSES)'''),
 ("md", "The DoS/DDoS cells hold almost all of the multiclass error (see notebook 08)."),
 ("code", '''import subprocess
print(subprocess.run([sys.executable, "make_figures.py"], capture_output=True, text=True).stdout)
from IPython.display import Image, display
for f in ["roc_curves.png", "precision_recall_curves.png", "kfold_cv_results.png"]:
    display(Image(filename="figures/" + f, width=900))'''),
])

# ---------------------------------------------------------------- 04
nb("04_baselines_and_paired_statistics.ipynb", [
 ("md", """# 04 Baselines and paired statistics

**Purpose.** Compare the seed-42 hybrid model with nine classical comparators on the same 7,500 test flows: paired accuracy difference with a paired-bootstrap 95% interval, odds ratio of discordant pairs, McNemar test with Holm adjustment; and Hedges' g for the five-seed parameter-matched controls.

**Method notes.** Variation over test flows is handled by paired tests on the same flows; variation over training runs by mean, standard deviation and Hedges' g. A five-pair Wilcoxon test is not used as evidence (its minimum two-sided p is 0.0625). Absence of a significant difference is not proof of equivalence.

**Inputs.** `inputs/baseline_predictions/*/predictions.npz`, `archive/*_predictions.npz`. **Outputs.** `results/ci_effect_sizes.json`."""),
 ("code", SETUP),
 ("code", '''import subprocess
r = subprocess.run([sys.executable, "ci_effect_sizes.py"], capture_output=True, text=True)
print("exit code", r.returncode, r.stderr[-300:])
res = json.load(open("results/ci_effect_sizes.json"))
res["headline_multiclass_seed42"], res["headline_binary_seed42"]'''),
 ("code", '''from scipy.stats import chi2
paired = res["paired_vs_baselines_multiclass_seed42"]
rows = []
for k, v in paired.items():
    b_, c_ = v["discordant_ours_only"], v["discordant_baseline_only"]
    rows.append({"comparator": k, "comparator acc": v["acc_baseline"], "diff (pp)": v["diff_pp"],
                 "CI low": v["diff_pp_ci95"][0], "CI high": v["diff_pp_ci95"][1], "odds ratio b/c": v["odds_ratio"],
                 "p (McNemar)": float(chi2.sf((abs(b_ - c_) - 1) ** 2 / (b_ + c_), 1))})
t = pd.DataFrame(rows).sort_values("comparator acc", ascending=False).reset_index(drop=True)
order = t["p (McNemar)"].argsort().values; m_ = len(t); run = 0; holm = np.empty(m_)
for rank, i in enumerate(order):
    run = max(run, min(1.0, t.loc[i, "p (McNemar)"] * (m_ - rank))); holm[i] = run
t["p (Holm)"] = holm
t'''),
 ("code", '''pd.DataFrame(res["matched_controls_vs_pqc_hedges_g"]).T'''),
])

# ---------------------------------------------------------------- 05
nb("05_classical_twin_training_and_cost.ipynb", [
 ("md", """# 05 Classical twin: training and cost

**Purpose.** Compare the complete hybrid pipeline with a *classical twin*: identical encoder, decoder, optimiser, schedule, validation-only checkpointing, explanation interface and audit chain; only the 8-qubit circuit is replaced by a dense 8x8 map with tanh (72 instead of 96 bottleneck parameters).

**What is stored.** Accuracy of the five re-trained twin models, and same-machine, single-thread timings of the simulated quantum model and the twin.
**Optional.** Set `RETRAIN = True` to retrain the twin (about 4 minutes per seed on one CPU thread; run the five seeds in parallel processes if you have the cores).

Absolute times depend on the machine; only the ratios between the two models are meaningful."""),
 ("code", SETUP),
 ("code", '''RETRAIN = False        # True: retrain the five twin models (a few minutes per seed)
RETIME  = False        # True: repeat the same-machine timing (needs the trained twin of seed 42)
import subprocess, glob
if RETRAIN:
    procs = [subprocess.Popen([sys.executable, "twin_pipeline.py", "multiclass", "train", str(s)]) for s in [42, 1, 7, 13, 2024]]
    [p.wait() for p in procs]
if RETIME:
    subprocess.run([sys.executable, "pqc_epoch_time.py"]); subprocess.run([sys.executable, "twin_pipeline.py", "multiclass", "time"])'''),
 ("code", '''rows = []
for f in sorted(glob.glob("results/twin_multiclass_seed*.json")):
    r = json.load(open(f))["seeds"]; s, v = list(r.items())[0]
    rows.append({"seed": int(s), "accuracy": v["acc"], "weighted F1": v["f1w"], "epochs run": v["epochs_run"]})
tw = pd.DataFrame(rows).set_index("seed"); display(tw)
print("mean accuracy %.4f, SD %.4f" % (tw["accuracy"].mean(), tw["accuracy"].std(ddof=1)))'''),
 ("code", '''T = json.load(open("results/twin_multiclass.json")); E = json.load(open("results/pqc_epoch_time.json"))
lat = T["latency_ms_per_sample"]; shp = T["kernelshap_forward_s_per_explanation"]
cost = pd.DataFrame([
  {"quantity": "parameters (total)", "quantum (simulated)": T["params"]["pqc"]["total"], "classical twin": T["params"]["twin"]["total"]},
  {"quantity": "training, s per epoch", "quantum (simulated)": E["pqc"]["epoch_s_extrapolated"], "classical twin": E["twin"]["epoch_s_extrapolated"]},
  {"quantity": "inference, ms/flow, batch 1", "quantum (simulated)": lat["1"]["pqc"], "classical twin": lat["1"]["twin"]},
  {"quantity": "inference, ms/flow, batch 128", "quantum (simulated)": lat["128"]["pqc"], "classical twin": lat["128"]["twin"]},
  {"quantity": "forward cost of one explanation, s", "quantum (simulated)": shp["pqc"], "classical twin": shp["twin"]}])
cost["ratio (quantum / twin)"] = cost["quantum (simulated)"] / cost["classical twin"]
cost'''),
])

# ---------------------------------------------------------------- 06
nb("06_robustness_comparison.ipynb", [
 ("md", """# 06 Robustness comparison

**Purpose.** Apply the paper's robustness protocol to the classical twin: four perturbations in the **original feature space** (packet loss, timing jitter, missing features, protocol-mix shift, applied before scaling and PCA), white-box FGSM and PGD in the projected space, and a gradient-free attack restricted to attacker-controllable features.

**Scope.** These results apply only to the perturbation models and attacks that were run, against single trained models. The quantum figures are those of the archived seed-42 model reported in the paper (cluster notebook 07).
**Optional.** `RERUN = True` repeats the twin evaluation for the five seeds (about 10 minutes)."""),
 ("code", SETUP),
 ("code", '''RERUN = False
import subprocess
if RERUN:
    subprocess.run([sys.executable, "robustness_twin_allseeds.py"])
A = json.load(open("results/twin_robustness_allseeds.json"))
def stat(kind, **kw):
    v = []
    for sd, r in A.items():
        if kind == "raw": v.append(next(x["accuracy"] for x in r["raw_space"] if x["perturbation"] == kw["p"] and x["severity"] == kw["s"]))
        elif kind == "adv": v.append(next(x["accuracy"] for x in r["adversarial_pca"] if x["attack"] == kw["a"] and x["epsilon"] == kw["e"]))
        else: v.append(next(x["accuracy"] for x in r["constrained"] if x["eps_rel"] == kw["e"]))
    v = np.array(v); return v.mean(), v.std(ddof=1)
# quantum values: archived seed-42 model, as reported in the paper (cluster notebook 07)
Q = {"jitter 20%": 0.5841, "missing 20%": 0.6733, "packet loss 20%": 0.7599, "PGD-10 eps=0.02": 0.5127, "constrained 2%": 0.5870}
Q0 = {"jitter 20%": 0.7919, "missing 20%": 0.7919, "packet loss 20%": 0.7919, "PGD-10 eps=0.02": 0.7919, "constrained 2%": 0.7880}
specs = {"jitter 20%": ("raw", dict(p="jitter", s=0.2), dict(p="jitter", s=0.0)),
         "missing 20%": ("raw", dict(p="missing", s=0.2), dict(p="missing", s=0.0)),
         "packet loss 20%": ("raw", dict(p="packet_loss", s=0.2), dict(p="packet_loss", s=0.0)),
         "PGD-10 eps=0.02": ("adv", dict(a="PGD-10", e=0.02), dict(a="PGD-10", e=0.0)),
         "constrained 2%": ("con", dict(e=0.02), dict(e=0.0))}
rows = []
for k, (kind, kw, kw0) in specs.items():
    m1, s1 = stat(kind, **kw); m0, _ = stat(kind, **kw0)
    rows.append({"condition": k, "quantum acc": Q[k], "quantum change (pp)": 100 * (Q[k] - Q0[k]),
                 "twin acc (mean of 5 seeds)": m1, "twin SD": s1, "twin change (pp)": 100 * (m1 - m0)})
pd.DataFrame(rows)'''),
])

# ---------------------------------------------------------------- 07
nb("07_audit_log_threat_model.ipynb", [
 ("md", """# 07 Audit log and threat model

**Purpose.** Demonstrate the SHA-256 hash-chain audit log and test which manipulations it detects.

**Threat model in brief.** The chain attests to the order and integrity of records *as written*. Full tamper detection depends on an independently protected **head digest and record count**:
without that anchor, an adversary with write access to the whole log can edit a record and recompute every later link, and internal verification passes. The chain says nothing about the correctness of the verdicts it records, provides integrity but not confidentiality, and gives order but not trusted time.
The mechanism supports tamper-evident decision provenance relevant to healthcare audit requirements, without establishing compliance with any specific regulatory framework."""),
 ("code", SETUP),
 ("code", '''import random
from audit import HashChainLog, make_event
rng = random.Random(0)
chain = HashChainLog()
for i in range(1000):
    chain.append(make_event(i, rng))
head, n = chain.head, len(chain.records)          # the anchor: stored OUTSIDE the log
print("valid chain:", chain.verify(), " head:", head[:16], "... records:", n)

chain.records[500]["payload"]["predicted_class"] ^= 1          # in-place edit
print("after in-place edit:", chain.verify())'''),
 ("code", '''# recompute-after-edit: internal verification passes, the external anchor catches it
def rehash(c, start):
    prev = c.records[start - 1]["hash"] if start else HashChainLog.GENESIS
    for r in c.records[start:]:
        r["prev"] = prev; r["hash"] = HashChainLog.digest(prev, r["payload"]); prev = r["hash"]
rehash(chain, 500)
print("internal verification after recomputing the links:", chain.verify())
print("anchored head still matches:", chain.head == head, "-> tampering", "NOT detected" if chain.head == head else "detected")'''),
 ("code", '''import subprocess
subprocess.run([sys.executable, "audit_threat_model.py"], capture_output=True)
pd.DataFrame(json.load(open("results/audit_threat_model.json"))).T'''),
])

# ---------------------------------------------------------------- 08
nb("08_dos_ddos_windowed_aggregation.ipynb", [
 ("md", """# 08 DoS versus DDoS: windowed aggregation

**Purpose.** About 97% of the multiclass error is one confusion between DoS and DDoS. This notebook tests whether temporal aggregation over neighbouring flows recovers the distinction, and which features carry it.

**Data layout.** Each CICIoMT2024 CSV is a concatenation of capture blocks, one per label, in capture order. A window statistic uses the previous 50 and 200 flows of the same capture only (causal).
**Protocols.** `chrono`: first 85% of every capture trains, last 15% tests. `heldout`: whole captures are held out (13 of 50, three repeats), the stricter test.

**Findings stored in `results/`.** On the 25 selected per-flow features the DoS/DDoS accuracy is 0.78 (chrono) / 0.59 (held-out); with windows it is 0.96 / 0.82. All 45 released features reach 99.9%, because inter-arrival time (IAT), a feature the selection discarded, is nearly constant with a class-specific offset of about 0.03%, which looks like a recording artefact; it is not used as a detection feature. Removing IAT leaves the windowed result unchanged.
The test uses a Random Forest, not the hybrid model, and each capture window contains a single attack class.

**To rerun** set `CICIOMT_DIR` to the folder with `CIC_IoMT_2024_WiFi_MQTT_train.csv` and `..._test.csv` (not redistributed; see the dataset page)."""),
 ("code", SETUP),
 ("code", '''CICIOMT_DIR = None     # e.g. r"D:\\data\\CIC IoMT 2024 WiFi MQTT"
import subprocess
if CICIOMT_DIR:
    for feats in ["all", "kept25", "noIAT"]:
        subprocess.run([sys.executable, "dos_ddos_window_test.py", CICIOMT_DIR], env={**os.environ, "FEATURES": feats})
    subprocess.run([sys.executable, "dos_ddos_feature_diagnosis.py", CICIOMT_DIR])'''),
 ("code", '''rows = []
for name in ["all", "kept25", "noIAT"]:
    r = json.load(open(f"results/dos_ddos_window_test_{name}.json"))
    rows.append({"features": {"all": "all 45", "kept25": "25 selected", "noIAT": "44 (all but IAT)"}[name],
                 "chrono per-flow": r["chrono"]["per_flow"]["acc"], "chrono + windows": r["chrono"]["per_flow+window"]["acc"],
                 "held-out per-flow": r["heldout_mean_sd_per_flow"][0], "held-out + windows": r["heldout_mean_sd_per_flow+window"][0],
                 "held-out SD (windows)": r["heldout_mean_sd_per_flow+window"][1]})
pd.DataFrame(rows)'''),
 ("code", '''D = json.load(open("results/dos_ddos_feature_diagnosis.json"))
print("dropped features:", D["dropped_features"])
print({k: round(D[k]["mean"], 4) for k in ["all_45", "kept_25", "dropped_20"]})
pd.Series(D["single_feature_depth3_acc"]).head(8).to_frame("single-feature depth-3 tree accuracy")'''),
])

# ---------------------------------------------------------------- 09
nb("09_figures_and_tables.ipynb", [
 ("md", """# 09 Figures and tables

**Purpose.** Regenerate the ROC, precision-recall, cross-validation and sample-efficiency figures from stored results, and map every result file to the table or figure of the paper it feeds.

| File in `results/` | Feeds |
|---|---|
| `ci_effect_sizes.json` | bootstrap intervals, paired differences, Hedges' g tables |
| `twin_multiclass_seed*.json`, `twin_multiclass.json`, `pqc_epoch_time.json` | classical-twin comparison table |
| `twin_robustness_allseeds.json` | twin robustness table |
| `audit_threat_model.json` | adversary-capability table |
| `dos_ddos_window_test_*.json`, `dos_ddos_feature_diagnosis.json` | DoS/DDoS windowed-aggregation table |

The multi-seed tables of the quantum model come from the cluster notebooks (`../../grid5000/notebooks/`)."""),
 ("code", SETUP),
 ("code", '''import subprocess
print(subprocess.run([sys.executable, "make_figures.py"], capture_output=True, text=True).stdout)
from IPython.display import Image, display
display(Image(filename="figures/sample_efficiency.png", width=700))
sorted(p.name for p in Path("results").glob("*.json"))'''),
])
