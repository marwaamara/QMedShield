"""
nb_defs.py -- cell definitions for every notebook.

One notebook per task. Each is headed by the question it answers, keeps
the original paper's idea intact, and saves every value it computes so that no
table, curve or statistic ever requires re-running the code.

`build_notebooks.py` prepends the inlined src/ bootstrap and writes the .ipynb.
"""
from __future__ import annotations


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


NOTEBOOKS = {}


def register(filename, title, remark, purpose, cells, needs="archive"):
    NOTEBOOKS[filename] = {"title": title, "remark": remark, "purpose": purpose,
                           "cells": cells, "needs": needs}


# ===========================================================================
# NB00 -- archive verification
# ===========================================================================
register(
    "00_verify_archive.ipynb",
    "00 - Verify the archived Grid5000 artefacts",
    "*None - this is the regression guard for every other notebook.*",
    """
Reloads the trained models from the archived Grid5000 run and re-computes the published
metrics. If predictions, accuracy or F1 deviate at all, the environment has drifted and
**no downstream result should be trusted**.

This also establishes that every number in this suite is real rather than synthetic: the
same weights, the same split and the same predictions that produced the paper.
""",
    [
        code("""
data = load_prepared_data(str(ARCHIVE_DIR))
pre  = load_preprocessors(str(ARCHIVE_DIR))
models = {}
for task, fn in (("binary", "hybrid_qcnn_binary (1).pt"),
                 ("multiclass", "hybrid_qcnn_multiclass (1).pt")):
    m, info = load_hybrid(task, str(ARCHIVE_DIR / fn), model=build_model(task, "full"))
    assert not info["missing"] and not info["unexpected"], info
    models[task] = m

print("archive loaded")
print("  splits       :", data['X_train_m'].shape[0], "/", data['X_val_m'].shape[0],
      "/", data['X_test_m'].shape[0])
print("  features     :", data['X_scaled'].shape[1], "raw ->", data['X_pca_m'].shape[1], "PCA")
print("  preprocessors:", sorted(pre))
print("  SMOTE applied:", not np.array_equal(data['X_train_m_smote'], data['X_train_m']),
      "(section 5.2 of the paper states SMOTE was NOT used)")
"""),
        code("""
# Parameter budget vs Table 5 of the paper
PAPER_PARAMS = {"binary": 1178, "multiclass": 6768}
for task in ("binary", "multiclass"):
    pb = models[task].param_breakdown()
    ok = "MATCH" if pb["total"] == PAPER_PARAMS[task] else "MISMATCH"
    print(f"{task:11s} {pb}  {models[task].footprint_kb():5.1f} KB  "
          f"paper={PAPER_PARAMS[task]}  [{ok}]")
"""),
        code("""
PUBLISHED = {"binary":     {"accuracy": 0.9962666666666666, "f1": 0.9980777152272415,
                            "auc": 0.9992968453310266},
             "multiclass": {"accuracy": 0.7918666666666667, "f1": 0.7708048050063878,
                            "auc": 0.8365806111722032}}
# Accuracy, F1 and the predictions are argmax-determined and must match EXACTLY.
# AUC is a rank statistic over float32 probabilities and additionally depends on
# the scikit-learn version's tie handling, so it carries a small tolerance.
TOL = {"accuracy": 0.0, "f1": 0.0, "auc": 1e-3}

model_ok, notes, results = True, [], {}
for task in ("binary", "multiclass"):
    sfx = "b" if task == "binary" else "m"
    r = evaluate(models[task],
                 make_loader(data[f"X_test_{sfx}"], data[f"y_test_{sfx}"], shuffle=False),
                 2 if task == "binary" else N_CLASSES)
    results[task] = r
    print(f"\\n{task}:")
    for k, ref in PUBLISHED[task].items():
        d = abs(r[k] - ref); ok = d <= TOL[k]
        if not ok and TOL[k] == 0.0: model_ok = False
        elif not ok: notes.append(f"{task}/{k} delta={d:.2e}")
        print(f"    {k:9s} archive={ref:.10f} recomputed={r[k]:.10f} delta={d:.2e} "
              f"[{'OK' if ok else 'CHECK'}]")
    arch = np.load(ARCHIVE_DIR / f"{task}_predictions.npz")
    agree = (r['_predictions'] == arch['predictions']).mean()
    model_ok &= (agree == 1.0)
    print(f"    per-sample agreement: {agree*100:.4f}%  "
          f"[{'OK' if agree == 1.0 else 'MODEL DRIFT'}]")

print("\\n" + ("VERIFIED - predictions, accuracy and F1 reproduce the paper exactly."
               if model_ok else "*** MODEL DRIFT - stop and investigate ***"))
if notes: print("  Note (not model drift): " + "; ".join(notes))
"""),
        code("""
for task in ("binary", "multiclass"):
    r = results[task]
    save_run(OUT_DIR, experiment="nb00_verify", variant="archive", task=task,
             split="random", seed=42,
             y_true=r['_labels'], y_pred=r['_predictions'], probs=r['_probabilities'],
             metrics={k: v for k, v in r.items()
                      if not k.startswith('_') and np.isscalar(v)},
             model=models[task], preprocessors=pre,
             params=models[task].param_breakdown(),
             class_names=CLASS_NAMES if task == "multiclass" else ["Benign", "Attack"],
             extra={"notebook": "NB00"})

check_completeness(OUT_DIR, "nb00_verify", required=("metrics", "predictions", "model"))
"""),
    ])

# ===========================================================================
# NB01 -- data, splits, qualitative curves
# ===========================================================================
register(
    "01_data_preparation.ipynb",
    "01 - Data characterisation, enlarged test set and split protocols",
    """**Question:** does the model generalise under more demanding evaluation protocols
(session-level, device-level, chronological splits) than a random train-test partition, which
may let correlated flows from the same capture session appear in both training and testing
subsets?""",
    """
Keeps the paper's data pipeline exactly (50,000-sample stratified pool, 25 features ->
16 PCA components) and adds two things:

1. **An enlarged, disjoint test set** so every class has n >= 100. The paper's per-class
   table reports F1 on n=4 (MQTT_Malformed) and n=15 (ARP_Spoofing), which is not defensible.
2. **Session / device / chronological split protocols**, built from per-flow provenance.

Also produces the qualitative figures: class distribution, PCA variance, t-SNE and the
feature-correlation structure - each saved as values, not only as PNGs.

> **Requires the raw CICIoMT2024 WiFi/MQTT CSVs** for provenance. Without them the notebook
> still runs the characterisation on the archived arrays and reports the split protocols as
> unavailable, stating why.
""",
    [
        code("""
data = load_prepared_data(str(ARCHIVE_DIR))
pre  = load_preprocessors(str(ARCHIVE_DIR))
X_raw, feat_names = recover_raw_features(str(ARCHIVE_DIR))
print("recovered original 25-feature matrix:", X_raw.shape)
print("features:", feat_names)
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
"""),
        code("""
# --- qualitative: class distribution across splits -------------------------
c = class_distribution({"train": data['y_train_m'], "val": data['y_val_m'],
                        "test": data['y_test_m']},
                       CLASS_NAMES, path=str(FIG / "class_distribution.png"))
save_curve(OUT_DIR, "nb01_data", "class_distribution", **c)

for i, n in enumerate(CLASS_NAMES):
    n_test = int((data['y_test_m'] == i).sum())
    flag = "  <-- too thin for a per-class F1" if n_test < 30 else ""
    print(f"  {n:16s} test n={n_test:5d}{flag}")
"""),
        code("""
# --- qualitative: PCA variance, correlation, t-SNE -------------------------
v = pca_variance(pre['pca_multi'].explained_variance_ratio_, n_keep=N_PCA,
                 path=str(FIG / "pca_variance.png"))
save_curve(OUT_DIR, "nb01_data", "pca_variance", **v)
print(f"cumulative explained variance at {N_PCA} components: "
      f"{v['cumulative'][N_PCA-1]*100:.2f}%  (paper reports 96.3%)")

h = correlation_heatmap(data['X_scaled'], feat_names,
                        path=str(FIG / "feature_correlation.png"))
save_curve(OUT_DIR, "nb01_data", "feature_correlation", **h)

t = tsne_projection(data['X_pca_m'], data['y_multiclass'], CLASS_NAMES,
                    path=str(FIG / "tsne.png"))
save_curve(OUT_DIR, "nb01_data", "tsne", **t)
"""),
        code("""
# --- enlarged test set + split protocols (need the raw CSVs) --------------
csvs = sorted(DATA_DIR.glob("*.csv"))
if not csvs:
    print("NO RAW CSVs in", DATA_DIR)
    print("  -> the enlarged test set and the session/device/chronological")
    print("     split protocols CANNOT be built: the archive holds only")
    print("     post-PCA arrays and no per-flow provenance (source capture file,")
    print("     packet order, device identity).")
    print("  -> download the CICIoMT2024 WiFi/MQTT partition into", DATA_DIR)
    print("     https://www.unb.ca/cic/datasets/iomt-dataset-2024.html")
    write_json(OUT_DIR / "nb01_data" / "splits_status.json",
               {"available": False,
                "blocked": ["enlarged_test_set", "session_split",
                            "device_split", "chronological_split"],
                "reason": "raw CICIoMT2024 CSVs absent; archive has no flow provenance"})
else:
    print("found", len(csvs), "CSV files -- building provenance and split protocols")
    # Provenance: source file = capture-session proxy, row order = time proxy.
    import pandas as pd
    frames = []
    for p in csvs:
        df = pd.read_csv(p)
        df["__source_csv"] = p.name
        df["__row_order"] = np.arange(len(df))
        frames.append(df)
    allflows = pd.concat(frames, ignore_index=True)
    print("  total flows:", len(allflows))
    allflows[["__source_csv", "__row_order"]].to_parquet(
        DATA_DIR / "flow_provenance.parquet")
    write_json(OUT_DIR / "nb01_data" / "splits_status.json",
               {"available": True, "n_flows": int(len(allflows)),
                "n_sources": int(allflows["__source_csv"].nunique())})
"""),
        code("""
save_run(OUT_DIR, experiment="nb01_data", variant="characterisation",
         task="multiclass", split="random", seed=42,
         metrics={"pca_cumvar_at_16": float(v['cumulative'][N_PCA-1]),
                  "n_test_min_class": int(np.bincount(data['y_test_m'],
                                                      minlength=N_CLASSES).min())},
         extra={"notebook": "NB01", "topics": "split protocols, enlarged test set",
                "feature_names": feat_names}, save_checkpoint=False)
check_completeness(OUT_DIR, "nb01_data", required=("metrics",))
"""),
    ])

# ===========================================================================
# NB02 / NB03 -- headline training with full curves
# ===========================================================================
for _task, _nb, _ep in (("binary", "02_qcnn_binary.ipynb", "EPOCHS_BINARY"),
                        ("multiclass", "03_qcnn_multiclass.ipynb", "EPOCHS_MULTICLASS")):
    register(
        _nb,
        f"0{'2' if _task=='binary' else '3'} - Hybrid QCNN, {_task} detection",
        """**Question:** how stable are the results across multiple random seeds (mean, standard
deviation and confidence intervals), rather than a single run?""",
        f"""
The paper's architecture is unchanged: 8-qubit, 4-layer circuit with angle-encoded data
re-uploading at every layer and the three-scale entangler (ring CNOT, skip-one CZ, cross-half
CNOT), embedded as a 96-parameter bottleneck between a classical encoder and decoder.

What is added is **{len('x')}multi-seed training** with mean +/- std and confidence intervals, and the
complete set of curves: learning curves with a seed band, ROC, precision-recall, calibration,
confusion matrix and per-class bars. Every array behind every figure is saved.

Checkpoint selection is on the **validation** split only. Test data is touched once, after
training.

> Training runs; use `QUICK_MODE=1` for a fast pass, or run on Grid5000 for the full
> {_task} schedule.
""",
        [
            code(f"""
TASK = "{_task}"
data = load_prepared_data(str(ARCHIVE_DIR))
sfx = "b" if TASK == "binary" else "m"
X_tr, y_tr = subsample_for_mode(data[f"X_train_{{sfx}}"], data[f"y_train_{{sfx}}"])
X_va, y_va = data[f"X_val_{{sfx}}"],   data[f"y_val_{{sfx}}"]
X_te, y_te = data[f"X_test_{{sfx}}"],  data[f"y_test_{{sfx}}"]
NC = 2 if TASK == "binary" else N_CLASSES
NAMES = ["Benign", "Attack"] if TASK == "binary" else CLASS_NAMES
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
print(f"{{TASK}}: train {{X_tr.shape}}  val {{X_va.shape}}  test {{X_te.shape}}  "
      f"epochs={{epochs_for(TASK)}}")
"""),
            code("""
# --- multi-seed training --------------------------------------
histories, evals = {}, {}
for seed in SEEDS:
    if is_done(OUT_DIR, f"nb0{'2' if TASK=='binary' else '3'}_{TASK}", "qcnn",
               TASK, "random", seed):
        print(f"seed {seed}: already done, skipping")
        continue
    print(f"\\n=== seed {seed} ===")
    set_global_seed(seed)
    model = build_model(TASK, "full")
    t0 = time.time()
    hist = train_model(model, make_loader(X_tr, y_tr),
                       make_loader(X_va, y_va, shuffle=False),
                       task=TASK, class_weights=compute_class_weights(y_tr, NC))
    r = evaluate(model, make_loader(X_te, y_te, shuffle=False), NC)
    histories[seed], evals[seed] = hist, r
    print(f"  test acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f}")

    save_run(OUT_DIR, experiment=f"nb0{'2' if TASK=='binary' else '3'}_{TASK}",
             variant="qcnn", task=TASK, split="random", seed=seed,
             y_true=r['_labels'], y_pred=r['_predictions'], probs=r['_probabilities'],
             metrics={k: v for k, v in r.items()
                      if not k.startswith('_') and np.isscalar(v)},
             history=hist, model=model, params=model.param_breakdown(),
             timings={"train_wall_s": time.time() - t0},
             class_names=NAMES, extra={"notebook": "NB02/NB03", "topic": "multi-seed statistics"})
"""),
            code("""
# --- aggregate across seeds -----------------------------------------------
EXP = f"nb0{'2' if TASK=='binary' else '3'}_{TASK}"
runs = load_all_runs(OUT_DIR, EXP)
print(f"{len(runs)} seed runs found\\n")
for metric in ("accuracy", "precision", "recall", "f1", "auc"):
    for row in summarise_runs(runs, metric, group_keys=("variant", "task")):
        print(f"  {metric:10s} {format_mean_std(row)}  "
              f"95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}]  n={row['n']}")
"""),
            code("""
# --- learning curves with the seed band (all values saved) ----------------
if histories:
    hs = {"QCNN": list(histories.values())}
    c = training_curves(hs, path=str(FIG / f"{TASK}_learning_curves.png"),
                        metrics=("train_loss", "train_acc", "val_f1", "val_auc"))
    save_curve(OUT_DIR, EXP, "learning_curves", **c)
else:
    print("no new histories this run (all seeds already complete);"
          " curves are in outputs/*/runs/*/history.{csv,npz}")
"""),
            code("""
# --- quantitative curves from the seed-42 run -----------------------------
p = load_predictions(OUT_DIR, EXP, "qcnn", TASK, "random", SEEDS[0])
y_true, y_pred, probs = p["y_true"], p["y_pred"], p["probs"]

save_curve(OUT_DIR, EXP, "roc",
           **roc_curves(y_true, probs, NAMES, path=str(FIG / f"{TASK}_roc.png")))
save_curve(OUT_DIR, EXP, "pr",
           **pr_curves(y_true, probs, NAMES, path=str(FIG / f"{TASK}_pr.png")))
save_curve(OUT_DIR, EXP, "calibration",
           **calibration_curve(y_true, probs, path=str(FIG / f"{TASK}_calibration.png")))

rep = per_class_report(y_true, y_pred, NAMES)
save_curve(OUT_DIR, EXP, "per_class", **per_class_bars(rep,
           path=str(FIG / f"{TASK}_per_class.png")))
save_curve(OUT_DIR, EXP, "confusion", **confusion_matrix_plot(
    rep["confusion_matrix"], NAMES, path=str(FIG / f"{TASK}_confusion.png")))
"""),
            code("""
check_completeness(OUT_DIR, EXP)
print("\\nEvery curve above is saved as CSV+NPZ; new figures need no re-training.")
"""),
        ], needs="training")

# ===========================================================================
# NB04 -- baselines, matched controls, McNemar, Pareto
# ===========================================================================
register(
    "04_baselines_multiseed.ipynb",
    "04 - Classical baselines, parameter-matched controls and paired tests",
    """**Question:** is the quantum approach preferable to highly optimised lightweight classical
models, and is the observed behaviour attributable to the quantum component rather than to
differences in network architecture (comparison against parameter-matched compact classical
neural networks)?""",
    """
Three families, one notebook:

**A. Paper baselines** (RF, GB, MLP, KNN, SVM) refitted from the archived `best_params` -
minutes, no cluster - now saving **per-sample predictions**, which is what makes the paired
McNemar tests possible.

**B. Parameter-matched controls**: the identical encoder/decoder scaffolding with the PQC
swapped for a linear, Fourier, tiny-MLP or identity bottleneck. This is the experiment that
decides whether the quantum block earns its place.

**C. Lightweight deployables**: int8-quantised MLP, pruned MLP, 20-tree forest, shallow GBDT,
compared on accuracy **and** footprint **and** energy - drawn as a Pareto frontier.

---
### Measured outcome (seed 42, multiclass)

| model | family | accuracy | F1 | footprint |
|---|---|---|---|---|
| Random Forest | paper | 0.8275 | 0.8184 | **104,337 KB** |
| Gradient Boosting | paper | 0.8209 | 0.8086 | 5,038 KB |
| KNN | paper | 0.8101 | 0.8019 | 4,649 KB |
| rf20 | lightweight | 0.8009 | 0.7718 | 729 KB |
| gbdt_shallow | lightweight | 0.7968 | 0.7672 | 477 KB |
| MLP | paper | 0.7964 | 0.7678 | 354 KB |
| **QCNN (ours)** | hybrid | **0.7919** | **0.7708** | **26.4 KB** |
| pruned_mlp | lightweight | 0.7824 | 0.7415 | 49 KB |
| int8_mlp | lightweight | 0.7815 | 0.7409 | **1.2 KB** |
| SVM (RBF) | paper | 0.7716 | 0.7142 | 4,111 KB |

**Paired McNemar, Holm-corrected:** RF, GB and KNN are significantly better than the QCNN
(p_holm = 2.0e-16, 1.7e-12, 7.7e-05); the QCNN is significantly better than SVM
(p_holm = 1.8e-07); and **QCNN vs MLP is not significant (p_holm = 0.117)**.

That last result matters: section 5.2 of the paper describes the QCNN-MLP comparison as *"a tie
in operational ranking rather than as a tested statistical claim"*. **The paired test supports
that description.**

**Reading the footprint axis:**

* The QCNN is **on** the accuracy/footprint Pareto frontier, and is 3,952x smaller than Random
  Forest, which at 104 MB cannot plausibly be replicated across edge nodes. That is a stronger
  and more quantitative version of the paper's compactness argument.
* But it does **not dominate** the frontier: `int8_mlp` reaches 0.7815 at **1.2 KB**, 22x
  smaller for 1.0 pp less accuracy. Highly optimised lightweight classical models are
  therefore a fair challenge on the footprint axis.
* Where the QCNN does separate from the tiny classical models is **F1**: 0.7708 against 0.7409
  and 0.7415, i.e. ~3 pp better minority-class handling at comparable accuracy. That, not raw
  size, is the defensible claim.
""",
    [
        code("""
data = load_prepared_data(str(ARCHIVE_DIR))
TASK = "multiclass"; sfx = "m"; NC = N_CLASSES
X_tr, y_tr = data[f"X_train_{sfx}"], data[f"y_train_{sfx}"]
X_te, y_te = data[f"X_test_{sfx}"],  data[f"y_test_{sfx}"]
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

# Archived best_params: refit directly instead of re-running GridSearchCV.
# NOTE: the archive stores these as repr STRINGS, e.g. "{'max_depth': 20, ...}",
# not as JSON objects -- so they must be parsed before being splatted into the
# estimator constructor.
import ast
arch_base = json.loads((ARCHIVE_DIR / "baseline_results.json").read_text())


def _parse_params(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip():
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return {}
    return {}


BEST = {k: _parse_params(v.get("best_params", {}))
        for k, v in arch_base[TASK].items() if k != "QCNN (Ours)"}
for k, v in BEST.items():
    print(f"  {k:18s} {v}")
"""),
        code("""
# --- A. paper baselines, WITH per-sample predictions ------------------
rows = []
for name in BASELINES:
    # Already run? Skip. Re-fitting a saved baseline changes nothing and costs
    # minutes that a deadline does not have.
    if is_done(OUT_DIR, "nb04_baselines", name.replace(" ", "_"), TASK,
               "random", 42):
        print(f"{name:18s} already done, skipping")
        continue
    est, meta = fit_baseline(name, X_tr, y_tr, seed=42, best_params=BEST.get(name))
    r = evaluate_sklearn(est, X_te, y_te, NC)
    kb = model_footprint_kb(est); npar = count_sklearn_params(est)
    rows.append({"model": name, "family": "paper", "accuracy": r["accuracy"],
                 "f1": r["f1"], "auc": r["auc"], "footprint_kb": kb, "n_params": npar})
    print(f"{name:18s} acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
          f"auc={r['auc']:.4f}  {kb:8.1f} KB  params~{npar}")
    save_run(OUT_DIR, experiment="nb04_baselines", variant=name.replace(" ", "_"),
             task=TASK, split="random", seed=42,
             y_true=r['_labels'], y_pred=r['_predictions'], probs=r['_probabilities'],
             metrics={k: v for k, v in r.items()
                      if not k.startswith('_') and np.isscalar(v)},
             model=est, params={"total": npar, "footprint_kb": kb},
             timings=meta, class_names=CLASS_NAMES,
             extra={"notebook": "NB04", "family": "paper", **meta})
"""),
        code("""
# --- paired McNemar: QCNN vs each baseline -------------------
qcnn = load_predictions(OUT_DIR, "nb00_verify", "archive", TASK, "random", 42)
family = {}
for name in BASELINES:
    p = load_predictions(OUT_DIR, "nb04_baselines", name.replace(" ", "_"),
                         TASK, "random", 42)
    family[name] = p["y_pred"]

tests = compare_against_family(qcnn["y_true"], qcnn["y_pred"], family)
for t in tests:
    print(f"QCNN vs {t['baseline']:18s} b={t['b']:4d} c={t['c']:4d} "
          f"p={t['p_value']:.3e} p_holm={t['p_holm']:.3e} "
          f"{'SIGNIFICANT' if t['significant_holm'] else 'n.s.'}  ({t['test']})")
save_sweep(OUT_DIR, "nb04_baselines", "mcnemar_vs_qcnn", tests)
"""),
        code("""
# --- B. parameter-matched controls ---------------------------
# Identical scaffolding; ONLY the bottleneck changes. This isolates the
# quantum contribution from the surrounding architecture, and it is the
# experiment that decides whether the quantum block earns its place.
matched = {b: build_model(TASK, b if b != "pqc" else "full")
           for b in MATCHED_BOTTLENECKS}
assert_scaffolding_identical(matched)   # gate G4
print("scaffolding assertion passed - encoder/decoder identical across controls\\n")
for b, m in matched.items():
    print(f"  {b:10s} {m.param_breakdown()}")
"""),
        code("""
# --- B (continued): TRAIN the matched controls ----------------------------
# `pqc` is not retrained here: it IS the headline model, and NB02/NB03 already
# produced it over the full seed set. Retraining it would spend cluster time
# reproducing a number we already have, and risk reporting two values for one
# model.
X_tr_q, y_tr_q = subsample_for_mode(X_tr, y_tr)
for b in MATCHED_BOTTLENECKS:
    if b == "pqc":
        print("pqc: taking the headline runs from NB02/NB03, not retraining")
        continue
    for seed in SEEDS:
        if is_done(OUT_DIR, "nb04_baselines", b, TASK, "random", seed):
            print(f"{b} seed {seed}: already done, skipping"); continue
        set_global_seed(seed)
        model = build_model(TASK, b)
        t0 = time.time()
        hist = train_model(model, make_loader(X_tr_q, y_tr_q),
                           make_loader(data[f"X_val_{sfx}"],
                                       data[f"y_val_{sfx}"], shuffle=False),
                           task=TASK,
                           class_weights=compute_class_weights(y_tr_q, NC))
        r = evaluate(model, make_loader(X_te, y_te, shuffle=False), NC)
        pb = model.param_breakdown()
        print(f"{b:10s} seed {seed}  acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
              f"params={pb['total']} (bottleneck {pb['bottleneck']})")
        rows.append({"model": b, "family": "matched", "accuracy": r["accuracy"],
                     "f1": r["f1"], "auc": r["auc"], "seed": seed,
                     "footprint_kb": model.footprint_kb(),
                     "n_params": pb["total"]})
        save_run(OUT_DIR, experiment="nb04_baselines", variant=b, task=TASK,
                 split="random", seed=seed,
                 y_true=r['_labels'], y_pred=r['_predictions'],
                 probs=r['_probabilities'],
                 metrics={k: v for k, v in r.items()
                          if not k.startswith('_') and np.isscalar(v)},
                 history=hist, model=model, params=pb,
                 timings={"train_wall_s": time.time() - t0},
                 class_names=CLASS_NAMES,
                 extra={"notebook": "NB04", "topic": "parameter-matched controls",
                        "family": "matched", "bottleneck": b})
"""),
        code("""
# --- B (continued): does the quantum bottleneck earn its place? -----------
matched_runs = [r for r in load_all_runs(OUT_DIR, "nb04_baselines")
                if r["variant"] in MATCHED_BOTTLENECKS and r["task"] == TASK]
for metric in ("accuracy", "f1"):
    for row in summarise_runs(matched_runs, metric, group_keys=("variant",)):
        print(f"  {metric:8s} {row['variant']:10s} {format_mean_std(row)}  "
              f"n={row['n']}")

# Paired McNemar, QCNN against each control on the same test rows.
qcnn_p = load_predictions(OUT_DIR, "nb00_verify", "archive", TASK, "random", 42)
fam_m = {}
for b in MATCHED_BOTTLENECKS:
    if b == "pqc":
        continue
    try:
        fam_m[b] = load_predictions(OUT_DIR, "nb04_baselines", b, TASK,
                                    "random", SEEDS[0])["y_pred"]
    except Exception:
        print(f"  {b}: not trained yet, skipped")
if fam_m:
    tests_m = compare_against_family(qcnn_p["y_true"], qcnn_p["y_pred"], fam_m)
    for t in tests_m:
        print(f"QCNN vs {t['baseline']:10s} b={t['b']:4d} c={t['c']:4d} "
              f"p_holm={t['p_holm']:.3e} "
              f"{'SIGNIFICANT' if t['significant_holm'] else 'n.s.'}")
    save_sweep(OUT_DIR, "nb04_baselines", "mcnemar_matched", tests_m)
    print("\\nIf the PQC is not significantly better than `linear` or `mlp96`,")
    print("report that: it is a finding about")
    print("where the compactness comes from, not a failure.")
"""),
        code("""
# --- C. lightweight classical deployables ---------------------
for name in LIGHTWEIGHT_BASELINES:
    if is_done(OUT_DIR, "nb04_baselines", name, TASK, "random", 42):
        print(f"{name:18s} already done, skipping")
        continue
    est = make_lightweight(name, seed=42, n_classes=NC)
    est.fit(X_tr, y_tr)
    r = evaluate_sklearn(est, X_te, y_te, NC)
    kb, npar = model_footprint_kb(est), count_sklearn_params(est)
    if name == "int8_mlp":
        q = quantise_mlp_int8(est)
        r = evaluate_sklearn(q["model"], X_te, y_te, NC)
        kb, npar = q["footprint_kb_int8"], q["n_params"]
    rows.append({"model": name, "family": "lightweight", "accuracy": r["accuracy"],
                 "f1": r["f1"], "auc": r["auc"], "footprint_kb": kb, "n_params": npar})
    print(f"{name:18s} acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
          f"{kb:8.1f} KB  params={npar}")
    save_run(OUT_DIR, experiment="nb04_baselines", variant=name, task=TASK,
             split="random", seed=42,
             y_true=r['_labels'], y_pred=r['_predictions'], probs=r['_probabilities'],
             metrics={k: v for k, v in r.items()
                      if not k.startswith('_') and np.isscalar(v)},
             model=est, params={"total": npar, "footprint_kb": kb},
             class_names=CLASS_NAMES,
             extra={"notebook": "NB04", "family": "lightweight"})
"""),
        code("""
# --- Pareto: accuracy vs deployment footprint -----------------------------
# `rows` only holds what THIS run computed; anything skipped above is on disk.
# Rebuild the full comparison from the manifests so a resumed run still produces
# the complete table.
# Variant names are stored with underscores ("Random_Forest"); `rows` carries
# the display name ("Random Forest"). Normalise, or a resumed run lists the
# same model twice.
_have = {str(r["model"]).replace(" ", "_") for r in rows}
for r in load_all_runs(OUT_DIR, "nb04_baselines"):
    if r["task"] != TASK or r["variant"] in _have:
        continue
    m = r.get("metrics") or {}
    if "accuracy" not in m:
        continue
    fam = (r.get("context", {}).get("extra", {}) or {}).get("family", "saved")
    rows.append({"model": r["variant"], "family": fam,
                 "accuracy": m.get("accuracy"), "f1": m.get("f1"),
                 "auc": m.get("auc"),
                 "footprint_kb": (r.get("params") or {}).get("footprint_kb"),
                 "n_params": (r.get("params") or {}).get("total")})

qm = json.loads((OUT_DIR / "nb00_verify" / "runs" /
                 f"archive__{TASK}__random__seed42" / "metrics.json").read_text())
rows.append({"model": "QCNN (ours)", "family": "hybrid",
             "accuracy": qm["accuracy"], "f1": qm["f1"], "auc": qm["auc"],
             "footprint_kb": 26.4, "n_params": 6768})

save_sweep(OUT_DIR, "nb04_baselines", "model_comparison", rows)
save_curve(OUT_DIR, "nb04_baselines", "pareto",
           **pareto_scatter(rows, "footprint_kb", "accuracy", "model",
                            path=str(FIG / "pareto_accuracy_vs_footprint.png")))
check_completeness(OUT_DIR, "nb04_baselines",
                   required=("metrics", "predictions", "model"))
"""),
    ])

# ===========================================================================
# NB05 -- ablation, corrected
# ===========================================================================
register(
    "05_ablation.ipynb",
    "05 - Ablation study, rebuilt",
    """**Question:** what does each component contribute, when every ablation variant shares an
identical classical encoder and decoder and differs only in its bottleneck?""",
    """
Every variant is built by one factory with an identical encoder and decoder;
`assert_scaffolding_identical()` runs **before** training, so the classical scaffolding cannot
differ between variants. Selection is on validation only, and epochs come from the shared
config.

Three variants beyond the paper give the **per-scale entanglement attribution**: `ring_only` (l=1), `ring_skip` (l in {1,2}), `ring_cross`
(l in {1,4}) against the full l in {1,2,4}.
""",
    [
        code("""
data = load_prepared_data(str(ARCHIVE_DIR))
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

# GATE G4: run BEFORE any training. Seconds now, versus an entire wasted sweep.
for TASK in ("binary", "multiclass"):
    variants = {v: (SimpleMLP(TASK) if v == "simple_mlp" else build_model(TASK, v))
                for v in ABLATION_VARIANTS}
    assert_scaffolding_identical(variants)
    print(f"{TASK}: G4 PASSED across {len(variants)} variants")
    for v, m in variants.items():
        print(f"   {v:18s} {m.param_breakdown()}")
"""),
        code("""
# --- what this job will train --------------------------------------------
# The full grid is 8 variants x 2 tasks x 5 seeds = 80 training runs, and at
# the measured ~258 s/epoch that does not fit into one walltime. Take a slice:
#
#   SHARD=3/8 ./run_all.sh 05        one eighth of the grid, round-robin
#   VARIANTS=ring_only,ring_skip     only these variants
#   TASKS=binary                     only this task
#   SEEDS=42,1                       only these seeds
#   ABLATION_SAMPLES=10000           identical reduced budget for EVERY variant
#
# Nothing is lost by splitting: every cell writes its own manifest and is
# skipped on re-entry, so two shards never repeat each other's work and a
# walltime kill costs one cell rather than the stage.
CELLS = cells_for(ABLATION_VARIANTS, ("binary", "multiclass"), SEEDS_ABLATION)
TOTAL = len(ABLATION_VARIANTS) * 2 * len(SEEDS_ABLATION)
print(describe_selection(TOTAL, len(CELLS)))
todo = [c for c in CELLS
        if not is_done(OUT_DIR, "nb05_ablation", c[0], c[1], "random", c[2])]
print(f"{len(todo)} still to train, {len(CELLS) - len(todo)} already complete\\n")
if ABLATION_SAMPLES:
    print(f"TRAINING BUDGET: {ABLATION_SAMPLES:,} rows for every variant "
          f"(identical across variants, so the comparison stays fair).")
    print("STATE THIS IN THE TABLE 8 CAPTION: the ablation rows are comparable")
    print("with each other but NOT with Tables 6 and 9, trained on 35,000.\\n")
for v, t, sd in todo:
    print(f"   {t:11s} {v:18s} seed {sd}")
"""),
        code("""
# --- train the assigned cells ---------------------------------------------
# Per-task data prepared once, even when the shard interleaves the two tasks.
_cache = {}


def _splits(task):
    if task not in _cache:
        sfx = "b" if task == "binary" else "m"
        X_tr, y_tr = subsample_for_mode(data[f"X_train_{sfx}"],
                                        data[f"y_train_{sfx}"])
        X_tr, y_tr = training_budget(X_tr, y_tr)   # ABLATION_SAMPLES, or full
        _cache[task] = (X_tr, y_tr,
                        data[f"X_val_{sfx}"], data[f"y_val_{sfx}"],
                        data[f"X_test_{sfx}"], data[f"y_test_{sfx}"],
                        2 if task == "binary" else N_CLASSES)
    return _cache[task]


for variant, TASK, seed in todo:
    X_tr, y_tr, X_va, y_va, X_te, y_te, NC = _splits(TASK)
    set_global_seed(seed)
    model = SimpleMLP(TASK) if variant == "simple_mlp" else build_model(TASK, variant)
    t0 = time.time()
    print(f"\\n=== {TASK} {variant} seed {seed}  "
          f"({len(X_tr):,} rows, up to {epochs_for(TASK)} epochs) ===", flush=True)
    hist = train_model(model, make_loader(X_tr, y_tr),
                       make_loader(X_va, y_va, shuffle=False),
                       task=TASK, class_weights=compute_class_weights(y_tr, NC))
    r = evaluate(model, make_loader(X_te, y_te, shuffle=False), NC)
    print(f"{TASK:10s} {variant:18s} seed={seed:5d} "
          f"acc={r['accuracy']:.4f} auc={r['auc']:.4f} "
          f"({(time.time()-t0)/60:.1f} min)")
    save_run(OUT_DIR, experiment="nb05_ablation", variant=variant, task=TASK,
             split="random", seed=seed,
             y_true=r['_labels'], y_pred=r['_predictions'],
             probs=r['_probabilities'],
             metrics={k: v for k, v in r.items()
                      if not k.startswith('_') and np.isscalar(v)},
             history=hist, model=model, params=model.param_breakdown(),
             timings={"train_wall_s": time.time() - t0},
             class_names=CLASS_NAMES if TASK == "multiclass"
                         else ["Benign", "Attack"],
             extra={"notebook": "NB05", "topic": "ablation parameter counts",
                    "train_rows": int(len(X_tr)),
                    "ablation_samples": ABLATION_SAMPLES,
                    "shard": f"{SHARD_INDEX}/{SHARD_COUNT}"})
"""),
        code("""
# --- is the grid complete? ------------------------------------------------
# The aggregation below runs on whatever exists, so a partly-finished grid
# still yields a table. A table built from 30 of 80 cells must not reach the
# paper unnoticed, so state plainly what is missing.
missing = [(v, t, sd)
           for v in ABLATION_VARIANTS for t in ("binary", "multiclass")
           for sd in SEEDS_ABLATION
           if not is_done(OUT_DIR, "nb05_ablation", v, t, "random", sd)]
full = len(ABLATION_VARIANTS) * 2 * len(SEEDS_ABLATION)
print(f"grid status: {full - len(missing)}/{full} cells complete")
if missing:
    print(f"{len(missing)} still missing -- submit the remaining shards before")
    print("treating the table below as final. Missing cells:")
    for v, t, sd in missing[:20]:
        print(f"   {t:11s} {v:18s} seed {sd}")
    if len(missing) > 20:
        print(f"   ... and {len(missing) - 20} more")
else:
    print("grid COMPLETE -- the table below is final.")
"""),
        code("""# --- ablation table, mean +/- std over seeds ------------------------------
# Restricted to SEEDS_ABLATION so every row is averaged over the SAME seeds.
# Extra seeds may exist on disk from earlier runs; they stay there, but a table
# whose rows have different n is not a table anyone can read.
runs = [r for r in load_all_runs(OUT_DIR, "nb05_ablation")
        if r["seed"] in SEEDS_ABLATION]
_extra = len(load_all_runs(OUT_DIR, "nb05_ablation")) - len(runs)
print(f"reporting seeds {SEEDS_ABLATION}"
      + (f"  ({_extra} run(s) on other seeds kept on disk, excluded here)"
         if _extra else ""))
for TASK in ("binary", "multiclass"):
    print(f"\\n=== {TASK} ===")
    sel = [r for r in runs if r["task"] == TASK]
    for metric in ("accuracy", "auc"):
        print(f"  -- {metric}")
        for row in summarise_runs(sel, metric, group_keys=("variant",)):
            print(f"     {row['variant']:18s} {format_mean_std(row)}  n={row['n']}")
"""),
        code("""
# --- per-scale entanglement attribution (new; section 3.3 defers this) ----
sel = [r for r in runs if r["task"] == "multiclass"]
by = {row["variant"]: row for row in summarise_runs(sel, "auc", group_keys=("variant",))}
if "full" in by:
    base = by["full"]["mean"]
    print("multiclass AUC contribution relative to the full entangler:")
    for v in ("no_entangle", "ring_only", "ring_skip", "ring_cross", "no_reupload"):
        if v in by:
            print(f"  {v:14s} {by[v]['mean']:.4f}   delta = "
                  f"{(base - by[v]['mean'])*100:+.2f} pp")
    save_sweep(OUT_DIR, "nb05_ablation", "entanglement_attribution",
               [{"variant": v, **{k: by[v][k] for k in ('mean','std','n')}}
                for v in by])
"""),
        code("""
# --- curves ---------------------------------------------------------------
sel = [r for r in runs if r["task"] == "multiclass"]
vals = {}
for row in summarise_runs(sel, "auc", group_keys=("variant",)):
    vals[row["variant"]] = row["values"]
if vals:
    save_curve(OUT_DIR, "nb05_ablation", "ablation_auc_by_seed",
               **seed_boxplot(vals, "multiclass AUC",
                              path=str(FIG / "ablation_auc_boxplot.png")))
check_completeness(OUT_DIR, "nb05_ablation")
"""),
    ], needs="training")

# ===========================================================================
# NB06 -- qubit / depth sweep
# ===========================================================================
register(
    "06_qubit_depth_sweep.ipynb",
    "06 - Why 8 qubits?",
    """**Question:** why 8 qubits?""",
    """
The paper justifies 8 qubits only by noting it equals the encoder output width and the
class count. This notebook supplies the empirical argument: accuracy against **exponentially
growing simulation cost**.

Every configuration gets an **identical, reduced budget** (same epochs, same training
subsample), so the comparison is fair across configurations rather than confounded by letting
each run to its own convergence. Both facts must be stated when reporting the results.
""",
    [
        code("""
data = load_prepared_data(str(ARCHIVE_DIR))
X_tr, y_tr = subsample_for_mode(data["X_train_m"][:SWEEP_SAMPLES],
                                data["y_train_m"][:SWEEP_SAMPLES])
X_va, y_va = data["X_val_m"], data["y_val_m"]
X_te, y_te = data["X_test_m"], data["y_test_m"]
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

configs = ([(q, N_LAYERS) for q in SWEEP_QUBITS] +
           [(N_QUBITS, L) for L in SWEEP_LAYERS if L != N_LAYERS])
print("configs:", configs)
print(f"identical budget for every cell: {SWEEP_EPOCHS} epochs, "
      f"{SWEEP_SAMPLES} training samples")
"""),
        code("""
rows = []
# The 12-qubit cells cost ~16x an 8-qubit one, so this sweep is also shardable:
#   SHARD=2/4 ./run_all.sh 06
SWEEP_CELLS = shard_items([(nq, L, seed) for (nq, L) in configs
                           for seed in SEEDS_SWEEP])
print(describe_selection(len(configs) * len(SEEDS_SWEEP), len(SWEEP_CELLS)))
for (nq, L, seed) in SWEEP_CELLS:
        tag = f"q{nq}_L{L}"
        if is_done(OUT_DIR, "nb06_sweep", tag, "multiclass", "random", seed):
            continue
        set_global_seed(seed)
        model = HybridIDS("multiclass", n_features=N_PCA, n_qubits=nq,
                          n_layers=L, bottleneck="pqc")
        t0 = time.time()
        hist = train_model(model, make_loader(X_tr, y_tr),
                           make_loader(X_va, y_va, shuffle=False),
                           task="multiclass",
                           class_weights=compute_class_weights(y_tr, N_CLASSES),
                           epochs_override=SWEEP_EPOCHS)
        r = evaluate(model, make_loader(X_te, y_te, shuffle=False), N_CLASSES)
        wall = time.time() - t0
        cm = quantum_cost_model(nq, L, len(X_te))
        rows.append({"n_qubits": nq, "n_layers": L, "seed": seed,
                     "accuracy": r["accuracy"], "f1": r["f1"], "auc": r["auc"],
                     "n_params": model.param_breakdown()["total"],
                     "statevector_dim": cm["statevector_dim"],
                     "sim_ops_per_sample": cm["sim_cost_per_sample_ops"],
                     "train_wall_s": wall})
        print(f"  q={nq:2d} L={L} seed={seed:5d} acc={r['accuracy']:.4f} "
              f"auc={r['auc']:.4f} dim={cm['statevector_dim']:5d} {wall:6.1f}s")
        save_run(OUT_DIR, experiment="nb06_sweep", variant=tag, task="multiclass",
                 split="random", seed=seed,
                 y_true=r['_labels'], y_pred=r['_predictions'], probs=r['_probabilities'],
                 metrics={k: v for k, v in r.items()
                          if not k.startswith('_') and np.isscalar(v)},
                 history=hist, model=model, params=model.param_breakdown(),
                 timings={"train_wall_s": wall}, class_names=CLASS_NAMES,
                 extra={"notebook": "NB06", "topic": "qubit choice",
                        "epochs_budget": SWEEP_EPOCHS,
                        "train_rows": int(len(X_tr)), **cm})
"""),
        code("""
# --- rebuild the sweep table from the SAVED RUNS ---------------------------
# Not from `rows`: `rows` holds only what this session trained, and writing it
# over sweep_qubit_depth.json would destroy the cells earlier jobs completed --
# which is exactly what happened before. The manifests are the record; the
# sweep file is a view of them.
recs, budgets = [], {}
for m in load_all_runs(OUT_DIR, "nb06_sweep"):
    ex = (m.get("context", {}) or {}).get("extra", {}) or {}
    met = m.get("metrics") or {}
    nq, L = ex.get("n_qubits"), ex.get("n_layers")
    if nq is None or L is None:              # older manifests: parse the tag
        import re as _re
        g = _re.match(r"q(\d+)_L(\d+)", m.get("variant", ""))
        if not g:
            continue
        nq, L = int(g.group(1)), int(g.group(2))
    if met.get("accuracy") is None:
        print(f"  skipping {m.get('variant')} seed {m['seed']}: no accuracy in "
              f"its manifest (the run did not finish)")
        continue
    # Cost figures are RECOMPUTED from (n_qubits, n_layers) rather than read
    # from the manifest: quantum_cost_model is a pure function of those two, and
    # recomputing means runs saved by any earlier version of this notebook still
    # plot. Reading `extra` would leave older manifests with None and break the
    # curve on `NoneType + NoneType`.
    cmr = quantum_cost_model(int(nq), int(L), len(data["X_test_m"]))
    recs.append({"n_qubits": int(nq), "n_layers": int(L), "seed": m["seed"],
                 "accuracy": met.get("accuracy"), "f1": met.get("f1"),
                 "auc": met.get("auc"),
                 "n_params": (m.get("params") or {}).get("total"),
                 "statevector_dim": cmr["statevector_dim"],
                 "sim_ops_per_sample": cmr["sim_cost_per_sample_ops"],
                 "epochs_budget": ex.get("epochs_budget"),
                 "train_wall_s": (m.get("timings") or {}).get("train_wall_s")})
    # Runs saved before 2026-09-06 carry no epochs_budget: SWEEP_EPOCHS was
    # never passed to the trainer then, so they factually ran the full
    # multiclass schedule. Record them as that rather than as a third,
    # phantom budget -- otherwise every legacy run looks like a mismatch.
    budgets.setdefault(ex.get("epochs_budget") or EPOCHS_MULTICLASS, []).append(
        f"q{nq}_L{L}_seed{m['seed']}")

if recs:
    save_sweep(OUT_DIR, "nb06_sweep", "qubit_depth", recs)
print(f"{len(recs)} sweep cells on disk "
      f"(grid is {len(configs) * len(SEEDS_SWEEP)})")

# Every cell must share one budget, or the curve compares schedules, not
# architectures. Say so loudly rather than plotting a misleading line.
if len(budgets) > 1:
    print("\\n*** MIXED EPOCH BUDGETS -- the sweep is NOT comparable as it stands:")
    for b, cells in budgets.items():
        print(f"      {b} epochs: {len(cells)} cell(s)  e.g. {cells[:3]}")
    print("    Finish the sweep with one setting (SWEEP_EPOCHS=<n>), or delete")
    print("    outputs/nb06_sweep/runs and redo it at a single budget.\\n")

missing = [(nq, L, sd) for (nq, L) in configs for sd in SEEDS_SWEEP
           if not is_done(OUT_DIR, "nb06_sweep", f"q{nq}_L{L}",
                          "multiclass", "random", sd)]
if missing:
    print(f"{len(missing)} cell(s) still missing; curves below cover only what "
          f"has run:")
    for nq, L, sd in missing[:12]:
        print(f"   q={nq:2d} L={L} seed={sd}")
"""),
        code("""
# --- curves: each one plotted only if its slice has data -------------------
q_rows = [r for r in recs if r["n_layers"] == N_LAYERS]
L_rows = [r for r in recs if r["n_qubits"] == N_QUBITS]

if q_rows:
    save_curve(OUT_DIR, "nb06_sweep", "accuracy_vs_qubits",
               **metric_vs_sweep(q_rows, "n_qubits", "accuracy",
                                 path=str(FIG / "sweep_accuracy_vs_qubits.png"),
                                 xlabel="qubits", ylabel="multiclass accuracy"))
    save_curve(OUT_DIR, "nb06_sweep", "simcost_vs_qubits",
               **metric_vs_sweep(q_rows, "n_qubits", "sim_ops_per_sample",
                                 path=str(FIG / "sweep_simcost_vs_qubits.png"),
                                 xlabel="qubits", ylabel="simulator ops / sample"))
    print(f"accuracy-vs-qubits drawn from {len(q_rows)} cells at L={N_LAYERS}: "
          f"qubits {sorted({r['n_qubits'] for r in q_rows})}")
else:
    print(f"no completed cells at L={N_LAYERS} yet -- the accuracy-vs-qubits "
          f"figure (the one motivating the qubit choice) needs them. Run the shards "
          f"covering the q*_L{N_LAYERS} configurations.")

if L_rows:
    save_curve(OUT_DIR, "nb06_sweep", "accuracy_vs_layers",
               **metric_vs_sweep(L_rows, "n_layers", "accuracy",
                                 path=str(FIG / "sweep_accuracy_vs_layers.png"),
                                 xlabel="layers", ylabel="multiclass accuracy"))
    print(f"accuracy-vs-layers drawn from {len(L_rows)} cells at "
          f"n_q={N_QUBITS}: layers {sorted({r['n_layers'] for r in L_rows})}")
else:
    print(f"no completed cells at n_q={N_QUBITS} yet -- depth curve skipped.")

check_completeness(OUT_DIR, "nb06_sweep")
"""),
    ], needs="training")

# ===========================================================================
# NB07 -- robustness
# ===========================================================================
register(
    "07_robustness_adversarial.ipynb",
    "07 - Realistic and adversarial robustness",
    """**Question:** how robust is the detector under realistic network perturbations (packet
loss, timing jitter, missing features, protocol variation) and adversarial attacks, rather than
only Gaussian noise in PCA space? Healthcare IoMT systems are vulnerable to adversarial
manipulation.""",
    """
What matters is **where** the perturbation is applied. Every realistic
perturbation here operates on the **original 25-feature space, before scaling and PCA**; the
full pipeline is then re-applied. The archived `StandardScaler` is inverted to recover raw
feature units, so this runs today without the source CSVs.

The PCA-space Gaussian sweep of the paper is retained, clearly **labelled a sanity baseline**.

Adversarial: FGSM and PGD white-box, plus a **constrained** attack restricted to
attacker-controllable features (packet sizes, rates, timings, TCP flags) - the only variant
that is operationally meaningful for a network IDS.
""",
    [
        code("""
data = load_prepared_data(str(ARCHIVE_DIR))
pre  = load_preprocessors(str(ARCHIVE_DIR))
X_raw, feat_names = recover_raw_features(str(ARCHIVE_DIR))
model, _ = load_hybrid("multiclass", str(ARCHIVE_DIR / "hybrid_qcnn_multiclass (1).pt"),
                       model=build_model("multiclass", "full"))
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

# Locate the test rows inside the 50k pool by exact PCA match, so the raw-space
# perturbations act on exactly the rows the paper evaluates.
from scipy.spatial import cKDTree
d, idx_test = cKDTree(data["X_pca_m"]).query(data["X_test_m"], k=1)
assert (d < 1e-9).all(), "test rows could not be located in the pool"
X_raw_test, y_test = X_raw[idx_test], data["y_test_m"]

scaler, pca = pre["scaler"], pre["pca_multi"]
predict = pipeline_predict_proba(model, scaler, pca)
base_pred = predict(X_raw_test).argmax(1)
arch = np.load(ARCHIVE_DIR / "multiclass_predictions.npz")
print(f"round-trip agreement with archived predictions: "
      f"{(base_pred == arch['predictions']).mean()*100:.4f}%")
print("feature groups:", {k: len(v) for k, v in column_groups(feat_names).items()})
"""),
        code("""
# --- realistic perturbations, RAW feature space ---------------------------
from sklearn.metrics import accuracy_score, f1_score
rng = np.random.default_rng(42)
rows = []
for kind in PERTURBATIONS:
    for sev in PERTURB_SEVERITIES:
        Xp = apply_perturbation(X_raw_test, kind, sev, feat_names, rng)
        pred = predict(Xp).argmax(1)
        rows.append({"space": "raw", "perturbation": kind, "severity": sev,
                     "accuracy": accuracy_score(y_test, pred),
                     "f1_weighted": f1_score(y_test, pred, average="weighted",
                                             zero_division=0)})
        print(f"{kind:14s} sev={sev:<5} acc={rows[-1]['accuracy']:.4f} "
              f"f1={rows[-1]['f1_weighted']:.4f}")
save_sweep(OUT_DIR, "nb07_robustness", "realistic_raw_space", rows)
save_curve(OUT_DIR, "nb07_robustness", "realistic_curves",
           **metric_vs_sweep(rows, "severity", "f1_weighted", "perturbation",
                             path=str(FIG / "robustness_realistic.png"),
                             xlabel="severity", ylabel="weighted F1"))
"""),
        code("""
# --- PCA-space Gaussian: the sweep of the paper, kept as a sanity baseline ---
g_rows = []
for s in PCA_GAUSS_SIGMAS:
    Xp = apply_pca_gaussian(data["X_test_m"], s, rng)
    r = evaluate(model, make_loader(Xp, y_test, shuffle=False), N_CLASSES)
    g_rows.append({"space": "pca", "perturbation": "gaussian", "sigma": s,
                   "accuracy": r["accuracy"], "f1_weighted": r["f1"], "auc": r["auc"]})
    print(f"sigma={s:<5} f1={r['f1']:.4f}  (Table 11 of the paper)")
save_sweep(OUT_DIR, "nb07_robustness", "pca_gaussian_sanity_baseline", g_rows)
print("\\nLABEL THIS A SANITY BASELINE, not the robustness result.")
"""),
        code("""
# --- adversarial: FGSM / PGD in PCA space ---------------------------------
adv = []
for eps in ADV_EPSILONS:
    for name, steps in (("FGSM", 1), ("PGD-10", 10)):
        Xa = data["X_test_m"] if eps == 0 else pgd(model, data["X_test_m"], y_test,
                                                   eps, steps=steps)
        r = evaluate(model, make_loader(Xa, y_test, shuffle=False), N_CLASSES)
        adv.append({"attack": name, "epsilon": eps, "space": "pca",
                    "accuracy": r["accuracy"], "f1_weighted": r["f1"]})
        print(f"{name:7s} eps={eps:<5} acc={r['accuracy']:.4f} f1={r['f1']:.4f}")
save_sweep(OUT_DIR, "nb07_robustness", "adversarial_pca", adv)
save_curve(OUT_DIR, "nb07_robustness", "adversarial_curves",
           **metric_vs_sweep(adv, "epsilon", "accuracy", "attack",
                             path=str(FIG / "robustness_adversarial.png"),
                             xlabel="epsilon (L-inf, PCA space)", ylabel="accuracy"))
"""),
        code("""
# --- constrained adversarial: attacker-controllable features only ---------
# The only operationally meaningful variant: an attacker can change packet
# sizes, rates, timings and TCP flags, not features the defender derives.
con = []
SUB = 1000   # bounded: this is a gradient-free coordinate search
for eps_rel in (0.0, 0.02, 0.05, 0.10):
    Xc = (X_raw_test[:SUB] if eps_rel == 0 else
          constrained_attack(model, X_raw_test[:SUB], y_test[:SUB], feat_names,
                             scaler, pca, eps_rel, steps=5))
    pred = predict(Xc).argmax(1)
    con.append({"attack": "constrained", "eps_rel": eps_rel, "n": SUB,
                "accuracy": accuracy_score(y_test[:SUB], pred)})
    print(f"constrained eps_rel={eps_rel:<5} acc={con[-1]['accuracy']:.4f}")
save_sweep(OUT_DIR, "nb07_robustness", "adversarial_constrained", con)
check_completeness(OUT_DIR, "nb07_robustness", required=("metrics",))
"""),
    ])

# ===========================================================================
# NB08 -- shots and device noise
# ===========================================================================
register(
    "08_shots_and_noise.ipynb",
    "08 - Finite shots and device noise",
    """**Question:** how sensitive is the detector to finite shot budgets and device noise, given
that the work relies on ideal quantum simulation rather than execution on real quantum
hardware?""",
    """
Converts "we used an ideal simulator" from an unaddressed limitation into a quantified
sensitivity analysis, and produces the shot budget that the cost model in NB11 consumes.

Shot-based **training** is not attempted: it needs parameter-shift gradients, which fail under
PennyLane's parameter broadcasting and are prohibitively slow otherwise. Inference under finite
shots is the deployment-relevant question.

A note carried into the code: finite-shot sampling requires **float64**; with float32 inputs
the sampler raises `probabilities do not sum to 1`.
""",
    [
        code("""
data = load_prepared_data(str(ARCHIVE_DIR))
model, _ = load_hybrid("multiclass", str(ARCHIVE_DIR / "hybrid_qcnn_multiclass (1).pt"),
                       model=build_model("multiclass", "full"))
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
N_EVAL = 1000
Xs, ys = data["X_test_m"][:N_EVAL], data["y_test_m"][:N_EVAL]
print(f"evaluating {N_EVAL} test samples per shot setting")
"""),
        code("""
import torch, pennylane as qml
from sklearn.metrics import accuracy_score, f1_score

def eval_shots(shots, nq=N_QUBITS, L=N_LAYERS):
    dev = qml.device("default.qubit", wires=nq, shots=shots)
    qn = build_qnode(dev, nq, L, diff_method=None, use_reuploading=True,
                     entangle_scales=ENTANGLE_SCALES)
    # float64 is REQUIRED: float32 trips "probabilities do not sum to 1".
    with torch.no_grad():
        z = model.classical_encoder(torch.tensor(Xs, dtype=torch.float32)).double()
        q = qn(z, model.bottleneck.q_weights.double())
        q = torch.stack([torch.as_tensor(v) for v in q], dim=-1).float()
        return model.classical_decoder(q).argmax(1).numpy()

rows = []
for shots in SHOT_SWEEP:
    pred = eval_shots(shots)
    lab = "ideal" if shots is None else str(shots)
    rows.append({"shots": lab, "shots_num": shots or 0,
                 "accuracy": accuracy_score(ys, pred),
                 "f1_weighted": f1_score(ys, pred, average="weighted", zero_division=0)})
    print(f"shots={lab:>6}  acc={rows[-1]['accuracy']:.4f} f1={rows[-1]['f1_weighted']:.4f}")
save_sweep(OUT_DIR, "nb08_shots", "shot_sweep", rows)
save_curve(OUT_DIR, "nb08_shots", "accuracy_vs_shots",
           shots=[r["shots"] for r in rows],
           accuracy=[r["accuracy"] for r in rows],
           f1=[r["f1_weighted"] for r in rows])
"""),
        code("""
# --- device noise: depolarising and amplitude damping ---------------------
def eval_noise(p_depol=0.0, p_damp=0.0, N=100, nq=N_QUBITS, L=N_LAYERS):
    dev = qml.device("default.mixed", wires=nq)
    @qml.qnode(dev, interface="torch")
    def noisy(inputs, weights):
        for layer in range(L):
            for i in range(nq):
                qml.RY(inputs[..., i] * ANGLE_SCALE, wires=i)
            for i in range(nq):
                qml.Rot(weights[layer, i, 0], weights[layer, i, 1],
                        weights[layer, i, 2], wires=i)
            for i in range(nq):
                qml.CNOT(wires=[i, (i + 1) % nq])
            if layer % 2 == 0:
                for i in range(0, nq, 2):
                    qml.CZ(wires=[i, (i + 2) % nq])
            else:
                for i in range(nq // 2):
                    qml.CNOT(wires=[i, i + nq // 2])
            if p_depol:
                for i in range(nq): qml.DepolarizingChannel(p_depol, wires=i)
            if p_damp:
                for i in range(nq): qml.AmplitudeDamping(p_damp, wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(nq)]

    with torch.no_grad():
        z = model.classical_encoder(torch.tensor(Xs[:N], dtype=torch.float32)).double()
        w = model.bottleneck.q_weights.double()
        outs = [torch.stack([torch.as_tensor(v) for v in noisy(z[k], w)]).float()
                for k in range(N)]
        return model.classical_decoder(torch.stack(outs)).argmax(1).numpy(), ys[:N]

nrows = []
for p in NOISE_DEPOL:
    pred, yy = eval_noise(p_depol=p)
    nrows.append({"channel": "depolarizing", "p": p,
                  "accuracy": accuracy_score(yy, pred)})
    print(f"depolarizing p={p:<6} acc={nrows[-1]['accuracy']:.4f}")
for p in NOISE_AMP_DAMP:
    pred, yy = eval_noise(p_damp=p)
    nrows.append({"channel": "amplitude_damping", "p": p,
                  "accuracy": accuracy_score(yy, pred)})
    print(f"amp-damping  p={p:<6} acc={nrows[-1]['accuracy']:.4f}")
save_sweep(OUT_DIR, "nb08_shots", "device_noise", nrows)
save_curve(OUT_DIR, "nb08_shots", "accuracy_vs_noise",
           **metric_vs_sweep(nrows, "p", "accuracy", "channel",
                             path=str(FIG / "noise_channels.png"),
                             xlabel="channel probability", ylabel="accuracy"))
"""),
        code("""
# --- headline numbers for the paper ---------------------------------
ideal_acc = [r for r in rows if r["shots"] == "ideal"][0]["accuracy"]
noise_base = [r for r in nrows if r["channel"] == "depolarizing" and r["p"] == 0.0][0]
print(f"ideal (analytic) accuracy on {N_EVAL} samples : {ideal_acc:.4f}")
for r in rows:
    if r["shots"] != "ideal":
        print(f"  {r['shots']:>5} shots -> {r['accuracy']:.4f} "
              f"({(r['accuracy']-ideal_acc)*100:+.2f} pp vs ideal)")

print(f"\\nnoise study uses a {100}-sample subset, so its control is the p=0 row "
      f"({noise_base['accuracy']:.4f}), NOT the shot-study baseline above -- "
      f"the two tables have different sample counts and must not be compared directly.")
for r in nrows:
    if r["p"] > 0:
        print(f"  {r['channel']:18s} p={r['p']:<6} {r['accuracy']:.4f} "
              f"({(r['accuracy']-noise_base['accuracy'])*100:+.2f} pp vs p=0)")

save_run(OUT_DIR, experiment="nb08_shots", variant="shots_noise", task="multiclass",
         split="random", seed=42,
         metrics={"ideal_accuracy": ideal_acc,
                  "accuracy_1024_shots": [r for r in rows
                                          if r["shots"] == "1024"][0]["accuracy"],
                  "noise_control_accuracy": noise_base["accuracy"],
                  "n_eval_shots": N_EVAL, "n_eval_noise": 100},
         extra={"notebook": "08", "topic": "shots and noise",
                "shot_sweep": rows, "device_noise": nrows},
         save_checkpoint=False)
check_completeness(OUT_DIR, "nb08_shots", required=("metrics",))
"""),
    ])

# ===========================================================================
# NB09 -- explainability
# ===========================================================================
register(
    "09_shap_explainability.ipynb",
    "09 - SHAP and LIME: fidelity, stability and inverse-map validation",
    """**Question:** does combining SHAP and LIME improve on either method alone, and how stable
and faithful are the explanations, given that the transformation of attributions from PCA space
back to the original feature space is heuristic?""",
    """
Three questions, answered quantitatively:

1. **Does the combination beat either method alone?** Deletion/insertion curves,
   comprehensiveness and sufficiency, computed identically for SHAP-only, LIME-only and a
   rank-fused ranking.
2. **Are the explanations stable?** Repeat-run variance (LIME is stochastic, so this is the
   most plausible concrete benefit of reporting both) and a local Lipschitz estimate.
3. **Is the PCA inverse map trustworthy?** It is treated as a heuristic and validated against
   permutation importance computed directly in the original 25-feature space.

If the fused ranking is no better than the better single method, that is reported.

> Needs `shap` and `lime`: `pip install shap lime`
""",
    [
        code("""
try:
    import shap, lime
    XAI_OK = True
except ImportError as e:
    XAI_OK = False
    print("MISSING:", e, "\\n  pip install shap lime")

data = load_prepared_data(str(ARCHIVE_DIR))
pre  = load_preprocessors(str(ARCHIVE_DIR))
X_raw, feat_names = recover_raw_features(str(ARCHIVE_DIR))
model, _ = load_hybrid("multiclass", str(ARCHIVE_DIR / "hybrid_qcnn_multiclass (1).pt"),
                       model=build_model("multiclass", "full"))
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

cfg = XAI_CONFIG
X_expl = data["X_test_m"][:cfg["subset"]]
X_bg   = data["X_train_m"][:cfg["bg_size"]]
y_expl = data["y_test_m"][:cfg["subset"]]
pca_names = [f"PC{i+1}" for i in range(N_PCA)]
f_pca = torch_predict_proba(model)
print(f"explaining {len(X_expl)} samples in {N_PCA}-D PCA space")
"""),
        code("""
if XAI_OK:
    shap_vals = kernel_shap(f_pca, X_expl[:60], X_bg, nsamples=cfg["nsamples"])
    lime_vals = lime_attributions(f_pca, X_expl[:60], data["X_train_m"],
                                  pca_names, num_samples=cfg["lime_samples"] * 50)
    imp_shap, imp_lime = global_importance(shap_vals), global_importance(lime_vals)
    imp_fused = fuse_rankings(imp_shap, imp_lime)
    print("top-5 PCA components")
    print("  SHAP :", [pca_names[i] for i in np.argsort(-imp_shap)[:5]])
    print("  LIME :", [pca_names[i] for i in np.argsort(-imp_lime)[:5]])
    print("  fused:", [pca_names[i] for i in np.argsort(-imp_fused)[:5]])
    save_curve(OUT_DIR, "nb09_xai", "global_importance",
               pca_component=np.array(pca_names), shap=imp_shap,
               lime=imp_lime, fused=imp_fused)
    save_curve(OUT_DIR, "nb09_xai", "importance_bars",
               **feature_importance_bars({"SHAP": imp_shap, "LIME": imp_lime},
                                         pca_names,
                                         path=str(FIG / "xai_importance.png")))
    save_curve(OUT_DIR, "nb09_xai", "shap_beeswarm",
               **shap_beeswarm_data(shap_vals, X_expl[:60], pca_names,
                                    path=str(FIG / "xai_beeswarm.png")))
"""),
        code("""
# --- 1. FIDELITY: does the combination help? -----------------
if XAI_OK:
    fid = []
    for label, imp in (("SHAP", imp_shap), ("LIME", imp_lime), ("SHAP+LIME", imp_fused)):
        d = deletion_curve(f_pca, X_expl[:60], imp)
        i = insertion_curve(f_pca, X_expl[:60], imp)
        cs = comprehensiveness_sufficiency(f_pca, X_expl[:60], imp, k=5)
        fid.append({"method": label, "deletion_auc": float(d["auc"][0]),
                    "insertion_auc": float(i["auc"][0]), **cs})
        save_curve(OUT_DIR, "nb09_xai", f"deletion_{label.replace('+','_')}", **d)
        save_curve(OUT_DIR, "nb09_xai", f"insertion_{label.replace('+','_')}", **i)
        print(f"{label:10s} deletion-AUC={d['auc'][0]:.4f} (lower=better)  "
              f"insertion-AUC={i['auc'][0]:.4f} (higher=better)  "
              f"compr={cs['comprehensiveness']:.4f}")
    save_sweep(OUT_DIR, "nb09_xai", "fidelity", fid)
    best = min(fid, key=lambda r: r["deletion_auc"])["method"]
    print(f"\\nMost faithful by deletion-AUC: {best}")
    print("If SHAP+LIME does not win, report that -- either result answers the question.")
"""),
        code("""
# --- 2. STABILITY -------------------------------------------
if XAI_OK:
    st = []
    st.append({"method": "SHAP",
               **stability_repeat(lambda s: kernel_shap(f_pca, X_expl[:20], X_bg,
                                                        nsamples=50, seed=s),
                                  repeats=3)})
    st.append({"method": "LIME",
               **stability_repeat(lambda s: lime_attributions(
                   f_pca, X_expl[:20], data["X_train_m"], pca_names,
                   num_samples=300, seed=s), repeats=3)})
    for r in st:
        print(f"{r['method']:6s} pairwise Spearman={r['mean_pairwise_spearman']:.4f} "
              f"top5 Jaccard={r['mean_top5_jaccard']:.3f} CV={r['mean_cv']:.4f}")
    lip = local_lipschitz(lambda X: kernel_shap(f_pca, X, X_bg, nsamples=50),
                          X_expl[:20], eps=cfg["perturb_eps"])
    print("SHAP local Lipschitz:", {k: round(v, 4) for k, v in lip.items()})
    save_sweep(OUT_DIR, "nb09_xai", "stability", st)
    write_json(OUT_DIR / "nb09_xai" / "lipschitz.json", lip)
"""),
        code("""
# --- 3. INVERSE-MAP VALIDATION ------------------------------
if XAI_OK:
    mapped = pca_inverse_map(imp_shap, pre["pca_multi"])
    from scipy.spatial import cKDTree
    d, idx = cKDTree(data["X_pca_m"]).query(data["X_test_m"][:cfg['subset']], k=1)
    f_raw = pipeline_predict_proba(model, pre["scaler"], pre["pca_multi"])
    reference = permutation_importance_pipeline(f_raw, X_raw[idx][:200],
                                                y_expl[:200], n_repeats=3)
    v = validate_inverse_map(mapped, reference, feat_names, k=5)
    print("inverse-map validation")
    for k in ("spearman", "kendall_tau", "jaccard_top5", "verdict"):
        print(f"  {k:14s} {v[k]}")
    print("  top-5 (mapped)   :", v["top_mapped"])
    print("  top-5 (reference):", v["top_reference"])
    write_json(OUT_DIR / "nb09_xai" / "inverse_map_validation.json", v)
    save_curve(OUT_DIR, "nb09_xai", "original_feature_importance",
               **feature_importance_bars(
                   {"PCA inverse map (heuristic)": mapped,
                    "permutation (direct)": reference},
                   feat_names, path=str(FIG / "xai_original_features.png")))

    ag = agreement(imp_shap, imp_lime, k=5)
    print("\\nSHAP vs LIME agreement:", {k: round(x, 4) if isinstance(x, float) else x
                                        for k, x in ag.items()})
    write_json(OUT_DIR / "nb09_xai" / "shap_lime_agreement.json", ag)
"""),
        code("""
if XAI_OK:
    save_run(OUT_DIR, experiment="nb09_xai", variant="shap_lime", task="multiclass",
             split="random", seed=42,
             metrics={"best_fidelity_method": 0.0},
             extra={"notebook": "NB09", "topics": "SHAP+LIME benefit, inverse-map validation",
                    "fidelity": fid, "agreement": ag, "inverse_map": v},
             save_checkpoint=False)
    check_completeness(OUT_DIR, "nb09_xai", required=("metrics",))
"""),
    ])

# ===========================================================================
# NB10 -- audit log
# ===========================================================================
register(
    "10_audit_log.ipynb",
    "10 - Tamper-evident audit log: measured, not asserted",
    """**Question:** what is the experimentally measured added value of the SHA-256 hash-chained
audit log within the IDS framework?""",
    """
Section 6.3 **asserts** tamper evidence, chronological integrity, O(T) verification and
sub-millisecond overhead. Here each is **measured**, against two comparison points: a plain append-only log (what a conventional IDS gives you) and a Merkle
tree (O(log T) inclusion proofs).

Needs neither a model nor a dataset - run it while the cluster allocation is still pending.
""",
    [
        code("""
rng = random.Random(42)
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

overhead = [measure_overhead(T, rng) for T in AUDIT_CHAIN_LENGTHS]
for r in overhead:
    print(f"T={r['n_records']:>7,}  chain={r['us_per_event_chain']:7.2f} us/ev  "
          f"overhead={r['us_overhead_vs_plain']:6.2f} us  "
          f"verify={r['verify_total_s']:6.3f}s ({r['us_per_record_verify']:5.2f} us/rec)  "
          f"size={r['size_mb']:6.2f} MB")
save_sweep(OUT_DIR, "nb10_audit", "overhead", overhead)
save_curve(OUT_DIR, "nb10_audit", "verification_cost",
           n_records=[r["n_records"] for r in overhead],
           us_per_record=[r["us_per_record_verify"] for r in overhead],
           total_verify_s=[r["verify_total_s"] for r in overhead],
           throughput=[r["throughput_events_per_s"] for r in overhead])
print("\\nper-record verification cost is flat in T -> total verification is O(T)")
"""),
        code("""
detection = []
for T in AUDIT_CHAIN_LENGTHS:
    for r in measure_detection(T, AUDIT_TRIALS[T], AUDIT_ATTACKS, rng):
        detection.append(r)
        loc = ("n/a  " if r["localisation_rate_pct"] is None
               else format(r["localisation_rate_pct"], "6.1f") + "%")
        print(f"T={r['n_records']:>7,}  {r['attack']:9s} trials={r['trials']:3d}  "
              f"detection={r['detection_rate_pct']:6.1f}%  localisation={loc}  "
              f"(append-only: {r['append_only_detection_pct']:.1f}%)")
save_sweep(OUT_DIR, "nb10_audit", "tamper_detection", detection)
"""),
        code("""
storage = project_storage(overhead[-1]["mb_per_million_events"], AUDIT_FLOW_RATES)
for r in storage:
    print(f"{r['flows_per_s']:>6,} flows/s -> {r['events_per_day']/1e6:6.2f} M/day, "
          f"{r['gb_per_day']:7.2f} GB/day, {r['gb_per_year']:8.1f} GB/year")
save_sweep(OUT_DIR, "nb10_audit", "storage_projection", storage)

comparison = [
    {"mechanism": "append-only log", "detection_pct": 0.0,
     "append_us": overhead[-1]["us_per_event_plain"], "proof_cost": "n/a"},
    {"mechanism": "SHA-256 hash chain (ours)",
     "detection_pct": float(np.mean([r["detection_rate_pct"] for r in detection])),
     "append_us": overhead[-1]["us_per_event_chain"], "proof_cost": "O(T)"},
    {"mechanism": "Merkle tree", "detection_pct": 100.0,
     "append_us": overhead[-1]["us_per_event_merkle"],
     "proof_cost": f"O(log T) = {overhead[-1]['merkle_proof_length']} hashes"},
]
save_sweep(OUT_DIR, "nb10_audit", "mechanism_comparison", comparison)
for c in comparison:
    print(f"  {c['mechanism']:28s} detect={c['detection_pct']:5.1f}%  "
          f"append={c['append_us']:6.2f} us  proof={c['proof_cost']}")
write_json(OUT_DIR / "nb10_audit" / "threat_model.json", THREAT_MODEL)
"""),
        code("""
save_run(OUT_DIR, experiment="nb10_audit", variant="hash_chain", task="audit",
         split="none", seed=42,
         metrics={"us_overhead_per_event": overhead[-1]["us_overhead_vs_plain"],
                  "throughput_events_per_s": overhead[-1]["throughput_events_per_s"],
                  "mean_detection_rate_pct": float(
                      np.mean([r["detection_rate_pct"] for r in detection])),
                  "us_per_record_verify": overhead[-1]["us_per_record_verify"]},
         extra={"notebook": "NB10", "topic": "audit log", "threat_model": THREAT_MODEL},
         save_checkpoint=False)
check_completeness(OUT_DIR, "nb10_audit", required=("metrics",))
"""),
    ], needs="none")

# ===========================================================================
# NB11 -- cost and energy
# ===========================================================================
register(
    "11_grid5000_timing_energy.ipynb",
    "11 - Operational cost, latency and energy",
    """**Question:** what are the operational costs of the model (latency, energy, quantum circuit
evaluations, preprocessing overhead), beyond its parameter count?""",
    """
The key point: **parameter count is not deployment cost.**

Measured here: latency (median, p95, p99 - a real-time IDS is sized by its tail), throughput,
preprocessing overhead timed separately, classical MACs, quantum circuit-evaluation counts,
the parameter-shift training cost on hardware, a QPU projection with its assumptions stated,
and energy.

Energy uses the Grid5000 wattmeters when available (a real measurement, on an **exclusive**
node); otherwise it falls back to a CPU-time estimate that is **explicitly labelled an
estimate**, so that an estimate is never presented as a measurement.

---
### Note on latency reporting

Section 4.1 reports *"approximately 12 ms for the binary detector and 15 ms for the multiclass
detector"* as **single-threaded median per-sample inference time**. Measured here:

| batch size | binary ms/sample | multiclass ms/sample |
|---|---|---|
| **1** | **88.4** | **97.9** |
| 8 | 12.7 | 12.3 |
| 32 | 3.6 | 3.4 |
| 128 | 1.2 | 1.3 |

The published figure matches **batch size 8**, not batch size 1. The phrase "computed over the
full 7,500-sample test set" suggests the whole set was run and the total divided, which
amortises the per-call overhead.

That distinction matters because parameter count alone does not reflect deployment cost: an IoMT
gateway classifying flows **one at a time** sees ~90 ms, not 12 ms. Latency should therefore be
reported **as a function of batch size**, stating the batching assumption, rather than as a
single number.

Preprocessing overhead (scaler + PCA) is 0.0003 ms/sample - entirely negligible.
""",
    [
        code("""
data = load_prepared_data(str(ARCHIVE_DIR))
pre  = load_preprocessors(str(ARCHIVE_DIR))
X_raw, feat_names = recover_raw_features(str(ARCHIVE_DIR))
models = {t: load_hybrid(t, str(ARCHIVE_DIR / f"hybrid_qcnn_{t} (1).pt"),
                         model=build_model(t, "full"))[0]
          for t in ("binary", "multiclass")}
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
print("Kwollect wattmeter available:", kwollect_available())
if not kwollect_available():
    print("  -> energy figures will be CPU-time ESTIMATES and labelled as such")
"""),
        code("""
# --- latency and throughput ----------------------------------------------
lat_rows = []
for task, m in models.items():
    f = torch_predict_proba(m)
    X = data["X_test_b" if task == "binary" else "X_test_m"]
    for r in measure_latency(f, X, COST_BATCHES, repeats=LATENCY_REPEATS):
        r["model"] = f"QCNN {task}"
        lat_rows.append(r)
        print(f"QCNN {task:10s} bs={r['batch_size']:4d} "
              f"median={r['median_ms_per_sample']:7.3f} ms/sample  "
              f"p95={r['p95_ms_per_sample']:7.3f}  p99={r['p99_ms_per_sample']:7.3f}  "
              f"{r['throughput_samples_per_s']:9.1f}/s")
save_sweep(OUT_DIR, "nb11_cost", "latency", lat_rows)
save_curve(OUT_DIR, "nb11_cost", "latency_vs_batch",
           **metric_vs_sweep(lat_rows, "batch_size", "median_ms_per_sample", "model",
                             path=str(FIG / "latency_vs_batch.png"),
                             xlabel="batch size", ylabel="ms / sample"))
"""),
        code("""
# --- preprocessing overhead (excluded from the parameter count) ------------
pp = measure_preprocessing_overhead(pre["scaler"], pre["pca_multi"], X_raw[:5000])
print("preprocessing:", {k: round(v, 6) for k, v in pp.items()})
write_json(OUT_DIR / "nb11_cost" / "preprocessing.json", pp)

# --- compute accounting ---------------------------------------------------
cm = quantum_cost_model(N_QUBITS, N_LAYERS, n_samples=len(data["X_test_m"]),
                        shots=1024, epochs=EPOCHS_MULTICLASS,
                        batch_size=BATCH_SIZE)
print("\\nquantum cost model")
for k, v in cm.items():
    print(f"  {k:42s} {v}")
write_json(OUT_DIR / "nb11_cost" / "quantum_cost_model.json", cm)

fl = classical_flops(models["multiclass"])
print(f"\\nclassical MACs per forward pass: {fl['total_macs']:,}")
write_json(OUT_DIR / "nb11_cost" / "classical_flops.json", fl)
"""),
        code("""
# --- QPU projection: assumptions stated, nothing invented ----------------
qpu = qpu_cost_projection(circuit_evals=len(data["X_test_m"]), shots=1024,
                          seconds_per_shot=1e-4)
print("QPU projection for one pass over the test set")
for k, v in qpu.items():
    print(f"  {k:24s} {v}")
print("\\nSupply the provider's real pricing to turn this into euros;")
print("queue wait, calibration and transpilation are EXCLUDED and must be stated so.")
write_json(OUT_DIR / "nb11_cost" / "qpu_projection.json", qpu)
"""),
        code("""
# --- energy ---------------------------------------------------------------
erows = []
for task, m in models.items():
    f = torch_predict_proba(m)
    X = data["X_test_b" if task == "binary" else "X_test_m"]
    with EnergyMeter(f"inference_{task}") as em:
        for _ in range(5):
            f(X)
    e = em.result
    e["model"] = f"QCNN {task}"
    e["n_inferences"] = len(X) * 5
    e["joules_per_1k_inferences"] = e["energy_joules"] / (len(X) * 5) * 1000
    e.update(co2_equivalent(e["energy_wh"]))
    erows.append(e)
    print(f"QCNN {task:10s} {e['source']:20s} "
          f"{e['energy_joules']:9.2f} J  "
          f"{e['joules_per_1k_inferences']:7.3f} J/1k inferences  "
          f"{e['co2e_grams']:.4f} gCO2e")
save_sweep(OUT_DIR, "nb11_cost", "energy", erows)

save_run(OUT_DIR, experiment="nb11_cost", variant="qcnn", task="multiclass",
         split="random", seed=42,
         metrics={"median_ms_per_sample": lat_rows[0]["median_ms_per_sample"],
                  "total_macs": fl["total_macs"],
                  "preproc_ms_per_sample": pp["total_preproc_ms_per_sample"]},
         extra={"notebook": "NB11", "topics": "operational cost, energy, circuit evaluations",
                "quantum_cost_model": cm, "qpu": qpu},
         save_checkpoint=False)
check_completeness(OUT_DIR, "nb11_cost", required=("metrics",))
"""),
    ])

# ===========================================================================
# NB12 -- aggregation
# ===========================================================================
register(
    "13_results_aggregation.ipynb",
    "13 - Aggregate every result into paper tables and figures",
    "*None - this is the assembly step.*",
    """
Reads `outputs/` and emits LaTeX tables and figures into `outputs/paper/`. **No experiment is
ever re-run to produce a table.** If a new table, curve or statistic is needed later, it is
added here and regenerated from the saved artefacts.

Ends with a completeness audit across every experiment, and an experiment coverage map
showing which experiments have saved evidence and which are still outstanding.
""",
    [
        code("""
PAPER_DIR.mkdir(parents=True, exist_ok=True)
experiments = sorted(p.name for p in OUT_DIR.iterdir()
                     if p.is_dir() and p.name != "paper")
print("experiments present:", experiments)
"""),
        code("""
def latex_table(rows, cols, caption, label, dp=4):
    lines = [r"\\begin{table}[htbp]", r"\\centering",
             f"\\\\caption{{{caption}}}", f"\\\\label{{{label}}}", r"\\small",
             r"\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\\toprule",
             " & ".join(r"\\textbf{" + c.replace("_", " ") + "}" for c in cols) + r" \\\\",
             r"\\midrule"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.{dp}f}" if isinstance(v, float) else str(v))
        lines.append(" & ".join(cells) + r" \\\\")
    lines += [r"\\bottomrule", r"\\end{tabular}", r"\\end{table}"]
    return "\\n".join(lines)


for exp in experiments:
    runs = load_all_runs(OUT_DIR, exp)
    if not runs:
        continue
    rows = []
    # Grouped by split as well as variant and task: NB16 reports the same model
    # under four evaluation protocols, and collapsing them would average a
    # session-level result into a random-split one.
    for metric in ("accuracy", "f1", "auc"):
        for row in summarise_runs(runs, metric,
                                  group_keys=("variant", "task", "split")):
            rows.append({"variant": row["variant"], "task": row["task"],
                         "split": row["split"], "metric": metric,
                         "mean": row["mean"], "std": row["std"],
                         "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                         "n_seeds": row["n"]})
    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(PAPER_DIR / f"{exp}_summary.csv", index=False)
        (PAPER_DIR / f"tab_{exp}.tex").write_text(latex_table(
            rows, ["variant", "task", "split", "metric", "mean", "std", "n_seeds"],
            f"{exp} results (mean over seeds)", f"tab:{exp}"), encoding="utf-8")
        print(f"  {exp}: {len(rows)} summary rows -> tab_{exp}.tex")
"""),
        code("""
# --- experiment coverage map ----------------------------------------
COVERAGE = {
 "lightweight classical baselines":   ("nb04_baselines", "sweep_model_comparison.json"),
 "non-classical comparators":         ("nb14_quantum", "sweep_quantum_comparators.json"),
 "multiclass gap (79.19%)":           ("nb15_gap", "error_decomposition.json"),
 "adversarial robustness":            ("nb07_robustness", "sweep_adversarial_pca.json"),
 "operational cost":                  ("nb11_cost", "sweep_latency.json"),
 "related-work comparison":           ("nb14_quantum", "literature_rows.json"),
 "energy":                           ("nb11_cost", "sweep_energy.json"),
 "SHAP+LIME benefit":                 ("nb09_xai", "sweep_fidelity.json"),
 "8-qubit choice":                    ("nb06_sweep", "sweep_qubit_depth.json"),
 "audit log":                        ("nb10_audit", "sweep_tamper_detection.json"),
 "parameter-matched controls":        ("nb04_baselines", "sweep_model_comparison.json"),
 "ablation parameter counts":         ("nb05_ablation", "runs_index.csv"),
 "multi-seed statistics":             ("nb03_multiclass", "runs_index.csv"),
 "split protocols":                   ("nb16_splits", "sweep_protocol_comparison.json"),
 "realistic perturbations":           ("nb07_robustness", "sweep_realistic_raw_space.json"),
 "XAI stability/inverse map":         ("nb09_xai", "inverse_map_validation.json"),
 "shots / ideal simulation":          ("nb08_shots", "sweep_shot_sweep.json"),
 "enlarged test set":                 ("nb16_splits", "enlarged_test_set.json"),
}
print(f"{'experiment':38s} {'status':10s} evidence")
missing = []
for remark, (exp, art) in COVERAGE.items():
    p = OUT_DIR / exp / art
    ok = p.exists()
    if not ok: missing.append(remark)
    print(f"{remark:38s} {'COVERED' if ok else 'MISSING':10s} {exp}/{art}")
print(f"\\n{len(COVERAGE)-len(missing)}/{len(COVERAGE)} experiments have saved evidence")
if missing:
    print("outstanding:", missing)
write_json(PAPER_DIR / "coverage_map.json",
           {"covered": [k for k in COVERAGE if k not in missing], "missing": missing})
"""),
        code("""
# --- final completeness audit --------------------------------------------
# Expected layers differ by experiment type: a training run must carry curves
# and weights, an inference-only run has no training history to save, and the
# audit-log study has neither a model nor per-sample predictions.
# A gate that flags every run would simply be ignored, so it is made specific.
EXPECTED = {
    "nb00_verify":     ("metrics", "predictions", "model"),
    "nb01_data":       ("metrics",),
    "nb02_binary":     ("metrics", "predictions", "curves", "model"),
    "nb03_multiclass": ("metrics", "predictions", "curves", "model"),
    "nb04_baselines":  ("metrics", "predictions", "model"),
    "nb05_ablation":   ("metrics", "predictions", "curves", "model"),
    "nb06_sweep":      ("metrics", "predictions", "curves", "model"),
    "nb07_robustness": ("metrics",),
    "nb08_shots":      ("metrics",),
    "nb09_xai":        ("metrics",),
    "nb10_audit":      ("metrics",),
    "nb11_cost":       ("metrics",),
    "nb12_cross":      ("metrics", "predictions", "model"),
    "nb14_quantum":    ("metrics", "predictions", "model"),
    "nb15_gap":        ("metrics", "predictions"),
    "nb16_splits":     ("metrics", "predictions", "model"),
}

all_ok = True
for exp in experiments:
    req = EXPECTED.get(exp, ("metrics",))
    df = check_completeness(OUT_DIR, exp, required=req)
    if not df.empty:
        all_ok &= bool(df["complete"].all())

print("\\n" + ("ALL EXPERIMENTS COMPLETE -- safe to release the allocation."
               if all_ok else
               "SOME CELLS INCOMPLETE -- fix before releasing the allocation."))
print("\\nOutputs written to:", PAPER_DIR)
print("Every table and figure regenerates from saved artefacts -- no re-training.")
"""),
    ], needs="none")
