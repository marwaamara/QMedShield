"""
nb_defs_quantum.py -- notebook 14, quantum and related-work comparators.

Kept in its own module for the same reason as nb_defs_cross.py: `nb_defs.py`
stays the core suite. Imported by build_notebooks.py after nb_defs.

Compares against quantum baselines (beyond classical machine learning baselines) and
against Al-Hawawreh & Hossain (2023), Inf. Fusion 99:101889.
"""
from __future__ import annotations

from nb_defs import code, register

register(
    "14_quantum_related_baselines.ipynb",
    "14 - Quantum and related-work comparators",
    """**Question:** how does the method compare against quantum baselines and against related
work, namely Al-Hawawreh, M., & Hossain, M. S. (2023), A privacy-aware framework for detecting
cyber attacks on internet of medical things systems using data fusion and quantum deep
learning, Information Fusion, 99, 101889?""",
    """
Five comparators, all on **the same split, the same preprocessing and the same schedule** as
the headline model, so every row is directly comparable:

| row | what it isolates |
|---|---|
| `qsvm_angle` | a quantum **kernel** method on our own encoding family |
| `qsvm_iqp` | a genuinely entangling kernel, not expressible as a product of overlaps |
| `vqc` | the quantum layer **without** the classical encoder/decoder - the mirror image of NB04's parameter-matched controls |
| `amp_qnn` | amplitude encoding: all 16 PCA components in 4 qubits, no decoder |
| `alhawawreh_qdl` | re-implementation of the fusion + quantum-deep-learning architecture of Al-Hawawreh & Hossain (2023) |

**Why a 3,000-sample QSVM is affordable.** The fidelity kernel
`k(x,y) = |<phi(x)|phi(y)>|^2` is normally evaluated pair by pair, which costs N^2 circuit
executions and is why quantum-kernel baselines are usually reported on a few hundred samples.
The states themselves need only N executions: simulate once per sample, stack into `S`, and the
entire Gram matrix is `|S S^H|^2`. That is an identity, not an approximation - and
`verify_kernel_matrix` checks it against pair-by-pair circuit evaluation before any number is
reported. The training set is still subsampled (Gram memory is quadratic); **the test set is
not**, so the rows stay comparable, and both counts are written into the result JSON.

**On the Al-Hawawreh row - read this before citing it.** The source framework fuses
heterogeneous modalities including patient biometric telemetry, which the CICIoMT2024 WiFi/MQTT
partition does not contain. The re-implementation therefore fuses the four modalities the flow
records *do* carry (timing, flags, protocol identity, size statistics) through per-modality
encoders into a shared fused representation, then a variational quantum layer. The
privacy-preserving component is orthogonal to detection accuracy and is not re-implemented.
Every such decision is in `ALHAWAWREH_DEVIATIONS` and is written into the result JSON.

> This row must be presented as **a faithful re-implementation under stated
> assumptions on our dataset**, never as the authors' own reported result. Numbers the authors
> report on their own data are transcribed separately in `LITERATURE_ROWS`, and this notebook
> prints them as OUTSTANDING until someone copies them from the article - it will not invent
> them, and it will not emit a blank table cell.

> Training runs for `vqc`, `amp_qnn` and `alhawawreh_qdl`; the two QSVM rows need no gradient
> descent at all. `QUICK_MODE=1` for a fast pass.
""",
    [
        code("""
data = load_prepared_data(str(ARCHIVE_DIR))
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
EXP = "nb14_quantum"
TASKS = ["multiclass", "binary"]
SFX = {"multiclass": "m", "binary": "b"}

# The reference row: our own model, as published.
REF = {t: json.loads((ARCHIVE_DIR / f"{t}_test_metrics.json").read_text())
       for t in TASKS}
for t in TASKS:
    print(f"reference QCNN {t:11s} acc={REF[t]['accuracy']:.4f} f1={REF[t]['f1']:.4f}")
print()
print("comparators:", QUANTUM_BASELINES)
print(f"QSVM train subsample: {QSVM_TRAIN_SUBSAMPLE} (test set NOT subsampled)")
rows = []
"""),
        code("""
# --- quantum kernel SVMs --------------------------------------
# No gradient descent: the kernel is the model. C is selected on VALIDATION.
# The kernel rows are cheap, but they are sharded too so that concurrent
# jobs never compute the same Gram matrix twice.
MY_QSVM = set(shard_items([(t, e) for t in TASKS for e in QSVM_EMBEDDINGS]))
print(describe_selection(len(TASKS) * len(QSVM_EMBEDDINGS), len(MY_QSVM)))
for TASK in TASKS:
    sfx, NC = SFX[TASK], (2 if TASK == "binary" else N_CLASSES)
    X_tr, y_tr = data[f"X_train_{sfx}"], data[f"y_train_{sfx}"]
    X_va, y_va = data[f"X_val_{sfx}"],   data[f"y_val_{sfx}"]
    X_te, y_te = data[f"X_test_{sfx}"],  data[f"y_test_{sfx}"]
    Xs, ys, sub_idx = stratified_subsample(X_tr, y_tr, QSVM_TRAIN_SUBSAMPLE, seed=42)

    for emb in QSVM_EMBEDDINGS:
        if (TASK, emb) not in MY_QSVM:
            continue
        name = f"qsvm_{emb}"
        if is_done(OUT_DIR, EXP, name, TASK, "random", 42):
            print(f"{name} {TASK}: already done, skipping"); continue
        t0 = time.time()
        est, K_te, K_tr, meta = fit_qsvm(Xs, ys, X_te, embedding=emb,
                                         X_val=X_va, y_val=y_va, seed=42)

        # The fast Gram path is an identity -- test it, do not trust it.
        A_tr, _ = to_angles(Xs)
        ver = verify_kernel_matrix(A_tr, K_tr, emb, n_pairs=50)
        print(f"  kernel check [{emb}] max|err|={ver['max_abs_error']:.2e} "
              f"{'PASS' if ver['passed'] else 'FAIL'}")
        assert ver["passed"], "state-vector Gram matrix disagrees with circuit evaluation"

        r = evaluate_sklearn(est, K_te, y_te, NC)
        rows.append({"model": name, "family": "quantum-kernel", "task": TASK,
                     "accuracy": r["accuracy"], "f1": r["f1"], "auc": r["auc"],
                     "n_params": int(est.support_.size), "seed": 42, **meta})
        print(f"{name:16s} {TASK:11s} acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
              f"auc={r['auc']:.4f}  C={meta['C']}  "
              f"({meta['kernel_wall_s']:.1f}s kernel, "
              f"{meta['circuit_evals']:,} circuit evals vs "
              f"{meta['circuit_evals_pairwise_equivalent']:,} pairwise)")

        save_run(OUT_DIR, experiment=EXP, variant=name, task=TASK,
                 split="random", seed=42,
                 y_true=r['_labels'], y_pred=r['_predictions'],
                 probs=r['_probabilities'],
                 metrics={k: v for k, v in r.items()
                          if not k.startswith('_') and np.isscalar(v)},
                 model=est, params={"total": int(est.support_.size)},
                 timings={"kernel_wall_s": meta["kernel_wall_s"],
                          "fit_wall_s": meta["fit_wall_s"],
                          "wall_s": time.time() - t0},
                 class_names=(["Benign", "Attack"] if TASK == "binary"
                              else CLASS_NAMES),
                 extra={"notebook": "NB14", "topic": "quantum comparators", **meta,
                        "kernel_verification": ver})
"""),
        code("""
# --- variational comparators without a classical decoder ------
# vqc     : the quantum layer alone -- NB04 removes the circuit and keeps the
#           scaffolding, this removes the scaffolding and keeps the circuit.
# amp_qnn : amplitude encoding, all 16 components in 4 qubits.
#
# ~82 node-hours over both tasks and three seeds, so this is shardable:
#   SHARD=2/6 ./run_all.sh 14
MY_VARIATIONAL = set(shard_items(
    [(n, t, s) for n in ("vqc", "amp_qnn") for t in TASKS for s in SEEDS_SWEEP]))
print(describe_selection(2 * len(TASKS) * len(SEEDS_SWEEP), len(MY_VARIATIONAL)))
for TASK in TASKS:
    sfx, NC = SFX[TASK], (2 if TASK == "binary" else N_CLASSES)
    X_tr, y_tr = subsample_for_mode(data[f"X_train_{sfx}"], data[f"y_train_{sfx}"])
    X_va, y_va = data[f"X_val_{sfx}"],  data[f"y_val_{sfx}"]
    X_te, y_te = data[f"X_test_{sfx}"], data[f"y_test_{sfx}"]
    cw = compute_class_weights(y_tr, NC)

    for name in ("vqc", "amp_qnn"):
        for seed in SEEDS_SWEEP:
            if (name, TASK, seed) not in MY_VARIATIONAL:
                continue
            if is_done(OUT_DIR, EXP, name, TASK, "random", seed):
                print(f"{name} {TASK} seed {seed}: already done, skipping"); continue
            set_global_seed(seed)
            model = build_quantum_baseline(name, TASK)
            print(f"\\n=== {name} {TASK} seed {seed}  {model.param_breakdown()} ===")
            t0 = time.time()
            hist = train_model(model, make_loader(X_tr, y_tr),
                               make_loader(X_va, y_va, shuffle=False),
                               task=TASK, class_weights=cw)
            r = evaluate(model, make_loader(X_te, y_te, shuffle=False), NC)
            print(f"  test acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f}")
            rows.append({"model": name, "family": "variational-no-decoder",
                         "task": TASK, "seed": seed, "accuracy": r["accuracy"],
                         "f1": r["f1"], "auc": r["auc"],
                         "n_params": model.param_breakdown()["total"],
                         "footprint_kb": model.footprint_kb()})
            save_run(OUT_DIR, experiment=EXP, variant=name, task=TASK,
                     split="random", seed=seed,
                     y_true=r['_labels'], y_pred=r['_predictions'],
                     probs=r['_probabilities'],
                     metrics={k: v for k, v in r.items()
                              if not k.startswith('_') and np.isscalar(v)},
                     history=hist, model=model,
                     params=model.param_breakdown(),
                     timings={"train_wall_s": time.time() - t0},
                     class_names=(["Benign", "Attack"] if TASK == "binary"
                                  else CLASS_NAMES),
                     extra={"notebook": "NB14", "topic": "quantum comparators",
                            "circuit_evals": int(model.n_circuit_evals)})
"""),
        code("""
# --- Al-Hawawreh & Hossain (2023) re-implementation -----------
# Raw 25-feature space: the fusion architecture is defined over feature
# MODALITIES, which only exist before PCA. The archive's scaler is inverted to
# recover raw units, and the split rows are located in the pool by exact PCA
# match -- the same mechanism NB07 uses for its raw-space perturbations.
X_raw_pool, feat_names = recover_raw_features(str(ARCHIVE_DIR))
Xg_pool, group_sizes, group_cols, group_names = group_feature_matrix(
    X_raw_pool, feat_names)
print("fusion modalities (a partition of all 25 features):")
for g, ns in group_names.items():
    print(f"  {g:9s} {len(ns):2d}  {ns}")

for k, v in ALHAWAWREH_DEVIATIONS.items():
    print(f"\\n[{k}] {v}")
"""),
        code("""
name = "alhawawreh_qdl"
MY_QDL = set(shard_items([(t, s) for t in TASKS for s in SEEDS_SWEEP]))
print(describe_selection(len(TASKS) * len(SEEDS_SWEEP), len(MY_QDL)))
for TASK in TASKS:
    sfx, NC = SFX[TASK], (2 if TASK == "binary" else N_CLASSES)
    pool_pca = data[f"X_pca_{sfx}"]
    i_tr = locate_pool_rows(pool_pca, data[f"X_train_{sfx}"])
    i_va = locate_pool_rows(pool_pca, data[f"X_val_{sfx}"])
    i_te = locate_pool_rows(pool_pca, data[f"X_test_{sfx}"])
    # Standardise the grouped raw features on TRAIN rows only.
    from sklearn.preprocessing import StandardScaler
    gsc = StandardScaler().fit(Xg_pool[i_tr])
    Xg_tr, y_tr = subsample_for_mode(gsc.transform(Xg_pool[i_tr]),
                                     data[f"y_train_{sfx}"])
    Xg_va, y_va = gsc.transform(Xg_pool[i_va]), data[f"y_val_{sfx}"]
    Xg_te, y_te = gsc.transform(Xg_pool[i_te]), data[f"y_test_{sfx}"]
    cw = compute_class_weights(y_tr, NC)

    for seed in SEEDS_SWEEP:
        if (TASK, seed) not in MY_QDL:
            continue
        if is_done(OUT_DIR, EXP, name, TASK, "random", seed):
            print(f"{name} {TASK} seed {seed}: already done, skipping"); continue
        set_global_seed(seed)
        model = build_quantum_baseline(name, TASK, group_sizes=group_sizes)
        print(f"\\n=== {name} {TASK} seed {seed}  {model.param_breakdown()} ===")
        t0 = time.time()
        hist = train_model(model, make_loader(Xg_tr, y_tr),
                           make_loader(Xg_va, y_va, shuffle=False),
                           task=TASK, class_weights=cw)
        r = evaluate(model, make_loader(Xg_te, y_te, shuffle=False), NC)
        print(f"  test acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f}")
        rows.append({"model": name, "family": "related-work-reimplementation",
                     "task": TASK, "seed": seed, "accuracy": r["accuracy"],
                     "f1": r["f1"], "auc": r["auc"],
                     "n_params": model.param_breakdown()["total"],
                     "footprint_kb": model.footprint_kb()})
        save_run(OUT_DIR, experiment=EXP, variant=name, task=TASK,
                 split="random", seed=seed,
                 y_true=r['_labels'], y_pred=r['_predictions'],
                 probs=r['_probabilities'],
                 metrics={k: v for k, v in r.items()
                          if not k.startswith('_') and np.isscalar(v)},
                 history=hist, model=model, params=model.param_breakdown(),
                 preprocessors={"group_scaler": gsc},
                 timings={"train_wall_s": time.time() - t0},
                 class_names=(["Benign", "Attack"] if TASK == "binary"
                              else CLASS_NAMES),
                 extra={"notebook": "NB14", "topic": "related-work comparison",
                        "deviations": ALHAWAWREH_DEVIATIONS,
                        "modalities": {g: ns for g, ns in group_names.items()},
                        "status": "re-implementation on CICIoMT2024, NOT the "
                                  "authors' reported numbers"})
"""),
        code("""
# --- aggregate: mean +/- std over seeds, and the comparison table ---------
runs = load_all_runs(OUT_DIR, EXP)
print(f"{len(runs)} comparator runs on disk\\n")
for metric in ("accuracy", "f1", "auc"):
    for row in summarise_runs(runs, metric, group_keys=("variant", "task")):
        print(f"  {metric:8s} {row['variant']:16s} {row['task']:11s} "
              f"{format_mean_std(row)}  n={row['n']}")

# ALWAYS rebuild from the manifests, never from `rows` alone. `rows` holds only
# what this shard trained, and writing it over sweep_quantum_comparators.json
# would delete the rows other shards produced -- the same way notebook 06 lost
# its L=4 records. The manifests are the record; the sweep file is a view.
_seen = {(r.get("model"), r.get("task"), r.get("seed")) for r in rows}
rows = rows + [{"model": r["variant"], "task": r["task"], "seed": r["seed"],
             "family": (r.get("context", {}).get("extra", {}) or {}).get("family",
                        "quantum"),
             **{k: v for k, v in (r.get("metrics") or {}).items()
                if k in ("accuracy", "f1", "auc")},
               "n_params": (r.get("params") or {}).get("total")}
              for r in runs
              if (r["variant"], r["task"], r["seed"]) not in _seen]
print(f"{len(rows)} comparator rows (this run + saved manifests)")

if rows:
    save_sweep(OUT_DIR, EXP, "quantum_comparators", rows)
"""),
        code("""
# --- paired McNemar: our QCNN against every quantum comparator ------
# Same test rows, same order, so the tests are properly paired.
for TASK in TASKS:
    ref = load_predictions(OUT_DIR, "nb00_verify", "archive", TASK, "random", 42)
    fam = {}
    for name in QUANTUM_BASELINES:
        seed = 42 if name.startswith("qsvm") else SEEDS_SWEEP[0]
        try:
            p = load_predictions(OUT_DIR, EXP, name, TASK, "random", seed)
        except Exception:
            print(f"  {name} {TASK}: no predictions yet, skipped"); continue
        fam[name] = p["y_pred"]
    if not fam:
        continue
    tests = compare_against_family(ref["y_true"], ref["y_pred"], fam)
    for t in tests:
        print(f"[{TASK:11s}] QCNN vs {t['baseline']:16s} b={t['b']:4d} c={t['c']:4d} "
              f"p_holm={t['p_holm']:.3e} "
              f"{'SIGNIFICANT' if t['significant_holm'] else 'n.s.'}")
    save_sweep(OUT_DIR, EXP, f"mcnemar_quantum_{TASK}", tests)
"""),
        code("""
# --- literature rows: transcribed, never invented -------------------------
lit = literature_table()
write_json(OUT_DIR / EXP / "literature_rows.json", lit)
if lit["outstanding"]:
    print("OUTSTANDING -- copy these numbers from the cited articles into "
          "config.LITERATURE_ROWS before the table is reported:")
    for r in lit["outstanding"]:
        print(f"  - {r['ref']}  ({r['method']})")
else:
    for r in lit["transcribed"]:
        print(f"  {r['ref']}: {r['reported']}")
print("\\n" + lit["note"])
"""),
        code("""
# --- accuracy vs footprint, quantum family --------------------------------
plot_rows = [r for r in rows if r.get("task") == "multiclass"
             and r.get("footprint_kb")]
if plot_rows:
    plot_rows = plot_rows + [{"model": "QCNN (ours)", "footprint_kb": 26.4,
                              "accuracy": REF["multiclass"]["accuracy"]}]
    save_curve(OUT_DIR, EXP, "quantum_pareto",
               **pareto_scatter(plot_rows, "footprint_kb", "accuracy", "model",
                                path=str(FIG / "quantum_comparators.png")))
check_completeness(OUT_DIR, EXP, required=("metrics", "predictions", "model"))
"""),
    ], needs="training")
