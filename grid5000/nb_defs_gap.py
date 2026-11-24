"""
nb_defs_gap.py -- notebook 15, the multiclass accuracy gap.

Investigates why multiclass accuracy (79.19%) is lower than binary accuracy.

Imported by build_notebooks.py after nb_defs.
"""
from __future__ import annotations

from nb_defs import code, register

register(
    "15_multiclass_gap.ipynb",
    "15 - Where the multiclass gap is, and whether it can be closed",
    """**Question:** the binary detector achieves excellent performance, whereas multiclass
accuracy is relatively lower (79.19%). Where does the gap come from, and can it be closed?""",
    """
The first question is not how to fix the gap but **where it is**. On the archived predictions
the answer is unusually concentrated:

| confusion | count | share of all error |
|---|---|---|
| DoS -> DDoS | 1,163 | 74.5% |
| DDoS -> DoS | 351 | 22.5% |
| **the two volumetric confusions together** | **1,514** | **97.0%** |
| everything else, all six remaining classes | 47 | 3.0% |

Merge DoS and DDoS into a single volumetric-flooding class and **the same untouched model scores
99.37%**. So the multiclass task is not broadly harder than the binary one: one binary decision
inside it is, and that decision carries 92% of the test set.

This notebook then asks whether that decision is recoverable, in the only order that yields a
clean answer:

1. **Error decomposition** - locate and quantify the gap (no training).
2. **Ceiling analysis** - tie a given DoS-vs-DDoS accuracy to the end-to-end 8-class number, so
   a gain on the subproblem is never reported as an equal gain overall.
3. **Representation probe** - fit tuned *classical* specialists on the volumetric rows alone, in
   both PCA-16 and raw-25 space. If they plateau too, the ceiling belongs to the feature set,
   not to the quantum model, and the paper can say so with evidence.
4. **Two-stage hierarchical detector** - a 7-way coarse stage plus a DoS-vs-DDoS specialist,
   both built from the same factory as the headline model.
5. **Feature-path variant** - route the raw features with the highest mutual information with
   the DoS/DDoS label around PCA, straight into the decoder. Tests whether variance-maximising
   PCA is discarding the discriminative directions.

**Report whatever lands.** If the hierarchy wins, it is a contribution; if it does not, the
decomposition plus the classical-probe ceiling is still a complete, quantitative answer -
considerably stronger than tuning until the number moves.

> Steps 1-3 need no training. Steps 4-5 train; `QUICK_MODE=1` for a fast pass.
""",
    [
        code("""
data = load_prepared_data(str(ARCHIVE_DIR))
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
EXP = "nb15_gap"
TASK, sfx, NC = "multiclass", "m", N_CLASSES

X_tr, y_tr = data["X_train_m"], data["y_train_m"]
X_va, y_va = data["X_val_m"],   data["y_val_m"]
X_te, y_te = data["X_test_m"],  data["y_test_m"]

# The archived model's own predictions -- the object of the analysis.
arch = np.load(ARCHIVE_DIR / "multiclass_predictions.npz")
y_true_arch, y_pred_arch = arch["labels"], arch["predictions"]
print(f"archived multiclass accuracy: {(y_true_arch == y_pred_arch).mean():.4f}")
"""),
        code("""
# --- 1. where does the error actually go? (no training) -------------------
dec = error_decomposition(y_true_arch, y_pred_arch, CLASS_NAMES)
print(f"{'true':16s} {'predicted':16s} {'count':>6s} {'% of error':>11s}")
for r in dec["top_confusions"]:
    print(f"{r['true']:16s} {r['pred']:16s} {r['count']:6d} "
          f"{100*r['share_of_error']:10.1f}%")
print(f"\\nvolumetric (DoS<->DDoS) confusions : {dec['volumetric_confusion_count']} "
      f"= {100*dec['volumetric_share_of_error']:.1f}% of all error")
print(f"accuracy, 8 classes                : {dec['accuracy']:.4f}")
print(f"accuracy, DoS and DDoS merged      : {dec['accuracy_merging_volumetric']:.4f}")
print(f"headroom held by ONE binary decision: "
      f"{100*dec['accuracy_headroom_if_volumetric_perfect']:.2f} pp")

write_json(OUT_DIR / EXP / "error_decomposition.json", dec)
save_sweep(OUT_DIR, EXP, "error_decomposition", dec["top_confusions"])
save_curve(OUT_DIR, EXP, "confusion_archived",
           **confusion_matrix_plot(dec["confusion_matrix"], CLASS_NAMES,
                                   path=str(FIG / "gap_confusion.png")))
"""),
        code("""
# --- 2. ceiling analysis --------------------------------------------------
# What a given DoS-vs-DDoS specialist accuracy is worth END TO END. The
# volumetric classes are 92% of the test set, so the mapping is nearly 1:1 --
# which is exactly why the subproblem number must never be quoted alone.
ceil_rows = ceiling_analysis(y_true_arch, y_pred_arch,
                             [0.78, 0.80, 0.85, 0.90, 0.95, 1.00])
for r in ceil_rows:
    print(f"specialist {100*r['specialist_accuracy']:5.1f}%  ->  overall "
          f"{100*r['implied_overall_accuracy']:5.2f}%")
save_sweep(OUT_DIR, EXP, "ceiling_analysis", ceil_rows)
save_curve(OUT_DIR, EXP, "ceiling",
           specialist_accuracy=[r["specialist_accuracy"] for r in ceil_rows],
           implied_overall=[r["implied_overall_accuracy"] for r in ceil_rows])
"""),
        code("""
# --- 3. representation probe: can ANYTHING separate DoS from DDoS? --------
# Tuned classical specialists, trained on the volumetric rows only, with the
# whole budget of that subproblem. Run in PCA-16 (what the model sees) and in
# raw-25 (everything the dataset records).
probe_rows = separability_probe(X_tr, y_tr, X_te, y_te, space="pca16")

X_raw_pool, feat_names = recover_raw_features(str(ARCHIVE_DIR))
i_tr = locate_pool_rows(data["X_pca_m"], X_tr)
i_te = locate_pool_rows(data["X_pca_m"], X_te)
probe_rows += separability_probe(X_raw_pool[i_tr], y_tr, X_raw_pool[i_te], y_te,
                                 space="raw25")

for r in probe_rows:
    print(f"{r['space']:7s} {r['probe']:22s} acc={r['accuracy']:.4f}"
          + (f"  f1={r['f1']:.4f}" if r["f1"] is not None else ""))

# What the archived hybrid model itself achieves on the same subproblem.
m_te = volumetric_mask(y_true_arch)
qcnn_sub = float((y_pred_arch[m_te] == y_true_arch[m_te]).mean())
print(f"\\narchived QCNN on the same subproblem: {qcnn_sub:.4f}")
probe_rows.append({"probe": "QCNN (archived, flat 8-class)", "space": "pca16",
                   "n_train": None, "n_test": int(m_te.sum()),
                   "accuracy": qcnn_sub, "f1": None, "auc": None,
                   "fit_wall_s": None})
save_sweep(OUT_DIR, EXP, "separability_probe", probe_rows)
print("\\nIf every probe plateaus near the same value, the ceiling is the FEATURE SET:")
print("CICIoMT2024 WiFi/MQTT flow records carry no source-address cardinality or")
print("dispersion, which is the property that distinguishes a distributed flood")
print("from a single-source one. State that as a measured limit, with this table.")
"""),
        code("""
# --- 4a. which raw features carry the DoS/DDoS signal? --------------------
# Mutual information, computed on TRAINING rows only. Selecting on the test
# split would manufacture the improvement the feature-path variant claims.
mi = mutual_information_ranking(X_raw_pool[i_tr], y_tr, feat_names)
for r in mi[:8]:
    print(f"  {r['rank']:2d}. {r['feature']:18s} MI={r['mutual_information']:.4f}")
save_sweep(OUT_DIR, EXP, "mutual_information_raw", mi)
TOP_COLS = [r["column"] for r in mi[:FEATURE_PATH_K]]
print(f"\\nrouting {FEATURE_PATH_K} features around PCA: "
      f"{[r['feature'] for r in mi[:FEATURE_PATH_K]]}")
"""),
        code("""
# --- 4b. two-stage hierarchical detector ----------------------------------
# Stage 1: 7-way coarse (DoS and DDoS merged). Stage 2: DoS-vs-DDoS specialist,
# trained on volumetric rows only -- which is what gives it a class balance the
# flat 8-way model never sees.
# ~61 node-hours for both approaches over three seeds -- shardable:
#   SHARD=2/4 ./run_all.sh 15
MY_HIER = shard_items([("hierarchical", s) for s in GAP_SEEDS])
print(describe_selection(len(GAP_SEEDS), len(MY_HIER)))
for _, seed in MY_HIER:
    if is_done(OUT_DIR, EXP, "hierarchical", TASK, "random", seed):
        print(f"hierarchical seed {seed}: already done, skipping"); continue
    set_global_seed(seed)
    two = TwoStageIDS()
    print(f"\\n=== hierarchical seed {seed}  {two.param_breakdown()} ===")
    t0 = time.time()

    Xc_tr, yc_tr = subsample_for_mode(X_tr, to_coarse(y_tr), seed)
    print("stage 1: 7-way coarse")
    h1 = train_model(two.stage1, make_loader(Xc_tr, yc_tr),
                     make_loader(X_va, to_coarse(y_va), shuffle=False),
                     task="multiclass",
                     class_weights=compute_class_weights(yc_tr, len(COARSE_NAMES)))

    mtr, mva = volumetric_mask(y_tr), volumetric_mask(y_va)
    Xs_tr, ys_tr = subsample_for_mode(X_tr[mtr], specialist_labels(y_tr)[mtr], seed)
    print("stage 2: DoS vs DDoS specialist")
    h2 = train_model(two.stage2, make_loader(Xs_tr, ys_tr),
                     make_loader(X_va[mva], specialist_labels(y_va)[mva],
                                 shuffle=False),
                     task="binary",
                     class_weights=compute_class_weights(ys_tr, 2))

    out = two.predict(X_te)
    r = {"accuracy": float((out["y_pred"] == y_te).mean())}
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    p_, r_, f_, _ = precision_recall_fscore_support(y_te, out["y_pred"],
                                                    average="weighted",
                                                    zero_division=0)
    r.update({"precision": float(p_), "recall": float(r_), "f1": float(f_)})
    m_te = volumetric_mask(y_te)
    r["specialist_accuracy"] = float(
        (out["specialist"][m_te] == specialist_labels(y_te)[m_te]).mean())
    r["coarse_accuracy"] = float((out["coarse"] == to_coarse(y_te)).mean())
    print(f"  8-class acc={r['accuracy']:.4f} f1={r['f1']:.4f}  "
          f"(coarse {r['coarse_accuracy']:.4f}, specialist "
          f"{r['specialist_accuracy']:.4f})  vs archived {dec['accuracy']:.4f}")

    save_run(OUT_DIR, experiment=EXP, variant="hierarchical", task=TASK,
             split="random", seed=seed,
             y_true=y_te, y_pred=out["y_pred"], metrics=r,
             history={f"stage1_{k}": v for k, v in h1.items()
                      if isinstance(v, list)},
             model=two, params=two.param_breakdown(),
             timings={"train_wall_s": time.time() - t0},
             class_names=CLASS_NAMES,
             extra={"notebook": "NB15", "topic": "multiclass gap", "approach": "hierarchical",
                    "stage1_history": {k: v for k, v in h1.items()
                                       if np.isscalar(v)},
                    "stage2_history": {k: v for k, v in h2.items()
                                       if np.isscalar(v)},
                    "cost_note": "two models and two forward passes on the "
                                 "volumetric fraction of traffic; parameters "
                                 "are the SUM of both stages"})
"""),
        code("""
# --- 5. feature-path variant ----------------------------------------------
# Identical encoder and quantum bottleneck; the decoder is k inputs wider and
# receives the top-MI raw features directly, bypassing PCA.
from sklearn.preprocessing import StandardScaler
rsc = StandardScaler().fit(X_raw_pool[i_tr][:, TOP_COLS])
R_tr = rsc.transform(X_raw_pool[i_tr][:, TOP_COLS])
R_va = rsc.transform(X_raw_pool[locate_pool_rows(data["X_pca_m"], X_va)][:, TOP_COLS])
R_te = rsc.transform(X_raw_pool[i_te][:, TOP_COLS])

MY_FP = shard_items([("feature_path", s) for s in GAP_SEEDS])
print(describe_selection(len(GAP_SEEDS), len(MY_FP)))
for _, seed in MY_FP:
    if is_done(OUT_DIR, EXP, "feature_path", TASK, "random", seed):
        print(f"feature_path seed {seed}: already done, skipping"); continue
    set_global_seed(seed)
    fp = FeaturePathIDS(TASK, k_raw=FEATURE_PATH_K, raw_columns=TOP_COLS)
    print(f"\\n=== feature_path seed {seed}  {fp.param_breakdown()} ===")
    t0 = time.time()
    # Same y and same seed -> subsample_for_mode draws the same indices, so the
    # PCA rows and the routed raw rows stay aligned. Asserted, not assumed.
    Xf_tr, yf_tr = subsample_for_mode(X_tr, y_tr, seed)
    Rf_tr, yr_tr = subsample_for_mode(R_tr, y_tr, seed)
    assert len(Xf_tr) == len(Rf_tr) and (yf_tr == yr_tr).all(),         "PCA and raw-path subsamples are misaligned"
    hist = train_feature_path(fp, (Xf_tr, Rf_tr, yf_tr), (X_va, R_va, y_va),
                              task=TASK,
                              class_weights=compute_class_weights(yf_tr, NC))
    r = evaluate_feature_path(fp, X_te, R_te, y_te, NC)
    extra_p = fp.param_breakdown()["extra_vs_flat"]
    print(f"  acc={r['accuracy']:.4f} f1={r['f1']:.4f}  vs archived "
          f"{dec['accuracy']:.4f}  (+{extra_p} parameters over the flat model)")

    save_run(OUT_DIR, experiment=EXP, variant="feature_path", task=TASK,
             split="random", seed=seed,
             y_true=r['_labels'], y_pred=r['_predictions'],
             probs=r['_probabilities'],
             metrics={k: v for k, v in r.items()
                      if not k.startswith('_') and np.isscalar(v)},
             history=hist, model=fp, params=fp.param_breakdown(),
             preprocessors={"raw_path_scaler": rsc},
             timings={"train_wall_s": time.time() - t0},
             class_names=CLASS_NAMES,
             extra={"notebook": "NB15", "topic": "multiclass gap", "approach": "feature_path",
                    "routed_features": [feat_names[c] for c in TOP_COLS],
                    "selection": "mutual information, TRAIN rows only"})
"""),
        code("""
# --- 6. did either approach close the gap? --------------------------------
runs = load_all_runs(OUT_DIR, EXP)
summary_rows = [{"approach": "archived flat 8-class", "accuracy": dec["accuracy"],
                 "f1": None, "n_params": 6768, "seeds": 1}]
for metric in ("accuracy", "f1"):
    for row in summarise_runs(runs, metric, group_keys=("variant",)):
        print(f"  {metric:8s} {row['variant']:14s} {format_mean_std(row)}  "
              f"95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}]  n={row['n']}")
        if metric == "accuracy":
            summary_rows.append({"approach": row["variant"],
                                 "accuracy": row["mean"], "std": row["std"],
                                 "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                                 "seeds": row["n"]})

# Paired McNemar against the archived model, same test rows.
fam = {}
for v in GAP_APPROACHES:
    try:
        fam[v] = load_predictions(OUT_DIR, EXP, v, TASK, "random",
                                  GAP_SEEDS[0])["y_pred"]
    except Exception:
        print(f"  {v}: no predictions yet, skipped")
if fam:
    tests = compare_against_family(y_true_arch, y_pred_arch, fam)
    for t in tests:
        print(f"archived vs {t['baseline']:14s} b={t['b']:4d} c={t['c']:4d} "
              f"p_holm={t['p_holm']:.3e} "
              f"{'SIGNIFICANT' if t['significant_holm'] else 'n.s.'}")
    save_sweep(OUT_DIR, EXP, "mcnemar_gap_approaches", tests)
save_sweep(OUT_DIR, EXP, "gap_approaches", summary_rows)
"""),
        code("""
check_completeness(OUT_DIR, EXP, required=("metrics", "predictions"))
print("\\nWhatever the outcome, sections 5.2 and 6 can now state WHERE the 20.8 pp")
print("goes, what a perfect DoS/DDoS decision would be worth, and whether any")
print("classifier at all can make that decision on these features.")
"""),
    ], needs="training")
