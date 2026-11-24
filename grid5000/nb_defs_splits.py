"""
nb_defs_splits.py -- notebook 16, evaluation protocols and the enlarged test set.

NB01 characterises the data and reports whether the protocols CAN be built.
This notebook is the experiment itself: it builds them, retrains under each one,
and produces the protocol comparison table -- plus the enlarged
per-class test set.

Imported by build_notebooks.py after nb_defs.
"""
from __future__ import annotations

from nb_defs import code, register

register(
    "16_split_protocols.ipynb",
    "16 - Session, device and chronological splits; enlarged test set",
    """**Question:** does the framework generalise under session-level, device-level and
chronological splits? A random train-test partition may let correlated flows from the same
capture session appear in both training and testing subsets, potentially giving optimistic
performance estimates.""",
    """
Four protocols, retrained under each:

| protocol | grouping | what it tests |
|---|---|---|
| `random` | none (stratified) | the protocol of the paper, kept as the reference row |
| `session` | source capture file | can the model generalise to a capture it never saw? |
| `device` | device / MAC / IP identity | can it generalise to an unseen device? |
| `chrono` | packet order within each capture | can it generalise forward in time? |

**The preprocessing is re-fitted inside every protocol** -- the ANOVA selector and the PCA as
well as the scaler. A session-level split whose PCA was fitted across all sessions leaks
through the preprocessing and reproduces the same flaw in a subtler form; `assert_fitted_on_train_only`
(validation gate **G5**) refuses to continue if any stage saw more rows than the training split
holds.

**Infeasible protocols are reported, not approximated.** The published CICIoMT2024 WiFi/MQTT
CSVs may carry no device identifier; if so, the device row states that as its result, naming the
columns that were searched for. Substituting some other grouping and calling it device-level
would be worse than reporting nothing.

**Enlarged test set.** The paper's per-class table reports F1 on
n = 4 (MQTT_Malformed) and n = 15 (ARP_Spoofing). Those estimates are not defensible. A disjoint
test set with n >= 100 per class is drawn from the rows the archived pool never touched -
membership is established by matching scaled feature vectors, and the match is conservative, so
a duplicated flow is dropped rather than risked. Inference is cheap: this costs minutes and no
retraining. The set is class-balanced by construction, so per-class metrics come from it while
prevalence-weighted metrics stay on the original split -- both are reported, and the difference
is stated.

> **Requires the raw CICIoMT2024 WiFi/MQTT CSVs in `DATA_DIR`.** The archive holds only
> post-PCA arrays and no per-flow provenance, so no amount of cleverness recovers capture
> identity or packet order from it. Without the CSVs this notebook writes a status file naming
> exactly what is blocked and why, and stops.
""",
    [
        code("""
EXP = "nb16_splits"
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
(OUT_DIR / EXP).mkdir(parents=True, exist_ok=True)

csvs = sorted(DATA_DIR.glob("*.csv"))
HAVE_CSV = bool(csvs)
if not HAVE_CSV:
    write_json(OUT_DIR / EXP / "splits_status.json",
               {"available": False,
                "blocked": ["session_split", "device_split",
                            "chronological_split", "enlarged_test_set"],
                "reason": "raw CICIoMT2024 WiFi/MQTT CSVs absent from "
                          f"{DATA_DIR}; the archive stores only post-PCA "
                          "arrays and no per-flow provenance (source capture "
                          "file, packet order, device identity)",
                "remedy": "https://www.unb.ca/cic/datasets/iomt-dataset-2024.html"})
    print("NO RAW CSVs in", DATA_DIR)
    print("  -> the split protocols and the enlarged test set CANNOT be built from the archive alone.")
    print("  -> download the CICIoMT2024 WiFi/MQTT partition and re-run.")
    print("  -> every later cell is a no-op; nothing below will fail loudly.")
else:
    print(f"found {len(csvs)} CSV files in {DATA_DIR}")
    # Profile them BEFORE any work. Every failure so far reduced to "these are
    # not the flow-feature CSVs the pipeline expects", and this says so in one
    # screen instead of surfacing as an empty matrix three cells later.
    profile_csvs(DATA_DIR)
"""),
        code("""
if HAVE_CSV:
    # PASS 1 -- labels and provenance only, read in chunks.
    # The WiFi/MQTT partition is several million flows. Loading every column of
    # every file and casting it to float64 needs many gigabytes, and a kernel
    # that exceeds what the node allows is killed by the OS -- which looks like
    # a notebook that simply stops, with no traceback. Only the label, the
    # capture file and the row position are needed to draw the pool.
    print("pass 1: scanning labels and provenance (2 columns, chunked)")
    raw = scan_flows(DATA_DIR)
    y_multi, y_bin = raw["y_multiclass"], raw["y_binary"]
    print(f"flows                : {raw['n_rows']:,}")
    print(f"numeric columns      : {len(raw['numeric_columns'])}")
    print(f"capture files        : {raw['n_sources']}")
    print(f"device column        : {raw['device_column'] or 'NONE FOUND'}")
    print(f"class distribution   : {np.bincount(y_multi, minlength=N_CLASSES).tolist()}")

    # The 50k pool keeps the protocols comparable with the setting of the paper;
    # a protocol evaluated on 5x the data would not be.
    pool = draw_pool(y_multi, POOL_SIZE, seed=42)
    print(f"pool                 : {len(pool):,} (stratified, seed 42)")

    # PASS 2 -- features for the pooled rows only, float32.
    print("pass 2: reading features for the pooled rows only")
    # raw["paths"] -- not a fresh glob: pass 1 may have set files aside (another
    # dataset in the same directory), and the pool indices are numbered over
    # the files it actually scanned.
    Xp, all_names = load_features_for(DATA_DIR, pool, raw["numeric_columns"],
                                      paths=raw["paths"])

    # What did we actually get? A silent mismatch here is what turns into an
    # opaque failure three cells later, so state it now.
    n_const = int((Xp.std(axis=0) == 0).sum())
    print(f"feature matrix       : {Xp.shape[0]:,} x {Xp.shape[1]} ({Xp.dtype})")
    print(f"constant columns     : {n_const} of {Xp.shape[1]}")
    print(f"first 8 features     : {all_names[:8]}")
    if Xp.shape[1] - n_const < 2:
        # The diagnosis goes INSIDE the exception, because a traceback is what
        # gets copied when something fails -- so the traceback must carry the
        # evidence rather than point at output that scrolled past.
        raise ValueError(
            f"only {Xp.shape[1] - n_const} non-constant feature(s) in the "
            f"pooled rows, from the CSVs in {DATA_DIR}. "
            + diagnose_features(DATA_DIR, Xp, all_names))
    if Xp.shape[1] - n_const < N_PCA:
        print(f"WARNING: {Xp.shape[1] - n_const} informative features is fewer "
              f"than the {N_PCA} PCA components the architecture expects; "
              f"training will stop with a clear message below.")
"""),
        code("""
if HAVE_CSV:
    # --- build every protocol, report feasibility explicitly -----------------
    raw_pool = {"source_csv": raw["source_csv"][pool],
                "row_order": raw["row_order"][pool],
                "device": (raw["device"][pool] if raw["device"] is not None
                           else None),
                "device_column": raw["device_column"],
                "n_sources": int(len(np.unique(raw["source_csv"][pool])))}
    yp_m, yp_b = y_multi[pool], y_bin[pool]   # Xp came from pass 2 already

    protocols, status = {}, []
    for proto in SPLIT_PROTOCOLS:
        info = protocol_indices(raw_pool, proto, yp_m, seed=42)
        if not info["feasible"]:
            print(f"{proto:8s} INFEASIBLE -- {info['reason']}")
            status.append({"protocol": proto, "feasible": False,
                           "reason": info["reason"]})
            continue
        protocols[proto] = info
        cc = info["class_counts"]["test"]
        cov = info.get("test_class_coverage", 1.0)
        flag = "" if cov >= 0.5 else "   <-- thin class coverage, read with care"
        print(f"{proto:8s} train={info['n_train']:6d} val={info['n_val']:5d} "
              f"test={info['n_test']:5d}  test classes present="
              f"{sum(c > 0 for c in cc)}/{N_CLASSES}{flag}")
        status.append({"protocol": proto, "feasible": True,
                       "n_train": info["n_train"], "n_val": info["n_val"],
                       "n_test": info["n_test"],
                       "test_class_counts": cc})
    save_sweep(OUT_DIR, EXP, "protocol_status", status)
    write_json(OUT_DIR / EXP / "splits_status.json",
               {"available": True, "protocols": status,
                "device_column": raw["device_column"],
                "n_sources": raw["n_sources"]})
"""),
        code("""
if HAVE_CSV:
    # --- QCNN under every feasible protocol -------------------
    # Preprocessing is re-fitted inside each protocol, and gate G5 checks it.
    #
    # ~146 node-hours over four protocols, two tasks and three seeds, so this is
    # the second-heaviest notebook after 05. Shard it:
    #   SHARD=3/12 ./run_all.sh 16
    MY_CELLS = set(shard_items([(p, t, s) for p in protocols
                                for t in ("multiclass", "binary")
                                for s in SEEDS_SWEEP]))
    print(describe_selection(len(protocols) * 2 * len(SEEDS_SWEEP),
                             len(MY_CELLS)))
    for proto, info in protocols.items():
        if not any(c[0] == proto for c in MY_CELLS):
            continue          # no cell of this protocol belongs to this shard
        tr, va, te = info["train_idx"], info["val_idx"], info["test_idx"]

        # Fitted ONCE per protocol, on the 8-class labels, exactly as the
        # original pipeline did -- the binary and multiclass models share the
        # preprocessing there too, and changing that here would confound the
        # protocol comparison with a preprocessing change.
        pre = FoldPreprocessor().fit(Xp[tr], yp_m[tr], all_names)
        assert_fitted_on_train_only(pre, tr)          # gate G5
        assert_no_leakage(tr, va, te)                 # gate G3

        # The architecture takes exactly N_PCA inputs. If the CSVs yield fewer
        # usable features, PCA silently returns fewer components and training
        # dies later with an opaque "mat1 and mat2 shapes cannot be multiplied".
        # Say what is actually wrong, here, before any compute.
        if pre.pca_.n_components_ != N_PCA:
            raise ValueError(
                f"the CSVs in {DATA_DIR} yield only "
                f"{pre.stage_counts['final']} usable numeric features, so PCA "
                f"returned {pre.pca_.n_components_} components -- the model "
                f"expects {N_PCA}. Check that DATA_DIR holds the full "
                f"CICIoMT2024 WiFi/MQTT flow-feature CSVs (the paper's Table 5 "
                f"reports 25 features after filtering), not a reduced export.")
        Z_tr, Z_va, Z_te = (pre.transform(Xp[tr]), pre.transform(Xp[va]),
                            pre.transform(Xp[te]))
        # Section 4.1 states that stages 1-2 leave 25 features and that no
        # supervised filter is applied. Print what actually happened on THIS
        # copy of the CSVs -- if the counts differ, section 4.1 needs correcting
        # and the manifest carries the evidence.
        print(f"  [{proto}] preprocessing {pre.stage_counts}  "
              f"anova_applied={pre.anova_applied}  "
              f"explained_variance="
              f"{pre.pca_.explained_variance_ratio_.sum()*100:.1f}% "
              f"(paper: 25 features, 96.3%)")

        for TASK in ("multiclass", "binary"):
            y_task = yp_m if TASK == "multiclass" else yp_b
            NC = N_CLASSES if TASK == "multiclass" else 2

            for seed in SEEDS_SWEEP:
                if (proto, TASK, seed) not in MY_CELLS:
                    continue
                if is_done(OUT_DIR, EXP, "qcnn", TASK, proto, seed):
                    print(f"qcnn {TASK} {proto} seed {seed}: done, skipping")
                    continue
                set_global_seed(seed)
                Zs, ys = subsample_for_mode(Z_tr, y_task[tr], seed)
                model = build_model(TASK, "full")
                print(f"\\n=== qcnn {TASK} {proto} seed {seed} ===")
                t0 = time.time()
                hist = train_model(model, make_loader(Zs, ys),
                                   make_loader(Z_va, y_task[va], shuffle=False),
                                   task=TASK,
                                   class_weights=compute_class_weights(ys, NC))
                r = evaluate(model, make_loader(Z_te, y_task[te], shuffle=False), NC)
                print(f"  acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f}")
                save_run(OUT_DIR, experiment=EXP, variant="qcnn", task=TASK,
                         split=proto, seed=seed,
                         y_true=r['_labels'], y_pred=r['_predictions'],
                         probs=r['_probabilities'], test_idx=te,
                         metrics={k: v for k, v in r.items()
                                  if not k.startswith('_') and np.isscalar(v)},
                         history=hist, model=model,
                         params=model.param_breakdown(),
                         preprocessors=pre.as_dict(),
                         timings={"train_wall_s": time.time() - t0},
                         class_names=(CLASS_NAMES if TASK == "multiclass"
                                      else ["Benign", "Attack"]),
                         extra={"notebook": "NB16", "topic": "split protocols",
                                "protocol": proto,
                                "preprocessing": pre.provenance()})
"""),
        code("""
if HAVE_CSV:
    # --- the same protocols for the strongest classical baselines ----------
    # A protocol that degrades every model equally says something different
    # from one that degrades only ours; without these rows the comparison
    # cannot be made.
    TOP3 = BASELINES[:3]
    for proto, info in protocols.items():
        tr, va, te = info["train_idx"], info["val_idx"], info["test_idx"]
        pre = FoldPreprocessor().fit(Xp[tr], yp_m[tr], all_names)
        assert_fitted_on_train_only(pre, tr)          # gate G5, again
        Z_tr, Z_te = pre.transform(Xp[tr]), pre.transform(Xp[te])
        for name in TOP3:
            v = name.replace(" ", "_")
            if is_done(OUT_DIR, EXP, v, "multiclass", proto, 42):
                print(f"{name} {proto}: done, skipping"); continue
            est, meta = fit_baseline(name, Z_tr, yp_m[tr], seed=42)
            r = evaluate_sklearn(est, Z_te, yp_m[te], N_CLASSES)
            print(f"{name:18s} {proto:8s} acc={r['accuracy']:.4f} f1={r['f1']:.4f}")
            save_run(OUT_DIR, experiment=EXP, variant=v, task="multiclass",
                     split=proto, seed=42,
                     y_true=r['_labels'], y_pred=r['_predictions'],
                     probs=r['_probabilities'],
                     metrics={k: val for k, val in r.items()
                              if not k.startswith('_') and np.isscalar(val)},
                     model=est, params={"total": count_sklearn_params(est)},
                     timings=meta, class_names=CLASS_NAMES,
                     extra={"notebook": "NB16", "topic": "split protocols",
                            "protocol": proto})
"""),
        code("""
if HAVE_CSV:
    # --- protocol comparison table -----------------------------------------
    runs = load_all_runs(OUT_DIR, EXP)
    rows = summarise_runs(runs, "accuracy", group_keys=("variant", "task", "split"))
    ref = {(r["variant"], r["task"]): r["mean"] for r in rows
           if r["split"] == "random"}
    table = []
    for r in sorted(rows, key=lambda r: (r["variant"], r["task"], r["split"])):
        base = ref.get((r["variant"], r["task"]))
        delta = (r["mean"] - base) if base is not None else None
        print(f"  {r['variant']:18s} {r['task']:11s} {r['split']:8s} "
              f"{format_mean_std(r)}"
              + (f"   {100*delta:+.2f} pp vs random" if delta is not None else ""))
        table.append({**{k: r[k] for k in ("variant", "task", "split", "mean",
                                           "std", "ci_low", "ci_high", "n")},
                      "delta_vs_random": delta})
    save_sweep(OUT_DIR, EXP, "protocol_comparison", table)
    save_curve(OUT_DIR, EXP, "protocol_comparison",
               **metric_vs_sweep([{"split": r["split"], "accuracy": r["mean"],
                                   "model": f"{r['variant']} ({r['task']})"}
                                  for r in table],
                                 "split", "accuracy", "model",
                                 path=str(FIG / "split_protocols.png"),
                                 xlabel="protocol", ylabel="accuracy"))
    print("\\nReport this block BESIDE the main table, not in an appendix "
          "(see the protocol comparison table): a drop under session or chronological evaluation is a "
          "result about deployment, not a footnote.")
"""),
        code("""
if HAVE_CSV:
    # --- enlarged test set, n >= 100 per class (enlarged test set) ---
    # Evaluated with the ARCHIVED scaler, PCA and weights, so no retraining is
    # involved and the numbers are directly comparable with the paper's per-class table.
    arch_data = load_prepared_data(str(ARCHIVE_DIR))
    pre_arch = load_preprocessors(str(ARCHIVE_DIR))

    # The archived pipeline selected 25 columns by name -- reuse them
    # directly, so the enlarged rows pass through EXACTLY the preprocessing
    # the archived weights were trained under. Re-fitting anything here would
    # compare the model against a pipeline it never saw.
    kept = [str(s) for s in arch_data["kept_names"]]
    missing = [k for k in kept if k not in all_names]
    if missing:
        print("archived feature names absent from the CSVs:", missing)
        print("-> cannot align the enlarged test set with the archived model")
    else:
        # The enlarged test set is drawn from rows OUTSIDE the pool, so its
        # features need their own pass -- again for selected rows only, never
        # for the whole partition.
        cols = [all_names.index(k) for k in kept]
        cand = np.setdiff1d(np.arange(raw["n_rows"]), pool)
        rng_c = np.random.default_rng(42)
        if len(cand) > 400_000:      # bound the scan; 400k is ample for n>=100/class
            cand = np.sort(rng_c.choice(cand, 400_000, replace=False))
        X_cand, _ = load_features_for(DATA_DIR, cand, all_names,
                                      paths=raw["paths"])
        X_scaled_all = pre_arch["scaler"].transform(X_cand[:, cols])
        y_cand = y_multi[cand]
        in_pool = identify_archived_pool_rows(X_scaled_all,
                                              arch_data["X_scaled"])
        print(f"candidate rows matching the archived 50k pool: "
              f"{int(in_pool.sum()):,} (excluded from the enlarged test set)")

        enl = enlarged_test_indices(y_cand, in_pool, seed=42)
        print("enlarged test counts per class:")
        for i, n in enumerate(enl["counts"]):
            print(f"  {CLASS_NAMES[i]:16s} n={n:6d}"
                  + ("   <-- still below target" if n < MIN_TEST_PER_CLASS else ""))

        Z_enl = pre_arch["pca_multi"].transform(X_scaled_all[enl["test_idx"]])
        y_enl = y_cand[enl["test_idx"]]
        model, _ = load_hybrid("multiclass",
                               str(ARCHIVE_DIR / "hybrid_qcnn_multiclass (1).pt"),
                               model=build_model("multiclass", "full"))
        r = evaluate(model, make_loader(Z_enl, y_enl, shuffle=False), N_CLASSES)
        print(f"\\narchived model on the enlarged test set: acc={r['accuracy']:.4f} "
              f"f1={r['f1']:.4f}  (n={len(y_enl):,})")
        rep = per_class_report(y_enl, r['_predictions'], CLASS_NAMES)
        for n, f1, sup in zip(rep["class_names"], rep["f1"], rep["support"]):
            print(f"  {n:16s} F1={f1:.4f}  n={sup}")

        write_json(OUT_DIR / EXP / "enlarged_test_set.json",
                   {k: v for k, v in enl.items() if k != "test_idx"})
        save_run(OUT_DIR, experiment=EXP, variant="archive_enlarged_test",
                 task="multiclass", split="enlarged", seed=42,
                 y_true=r['_labels'], y_pred=r['_predictions'],
                 probs=r['_probabilities'], test_idx=enl["test_idx"],
                 metrics={k: v for k, v in r.items()
                          if not k.startswith('_') and np.isscalar(v)},
                 model=model, params=model.param_breakdown(),
                 class_names=CLASS_NAMES,
                 extra={"notebook": "NB16", "topic": "enlarged test set",
                        "counts": enl["counts"],
                        "true_prevalence": enl["true_prevalence"],
                        "interpretation": enl["note"]})
"""),
        code("""
if HAVE_CSV:
    check_completeness(OUT_DIR, EXP, required=("metrics", "predictions"))
else:
    print("nothing to audit: the protocols need the raw CSVs.")
"""),
    ], needs="training")
