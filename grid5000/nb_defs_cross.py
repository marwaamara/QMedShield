"""
nb_defs_cross.py -- notebook 12, cross-dataset generalisation.

Kept in its own module so `nb_defs.py` stays the core suite. Imported by
build_notebooks.py after nb_defs.
"""
from __future__ import annotations

from nb_defs import code, register

register(
    "12_cross_dataset_wustl.ipynb",
    "12 - Cross-dataset generalisation on WUSTL-EHMS-2020",
    """**Question:** does the framework generalise to a second, independent healthcare IoT
dataset in realistic deployment scenarios?""",
    """
Follows a frozen-transfer / fine-tune design, applied to **this** paper's architecture: the 8-qubit, 4-layer circuit with the three-scale
entangler, trained on CICIoMT2024 and evaluated on a second, independent healthcare IoT
benchmark.

Two regimes, as in the original:

1. **Frozen transfer** - encoder, circuit and decoder held fixed; only the input projection is
   refitted to the new feature space. Tests whether the learned representation transfers at all.
2. **Fine-tuned head** - the classical decoder is retrained while the quantum core stays
   frozen. This is the deployment-realistic regime the paper argues for: the compact
   quantum core ships once, and each site adapts only its head.

This is reported as supporting evidence for
generalisation, never as a headline claim.

> Optional. Place `wustl-ehms-2020.csv` in `data/`.
""",
    [
        code("""
if not WUSTL_PATH.exists():
    print("WUSTL-EHMS-2020 not found at", WUSTL_PATH)
    print("  This notebook is OPTIONAL -- it strengthens the generalisation argument")
    print("  but it is not required for the main results.")
    print("  To enable: place wustl-ehms-2020.csv in", DATA_DIR)
    write_json(OUT_DIR / "nb12_cross" / "status.json",
               {"available": False, "reason": "dataset absent", "optional": True})
    WUSTL_OK = False
else:
    import pandas as pd
    df = pd.read_csv(WUSTL_PATH)
    print("loaded", df.shape)
    WUSTL_OK = True
"""),
        code("""
if WUSTL_OK:
    # Binary labels + numeric feature matrix, as in the frozen-transfer design.
    label_col = next((c for c in df.columns
                      if c.lower() in ("label", "attack", "class", "target")), None)
    assert label_col, f"no label column among {list(df.columns)[:20]}"
    y = (df[label_col].astype(str).str.lower()
         .apply(lambda s: 0 if any(k in s for k in ("normal", "benign", "0")) else 1)).values
    Xnum = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    Xnum = Xnum.fillna(Xnum.mean()).values
    print(f"features {Xnum.shape}  benign={int((y==0).sum())}  attack={int((y==1).sum())}")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    Xtr, Xte, ytr, yte = train_test_split(Xnum, y, test_size=0.3, stratify=y,
                                          random_state=42)
    # The input projection is refitted to this dataset's feature space; the
    # quantum core is what is being tested for transfer, not the preprocessing.
    sc = StandardScaler().fit(Xtr)
    pc = PCA(n_components=N_PCA, random_state=42).fit(sc.transform(Xtr))
    Xtr_p, Xte_p = pc.transform(sc.transform(Xtr)), pc.transform(sc.transform(Xte))
    print("PCA explained variance:", round(float(pc.explained_variance_ratio_.sum()), 4))
"""),
        code("""
if WUSTL_OK:
    # --- regime 1: frozen transfer ----------------------------------------
    src, _ = load_hybrid("binary", str(ARCHIVE_DIR / "hybrid_qcnn_binary (1).pt"),
                         model=build_model("binary", "full"))
    r_frozen = evaluate(src, make_loader(Xte_p, yte, shuffle=False), 2)
    print(f"frozen transfer   acc={r_frozen['accuracy']:.4f} "
          f"f1={r_frozen['f1']:.4f} auc={r_frozen['auc']:.4f}")
    save_run(OUT_DIR, experiment="nb12_cross", variant="frozen", task="binary",
             split="wustl", seed=42,
             y_true=r_frozen['_labels'], y_pred=r_frozen['_predictions'],
             probs=r_frozen['_probabilities'],
             metrics={k: v for k, v in r_frozen.items()
                      if not k.startswith('_') and np.isscalar(v)},
             model=src, preprocessors={"scaler": sc, "pca": pc},
             class_names=["Benign", "Attack"],
             extra={"notebook": "12", "regime": "frozen_transfer",
                    "dataset": "WUSTL-EHMS-2020"})
"""),
        code("""
if WUSTL_OK:
    # --- regime 2: fine-tune the decoder, quantum core frozen -------------
    import torch, copy
    ft = copy.deepcopy(src)
    for p_ in ft.classical_encoder.parameters():
        p_.requires_grad_(False)
    for p_ in ft.bottleneck.parameters():
        p_.requires_grad_(False)
    n_train = sum(p_.numel() for p_ in ft.parameters() if p_.requires_grad)
    print(f"fine-tuning {n_train} decoder parameters; quantum core frozen")

    set_global_seed(42)
    hist = train_model(ft, make_loader(Xtr_p, ytr),
                       make_loader(Xte_p, yte, shuffle=False),
                       task="binary", class_weights=compute_class_weights(ytr, 2))
    r_ft = evaluate(ft, make_loader(Xte_p, yte, shuffle=False), 2)
    print(f"fine-tuned head   acc={r_ft['accuracy']:.4f} "
          f"f1={r_ft['f1']:.4f} auc={r_ft['auc']:.4f}")
    save_run(OUT_DIR, experiment="nb12_cross", variant="finetuned", task="binary",
             split="wustl", seed=42,
             y_true=r_ft['_labels'], y_pred=r_ft['_predictions'],
             probs=r_ft['_probabilities'],
             metrics={k: v for k, v in r_ft.items()
                      if not k.startswith('_') and np.isscalar(v)},
             history=hist, model=ft, preprocessors={"scaler": sc, "pca": pc},
             params={"trainable": n_train}, class_names=["Benign", "Attack"],
             extra={"notebook": "12", "regime": "finetune_head",
                    "dataset": "WUSTL-EHMS-2020"})
"""),
        code("""
if WUSTL_OK:
    FIG = ROOT / "figures"
    FIG.mkdir(exist_ok=True)
    save_curve(OUT_DIR, "nb12_cross", "roc_frozen",
               **roc_curves(r_frozen['_labels'], r_frozen['_probabilities'],
                            ["Benign", "Attack"],
                            path=str(FIG / "wustl_roc_frozen.png")))
    save_curve(OUT_DIR, "nb12_cross", "roc_finetuned",
               **roc_curves(r_ft['_labels'], r_ft['_probabilities'],
                            ["Benign", "Attack"],
                            path=str(FIG / "wustl_roc_finetuned.png")))
    save_curve(OUT_DIR, "nb12_cross", "finetune_learning_curves",
               **training_curves({"fine-tune": hist},
                                 path=str(FIG / "wustl_finetune_curves.png"),
                                 metrics=("train_loss", "val_f1")))
    save_sweep(OUT_DIR, "nb12_cross", "transfer_comparison", [
        {"regime": "frozen transfer", "accuracy": r_frozen["accuracy"],
         "f1": r_frozen["f1"], "auc": r_frozen["auc"], "trainable_params": 0},
        {"regime": "fine-tuned head", "accuracy": r_ft["accuracy"],
         "f1": r_ft["f1"], "auc": r_ft["auc"], "trainable_params": n_train},
    ])
    check_completeness(OUT_DIR, "nb12_cross")
"""),
    ], needs="optional")
