"""
persistence.py -- comprehensive artefact persistence.

The design goal is:

    "store enough raw artefacts that any future statistical analysis
     (Friedman, McNemar, bootstrap, calibration, new SHAP samples, threshold
     sweeps, fairness metrics, etc.) can be added WITHOUT re-running any
     training."

Persisting every artefact avoids re-runs. Gaps in earlier Grid5000 runs that
would each have forced a re-run:

  * noise_results.json          kept F1 ONLY -> no accuracy/AUC/per-class, no
                                predictions, so no confidence intervals
  * sample_efficiency_results    F1 only
  * ablation_*_partial.json      7 scalars per variant; no weights, no curves,
                                no predictions
  * baselines                    best_params kept but NOT the fitted models and
                                NOT per-sample predictions -> the paired
                                McNemar tests were impossible
  * XAI                          PNG figures only; no SHAP/LIME values

Accordingly EVERY run persists five layers:

  1. scalars   metrics.json          -- the numbers that go in a table
  2. vectors   predictions.npz       -- y_true, y_pred, probs, test_idx
  3. curves    curves/*.csv + *.npz  -- every value behind every plotted line
  4. state     checkpoint.pt / model.joblib + preprocessors/
  5. context   manifest.json         -- config, seed, versions, timing, OAR id

A run is only considered COMPLETE when all applicable layers exist;
`check_completeness()` reports what is missing and is designed to be called
BEFORE the compute allocation is released.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# Provenance (layer 5)
# =============================================================================
def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def _versions() -> Dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "sklearn", "torch", "pennylane",
                "shap", "lime", "statsmodels", "scipy"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            out[mod] = "absent"
    return out


def build_context(seed: Optional[int] = None,
                  extra: Optional[Dict] = None) -> Dict:
    ctx = {
        "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "git_commit": _git_commit(),
        "versions": _versions(),
        "seed": seed,
        "job": {k: os.environ.get(k) for k in
                ("OAR_JOB_ID", "OAR_ARRAY_INDEX", "SLURM_JOB_ID",
                 "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    if extra:
        ctx.update(extra)
    return ctx


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_jsonable), encoding="utf-8")
    return path


# =============================================================================
# Run directories
# =============================================================================
def run_dir(out_dir, experiment: str, variant: str = "main",
            task: str = "multiclass", split: str = "random",
            seed: int = 42) -> Path:
    """One directory per (experiment, variant, task, split, seed) cell.

    A walltime kill therefore costs one cell, never a whole stage, and
    re-submitting can skip completed cells (see `is_done`).
    """
    p = (Path(out_dir) / experiment / "runs" /
         f"{variant}__{task}__{split}__seed{seed}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_done(out_dir, experiment: str, variant: str = "main",
            task: str = "multiclass", split: str = "random",
            seed: int = 42) -> bool:
    d = (Path(out_dir) / experiment / "runs" /
         f"{variant}__{task}__{split}__seed{seed}")
    return (d / "manifest.json").exists()


# =============================================================================
# Layer 3 -- curves
# =============================================================================
def save_curve(out_dir, experiment: str, curve_name: str, **series) -> Path:
    """Persist EVERY value behind a plotted line, as CSV and NPZ.

    Figures are not data. Saving the underlying values means that changing an
    axis, adding a confidence band, or re-plotting at a different DPI never
    requires recomputing the experiment.
    """
    d = Path(out_dir) / experiment / "curves"
    d.mkdir(parents=True, exist_ok=True)

    arrays = {k: np.asarray(v) for k, v in series.items()}

    # A sweep that ran only part of its grid produces an empty series. Saving
    # nothing and saying so is right; crashing on max() of an empty sequence
    # loses the rest of the cell, including the runs that DID complete.
    if not arrays or all(len(v) == 0 for v in arrays.values()):
        print(f"[persist] {experiment}/{curve_name}: no data yet, curve skipped "
              f"(the runs behind it are still missing)", flush=True)
        return None

    np.savez_compressed(d / f"{curve_name}.npz", **arrays)

    n = max(len(v) for v in arrays.values())
    df = pd.DataFrame({k: pd.Series(list(v)) for k, v in arrays.items()},
                      index=range(n))
    csv = d / f"{curve_name}.csv"
    df.to_csv(csv, index=False)
    return csv


# =============================================================================
# Layer 1/2 -- sweeps in LONG format
# =============================================================================
def save_sweep(out_dir, experiment: str, sweep_name: str,
               records: Sequence[Dict]) -> Path:
    """Persist a sweep as a long-format table carrying EVERY metric.

    Every metric is kept (accuracy, precision, recall, F1 and AUC), so the
    analysis can be given confidence intervals without re-running it.
    """
    d = Path(out_dir) / experiment
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / f"sweep_{sweep_name}.json",
               {"records": list(records), "context": build_context()})
    df = pd.DataFrame(list(records))
    csv = d / f"sweep_{sweep_name}.csv"
    df.to_csv(csv, index=False)
    return csv


# =============================================================================
# The main entry point
# =============================================================================
def save_run(out_dir, *, experiment: str, variant: str = "main",
             task: str = "multiclass", split: str = "random", seed: int = 42,
             y_true=None, y_pred=None, probs=None, test_idx=None,
             metrics: Optional[Dict] = None,
             history: Optional[Dict] = None,
             model=None, preprocessors: Optional[Dict] = None,
             params: Optional[Dict] = None,
             timings: Optional[Dict] = None,
             class_names: Optional[Sequence[str]] = None,
             extra: Optional[Dict] = None,
             save_checkpoint: bool = True) -> Path:
    """Persist all five artefact layers for one run cell."""
    d = run_dir(out_dir, experiment, variant, task, split, seed)
    files: Dict[str, str] = {}

    # -- layer 2: per-sample vectors (paired significance tests need these) --
    if y_true is not None and y_pred is not None:
        np.savez_compressed(
            d / "predictions.npz",
            y_true=np.asarray(y_true), y_pred=np.asarray(y_pred),
            probs=np.asarray(probs) if probs is not None else np.array([]),
            test_idx=np.asarray(test_idx) if test_idx is not None else np.array([]),
        )
        files["predictions"] = "predictions.npz"
        write_json(d / "per_class.json",
                   per_class_report(y_true, y_pred, class_names))
        files["per_class"] = "per_class.json"

    # -- layer 1: scalars ---------------------------------------------------
    if metrics:
        write_json(d / "metrics.json",
                   {k: v for k, v in metrics.items() if not k.startswith("_")})
        files["metrics"] = "metrics.json"

    # -- layer 3: curves ----------------------------------------------------
    if history:
        series = {k: v for k, v in history.items()
                  if isinstance(v, (list, tuple, np.ndarray))}
        if series:
            np.savez_compressed(d / "history.npz",
                                **{k: np.asarray(v) for k, v in series.items()})
            pd.DataFrame({k: pd.Series(list(v)) for k, v in series.items()}
                         ).to_csv(d / "history.csv", index=False)
            files["curves"] = "history.{csv,npz}"
        write_json(d / "history_scalars.json",
                   {k: v for k, v in history.items() if np.isscalar(v)})

    # -- layer 4: state -----------------------------------------------------
    if model is not None and save_checkpoint:
        try:
            import torch
            if hasattr(model, "state_dict"):
                torch.save(model.state_dict(), d / "checkpoint.pt")
                files["model"] = "checkpoint.pt"
                if params is None:
                    params = {"total": int(sum(p.numel() for p in model.parameters()
                                               if p.requires_grad))}
            else:
                raise AttributeError
        except Exception:
            try:
                import joblib
                joblib.dump(model, d / "model.joblib")
                files["model"] = "model.joblib"
            except Exception as e:      # never let persistence kill a run
                files["model_error"] = str(e)

    if preprocessors:
        # Same rule as the model branch above: persistence must never kill a
        # run. A long-path failure on Windows, a full disk, or an unpicklable
        # transformer costs the preprocessors, not the metrics and predictions
        # that took the hours to produce -- and the failure is recorded rather
        # than swallowed.
        import joblib
        pdir = d / "preprocessors"
        saved, failed = [], {}
        try:
            pdir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            failed["__mkdir__"] = str(e)
        for k, v in preprocessors.items():
            try:
                joblib.dump(v, pdir / f"{k}.joblib")
                saved.append(k)
            except Exception as e:
                failed[k] = f"{type(e).__name__}: {e}"
        if saved:
            files["preprocessors"] = ",".join(sorted(saved))
        if failed:
            files["preprocessors_error"] = json.dumps(failed)
            print(f"[persist] WARNING: preprocessors not saved: {failed}",
                  flush=True)

    # -- layer 5: context ---------------------------------------------------
    write_json(d / "manifest.json", {
        "experiment": experiment, "variant": variant, "task": task,
        "split": split, "seed": seed,
        "files": files, "params": params or {}, "timings": timings or {},
        "metrics": {k: v for k, v in (metrics or {}).items()
                    if not k.startswith("_")},
        "context": build_context(seed=seed, extra=extra),
    })

    _append_index(out_dir, experiment, variant, task, split, seed, metrics, params)
    print(f"[persist] {experiment}/{variant}__{task}__{split}__seed{seed}"
          f" -> {sorted(files)}", flush=True)
    return d


def per_class_report(y_true, y_pred, class_names=None) -> Dict:
    """Per-class precision/recall/F1/support + confusion matrix.

    Always saved, never only the weighted average -- the per-class table is a
    paper table in its own right (Table 10 of the paper).
    """
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    labels = sorted(set(np.asarray(y_true).tolist()) |
                    set(np.asarray(y_pred).tolist()))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels, zero_division=0)
    return {
        "labels": [int(x) for x in labels],
        "class_names": list(class_names) if class_names else
                       [str(x) for x in labels],
        "precision": p.tolist(), "recall": r.tolist(),
        "f1": f1.tolist(), "support": sup.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                             labels=labels).tolist(),
    }


def _append_index(out_dir, experiment, variant, task, split, seed,
                  metrics, params) -> None:
    """Maintain runs_index.csv so aggregation is a one-line DataFrame read."""
    idx = Path(out_dir) / experiment / "runs_index.csv"
    row = {"experiment": experiment, "variant": variant, "task": task,
           "split": split, "seed": seed,
           "n_params": (params or {}).get("total")}
    for k, v in (metrics or {}).items():
        if not k.startswith("_") and np.isscalar(v):
            row[k] = v
    df = pd.DataFrame([row])
    if idx.exists():
        df = pd.concat([pd.read_csv(idx), df], ignore_index=True)
        df = df.drop_duplicates(subset=["experiment", "variant", "task",
                                        "split", "seed"], keep="last")
    idx.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(idx, index=False)


# =============================================================================
# Loading (for aggregation -- no experiment is ever re-run to make a table)
# =============================================================================
def load_index(out_dir, experiment: str) -> pd.DataFrame:
    p = Path(out_dir) / experiment / "runs_index.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_predictions(out_dir, experiment: str, variant="main",
                     task="multiclass", split="random", seed=42) -> Dict:
    d = (Path(out_dir) / experiment / "runs" /
         f"{variant}__{task}__{split}__seed{seed}")
    z = np.load(d / "predictions.npz")
    return {k: z[k] for k in z.files}


def load_all_runs(out_dir, experiment: str) -> list:
    root = Path(out_dir) / experiment / "runs"
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        m = d / "manifest.json"
        if m.exists():
            out.append(json.loads(m.read_text(encoding="utf-8")))
    return out


# =============================================================================
# Completeness gate -- call BEFORE releasing the Grid5000 allocation
# =============================================================================
REQUIRED_LAYERS = ("metrics", "predictions", "curves", "model")


def check_completeness(out_dir, experiment: Optional[str] = None,
                       required: Iterable[str] = REQUIRED_LAYERS,
                       strict: bool = False) -> pd.DataFrame:
    """Report which run cells are missing which artefact layer.

    Run this before the cluster allocation is released, so that any gap can be
    fixed while the cluster is still reserved.
    """
    out_dir = Path(out_dir)
    exps = ([experiment] if experiment
            else [p.name for p in out_dir.iterdir() if p.is_dir()])
    rows = []
    for exp in exps:
        for man in load_all_runs(out_dir, exp):
            present = set(man.get("files", {}))
            missing = [l for l in required if l not in present]
            rows.append({"experiment": exp, "variant": man["variant"],
                         "task": man["task"], "split": man["split"],
                         "seed": man["seed"],
                         "present": ",".join(sorted(present)),
                         "missing": ",".join(missing), "complete": not missing})
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[completeness] no runs found under {out_dir}")
        return df

    n_bad = int((~df["complete"]).sum())
    print(f"\n{'='*70}\nARTEFACT COMPLETENESS -- {len(df)} run cells\n{'='*70}")
    for _, r in df.iterrows():
        flag = "OK " if r["complete"] else "GAP"
        print(f"  [{flag}] {r['experiment']}/{r['variant']}__{r['task']}"
              f"__{r['split']}__seed{r['seed']}   missing: {r['missing'] or '-'}")
    if n_bad:
        msg = (f"\n  {n_bad} incomplete cell(s). FIX BEFORE RELEASING THE "
               f"ALLOCATION -- regenerating any of these means re-running the job.")
        print(msg)
        if strict:
            raise RuntimeError(msg)
    else:
        print("\n  All cells complete. Safe to release the allocation.")

    write_json(out_dir / "_completeness_report.json", rows)
    return df


# =============================================================================
# Notebook dependency guards
# =============================================================================
# Notebooks form a DAG: 04 needs 00's QCNN predictions for the paired McNemar
# tests, 13 needs everything. On a cluster the failure mode is a KeyError or a
# FileNotFoundError hundreds of lines into a job that has already burned an
# hour of walltime. These guards fail in the FIRST cell with a message that
# says which notebook to run first.

NOTEBOOK_DEPENDENCIES = {
    "nb00_verify":     [],
    "nb01_data":       [],
    "nb02_binary":     [],
    "nb03_multiclass": [],
    "nb04_baselines":  ["nb00_verify"],   # McNemar needs the QCNN predictions
    "nb05_ablation":   [],
    "nb06_sweep":      [],
    "nb07_robustness": [],
    "nb08_shots":      [],
    "nb09_xai":        [],
    "nb10_audit":      [],
    "nb11_cost":       [],
    "nb12_cross":      [],
    "nb14_quantum":    ["nb00_verify"],   # McNemar needs the QCNN predictions
    "nb15_gap":        [],
    "nb16_splits":     [],
    "nb13_aggregate":  ["nb00_verify"],   # everything else is optional here
}

NOTEBOOK_FILES = {
    "nb00_verify": "00_verify_archive.ipynb",
    "nb01_data": "01_data_preparation.ipynb",
    "nb02_binary": "02_qcnn_binary.ipynb",
    "nb03_multiclass": "03_qcnn_multiclass.ipynb",
    "nb04_baselines": "04_baselines_multiseed.ipynb",
    "nb05_ablation": "05_ablation.ipynb",
    "nb06_sweep": "06_qubit_depth_sweep.ipynb",
    "nb07_robustness": "07_robustness_adversarial.ipynb",
    "nb08_shots": "08_shots_and_noise.ipynb",
    "nb09_xai": "09_shap_explainability.ipynb",
    "nb10_audit": "10_audit_log.ipynb",
    "nb11_cost": "11_grid5000_timing_energy.ipynb",
    "nb12_cross": "12_cross_dataset_wustl.ipynb",
    "nb14_quantum": "14_quantum_related_baselines.ipynb",
    "nb15_gap": "15_multiclass_gap.ipynb",
    "nb16_splits": "16_split_protocols.ipynb",
    "nb13_aggregate": "13_results_aggregation.ipynb",
}


def require_experiment(out_dir, experiment: str, hint: str = "") -> None:
    """Fail immediately if a prerequisite experiment has produced nothing."""
    root = Path(out_dir) / experiment
    runs = root / "runs"
    ok = runs.exists() and any(
        (d / "manifest.json").exists() for d in runs.iterdir() if d.is_dir())
    if not ok:
        nb = NOTEBOOK_FILES.get(experiment, experiment)
        raise FileNotFoundError(
            f"\nMISSING PREREQUISITE: '{experiment}' has no completed runs under "
            f"{root}.\n"
            f"  Run  {nb}  first.\n"
            + (f"  {hint}\n" if hint else "")
        )


def require_archive(archive_dir) -> None:
    """Fail immediately if the archived Grid5000 artefacts are not reachable.

    On the cluster the package is usually staged into $HOME, so the default
    relative path can miss. Setting ARCHIVE_DIR fixes it.
    """
    p = Path(archive_dir)
    needed = ["prepared_data.npz", "scaler.joblib",
              "hybrid_qcnn_binary (1).pt", "hybrid_qcnn_multiclass (1).pt"]
    missing = [f for f in needed if not (p / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"\nARCHIVE NOT FOUND at {p}\n"
            f"  missing: {missing}\n"
            f"  Set ARCHIVE_DIR to the folder holding salvaged_g5k_artifacts, e.g.\n"
            f"    export ARCHIVE_DIR=$HOME/qmedshield_g5k/salvaged_g5k_artifacts\n"
        )
    print(f"[deps] archive OK: {p}")


def require_packages(*names: str) -> None:
    """Fail immediately if an optional dependency is absent."""
    missing = []
    for n in names:
        try:
            __import__(n)
        except ImportError:
            missing.append(n)
    if missing:
        raise ImportError(
            f"\nMISSING PACKAGES: {missing}\n  pip install {' '.join(missing)}\n")
    print(f"[deps] packages OK: {list(names)}")


def preflight(out_dir, archive_dir, experiment: str,
              packages: Sequence[str] = (), needs_data: bool = False,
              data_dir=None) -> None:
    """One call at the top of every notebook: verify everything it needs.

    Cheap, and it converts a mid-job crash on the cluster into an immediate,
    actionable message.
    """
    print(f"[preflight] {experiment}")
    for dep in NOTEBOOK_DEPENDENCIES.get(experiment, []):
        require_experiment(out_dir, dep)
        print(f"[deps] prerequisite OK: {dep}")
    if archive_dir is not None:
        require_archive(archive_dir)
    if packages:
        require_packages(*packages)
    if needs_data and data_dir is not None:
        csvs = list(Path(data_dir).glob("*.csv"))
        print(f"[deps] raw CSVs in {data_dir}: {len(csvs)}"
              + ("" if csvs else "  (split protocols will be skipped)"))
    print("[preflight] all checks passed\n")
