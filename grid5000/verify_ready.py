"""
verify_ready.py -- preflight this flat notebook package BEFORE burning walltime.

    python verify_ready.py

Checks, in order:
  1. every required package imports, at a usable version
  2. the archive is present and complete
  3. the archived weights still reproduce the published metrics BIT-EXACTLY
  4. every notebook parses and its inlined bootstrap compiles
  5. the documented run order satisfies the only two dependencies that exist
  6. outputs/, figures/ and logs/ are writable

Exit code 0 means it is safe to submit. Anything else names the problem.

The notebooks are standalone -- every helper is inlined in their first cell --
so this script deliberately does NOT import a src/ package. It rebuilds what it
needs by executing notebook 00's bootstrap cell, which is exactly the code the
notebooks themselves will run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAIL: list[str] = []
WARN: list[str] = []

PUBLISHED = {"binary": 0.9962666666666666, "multiclass": 0.7918666666666667}


def check(label, fn):
    try:
        detail = fn()
        print(f"  [ OK ] {label}" + (f"  {detail}" if detail else ""))
        return True
    except Exception as e:
        print(f"  [FAIL] {label}\n         {type(e).__name__}: {e}")
        FAIL.append(label)
        return False


def warn(label, fn):
    try:
        detail = fn()
        print(f"  [ OK ] {label}" + (f"  {detail}" if detail else ""))
    except Exception as e:
        print(f"  [WARN] {label}\n         {type(e).__name__}: {e}")
        WARN.append(label)


print("=" * 72)
print("1. PACKAGES")
print("=" * 72)


def _v(mod):
    m = __import__(mod)
    return getattr(m, "__version__", "?")


for mod in ("numpy", "pandas", "scipy", "sklearn", "torch", "pennylane",
            "statsmodels", "matplotlib", "joblib"):
    check(f"import {mod}", lambda m=mod: _v(m))
for mod in ("shap", "lime", "seaborn", "pyarrow", "requests"):
    warn(f"import {mod}  (optional)", lambda m=mod: _v(m))

print("\n" + "=" * 72)
print("2. ARCHIVE")
print("=" * 72)

ARCHIVE = Path(__import__("os").environ.get("ARCHIVE_DIR",
                                            HERE / "salvaged_g5k_artifacts"))
NEEDED = ["prepared_data.npz", "scaler.joblib", "pca_multi.joblib",
          "hybrid_qcnn_binary (1).pt", "hybrid_qcnn_multiclass (1).pt",
          "binary_test_metrics.json", "multiclass_test_metrics.json",
          "binary_predictions.npz", "multiclass_predictions.npz"]


def _archive():
    missing = [f for f in NEEDED if not (ARCHIVE / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing from {ARCHIVE}: {missing}")
    return f"{ARCHIVE}"


check("archive complete", _archive)

def _csvs():
    found = sorted((HERE / "data").glob("*.csv")) if (HERE / "data").exists() else []
    if not found:
        raise FileNotFoundError(
            "no CSVs in data/; notebooks 01 and 16 will report the session, "
            "device and chronological protocols and the enlarged "
            "test set as unavailable, and say why")
    return f"{len(found)} files in data/"


warn("raw CICIoMT2024 CSVs (notebooks 01 and 16 only)", _csvs)


print("\n" + "=" * 72)
print("3. BIT-EXACT REPRODUCTION")
print("=" * 72)

# Dated copies (…__FIXED_YYYY-MM-DD.ipynb) are convenience duplicates for
# moving files around; they hold the same code as the canonical notebook and
# are not part of the run order.
NOTEBOOKS = sorted(p for p in HERE.glob("*.ipynb") if "__FIXED_" not in p.name)


def _bootstrap_namespace():
    """Execute notebook 00's inlined bootstrap and return its globals."""
    nb = json.loads((HERE / "00_verify_archive.ipynb").read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    # "__main__" and not a private name: @dataclass resolves its module through
    # sys.modules[cls.__module__], and a name that is not registered there fails
    # with an opaque AttributeError on None.
    ns: dict = {"__name__": "__main__"}
    # Cell 0 is the editable SETTINGS cell, cell 1 the inlined bootstrap.
    # Run both, in order, exactly as JupyterLab does -- the settings cell is
    # what makes the suite drivable without a shell, so verify it too.
    for _k, _cell in enumerate(code_cells[:2]):
        exec(compile("".join(_cell["source"]), f"cell{_k}", "exec"), ns)
    return ns


NS = None
if not FAIL:
    def _boot():
        global NS
        NS = _bootstrap_namespace()
        return f"{len(NS)} names"

    check("notebook 00 bootstrap executes", _boot)


def _repro(task, weights, sfx, nc):
    data = NS["load_prepared_data"](str(ARCHIVE))
    model, _ = NS["load_hybrid"](task, str(ARCHIVE / weights),
                                 model=NS["build_model"](task, "full"))
    r = NS["evaluate"](model, NS["make_loader"](data[f"X_test_{sfx}"],
                                                data[f"y_test_{sfx}"],
                                                shuffle=False), nc)
    delta = abs(r["accuracy"] - PUBLISHED[task])
    if delta > 0:
        raise AssertionError(
            f"accuracy drifted by {delta:.2e} ({r['accuracy']:.10f} vs "
            f"published {PUBLISHED[task]:.10f}) -- the environment does not "
            f"match the one the weights were trained in. DO NOT run the suite.")
    return f"accuracy {r['accuracy']:.10f}  delta 0"


if NS is not None:
    check("binary reproduces exactly",
          lambda: _repro("binary", "hybrid_qcnn_binary (1).pt", "b", 2))
    check("multiclass reproduces exactly",
          lambda: _repro("multiclass", "hybrid_qcnn_multiclass (1).pt", "m", 8))

print("\n" + "=" * 72)
print("4. NOTEBOOKS")
print("=" * 72)

check(f"{len(NOTEBOOKS)} notebooks found", lambda: "")
for p in NOTEBOOKS:
    def _parse(p=p):
        nb = json.loads(p.read_text(encoding="utf-8"))
        # Per cell, as Jupyter compiles them: concatenating would misreport
        # the bootstrap's `from __future__` import as misplaced now that the
        # editable settings cell precedes it.
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                compile("".join(c["source"]), f"{p.name}[cell {i}]", "exec")
        return f"{len(nb['cells'])} cells"

    check(f"{p.name} parses", _parse)

print("\n" + "=" * 72)
print("5. RUN ORDER")
print("=" * 72)

RUN_ORDER = ["00_verify_archive", "01_data_preparation", "02_qcnn_binary",
             "03_qcnn_multiclass", "04_baselines_multiseed", "05_ablation",
             "06_qubit_depth_sweep", "07_robustness_adversarial",
             "08_shots_and_noise", "09_shap_explainability", "10_audit_log",
             "11_grid5000_timing_energy", "12_cross_dataset_wustl",
             "14_quantum_related_baselines", "15_multiclass_gap",
             "16_split_protocols", "13_results_aggregation"]

# The only dependencies that exist: 04 and 14 need 00's QCNN predictions for the
# paired McNemar tests, and 13 aggregates, so it runs last.
DEPENDS = {"04_baselines_multiseed": ["00_verify_archive"],
           "14_quantum_related_baselines": ["00_verify_archive"],
           "13_results_aggregation": ["00_verify_archive"]}


def _dag():
    seen = set()
    for nb in RUN_ORDER:
        for dep in DEPENDS.get(nb, []):
            assert dep in seen, f"{nb} needs {dep}, which runs later"
        seen.add(nb)
    on_disk = {p.stem for p in NOTEBOOKS}
    missing = [nb for nb in RUN_ORDER if nb not in on_disk]
    assert not missing, f"listed in the run order but absent: {missing}"
    # A notebook MISSING from the folder is fatal. An extra one is a copy the
    # user made while moving files -- worth naming, since a stale copy is how
    # the wrong build gets run, but not a reason to block the suite.
    extra = [nb for nb in sorted(on_disk) if nb not in RUN_ORDER]
    if extra:
        print(f"         NOTE: extra notebook(s) present, not part of the run "
              f"order: {extra}")
        print(f"         Check their build stamp before running them, or "
              f"delete them to avoid confusion.")
    return f"{len(RUN_ORDER)} notebooks, order satisfiable"


check("run order satisfies every dependency", _dag)
assert RUN_ORDER[-1] == "13_results_aggregation"
print("         13_results_aggregation runs LAST -- it aggregates the rest")

print("\n" + "=" * 72)
print("6. WRITABLE PATHS")
print("=" * 72)


def _writable(name):
    d = HERE / name
    d.mkdir(exist_ok=True)
    probe = d / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    return str(d)


for name in ("outputs", "figures", "logs"):
    check(f"{name}/ writable", lambda n=name: _writable(n))

print("\n" + "=" * 72)
if FAIL:
    print(f"VERIFICATION FAILED -- {len(FAIL)} problem(s):")
    for f in FAIL:
        print(f"  - {f}")
    print("\nFix these before submitting any job.")
    sys.exit(1)

print("VERIFICATION PASSED -- the package is safe to run.")
if WARN:
    print(f"\n{len(WARN)} optional item(s) unavailable:")
    for w in WARN:
        print(f"  - {w}")
    print("  Notebook 09 needs shap+lime; notebooks 01 and 16 need the raw CSVs.")
    print("  Everything else will run.")
print("=" * 72)
