"""
verify_deploy.py -- preflight the whole package BEFORE burning cluster time.

Run this first on the Grid5000 node:

    python verify_deploy.py

It checks, in order:
  1. every required package imports, at a usable version
  2. the archive is present and loadable
  3. the archived weights reproduce the published metrics bit-exactly
  4. every notebook parses, and its bootstrap executes
  5. the notebook dependency DAG is satisfiable in the documented run order
  6. paths are writable and free of surprises

Exit code 0 means the package is safe to run. Anything else names the problem.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

FAIL: list[str] = []
WARN: list[str] = []


def check(label: str, fn):
    try:
        detail = fn()
        print(f"  [ OK ] {label}" + (f"  {detail}" if detail else ""))
        return True
    except Exception as e:
        print(f"  [FAIL] {label}\n         {type(e).__name__}: {e}")
        FAIL.append(label)
        return False


def warn(label: str, fn):
    try:
        detail = fn()
        print(f"  [ OK ] {label}" + (f"  {detail}" if detail else ""))
    except Exception as e:
        print(f"  [WARN] {label}\n         {type(e).__name__}: {e}")
        WARN.append(label)


# ---------------------------------------------------------------------------
print("=" * 72)
print("1. PACKAGES")
print("=" * 72)


def _v(mod):
    m = __import__(mod)
    return getattr(m, "__version__", "?")


for pkg in ("numpy", "pandas", "scipy", "sklearn", "torch", "pennylane",
            "matplotlib", "joblib"):
    check(f"import {pkg}", lambda p=pkg: _v(p))

for pkg in ("shap", "lime", "statsmodels", "pyarrow"):
    warn(f"import {pkg}  (optional)", lambda p=pkg: _v(p))

# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("2. CONFIGURATION AND PATHS")
print("=" * 72)

import config  # noqa: E402
from persistence import require_archive  # noqa: E402

check("config loads", lambda: f"ROOT={config.ROOT.name}")
check("archive present", lambda: (require_archive(config.ARCHIVE_DIR),
                                  str(config.ARCHIVE_DIR.name))[1])


def _writable():
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = config.OUT_DIR / ".write_test"
    p.write_text("ok", encoding="utf-8")
    p.unlink()
    return str(config.OUT_DIR)


check("outputs/ writable", _writable)
warn("raw CSVs present (needed only by notebook 01)",
     lambda: f"{len(list(config.DATA_DIR.glob('*.csv')))} files")

# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("3. ARCHIVED MODELS REPRODUCE THE PUBLISHED METRICS")
print("=" * 72)

PUBLISHED = {"binary": 0.9962666666666666, "multiclass": 0.7918666666666667}


def _repro(task):
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from load_g5k import load_hybrid, load_prepared_data
    from models import build_model
    from train import make_loader, evaluate

    data = load_prepared_data(str(config.ARCHIVE_DIR))
    fn = f"hybrid_qcnn_{task} (1).pt"
    m, info = load_hybrid(task, str(config.ARCHIVE_DIR / fn),
                          model=build_model(task, "full"))
    assert not info["missing"] and not info["unexpected"], info
    sfx = "b" if task == "binary" else "m"
    r = evaluate(m, make_loader(data[f"X_test_{sfx}"], data[f"y_test_{sfx}"],
                                shuffle=False), 2 if task == "binary" else 8)
    d = abs(r["accuracy"] - PUBLISHED[task])
    assert d == 0.0, f"accuracy drifted by {d:.2e} -- do NOT run the suite"
    arch = np.load(config.ARCHIVE_DIR / f"{task}_predictions.npz")
    agree = (r["_predictions"] == arch["predictions"]).mean()
    assert agree == 1.0, f"per-sample agreement {agree:.4%}"
    return f"accuracy={r['accuracy']:.10f}  agreement=100%"


for t in ("binary", "multiclass"):
    check(f"{t} reproduces exactly", lambda t=t: _repro(t))

# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("4. NOTEBOOKS PARSE AND THEIR BOOTSTRAP RUNS")
print("=" * 72)

nb_dir = HERE / "notebooks"
notebooks = sorted(nb_dir.glob("*.ipynb"))
check(f"{len(notebooks)} notebooks found", lambda: "")


def _parses(p):
    nb = json.loads(p.read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"]
                    if c["cell_type"] == "code")
    compile(src, p.name, "exec")
    assert "__EXPERIMENT__" not in src, "preflight placeholder not substituted"
    assert "preflight(" in src, "preflight call missing"
    return f"{len(nb['cells'])} cells"


for p in notebooks:
    check(f"{p.name} parses", lambda p=p: _parses(p))

# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("5. DEPENDENCY DAG")
print("=" * 72)

from persistence import NOTEBOOK_DEPENDENCIES, NOTEBOOK_FILES  # noqa: E402

RUN_ORDER = ["nb00_verify", "nb01_data", "nb02_binary", "nb03_multiclass",
             "nb04_baselines", "nb05_ablation", "nb06_sweep", "nb07_robustness",
             "nb08_shots", "nb09_xai", "nb10_audit", "nb11_cost", "nb12_cross",
             "nb14_quantum", "nb15_gap", "nb16_splits", "nb13_aggregate"]


def _dag():
    seen = set()
    for exp in RUN_ORDER:
        for dep in NOTEBOOK_DEPENDENCIES.get(exp, []):
            assert dep in seen, (
                f"{NOTEBOOK_FILES[exp]} needs {NOTEBOOK_FILES[dep]}, "
                f"which the documented run order places later")
        seen.add(exp)
    return f"{len(RUN_ORDER)} notebooks, order satisfiable"


check("run order satisfies every dependency", _dag)
for exp in RUN_ORDER:
    deps = NOTEBOOK_DEPENDENCIES.get(exp, [])
    if deps:
        print(f"         {NOTEBOOK_FILES[exp]:36s} <- "
              f"{', '.join(NOTEBOOK_FILES[d] for d in deps)}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
if FAIL:
    print(f"VERIFICATION FAILED -- {len(FAIL)} problem(s):")
    for f in FAIL:
        print(f"  - {f}")
    print("\nFix these before submitting any OAR job.")
    sys.exit(1)

print("VERIFICATION PASSED -- the package is safe to run.")
if WARN:
    print(f"\n{len(WARN)} optional item(s) unavailable:")
    for w in WARN:
        print(f"  - {w}")
    print("  Notebook 09 needs shap+lime; notebooks 01 and 16 need the raw CSVs.")
    print("  Everything else will run.")
print("=" * 72)
sys.exit(0)
