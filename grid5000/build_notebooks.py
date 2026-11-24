"""
build_notebooks.py -- generate every standalone notebook.

    python build_notebooks.py

Each notebook inlines the whole `src/`
bootstrap, so it has NO `src/` dependency and can be dropped on Grid5000 by
itself. Edit the originals in `src/`, then re-run this script.

Every notebook:
  * is headed by the question it answers (a short Purpose section),
  * keeps the original paper's architecture and data pipeline intact,
  * saves all five artefact layers through src/persistence.py,
  * ends with check_completeness(), so a missing layer is caught BEFORE the
    compute allocation is released.
"""
from __future__ import annotations

import json
from pathlib import Path

import nb_defs
import nb_defs_cross    # noqa: F401  -- registers notebook 12
import nb_defs_quantum  # noqa: F401  -- registers notebook 14
import nb_defs_gap      # noqa: F401  -- registers notebook 15
import nb_defs_splits   # noqa: F401  -- registers notebook 16

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
OUT = HERE / "notebooks"

# Inlined into every notebook, in dependency order.
BOOTSTRAP_MODULES = ["config", "persistence", "circuits", "models", "train",
                     "load_g5k", "audit", "perturb", "stats", "plots",
                     "baselines", "xai", "cost", "qbaselines", "hierarchical",
                     "splits"]

LOCAL = tuple(BOOTSTRAP_MODULES)

# Preflight configuration per notebook:
#   (experiment key it writes, needs the archive, extra packages, wants raw CSVs)
PREFLIGHT = {
    "00_verify_archive.ipynb":         ("nb00_verify",     True,  (), False),
    "01_data_preparation.ipynb":       ("nb01_data",       True,  (), True),
    "02_qcnn_binary.ipynb":            ("nb02_binary",     True,  (), False),
    "03_qcnn_multiclass.ipynb":        ("nb03_multiclass", True,  (), False),
    "04_baselines_multiseed.ipynb":    ("nb04_baselines",  True,  (), False),
    "05_ablation.ipynb":               ("nb05_ablation",   True,  (), False),
    "06_qubit_depth_sweep.ipynb":      ("nb06_sweep",      True,  (), False),
    "07_robustness_adversarial.ipynb": ("nb07_robustness", True,  ("scipy",), False),
    "08_shots_and_noise.ipynb":        ("nb08_shots",      True,  (), False),
    "09_shap_explainability.ipynb":    ("nb09_xai",        True,  ("shap", "lime"), False),
    "10_audit_log.ipynb":              ("nb10_audit",      False, (), False),
    "11_grid5000_timing_energy.ipynb": ("nb11_cost",       True,  (), False),
    "12_cross_dataset_wustl.ipynb":    ("nb12_cross",      True,  (), False),
    "13_results_aggregation.ipynb":    ("nb13_aggregate",  False, (), False),
    # Without these three the fallback derives the experiment key from the file
    # name ("14", "15", "16"), so preflight checks the wrong experiment and the
    # nb00 prerequisite of notebook 14 goes unchecked.
    "14_quantum_related_baselines.ipynb": ("nb14_quantum",  True,  (), False),
    "15_multiclass_gap.ipynb":         ("nb15_gap",        True,  ("scipy",), False),
    # needs_data stays False for 16: the notebook reports missing CSVs itself,
    # in writing, instead of failing preflight.
    "16_split_protocols.ipynb":        ("nb16_splits",     True,  ("scipy",), False),
}


def build_bootstrap() -> str:
    parts = [
        "from __future__ import annotations",
        "# " + "=" * 66,
        "# STANDALONE BOOTSTRAP -- every src/ module inlined.",
        "# This notebook has NO src/ dependency: drop it on Grid5000 and run it.",
        "# Edit the originals in src/, then re-run build_notebooks.py",
        "# " + "=" * 66,
        "",
        "import os, sys, time, json, platform, random, warnings",
        "warnings.filterwarnings('ignore')",
        "import matplotlib",
        "if not os.environ.get('DISPLAY') and not os.environ.get('JPY_PARENT_PID'):",
        "    matplotlib.use('Agg')",
        "    print('[setup] no display -> Agg backend (figures saved, not shown)')",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "",
    ]

    for mod in BOOTSTRAP_MODULES:
        path = SRC / f"{mod}.py"
        if not path.exists():
            print(f"  ! missing src/{mod}.py -- skipped")
            continue

        lines, in_local_import = [], False
        for ln in path.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if s.startswith("from __future__"):
                continue
            # Multi-line imports must be commented through their closing paren,
            # or the continuation lines become an IndentationError.
            if in_local_import:
                lines.append("# " + ln + "   # inlined")
                if ")" in s:
                    in_local_import = False
                continue
            if any(s.startswith(f"from {m} import") or s == f"import {m}"
                   for m in LOCAL):
                lines.append("# " + ln + "   # inlined")
                if "(" in s and ")" not in s:
                    in_local_import = True
                continue
            lines.append(ln)

        parts += ["# " + "=" * 66,
                  f"# === Inlined from src/{mod}.py ===",
                  "# " + "=" * 66,
                  "\n".join(lines), ""]

    parts += [
        "print(summary())",
        "OUT_DIR.mkdir(parents=True, exist_ok=True)",
        "(ROOT / 'figures').mkdir(parents=True, exist_ok=True)",
        "",
        "# --- PREFLIGHT ------------------------------------------------------",
        "# Verify every prerequisite BEFORE any compute. On the cluster this",
        "# turns a crash deep inside an hour-long job into an immediate message",
        "# naming the notebook that has to run first.",
        "preflight(OUT_DIR, ARCHIVE_DIR if __NEEDS_ARCHIVE__ else None,",
        "          __EXPERIMENT__, packages=__PACKAGES__,",
        "          needs_data=__NEEDS_DATA__, data_dir=DATA_DIR)",
    ]
    return "\n".join(parts)


# Editable knobs, rendered as the FIRST cell of every notebook, so the suite can
# be driven entirely from JupyterLab -- no shell, no environment variables.
SETTINGS_CELL = """# =====================================================================
# SETTINGS -- edit this cell, then Run All.  "" means "use the default".
# This cell must run BEFORE the bootstrap cell below it.
# =====================================================================
import os

# Printed first, so every log says which build produced it. Quote this line
# when reporting a problem: an old notebook left on the cluster is otherwise
# indistinguishable from a bug in the current one.
print("build:", "__BUILD_ID__")

os.environ["SHARD"]            = ""   # "3/8"   run only shard 3 of 8    (05,06,14,15,16)
os.environ["SEEDS"]            = ""   # "42,1"  only these seeds         (every notebook)
os.environ["SEEDS_ABLATION"]   = ""   # "42,1,7" ablation seeds          (05)
os.environ["SWEEP_EPOCHS"]     = ""   # "40"    cheaper qubit sweep      (06)
os.environ["ABLATION_SAMPLES"] = ""   # "10000" smaller ablation budget  (05)
os.environ["QUICK_MODE"]       = ""   # "1"     fast smoke pass -- NOT results
os.environ["KWOLLECT"]         = ""   # "1"     wattmeter energy         (11, exclusive node)
os.environ["WUSTL"]            = ""   # "1"     optional second dataset  (12)

# Paths -- only if these folders are NOT beside this notebook:
# os.environ["ARCHIVE_DIR"] = "/home/USER/salvaged_g5k_artifacts"
# os.environ["DATA_DIR"]    = "/home/USER/data"
# os.environ["OUT_DIR"]     = "/home/USER/outputs"

_on = {k: v for k, v in os.environ.items() if v and k in (
    "SHARD", "SEEDS", "SEEDS_ABLATION", "SWEEP_EPOCHS", "ABLATION_SAMPLES",
    "QUICK_MODE", "KWOLLECT", "WUSTL", "ARCHIVE_DIR", "DATA_DIR", "OUT_DIR")}
print("settings:", _on if _on else "all defaults")
"""


HEADER = """
# {title}

## Purpose

{remark}

{purpose}

---
### Persistence contract

Every result below is written to `outputs/` in five layers -- **metrics**, per-sample
**predictions**, **curves** (every value behind every plotted line, as CSV *and* NPZ),
**model** weights with fitted preprocessors, and **context** (config, seed, versions,
timings, OAR job id).

So a new table, a new curve, an extra statistic or a different figure style can all be
produced later **without re-running any of this code**. The final cell audits completeness;
run it before releasing the Grid5000 allocation.
"""


def write(name: str, meta: dict, bootstrap: str) -> None:
    exp, needs_arch, pkgs, needs_data = PREFLIGHT.get(
        name, (name.split("_")[0], True, (), False))
    settings = SETTINGS_CELL.replace("__BUILD_ID__", BUILD_ID)
    boot = (bootstrap
            .replace("__EXPERIMENT__", repr(exp))
            .replace("__NEEDS_ARCHIVE__", repr(needs_arch))
            .replace("__PACKAGES__", repr(tuple(pkgs)))
            .replace("__NEEDS_DATA__", repr(needs_data)))

    cells = [
        {"cell_type": "markdown", "metadata": {},
         "source": HEADER.format(title=meta["title"], remark=meta["remark"],
                                 purpose=meta["purpose"]).strip("\n").splitlines(True)},
        # Editable settings, FIRST, so a notebook run straight from JupyterLab
        # needs no shell and no environment variables.
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": settings.splitlines(True)},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": boot.splitlines(True)},
    ] + meta["cells"]

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.x"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")

    # Compile each cell SEPARATELY, the way Jupyter does. Concatenating them
    # would flag the bootstrap's `from __future__ import annotations` as
    # misplaced merely because the settings cell precedes it -- a false alarm
    # about code that runs correctly.
    status = "OK"
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        try:
            compile("".join(c["source"]), f"{name}[cell {i}]", "exec")
        except SyntaxError as e:
            status = f"SYNTAX ERROR cell {i} line {e.lineno}: {e.msg}"
            break
    print(f"  {name:38s} {meta['needs']:9s} {len(cells):2d} cells  [{status}]")


if __name__ == "__main__":
    print("building standalone notebooks...\n")
    boot = build_bootstrap()
    print(f"  bootstrap: {len(boot):,} chars from {len(BOOTSTRAP_MODULES)} modules\n")
    import datetime as _dt, hashlib as _hl
    BUILD_ID = (_dt.datetime.now().strftime("%Y-%m-%d %H:%M") + " "
                + _hl.sha256(boot.encode()).hexdigest()[:8])
    print("  build id :", BUILD_ID, "\\n")
    for name in sorted(nb_defs.NOTEBOOKS):
        write(name, nb_defs.NOTEBOOKS[name], boot)

    print(f"\n{len(nb_defs.NOTEBOOKS)} notebooks written to notebooks/")
    print("\nRun order:")
    print("  00  verify the archive          <- ALWAYS FIRST (regression guard)")
    print("  01  data + splits               (needs raw CSVs for split protocols)")
    print("  02/03 headline training         (cluster; QUICK_MODE=1 for a fast pass)")
    print("  04  baselines + matched         (refit is minutes; matched needs cluster)")
    print("  05  ablation                    (cluster)")
    print("  06  qubit/depth sweep           (cluster)")
    print("  07  robustness                  (archive only -- no cluster)")
    print("  08  shots + noise               (archive only)")
    print("  09  explainability              (archive only; pip install shap lime)")
    print("  10  audit log                   (nothing needed at all)")
    print("  11  cost + energy               (archive only)")
    print("  12  cross-dataset WUSTL         (optional)")
    print("  14  quantum comparators         (cluster; QSVM rows need no training)")
    print("  15  multiclass gap              (steps 1-3 archive only; 4-5 cluster)")
    print("  16  split protocols             (cluster; NEEDS the raw CSVs)")
    print("  13  aggregate -> paper tables   (reads outputs/ only; run LAST)")
