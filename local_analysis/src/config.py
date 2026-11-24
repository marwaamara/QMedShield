"""
config.py -- single source of truth for the experimental configuration.

Standalone notebooks, QUICK_MODE, environment overrides and comprehensive
persistence are all driven from this file.

IMPORTANT -- experimental design
--------------------------------
The experimental design is the one in the paper (79.19% multiclass accuracy,
8-qubit circuit, SHA-256 audit log):

    classes             8 (7 attacks + benign)
    PCA components      16
    quantum layers      4
    sample budget       50,000
    baselines           RF/GB/SVM/KNN/MLP
    tasks               binary + multiclass

Every constant below reproduces the paper, verified bit-exact against the
archived Grid5000 weights.

Environment overrides
---------------------
QUICK_MODE=1    switch every slow knob to its iteration-friendly value
DATA_DIR=/p     override the data directory
OUT_DIR=/p      override the outputs directory
"""
from __future__ import annotations

import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
# ROOT defaults to the package root, NOT the cwd, so data/ and outputs/ resolve
# identically whether the code is launched from the package root, from
# notebooks/, or by an OAR job in $HOME.
#
# `__file__` is undefined once this module is INLINED into a standalone
# notebook, which is the normal way these run on Grid5000 -- so fall back to
# walking up from the working directory.
def _find_pkg_root() -> Path:
    try:
        return Path(__file__).resolve().parent.parent      # running from src/
    except NameError:
        pass
    here = Path.cwd().resolve()
    for cand in (here, *here.parents):
        if (cand / "src").is_dir() or (cand / "notebooks").is_dir():
            return cand
    return here


_PKG_ROOT = _find_pkg_root()
ROOT      = Path(os.environ.get("ROOT", _PKG_ROOT))
DATA_DIR  = Path(os.environ.get("DATA_DIR", ROOT / "data"))
OUT_DIR   = Path(os.environ.get("OUT_DIR",  ROOT / "outputs"))
LOG_DIR   = ROOT / "logs"
PAPER_DIR = OUT_DIR / "paper"

# Archived Grid5000 artefacts (trained weights + fitted preprocessing).
# These let every INFERENCE-ONLY notebook run with no retraining at all.
#
# Resolution order matters. The BUNDLED copy must win, so that a deployed
# package is genuinely self-contained: if the sibling development folder were
# preferred, a deploy folder would silently read the archive from outside
# itself and would then break the moment it is copied to the cluster.
def _find_archive() -> Path:
    env = os.environ.get("ARCHIVE_DIR")
    if env:
        return Path(env)
    candidates = [
        ROOT / "salvaged_g5k_artifacts",                     # bundled (deploy)
        ROOT.parent / "code_archive" / "salvaged_g5k_artifacts",
        ROOT.parent / "salvaged_g5k_artifacts",
    ]
    for c in candidates:
        if (c / "prepared_data.npz").exists():
            return c
    return candidates[0]


ARCHIVE_DIR = _find_archive()

# =============================================================================
# Run mode
# =============================================================================
QUICK_MODE = os.environ.get("QUICK_MODE", "0") == "1"

# =============================================================================
# Seeds -- multiple random seeds, reporting mean performance together with
# standard deviations or confidence intervals.
# Seed 42 first: it is the seed of the paper's original runs, so its row stays
# directly comparable with the published tables.
# =============================================================================
SEEDS_FULL  = [42, 1, 7, 13, 2024]
SEEDS_QUICK = [42, 1, 7]
SEEDS = SEEDS_QUICK if QUICK_MODE else SEEDS_FULL

# Expensive sweeps use a reduced seed set.
SEEDS_SWEEP = SEEDS[:3]

# --- ablation seed budget ----------------------------------------------------
# MEASURED on Grid'5000: 267 s/epoch at 8 qubits on 35,000 rows, so one binary
# ablation cell is ~3.6 h and one multiclass cell ~8.7 h. The 8-variant grid is
# therefore ~98 node-hours per seed, i.e. ~490 h at five seeds -- more than the
# whole compute allocation, and spent on the LEAST informative axis:
# the binary variants already agree to within 0.001 accuracy across seeds.
#
# Three seeds still give a mean, a standard deviation and a CI for every row,
# which is what multi-seed reporting needs; five would tighten the interval by ~20% at
# 1.7x the cost. The headline models (Tables 6/9) keep all five seeds.
#
# Override with SEEDS_ABLATION=42,1,7,13,2024 if the allocation allows it.
_SA = _env_csv_early = os.environ.get("SEEDS_ABLATION", "").strip()
SEEDS_ABLATION = ([int(t) for t in _SA.split(",") if t.strip()] if _SA
                  else SEEDS[:3])

# =============================================================================
# Dataset -- CICIoMT2024 WiFi/MQTT partition
# =============================================================================
DATASET      = "CICIoMT2024-WiFi-MQTT"
N_CLASSES    = 8
CLASS_NAMES  = ["Benign", "ARP_Spoofing", "Reconnaissance", "MQTT_Malformed",
                "MQTT_DoS", "MQTT_DDoS", "DoS", "DDoS"]

POOL_SIZE    = 50_000 if not QUICK_MODE else 10_000
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
N_FEATURES   = 25
N_PCA        = 16

# --- enlarged test set -------------------------------------------------------
# The paper's per-class table reports F1 on n=4 (MQTT_Malformed) and n=15
# (ARP_Spoofing). Those estimates are not defensible. Draw a DISJOINT enlarged
# test set so every class has at least MIN_TEST_PER_CLASS samples. Inference is
# cheap, so this costs minutes and no retraining.
ENLARGED_TEST       = True
MIN_TEST_PER_CLASS  = 100
TEST_SIZE_TARGET    = 20_000

# --- evaluation protocols: random / session / device / chronological splits --
SPLIT_PROTOCOLS = ["random", "session", "device", "chrono"]
SPLIT_DEFAULT   = "random"   # headline; the others reported beside it

# =============================================================================
# Architecture -- section 3.3 of the paper
# =============================================================================
N_QUBITS  = 8
N_LAYERS  = 4
ENTANGLE_SCALES = (1, 2, 4)     # ring CNOT / skip-one CZ / cross-half CNOT
USE_REUPLOADING = True

# Encoding convention: R_Y(z_j), NO pi factor. Matches eq. (3) of the paper and
# the config that trained the archived weights. A pi-scaled variant exists in
# smote_correction_optimized_V2 and is WRONG: loading the archived weights into
# it yields 2.9% binary accuracy against the published 99.63%.
ANGLE_SCALE = 1.0

# =============================================================================
# Training -- Table 4 of the paper (verified against the archived Config)
# =============================================================================
EPOCHS_BINARY     = 50  if not QUICK_MODE else 3
EPOCHS_MULTICLASS = 120 if not QUICK_MODE else 5
LR_BINARY         = 5e-3
LR_MULTICLASS     = 2e-3
LR_MIN_BINARY     = 5e-5
LR_MIN_MULTICLASS = 2e-5
BATCH_SIZE        = 32          # Table 4 of the paper
WEIGHT_DECAY      = 1e-3
GRAD_CLIP         = 1.0
CLASS_WEIGHTING   = "sqrt_inverse"

EARLY_STOP_PATIENCE   = 15
EARLY_STOP_MIN_EPOCHS = 30

# =============================================================================
# Quantum backend -- MEASURED, not assumed (bench_backend.py, task P0.7)
#   default.qubit   + backprop    3.34 ms/sample  ->   3.9 h per multiclass run
#   lightning.qubit + adjoint    27.63 ms/sample  ->  32.2 h per multiclass run
# At 8 qubits the state vector is 256 amplitudes, so per-execution overhead
# dominates; default.qubit keeps the whole batch in one broadcast torch tensor.
# This is also what the paper reports in section 4.1.
# =============================================================================
QUANTUM_DEVICE = "default.qubit"
DIFF_METHOD    = "backprop"

# --- shots, circuit evaluation, ideal simulation -----------------------------
SHOT_SWEEP        = [128, 512, 1024, 4096, None]   # None == analytic/ideal
NOISE_DEPOL       = [0.0, 0.001, 0.005, 0.01, 0.02]
NOISE_AMP_DAMP    = [0.001, 0.005, 0.01]
SHOTS_TRAINING_SUPPORTED = False   # parameter-shift + broadcasting is unusable

# --- qubit / depth sweep (motivates the 8-qubit choice) ---------------------
# 7 configs, not a full grid. Sweeps use a reduced, IDENTICAL budget
# for every config so the comparison is fair across configurations.
SWEEP_QUBITS  = [4, 6, 8, 10, 12]     # at L = N_LAYERS
SWEEP_LAYERS  = [2, 4, 6]             # at n_q = N_QUBITS
SWEEP_SAMPLES = 10_000

# Epoch budget for the sweep, passed to train_model as `epochs_override`. It
# defaults to the multiclass schedule (roughly 3.6 h per cell), so that cells
# completed at 120 epochs stay comparable with cells trained later.
#
# For a FRESH sweep, SWEEP_EPOCHS=40 cuts the stage from ~170 to ~58 node-hours.
# Do not switch mid-sweep: mixing budgets makes the accuracy-versus-qubits
# curve meaningless, and the notebook checks for exactly that.
_SE = os.environ.get("SWEEP_EPOCHS", "").strip()
SWEEP_EPOCHS = int(_SE) if _SE else EPOCHS_MULTICLASS

# =============================================================================
# Baselines -- Table 6/7 grids of the paper, plus matched controls
# =============================================================================
BASELINES = ["Random Forest", "Gradient Boosting", "MLP", "KNN", "SVM (RBF)"]

# Family A: identical encoder/decoder scaffolding, bottleneck swapped. This is
# what isolates the quantum contribution (comparison against parameter-matched
# compact classical neural networks).
MATCHED_BOTTLENECKS = ["pqc", "linear", "fourier", "mlp96", "identity"]
# Family B: highly optimised lightweight classical deployables.
LIGHTWEIGHT_BASELINES = ["int8_mlp", "pruned_mlp", "gbdt_shallow", "rf20"]

# --- ablation variants -------------------------------------------------------
# Every variant shares the IDENTICAL encoder and decoder; only the bottleneck
# changes. Enforced by assert_scaffolding_identical() before any training.
ABLATION_VARIANTS = ["full", "no_reupload", "no_entangle",
                     "ring_only", "ring_skip", "ring_cross",
                     "classical_matched", "simple_mlp"]

# =============================================================================
# Quantum and related-work comparators
#   Quantum baselines, and a comparison against a related method:
#   Al-Hawawreh & Hossain, Information Fusion 99:101889, 2023
# =============================================================================
QUANTUM_BASELINES = ["qsvm_angle", "qsvm_iqp", "vqc", "amp_qnn", "alhawawreh_qdl"]

# QSVM: the fidelity Gram matrix is O(N^2) in memory, so the training set is
# subsampled (stratified, seeded). The test set is NOT subsampled, so the row
# stays directly comparable with every other model in the paper. Both numbers
# are recorded in the result JSON and must be stated alongside the results.
QSVM_TRAIN_SUBSAMPLE = 3_000 if not QUICK_MODE else 500
QSVM_C_GRID          = [0.5, 1.0, 4.0, 16.0]
QSVM_EMBEDDINGS      = ["angle", "iqp"]

# VQC / amplitude-QNN: no classical decoder, so the head is the measurement
# itself. Depth matches the headline circuit for a like-for-like comparison.
VQC_LAYERS      = N_LAYERS
AMP_QNN_QUBITS  = 4            # 2^4 = 16 amplitudes = the PCA dimension
AMP_QNN_LAYERS  = N_LAYERS

# --- Al-Hawawreh & Hossain (2023) --------------------------------------------
# The source paper fuses heterogeneous sensor and network modalities before a
# quantum deep-learning classifier. CICIoMT2024 WiFi/MQTT carries no biometric
# channel, so the re-implementation fuses the three FEATURE GROUPS the flow
# features fall into. Every such decision is a documented deviation, written
# into the result JSON, and the row must be presented as a faithful
# re-implementation under stated assumptions -- never as the authors' own
# reported result.
# A DISJOINT modality partition of the 25 flow features -- every feature lands
# in exactly one modality, which is what a fusion architecture requires. The
# overlapping keyword sets in perturb.column_groups() are deliberately
# overlapping (a feature can be perturbed by several mechanisms) and are the
# wrong thing to fuse over; this partition is defined once, here, and asserted
# to be a partition in qbaselines.group_feature_matrix().
ALHAWAWREH_MODALITIES = {
    "timing":   ["duration", "rate", "iat", "interval"],
    "flags":    ["flag", "fin", "syn", "rst", "psh", "ack", "urg", "ece", "cwr"],
    "protocol": ["http", "https", "dns", "tcp", "udp", "arp", "icmp", "proto"],
    "size":     ["header", "length", "size", "tot", "sum", "min", "max",
                 "avg", "std", "variance", "number", "count"],
}
ALHAWAWREH_GROUPS      = list(ALHAWAWREH_MODALITIES)
ALHAWAWREH_GROUP_DIM   = 8     # per-group dense encoder width
ALHAWAWREH_FUSION_DIM  = 16    # fused representation before the quantum layer

# Reported numbers from the published papers are TRANSCRIBED, never inferred.
# `reported` stays None until a value is copied from the cited article; NB14
# prints the outstanding rows rather than emitting an empty table cell, so an
# untranscribed number cannot silently reach a results table.
LITERATURE_ROWS = [
    {"ref": "Al-Hawawreh & Hossain (2023), Information Fusion 99:101889",
     "method": "Data fusion + quantum deep learning",
     "dataset": "(as reported by the authors)", "reported": None},
]

# =============================================================================
# Multiclass gap -- multiclass accuracy is lower than binary (79.19%)
# -----------------------------------------------------------------------------
# 97.0% of the paper model's multiclass error is a single confusion:
# 1,163 DoS -> DDoS plus 351 DDoS -> DoS out of 1,561 errors. Merging the two
# volumetric classes lifts accuracy to 99.37% with no retraining, which places
# the gap in one binary decision rather than in the 8-class problem.
# =============================================================================
VOLUMETRIC_CLASSES = (6, 7)          # DoS, DDoS in CLASS_NAMES order
COARSE_NAMES = ["Benign", "ARP_Spoofing", "Reconnaissance", "MQTT_Malformed",
                "MQTT_DoS", "MQTT_DDoS", "Volumetric_Flood"]
GAP_APPROACHES = ["hierarchical", "feature_path"]
GAP_SEEDS      = SEEDS_SWEEP          # exploratory; 3 seeds
# Raw features routed around PCA straight into the decoder (feature_path).
FEATURE_PATH_K = 4                    # selected by mutual information, on TRAIN only
# Classical specialists used to measure the REPRESENTATION ceiling: if a tuned
# random forest on the raw 25 features cannot separate DoS from DDoS either,
# the gap is a property of the feature set, not of the quantum model.
GAP_PROBES = ["Random Forest", "Gradient Boosting", "MLP"]

# =============================================================================
# Robustness -- adversarial attacks and realistic perturbations.
# Applied in the ORIGINAL 25-feature space, BEFORE scaling and PCA.
# =============================================================================
PERTURBATIONS = ["missing", "jitter", "packet_loss", "protocol_var"]
PERTURB_SEVERITIES = [0.0, 0.01, 0.05, 0.10, 0.20]
ADV_ATTACKS   = ["FGSM", "PGD-10", "transfer", "constrained"]
ADV_EPSILONS  = [0.0, 0.01, 0.02, 0.05, 0.10]
# The PCA-space Gaussian sweep of the paper is kept only as a labelled baseline.
PCA_GAUSS_SIGMAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]

# =============================================================================
# Explainability -- benefit of combining SHAP and LIME; heuristic inverse
# mapping; explanation stability and fidelity
# =============================================================================
XAI_CONFIG = {
    "subset": 300, "nsamples": 100, "bg_size": 200,
    "deletion_samples": 30, "lime_samples": 20,
    "stability_repeats": 10,      # LIME is stochastic -> variance is the point
    "perturb_eps": 0.01,          # local Lipschitz probe
}

# =============================================================================
# Audit log -- experimental demonstration of its added value
# =============================================================================
AUDIT_CHAIN_LENGTHS = [1_000, 10_000, 100_000]
AUDIT_TRIALS        = {1_000: 100, 10_000: 50, 100_000: 10}
AUDIT_ATTACKS       = ["modify", "delete", "reorder", "truncate"]
AUDIT_FLOW_RATES    = [10, 100, 1_000, 10_000]   # flows/s, storage projection

# =============================================================================
# Cost / energy -- operational cost, energy and circuit evaluations
# =============================================================================
LATENCY_REPEATS  = 200
COST_BATCHES     = [1, 8, 32, 128]
KWOLLECT_ENABLED = os.environ.get("KWOLLECT", "0") == "1"   # Grid5000 wattmeter

# =============================================================================
# Optional second dataset (generalisation)
# =============================================================================
WUSTL_ENABLED = os.environ.get("WUSTL", "0") == "1"
WUSTL_PATH    = DATA_DIR / "wustl-ehms-2020.csv"


# =============================================================================
# Work selection -- run a SLICE of a notebook per job
# -----------------------------------------------------------------------------
# The ablation is 8 variants x 2 tasks x 5 seeds = 80 training runs. At the
# measured ~258 s/epoch that is far beyond any single walltime, so the grid has
# to be spread over several jobs. Every run cell already writes its own manifest
# and is skipped on re-entry, so splitting costs nothing and loses nothing.
#
# Three ways to take a slice, all composable, all read from the environment so
# the notebook file itself never changes:
#
#   VARIANTS=ring_only,ring_skip   only these variants
#   TASKS=binary                   only this task
#   SEEDS=42,1                     only these seeds
#   SHARD=3/8                      shard 3 of 8, round-robin over the cell list
#
# SHARD is the one to use for an OAR array job: submit 8 jobs, give each its
# index, and they cover the grid between them without coordinating. Because the
# assignment is round-robin over a deterministically sorted list, every shard
# gets a comparable mix of cheap (binary) and expensive (multiclass) cells.
# =============================================================================
def _env_csv(name: str):
    """Parse VAR="a,b,c" into a list; None when unset or empty."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


_SEEDS_ENV = _env_csv("SEEDS")
if _SEEDS_ENV:
    SEEDS = [int(t) for t in _SEEDS_ENV]
    SEEDS_SWEEP = SEEDS[:3]

TASKS_SELECTED = _env_csv("TASKS")
VARIANTS_SELECTED = _env_csv("VARIANTS")

_SHARD = os.environ.get("SHARD", "").strip()
if _SHARD:
    _i, _n = (int(x) for x in _SHARD.split("/"))
    if not (1 <= _i <= _n):
        raise ValueError(f"SHARD={_SHARD}: index must be in 1..N")
    SHARD_INDEX, SHARD_COUNT = _i, _n
else:
    SHARD_INDEX, SHARD_COUNT = 1, 1


def cells_for(variants, tasks, seeds) -> list:
    """Every (variant, task, seed) this job is responsible for.

    Sorted deterministically, then filtered by VARIANTS / TASKS / SEEDS and
    finally by SHARD. Two jobs with different SHARD indices therefore never
    train the same cell, and together they cover the grid exactly.
    """
    sel_v = VARIANTS_SELECTED or list(variants)
    sel_t = TASKS_SELECTED or list(tasks)
    sel_s = [int(x) for x in (seeds or [])]

    unknown_v = [v for v in sel_v if v not in list(variants)]
    unknown_t = [t for t in sel_t if t not in list(tasks)]
    if unknown_v:
        raise ValueError(f"VARIANTS={unknown_v} not in {list(variants)}")
    if unknown_t:
        raise ValueError(f"TASKS={unknown_t} not in {list(tasks)}")

    cells = sorted(((v, t, s) for v in sel_v for t in sel_t for s in sel_s),
                   key=lambda c: (c[1], c[0], c[2]))
    return [c for k, c in enumerate(cells) if k % SHARD_COUNT == SHARD_INDEX - 1]


def shard_items(items) -> list:
    """Round-robin slice of any deterministic list, using the same SHARD.

    `cells_for` is the ablation-shaped version of this; sweeps whose work is not
    (variant, task, seed) use this one directly.
    """
    return [x for k, x in enumerate(list(items)) if k % SHARD_COUNT == SHARD_INDEX - 1]


def describe_selection(total: int, mine: int) -> str:
    bits = [f"shard {SHARD_INDEX}/{SHARD_COUNT}"]
    if VARIANTS_SELECTED:
        bits.append(f"variants={VARIANTS_SELECTED}")
    if TASKS_SELECTED:
        bits.append(f"tasks={TASKS_SELECTED}")
    if _SEEDS_ENV:
        bits.append(f"seeds={SEEDS}")
    return (f"{mine} of {total} cells assigned to this job ({', '.join(bits)}). "
            f"Cells already carrying a manifest are skipped.")


# --- ablation training budget ------------------------------------------------
# The ablation is a RELATIVE comparison, so it does not need the headline's full
# training set -- but every variant must get the IDENTICAL budget, or the
# comparison measures the budget instead of the architecture.
#
# ABLATION_SAMPLES=10000 cuts a binary cell from ~3.6 h to ~1.0 h and a
# multiclass cell from ~8.7 h to ~2.5 h. If you use it, say so in the caption:
# the ablation rows are then comparable with each other but NOT with Tables 6
# and 9, which are trained on the full 35,000.
# Unset (the default) means the full training split, identical to the headline.
_AS = os.environ.get("ABLATION_SAMPLES", "").strip()
ABLATION_SAMPLES = int(_AS) if _AS else None


def training_budget(X, y, n: int = None, seed: int = 42):
    """Stratified cap applied identically to every ablation variant."""
    n = ABLATION_SAMPLES if n is None else n
    if not n or len(X) <= n:
        return X, y
    import numpy as _np
    rng = _np.random.default_rng(seed)
    idx = []
    for c in _np.unique(y):
        pos = _np.flatnonzero(y == c)
        take = max(1, int(round(len(pos) / len(y) * n)))
        idx.append(rng.choice(pos, min(take, len(pos)), replace=False))
    idx = _np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def summary() -> str:
    return "\n".join([
        "=== Experiment configuration ===",
        f"  MODE          : {'QUICK' if QUICK_MODE else 'FULL'}",
        f"  ROOT          : {ROOT}",
        f"  DATA_DIR      : {DATA_DIR}",
        f"  OUT_DIR       : {OUT_DIR}",
        f"  ARCHIVE_DIR   : {ARCHIVE_DIR}",
        f"  SEEDS         : {SEEDS}",
        f"  DATASET       : {DATASET}  ({N_CLASSES} classes)",
        f"  POOL / PCA    : {POOL_SIZE} samples, {N_FEATURES} -> {N_PCA}",
        f"  QCNN          : {N_QUBITS} qubits / {N_LAYERS} layers, "
        f"scales={ENTANGLE_SCALES}",
        f"  BACKEND       : {QUANTUM_DEVICE} / {DIFF_METHOD}",
        f"  EPOCHS        : binary {EPOCHS_BINARY}, multiclass {EPOCHS_MULTICLASS}",
        f"  ENLARGED TEST : {ENLARGED_TEST} (min {MIN_TEST_PER_CLASS}/class)",
        f"  SHARD         : {SHARD_INDEX}/{SHARD_COUNT}"
        + (f"  VARIANTS={VARIANTS_SELECTED}" if VARIANTS_SELECTED else "")
        + (f"  TASKS={TASKS_SELECTED}" if TASKS_SELECTED else ""),
        f"  ABLATION_SAMP : {ABLATION_SAMPLES or 'full training split'}",
    ])


# =============================================================================
# Compatibility shim
# -----------------------------------------------------------------------------
# circuits.py / models.py / train.py were verified BIT-EXACT against the
# archived Grid5000 weights (100% per-sample agreement on both tasks). They are
# ported here unchanged so that verification still holds; this shim exposes the
# structured accessors they expect on top of the flat constants above.
# =============================================================================
from dataclasses import dataclass, field, asdict as _asdict
from typing import Dict, Tuple


@dataclass(frozen=True)
class _Arch:
    N_QUBITS: int = N_QUBITS
    N_LAYERS: int = N_LAYERS
    ENC_BINARY: Tuple[int, ...] = (32,)
    DEC_BINARY: Tuple[int, ...] = (16,)
    ENC_MULTICLASS: Tuple[int, ...] = (64, 32)
    DEC_MULTICLASS: Tuple[int, ...] = (64, 32)
    DROPOUT_ENC: float = 0.20
    DROPOUT_DEC_BINARY: float = 0.20
    DROPOUT_DEC_MULTI: Tuple[float, float] = (0.15, 0.10)


@dataclass(frozen=True)
class _Train:
    EPOCHS_BINARY: int = EPOCHS_BINARY
    EPOCHS_MULTICLASS: int = EPOCHS_MULTICLASS
    LR_BINARY: float = LR_BINARY
    LR_MIN_BINARY: float = LR_MIN_BINARY
    LR_MULTICLASS: float = LR_MULTICLASS
    LR_MIN_MULTICLASS: float = LR_MIN_MULTICLASS
    BATCH_SIZE: int = BATCH_SIZE
    WEIGHT_DECAY: float = WEIGHT_DECAY
    GRAD_CLIP: float = GRAD_CLIP
    EARLY_STOP_PATIENCE: int = EARLY_STOP_PATIENCE
    EARLY_STOP_MIN_EPOCHS: int = EARLY_STOP_MIN_EPOCHS
    CHECKPOINT_EVERY_N_EPOCHS: int = 10
    SMOKE_EPOCHS: int = 2
    SMOKE_SAMPLES: int = 2000


@dataclass(frozen=True)
class _Quantum:
    BACKEND_PREFERENCE: Tuple[str, ...] = (QUANTUM_DEVICE,)
    BACKEND_FALLBACK: str = "lightning.qubit"
    DIFF_METHOD_PREFERRED: str = DIFF_METHOD
    DIFF_METHOD_FALLBACK: str = "adjoint"
    SHOTS_TRAINING_SUPPORTED: bool = SHOTS_TRAINING_SUPPORTED


@dataclass(frozen=True)
class _Data:
    PCA_COMPONENTS: int = N_PCA
    N_CLASSES: int = N_CLASSES


@dataclass(frozen=True)
class _CFG:
    arch: _Arch = field(default_factory=_Arch)
    train: _Train = field(default_factory=_Train)
    quantum: _Quantum = field(default_factory=_Quantum)
    data: _Data = field(default_factory=_Data)

    def as_dict(self) -> Dict:
        return {"arch": _asdict(self.arch), "train": _asdict(self.train),
                "quantum": _asdict(self.quantum), "data": _asdict(self.data),
                "seeds": list(SEEDS), "class_names": list(CLASS_NAMES),
                "dataset": DATASET, "angle_scale": ANGLE_SCALE}


CFG = _CFG()


def epochs_for(task: str, smoke: bool = False) -> int:
    if smoke:
        return CFG.train.SMOKE_EPOCHS
    return EPOCHS_BINARY if task == "binary" else EPOCHS_MULTICLASS


def lr_for(task: str):
    if task == "binary":
        return LR_BINARY, LR_MIN_BINARY
    return LR_MULTICLASS, LR_MIN_MULTICLASS


def n_classes_for(task: str) -> int:
    return 2 if task == "binary" else N_CLASSES


def set_global_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches. Called once per run cell."""
    import random
    import numpy as _np
    import torch
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


RESULTS_DIR = str(OUT_DIR)


# =============================================================================
# QUICK_MODE training-set cap
# -----------------------------------------------------------------------------
# QUICK_MODE reduces epochs and seeds, but the notebooks load their splits from
# the ARCHIVE (35,000 training rows), which POOL_SIZE does not affect. Without
# a cap, a "quick" ablation is 8 variants x 2 tasks x 3 seeds x 3 epochs at
# ~430 s/epoch -- about 17 hours, which defeats the purpose of a smoke test.
#
# subsample_for_mode() is applied to TRAINING data only. Validation and test
# splits stay whole, so the metrics remain interpretable.
# =============================================================================
QUICK_TRAIN_SAMPLES = 2_000


def subsample_for_mode(X, y, seed: int = 42):
    """Cap the training set in QUICK_MODE; return it untouched otherwise."""
    if not QUICK_MODE or len(X) <= QUICK_TRAIN_SAMPLES:
        return X, y
    import numpy as _np
    rng = _np.random.default_rng(seed)
    # Stratified so every class survives the cut -- an unstratified sample
    # would drop the rare classes and make the smoke test meaningless.
    idx = []
    for c in _np.unique(y):
        pos = _np.flatnonzero(y == c)
        take = max(1, int(round(len(pos) / len(y) * QUICK_TRAIN_SAMPLES)))
        idx.append(rng.choice(pos, min(take, len(pos)), replace=False))
    idx = _np.concatenate(idx)
    rng.shuffle(idx)
    print(f"[QUICK_MODE] training set capped {len(X)} -> {len(idx)} (stratified)")
    return X[idx], y[idx]
