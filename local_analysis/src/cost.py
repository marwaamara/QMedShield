"""
cost.py -- operational cost, latency and energy.

Parameter count alone does not reflect deployment cost for quantum machine
learning systems: practical execution also depends on quantum circuit
evaluation, shot requirements, preprocessing overhead, classical control and
hardware constraints. This module measures those quantities.

The key point: PARAMETER COUNT IS NOT DEPLOYMENT COST. A
1,178-parameter model that needs one circuit execution per inference, at some
shot budget, on hardware with a queue, is not cheaper than a 500-tree forest
just because it serialises to 4.6 KB.

Energy: on Grid5000 the node wattmeters (Kwollect) give measured wall power,
which is a real measurement rather than an estimate. Off-cluster this module
falls back to a CPU-time-based estimate and LABELS it as such -- an estimate
presented as a measurement would be misleading.
"""
from __future__ import annotations

import os
import platform
import time
from typing import Dict, List, Optional, Sequence

import numpy as np


# =============================================================================
# Latency and throughput
# =============================================================================
def measure_latency(predict_fn, X: np.ndarray, batch_sizes: Sequence[int] = (1,),
                    repeats: int = 200, warmup: int = 10) -> List[Dict]:
    """Per-sample latency at several batch sizes.

    Median plus p95/p99, because a real-time IDS is sized by its tail, not its
    mean.
    """
    rows = []
    for bs in batch_sizes:
        Xb = X[:bs] if len(X) >= bs else np.repeat(X, bs // len(X) + 1, 0)[:bs]

        for _ in range(warmup):
            predict_fn(Xb)

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            predict_fn(Xb)
            times.append(time.perf_counter() - t0)
        t = np.asarray(times)

        rows.append({
            "batch_size": bs,
            "median_ms_per_batch": float(np.median(t) * 1e3),
            "median_ms_per_sample": float(np.median(t) * 1e3 / bs),
            "p95_ms_per_sample": float(np.percentile(t, 95) * 1e3 / bs),
            "p99_ms_per_sample": float(np.percentile(t, 99) * 1e3 / bs),
            "throughput_samples_per_s": float(bs / np.median(t)),
            "repeats": repeats,
        })
    return rows


def measure_preprocessing_overhead(scaler, pca, X_raw: np.ndarray,
                                   repeats: int = 100) -> Dict:
    """Scaler + PCA cost, timed separately.

    This is excluded from the parameter-count story. It is
    paid on every inference in deployment, so it belongs in the cost table.
    """
    def timeit(fn, n):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    Xs = scaler.transform(X_raw)
    t_scale = timeit(lambda: scaler.transform(X_raw), repeats)
    t_pca = timeit(lambda: pca.transform(Xs), repeats)
    n = len(X_raw)
    return {"n_samples": n,
            "scaler_ms_per_sample": t_scale * 1e3 / n,
            "pca_ms_per_sample": t_pca * 1e3 / n,
            "total_preproc_ms_per_sample": (t_scale + t_pca) * 1e3 / n}


# =============================================================================
# Compute accounting
# =============================================================================
def quantum_cost_model(n_qubits: int, n_layers: int, n_samples: int,
                       shots: Optional[int] = None,
                       epochs: int = 0, batch_size: int = 32) -> Dict:
    """Circuit evaluations, gate counts and shot budget.

    Gate count per layer: n_q RY (re-upload) + n_q Rot + the entangler
    (n_q ring CNOT, plus n_q/2 skip CZ on even layers or n_q/2 cross CNOT on
    odd layers) -- so about 3.5*n_q per layer on average.

    Training with the hardware-compatible parameter-shift rule costs
    2*|theta_q| circuit evaluations per gradient, which is the number that makes
    "compact" and "cheap to train on hardware" two different claims.
    """
    n_params_q = 3 * n_layers * n_qubits
    gates_per_layer = n_qubits * 2 + n_qubits + n_qubits // 2
    total_gates = gates_per_layer * n_layers

    inference_evals = n_samples
    inference_shots = n_samples * shots * n_qubits if shots else None

    steps = (n_samples // batch_size) * epochs if epochs else 0
    train_evals_ps = steps * batch_size * 2 * n_params_q if steps else 0

    return {
        "n_qubits": n_qubits, "n_layers": n_layers,
        "n_quantum_params": n_params_q,
        "gates_per_layer": gates_per_layer, "total_gates": total_gates,
        "circuit_depth_estimate": total_gates,
        "statevector_dim": 2 ** n_qubits,
        "sim_cost_per_sample_ops": n_layers * n_qubits * (2 ** n_qubits),
        "inference_circuit_evals": inference_evals,
        "inference_total_shots": inference_shots,
        "training_steps": steps,
        "training_circuit_evals_parameter_shift": train_evals_ps,
        "note": ("parameter-shift training cost = 2*|theta_q| circuit "
                 "evaluations per gradient step; adjoint/backprop is a "
                 "simulator-only shortcut unavailable on hardware"),
    }


def qpu_cost_projection(circuit_evals: int, shots: int,
                        seconds_per_shot: float = 1e-4,
                        price_per_second_eur: float = 0.0,
                        price_per_task_eur: float = 0.0,
                        tasks: int = 1) -> Dict:
    """What the workload would cost on real NISQ hardware, today.

    Defaults are zero so nothing is invented: supply the numbers from whichever
    provider is being cited, and the report states the arithmetic and its
    assumptions rather than a guess. Queueing and calibration are excluded and
    must be stated as excluded.
    """
    total_shots = circuit_evals * shots
    qpu_seconds = total_shots * seconds_per_shot
    return {
        "circuit_evals": circuit_evals, "shots_per_eval": shots,
        "total_shots": total_shots,
        "qpu_seconds": qpu_seconds, "qpu_hours": qpu_seconds / 3600.0,
        "cost_eur": qpu_seconds * price_per_second_eur +
                    tasks * price_per_task_eur,
        "assumptions": {"seconds_per_shot": seconds_per_shot,
                        "price_per_second_eur": price_per_second_eur,
                        "price_per_task_eur": price_per_task_eur,
                        "excluded": ["queue wait", "calibration",
                                     "transpilation", "error mitigation"]},
    }


def classical_flops(model) -> Dict:
    """Forward-pass multiply-accumulates for the classical layers.

    The comparable axis to the quantum circuit-evaluation count: both are
    per-inference work, and neither is the parameter count.
    """
    import torch.nn as nn

    total, per_layer = 0, {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            f = mod.in_features * mod.out_features
            per_layer[name] = f
            total += f
    return {"total_macs": total, "per_layer": per_layer}


# =============================================================================
# Energy
# =============================================================================
def kwollect_available() -> bool:
    return os.environ.get("KWOLLECT", "0") == "1" and "grid5000" in platform.node()


def fetch_kwollect_energy(node: str, t_start: float, t_end: float,
                          site: Optional[str] = None) -> Dict:
    """Measured wall power from the Grid5000 wattmeters.

    Requires an EXCLUSIVE node reservation: with co-located jobs the reading
    describes the machine, not the experiment.
    """
    import requests

    site = site or os.environ.get("OAR_SITE", "lyon")
    url = (f"https://api.grid5000.fr/stable/sites/{site}/metrics"
           f"?nodes={node}&metrics=wattmetre_power_watt"
           f"&start_time={int(t_start)}&end_time={int(t_end)}")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        pts = r.json()
        watts = np.array([p["value"] for p in pts], float)
        if len(watts) == 0:
            return {"available": False, "reason": "no samples returned"}
        dur = t_end - t_start
        return {"available": True, "source": "kwollect_wattmeter",
                "mean_watts": float(watts.mean()),
                "peak_watts": float(watts.max()),
                "duration_s": dur,
                "energy_joules": float(watts.mean() * dur),
                "energy_wh": float(watts.mean() * dur / 3600.0),
                "n_samples": len(watts)}
    except Exception as e:
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}


def estimate_energy_from_cpu(cpu_seconds: float, tdp_watts: float = 105.0,
                             n_cores_used: int = 1,
                             n_cores_total: Optional[int] = None) -> Dict:
    """Fallback ESTIMATE when no wattmeter is available.

    Explicitly labelled an estimate. It must not be presented as a
    measurement; reports must say which of the two they contain.
    """
    total = n_cores_total or os.cpu_count() or 1
    watts = tdp_watts * (n_cores_used / total)
    j = watts * cpu_seconds
    return {"available": True, "source": "cpu_time_estimate",
            "is_estimate": True,
            "assumed_tdp_watts": tdp_watts, "cores_used": n_cores_used,
            "cores_total": total, "effective_watts": watts,
            "cpu_seconds": cpu_seconds,
            "energy_joules": j, "energy_wh": j / 3600.0,
            "caveat": "estimate from CPU time and assumed TDP, NOT a measurement"}


class EnergyMeter:
    """Context manager: wattmeter if available, CPU-time estimate otherwise."""

    def __init__(self, label: str = "run", n_cores_used: int = 1):
        self.label = label
        self.n_cores_used = n_cores_used
        self.result: Dict = {}

    def __enter__(self):
        self.t0 = time.time()
        self.c0 = time.process_time()
        return self

    def __exit__(self, *exc):
        wall = time.time() - self.t0
        cpu = time.process_time() - self.c0
        if kwollect_available():
            e = fetch_kwollect_energy(platform.node(), self.t0, time.time())
            if not e.get("available"):
                e = estimate_energy_from_cpu(cpu, n_cores_used=self.n_cores_used)
        else:
            e = estimate_energy_from_cpu(cpu, n_cores_used=self.n_cores_used)
        e.update({"label": self.label, "wall_seconds": wall, "cpu_seconds": cpu})
        self.result = e
        return False


def co2_equivalent(energy_wh: float, g_per_kwh: float = 56.0) -> Dict:
    """CO2e. Default is the French grid intensity (~56 gCO2e/kWh), which is
    where Grid5000 runs; state the factor in the report, since a US or
    German grid would give a very different number for identical compute."""
    return {"energy_wh": energy_wh, "grid_g_per_kwh": g_per_kwh,
            "co2e_grams": energy_wh / 1000.0 * g_per_kwh}
