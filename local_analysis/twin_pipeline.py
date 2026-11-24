"""
twin_pipeline.py -- classical twin of the complete pipeline: a direct comparison between the
proposed framework and an otherwise identical classical implementation.

Builds the SAME pipeline twice -- identical classical encoder, decoder, schedule,
validation-only checkpointing, KernelSHAP explanation, SHA-256 hash-chain audit --
and swaps only the 8-qubit PQC for the parameter-matched dense linear bottleneck.
Measures, on this machine, under identical single-thread CPU conditions:
  * accuracy / weighted F1 over seeds (classical twin, trained here)
  * parameter count and footprint
  * training wall-clock per epoch
  * inference latency vs batch size
  * explanation cost (KernelSHAP seconds per explained flow)
  * audit overhead per decision
The PQC accuracy over seeds is NOT retrained here (it is the Grid'5000 result reported
in the paper); the PQC is instantiated only to time it under the same conditions.
"""
import json, sys, time, random, statistics as st
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent / "src"))
torch.set_num_threads(1)
from config import set_global_seed
from models import build_model
from train import make_loader, compute_class_weights, train_model, evaluate
from audit import HashChainLog, make_event
from xai import kernel_shap, torch_predict_proba
from sklearn.metrics import f1_score, accuracy_score

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
D = np.load(Path(__file__).parent / "archive" / "prepared_data.npz")
task = sys.argv[1]; MODE = sys.argv[2]            # MODE: "train <seed>" or "time"
SEEDS = [int(sys.argv[3])] if MODE == "train" else []
k = "m" if task == "multiclass" else "b"
Xtr, ytr, Xva, yva, Xte, yte = (D[f"X_train_{k}"], D[f"y_train_{k}"], D[f"X_val_{k}"],
                                D[f"y_val_{k}"], D[f"X_test_{k}"], D[f"y_test_{k}"])
ncls = 8 if task == "multiclass" else 2

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()

def latency(model, bs, rep=30):
    model.eval(); x = torch.tensor(Xte[:bs], dtype=torch.float32)
    with torch.no_grad():
        model(x); ts = []
        for _ in range(rep):
            t = time.perf_counter(); model(x); ts.append(time.perf_counter() - t)
    return 1e3 * st.median(ts) / bs

res = {"task": task, "seeds": {}}
for s in SEEDS:
    set_global_seed(s)
    m = build_model(task, "linear")
    cw = compute_class_weights(ytr, ncls)
    h = train_model(m, make_loader(Xtr, ytr), make_loader(Xva, yva, shuffle=False), task,
                    class_weights=cw, verbose=False)
    p = predict(m, Xte)
    res["seeds"][s] = {"acc": float(accuracy_score(yte, p)),
                       "f1w": float(f1_score(yte, p, average="weighted")),
                       "epochs_run": h["epochs_run"], "train_wall_s": h["train_wall_s"],
                       "s_per_epoch": h["train_wall_s"] / h["epochs_run"]}
    np.save(OUT / f"twin_{task}_seed{s}_pred.npy", p)
    print(task, s, res["seeds"][s], flush=True)
    last = m
    json.dump(res, open(OUT / f"twin_{task}_seed{s}.json", "w"), indent=1)
    torch.save(m.state_dict(), OUT / f"twin_{task}_seed{s}.pt")

if MODE == "train": sys.exit(0)
last = build_model(task, "linear"); last.load_state_dict(torch.load(OUT / f"twin_{task}_seed42.pt"))
# ---- same-conditions timing: classical twin vs PQC (weights irrelevant for cost) -----
pqc = build_model(task, "full")
res["params"] = {"twin": last.param_breakdown(), "pqc": pqc.param_breakdown()}
res["latency_ms_per_sample"] = {str(bs): {"twin": latency(last, bs), "pqc": latency(pqc, bs, 5 if bs == 1 else 3)}
                                for bs in (1, 32, 128)}
print(res["latency_ms_per_sample"], flush=True)

def shap_cost(model, n_coal=100, n_bg=50, bs=128):
    """Wall-clock of the forward evaluations ONE KernelSHAP explanation needs
    (n_coal coalitions x n_bg background points = 5,000 evaluations), run in batches of bs.
    The shap package cannot be imported in this environment (numba/NumPy conflict), so the
    explainer's own arithmetic is excluded; it is identical for both models."""
    model.eval(); X = torch.tensor(np.tile(Xte[:128], (int(np.ceil(n_coal * n_bg / 128)), 1))[: n_coal * n_bg], dtype=torch.float32)
    t = time.perf_counter()
    with torch.no_grad():
        for k in range(0, len(X), bs): model(X[k:k + bs])
    return time.perf_counter() - t
res["kernelshap_forward_s_per_explanation"] = {"twin": shap_cost(last), "pqc": shap_cost(pqc)}
print(res["kernelshap_forward_s_per_explanation"], flush=True)

rng = random.Random(0); chain = HashChainLog(); ev = [make_event(i, rng, ncls) for i in range(5000)]
t = time.perf_counter(); [chain.append(e) for e in ev]; res["audit_us_per_event"] = (time.perf_counter() - t) / 5000 * 1e6
res["audit_verify_ok"] = chain.verify()[0]
json.dump(res, open(OUT / f"twin_{task}.json", "w"), indent=1)
print("DONE")
