"""
robustness_twin.py -- same robustness protocol as notebook 07 applied to the
classical-twin model (dense-linear bottleneck, seed 42), so quantum and classical
bottlenecks are compared under identical perturbations, rows and budgets.
PQC numbers are the ones reported in the paper (robustness tables).
"""
import sys, json, numpy as np, torch
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score
H = Path(__file__).parent; sys.path.insert(0, str(H / "src")); torch.set_num_threads(1)
from models import build_model
from load_g5k import load_prepared_data, load_preprocessors, recover_raw_features
from perturb import apply_perturbation, pgd, constrained_attack
from xai import pipeline_predict_proba
A = str(H / "archive"); R = H / "results"
data = load_prepared_data(A); pre = load_preprocessors(A); X_raw, names = recover_raw_features(A)
d, idx = cKDTree(data["X_pca_m"]).query(data["X_test_m"], k=1); assert (d < 1e-9).all()
Xr, y = X_raw[idx], data["y_test_m"]; scaler, pca = pre["scaler"], pre["pca_multi"]
out = {}
for seed in (42,):
    m = build_model("multiclass", "linear"); m.load_state_dict(torch.load(R / f"twin_multiclass_seed{seed}.pt")); m.eval()
    pred = pipeline_predict_proba(m, scaler, pca)
    rng = np.random.default_rng(42); rows = []
    for kind in ["missing", "jitter", "packet_loss", "protocol_var"]:
        for sev in [0.0, 0.01, 0.05, 0.10, 0.20]:
            Xp = apply_perturbation(Xr, kind, sev, names, rng)
            rows.append({"perturbation": kind, "severity": sev, "accuracy": float(accuracy_score(y, pred(Xp).argmax(1)))})
    out["raw_space"] = rows
    adv = []
    for eps in [0.0, 0.01, 0.02, 0.05, 0.10]:
        for name, steps in (("FGSM", 1), ("PGD-10", 10)):
            Xa = data["X_test_m"] if eps == 0 else pgd(m, data["X_test_m"], y, eps, steps=steps)
            with torch.no_grad():
                a = float((m(torch.tensor(Xa, dtype=torch.float32)).argmax(1).numpy() == y).mean())
            adv.append({"attack": name, "epsilon": eps, "accuracy": a})
    out["adversarial_pca"] = adv
    con = []
    for e in (0.0, 0.02, 0.05, 0.10):
        Xc = Xr[:1000] if e == 0 else constrained_attack(m, Xr[:1000], y[:1000], names, scaler, pca, e, steps=5)
        con.append({"eps_rel": e, "accuracy": float(accuracy_score(y[:1000], pred(Xc).argmax(1)))})
    out["constrained"] = con
json.dump(out, open(R / "twin_robustness.json", "w"), indent=1); print(json.dumps(out)[:3000])
