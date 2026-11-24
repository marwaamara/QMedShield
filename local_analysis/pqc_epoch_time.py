"""Time PQC vs twin training steps on THIS machine, single thread, identical loader/optimiser;
epoch time = median step time x steps per epoch (35,000 / 32 = 1,094)."""
import sys, time, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "src"); torch.set_num_threads(1)
from models import build_model
D = np.load("prepared_data.npz"); X = torch.tensor(D["X_train_m"][:32 * 40], dtype=torch.float32); y = torch.tensor(D["y_train_m"][:32 * 40])
out = {}
for name, var, n in (("pqc", "full", 25), ("twin", "linear", 25)):
    m = build_model("multiclass", var); opt = torch.optim.AdamW(m.parameters(), 2e-3); lf = nn.CrossEntropyLoss(); ts = []
    for i in range(n):
        xb, yb = X[i * 32:(i + 1) * 32], y[i * 32:(i + 1) * 32]
        t = time.perf_counter(); opt.zero_grad(); lf(m(xb), yb).backward(); opt.step(); ts.append(time.perf_counter() - t)
    out[name] = {"median_step_s": float(np.median(ts[3:])), "epoch_s_extrapolated": float(np.median(ts[3:]) * 1094)}
json.dump(out, open("results/pqc_epoch_time.json", "w"), indent=1); print(out)
