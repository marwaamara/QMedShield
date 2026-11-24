"""
Diagnosis: why does a Random Forest separate DoS from DDoS at ~99.9% (even on held-out captures) on the
45 released features, while specialists trained on the 25 selected features plateau at ~0.81?  Hypothesis: the
feature selection (25 of 45) discarded the discriminative features.

For TCP_IP-DoS vs TCP_IP-DDoS, held-out-capture protocol (same 3 repeats as dos_ddos_window_test.py):
  (a) all 45 features          (b) the 25 features kept by the pipeline
  (c) the 20 features the pipeline dropped      (d) each dropped feature alone (single-feature depth-limited tree)
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

H = Path(__file__).parent
folder = sys.argv[1]
KEPT = [str(x) for x in np.load(H / "archive" / "prepared_data.npz", allow_pickle=True)["kept_names"]]
PER_BLOCK = 10000
blocks = []   # (capture, y, DataFrame)
for fname in ("CIC_IoMT_2024_WiFi_MQTT_train.csv", "CIC_IoMT_2024_WiFi_MQTT_test.csv"):
    cur, buf = None, []

    def flush():
        if cur is None or not buf or not (cur.startswith("TCP_IP-DoS") or cur.startswith("TCP_IP-DDoS")):
            return
        df = pd.concat(buf)
        df = df.iloc[np.sort(np.random.default_rng(0).choice(len(df), min(PER_BLOCK, len(df)), replace=False))]
        blocks.append((fname[:20] + cur, 1 if cur.startswith("TCP_IP-DDoS") else 0, df.drop(columns=["label"])))

    for chunk in pd.read_csv(Path(folder) / fname, chunksize=500000, dtype={"label": str}):
        lab = chunk["label"].to_numpy()
        for seg in np.split(np.arange(len(chunk)), np.flatnonzero(lab[1:] != lab[:-1]) + 1):
            l = lab[seg[0]]
            if l != cur:
                flush(); cur, buf = l, []
            buf.append(chunk.iloc[seg])
    flush()
cols = list(blocks[0][2].columns)
dropped = [c for c in cols if c not in KEPT]
print(len(blocks), "captures;", len(cols), "features;", len(dropped), "dropped:", dropped, flush=True)
ys = np.array([b[1] for b in blocks])


def run(featset, rep):
    rng = np.random.default_rng(100 + rep); test = set()
    for c in (0, 1):
        cand = np.flatnonzero(ys == c); test |= set(rng.choice(cand, max(1, int(round(0.25 * len(cand)))), replace=False).tolist())
    tr = [b for i, b in enumerate(blocks) if i not in test]; te = [b for i, b in enumerate(blocks) if i in test]
    X = lambda L: np.nan_to_num(np.vstack([b[2][featset].to_numpy(np.float64) for b in L]), nan=0, posinf=0, neginf=0)
    y = lambda L: np.concatenate([np.full(len(b[2]), b[1]) for b in L])
    Xtr, ytr = X(tr), y(tr)
    i = np.random.default_rng(1).choice(len(ytr), min(150000, len(ytr)), replace=False)
    m = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, min_samples_leaf=2).fit(Xtr[i], ytr[i])
    return float(accuracy_score(y(te), m.predict(X(te))))


out = {"n_captures": len(blocks), "dropped_features": dropped}
for name, fs in (("all_45", cols), ("kept_25", [c for c in KEPT if c in cols]), ("dropped_20", dropped)):
    a = [run(fs, r) for r in range(3)]
    out[name] = {"acc_per_rep": a, "mean": float(np.mean(a))}; print(name, out[name], flush=True)

# single-feature screening (depth-3 tree), same held-out protocol, rep 0
rng = np.random.default_rng(100); test = set()
for c in (0, 1):
    cand = np.flatnonzero(ys == c); test |= set(rng.choice(cand, max(1, int(round(0.25 * len(cand)))), replace=False).tolist())
tr = [b for i, b in enumerate(blocks) if i not in test]; te = [b for i, b in enumerate(blocks) if i in test]
single = {}
for c in cols:
    Xtr = np.nan_to_num(np.concatenate([b[2][c].to_numpy(np.float64) for b in tr]))[:, None]; ytr = np.concatenate([np.full(len(b[2]), b[1]) for b in tr])
    Xte = np.nan_to_num(np.concatenate([b[2][c].to_numpy(np.float64) for b in te]))[:, None]; yte = np.concatenate([np.full(len(b[2]), b[1]) for b in te])
    single[c] = float(accuracy_score(yte, DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr).predict(Xte)))
out["single_feature_depth3_acc"] = dict(sorted(single.items(), key=lambda kv: -kv[1]))
out["single_feature_kept"] = {k: v for k, v in out["single_feature_depth3_acc"].items() if k in KEPT}
print(json.dumps(out["single_feature_depth3_acc"], indent=1))
json.dump(out, open(H / "results" / "dos_ddos_feature_diagnosis.json", "w"), indent=1)
