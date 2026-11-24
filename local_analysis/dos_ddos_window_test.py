"""
dos_ddos_window_test.py -- DoS versus DDoS: temporal aggregation over neighbouring flows.

Question: do causal temporal-aggregation / flow-distribution statistics over *neighbouring flows*
(a proxy for source diversity, which the released flow records do not carry) separate DoS from DDoS
better than the per-flow features do?

DATA LAYOUT (verified on the released files): each CSV is a concatenation of capture blocks, one per
label value (train: 51 blocks, test: 21), in capture order. A "capture" = one contiguous label block.
Window statistics are computed INSIDE a block and are causal (row i uses rows i-w..i-1 only).

TASK: TCP_IP-DoS-* (class 0) versus TCP_IP-DDoS-* (class 1), WiFi volumetric floods.

TWO PROTOCOLS
  chrono   : within every capture, first 85% of rows train, last 15% test (as the chronological
             protocol of the paper). Weak: the test rows come from captures seen in training.
  heldout  : whole captures are held out (about 25% of captures per class, 3 random repeats). The model
             must separate DoS from DDoS in capture sessions it has never seen -- the stricter test.
FEATURES
  per_flow         : the 45 released numeric features
  per_flow+window  : + causal rolling mean and std over the previous 50 and 200 flows of every feature

USAGE   python dos_ddos_window_test.py "<folder with the two CSVs>"
Output  results/dos_ddos_window_test.json
"""
import sys, json, time, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

W = (50, 200)
SUBSET = os.environ.get("FEATURES", "all")   # all | kept25 | noIAT
KEPT25 = [str(x) for x in np.load(Path(__file__).parent / "archive" / "prepared_data.npz", allow_pickle=True)["kept_names"]]


def select(df):
    df = df.drop(columns=["label"])
    if SUBSET == "kept25": return df[KEPT25]
    if SUBSET == "noIAT": return df.drop(columns=["IAT"])
    return df

PER_BLOCK = 10000          # rows kept per capture block after window features are computed
RF = dict(n_estimators=100, n_jobs=-1, random_state=42, min_samples_leaf=2)
MAX_TRAIN = 200000


def window_feats(X, w):
    z = np.zeros((1, X.shape[1]))
    c, c2 = np.cumsum(np.vstack([z, X]), 0), np.cumsum(np.vstack([z, X ** 2]), 0)
    n = np.arange(len(X)); lo = np.maximum(0, n - w); cnt = (n - lo)[:, None].clip(1)
    mean = (c[n] - c[lo]) / cnt
    var = np.maximum((c2[n] - c2[lo]) / cnt - mean ** 2, 0)
    return np.hstack([mean, np.sqrt(var)]).astype(np.float32)


def load_blocks(folder):
    blocks = []   # (capture_id, y, pos_fraction, X_flow, X_window)
    for fname in ("CIC_IoMT_2024_WiFi_MQTT_train.csv", "CIC_IoMT_2024_WiFi_MQTT_test.csv"):
        t0 = time.time()
        state = {"cur": None, "buf": []}

        def flush():
            lab, frames = state["cur"], state["buf"]
            if lab is None or not frames:
                return
            if not (lab.startswith("TCP_IP-DoS") or lab.startswith("TCP_IP-DDoS")):
                return
            df = pd.concat(frames)
            X = np.nan_to_num(select(df).to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            F = np.hstack([window_feats(X, w) for w in W])
            n = len(X)
            idx = np.sort(np.random.default_rng(0).choice(n, min(PER_BLOCK, n), replace=False))
            blocks.append((fname[:20] + lab, 1 if lab.startswith("TCP_IP-DDoS") else 0, idx / n,
                           X[idx].astype(np.float32), F[idx]))

        for chunk in pd.read_csv(Path(folder) / fname, chunksize=500000, dtype={"label": str}):
            # keep original row order: split the chunk into runs of equal label
            lab = chunk["label"].to_numpy()
            change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
            for seg in np.split(np.arange(len(chunk)), change):
                l = lab[seg[0]]
                if l != state["cur"]:
                    flush(); state["cur"], state["buf"] = l, []
                state["buf"].append(chunk.iloc[seg])
        flush()
        print(f"{fname}: done in {time.time()-t0:.0f}s, blocks so far {len(blocks)}", flush=True)
    return blocks


def fit_eval(Xtr, ytr, Xte, yte):
    if len(ytr) > MAX_TRAIN:
        i = np.random.default_rng(1).choice(len(ytr), MAX_TRAIN, replace=False)
        Xtr, ytr = Xtr[i], ytr[i]
    p = RandomForestClassifier(**RF).fit(Xtr, ytr).predict(Xte)
    return {"acc": float(accuracy_score(yte, p)), "bal_acc": float(balanced_accuracy_score(yte, p))}


def feats(b, which):
    return b[3] if which == "flow" else np.hstack([b[3], b[4]])


def main(folder):
    blocks = load_blocks(folder)
    ys = np.array([b[1] for b in blocks])
    out = {"n_blocks": len(blocks), "n_dos_blocks": int((ys == 0).sum()), "n_ddos_blocks": int((ys == 1).sum()),
           "rows_per_block_max": PER_BLOCK, "windows": list(W)}
    # ---- chronological within capture
    out["chrono"] = {}
    for name, which in (("per_flow", "flow"), ("per_flow+window", "both")):
        Xtr = np.vstack([feats(b, which)[b[2] < 0.85] for b in blocks])
        ytr = np.concatenate([np.full(int((b[2] < 0.85).sum()), b[1]) for b in blocks])
        Xte = np.vstack([feats(b, which)[b[2] >= 0.85] for b in blocks])
        yte = np.concatenate([np.full(int((b[2] >= 0.85).sum()), b[1]) for b in blocks])
        out["chrono"][name] = fit_eval(Xtr, ytr, Xte, yte)
        out["chrono"].update(n_train=int(len(ytr)), n_test=int(len(yte)), majority_floor=float(max(yte.mean(), 1 - yte.mean())))
        print("chrono", name, out["chrono"][name], flush=True)
    # ---- held-out captures
    out["heldout"] = []
    for rep in range(3):
        rng = np.random.default_rng(100 + rep); test_ids = set()
        for c in (0, 1):
            cand = np.flatnonzero(ys == c)
            test_ids |= set(rng.choice(cand, max(1, int(round(0.25 * len(cand)))), replace=False).tolist())
        trs = [b for i, b in enumerate(blocks) if i not in test_ids]
        tes = [b for i, b in enumerate(blocks) if i in test_ids]
        ytr = np.concatenate([np.full(len(b[3]), b[1]) for b in trs])
        yte = np.concatenate([np.full(len(b[3]), b[1]) for b in tes])
        r = {"rep": rep, "n_train_captures": len(trs), "n_test_captures": len(tes),
             "majority_floor": float(max(yte.mean(), 1 - yte.mean()))}
        for name, which in (("per_flow", "flow"), ("per_flow+window", "both")):
            r[name] = fit_eval(np.vstack([feats(b, which) for b in trs]), ytr,
                               np.vstack([feats(b, which) for b in tes]), yte)
        out["heldout"].append(r); print("heldout", r, flush=True)
    for k in ("per_flow", "per_flow+window"):
        a = [r[k]["acc"] for r in out["heldout"]]
        out["heldout_mean_sd_" + k] = [float(np.mean(a)), float(np.std(a, ddof=1))]
    Path(__file__).parent.joinpath("results").mkdir(exist_ok=True)
    out["feature_subset"] = SUBSET
    json.dump(out, open(Path(__file__).parent / "results" / f"dos_ddos_window_test_{SUBSET}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
