"""
audit_threat_model.py -- tamper-evidence assumptions of the hash-chain audit log.
Extends the original harness (modify / delete / reorder / truncate) with the two attacks
a hash chain does NOT stop on its own:
  * RECOMPUTE: an adversary with write access to the whole log edits a record and then
    recomputes every later link. Internal verification passes.
  * TRUNCATE-AND-CONTINUE: tail removed, then new records appended (head differs).
Each is evaluated (a) with internal verification only, (b) with the head digest AND record
count anchored in an independent trusted store.
"""
import sys, random, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from audit import HashChainLog, make_event
rng = random.Random(0); N = 2000; TR = 100
def build():
    c = HashChainLog(); [c.append(make_event(i, rng)) for i in range(N)]; return c
def rehash(c, start):
    prev = c.records[start - 1]["hash"] if start else HashChainLog.GENESIS
    for r in c.records[start:]:
        r["prev"] = prev; r["hash"] = HashChainLog.digest(prev, r["payload"]); prev = r["hash"]
res = {}
for mode in ("recompute", "truncate_continue", "recompute_last_only"):
    internal = anchored = 0
    for _ in range(TR):
        c = build(); head, n0 = c.head, len(c.records)
        if mode == "recompute":
            i = rng.randrange(N); c.records[i]["payload"]["predicted_class"] ^= 1; rehash(c, i)
        elif mode == "recompute_last_only":
            i = N - 1; c.records[i]["payload"]["predicted_class"] ^= 1; rehash(c, i)
        else:
            k = rng.randrange(N // 2, N); del c.records[k:]
            for j in range(k, N): c.append(make_event(j, rng))
        ok, _ = c.verify()
        internal += (not ok)
        anchored += ((not ok) or c.head != head or len(c.records) != n0)
    res[mode] = {"detected_internal_only_pct": internal, "detected_with_external_anchor_pct": anchored, "trials": TR}
print(json.dumps(res, indent=1)); json.dump(res, open(Path(__file__).parent / "results" / "audit_threat_model.json", "w"), indent=1)
