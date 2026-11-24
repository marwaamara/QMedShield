"""
audit.py -- SHA-256 hash-chained audit log, with a measurement harness.

Section 6.3 of the paper asserts tamper evidence, chronological integrity,
O(T) verification and sub-millisecond overhead. This module turns each
assertion into something measurable, and provides two comparison points: a
plain append-only log (what a conventional IDS gives you) and a Merkle tree
(what you would use if you needed O(log T) inclusion proofs).

Requires no model and no dataset.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from typing import Callable, Dict, List, Optional, Tuple


# =============================================================================
# The logs
# =============================================================================
class HashChainLog:
    """The audit log of section 6.3 of the paper.

    Record i commits to (hash_{i-1}, payload_i), so the chain head
    authenticates the entire history: altering any record invalidates every
    subsequent link.
    """

    GENESIS = "0" * 64

    def __init__(self):
        self.records: List[Dict] = []

    @staticmethod
    def digest(prev_hash: str, payload: Dict) -> str:
        return hashlib.sha256(
            (prev_hash + json.dumps(payload, sort_keys=True)).encode()
        ).hexdigest()

    def append(self, payload: Dict) -> str:
        prev = self.records[-1]["hash"] if self.records else self.GENESIS
        h = self.digest(prev, payload)
        self.records.append({"prev": prev, "payload": payload, "hash": h})
        return h

    @property
    def head(self) -> str:
        return self.records[-1]["hash"] if self.records else self.GENESIS

    def verify(self) -> Tuple[bool, Optional[int]]:
        """Return (is_valid, index_of_first_corruption). O(T) by construction."""
        prev = self.GENESIS
        for i, rec in enumerate(self.records):
            if rec["prev"] != prev:
                return False, i
            if self.digest(rec["prev"], rec["payload"]) != rec["hash"]:
                return False, i
            prev = rec["hash"]
        return True, None


class AppendOnlyLog:
    """Control: a conventional IDS log. No linkage, so a modified record is
    indistinguishable from a genuine one -- detection rate is identically 0."""

    def __init__(self):
        self.records: List[Dict] = []

    def append(self, payload: Dict) -> None:
        self.records.append({"payload": payload})

    def verify(self) -> Tuple[bool, Optional[int]]:
        return True, None


class MerkleLog:
    """Alternative: a Merkle tree over the same payloads.

    Same tamper evidence, but inclusion proofs are O(log T) rather than O(T).
    The trade-off is that the whole tree must be rebuilt (or incrementally
    maintained) rather than appended to in constant time -- which is why the
    paper's single-node, low-latency setting favours the chain.
    """

    def __init__(self):
        self.leaves: List[str] = []

    def append(self, payload: Dict) -> None:
        self.leaves.append(
            hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())

    def root(self) -> str:
        if not self.leaves:
            return "0" * 64
        level = list(self.leaves)
        while len(level) > 1:
            if len(level) % 2:
                level.append(level[-1])
            level = [hashlib.sha256((level[i] + level[i + 1]).encode()).hexdigest()
                     for i in range(0, len(level), 2)]
        return level[0]

    def proof_length(self) -> int:
        n, d = max(1, len(self.leaves)), 0
        while n > 1:
            n = (n + 1) // 2
            d += 1
        return d


# =============================================================================
# Event construction
# =============================================================================
def make_event(i: int, rng: random.Random, n_classes: int = 8) -> Dict:
    """A realistic IDS decision record.

    Section 6.3 states that the prediction, the SHAP/LIME explanatory features
    and a SHA-256 digest of the input are all committed, so the measured
    overhead must be for a payload of that size -- not for a toy string.
    """
    return {
        "event_id": i,
        "timestamp": 1.7e9 + i * 0.01,
        "predicted_class": rng.randrange(n_classes),
        "confidence": round(rng.random(), 6),
        "top_features": [f"pca_{rng.randrange(16)}" for _ in range(5)],
        "shap_values": [round(rng.uniform(-1, 1), 6) for _ in range(5)],
        "lime_features": [f"feat_{rng.randrange(25)}" for _ in range(5)],
        "input_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
    }


# =============================================================================
# Tamper harness
# =============================================================================
def tamper_inplace(chain: HashChainLog, mode: str,
                   rng: random.Random) -> Tuple[Optional[int], Callable[[], None]]:
    """Corrupt the chain IN PLACE; return (corrupted_index, undo).

    Deep-copying a 10^5-record chain per trial would dominate runtime and
    measure the copy, not the verification. Mutate and restore instead.
    """
    recs = chain.records
    n = len(recs)

    if mode == "modify":
        i = rng.randrange(n)
        old = recs[i]["payload"]["predicted_class"]
        recs[i]["payload"]["predicted_class"] = old ^ 1        # flip a verdict
        return i, lambda: recs[i]["payload"].__setitem__("predicted_class", old)

    if mode == "delete":
        i = rng.randrange(1, n)
        old_rec = recs.pop(i)
        return i, lambda: recs.insert(i, old_rec)

    if mode == "reorder":
        i = rng.randrange(n - 1)

        def swap():
            recs[i], recs[i + 1] = recs[i + 1], recs[i]
        swap()
        return i, swap

    if mode == "truncate":
        i = rng.randrange(n // 2, n)
        tail = recs[i:]
        del recs[i:]
        return None, lambda: recs.extend(tail)

    raise ValueError(f"unknown tamper mode: {mode}")


def detect_once(chain: HashChainLog, mode: str, rng: random.Random,
                head_before: str) -> Tuple[bool, bool]:
    """One tamper trial. Returns (detected, localised)."""
    i_corrupt, undo = tamper_inplace(chain, mode, rng)
    valid, idx = chain.verify()

    if mode == "truncate":
        # An intact prefix still verifies; truncation is caught by comparing
        # the retained head against the independently published one.
        detected = (not valid) or (chain.head != head_before)
    else:
        detected = not valid

    localised = bool(detected and i_corrupt is not None and idx is not None
                     and idx <= i_corrupt + 1)
    undo()
    return detected, localised


# =============================================================================
# Measurements
# =============================================================================
def measure_overhead(n_records: int, rng: random.Random) -> Dict:
    """Append cost, verification cost and storage for one chain length."""
    events = [make_event(i, rng) for i in range(n_records)]

    t0 = time.perf_counter()
    chain = HashChainLog()
    for e in events:
        chain.append(e)
    t_chain = time.perf_counter() - t0

    t0 = time.perf_counter()
    plain = AppendOnlyLog()
    for e in events:
        plain.append(e)
    t_plain = time.perf_counter() - t0

    t0 = time.perf_counter()
    merkle = MerkleLog()
    for e in events:
        merkle.append(e)
    _ = merkle.root()
    t_merkle = time.perf_counter() - t0

    t0 = time.perf_counter()
    chain.verify()
    t_verify = time.perf_counter() - t0

    size_mb = len(json.dumps(chain.records).encode()) / 1e6

    return {
        "n_records": n_records,
        "us_per_event_chain": t_chain / n_records * 1e6,
        "us_per_event_plain": t_plain / n_records * 1e6,
        "us_per_event_merkle": t_merkle / n_records * 1e6,
        "us_overhead_vs_plain": (t_chain - t_plain) / n_records * 1e6,
        "throughput_events_per_s": n_records / t_chain,
        "verify_total_s": t_verify,
        "us_per_record_verify": t_verify / n_records * 1e6,
        "size_mb": size_mb,
        "mb_per_million_events": size_mb / n_records * 1e6,
        "merkle_proof_length": merkle.proof_length(),
    }


def measure_detection(n_records: int, trials: int, attacks: List[str],
                      rng: random.Random) -> List[Dict]:
    chain = HashChainLog()
    for i in range(n_records):
        chain.append(make_event(i, rng))
    head = chain.head

    rows = []
    for mode in attacks:
        det = loc = 0
        for _ in range(trials):
            d, l = detect_once(chain, mode, rng, head)
            det += d
            loc += l
        # Truncation leaves an intact PREFIX, so verify() returns valid and
        # yields no corruption index: the attack is caught by comparing the
        # retained head, not by walking the chain. Localisation is therefore
        # not applicable rather than 0% -- reporting 0% would read as a failure.
        rows.append({
            "n_records": n_records, "attack": mode, "trials": trials,
            "detection_rate_pct": det / trials * 100,
            "localisation_rate_pct": (None if mode == "truncate"
                                      else loc / trials * 100),
            "localisation_note": ("n/a -- detected by head comparison"
                                  if mode == "truncate" else ""),
            "append_only_detection_pct": 0.0,   # the control, by construction
        })
    return rows


def project_storage(mb_per_million: float, flows_per_s: List[int]) -> List[Dict]:
    """Storage projection at realistic hospital flow rates."""
    rows = []
    for r in flows_per_s:
        per_day = r * 86_400
        rows.append({
            "flows_per_s": r,
            "events_per_day": per_day,
            "gb_per_day": per_day / 1e6 * mb_per_million / 1000,
            "gb_per_year": per_day * 365 / 1e6 * mb_per_million / 1000,
        })
    return rows


# =============================================================================
# Threat model -- what the mechanism does and does not provide
# =============================================================================
THREAT_MODEL = {
    "detects": [
        "silent modification of a past detection verdict or its explanation",
        "deletion of an inconvenient alert",
        "reordering of records to misrepresent the timeline",
        "truncation of the log tail (given an independently retained head)",
    ],
    "does_not_detect": [
        "an adversary who rewrites the whole chain AND replaces the retained "
        "head -- tamper evidence requires the head to be published or held "
        "outside the compromised node",
        "suppression of an event before it is ever logged",
        "a compromised model producing wrong but faithfully logged verdicts",
    ],
    "regulatory_mapping": {
        "HIPAA 164.312(b)": "audit controls -- record and examine activity",
        "HIPAA 164.312(c)(1)": "integrity -- protect from improper alteration",
        "EU MDR Annex I 17.2": "repeatability, reliability and traceability "
                               "of software decisions",
    },
}
