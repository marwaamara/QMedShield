"""
splits.py -- evaluation protocols and the enlarged test set.

A random train-test partition may allow correlated network flows originating
from the same capture sessions to appear in both training and testing subsets,
potentially leading to optimistic performance estimates. Session-level,
device-level and chronological splits provide stronger evidence regarding the
generalization ability of the framework in realistic deployment scenarios.

The pipeline of the paper is reproduced as section 4.1 describes it -- constant-column
removal, |r| > 0.95 correlation filter, 50,000-sample stratified pool,
StandardScaler, PCA-16, and *no* supervised univariate filter -- with ONE
correction:

    every fitting step runs on the TRAINING PORTION of each protocol only.

That includes the PCA, not merely the classifier. A
session-level split whose PCA was fitted over all sessions leaks through the
preprocessing and reproduces, in a subtler form, exactly the flaw that grouped
splits are meant to expose. `assert_fitted_on_train_only` (validation gate G5) refuses to continue if
any component saw more rows than the training split contains.

Protocols
---------
random   stratified, seeded -- the protocol of the paper, kept for comparison
session  grouped by source capture file; no capture contributes to two splits
device   grouped by device identity; INFEASIBLE unless the CSVs carry a device,
         MAC or IP column -- reported as infeasible with the reason, never
         silently approximated by something else
chrono   time-ordered within each capture: first 70% train, next 15% val, last
         15% test, so the model is always tested on later traffic
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import (
    MIN_TEST_PER_CLASS,
    N_CLASSES,
    N_FEATURES,
    N_PCA,
    POOL_SIZE,
    TEST_SIZE_TARGET,
    TRAIN_FRAC,
    VAL_FRAC,
)

# Placeholder tokens that flow exporters write instead of a number. Without
# these, pandas reads the whole column as text, every value coerces to 0, the
# column looks constant, and the failure surfaces far from its cause -- which
# is exactly what "only 1 non-constant feature" means.
NA_TOKENS = ["n/a", "N/A", "n.a.", "na", "NA", "-", "--", "?", "NULL", "null",
             "None", "none", "NaN", "nan", "inf", "-inf", "Infinity",
             "-Infinity", "#DIV/0!", "#VALUE!"]

# Candidate column names that would make a device-level split possible.
DEVICE_COLUMN_CANDIDATES = ("device", "device_id", "mac", "src_mac", "srcmac",
                            "eth.src", "src_ip", "srcip", "ip.src", "source_ip")


# =============================================================================
# Labels -- byte-for-byte the taxonomy of the paper
# =============================================================================
def parse_label(label_str) -> Tuple[int, int]:
    """Raw CICIoMT2024 label -> (binary, 8-class). Copied unchanged from the
    original `01_Data_Preparation`; changing it would change the task."""
    label = str(label_str).lower().strip()
    binary = 0 if "benign" in label else 1
    if "benign" in label:
        multiclass = 0
    elif "spoof" in label or "arp" in label:
        multiclass = 1
    elif any(k in label for k in ("recon", "scan", "vulscan", "os_detect",
                                  "portscan")):
        multiclass = 2
    elif "malformed" in label:
        multiclass = 3
    elif "mqtt" in label and "ddos" in label:
        multiclass = 5
    elif "mqtt" in label and "dos" in label:
        multiclass = 4
    elif "ddos" in label:
        multiclass = 7
    elif "dos" in label:
        multiclass = 6
    else:
        multiclass = 1
    return binary, multiclass


# =============================================================================
# Raw CSVs -> flows + provenance
# =============================================================================
def load_raw_flows(data_dir) -> Dict:
    """Read every CSV in `data_dir`, attaching capture-session provenance.

    `source_csv` is the capture-session proxy and `row_order` the chronological
    proxy: the archive preserved neither, which is why the split protocols
    cannot be built from it and need the CSVs back.
    """
    from pathlib import Path

    import pandas as pd

    paths = sorted(Path(data_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no CSVs in {data_dir}. The CICIoMT2024 WiFi/MQTT partition is "
            f"required for the session/device/chronological protocols: "
            f"https://www.unb.ca/cic/datasets/iomt-dataset-2024.html")

    frames = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        df["__source_csv"] = p.name
        df["__row_order"] = np.arange(len(df))
        if "label" not in df.columns:
            # Several distributions encode the class in the file name only.
            df["label"] = p.stem
        frames.append(df)
    flows = pd.concat(frames, ignore_index=True)

    parsed = flows["label"].apply(parse_label)
    y_binary = np.array([t[0] for t in parsed], dtype=np.int64)
    y_multi = np.array([t[1] for t in parsed], dtype=np.int64)

    device_col = next((c for c in flows.columns
                       if c.lower() in DEVICE_COLUMN_CANDIDATES), None)

    return {"flows": flows, "y_binary": y_binary, "y_multiclass": y_multi,
            "source_csv": flows["__source_csv"].to_numpy(),
            "row_order": flows["__row_order"].to_numpy(),
            "device": (flows[device_col].to_numpy() if device_col else None),
            "device_column": device_col,
            "n_sources": int(flows["__source_csv"].nunique()),
            "paths": [str(p) for p in paths]}


def _peek(path, n: int = 2000):
    """First rows of a CSV, with delimiter sniffing.

    A file exported with ';' or a tab reads back as ONE column of joined text
    under the default comma. Everything downstream then looks empty for reasons
    that have nothing to do with the data, so detect it here.
    """
    import pandas as pd

    head = pd.read_csv(path, nrows=n, low_memory=False,
                       na_values=NA_TOKENS, keep_default_na=True)
    if head.shape[1] == 1:
        sniffed = pd.read_csv(path, nrows=n, sep=None, engine="python",
                              na_values=NA_TOKENS)
        if sniffed.shape[1] > 1:
            raise ValueError(
                f"{path.name} parses as a single column with the default "
                f"comma separator but as {sniffed.shape[1]} columns when the "
                f"delimiter is sniffed. Re-export it as comma-separated, or "
                f"tell me and I will thread a `sep` setting through the "
                f"loader.")
    return head


def _numeric_columns(head, columns=None) -> List[str]:
    """Numeric feature columns, decided from a 200-row peek."""
    import pandas as pd

    out, rejected = [], []
    for c in (columns if columns is not None else head.columns):
        if c == "label" or str(c).startswith("__"):
            continue
        vals = head[c]
        # Judge ONLY the cells that hold something. A column that is blank in
        # the peeked rows says nothing about whether it is numeric, and
        # rejecting it here is how 25 features silently became 1.
        nonempty = vals[vals.notna() & (vals.astype(str).str.strip() != "")]
        if len(nonempty) == 0:
            rejected.append(f"{c}(empty in peek)")
            continue
        ok = pd.to_numeric(nonempty, errors="coerce").notna().mean()
        if ok >= 0.9:
            out.append(c)
        else:
            sample = nonempty.astype(str).iloc[0][:20]
            rejected.append(f"{c}(e.g. {sample!r})")
    if rejected:
        print(f"  non-numeric columns ignored ({len(rejected)}): "
              f"{rejected[:6]}{' ...' if len(rejected) > 6 else ''}", flush=True)
    return out


def profile_csvs(data_dir, n: int = 2000) -> None:
    """Print what the CSVs in `data_dir` actually contain.

    Cheap (a few thousand rows per file) and worth running before anything
    else: every failure so far has come down to the files not being the
    flow-feature export the pipeline expects, and this says so in one screen.
    """
    from pathlib import Path

    import pandas as pd

    paths = sorted(Path(data_dir).glob("*.csv"))
    print(f"{len(paths)} CSV file(s) in {data_dir}" + chr(10))
    for path in paths:
        try:
            head = _peek(path, n)
        except Exception as e:
            print(f"{path.name}: UNREADABLE -- {type(e).__name__}: {e}")
            continue
        num, txt = [], []
        missing_frac = {}
        for c in head.columns:
            v = head[c]
            ne = v[v.notna() & (v.astype(str).str.strip() != "")]
            missing_frac[c] = 1.0 - len(ne) / max(len(v), 1)
            frac = (pd.to_numeric(ne, errors="coerce").notna().mean()
                    if len(ne) else 0.0)
            (num if frac >= 0.9 else txt).append(c)
        mostly_missing = [c for c, f in missing_frac.items() if f > 0.5]
        if mostly_missing:
            print(f"   >50% missing/placeholder in {len(mostly_missing)} "
                  f"column(s): {mostly_missing[:8]}")
        size_mb = path.stat().st_size / 1e6
        print(f"{path.name}  ({size_mb:,.0f} MB)")
        print(f"   columns : {len(head.columns)}   numeric: {len(num)}   "
              f"non-numeric/empty: {len(txt)}")
        print(f"   numeric : {num[:12]}{' ...' if len(num) > 12 else ''}")
        if txt:
            print(f"   other   : {txt[:8]}{' ...' if len(txt) > 8 else ''}")
        with open(path, "r", errors="replace") as fh:
            print(f"   header  : {fh.readline().strip()[:160]}")
            print(f"   row 1   : {fh.readline().strip()[:160]}")
        print()


def diagnose_features(data_dir, X=None, names=None, n: int = 5) -> str:
    """A self-contained diagnosis, formatted to be pasted into a message.

    Raised INSIDE the exception rather than merely printed: a traceback is what
    gets copied when something fails, so the traceback has to carry the
    evidence. Reports the raw bytes of the first lines, the field counts under
    each candidate separator, per-column parse rates, and -- if a matrix was
    built -- which columns came out constant.
    """
    from pathlib import Path

    import pandas as pd

    out = ["", "=" * 68, "DIAGNOSIS -- paste this whole block when reporting",
           "=" * 68]
    paths = sorted(Path(data_dir).glob("*.csv"))
    out.append(f"{len(paths)} CSV file(s) in {data_dir}")

    for path in paths[:3]:
        out.append("")
        out.append(f"--- {path.name}  ({path.stat().st_size / 1e6:,.1f} MB) ---")
        try:
            with open(path, "r", errors="replace") as fh:
                raw = [fh.readline().rstrip() for _ in range(3)]
        except Exception as e:
            out.append(f"  unreadable: {e}")
            continue
        for i, line in enumerate(raw):
            out.append(f"  line {i}: {line[:150]}")
        for sep, nm in ((",", "comma"), (";", "semicolon"), ("	", "tab"),
                        ("|", "pipe")):
            counts = [len(l.split(sep)) for l in raw if l]
            out.append(f"  fields per line, {nm:9s} separator: {counts}")

        try:
            head = pd.read_csv(path, nrows=500, low_memory=False,
                               na_values=NA_TOKENS, keep_default_na=True)
            out.append(f"  pandas sees {head.shape[1]} column(s), "
                       f"{len(head)} row(s) peeked")
            rates = []
            for c in head.columns[:n]:
                v = head[c]
                ne = v[v.notna()]
                rate = (pd.to_numeric(ne, errors="coerce").notna().mean()
                        if len(ne) else 0.0)
                rates.append(f"{str(c)[:18]}: {len(ne)}/{len(v)} present, "
                             f"{100 * rate:.0f}% numeric")
            out.append(f"  first {n} columns -> " + " | ".join(rates))
            allna = [str(c) for c in head.columns if head[c].notna().sum() == 0]
            if allna:
                out.append(f"  columns with NO values at all ({len(allna)}): "
                           f"{allna[:10]}")
        except Exception as e:
            out.append(f"  pandas failed: {type(e).__name__}: {e}")

    if X is not None:
        import numpy as _np
        sd = _np.asarray(X).std(axis=0)
        const = [names[i] if names else i for i in range(len(sd)) if sd[i] == 0]
        out.append("")
        out.append(f"pooled matrix: {X.shape[0]} x {X.shape[1]}, "
                   f"{len(const)} constant column(s)")
        out.append(f"  constant: {const[:12]}{' ...' if len(const) > 12 else ''}")
        live = [(names[i] if names else i, float(sd[i]))
                for i in range(len(sd)) if sd[i] > 0][:5]
        out.append(f"  non-constant (name, std): {live}")
    out.append("=" * 68)
    return chr(10).join(out)


def scan_flows(data_dir, chunksize: int = 500_000) -> Dict:
    """PASS 1: labels and provenance only, read in chunks.

    Why this exists
    ---------------
    The CICIoMT2024 WiFi/MQTT partition is several million flows. Reading every
    column of every file into one DataFrame and then casting the whole thing to
    float64 needs many gigabytes, and a Jupyter kernel that exceeds what the
    node allows is killed by the OS -- which surfaces as a notebook that simply
    stops, with no Python traceback to explain it.

    Only three things are needed from every row: its label, which capture it
    came from, and its position in that capture. Those are at most two columns,
    so this pass costs tens of megabytes however large the CSVs are. Features
    are read afterwards, for the pooled rows only (`load_features_for`).

    Every decision is made PER FILE. The CIC distribution ships one CSV per
    attack class with the class in the FILE NAME and no `label` column at all,
    while other exports carry a `label` column; both occur, sometimes in the
    same directory, and a header taken from the first file does not describe
    the rest.
    """
    from pathlib import Path

    import numpy as _np
    import pandas as pd

    paths = sorted(Path(data_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no CSVs in {data_dir}. The CICIoMT2024 WiFi/MQTT partition is "
            f"required for the session/device/chronological protocols: "
            f"https://www.unb.ca/cic/datasets/iomt-dataset-2024.html")

    heads = {p: _peek(p) for p in paths}

    # Keep only the files that share the DOMINANT schema.
    #
    # A data directory legitimately holds more than one dataset -- notebook 12
    # reads WUSTL-EHMS-2020 from the same place. Intersecting column names
    # across everything then collapses to whatever the schemas happen to share
    # (with WUSTL present, exactly one column: `Rate`), and the run dies with
    # "1 non-constant feature". Group by column signature, take the group
    # holding the most data, and say which files were set aside.
    groups: Dict[tuple, list] = {}
    for q in paths:
        groups.setdefault(tuple(heads[q].columns), []).append(q)
    if len(groups) > 1:
        chosen = max(groups.values(),
                     key=lambda g: sum(x.stat().st_size for x in g))
        dropped = [x.name for g in groups.values() if g is not chosen
                   for x in g]
        print(f"  {len(groups)} different schemas in this directory; using the "
              f"{len(chosen)} file(s) with the dominant one", flush=True)
        print(f"  ignored (different dataset): {dropped}", flush=True)
        paths = sorted(chosen)

    common = list(heads[paths[0]].columns)
    if any(tuple(heads[q].columns) != tuple(common) for q in paths):
        common = [c for c in heads[paths[0]].columns
                  if all(c in heads[q].columns for q in paths)]
        print(f"  columns common to the selected files: {len(common)}",
              flush=True)

    cols = _numeric_columns(heads[paths[0]], common)

    # The device column only counts if EVERY file has it -- a device split
    # built from a column that half the captures lack is not a device split.
    device_col = next((c for c in common
                       if c.lower() in DEVICE_COLUMN_CANDIDATES), None)
    any_device = next((c for p in paths for c in heads[p].columns
                       if c.lower() in DEVICE_COLUMN_CANDIDATES), None)
    if device_col is None and any_device:
        print(f"  '{any_device}' exists in some files but not all -- the "
              f"device protocol will be reported infeasible", flush=True)

    labels, sources, orders, devices = [], [], [], []
    for path in paths:
        has_label = "label" in heads[path].columns
        want = [c for c in ("label" if has_label else None, device_col) if c]
        # Nothing to read? Take the first column purely to count the rows.
        usecols = want or [heads[path].columns[0]]

        n_file = 0
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize,
                                 low_memory=False, na_values=NA_TOKENS,
                                 keep_default_na=True):
            n = len(chunk)
            labels.append(chunk["label"].to_numpy() if has_label
                          else _np.full(n, path.stem, dtype=object))
            sources.append(_np.full(n, path.name, dtype=object))
            orders.append(_np.arange(n_file, n_file + n))
            if device_col:
                devices.append(chunk[device_col].to_numpy())
            n_file += n
        print(f"  scanned {path.name}: {n_file:,} rows"
              f"{'' if has_label else '  (class taken from the file name)'}",
              flush=True)

    raw_labels = _np.concatenate(labels)

    # Parse each DISTINCT label once and map. A per-row Python call would take
    # minutes on a multi-million-row partition; there are ~18 distinct labels.
    uniq, inverse = _np.unique(raw_labels, return_inverse=True)
    table = _np.array([parse_label(v) for v in uniq], dtype=_np.int64)
    print(f"  {len(uniq)} distinct label(s) -> 8-class taxonomy: "
          f"{ {str(u): int(t[1]) for u, t in zip(uniq[:10], table[:10])} }",
          flush=True)

    return {"y_binary": table[inverse, 0],
            "y_multiclass": table[inverse, 1],
            "source_csv": _np.concatenate(sources),
            "row_order": _np.concatenate(orders),
            "device": (_np.concatenate(devices) if device_col else None),
            "device_column": device_col,
            "n_sources": len(paths),
            "paths": [str(x) for x in paths],
            "numeric_columns": cols,
            "n_rows": int(len(raw_labels))}


def load_features_for(data_dir, keep_idx, numeric_columns: Sequence[str],
                      dtype=np.float32, chunksize: int = 200_000,
                      paths: Optional[Sequence] = None):
    """PASS 2: the numeric features of `keep_idx` (GLOBAL row indices) only.

    float32 by default: the model consumes float32 anyway, and float64 would
    double the footprint for no gain. Returns (X, feature_names) with the rows
    in the same order as `keep_idx`.
    """
    from pathlib import Path

    import pandas as pd

    keep = np.sort(np.asarray(keep_idx))
    # MUST be the same file list, in the same order, that pass 1 scanned:
    # `keep_idx` are global row numbers over that sequence. Re-globbing here
    # would silently include files pass 1 excluded and shift every offset.
    paths = ([Path(x) for x in paths] if paths is not None
             else sorted(Path(data_dir).glob("*.csv")))

    blocks, taken, offset = [], [], 0
    for path in paths:
        for chunk in pd.read_csv(path, usecols=list(numeric_columns),
                                 chunksize=chunksize, low_memory=False,
                                 na_values=NA_TOKENS, keep_default_na=True):
            n = len(chunk)
            g = np.arange(offset, offset + n)
            # Vectorised: `keep` is sorted, so searchsorted beats a per-row
            # membership test by orders of magnitude at millions of rows.
            pos = np.searchsorted(keep, g)
            pos[pos >= len(keep)] = 0
            sel = keep[pos] == g
            if sel.any():
                # `usecols` returns columns in FILE order, which may differ
                # between captures; reindex so column j means the same feature
                # in every block.
                block = chunk.loc[sel, list(numeric_columns)]
                block = block.apply(pd.to_numeric, errors="coerce")
                block = block.fillna(0.0).replace([np.inf, -np.inf], 0.0)
                blocks.append(block.to_numpy(dtype=dtype))
                taken.append(g[sel])
            offset += n

    if not blocks:
        raise ValueError("no rows selected; keep_idx does not match the CSVs")

    X = np.vstack(blocks)
    order = np.argsort(np.concatenate(taken))
    X = X[order]

    # Restore the caller's ordering of keep_idx (np.sort above sorted it).
    back = np.argsort(np.argsort(np.asarray(keep_idx)))
    print(f"  loaded features for {len(X):,} rows "
          f"({X.nbytes / 1e6:.0f} MB, {X.dtype})", flush=True)
    return X[back], list(numeric_columns)


def numeric_matrix(flows) -> Tuple[np.ndarray, List[str]]:
    """All numeric feature columns, NaN/inf-cleaned. No selection yet: feature
    selection is label-dependent and therefore belongs inside the fold."""
    import pandas as pd

    cols = [c for c in flows.columns
            if c != "label" and not str(c).startswith("__")]
    X = flows[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return X.to_numpy(dtype=np.float64), list(cols)


# =============================================================================
# Protocols
# =============================================================================
def _stratified_three_way(y: np.ndarray, seed: int,
                          frac=(TRAIN_FRAC, VAL_FRAC)) -> Tuple[np.ndarray, ...]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=frac[0], stratify=y,
                                random_state=seed)
    rel = frac[1] / (1.0 - frac[0])
    va, te = train_test_split(rest, train_size=rel, stratify=y[rest],
                              random_state=seed)
    return tr, va, te


def _grouped_three_way(groups: np.ndarray, seed: int) -> Tuple[np.ndarray, ...]:
    """Assign whole groups to train/val/test by cumulative size."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    sizes = {g: int((groups == g).sum()) for g in uniq}
    total = sum(sizes.values())

    buckets: Dict[str, List] = {"train": [], "val": [], "test": []}
    acc = 0
    for g in uniq:
        frac = acc / total
        key = ("train" if frac < TRAIN_FRAC
               else "val" if frac < TRAIN_FRAC + VAL_FRAC else "test")
        buckets[key].append(g)
        acc += sizes[g]

    # A protocol that leaves a split empty is not a protocol.
    for key in ("train", "val", "test"):
        if not buckets[key]:
            donor = max(buckets, key=lambda k: len(buckets[k]))
            if len(buckets[donor]) < 2:
                raise ValueError("too few groups for a grouped split")
            buckets[key].append(buckets[donor].pop())

    sel = lambda gs: np.flatnonzero(np.isin(groups, gs))
    return sel(buckets["train"]), sel(buckets["val"]), sel(buckets["test"])


def _chronological(source: np.ndarray, order: np.ndarray) -> Tuple[np.ndarray, ...]:
    """First 70% / next 15% / last 15% of every capture, in packet order.

    Splitting inside each capture rather than across captures keeps every class
    present in every split -- a global time cut would put whole attack families
    on one side, which measures class coverage rather than temporal
    generalisation.
    """
    tr, va, te = [], [], []
    for s in np.unique(source):
        rows = np.flatnonzero(source == s)
        rows = rows[np.argsort(order[rows])]
        n = len(rows)
        a, b = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
        tr.append(rows[:a]); va.append(rows[a:b]); te.append(rows[b:])
    return (np.sort(np.concatenate(tr)), np.sort(np.concatenate(va)),
            np.sort(np.concatenate(te)))


def protocol_indices(raw: Dict, protocol: str, y: np.ndarray,
                     seed: int = 42) -> Dict:
    """Train/val/test indices for one protocol, plus a feasibility verdict.

    An infeasible protocol returns `{"feasible": False, "reason": ...}` instead
    of an approximation. Reporting "device-level split unavailable because the
    published CSVs carry no device identifier" is a result; silently
    substituting something else is not.
    """
    if protocol == "random":
        tr, va, te = _stratified_three_way(y, seed)
    elif protocol == "session":
        if raw["n_sources"] < 3:
            return {"feasible": False,
                    "reason": f"only {raw['n_sources']} capture file(s); a "
                              f"session-level split needs at least 3"}
        tr, va, te = _grouped_three_way(raw["source_csv"], seed)
    elif protocol == "device":
        if raw["device"] is None:
            return {"feasible": False,
                    "reason": "no device / MAC / IP column in the published "
                              "CICIoMT2024 WiFi-MQTT CSVs, so device identity "
                              "cannot be recovered; searched for "
                              + ", ".join(DEVICE_COLUMN_CANDIDATES)}
        if len(np.unique(raw["device"])) < 3:
            return {"feasible": False,
                    "reason": f"only {len(np.unique(raw['device']))} distinct "
                              f"devices in column '{raw['device_column']}'"}
        tr, va, te = _grouped_three_way(raw["device"], seed)
    elif protocol == "chrono":
        tr, va, te = _chronological(raw["source_csv"], raw["row_order"])
    else:
        raise ValueError(f"unknown protocol: {protocol}")

    assert_disjoint(tr, va, te)

    # A capture-grouped split is only meaningful if the test split still
    # contains several classes. The CIC distribution ships ONE CSV PER ATTACK
    # CLASS, so grouping by capture file is then identical to holding out whole
    # classes: the model is asked to label traffic of a class it never saw, and
    # the resulting accuracy measures nothing about session generalisation.
    # Report that as infeasible rather than publishing the number.
    n_tr_cls = len(np.unique(y[tr]))
    n_te_cls = len(np.unique(y[te]))
    if n_te_cls < 2 or n_tr_cls < 2:
        return {"feasible": False,
                "reason": (f"{protocol} split is degenerate: the test split "
                           f"holds {n_te_cls} class(es) and the training split "
                           f"{n_tr_cls}. Each capture file carries a single "
                           f"class, so grouping by capture is a class holdout, "
                           f"not a {protocol} split. Report this rather than "
                           f"the accuracy it would produce.")}

    coverage = n_te_cls / len(np.unique(y))
    return {"feasible": True, "protocol": protocol, "seed": seed,
            "test_class_coverage": coverage,
            "train_idx": tr, "val_idx": va, "test_idx": te,
            "n_train": int(len(tr)), "n_val": int(len(va)),
            "n_test": int(len(te)),
            "class_counts": {"train": np.bincount(y[tr], minlength=N_CLASSES).tolist(),
                             "val": np.bincount(y[va], minlength=N_CLASSES).tolist(),
                             "test": np.bincount(y[te], minlength=N_CLASSES).tolist()}}


def assert_disjoint(train_idx, val_idx, test_idx) -> None:
    """Gate G3, at the index level."""
    tr, va, te = set(map(int, train_idx)), set(map(int, val_idx)), set(map(int, test_idx))
    for a, b, na, nb in ((tr, te, "train", "test"), (va, te, "val", "test"),
                         (tr, va, "train", "val")):
        if a & b:
            raise AssertionError(f"LEAKAGE: {len(a & b)} indices shared between "
                                 f"{na} and {nb}")


# =============================================================================
# Preprocessing -- fitted INSIDE the fold (gate G5)
# =============================================================================
class FoldPreprocessor:
    """Constant filter -> correlation filter -> scaler -> PCA.

    The pipeline of the paper, with every stage fitted on the training rows of the
    current protocol and nothing else. `n_fit_rows` records how many rows the
    fit saw, so gate G5 can check it against the training split.

    On the supervised filter -- read this before changing it
    --------------------------------------------------------
    Section 4.1 of the paper is explicit:

        "We do not apply a supervised univariate filter (such as ANOVA
         SelectKBest) in addition to the correlation filter: the 25 features
         surviving stage 2 already span the operationally meaningful traffic
         descriptors."

    So the default here is NO supervised selection: whatever survives stages 1
    and 2 goes forward, exactly as the paper describes. The archived run kept
    25 features and the paper reports 25, so on that data the two agree.

    If stage 2 leaves MORE than `n_features` on your copy of the CSVs, the
    paper's claim does not hold there, and that must not be papered over: `fit`
    then applies an ANOVA cap to stay dimensionally comparable with the
    archived model, sets `anova_applied = True`, and prints a warning naming
    the counts. Report those counts if it happens -- section 4.1 would need
    correcting. Fitting the ANOVA on training rows only keeps it leakage-free
    either way (gate G5).
    """

    def __init__(self, n_features: int = N_FEATURES, n_pca: int = N_PCA,
                 corr_threshold: float = 0.95, seed: int = 42):
        self.n_features, self.n_pca = n_features, n_pca
        self.corr_threshold, self.seed = corr_threshold, seed
        self.n_fit_rows: Optional[int] = None
        self.kept_names: List[str] = []
        self.selector_ = None            # stays None when the paper's claim holds
        self.anova_applied = False
        self.stage_counts: Dict[str, int] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        self.n_fit_rows = int(len(X))
        names = list(feature_names)
        self.stage_counts["input"] = len(names)

        # stage 1 -- constant columns
        self.non_const_ = np.std(X, axis=0) > 0
        Xf = X[:, self.non_const_]
        names = [n for n, k in zip(names, self.non_const_) if k]
        self.stage_counts["after_constant_filter"] = len(names)

        if len(names) < 2:
            raise ValueError(
                f"only {len(names)} feature(s) survive the constant-column "
                f"filter, out of {self.stage_counts['input']} numeric columns "
                f"read. Every other column is constant across the sampled "
                f"rows, which normally means the features did not parse as "
                f"numbers (wrong delimiter, thousands separators, or a raw "
                f"packet export rather than the flow-feature CSVs). The "
                f"paper's Table 5 expects 25 informative features. Surviving: "
                f"{names}")

        # stage 2 -- |r| > 0.95 redundancy filter
        # atleast_2d: np.corrcoef returns a SCALAR for a single column, and
        # np.triu then fails with an opaque "tri() missing argument 'N'".
        corr = np.atleast_2d(np.nan_to_num(np.corrcoef(Xf.T), nan=0.0))
        upper = np.triu(np.abs(corr), k=1)
        drop = {j for i in range(upper.shape[0])
                for j in range(i + 1, upper.shape[1])
                if upper[i, j] > self.corr_threshold}
        self.corr_keep_ = [i for i in range(Xf.shape[1]) if i not in drop]
        Xf = Xf[:, self.corr_keep_]
        names = [names[i] for i in self.corr_keep_]
        self.stage_counts["after_correlation_filter"] = len(names)

        # No supervised filter -- unless stage 2 left more than the paper says.
        if len(names) > self.n_features:
            from sklearn.feature_selection import SelectKBest, f_classif
            self.selector_ = SelectKBest(f_classif, k=self.n_features).fit(Xf, y)
            Xf = self.selector_.transform(Xf)
            mask = self.selector_.get_support()
            names = [n for n, m in zip(names, mask) if m]
            self.anova_applied = True
            print(f"[splits] WARNING: stage 2 left "
                  f"{self.stage_counts['after_correlation_filter']} features, "
                  f"but section 4.1 reports {self.n_features} and states that "
                  f"no supervised filter is applied. An ANOVA cap to "
                  f"{self.n_features} was applied to stay comparable with the "
                  f"archived model. REPORT THIS -- section 4.1 needs correcting.")
        elif len(names) < self.n_features:
            print(f"[splits] WARNING: only {len(names)} features survive stages "
                  f"1-2, against the {self.n_features} the paper reports. The "
                  f"CSVs in data/ may not be the same WiFi/MQTT partition.")

        self.kept_names = names
        self.stage_counts["final"] = len(names)

        self.scaler_ = StandardScaler().fit(Xf)
        self.pca_ = PCA(n_components=min(self.n_pca, Xf.shape[1]),
                        random_state=self.seed).fit(self.scaler_.transform(Xf))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xf = X[:, self.non_const_][:, self.corr_keep_]
        if self.selector_ is not None:
            Xf = self.selector_.transform(Xf)
        return self.pca_.transform(self.scaler_.transform(Xf))

    def as_dict(self) -> Dict:
        out = {"scaler": self.scaler_, "pca": self.pca_}
        if self.selector_ is not None:
            out["selector"] = self.selector_
        return out

    def provenance(self) -> Dict:
        """What the paper claims against what actually happened. Written into
        the run manifest so the difference cannot be lost."""
        return {"stage_counts": self.stage_counts,
                "kept_features": list(self.kept_names),
                "anova_applied": self.anova_applied,
                "paper_claim": "section 4.1: no supervised univariate filter; "
                               "25 features survive stages 1-2",
                "explained_variance": float(
                    self.pca_.explained_variance_ratio_.sum()),
                "n_fit_rows": self.n_fit_rows}


def assert_fitted_on_train_only(pre: FoldPreprocessor, train_idx) -> None:
    """Validation gate G5.

    The whole point of a session-level split is that nothing crosses the
    session boundary. If the PCA was fitted on more rows than the training
    split holds, it crossed, and every number downstream is optimistic in
    precisely the way grouped-split evaluation is designed to catch.
    """
    if pre.n_fit_rows is None:
        raise AssertionError("gate G5: preprocessor was never fitted")
    if pre.n_fit_rows != len(train_idx):
        raise AssertionError(
            f"gate G5 FAILED: preprocessing was fitted on {pre.n_fit_rows} rows "
            f"but the training split holds {len(train_idx)}. Fit the scaler, "
            f"the ANOVA selector and the PCA inside the fold.")


# =============================================================================
# Pool and enlarged test set
# =============================================================================
def draw_pool(y: np.ndarray, size: int = POOL_SIZE, seed: int = 42) -> np.ndarray:
    """The 50,000-sample stratified pool of the paper."""
    from sklearn.model_selection import train_test_split

    if len(y) <= size:
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=size, stratify=y,
                              random_state=seed)
    return np.sort(idx)


def identify_archived_pool_rows(X_scaled_all: np.ndarray,
                                X_scaled_pool: np.ndarray,
                                tol: float = 1e-6) -> np.ndarray:
    """Boolean mask of rows that match the archived 50k pool.

    Used to keep the enlarged test set DISJOINT from the data the archived
    model was trained on. Matching is conservative: any row within `tol` of a
    pool row is excluded, so a duplicated flow is dropped rather than risked.
    """
    from scipy.spatial import cKDTree

    d, _ = cKDTree(np.asarray(X_scaled_pool)).query(np.asarray(X_scaled_all), k=1)
    return d < tol


def enlarged_test_indices(y: np.ndarray, excluded: np.ndarray,
                          min_per_class: int = MIN_TEST_PER_CLASS,
                          target: int = TEST_SIZE_TARGET,
                          seed: int = 42) -> Dict:
    """Draw a disjoint test set with at least `min_per_class` rows per class.

    The paper's per-class table reports F1 on n=4 and n=15. Those estimates
    are not defensible at any confidence level, and inference is cheap, so the
    fix is simply to test on more rows -- while recording the TRUE prevalence
    separately, because the enlarged set is deliberately not prevalence-faithful
    and its weighted metrics must be read with that in mind.
    """
    rng = np.random.default_rng(seed)
    avail = np.flatnonzero(~np.asarray(excluded))
    y_av = np.asarray(y)[avail]

    chosen, shortfalls = [], {}
    for c in range(N_CLASSES):
        pool = avail[y_av == c]
        take = min(len(pool), max(min_per_class, int(target / N_CLASSES)))
        if len(pool) < min_per_class:
            shortfalls[int(c)] = int(len(pool))
        if take:
            chosen.append(rng.choice(pool, size=take, replace=False))

    idx = np.sort(np.concatenate(chosen)) if chosen else np.array([], dtype=int)
    counts = np.bincount(np.asarray(y)[idx], minlength=N_CLASSES)
    true_prev = np.bincount(np.asarray(y), minlength=N_CLASSES) / len(y)
    return {"test_idx": idx, "counts": counts.tolist(),
            "true_prevalence": true_prev.tolist(),
            "min_per_class_target": int(min_per_class),
            "classes_below_target": shortfalls,
            "note": "class-balanced by construction; weighted metrics on this "
                    "set do NOT reflect operational prevalence -- report "
                    "per-class metrics here and prevalence-weighted metrics on "
                    "the original test split."}
