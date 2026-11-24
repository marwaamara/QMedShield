# QMedShield — Grid'5000 package

Self-contained. No spaces in any path. Copy the whole folder and run.

```bash
rsync -av qmedshield_g5k/ g5k:~/qmedshield_g5k/
```

---

## 1. One-time setup (frontend)

```bash
cd ~/qmedshield_g5k
./oar/setup_env.sh                 # builds ~/qmed_env from requirements.txt
source ~/qmed_env/activate         # or: source ~/qmed_env/bin/activate
```

## 2. Verify before spending any walltime

```bash
python verify_deploy.py
```

Checks packages, the archive, that the archived weights **reproduce the published metrics
bit-exactly**, that all 17 notebooks parse with their preflight substituted, and that the run
order satisfies every dependency. Exit code 0 means it is safe to submit. Anything else names
the problem.

Expect one warning if `shap`/`lime` are absent (only notebook 09 needs them) and one if the raw
CSVs are absent (notebooks 01 and 16 need them; 16 cannot build the session/device/chronological protocols or the enlarged test set without them).

## 3. Submit

```bash
./oar/submit_archive_only.sh    # 00,01,04,07,08,09,10,11,13 — 1 node, ~6 h, NO training
./oar/submit_training.sh        # 02,03,04,05,06,14,15,16   — 1 node, ~24 h
./oar/submit_energy.sh          # 11 with wattmeters       — EXCLUSIVE node
```

Or interactively:

```bash
./run_all.sh                    # everything, in dependency order
QUICK_MODE=1 ./run_all.sh       # fast pass to validate the DAG first
./run_all.sh 07 08 11           # only these
```

**Run `QUICK_MODE=1 ./run_all.sh` once before the real submission.** It exercises every code
path with 3 seeds and 2-epoch training in a fraction of the time, and it is the cheapest way to
discover a problem that would otherwise surface an hour into a 24-hour job.

---

## 4. Run order and dependencies

Only two hard links exist; everything else is independent and can run in any order or in
parallel:

```
00_verify_archive          (no dependencies)  <- ALWAYS FIRST
  └─> 04_baselines_multiseed        needs 00's QCNN predictions for paired McNemar
  └─> 14_quantum_related_baselines  same, for the quantum comparators
  └─> 13_results_aggregation        needs 00 at minimum; reads whatever else exists
                                    and must run LAST
```

Every notebook calls `preflight()` in its first cell, which fails immediately with a message
naming the notebook to run first — rather than crashing deep inside a long job.

| # | Notebook | Needs | Approx. cost |
|---|---|---|---|
| 00 | verify_archive | archive | seconds |
| 01 | data_preparation | archive (+ raw CSVs for splits) | minutes |
| 02 | qcnn_binary | archive | training, 5 seeds |
| 03 | qcnn_multiclass | archive | training, 5 seeds — long |
| 04 | baselines_multiseed | **00** | ~30 min for the classical rows; the parameter-matched controls train (4 variants × 5 seeds) |
| 05 | ablation | archive | training, 8 variants × 2 tasks × 5 seeds — longest |
| 06 | qubit_depth_sweep | archive | training, 7 configs × 3 seeds; 12-qubit cells dominate |
| 07 | robustness_adversarial | archive | ~20 min, inference only |
| 08 | shots_and_noise | archive | ~15 min, inference only |
| 09 | shap_explainability | archive + `shap`,`lime` | ~30 min |
| 10 | audit_log | nothing | ~2 min |
| 11 | grid5000_timing_energy | archive | ~10 min |
| 12 | cross_dataset_wustl | WUSTL CSV | optional |
| 14 | quantum_related_baselines | **00** | QSVM rows minutes; vqc/amp_qnn/QDL train, 3 seeds |
| 15 | multiclass_gap | archive | steps 1-3 minutes; hierarchical + feature-path train, 3 seeds |
| 16 | split_protocols | archive + **raw CSVs** | training, 3 protocols × 2 tasks × 3 seeds |
| 13 | results_aggregation | **00** | seconds — **run last** |

Training notebooks **skip any `(variant, task, split, seed)` cell whose manifest already
exists**, so re-submitting after a walltime kill re-runs only what died. Nothing is wasted.

---

## 5. Environment variables

| variable | effect |
|---|---|
| `QUICK_MODE=1` | 3 seeds, 2-epoch training, 10k pool — validates the DAG fast |
| `ARCHIVE_DIR` | where `salvaged_g5k_artifacts/` lives (auto-detected; set it if you move things) |
| `DATA_DIR` | where the raw CICIoMT2024 CSVs live |
| `OUT_DIR` | where results are written (default `./outputs`) |
| `KWOLLECT=1` | pull wattmeter energy for notebook 11 — requires an **exclusive** node |

`run_all.sh` sets `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to `nproc` automatically.

---

## 6. What comes back

```
outputs/<experiment>/
  runs/<variant>__<task>__<split>__seed<n>/
      metrics.json        table scalars
      predictions.npz     y_true, y_pred, probs, test_idx  (paired tests)
      history.{csv,npz}   every value behind every learning curve
      checkpoint.pt       weights (+ preprocessors/)
      manifest.json       config, seed, versions, timings, OAR job id
  curves/*.{csv,npz}      every value behind every figure
  sweep_*.{json,csv}      long-format sweep tables
  runs_index.csv          one-line DataFrame aggregation
outputs/paper/            LaTeX tables + coverage_map.json
figures/*.png            300 DPI
```

**Copy back `outputs/` and `figures/`.** With those, any new table, curve, confidence interval
or statistic can be produced later **without returning to the cluster** — which is the entire
point of the persistence layer.

Before releasing the allocation, run notebook 13: it prints a completeness audit and an
experiment coverage map. If a run cell is missing an artefact layer, regenerating it means
re-running the job — so fix it while the reservation is still live.

---

## 7. Quantum backend

`default.qubit` + `backprop`, pinned deliberately. Measured at n_q=8, L=4, batch 64:

| backend | gradient | ms/sample | extrapolated multiclass run |
|---|---|---|---|
| `default.qubit` | `backprop` | **3.34** | **3.9 h** |
| `lightning.qubit` | `adjoint` | 27.63 | 32.2 h |

At 8 qubits the state vector is only 256 amplitudes, so per-execution overhead dominates and
`default.qubit` — which keeps the whole batch in one broadcast torch tensor — wins by 8.3×.
This is also what the paper reports in section 4.1.

Re-run `python src/../bench_backend.py` on the node if you want to confirm on that hardware;
thread count and BLAS build shift the ratio.

---

## 8. If something fails

| symptom | cause | fix |
|---|---|---|
| `ARCHIVE NOT FOUND` | package moved | `export ARCHIVE_DIR=$PWD/salvaged_g5k_artifacts` |
| `MISSING PREREQUISITE: nb00_verify` | ran 04 or 13 first | run `00_verify_archive.ipynb` |
| `MISSING PACKAGES: ['shap']` | optional deps | `pip install shap lime`, or skip notebook 09 |
| `probabilities do not sum to 1` | float32 in shot sampling | already fixed — notebook 08 casts to float64 |
| model drift in `verify_deploy.py` | wrong package versions | reinstall from `requirements.txt`; **do not run the suite** |
| notebook 01 skips the splits | raw CSVs absent | download CICIoMT2024 WiFi/MQTT into `data/` |
| notebook 16 writes only `splits_status.json` | raw CSVs absent | same — the split protocols and the enlarged test set need the CSVs, the archive has no flow provenance |
| notebook 16 reports `device` infeasible | published CSVs carry no device/MAC/IP column | expected; report it as the result, do not substitute another grouping |
| `gate G5 FAILED` in notebook 16 | preprocessing fitted outside the fold | a real bug — the protocol result would be optimistic; fix before reporting |
