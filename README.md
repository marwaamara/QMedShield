# QMedShield: hybrid quantum-classical intrusion detection for healthcare IoMT WiFi/MQTT networks

Code, notebooks and numerical results accompanying the manuscript

> **QMedShield: A Hybrid Quantum-Classical Intrusion Detection System for Healthcare IoMT WiFi/MQTT Networks**
> Marwen Amara, Marwa Amara, Sami Mnasri, Thierry Val
> *Journal of Network and Computer Applications* (Elsevier), manuscript JNCA-D-26-01837.

**Status: under peer review.** The repository reflects the current version of the manuscript and may change before publication.
A versioned, citable archive with a DOI will be deposited on **Zenodo after acceptance**; this README will then be updated with the DOI and the final reference.

---

## The idea in brief

Intrusion detection for hospital IoMT networks has to be accurate on WiFi and MQTT traffic, small enough to replicate across edge gateways,
robust to the perturbations real deployments produce, explainable to the analyst who acts on an alert, and auditable afterwards.
QMedShield targets these requirements in one end-to-end differentiable pipeline:

1. a classical encoder maps 16 PCA components of 25 network-flow features to 8 angles;
2. an 8-qubit parameterized quantum circuit (4 layers, data re-uploading at every layer, three-scale entanglement) is simulated on a state-vector backend;
3. a classical decoder produces the decision (binary, or eight operational attack classes);
4. every decision is explained with SHAP and LIME, mapped back through the PCA loading matrix to the original features, and committed to a SHA-256 hash chain.

The repository is built to answer a more specific question than most quantum intrusion-detection studies ask:
**which properties of such a detector are attributable to the quantum component, and which survive deployment-style stress?**
It therefore contains parameter-matched classical bottlenecks, a classical twin of the whole pipeline (same encoder, decoder, optimiser, explanation and audit layers, with the circuit replaced by a dense map),
multi-seed statistics, a chronological evaluation, robustness in the original feature space (including a constrained adversarial attack), finite-shot and noisy simulation,
explanation-fidelity measurements, and a tamper-detection harness with an explicit threat model.

### What the experiments show (as reported in the manuscript)

* The integrated framework reaches 99.63 % binary and 79.19 % eight-class accuracy in the reference run (78.39 ± 0.69 % over five seeds) with 1,178 / 6,768 parameters (4.6 / 26.4 KB).
* **No quantum-specific advantage is demonstrated.** Parameter-matched classical bottlenecks and a classical twin show no statistically reliable accuracy difference from the quantum circuit; in simulation the twin is far cheaper to train, run and explain. All quantum results are simulated; nothing is claimed about quantum hardware.
* 97 % of the multiclass error is a DoS/DDoS confusion that the 25 selected per-flow features do not resolve; causal windowed aggregation over neighbouring flows recovers much of it (Random Forest test only).
* Multiclass accuracy drops by six to nine points under a chronological protocol for every model tested; a constrained adversarial attack with a 2 % budget removes about 20 points.
* Inverse-mapped (PCA) explanations agree only weakly with direct attributions.
* The hash chain detects modification, deletion, reordering and truncation, but full tamper detection depends on an independently protected head digest and record count.

---

## Where to start

| You want to | Go to |
|---|---|
| **Reproduce the analysis, statistics, cost and robustness comparisons on a laptop** | [`local_analysis/notebooks/`](local_analysis/notebooks/) (start with `00_start_here.ipynb`) |
| **See or rerun the main experiments** (headline models, baselines, ablation, qubit/depth sweep, shots and noise, explanations, audit log, cost, quantum baselines, split protocols) | [`grid5000/`](grid5000/) |

### Main experiments: executed on Grid'5000

The training-heavy experiments were run on the **Grid'5000 testbed** (CPU-only nodes; about 267 s per training epoch at 8 qubits, 8.7 h per multiclass run under state-vector simulation).
`grid5000/notebooks/` holds the 17 notebooks **with their recorded outputs** (including the cluster environment and timings), `grid5000/oar/` the OAR job scripts, `grid5000/src/` the Python modules,
and `grid5000/README_GRID5000.md` the deployment instructions. Each notebook is standalone: copy it to a node and run it.

| Notebook | Content |
|---|---|
| 00 verify archive | integrity check of the archived weights before any experiment |
| 01 data preparation | data characterisation, enlarged test set, split protocols |
| 02, 03 hybrid models | headline binary and multiclass models |
| 04 baselines | tuned and lightweight classical baselines, parameter-matched controls, paired tests |
| 05 ablation | eight circuit variants over five seeds |
| 06 qubit/depth sweep | circuit width and depth |
| 07 robustness | raw-space perturbations and adversarial attacks |
| 08 shots and noise | finite-shot and device-noise simulation |
| 09 explainability | SHAP, LIME, fidelity and stability, inverse-map validation |
| 10 audit log | hash chain, tamper harness, overhead |
| 11 timing and energy | training cost, latency against batch size, energy estimate |
| 12 cross-dataset | WUSTL-EHMS evaluation |
| 13 aggregation | collects the results into the paper's tables |
| 14 quantum baselines | quantum-kernel SVMs, variational classifiers, re-implementation of a related quantum IoMT detector |
| 15 multiclass gap | decomposition of the DoS/DDoS error, specialist and hierarchical detectors |
| 16 split protocols | random vs chronological protocols with preprocessing re-fitted inside each training partition |

### Laptop analyses: `local_analysis/`

Run from stored artefacts, no cluster needed. Notebooks `00`-`09` (executed, with outputs) are listed in `local_analysis/notebooks/00_start_here.ipynb`; the underlying scripts sit next to them
(`ci_effect_sizes.py`, `twin_pipeline.py`, `robustness_twin_allseeds.py`, `audit_threat_model.py`, `dos_ddos_window_test.py`, `make_figures.py`, ...).

```bash
git clone <this repository>
cd QMedShield
pip install -r requirements.txt
cd local_analysis/notebooks
jupyter lab            # open 00_start_here.ipynb and run the notebooks in order
```

## Data

The experiments use the WiFi/MQTT partition of **CICIoMT2024** (Dadkhah et al., *Internet of Things* 28, 101351, 2024; <https://www.unb.ca/cic/datasets/>).
The raw CSV files are **not** redistributed here. Derived artefacts needed by the laptop notebooks (PCA features, labels, fitted scaler and PCA, stored predictions) are in `local_analysis/archive/`.

Preprocessing note: in the main experiments the constant and correlation filters, the ANOVA selector (25 of 35 features), the scaler and the PCA (16 components) were fitted once on the 50,000-flow sample before splitting.
Cluster notebook 16 re-fits every step inside each training partition. The manuscript discusses the consequence.

## Environment

Laptop notebooks: Python 3.13, PyTorch 2.12, PennyLane 0.44, scikit-learn 1.8, SciPy, NumPy, pandas, matplotlib (`requirements.txt`).
Cluster runs: the pins in `grid5000/requirements.txt`. Seeds used throughout: 42, 1, 7, 13, 2024.

## Authors

Marwen Amara, Marwa Amara (corresponding author), Sami Mnasri, Thierry Val. See `CITATION.cff`.

## Citation

Please cite the manuscript once published; until then see `CITATION.cff`. A Zenodo DOI will be added after acceptance.

## License

Code is released under the MIT License (see `LICENSE`). The CICIoMT2024 dataset has its own terms of use.
