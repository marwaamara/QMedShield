#!/usr/bin/env bash
# run_all.sh -- execute the whole suite sequentially, in dependency order.
#
#   ./run_all.sh                 full run
#   QUICK_MODE=1 ./run_all.sh    fast pass (fewer seeds/epochs) to validate the DAG
#   ./run_all.sh 07 08 11        run only the listed notebook numbers
#
# Each notebook is executed in place with nbconvert, so its outputs are stored
# in the .ipynb as well as under outputs/.
set -u

cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc 2>/dev/null || echo 4)}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONIOENCODING=utf-8
export ARCHIVE_DIR=${ARCHIVE_DIR:-$PWD/salvaged_g5k_artifacts}

mkdir -p logs outputs figures

ALL=(00_verify_archive 01_data_preparation 02_qcnn_binary 03_qcnn_multiclass \
     04_baselines_multiseed 05_ablation 06_qubit_depth_sweep \
     07_robustness_adversarial 08_shots_and_noise 09_shap_explainability \
     10_audit_log 11_grid5000_timing_energy 12_cross_dataset_wustl \
     13_results_aggregation)

if [ $# -gt 0 ]; then
  SELECTED=()
  for want in "$@"; do
    for nb in "${ALL[@]}"; do
      case "$nb" in "$want"*) SELECTED+=("$nb");; esac
    done
  done
else
  SELECTED=("${ALL[@]}")
fi

echo "=== verifying the package before running anything ==="
python verify_deploy.py || { echo "verification FAILED -- aborting"; exit 1; }

FAILED=()
for nb in "${SELECTED[@]}"; do
  echo
  echo "==================================================================="
  echo "  $nb   ($(date -Is))"
  echo "==================================================================="
  jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=-1 \
      "notebooks/${nb}.ipynb" > "logs/${nb}.log" 2>&1
  if [ $? -eq 0 ]; then
    echo "  OK  -> logs/${nb}.log"
  else
    echo "  FAILED -> logs/${nb}.log (last 20 lines)"
    tail -20 "logs/${nb}.log"
    FAILED+=("$nb")
  fi
done

echo
echo "==================================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "  ALL ${#SELECTED[@]} NOTEBOOKS COMPLETED"
else
  echo "  ${#FAILED[@]} FAILED: ${FAILED[*]}"
fi
echo "  results: outputs/    figures: figures/    tables: outputs/paper/"
echo "==================================================================="
[ ${#FAILED[@]} -eq 0 ]
