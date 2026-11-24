#!/usr/bin/env bash
# Notebooks that need NO training: 00, 01, 04, 07, 08, 09, 10, 11, 13.
# One node, a few hours. Run this stage first -- it produces evidence for
# eight experiments without touching the training budget. Notebooks 14
# and 15 are omitted here: their QSVM rows and their error decomposition need no
# training, but their remaining cells do -- run them with submit_training.sh.
cd "$(dirname "$0")/.."
oarsub -l host=1/core=32,walltime=6:00:00 \
       -O logs/archive_only.%jobid%.out -E logs/archive_only.%jobid%.err \
       "./run_all.sh 00 01 04 07 08 09 10 11 13"
