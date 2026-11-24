#!/usr/bin/env bash
# Notebooks that train: 02, 03, 04(matched controls), 05, 06, 14, 15, 16.
# 04 also appears in submit_archive_only.sh: its paper-baseline and
# lightweight rows need no training, its parameter-matched controls do. Every
# cell is skipped once its manifest exists, so running it twice is free.
# 05 (ablation) is the long pole: 8 variants x 2 tasks x 5 seeds.
# Every run cell is skipped if its manifest already exists, so re-submitting
# after a walltime kill only re-runs what actually died.
cd "$(dirname "$0")/.."
oarsub -l host=1/core=32,walltime=24:00:00 \
       -O logs/training.%jobid%.out -E logs/training.%jobid%.err \
       "./run_all.sh 02 03 04 05 06 14 15 16"
