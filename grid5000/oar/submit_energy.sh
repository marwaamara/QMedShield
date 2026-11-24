#!/usr/bin/env bash
# Notebook 11 with wattmeter energy measurement.
#
# MUST be an EXCLUSIVE node on a site with Kwollect wattmeters: with jobs
# co-located on the same machine the power reading describes the machine, not
# the experiment. Check site support first:
#   curl -s "https://api.grid5000.fr/stable/sites/lyon/metrics" | head
cd "$(dirname "$0")/.."
KWOLLECT=1 oarsub -t exotic -l host=1,walltime=4:00:00 \
       -O logs/energy.%jobid%.out -E logs/energy.%jobid%.err \
       "KWOLLECT=1 ./run_all.sh 11"
