#!/usr/bin/env bash
# Run one notebook as an OAR ARRAY of N independent shards.
#
#   ./oar/submit_shards.sh 05 16          notebook 05 over 16 shards
#   ./oar/submit_shards.sh 16 12          notebook 16 over 12 shards
#   WALLTIME=6:00:00 ./oar/submit_shards.sh 14 6
#   SEEDS=42,1 ./oar/submit_shards.sh 15 2
#
# Shardable notebooks: 05 (ablation), 06 (qubit/depth sweep), 14 (quantum
# comparators), 15 (multiclass gap), 16 (split protocols). Others ignore SHARD.
#
# MEASURED cost basis (Grid'5000, 8 qubits, 35,000 rows, batch 32):
#   267 s/epoch  ->  binary cell ~3.6 h,  multiclass cell ~8.7 h
#
#   notebook  cells  node-hours   suggested shards
#   05         48*      ~290      16
#   06         21        ~58       4
#   14         22        ~82       6
#   15          6        ~61       4
#   16         24       ~146      12
#   * at SEEDS_ABLATION = 3 seeds
#
# Assignment is round-robin over a deterministically sorted cell list, so shards
# never collide and together cover the grid exactly. Every cell writes its own
# manifest and is skipped on re-entry: a walltime kill costs one cell, and
# re-submitting the same array resumes precisely where it stopped.
set -u
cd "$(dirname "$0")/.."

NB=${1:?usage: submit_shards.sh <notebook-number> [n_shards]}
N=${2:-8}
WALL=${WALLTIME:-12:00:00}

echo "notebook $NB over $N shards, walltime $WALL each"
[ -n "${SEEDS:-}" ]            && echo "  SEEDS=$SEEDS"
[ -n "${ABLATION_SAMPLES:-}" ] && echo "  ABLATION_SAMPLES=$ABLATION_SAMPLES"

oarsub -l host=1/core=32,walltime=$WALL \
       --array $N \
       -O logs/nb${NB}.%jobid%.out -E logs/nb${NB}.%jobid%.err \
       "SHARD=\$((OAR_ARRAY_INDEX))/$N SEEDS=${SEEDS:-} ABLATION_SAMPLES=${ABLATION_SAMPLES:-} ./run_all.sh $NB"

echo
echo "when the array finishes, run the notebook once more to aggregate:"
echo "  ./run_all.sh $NB      # trains nothing new; prints the grid status"
