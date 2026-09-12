#!/usr/bin/env bash
# Full replication pipeline. One GPU process at a time. Logs go to results/logs.
# Usage: bash scripts/run_all.sh [STEPS]   (default 12000 batched env steps of 256 envs)
# Every step is skipped when its output exists, so a rerun resumes where it stopped.
# SWARM_SIM=mjlab selects the MuJoCo Warp task. RUN_DIR selects where results, data,
# checkpoints, and figures go (default: the repository root).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${RUN_DIR:-$ROOT}"
cd "${RUN_DIR:-$ROOT}"
STEPS="${1:-12000}"
P="$ROOT/.venv/bin/python"
S="$ROOT/scripts"
V="$ROOT/viewer"
CK=checkpoints/belief_best.pt
mkdir -p results/logs
run() { local out="$1"; shift; if [ -e "$out" ]; then echo "== skip $out"; return; fi
        echo "== $(date +%H:%M:%S) $*"; "$@"; }
run results/controls.json      $P $S/run_controls.py                            > results/logs/controls.log 2>&1
run data/offline.pt            $P $S/collect_offline.py --envs 256 --steps 800  > results/logs/collect.log 2>&1
run checkpoints/belief.pt      $P $S/train.py --obs belief --steps "$STEPS" --out belief > results/logs/train_belief.log 2>&1
run checkpoints/full.pt        $P $S/train.py --obs full   --steps "$STEPS" --out full   > results/logs/train_full.log 2>&1
run results/wm_error.json      $P $S/world_model_error.py --ckpt $CK             > results/logs/wm_error.log 2>&1
for d in -1 0 2; do
  run results/h1_depth$d.json  $P $S/evaluate.py --ckpt $CK --depth $d --lag 1 --out results/h1_depth$d.json > results/logs/h1_depth$d.log 2>&1
done
for w in h2 h3 h4 robust transfer; do
  run results/$w.json          $P $S/sweep.py --ckpt $CK --which $w             > results/logs/sweep_$w.log 2>&1
done
run results/ablations.json     $P $S/ablations.py --ckpt $CK                    > results/logs/ablations.log 2>&1
run results/episodes.json      $P $S/record_episodes.py --ckpt $CK              > results/logs/record.log 2>&1
run figures/learning_curves.png $P $S/plot.py                                   > results/logs/plot.log 2>&1
run figures/viewer.html        $P $V/build.py                                    > results/logs/viewer.log 2>&1
echo "== $(date +%H:%M:%S) done"
