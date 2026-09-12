#!/usr/bin/env bash
# Full replication pipeline. One GPU process at a time. Logs go to results/logs.
# Usage: bash scripts/run_all.sh [STEPS]   (default 12000 batched env steps of 256 envs)
# Every step is skipped when its output exists, so a rerun resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS="${1:-12000}"
P=.venv/bin/python
CK=checkpoints/belief_best.pt
mkdir -p results/logs
run() { local out="$1"; shift; if [ -e "$out" ]; then echo "== skip $out"; return; fi
        echo "== $(date +%H:%M:%S) $*"; "$@"; }
run results/controls.json      $P scripts/run_controls.py                            > results/logs/controls.log 2>&1
run data/offline.pt            $P scripts/collect_offline.py --envs 256 --steps 800  > results/logs/collect.log 2>&1
run checkpoints/belief.pt      $P scripts/train.py --obs belief --steps "$STEPS" --out belief > results/logs/train_belief.log 2>&1
run checkpoints/full.pt        $P scripts/train.py --obs full   --steps "$STEPS" --out full   > results/logs/train_full.log 2>&1
run results/wm_error.json      $P scripts/world_model_error.py --ckpt $CK             > results/logs/wm_error.log 2>&1
for d in -1 0 2; do
  run results/h1_depth$d.json  $P scripts/evaluate.py --ckpt $CK --depth $d --lag 1 --out results/h1_depth$d.json > results/logs/h1_depth$d.log 2>&1
done
for w in h2 h3 h4 robust transfer; do
  run results/$w.json          $P scripts/sweep.py --ckpt $CK --which $w             > results/logs/sweep_$w.log 2>&1
done
run results/ablations.json     $P scripts/ablations.py --ckpt $CK                    > results/logs/ablations.log 2>&1
run results/episodes.json      $P scripts/record_episodes.py --ckpt $CK              > results/logs/record.log 2>&1
run figures/learning_curves.png $P scripts/plot.py                                   > results/logs/plot.log 2>&1
run figures/viewer.html        $P viewer/build.py                                    > results/logs/viewer.log 2>&1
echo "== $(date +%H:%M:%S) done"
