#!/usr/bin/env bash
# Third program: mechanism controls on the extra seeds, the staleness curve, search during
# collection, a momentum 2D variant, two more 2D seeds with H2, H2 on the extra mjlab seeds.
# The GPU idles 15 minutes after every training run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; P="$ROOT/.venv/bin/python"; S="$ROOT/scripts"
cool() { echo "== $(date +%H:%M:%S) cooling 15 minutes"; sleep 900; }
stage() { echo "== $(date +%H:%M:%S) $*"; }

stage "A mechanism controls on mjlab seeds 1 and 2"
for seed in 1 2; do D="runs/mjlab_seed$seed"
  [ -f "$D/results/critic_action_span.json" ] || (cd "$D" && SWARM_SIM=mjlab $P $S/critic_span.py --ckpt checkpoints/belief_best.pt --out results/critic_action_span.json > results/logs/span.log 2>&1)
  [ -f "$D/results/wm_control.json" ] || (cd "$D" && SWARM_SIM=mjlab $P $S/day1_controls.py --ckpt checkpoints/belief_best.pt --only F3 --out results/wm_control.json > results/logs/wm_control.log 2>&1)
done
stage "B staleness curve, lags 2 and 8, three mjlab seeds"
for D in runs/mjlab runs/mjlab_seed1 runs/mjlab_seed2; do
  [ -f "$D/results/lag_curve.json" ] || (cd "$D" && SWARM_SIM=mjlab $P $S/lag_curve.py --ckpt checkpoints/belief_best.pt --out results/lag_curve.json > results/logs/lag_curve.log 2>&1)
done
stage "C search during collection on mjlab"
D="runs/mjlab_collect"; mkdir -p "$D/results/logs" "$D/data" "$D/checkpoints"
[ -e "$D/data/offline.pt" ] || ln -s "$ROOT/runs/mjlab/data/offline.pt" "$D/data/offline.pt"
if [ ! -f "$D/checkpoints/belief_best.pt" ]; then
  (cd "$D" && SWARM_SIM=mjlab $P $S/train.py --obs belief --steps 24000 --seed 0 --collect-depth 2 --out belief > results/logs/train_belief.log 2>&1); cool; fi
[ -f "$D/results/day1_controls.json" ] || (cd "$D" && SWARM_SIM=mjlab $P $S/day1_controls.py --ckpt checkpoints/belief_best.pt --only F1 --out results/day1_controls.json > results/logs/day1_f1.log 2>&1)
stage "D momentum 2D variant"
D="runs/2d_momentum"; mkdir -p "$D/results/logs" "$D/data" "$D/checkpoints"
[ -f "$D/results/controls.json" ] || (cd "$D" && SWARM_SIM=2d_momentum $P $S/run_controls.py > results/logs/controls.log 2>&1)
[ -f "$D/data/offline.pt" ] || (cd "$D" && SWARM_SIM=2d_momentum $P $S/collect_offline.py --envs 256 --steps 800 > results/logs/collect.log 2>&1)
if [ ! -f "$D/checkpoints/belief_best.pt" ]; then
  (cd "$D" && SWARM_SIM=2d_momentum $P $S/train.py --obs belief --steps 12000 --out belief > results/logs/train_belief.log 2>&1); cool; fi
[ -f "$D/results/day1_controls.json" ] || (cd "$D" && SWARM_SIM=2d_momentum $P $S/day1_controls.py --ckpt checkpoints/belief_best.pt --out results/day1_controls.json > results/logs/day1.log 2>&1)
[ -f "$D/results/critic_action_span.json" ] || (cd "$D" && SWARM_SIM=2d_momentum $P $S/critic_span.py --ckpt checkpoints/belief_best.pt --out results/critic_action_span.json > results/logs/span.log 2>&1)
stage "E two more 2D seeds with the fair arms and H2"
for seed in 1 2; do D="runs/2d_seed$seed"; mkdir -p "$D/results/logs" "$D/data" "$D/checkpoints"
  [ -e "$D/data/offline.pt" ] || ln -s "$ROOT/data/offline.pt" "$D/data/offline.pt"
  if [ ! -f "$D/checkpoints/belief_best.pt" ]; then
    (cd "$D" && SWARM_SIM=2d $P $S/train.py --obs belief --steps 12000 --seed $seed --out belief > results/logs/train_belief.log 2>&1); cool; fi
  [ -f "$D/results/day1_controls.json" ] || (cd "$D" && SWARM_SIM=2d $P $S/day1_controls.py --ckpt checkpoints/belief_best.pt --only F2 --out results/day1_controls.json > results/logs/day1.log 2>&1)
  [ -f "$D/results/h2.json" ] || (cd "$D" && SWARM_SIM=2d $P $S/sweep.py --ckpt checkpoints/belief_best.pt --which h2 > results/logs/sweep_h2.log 2>&1)
done
stage "F H2 on the extra mjlab seeds"
for D in runs/mjlab_seed1 runs/mjlab_seed2; do
  [ -f "$D/results/h2.json" ] || (cd "$D" && SWARM_SIM=mjlab $P $S/sweep.py --ckpt checkpoints/belief_best.pt --which h2 > results/logs/sweep_h2.log 2>&1)
done
cool
stage "day3 done"
