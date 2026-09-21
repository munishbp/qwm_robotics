# Experiment methodology

This document is the replication recipe. It states what runs, in what order, with what settings,
on what hardware, and how each hypothesis is decided from the outputs. The reasons behind each
design choice are in `design.md`. The equations are in `math.md`. The numbers are in `results.md`.

## 1. Scope

The proposal targets mjlab (MuJoCo Warp). The main study runs the method on a batched 2D rigid
body transport simulator written in PyTorch (`swarm/env.py`), which the proposal's risk section
asks for as the first version. Every hypothesis concerns the search mechanism under
decentralization and staleness, not contact physics, so the 2D task keeps the question intact.

The same task also runs on mjlab (`swarm/env_mjlab.py`) with the identical interface, observation
layout, message model, and success test, and real contact physics: an 8 kg box with Coulomb
friction 0.5, pushers that push through contact under a 10 N force limit, grippers that latch
kinematically and pull with 3 N while unloading the payload by 25 N each, and static walls. The
friction facts of `design.md` section 3.3 hold in that physics (tests in
`tests/test_env_mjlab.py`). `SWARM_SIM=mjlab` selects it in every script. The mjlab robots run at
2.5 m/s instead of 1.5 because a velocity servo accelerates and the 2D robots move instantly;
every other constant is shared. See `mjlab_port.md` for the mapping and its measurements.

## 2. Hardware and software

| Item | Value |
|---|---|
| GPU | one NVIDIA GeForce RTX 5090, 32 GB, driver 610.43.02 |
| CPU and RAM | shared workstation, 46 GB RAM, other jobs running at the same time |
| Python | 3.13 |
| PyTorch | 2.9.0+cu128 |
| Lock files | `uv.lock`, `pylock.toml` |
| Memory caps | 25 percent of the GPU per process, 3 GB host RSS watchdog (`swarm/compute.py`) |

Every script is deterministic given its seed except for CUDA kernel nondeterminism in scatter
and attention kernels. Rerunning with the same seed reproduces the numbers to within evaluation
noise, not bit for bit. The reported sweeps were run before the evaluation seeded the torch RNG
per batch, so a cell that appears in several sweep groups differs between them by up to four
points (results.md section 13.9 lists the repeats); the code now seeds per batch.

## 3. The task

See `design.md` section 3 for the constants. In short: a team of 3 pushers, 2 grippers, and 1
scout moves a rectangular payload to a goal pose inside a 10 by 10 m arena. The payload moves
only when the net force exceeds a friction threshold that latched grippers lower. Pushers push,
grippers latch and pull weakly, the scout only senses. Reward is 1 on success and 0 otherwise.
The episode ends on success or after 150 steps of 0.1 s.

Each robot observes its own position, its type, the goal, and the payload pose if the payload
center is within its sensing range (3 m for pushers and grippers, 15 m for the scout). Robots
exchange messages within 6 m unless the payload blocks the line between them. Goals lie 1.5 to
3 m from the start pose with an orientation change of up to 45 degrees.

## 4. Pipeline

`scripts/run_all.sh` runs every step below in order. Each step is one process. The full run
takes about 2.5 hours on the 2D simulator and about 3.5 hours on mjlab, on the hardware above.
`SWARM_SIM=mjlab RUN_DIR=$PWD/runs/mjlab bash scripts/run_all.sh 24000` is the mjlab run.

| Step | Command | Output | Purpose |
|---|---|---|---|
| 1 | `scripts/run_controls.py` | `results/controls.json` | Task solvable by the scripted team, unsolvable by one robot, throughput |
| 2 | `scripts/collect_offline.py --envs 256 --steps 800` | `data/offline.pt` | 204,800 scripted transitions with action noise 0.2 |
| 3 | `scripts/train.py --obs belief --steps 12000` (24000 on mjlab) | `checkpoints/belief_best.pt`, `results/train_belief.jsonl` | The decentralized belief agent. The best checkpoint by evaluation success is the snapshot every sweep uses |
| 4 | `scripts/train.py --obs full --steps 12000` (24000 on mjlab) | `checkpoints/full.pt`, `results/train_full.jsonl` | Centralized full state baseline |
| 5 | `scripts/world_model_error.py` | `results/wm_error.json` | Open loop latent and decoded error against horizon |
| 6 | `scripts/evaluate.py --depth -1, 0, 2 --lag 1` | `results/h1_depth*.json` | H1 |
| 7 | `scripts/sweep.py --which h2 h3 h4 robust transfer` | `results/h2.json` and so on | H2, H3, H4, robustness, transfer |
| 8 | `scripts/record_episodes.py` | `results/episodes.json` | Episode replays for the viewer |
| 9 | `scripts/plot.py`, `viewer/build.py` | `figures/` | Figures and the interactive viewer |

## 5. Training settings

| Setting | Value |
|---|---|
| Parallel envs | 256 |
| Batched env steps | 12,000 on 2D (3.07 million transitions), 24,000 on mjlab (6.14 million) |
| Updates per batched env step | 4 |
| Batch | 256 rows, half offline and half online, times 6 robots |
| Critic ensemble | 10 heads, target is the mean of 2 random heads clamped to [0, 1] (RLPD uses the minimum; see design.md 6.4) |
| Discount, Polyak rate, learning rate | 0.99, 0.005, 3e-4 |
| Target entropy | -3 |
| Training staleness | lag 1 |
| Latent size, hidden size | 64, 256 |
| Frame stack | 3 observations plus the last action |
| Seed | 0 |

The world model and the decoder train from the first update on the same batches, with a stop
gradient on the latents. There is no separate pretraining phase because the encoders are random
at the start, and training on the offline half of every batch is the same data the proposal
pretrains on.

Evaluation during training runs 2 batches of 128 envs with the mean action every 500 steps and
records the first episode of every env. The checkpoint with the best evaluation success is the
snapshot for every test time experiment (`checkpoints/belief_best.pt`). The critic showed a late
decline in every training run, so the final checkpoint is not used.

## 5b. The programs after the first pipeline

The first pipeline (section 4) produced the seed 0 snapshots and sweeps of `results.md` sections
3 to 13. Four more driver scripts ran afterwards, each resumable by output file and each idling
the GPU for 15 minutes after a training run:

| Script | What it runs | Results sections |
|---|---|---|
| `scripts/day1.sh` | The first day controls on both snapshots (`scripts/day1_controls.py`): forward correction off, random world model, decoded and random scorers, leader with one fresh election, and on mjlab the fair H1 arms; then the demonstration quality control (a 2D buffer at 61 percent success, a retrain, `scripts/critic_span.py`) | 14 |
| `scripts/day2.sh` | Two more mjlab training seeds with the fair H1 arms on each, then the mjlab sweep rerun with per batch seeding | 13.8, 16 |
| `scripts/day3.sh` | Mechanism controls on the extra seeds, the staleness curve at lags 2 and 8 (`scripts/lag_curve.py`), search during collection, the momentum 2D variant, two more 2D seeds with H2, H2 on the extra mjlab seeds | 16 to 19 |
| `scripts/day4.sh` | The fair H1 arms rerun on the three mjlab seeds with per episode outcomes for the paired bootstrap | 16 |

**The programs of sections 21 to 23.** No driver script. Each results section states its command.
The gate and the pessimistic score are test time settings on existing snapshots:
`scripts/sweep.py --which gate` and `--which lcb`, at lag 1, 256 envs, and 6 batches, which is the
margin protocol of `next_steps.md`. Each grid has the two no search baselines (mean action and
sampled policy), the ungated search, and the shuffled gate as the control. `--lcb` sets the
pessimistic score for every other sweep group, and `--tag` names a second output file. The H2
sweeps of section 23 use 256 envs and 3 batches. The n step runs are `scripts/train.py --n-step 5`
in a new run directory with the offline data of the matching one step seed, so the target is the
only change. The threshold `gate = 1.0` and the weight `lcb = 2.0` were chosen on seed 0 and then
applied to seeds 1 and 2 without a new search.

**The fair H1 arms.** Four arms on the same env seeds: the sampled policy (the policy with its
exploration noise), a random root candidate, depth 0 (critic argmax over nine candidates), and
depth 6 search with beta 0.9. 256 envs and 5 batches per cell, env seeds 1000 to 1004, so every
arm scores the same 1,280 first episodes. The decision rule for H1 against the fair baseline:
depth 6 minus the sampled policy, paired over episodes, with a bootstrap 95 percent interval
that excludes zero; across seeds the sign must agree in all three. The reward hacking audit is
`scripts/audit.py` (section 15 of the results).

## 6. Evaluation protocol

Every reported number is the mean over 3 batches, each batch from a different env seed (1000,
1001, 1002), with the standard error over batches. H1 uses 256 envs per batch. The sweeps use
128 envs per batch, which keeps their 76 cells under two hours. A batch runs 150 steps from reset
and scores the first episode of every env, so every env contributes exactly one episode. The
mean action is used unless search is on.

Search settings unless a sweep varies them: 8 sampled candidates plus the mean action, beam 4,
tree search discount 0.5, independent mode. The no search baseline is the mean action, reported
as depth -1. Depth 0 is the critic argmax over the root candidates with no world model call.

## 7. Decision rules

| Hypothesis | Measure | Confirmed when | Refuted when |
|---|---|---|---|
| H1 | success at lag 1: depth -1 against depth 2 | depth 2 exceeds depth -1 by more than 2 standard errors | otherwise |
| H2 | best depth per lag over depths 0 to 6, at lags 0, 1, 2, 4 | the best depth is non increasing in lag and drops by at least one level from lag 0 to lag 4 | the best depth stays flat or rises |
| H3 | best beta per lag at depth 2 | the best beta is non increasing in lag | otherwise |
| H4 | success at matched compute: independent, leader, round robin | leader exceeds independent by more than 2 standard errors at some lag | otherwise |

A hypothesis whose measure is within noise at every cell is reported as not resolved, with the
noise level stated. That is a result about the task, not a failure of the protocol.

## 8. What is not controlled

- One training seed. The sweeps hold the snapshot fixed, so cell to cell comparisons are paired
  and share the same training noise. Claims across training runs (belief against full) rest on one
  seed each and are reported as such.
- The scripted controller is not optimal, so the offline data is a floor, not a ceiling.
- The 2D simulator has no robot to robot collisions and quasi static payload motion.

## 9. Files that a reader needs

- `swarm/env.py` for the physics and observation, `swarm/scripted.py` for the demo source.
- `swarm/belief.py` for the message table, forward correction, and fusion.
- `swarm/search.py` for the tree search.
- `scripts/sweep.py` for the exact grids.
- `tests/` for the checks that ran before training.
