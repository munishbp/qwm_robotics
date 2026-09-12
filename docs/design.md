# System design

This document fixes the interfaces before any code exists. Every module and every agent follows it.
Where it differs from the proposal, section 8 states the difference and the reason.

## 1. Problem

Test whether test time world model search still helps a decentralized heterogeneous team when each
robot must imagine its teammates from stale information. The four hypotheses H1 to H4 are in the
proposal. The deliverable is one trained snapshot and a set of test time sweeps on it.

## 2. Compute rules

These rules protect a nine day job that runs on the same machine. They are not optional.

- One training or evaluation process at a time on the GPU.
- Every script calls `swarm.compute.limit_memory()` first. It sets the per process GPU memory
  fraction to 0.25 (about 8 GB of 32 GB) and sets `expandable_segments`.
- Parallel environments: at most 512 for training, at most 256 for evaluation.
- Replay buffers live on the GPU and are sized in the config. The default online buffer holds
  1,000,000 transitions. The offline buffer holds 200,000.
- Host RAM for one process stays under 3 GB. No process loads a dataset into host memory twice.
- Smoke test every script at `--smoke` scale (16 envs, 200 steps) before the full run.
- Never kill or renice a process you did not start.

## 3. Environment: `swarm/env.py`

A batched 2D transport task in PyTorch. All state tensors have a leading env dimension `E`.
The team has `K` robots. Types: `0 pusher`, `1 gripper`, `2 scout`. The default team is
`(pusher, pusher, pusher, gripper, gripper, scout)`, so `K = 6`.

### 3.1 World

- Arena: square, half size `A = 5.0` m. Robots and the payload stay inside it.
- Payload: rectangle with half extents `(hx, hy) = (0.8, 0.4)` m. Pose `(x, y, theta)`.
- Robots: discs of radius `0.2` m. State is position `(x, y)`. No heading. A robot never enters the
  payload rectangle. When the payload stands against the wall, a robot squeezed between them may
  leave the arena by up to one diameter. Non penetration wins because contact, force, and occlusion
  depend on it.
- Time step `dt = 0.1` s. Episode length `T = 150` steps.
- Goal: a pose `(gx, gy, gtheta)`. Reset samples the payload pose and a goal at distance 1.5 to
  3.0 m. The goal orientation differs from the start orientation by a random angle in
  `[-pi/4, pi/4]`. (The first draft used 2.5 to 4.0 m and `[-pi/2, pi/2]`. The cloned policy
  plateaued at 6 percent success there, which leaves no room to measure a search effect.) Robots start at random positions at least
  1.0 m from the payload.

### 3.2 Actions

`a[e, k] = (vx, vy, u)` in `[-1, 1]^3`.

- Every robot moves with velocity `clip(vx, vy) * v_max`, `v_max = 1.5` m/s. A robot cannot enter
  the payload rectangle. The env projects it out along the closest boundary normal.
- Pusher: if the robot disc edge is within `d_c = 0.25` m of the payload boundary (robot center within
  `0.45` m) and `u > 0`, it applies a
  force `u * F_p` at the closest boundary point along the inward normal. `F_p = 1.0` N.
- Gripper: if `u > 0` and within `d_c`, it latches at the closest boundary point. The latch point is
  fixed in the payload frame. While latched, the gripper sits at the latch point and applies force
  `F_g * clip(vx, vy)` at that point, `F_g = 0.3` N, in any direction. `u <= 0` releases.
- Scout: `u` has no effect. The scout applies no force.

### 3.3 Payload dynamics

Quasi static with a Coulomb style threshold. `n_latched` is the number of latched grippers.

```
F_s   = max(F_0 - dF * n_latched, F_min)      F_0 = 3.5, dF = 1.0, F_min = 0.5
tau_s = 0.5 * F_s * hx
F_net = sum of forces, tau_net = sum of (r x F) with r from the payload center
v     = max(|F_net| - F_s, 0) / c_t * F_net / |F_net|     c_t = 2.0
omega = max(|tau_net| - tau_s, 0) / c_r * sign(tau_net)  c_r = 2.0
pose += (v, omega) * dt
```

Consequences that the controls in section 7 verify:

- Three pushers with no latch: `3.0 < 3.5`. Pushers alone cannot move the payload.
- Two grippers latched, no pushers: `0.6 < 1.5`. Grippers alone cannot move the payload.
- Two grippers latched plus two pushers: `2.0 + 0.6 > 1.5`. The team moves it.
- One robot of any type cannot move it.

### 3.4 Success and reward

Success when `|(x, y) - (gx, gy)| < 0.3` m and `|wrap(theta - gtheta)| < 0.2` rad. Reward is
`1.0` on the success step and `0.0` otherwise. Success terminates the episode. Reaching `T` truncates
it. The env exposes `terminated` and `truncated` separately.

### 3.5 Observation

`local[e, k]` is a 16 float vector, in this order:

| Index | Content |
|---|---|
| 0:2 | own position / A |
| 2:5 | type one hot |
| 5:9 | goal relative to own position: `(gx - x, gy - y) / A, cos(gtheta), sin(gtheta)` |
| 9 | payload visible flag |
| 10:14 | payload relative to own position, times the visible flag: `(px - x, py - y) / A, cos, sin` |
| 14 | latched flag |
| 15 | time fraction `t / T` |

The payload is visible if its center is within the sensing range of the type. Ranges: pusher `3.0`,
gripper `3.0`, scout `15.0` m, which exceeds the arena diagonal, so the scout always sees it.

`full[e, k]` is the centralized observation for the oracle baseline: `local[e, k]` followed by the
absolute payload pose `(px / A, py / A, cos, sin)` and every robot's `(x / A, y / A, latched)` in index
order. Length `16 + 4 + 3K`.

`comm[e, i, j]` is true when robot `i` can receive a message from robot `j`. It requires `i != j`,
distance under `R_c = 6.0` m, and no occlusion. (The first draft used 2.0 m sensing and 4.0 m
communication. Robots that start outside both ranges have no way to find the payload, and the
task was too hard for the budget.) The payload occludes when the segment from `i` to `j`
intersects the payload rectangle.

### 3.6 API

```python
env = TransportEnv(num_envs, team=DEFAULT_TEAM, device="cuda", cfg=EnvConfig())
obs = env.reset()                      # dict: local [E,K,16], full [E,K,16+4+3K], comm [E,K,K]
obs, reward, terminated, truncated, info = env.step(actions)   # actions [E,K,3]
state = env.state()                    # dict: payload [E,3], goal [E,3], robot_pos [E,K,2], latched [E,K]
```

`step` auto resets an env whose episode ended. In that case `obs` is the first observation of the
new episode, and `info["final_local"]` holds the last observation of the ended episode for every env
(equal to `obs["local"]` for envs that did not end). `info["success"]` is the terminated flag.

## 4. Scripted controller: `swarm/scripted.py`

`ScriptedController(env, noise=0.0).act() -> actions [E,K,3]`. It reads the full state.

1. Compute the desired payload motion: the unit vector toward the goal, and the sign of the angle
   error.
2. Grippers move to the two side faces of the payload (the faces parallel to the goal direction)
   and latch.
3. Pushers move to the face opposite the goal direction. The target point along that face shifts by
   `+/- 0.5 hx` in the direction that produces the needed torque. Pushers push with `u = 1` while
   within `d_c`.
4. Within 0.6 m of the goal position, pushers spread along the face to correct the angle first.
5. Gaussian noise with standard deviation `noise` is added to every action and the result is
   clipped.

The controller does not have to be optimal. It has to succeed often enough to fill the offline
buffer. The target is above 60 percent success at `noise = 0`.

## 5. Replay buffer: `swarm/buffer.py`

Per env timelines so that a stale message at `t - L` can be read from the buffer.

Stored per row `(env, t)`: `local [K,16]`, `full [K, F]`, `comm [K,K]`, `action [K,3]`, `reward`,
`terminated`, `truncated`, `next_local [K,16]`, `next_full [K,F]`, `next_comm [K,K]`, `ep_start`
(the row index where the episode began), `target` decoder targets `[K, 6]`: the payload pose
relative to the robot `(dx / A, dy / A, cos theta, sin theta)`, the own latched flag, and the
payload visible flag. The decoder reports position error in metres by multiplying by `A`.

Sampling returns a batch of row indices `(env, t)` plus `(env, max(t - L, ep_start))` for any
requested lag `L`. The offline buffer uses the same class. RLPD sampling draws half of each batch
from each buffer.

## 6. Learning system

### 6.1 Latent space and encoders: `swarm/nets.py`

- Encoder input for robot `k`: the last 3 local observations stacked (48) and the last action (3).
  The stack repeats the first observation at an episode start.
- One encoder per type, output `d = 64`. `e = enc_type(o_stack)`.
- Every latent in the system lives in this one 64 dimensional space: the fresh own encoding, the
  received messages, the rolled forward estimates, and the world model state.

### 6.2 Messages and forward correction: `swarm/belief.py`

A message from robot `j` carries `(e_j, a_j, unc_j, type_j)` stamped at its send time. Robot `i`
keeps a table of the last received message from every teammate and its age. Every robot's table is
initialized at reset with every teammate's step 0 encoding, so no entry is ever missing.

At step `t` with staleness `L`, robot `i` uses the message from step `t - L`. Forward correction
rolls the message forward `L` steps with the world model. Step one uses the recorded action from
the message. Every later step uses the mean action of the policy on the self only belief of the
estimate. The action context during the roll pools the recorded actions of the other messages in
the same table and stays fixed, because the receiver has no newer information. The receiver does
not see the sender's own table, so this is an approximation, and it is the same one the search
makes.

Delivery: a message sent at `t - L` arrives if the link is up at `t` and it survives the dropout
draw. A robot that hears nothing keeps its older entry. Ages are capped at `AGE_MAX = 8`, which
models a channel with a bounded delay.

### 6.3 Fusion

`b_i = fuse(e_i, {(z_hat_j, age_j, unc_j, type_j)})`. One multi head attention layer with the own
encoding as the query and the own encoding plus every estimate as keys and values, then a residual
MLP. The age and type of each estimate are concatenated to the value before the key and value
projections. Uncertainty is not a fusion feature: the offline buffer has no critic, so a stored
uncertainty would mark the data source inside every batch. Uncertainty serves the leader election
only. The output is in the same 64 dimensional space. Permutation invariant over
teammates, so team size can change at test time.

### 6.4 Policy, critic, uncertainty

- Actor: tanh Gaussian `pi(a | b)`, hidden 256, 2 layers.
- Critic ensemble: `M = 10` heads `Q_m(b, a)`, hidden 256, 2 layers, LayerNorm after each hidden
  layer, implemented with batched weights so one forward pass evaluates all heads.
- Target: the mean of two random heads, clamped to `[0, 1]`. RLPD uses the minimum. Here the
  heads disagree by 0.02 to 0.03 near the goal, the minimum sits 0.56 of that below the mean on
  every bootstrap, and over the 90 step horizon that compounded to a value of zero in two runs.
  The reward is one terminal unit, so every true value lies in `[0, 1]`, and the clamp bounds the
  overestimation the minimum was there to prevent. Actor objective uses the mean over all heads.
- Discount `gamma = 0.99`. Adam with learning rate `3e-4` for every module. Fusion has 4 heads.
- Uncertainty `unc_i = std_m Q_m(b_i, mu(b_i))`, the critic ensemble spread at the mean action.
- Entropy temperature `alpha` is learned with target entropy `-3`, starting at `0.1`.
- The critic target does not back up the entropy term (RLPD `backup_entropy = False`). Under a
  sparse terminal reward the entropy stream would reward a long episode and the actor would stall.
- The bootstrap action is the next action recorded in the buffer, a SARSA target on the behavior
  data, with the policy mean only where the next row is unavailable. Two earlier choices failed.
  A sampled bootstrap valued the noisy collection policy and decayed by 0.8 per step. A bootstrap
  on the policy mean held for 4,000 steps and then collapsed: the mean lies about 0.45 from the
  data action under partial observability, and once the critic learned action dependence that
  mean became an out of distribution query. The critic therefore values the behavior data, which
  is 99 percent successful in the offline half, and the search uses it to rank candidates.

### 6.5 World model: `swarm/world_model.py`

`f(z, a_self, c) -> z + g(z, a_self, c)`. `c` is the mean over teammates of `h(type_j, a_j)`, a
permutation invariant action context. `g` and `h` are MLPs, hidden 256. The loss is the mean
squared error between `f(sg(e_t), a_t, c_t)` and `sg(e_{t+1})` on real transitions. `sg` is the stop
gradient. A decoder `dec(z) -> state [6]` is trained with `sg(z)` for error reporting only.

The model is pretrained on the offline buffer encoded by the trained encoders, then trained online
with the same update count as the critic.

### 6.6 RLPD update: `swarm/rlpd.py`

One update:

1. Sample `B / 2` rows from the offline buffer and `B / 2` from the online buffer, `B = 256`.
2. Build beliefs for all `K` robots at `t` and `t + 1` with the current encoders, the message
   procedure of 6.2 at the training staleness `L_train = 1`, and the fusion of 6.3.
3. Critic loss on `B * K` per robot transitions with the shared team reward, on a detached
   belief. The critic never shapes the representation. A bootstrapped loss that shapes the fusion
   has a degenerate fixed point where the belief goes constant, and every run that allowed it
   collapsed between steps 3,500 and 5,000 (the belief spread across states halved and the
   terminal rows could no longer be fit).
3b. Representation loss into the encoders and the fusion: the cloning term of step 4 plus a
   decoder from the belief to the decoder target (payload pose relative to the robot, latch flag,
   visible flag), weight `1.0`. This is privileged state at training time only. It is what forces
   the fusion to recover the payload pose from a teammate's message.
4. Actor loss and temperature loss, plus a behavior cloning term on the offline half of the batch,
   `bc_weight = 1.0` times the squared error between the policy mean and the recorded action. This
   is a deviation from RLPD. Under the sparse reward the critic stays flat in the action for far
   longer than the budget allows, and without the term the actor stayed near zero velocity. The
   cloning gradient also flows into the encoders and the fusion (step 3b). The SAC actor term
   runs on a detached belief.
5. World model loss and decoder loss with stop gradient on the latents.
6. Polyak update of the target critic with `tau = 0.005`.

`utd` updates per batched env step. Default `utd = 4` with 256 envs.

## 7. Search: `swarm/search.py`

At every decision step, for every searching robot `i` and every env, batched:

1. Root estimates: `z_i = e_i` fresh, `z_j = rolled message` for every teammate `j`.
2. Sample `N = 8` candidate own actions from `pi(b_i)` plus the mean action. Imagine every
   teammate's action as `mu(fuse(z_j, others))` from the same estimates. Each root candidate is a
   joint action. Its root score is `Q_0 = Q(b_i, a_i)`, the mean over the critic ensemble.
3. Roll every path one step with `f` and its joint action. Every slot rolls, the own slot
   included. Fuse the child beliefs.
4. Add `beta^d Q(b_i^(d), mu(b_i^(d)))` to the path score. Keep the `J = 4` best paths.
5. Expand every survivor with the mean joint action plus `N` sampled own actions, the teammates
   again at their imagined mean. Repeat steps 3 to 5 to depth `D`.
6. The score of a root action is the best path from it:
   `S = (Q_0 + sum_{d=1}^{D} beta^d Q_d) / (sum_{d=0}^{D} beta^d)`.
   `beta` is the tree search discount, default `0.5`. `D = 0` or `beta = 0` reduces the search to
   the critic argmax over the root candidates with no world model call. The no search baseline is
   the mean action `mu(b)`, and the sweeps report it as depth `-1`.
7. Act with the best root action.

Every candidate is rolled before it is scored, so the number of world model calls grows with the
number of candidates: `(N + 1) K` calls at depth 1 and `J (N + 1) K` at every deeper level, per
searching robot. Teammate estimates roll forward one step per depth together with the own belief
and their age features stay fixed, so the staleness of a teammate estimate is the same at every
depth. Rows are processed in chunks of 512 to bound memory.

Sweep values: depth `-1, 0, 1, 2, 4, 6`, staleness `0, 1, 2, 4`, beta `0.0` to `1.0`.

Leader modes for H4:

- `independent`: every robot searches its own action. Compute per step is `K` searches.
- `leader`: every robot `i` elects the lowest entry of its own uncertainty row, which holds its
  own uncertainty fresh and every teammate's as of the message. A robot that elects itself runs one
  search over joint candidates with `N K` samples per node and broadcasts the joint action. A robot
  that elected someone else executes its slot of that leader's broadcast, or its mean action when
  its elected leader did not elect itself. Because every candidate is rolled, one joint search
  costs the same as `K` independent searches. The evaluation reports the leader disagreement rate
  (the share of robots whose choice differs from the election under fresh uncertainty), the number
  of searches per env, and the share of robots following a leader.
- `round_robin`: the leader index is `t mod K` for every robot, same joint search.

## 8. Differences from the proposal

- Physics is a batched 2D rigid body model in PyTorch, not mjlab. The proposal's risk section asks
  for a trivial version of the task first. This is it. The hypotheses concern the search mechanism,
  not contact physics. mjlab is future work.
- Angular sensing sectors are dropped. Range and payload occlusion remain. This keeps the task
  partially observed and keeps the code small.
- Uncertainty comes from the critic ensemble, which RLPD already provides, instead of a separate
  world model ensemble.
- The team reward is shared. The critic is per robot on its own belief and own action. Teammates are
  part of the environment for the critic. The world model takes the joint action.

## 9. Experiment protocol

All runs use seed `0` unless stated. Evaluation uses 256 envs and reports the mean success rate over
at least 3 evaluation batches, with the standard error.

| Step | Script | Output |
|---|---|---|
| Controls | `scripts/run_controls.py` | `results/controls.json` |
| Offline buffer | `scripts/collect_offline.py` | `data/offline.pt` (ignored by git) |
| Train centralized and decentralized | `scripts/train.py --obs full`, `--obs belief` | `results/train_*.jsonl`, `checkpoints/*.pt` |
| World model error | `scripts/world_model_error.py` | `results/wm_error.json` |
| H1 | `scripts/evaluate.py --depth -1`, `--depth 0`, and `--depth 2` at lag 1 | `results/h1.json` |
| H2 | `scripts/sweep.py --which h2` | `results/h2.json` |
| H3 | `scripts/sweep.py --which h3` | `results/h3.json` |
| H4 | `scripts/sweep.py --which h4` | `results/h4.json` |
| Figures | `scripts/plot.py` | `figures/*.png` |
