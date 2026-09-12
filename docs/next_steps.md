# Next steps

Three drafts from three lenses, written after both runs of the study (`results.md`), merged here
in the order to read them: what would break the conclusions, what would make the search earn its
margin, and what would make the physics honest. Each item names the file and function to change,
the measurement that decides it, and the cost in GPU minutes on the pipeline as it exists.

## The first day

**Status (2026-09-12, evening).** Items 2 to 8 below ran; `results.md` section 14 reports them.
Item 1 was dropped because the bounded lift explained the transfer loss. The findings: the fair
H1 resolves on mjlab (depth 6 beats the sampled policy at lags 0 and 1, not at lag 4); the world
model rollout carries the gain; the critic argmax never beats a random candidate; forward
correction explains the 2D staleness rise; leader mode without fallback is refuted on both
simulators; the critic span ratio r rises with noisier demonstrations but does not predict
whether search helps, so item 6 of the algorithm section is answered in the negative.

The cheapest experiments decide the most. In order of information per GPU minute:

| # | Experiment | Decides | GPU minutes |
|---|---|---|---|
| 1 | Scripted controller and messages masked at 9, 12, 16 robots on mjlab (skeptic F11) | Whether the mjlab transfer loss is crowding or the fusion | 4 |
| 2 | Forward correction off across lags on 2D (skeptic F5a) | Which explanation of the 2D staleness rise survives | 2 |
| 3 | A random world model at depth 2 and 6 on both simulators (skeptic F3) | Whether the rollout's 8 points are the model's | 5 |
| 4 | A decoded state scorer and the scripted value as scorer (skeptic F2) | Whether the search code itself is sound | 8 |
| 5 | Leader mode with no fallback (skeptic F7) | Whether conclusion 3 holds without the confound | 8 |
| 6 | The action signal to noise ratio r on both snapshots (algorithm 1) | Whether r predicts when search helps | 10 |
| 7 | Re baseline H1 against the sampled policy and a random candidate (skeptic F1) | Whether the mjlab H1 survives the fair baseline | 23 |
| 8 | Collect a 2D offline buffer at 65 percent success (more action noise), retrain, measure the critic span (reviewer 1) | Whether the flat 2D critic comes from near perfect demonstrations rather than the physics | 35 |

One physics item is already done: the unloading force is now bounded at 80 percent of the
payload weight (`swarm/env_mjlab.py`), which the physics draft below identified as the likely
cause of the mjlab transfer loss (four latched grippers lifted 100 N against a 78.5 N weight).
The mjlab transfer cells were rerun after the change and `results.md` section 13.7 reports both.

The longer items (three training seeds per simulator, n step targets, search during collection,
a real latch, robot to robot collision) follow in the sections below with their costs.

---

## What would change the conclusions

This section treats the six conclusions of `results.md` section 11 and the four bullets of section 13.8 as
claims to break. Each part names the strongest rival explanation the study did not rule out, the cheapest
experiment that separates the two, and the cost. A 2D training run costs 27 GPU minutes and an mjlab run at
24,000 steps costs 66. One cell is one call of `swarm.evaluate.evaluate` over 3 batches and costs 15 to 100
seconds; price it at 0.5 minutes for a shallow 2D setting, 1.0 for an mjlab depth 2 setting, and 1.7 for depth 6
or a 16 robot team. A cell at `--batches 5` costs 5/3 of the same cell at 3 batches. Nine flags below are new
and the text marks each one.

### Conclusion 1: search helps when, and only when, the critic can rank actions

**Rival.** The mjlab gain is not the critic ranking actions. It is the gain of leaving the deterministic mean,
plus a beam that keeps any candidate other than the mean. The sampled policy alone reaches 65.4 percent at lag 1
against 45.1 for the mean. The critic argmax reaches 61.1, below the sampled policy. Depth 2 reaches 69.5, which
is 4.1 points above the sampled policy against a 2 SE of about 2.9. A selector that picked one of the nine
candidates at random would score near 65 without reading the critic, and the study never ran that arm. A second
rival covers 2D: the critic may be correct and flat there, because the nine candidates may lead to the same
outcome. A flat critic is then correct.

**F1, the re-baselined H1.** `scripts/evaluate.py` with `SWARM_SIM=mjlab`, at lag 0 and lag 1, `--envs 256
--batches 5`, env seeds 1000 to 1004, on the existing snapshot. Four arms: `--depth -1 --policy sample`;
`--depth 0 --rank random` (new, a random root candidate in place of the argmax); `--depth 0`; `--depth 6 --beta
0.9`. Add `--policy noisy --noise-scale s` (new) for s in 0.1, 0.25, 0.5, 0.75, 1.0 at lag 1 on both simulators.
Cells: 8 plus 10. Decision rule: conclusion 1 survives only if depth 6 beats both the sampled policy and the
random candidate arm by more than 2 SE at both lags. If the random candidate arm matches depth 0, the critic
does not rank actions on mjlab either and the conclusion inverts. Cost: **23 GPU minutes**.

**F9, do the candidates differ at all.** New `scripts/candidate_return.py`, about 40 lines, on
`swarm.rollout.Runner`. At 128 saved 2D states it enumerates the nine root candidates, executes each one for one
step, follows the mean policy to the end, and reports the spread of realized return plus the rank correlation
between the critic value and that return. Seeds 1000 to 1002. Decision rule: a return spread under 0.05 means
the 2D critic is flat because the task is flat, and "quasi static physics did not produce action dependence"
becomes "the nine candidates do not differ". A large spread with a near zero correlation means the critic is
wrong and the conclusion stands. Cost: 9 arms at 1 cell, **9 GPU minutes**.

### Conclusion 2: staleness degrades the search in the predicted direction

**Rival.** The mjlab best depth is 6, 6, 6, 4, and 6 is the largest depth in the grid. At three of four lags the
argmax sits on the edge of the axis, so the grid is censored at the top and no collapse has been observed. The
one drop, from 6 to 4 at lag 4, comes from cells 2.1 points apart against a 2 SE of about 8.6 (63.0 ± 1.6
against 60.9 ± 4.3), and every depth from 2 to 6 lies inside the noise of every other at every lag. The best
depth statistic is an argmax over five noisy cells, and an argmax over a flat row produces a descending sequence
by chance often enough to matter. The proposal's collapse claim is not supported. It is only not refuted.

**F6, uncensor the depth axis.** `scripts/sweep.py --which h2` with `SWARM_SIM=mjlab`, `DEPTHS` extended with 8
and 12 (one line in the module constant), `--envs 128 --batches 7`, env seeds 1000 to 1006, depths 2, 6 and 12
at lags 0 and 4 only, which is 6 cells. Record the per env success vector in the cell JSON (new, one line in
`swarm.evaluate.run_batch`), so a paired bootstrap over envs tests the ordering at no GPU cost. Pre-registered
decision rule: the collapse claim is supported only if depth 6 beats depth 2 at lag 0 by more than 2 SE, and the
best depth at lag 4 sits at least one grid level below the best depth at lag 0 under a paired bootstrap at the
95 percent level. If depth 12 does not fall below depth 6 at any lag, report that no falloff was observed and
that H2's headline is untested; conclusion 2 then keeps only its H3 half and its depth 1 minus depth 0 half,
which need no argmax. Cost: **24 GPU minutes**.

### Conclusion 3: a broadcast joint search loses to independent search

**Rival.** The leader arm is a mixture, on both simulators, not on 2D alone. In `swarm/search.py` a robot whose
elected leader did not elect itself acts with its mean action: 44 percent of robots on 2D, 31 percent on mjlab.
Round robin forces every robot to follow but changes the election rule at the same time. No arm holds the
election fixed and removes the fallback.

**F7, a leader mode where nobody falls back.** New mode `leader_nofallback` in `SearchConfig.mode`, about 5
lines: every robot that at least one robot elects runs the joint search, and every robot follows its elected
leader. `scripts/evaluate.py --mode leader_nofallback --depth 2 --lag 1 --envs 256 --batches 3` on both
simulators, beside the three existing modes, which is 8 cells. Match compute by cutting `--candidates` until
`ms_per_step` equals the independent arm. Decision rule: conclusion 3 stands as written only if
`leader_nofallback` loses to independent search by more than 2 SE on both simulators. If it beats independent
search on 2D, the 2D H4 verdict was about the search and the fallback sentence is wrong. Cost: **8 GPU
minutes**.

### Conclusion 4: the belief pipeline works in both physics

**Rival A, the staleness rise.** The 2D no search baseline rises with lag (38.8, 50.3, 63.8, 61.7) and with
dropout (50.3, 55.7, 59.4, 63.5), while the mjlab baseline is flat on both axes. Three explanations compete and
the study separates none: (a) the fusion over trusts a fresh message and the age feature lets it down weight an
old one, (b) the rolled estimate is a smoother and better input than a raw encoding, (c) a train test mismatch,
because training runs at lag 1. Explanation (c) predicts a peak at lag 1 and predicts that dropout, which raises
the effective age above 1, should hurt. Dropout helps, so (c) explains the lag 0 dip and nothing else. Both (a)
and (b) fit the lag rise and the dropout rise. The mjlab flatness fits (b) too, because 2D leans on messages far
harder: masking costs 36.8 points on 2D and 16.2 on mjlab.

**F5a and F5b, the roll and the training lag.** F5a adds `--roll-teammates off` to `scripts/evaluate.py` (new),
which feeds the raw stale encoding to the fusion in place of the forward corrected one; run lags 0, 1, 2 and 4
on 2D, `--envs 128 --batches 3`, seeds 1000 to 1002, which is 4 cells and **2 GPU minutes**. If the curve
flattens, the cause is (b), and "the fusion prefers older estimates" becomes "the forward correction improves
with the steps it integrates". If the rise survives, the cause is (a) or (c), and F5b separates them:
`scripts/train.py --obs belief --lag 4 --steps 12000 --out belief_lag4` on 2D, then `scripts/ablations.py --ckpt
checkpoints/belief_lag4_best.pt`, which is 11 cells and **31 GPU minutes**. A peak at lag 4 and a drop at lag 1
make the mismatch the cause and the fusion claim comes out. A peak still at lag 2 clears it.

**Rival B and F11, the mjlab transfer loss.** The mean policy falls to 25.8 percent at 16 robots on mjlab and
reaches 95.6 on 2D. Crowding and a fusion that fails on a longer message table predict the same number, and the
scripted controller separates them, because it has no fusion and no learned weights. Add `--team` to
`scripts/run_controls.py` (new, about 3 lines; `TEAMS` already lives in `scripts/common.py`). Run
`SWARM_SIM=mjlab python scripts/run_controls.py --team p4g4s1`, then `p6g5s1`, then `p8g7s1`, at 256 envs, seed
0, then the masked arm of the learner at the same three sizes: `scripts/evaluate.py --team p8g7s1 --no-messages
--depth -1`. Cells: 3 scripted rollouts and 3 masked cells. Decision rule: a scripted fall from 83.6 to below 50
percent at 16 robots makes crowding the cause and the claim holds. A scripted controller above 80 percent puts
the cause in the fusion or the actor, and the sentence comes out; the masked arm then says whether the loss
tracks the table length. Cost: **4 GPU minutes**.

### Conclusion 5: the sampled policy is the baseline a fair claim must beat

**Rival and experiment.** The sampled policy is not a designed baseline either, so beating it settles nothing.
Its noise scale is the standard deviation of an actor whose alpha decayed toward zero while the critic stayed
flat in the action, which `swarm/rlpd.py` documents, so that scale is an accident of training. A baseline that
tunes one scalar is fairer and far cheaper than a search at 66.6 ms per step, and if a tuned scale reaches 70
percent on mjlab the search has no margin left to defend. The 2D side is understated too: against the sampled
policy at lag 1 (62.2) the 2D depth 2 search (43.2) loses 19 points, not 9. The test is the five noise cells of
F1 plus a sampled baseline row across the mjlab H2 grid. `scripts/sweep.py` hard codes `policy = "search" if
depth >= 0 else "mean"` in `cell()`, so that row needs a sentinel depth of -2 or a new `--baseline sample` flag;
run it at lags 0, 1, 2 and 4, `--envs 128 --batches 5`. Decision rule: report every H2 gain against the sampled
row, and if the gain at the best depth falls inside 2 SE at three of four lags, restate conclusion 2's "+35
falling to +24" in the smaller units. Cost: **4 GPU minutes**, inside the F1 budget.

### Conclusion 6: nine documented changes were needed to train at all

**Rivals.** The nine changes are a path through one seed. Each change followed one failed run, and a run that
fails at seed 0 can succeed at seed 1 with no change. Nothing measures the seed spread, so nothing separates
"this change was needed" from "this run was unlucky", and the same gap makes every cross-run claim unsafe:
belief against full, 2D against mjlab, and the sign of H1. The second rival is the target itself.
`RLPDConfig.target_policy = "data"` means the critic values the behavior mixture of the scripted controller and
the collection policy, not the learned policy. A critic that values a mixture has no reason to rank the learned
policy's own candidates, so the 2D flatness may belong to the target and not to the physics. The mean target
collapsed in run 7, before the run 8 and run 9 fixes existed, and it has never been tried on the final recipe.

**F4, three training seeds per simulator.** `scripts/train.py --obs belief --seed 1 --out belief_s1` and `--seed
2 --out belief_s2` on both simulators, then H1 at lag 1 with the four arms of F1 on each snapshot. Decision
rule: the sign of the H1 gap must agree across 3 of 3 seeds within a simulator. One positive 2D seed turns
conclusion 1 into a claim about one snapshot. Report the seed spread of best evaluation success; every later
necessity claim has to exceed it. Cost: 2 times 27 plus 2 times 66 plus 16 cells, **202 GPU minutes**.

**F8, a critic that values the learned policy.** Add `--target-policy` to `scripts/train.py` (new, one line; the
config field already accepts `mean`). Run `--obs belief --target-policy mean --steps 12000 --out belief_mean` on
2D, which now carries the target copies, the [0, 1] clamp, the cloning anchor, and the belief decoder. Then run
H1 at lag 1 and the spread diagnostic of F9. Decision rule: no collapse at step 5,000, a candidate spread above
0.01, and a positive H1 together put the 2D flatness on the SARSA target and make conclusion 1's physics
explanation wrong. A second collapse makes the SARSA target load bearing and conclusion 1 survives with that
caveat stated. Cost: **31 GPU minutes**.

**F12, leave one in.** On 2D only, retrain with `--target-reduce min`, with `--bc-weight 0`, and with
`--dec-b-weight 0`, one seed each. Decision rule: a change counts as necessary only when removing it drops the
best evaluation success by more than the seed spread F4 measures. Run it after F4, never before. Cost: **81 GPU
minutes**.

### The two task changes that no result controls for

**F10.** The 2D task was eased once: sensing 3 m, communication 6 m, goals 1.5 to 3 m. The mjlab robots run at
2.5 m/s in place of 1.5 because a velocity servo accelerates. The eased 2D task gives every robot more of its
own sensing, which is one way to flatten a critic in the action, and the higher mjlab speed raises the scripted
ceiling from 52 to 75 percent at 64 envs, which is one way to make actions matter. Either change can set the
sign of H1. Two arms follow. First, 2D retrained on the pre-ease ranges, sensing 1.5 m and communication 3 m,
through new `--sensing` and `--comm` flags on `scripts/train.py`, then H1 at lag 1. Second, mjlab retrained at
`v_max=1.5` through a new `SWARM_V_MAX` variable read in `scripts.common.env_config`, then H1 at lag 1. Decision
rule: the sign of the H1 gap must not change in either arm, and the comparison is the gap, not the level,
because the lower speed also lowers the scripted ceiling. A positive H1 on the harder 2D task puts the flat
critic on the eased task and not on quasi static physics. A negative H1 on mjlab at 1.5 m/s puts the mjlab
confirmation on the speed and not on contact physics. Cost: **101 GPU minutes**.

### Controls that prove the search implementation

**F2, the positive control: a scorer that cannot be flat.** Every claim above assumes `swarm/search.py` does
what its docstring says, and no experiment in the study tests that. New flag `--scorer decoded` in
`scripts/evaluate.py`, about 20 lines, replaces `_q` in the search with a value built from the belief decoder
that already exists, `nets.dec_b`, which maps a belief to the payload pose: `V(b) = 1 - clamp(w_p
||p_hat - p_goal|| + w_a |theta_hat - theta_goal|, 0, 1)`. Add `--scorer scripted`, which scores a candidate
by the progress measure `swarm/scripted.py` already computes. Run depth 0 and depth 2 at lag 1 on both
simulators, `--envs 256 --batches 3`, seeds 1000 to 1002, which is 8 cells. Decision rule: if the decoded scorer
at depth 2 beats no search on 2D by more than 2 SE, the search code and the world model are sound and the 2D
failure lies in the critic, which is what conclusion 1 asserts. If the decoded scorer also loses on 2D, the
fault is in the search or in the rollout, and every conclusion that names the critic comes out until the search
is fixed. Cost: **8 GPU minutes**.

**F3, the negative control: a world model that knows nothing.** New flag `--wm random` in `scripts/evaluate.py`,
about 3 lines, which reinitializes `nets.wm` after `Agent.load`. Run depth 2 and depth 6 at lag 1 on both
simulators, `--envs 256 --batches 3`, which is 4 cells. Decision rule: on mjlab the random model must cost more
than 2 SE against the trained model at depth 6. A match means the mjlab gain comes from re-ranking root
candidates and not from imagination, and the sentence "the world model rollout accounts for 8 of them" fails. On
2D the two arms should agree. Cost: **5 GPU minutes**.

### Ranked by information per GPU minute

The total is 529 GPU minutes, about 8.8 hours on the one RTX 5090. The first six items cost 36 minutes together
and they decide four of the six conclusions.

| Rank | Experiment | Cost | Conclusion at stake |
|---|---|---|---|
| 1 | F11 scripted controller at 9, 12 and 16 robots on mjlab | 4 min | 4 |
| 2 | F5a roll turned off across lags on 2D | 2 min | 4, 2 |
| 3 | F3 randomly initialized world model, both simulators | 5 min | 1 |
| 4 | F2 decoded state scorer, both simulators | 8 min | 1 |
| 5 | F7 leader mode with no fallback, both simulators | 8 min | 3 |
| 6 | F9 candidate return spread and rank correlation on 2D | 9 min | 1 |
| 7 | F1 H1 re-baselined on sampled, random and noisy arms | 23 min | 1, 5 |
| 8 | F6 depth axis extended to 12 at 7 batches | 24 min | 2 |
| 9 | F5b 2D retrained at lag 4 | 31 min | 4 |
| 10 | F8 mean action target on the final recipe, 2D | 31 min | 1, 6 |
| 11 | F4 three training seeds per simulator | 202 min | every one |
| 12 | F12 leave one in on three of the nine changes | 81 min | 6 |
| 13 | F10 pre-ease 2D and 1.5 m/s mjlab | 101 min | 1 |


---

## Making the search earn its margin

### The problem

The search has to beat the sampled policy, not the mean action. On mjlab it beats the mean action by
24.5 points and the sampled policy by 4 to 7 points, and only depth 4 to 6 reaches 2 SE
(`docs/results.md` sections 13.3 and 13.7). On the 2D task the same code loses 9 points, because the
critic spans 0.003 across the policy's own candidates and the argmax ranks noise (section 5). The
ranking signal is weak, and the search spends compute on it.

### The quantity that unites the two runs

Define the action signal to noise ratio at a belief `b`:

    r(b) = (max_c Q(b, a_c) - min_c Q(b, a_c)) / std_m Q_m(b, mu(b))

The numerator is the span of the ensemble mean over the nine root candidates of `swarm/search.py`
`_candidates`. The denominator is the ensemble spread that `swarm/rollout.py` computes as `unc`.
`docs/math.md` section 4 measures that spread at 0.02 to 0.03 near the goal, and `docs/results.md`
section 5 measures the 2D span at 0.003. So 2D sits near r = 0.1 and its argmax reads noise. The
mjlab critic must sit above 1, because its argmax alone adds 16 points.

### A protocol requirement that comes first

The effect under test is 4 to 7 points. A 3 batch cell at 128 envs carries a standard error of 0.3
to 4.3 points in the current tables, so 2 SE reaches 8 points in the worst cells. Run every margin
cell at 256 envs and 6 batches. That doubles the cell cost, and the costs below include the doubling.
Planning figures: a 2D training run is 27 minutes, an mjlab run at 24,000 steps is 66 minutes, and
one sweep cell is 15 to 100 seconds. Budget 2 minutes per margin cell.

### The ranked experiments

#### 1. Measure r, and add a matched random score control

**Hypothesis.** r predicts whether search helps. r below 1 predicts a loss, r above 1 predicts a
gain, and the size of the gain tracks r.

**Change.** Add `scripts/action_signal.py`. It loads a checkpoint, builds beliefs with
`swarm/belief.py` `beliefs`, samples eight actions with `swarm/nets.py` `Actor.sample_n`, and reports
r per lag. For the training curve, return `q_span` and `q_std` from `swarm/rlpd.py` `Agent.update` on
`d["b"]`; `scripts/train.py` writes every returned key to `results/train_*.jsonl` already. For the
control, add a `score` field to `SearchConfig` and return `torch.rand_like(q)` from `_q`.

**Cost.** The probe is under 1 GPU minute for both snapshots. The control is 4 cells, 8 to 16 GPU
minutes. The training curve is free, because it rides every run below.

**Verdict.** Confirms if r is near 0.1 on the 2D snapshot, above 1 on the mjlab snapshot, and the
random score control loses the whole 24.5 points on mjlab. Refutes if the mjlab control keeps most of
the gain, which would put the gain in the candidate machinery and not in the ranking.

#### 2. Score candidates pessimistically instead of by the ensemble mean

**Hypothesis.** The argmax over 9 noisy heads selects the candidate with the largest head error. The
2D evidence is direct: the argmax picks the candidate farthest from the policy mean, 0.34 against
0.28. A lower confidence bound removes that selection.

**Change.** `swarm/search.py` `_q` returns `q.mean(0) - cfg.lcb * q.std(0)`. Add `lcb: float = 0.0`
to `SearchConfig` and a `--lcb` flag to `scripts/evaluate.py` and `scripts/sweep.py` `cell`.

**Cost.** 16 cells: lcb in {0, 0.5, 1, 2} times depth {0, 2} on both simulators, 30 to 60 GPU minutes,
no training run.

**Verdict.** Confirms if 2D depth 0 rises from 42.3 toward the 52.3 of the mean action as lcb grows,
and the mjlab margin over the sampled policy passes 7 points at depth 2. Refutes if the curve is flat
in lcb on both simulators, which would leave the whole burden of the critic on item 4.

#### 3. Give the search more candidates, a wider beam, and a wider proposal

**Hypothesis.** A ranking that carries signal earns more margin from a larger and wider candidate
set. The slope of success against the candidate count measures r from the outside. The 2D run
predicts a negative slope and the mjlab run predicts a positive one.

**Change.** `scripts/sweep.py` already forwards `--candidates` and `--beam`. Add a proposal
temperature: give `SearchConfig` a `temp` field, add a `scale` argument to `swarm/nets.py`
`Actor.sample_n` that multiplies `log_std.exp()`, and pass it from `swarm/search.py` `_candidates`.
The baseline arm is `scripts/evaluate.py --depth -1 --policy sample`.

**Cost.** On mjlab: candidates in {2, 4, 8, 16, 32} times depth {0, 2} is 10 cells, beam in
{1, 2, 4, 8} is 4 cells, temperature in {1.0, 1.5, 2.0, 3.0} times depth {0, 2} is 8 cells, plus 1
baseline cell. A cell at 32 candidates costs about four times a cell at 8. 60 to 90 GPU minutes, plus
4 cells on 2D as the falsification arm at 10 GPU minutes.

**Verdict.** Confirms if success rises with the candidate count on mjlab and the best cell beats the
sampled policy by more than 7 points at 2 SE. A temperature above 1 that adds points says the policy
spread is too narrow. Refutes if success is flat or falls with the candidate count, which is the
winner's curse and caps what any candidate change can buy.

#### 4. Train the 2D critic with n step or Monte Carlo SARSA targets

**Hypothesis.** The 2D flatness is a credit assignment artifact of the one step target. Under a
sparse terminal reward a one step target teaches the state dependence first and the action dependence
last. An n step return attaches the outcome to the action taken, so r rises inside the same budget.

**Change.** `swarm/rlpd.py`. Add `n_step: int = 1` to `RLPDConfig`. `Agent._batch` gathers rows t to
t + n and returns the discounted reward sum and the row at t + n. `Agent.update` builds
`y = sum_m gamma^m r_{t+m} + gamma^n (1 - terminated) Q_target(b_{t+n}, a_{t+n})`, then applies the
existing `[0, 1]` clamp. The target stays SARSA on the behavior data, so it stays consistent with the
behavior policy that `docs/math.md` section 3 defends. Two windows are short. An episode that ends
inside the window stops the sum at the terminal row and zeroes the bootstrap. A window past
`buf.t - 1` shortens for that row only. `swarm/buffer.py` holds `ep_start` and `terminated` per row.

**Cost.** Three 2D runs at n in {1, 5, 20} is 81 GPU minutes, and the H1 triple at each run is 20
more. 100 GPU minutes to decide. The mjlab confirmation is 66 plus 20 minutes, and it runs only if
2D confirms.

**Verdict.** Confirms if r on the 2D snapshot rises above 1 and depth 0 rises from 42.3 above 52.3.
The item 1 curve shows the rise before step 6,000, so a failing arm stops early. Refutes if r stays
below 1 at n = 20, which puts the flatness in the quasi static task and not in the target.

#### 5. Run the search during data collection, as the proposal asks

**Hypothesis.** The critic trains on the behavior policy and answers queries at the search's
distribution. Collecting with the search closes that gap, and it raises the data quality too.

**Change.** None in the library. `scripts/train.py` already takes `--collect-depth`. Add
`--collect-candidates` and pass it into the `SearchConfig` that `scripts/train.py` builds, so the
collection search runs narrower than the evaluation search.

**Cost.** This is the expensive item. A depth 2 search costs 46 ms per step against 28 ms for the mean
at 128 envs, and collection is the part that grows. Budget 2.5 to 3.5 times the mjlab run, so 165 to
230 GPU minutes plus 20 minutes of cells. The cheap variant collects at depth 1 with 4 candidates
after step 6,000, at 100 to 130 GPU minutes.

**Verdict.** Confirms if the margin over the sampled policy at depth 2 passes 7 points at 2 SE and r
on the new snapshot is higher than on the current one. Refutes if the margin does not move or the run
trains worse, because a search narrows the data and a narrower buffer can lower r.

#### 6. Value the learned policy again, with a support constraint

**Hypothesis.** The SARSA target makes the critic rank the behavior mixture, not the policy the search
samples from, as `docs/results.md` section 12 states. The run 7 collapse came from an out of
distribution query before the target representation existed. With the run 9 recipe the mean action
target may hold, and a critic that values its own policy is the critic the search queries.

**Change.** `RLPDConfig.target_policy = "mean"` exists. Add `mean_target_radius: float`. In
`Agent.update`, use the policy mean in the bootstrap only where
`||mu(b') - a_next|| < mean_target_radius`, and use the recorded action elsewhere. The measured gap is
0.45 in `docs/math.md` section 3, so sweep the radius at 0.2, 0.45, and infinity. The named failure is
the run 7 collapse. Stop the arm when the terminal row fit falls below 0.5 after step 5,000.

**Cost.** One 2D run at 27 minutes decides whether the collapse returns. The mjlab run is 66 minutes
and it starts only after the 2D arm survives. With cells, 35 GPU minutes to gate and 90 to complete.

**Verdict.** Confirms if the run reaches the end without a collapse and r on the snapshot rises.
Refutes if the collapse returns at every radius, which closes the question and confirms the current
recipe.

#### 7. Remove the cloning term late in training

**Hypothesis.** The cloning term pins the policy mean on the scripted action for the whole run, and it
holds the proposal distribution at the scripted spread. Removing it late lets the critic improve the
policy, which raises the baseline and the candidate set together.

**Change.** `swarm/rlpd.py`. Add `bc_off_step: int` to `RLPDConfig`, set the cloning weight to zero in
`Agent.update` after that step, and add the flag to `scripts/train.py`. The named failure is run 2,
where the actor drifts to near zero velocity. Item 1 predicts that failure whenever r is below 1, so
run this arm only after item 4 or item 6 raises r. The monitor is `train_success`.

**Cost.** One 2D run at 27 minutes and one mjlab run at 66 minutes, plus 20 minutes of cells. 115 GPU
minutes for the pair. Gate on the 2D arm.

**Verdict.** Confirms if the mean policy rises toward the sampled policy and the search keeps its
margin over it. Refutes if success falls after the cloning weight drops, which makes the cloning term
load bearing at this budget.

#### 8. Score candidates by decoded progress toward the goal instead of Q

**Hypothesis.** The tree and the world model are sound, and the critic is the only weak part. A score
that reads the decoded payload to goal distance from the rolled latent tests that claim directly,
because it holds the tree fixed and swaps the score.

**Change.** The decoder of `swarm/nets.py` predicts the payload pose relative to the robot. The goal
relative to the robot sits in the local observation and is never masked, but no head decodes it. Fit
a two output head on the frozen encoder from the stored offline buffer, then score a path by
`-||dec_goal(z) - dec(z)[0:2]||`, wired behind the `score` field of item 1. The permanent version
extends `decoder_target` in `swarm/env.py` `TransportEnv.state` to 8 columns and sets
`Decoder(out_dim=8)`; `swarm/env_mjlab.py` inherits both.

**Cost.** The head fit on stored transitions is about 5 GPU minutes, and the sweep is 8 to 12 cells at
20 to 30 GPU minutes. A retrain follows only if the post hoc head fits worse than the current decoder.

**Verdict.** Confirms if progress scoring beats Q scoring on 2D by more than 9 points, which recovers
the H1 loss and puts the fault in the critic alone. Refutes if progress scoring also loses on 2D,
which puts part of the fault in the tree or in the teammate imagination. This item ranks last because
the score is hand designed and it needs a goal metric that this task happens to provide. It answers a
different question from the project's, because the QWM claim is about searching on a learned critic.
Keep it as a diagnostic and do not report it as a method.

### What I run first, and why

Run item 1 first. It costs under 1 GPU minute for the snapshot probe and nothing for the training
curve, and it turns the study's central explanation into a measured number that predicts the result
of every other item. Items 2 and 3 are pointless if the random score control keeps the mjlab gain.
Items 4, 6, and 7 each spend 27 to 115 GPU minutes to move r, and none of them has a stopping rule
until r is instrumented. The 2D span of 0.003 and the mjlab gain of 24.5 points are two readings of
one quantity that the pipeline never logged. Log it, then spend GPU time.

Run item 2 second, because it can recover points with no training run. Run item 3 third, because its
slope measures r from the outside. Start item 4 in parallel on the 2D task, because its verdict is
readable from the item 1 curve before each run ends.


---

## Physics fidelity and scale

The mjlab port produced the main positive result. It also simplifies the physics in five ways that
`docs/mjlab_port.md` section 5 and `docs/results.md` section 12 record: a kinematic latch, a central unloading force,
one contact point for a cylinder against a box, no robot to robot collision, and a top speed of 2.5 m/s against 1.5
m/s in the 2D task. This section ranks the work that removes those simplifications, cheap tests of large risks first.

One measurement drives the ranking. On 2D the mean policy improves with team size: 50, 89, 93, 96 percent at 6, 9, 12,
16 robots. On mjlab it falls: 47, 29, 31, 26 percent, and search recovers most of the loss (71, 69, 54, 43).
`docs/results.md` section 13.7 reads the fall as crowding. Three of the five simplifications produce the same fall, so
the reading is not yet safe. Every hour estimate is engineering time. Every GPU minute estimate assumes the measured
66 minutes for a 24,000 step belief run at 256 envs and the measured 11,850 env steps per second.

### 1. A real lift instead of the central unloading force

**The change and the risk it retires.** `MjlabTransportEnv.step` in `swarm/env_mjlab.py` computes
`lift = mj.lift_force * self.latched.sum(1)` and writes it to `wrench[:, 2]`. Nothing bounds it. `mj.lift_force` is 25
N and the payload weighs 78.5 N, so four latched grippers lift 100 N. The payload leaves the floor, the friction goes
to zero, and the box becomes a free body that two 3 N pulls steer. The default team has two grippers, so training
never meets this state. The transfer teams have 4, 5, and 7 grippers, and the no search column drops from 47.4 to 28.6
percent at exactly the step from two grippers to four. Stage A is one line: clamp the total lift below the payload
weight. It retires the risk that the headline transfer result measures a lift off and not crowding. Stage B moves the
lift to the latch point, so the wrench carries the pitch moment a real gripper applies and the box loads two floor
contacts instead of four. It retires a fiat: the port replaced one threshold that falls by decree with a lift that
also falls by decree, applied where it cannot tip.

**The tipping risk.** The payload half extents are 0.8 by 0.4 by 0.15 m. One 25 N lift at a long face gives 10 N.m
about the far edge against a restoring 31.4 N.m. Two latches at the same face give 40 N.m against 62.8 N.m, a margin
of 1.6. Stage B must test tipping, because a tipped payload breaks two readers: `_read_state` computes the yaw from
`xquat` with the flat box formula, and `rect_contact` treats the payload as a plane rectangle. Stage B must reject an
episode whose pitch exceeds a bound and report the rejection. The clamp of the payload center to `center_limit` at the
end of `step` is the wrong pattern, because it hides a non physical state instead of naming it.

**Cost and measurement.** Stage A: 1 hour, 30 GPU minutes. Stage B: 10 hours, 120 GPU minutes. Re run the transfer
grid of `docs/results_tables_mjlab.md` with stage A alone, on the same snapshot and seeds. If the no search column
stops falling with team size, the crowding claim is wrong. If it still falls, crowding survives and items 3 and 4
become the explanation. Log the fraction of substeps in which the payload has no floor contact, before and after.

### 2. The drive force and the top speed

**The change and the risk it retires.** `MjlabConfig.force_limit` is `(10.0, 3.0, 3.0)` newtons, and `build_spec`
writes that number into the `forcerange` of both slide actuators. One number caps two unrelated things: how hard the
robot pushes the payload, and how fast it accelerates its own 2 kg body. A pusher accelerates at 5 m/s2 and needs five
control steps to reach 2.5 m/s. A gripper and a scout accelerate at 1.5 m/s2 and need 1.7 s. The 2D task gives every
type 1.5 m/s at once. A gripper's pull does not come from its actuator: `step` applies it through `xfrc_applied` at
`mj.grip_force`, a separate 3 N. So raising the gripper `forcerange` to 10 N changes only how fast a gripper reaches
its latch point, and the friction facts of `docs/mjlab_port.md` section 1 hold. The scout must still not move the
payload, so exclude it from the payload collision with `contype` and `conaffinity`. The step retires two confounds.
Section 2 of that document raises the top speed from 1.5 to 2.5 m/s and reports a scripted gain from 52 to 75 percent,
which may be the speed or the acceleration ramp. And p8g7s1 has seven gripper class robots at 1.5 m/s2, so it is a
slow team for a reason the task never intended.

**Cost and measurement.** 4 hours, 60 GPU minutes. Record `qvel` over the first ten control steps after a reset and
report the time to 90 percent of the commanded speed, per type. Then run `scripts/run_controls.py` at 256 envs at 2.5
m/s as today, at 1.5 m/s as today, and at 1.5 m/s with every drive force at 10 N. If the third matches 83.6 percent,
the speed mismatch is an artifact of the force limit and the port returns to 1.5 m/s. If not, the port keeps 2.5 m/s
and `docs/mjlab_port.md` states the servo lag as a known difference.

### 3. Robot to robot collision

**The change and the risk it retires.** `build_spec` gives every robot geom `contype=2, conaffinity=1`, so no robot
pair collides. Set both to 3. Robot against robot then matches on bit 0 and bit 1, robot against payload still matches
because the payload uses `contype=5, conaffinity=5`, and robot against floor and wall still do not match because those
use bit 2. The change is one line. The cost is `nconmax`, today 24, and `njmax`, today 96: at 16 robots there are 120
robot pairs per world, and `mujoco_warp/_src/collision_driver.py` routes a cylinder against a cylinder to the convex
path. Today several robots occupy one point on one payload face, their contact normals cancel inside the solver, and
the servos stall. That is a simulation artifact. With collision on, robots block each other, which is real crowding.
Both effects move the success number the same way, so the transfer result cannot separate a policy failure from a pile
up. The 2D task also omits robot to robot collision and its larger teams improve, so the omission alone does not
explain the fall.

**Cost and measurement.** 3 hours, 90 GPU minutes. Before the change, log the mean count of robots whose centers lie
within one diameter of each other, per team size. Then turn collision on, raise `nconmax` and `njmax` until the solver
stops saturating, and re run the transfer grid. Report the slope of success against team size in both settings, and
throughput in both. A slope that stays negative means the policy does not generalize, which is the interesting result.
A flat slope means the port was the problem.

### 4. Contact points for the cylinder against the box

**The change and the risk it retires.** `docs/mjlab_port.md` section 5 states that a cylinder against a box gets at
most one contact point. This is not a flag. In `mujoco_warp/_src/collision_convex.py` the multicontact expansion sits
behind a `wp.static` guard at line 860 that admits only BOX and MESH on both sides, and a cylinder is neither, so the
branch is compiled out. MULTICCD is a disable flag (`use_multiccd = m.opt.disableflags & DisableBit.MULTICCD == 0`,
line 1195), so multicontact is already on and enabling it again changes nothing. The fix is the geom type: make the
pusher a box in `build_spec`. Box against box takes the multicontact path unconditionally, and line 1198 allocates
four polygon slots for it. A robot has only two slide joints, so a box pusher keeps a fixed world orientation and
meets a rotated face along an edge, which gives two points, not four. Two points already give a moment arm, so the
pusher applies a torque through its own patch. Today the payload rotates only from off center pushes against the floor
contacts, and the angle error is 0.140 rad against a 0.2 rad tolerance. A flat face against a flat face needs a hinge
about z and a third actuator, which changes `ACT_DIM` and invalidates every checkpoint. Do not do that here. One side
effect may pay for the work: line 1193 sets `epa_iterations` to 16 when every convex pair is box against box and to
`m.opt.ccd_iterations`, default 50, otherwise.

**Cost and measurement.** 6 hours, 60 GPU minutes. Log the contact count between a pusher geom and the payload geom
over an episode, before and after. Then command a pure yaw goal with one pusher and measure the payload yaw rate. Then
re run `scripts/run_controls.py` and report success, angle error, and throughput. Keep the step if the angle error
falls and throughput does not.

### 5. A real latch as a per world equality constraint

**The change and the risk it retires.** Replace the kinematic latch in `step`, which writes the gripper `qpos` every
substep, with an equality constraint per gripper, declared inactive in `build_spec` and toggled per env. The kinematic
latch gives the gripper infinite strength in the normal direction and zero mass, so it cannot be pulled off and adds
no inertia. A connect constraint can break under load if the port gives it a `solimp` ceiling, and it feeds the
payload's momentum back into the gripper.

**What I verified in the mjlab and mujoco_warp source.** `Data.eq_active` has shape `(nworld, neq)`
(`mujoco_warp/_src/types.py` line 2229) and `put_data` fills it per world from `eq_active0` (`io.py` line 1815), so a
latch toggle is an in place write through the `WarpBridge` and needs no graph recapture. `Model.eq_data` has shape
`("*", "neq", vec11)` (`types.py` line 1744), and the leading dimension is the world dimension, 1 by default. Every
equality kernel reads `eq_data[worldid % eq_data.shape[0], eqid]` (`constraint.py` lines 239, 562, 707, 1058, 1521),
so after expansion each world reads its own row. This is the fact the earlier draft could not confirm.
`Simulation.expand_model_fields` accepts any name that exists on `mj_model`, and `MjModel` has `eq_data`;
`sim/randomization.py` tiles any field whose `shape[0]` is 1; `expand_model_fields` then calls `create_graph()`, so
the expansion happens once at construction and never per latch. The connect equality reads `anchor1 = data[0:3]` in
body 1's frame and `anchor2 = data[3:6]` in body 2's frame and drives the two world points together with three rows
(`constraint.py` lines 239 to 256), so a latch writes three floats per env and the gripper side anchor is a constant.
`mjwarp.reset_data` restores `eq_active` from `eq_active0` (`io.py` line 2504), but the env never calls
`Simulation.reset`: `_reset_masked` writes `qpos` and `qvel` and calls `forward`. So the env must clear `eq_active`
itself on reset, or a latch leaks into the next episode.

**Use connect, not weld.** The earlier draft proposed `mjEQ_WELD`, which fixes the relative orientation and takes six
rows. The gripper body has two slide joints and no rotational freedom, so a weld would pin the payload yaw to the
world through a single gripper. That is stronger than the 2D rule, where a latched gripper applies a force at a point
and the payload still turns. `mjEQ_CONNECT` matches that rule and costs half the rows.

**What I could not verify.** No domain randomization term in `mjlab/envs/mdp/dr/` targets `eq_data`, so the expansion
path is untested in this tree for this field. I did not confirm that a connect constraint and a friction contact on
one payload converge at the current 30 solver iterations, nor that the active equality rows stay inside `njmax`, today
96, at 16 robots. I ran nothing.

**Cost and measurement.** 18 hours, 180 GPU minutes. Latch, hold 1500 physics steps, and assert the anchor drift stays
under 1 mm with `torch.isfinite` true on `qpos` and `qvel`. Then re run the three friction facts of
`tests/test_env_mjlab.py`, then `scripts/run_controls.py`. The step holds at or above 83.6 percent scripted success
with under 20 percent throughput loss.

### 6. Throughput and memory at 512 envs

**The change and the risk it retires.** Raise `MAX_TRAIN_ENVS` in `swarm/compute.py` and measure. Throughput is 2,900,
6,100, and 11,850 env steps per second at 64, 128, and 256 envs. That is close to linear, so the GPU is latency bound
and not compute bound, and 512 envs should reach about 22,000. Process memory is 718 MiB and flat from 4 to 256 envs,
so the Warp context and the compiled kernel modules dominate and the per world `Data` does not. 512 envs should stay
near 1 GiB. The replay buffer in `swarm/buffer.py` grows with the env count, and `swarm/compute.py` exits above 8 GB.
Warning: `SimulationCfg.nconmax` is a per world allocation, so items 3 and 4 multiply contact memory by the env count.
Measure 512 envs after items 3 and 4. The step retires the first limitation of `docs/results.md` section 12: the mjlab
result is a single seed at 24,000 steps. Throughput buys seeds.

**Cost and measurement.** 5 hours, 120 GPU minutes. Report env steps per second and peak process memory at 256 and 512
envs, with the Warp allocation counted, because the PyTorch fraction cap does not bound it. Then run three seeds of
the belief run and report the standard error of the best evaluation. The step pays off if three seeds cost under three
times 66 minutes.

### 7. Domain randomization of the mass and the friction

**The change and the risk it retires.** Call `Simulation.expand_model_fields(("body_mass", "geom_friction"))` once in
`_build_sim`, then write the per world slices in `_write_state` on every reset, and call
`Simulation.recompute_constants(RecomputeLevel.set_const)` after a `body_mass` write. Do not adopt the manager based
event stack: the env builds `Simulation` directly, and two lines are simpler. Warning: `recompute_constants` runs
outside the captured graph, and episodes end on different steps in different envs, so a mass write happens on most
control steps. Measure that cost first. If it is large, randomize the friction only, which needs no recompute. The
step retires this risk: every mjlab number comes from one mass and one friction, so the study cannot say whether the
policy learned the task or the constant.

**Does the learner survive it?** The critic target clamp in `swarm/rlpd.py` line 133 bounds the TD target to `[0, 1]`.
The reward stays 0 or 1 and gamma stays below 1, so the clamp survives randomization unchanged. The friction facts do
not. The threshold is `mu * m * g`, today 39.2 N, and three pushers apply 30 N. If mu and m each vary by 15 percent,
the low corner gives 28.3 N, three pushers alone move the payload, the `single_pusher` control stops reading 0, and
the task stops requiring cooperation. At 10 percent each the low corner gives 31.8 N, a margin of 1.8 N, so 10 percent
is near the ceiling at the current mass. Raise the base mass for a wider range.

**Cost and measurement.** 8 hours, 300 GPU minutes. First assert the friction facts at all four corners of the sampled
range, in `tests/test_env_mjlab.py`. Then train one seed with randomization on and evaluate it on the nominal model
and on four held out corners. Report the spread and the throughput with and without the recompute.

### 8. A real robot morphology for the pusher

**The change and the risk it retires.** Replace the cylinder pusher with `unitree_go1` from
`mjlab/src/mjlab/asset_zoo/robots/unitree_go1`. The Go1 spec holds 13 joints, one free joint and twelve hinges, so 18
degrees of freedom per robot against 2 today. Six robots plus the payload take the world from 24 to 114 degrees of
freedom, and each robot adds four foot contacts on a floor the robots do not touch today, so `nconmax` must rise well
above 24. The pusher today has no mass on the floor, no yaw, and no way to fall over. Whether the search result
survives a robot that can fail is the question the proposal ultimately asks, and no item above answers it.

**What changes in the action space.** Everything. The action today is `(vx, vy, u)`, which the port maps to two slide
velocity servos. A Go1 takes twelve joint position targets, so the task needs a locomotion policy under the transport
policy, and the transport action becomes a velocity command into it. That is a second learning problem, and it is why
this item ranks last. `ACT_DIM` stays at 3 only if the locomotion policy is frozen and pre trained.

**Cost and measurement.** 40 hours or more, 1,200 GPU minutes or more, excluding the locomotion policy. Do not start
until items 1 to 5 are done. The gate is then the scripted controller: it must reach 60 percent with the Go1 pusher
before any learning run starts. If it cannot, report and stop.

### What to do first

Do item 1, stage A. It is one line, it costs an hour, and it decides whether the headline transfer result of
`docs/results.md` section 13.7 measures crowding or measures a payload that leaves the floor at four latched grippers.
Every other item is easier to interpret once that is answered.
