# Reviewer 1: methods, correctness, and statistical support

## Summary

The paper asks whether test time world model search (the QWM recipe) helps a decentralized
heterogeneous team, and how staleness of teammate information changes the useful search depth and the
tree search discount. The authors build a batched 2D transport task and a mjlab (MuJoCo Warp) port of
the same task, train one RLPD style agent per simulator, and run four hypotheses (H1 search helps,
H2 best depth falls with lag, H3 best discount falls with lag, H4 an elected leader beats independent
search) plus robustness, transfer, and ablation grids. The headline is a disagreement: on 2D depth 2
search loses 9.1 points against the mean action (52.3 to 43.2), on mjlab it gains 24.5 points (45.1
to 69.5). The authors attribute the split to whether the critic can rank actions, and report the 2D
critic as flat (values span 0.003). The engineering is careful, the tables regenerate exactly from
the JSON files, and the authors state several limitations themselves. I cannot recommend acceptance
in this form. The mjlab positive does not separate from the sampled policy baseline at any cell of
the grid, the two governing documents define H1 differently and disagree on the 2D verdict, the
claimed 2D mechanism is never measured, and the reported standard errors exclude a 3 to 4 point run
to run variance that the repository itself exposes.

## Strengths

- The pipeline reproduces its own tables. `scripts/summarize.py --results results` and
  `--results runs/mjlab/results` regenerate `docs/results_tables.md` and
  `docs/results_tables_mjlab.md` byte for byte, and every table line of both appears verbatim in
  `docs/results.md`. I found no arithmetic error in any table.
- The verdict arithmetic is correct where stated. H1 2D (`results.md:119`) gives -9.1 with 2 SE 4.6;
  the JSON gives -9.11 and 4.61. H1 mjlab (`results.md:388`) gives +24.5 with 2 SE 5.3; the JSON
  gives +24.47 and 5.32.
- The compute model of `math.md` equation (31) matches measurement. With N=8 and J=4 it predicts 9,
  45, 117, 189 rolls at depths 1, 2, 4, 6; the measured 2D increments over depth 0 at lag 1 are 6,
  42, 117, 192 ms (`results.md:156`). Leader and independent both cost 68 ms (`results.md:218-223`),
  so the H4 compute match is real.
- The search score at `swarm/search.py:110,120` implements equation (29) of `math.md:441` as written,
  and the argmax over surviving paths mapped back to the root (line 137) matches the pseudocode.
  `swarm/nets.py:217-220,237-240` keeps Polyak copies of the encoders and the fusion,
  `swarm/rlpd.py:97` uses them for the bootstrap belief, and the SARSA target at `swarm/rlpd.py:125`
  matches `design.md:214-220`. `tests/test_env.py` and `tests/test_env_mjlab.py` check the three
  friction facts in both physics, the occlusion mask, and the reset invariants.
- Conclusion 5 states that the sampled policy is the baseline a fair claim must beat. That is honest,
  and it is the right baseline.

## Weaknesses

- **The mjlab positive does not separate from the sampled policy at any cell.** `results.md:399-402`
  claims the margin over a random sample "reaches 2 SE only at the deeper settings (depth 6 at lag 0:
  72.7 ± 4.3 against 65.4 ± 2.2)". That cell fails the authors' own rule: the gap is 7.3 and 2 SE is
  9.8. I recomputed all twenty mjlab H2 cells against the sampled policy. Exactly one clears 2 SE
  (depth 2 at lag 0, +5.5 against 5.2), and it compares a lag 0 search with a lag 1 sampled policy,
  because the sampled policy was never measured at lag 0 or 4. Conclusion 5 (`results.md:560-563`)
  repeats the claim. The headline compares against a mean action the authors say "stalls in contact
  configurations that any perturbation breaks" (`results.md:398`).
- **The two governing documents define H1 differently and disagree on the 2D verdict.**
  `math.md:568` sets H1 to `p(D=2) - p(D=0)`, "both at L=1", and says it "isolates the world model
  and nothing else". `methodology.md:111` sets it to depth -1 against depth 2, and
  `scripts/summarize.py` implements the second. Under `math.md`'s own H1 the 2D result is +0.9 with
  2 SE 5.4, that is **not resolved**, not "refuted". The central negative claim
  (`results.md:17,119,122`) rests on one definition and is contradicted by the other.
- **The reported plus or minus excludes the search sampling variance, which is 3 to 4 points.** The
  same configuration (depth 2, lag 1, independent, default team, 128 envs, 3 batches, seeds 1000 to
  1002, same checkpoint) appears in four result files. On 2D it reads 45.6, 43.8, 42.4, 44.0, spread
  3.1, against printed standard errors of 1.4 to 4.3. On mjlab it reads 70.3, 67.2, 66.4, 69.0, and
  `transfer_unclamped_lift.json` adds 70.8, spread 4.4, against printed standard errors of 0.9 to
  2.3. `scripts/run_all.sh` shows the cause: each sweep runs in one process seeded once, so a cell's
  candidate sampling depends on the RNG position every earlier cell left. Several conclusions sit
  inside that floor: the mjlab "value of one imagined step falls from +8.3 to +5.2" (conclusion 2,
  `results.md:548`) is a 3.1 point change, the "4 to 7 points" search margin is 4 to 7 points, and
  the H4 2D gaps are 6.0 and 8.6 points.
- **The claimed 2D mechanism is asserted, not measured.** `results.md:23,127,541` state a critic span
  of 0.003, a 1.4 percent mean action selection rate, and candidate distances of 0.34 against 0.28.
  None of the three exists in any file under `results/` or `runs/mjlab/results/`; I enumerated every
  JSON key in both. `next_steps.md:276` lists "return `q_span` and `q_std` from `swarm/rlpd.py`
  `Agent.update`" as future work, and `next_steps.md:431` says the quantity holds "until r is
  instrumented". The central mechanism is unimplemented, which contradicts `results.md:3`, "Every
  number below comes from the files in `results/` and is regenerated by `python scripts/summarize.py`".
- **A simpler mechanism explains the split and is never tested.** `results/offline.json` records an
  offline success rate of 0.988 on 2D; `runs/mjlab/results/offline.json` records 0.652. The critic is
  a SARSA critic on that data, and `design.md:219` says so: "The critic therefore values the behavior
  data, which is 99 percent successful in the offline half". A critic fit to behavior that succeeds
  98.8 percent of the time has almost no outcome variance left to attribute to the action, so Q
  collapses to a function of the state. On mjlab the behavior fails a third of the time, so the
  action changes the outcome. This explains the flat 2D critic, the ranked mjlab critic, and every
  number in the paper, and it has nothing to do with "quasi static physics" (`results.md:526-528`).
  The test is cheap: retrain the 2D critic on a degraded offline buffer.
- **The cross simulator claim is confounded by the robot speed.** `scripts/common.py:env_config`
  returns `EnvConfig(v_max=2.5)` for mjlab and 1.5 for 2D. `mjlab_port.md:2` says the change matters:
  "At 1.5 m/s the scripted controller reaches 52 percent; at 2.5 m/s it reaches 75 percent". A 1.67
  times larger action scale raises the action dependence of the critic directly, which is the exact
  quantity said to distinguish the simulators. Neither simulator was run at the other's speed.
- **The consistency gate of `math.md` section 11.3 fails on mjlab and was never run there.**
  `math.md:604-607` requires the beta 0 column to reproduce the D=0 column within 2 SE, and says "If
  it does not, the search code has a defect and neither grid is interpretable". At lag 1 on mjlab the
  cells read 58.3 ± 1.4 and 62.2 ± 1.1, gap 3.9 against a threshold of 3.7. The gate fails.
  `results.md:204-206` reports it only for 2D. The failing simulator carries the positive result.
  Note also that `swarm/search.py:104` sets `D = 0` whenever `beta == 0`, so the two columns are the
  same code path and the check can only measure run to run noise, never correctness.
- **The H4 negative is an artifact of the scoring function.** In leader and round robin mode the
  search samples a joint action over all K slots (`swarm/search.py:57-58`), but the root score reads
  the critic at the leader's own belief with the leader's own action only (`swarm/search.py:110`),
  and every deeper term does the same (line 120). A "joint search" therefore ranks 49 joint
  candidates by one robot's individual Q, and the five teammate actions in the winner are unscored
  samples. `math.md:428-431` prescribes the same, so this is a design choice, not a coding slip, but
  it means conclusion 3 (`results.md:551`) tests a joint search with no team objective. The beam is
  also unmatched: the leader keeps 4 of 49 root candidates, the independent searcher 4 of 9.
- **The nine training changes carry no evidence and cannot be ablated from this repository.**
  `results.md:42-52` tables nine runs with a symptom, a cause, and a change, and `results.md:565`
  calls the recipe a result. None has a run in `results/`. `scripts/train.py:30-42` exposes no flag
  for `backup_entropy`, `bc_weight`, `dec_b_weight`, `target_policy`, `target_reduce`, `target_min`,
  or `target_max`; all seven are hard coded in `swarm/rlpd.py:21-65`. The causal claims are post hoc
  readings of single seed curves. The table also omits a change: `swarm/rlpd.py:63` sets
  `target_reduce = "mean"` and `design.md:202-206` explains why the minimum was dropped, but run 4
  of the table reports only the clamp.
- **"2 SE" is not a 95 percent rule with three batches.** `swarm/evaluate.py:70-79` takes the sample
  standard deviation over 3 batch means, so every interval has 2 degrees of freedom. Two standard
  errors then cover about 82 percent, not 95, and the small sample bias of `s` lowers the estimate a
  further 11 percent. I measured that bias: the mean printed standard error over the 87 cells of each
  run is 0.0189 and 0.0191, against a mean binomial standard error of 0.0235 and 0.0236. The
  consequence is concrete. H4 on 2D is **confirmed** (`results.md:225,228`) on a lag 4 gap of 8.6
  against 2 SE 7.5; a t test with 2 degrees of freedom gives p = 0.15. The verdict does not survive
  its own data.
- **The H2 and H3 rules fire about one time in ten under a pure noise null.** The measure is an
  argmax over 5 depths or 7 betas at 4 lags, with no noise test in `scripts/summarize.py`. Simulating
  a uniform argmax null, the H2 rule fires with probability 0.104 and the H3 rule with 0.084.
  `math.md:588` adds a third condition to fix exactly this, "at the smallest L the peak beats D=0 by
  more than 2 se", and calls it the guard "so that D*(L) marks a real peak and not the argmax of
  noise". `methodology.md:112` drops it and `scripts/summarize.py` does not implement it. It happens
  to pass on both runs, but the shipped rule is the weaker one. The reported 2D sequence is also
  partly arbitrary: at lag 1 depths 2 and 4 tie exactly at 0.4557291667, and the strict `>` in
  `scripts/summarize.py` breaks the tie toward the lower depth, which is what prints "4, 2, 0, 0".
- **The pairing the authors claim is never used.** `methodology.md:121` and `results.md:573` say cell
  to cell comparisons are "paired on the same snapshot and the same first episodes" and call that
  "the reliable part". The env seeds are shared, but `swarm/evaluate.py` saves only batch means, so
  no paired statistic exists, and the difference standard error assumes independence. A McNemar test
  on the 384 paired episodes would be far more powerful than the 2 degree of freedom test used, and
  the per env outcomes it needs were discarded.
- **The mjlab result files mix two versions of the physics, and the "unaffected" claim is untested.**
  `swarm/env_mjlab.py` has an mtime of 15:57. Every mjlab result file except `transfer.json` (16:01)
  predates it: `h1_*` 15:06, `h2` 15:17, `h3` 15:26, `h4` 15:28, `robust` 15:31, `ablations` 15:38,
  `belief_best.pt` 14:28. `results.md:519` argues "The default team has two grippers, so the snapshot
  and every other cell were unaffected". The arithmetic supports it (`swarm/env_mjlab.py:262` clamps
  at 0.8 x 78.5 = 62.8 N and two grippers lift 50 N), but nothing was rerun to confirm it, and the
  two runs of the default cell disagree by 4.4 points (66.4 against 70.8). The same pattern holds on
  2D: `swarm/belief.py` and `swarm/rollout.py` are 12:15 and `swarm/search.py` is 12:30, while `h1_*`
  is 11:47, `h2` 12:04, `h3` 12:17, `h4` 12:20, `robust` 12:23. `scripts/run_all.sh` skips any step
  whose output exists, so a partial rerun keeps the stale files.
- **The lock files do not cover mjlab.** `methodology.md:31` lists `uv.lock` and `pylock.toml` as the
  software record. Neither contains `mjlab`, `mujoco`, or `warp`. The installed environment runs
  mjlab 1.6.0, mujoco 3.11.0, warp-lang 1.17.0, and `pyproject.toml` pins a bare git commit with no
  lock over its dependency tree. The half of the study carrying the positive result cannot be rebuilt
  from the locks.
- **Snapshot selection is an unaccounted optimization on the evaluation metric.**
  `scripts/train.py:91` keeps the argmax over every evaluation. On mjlab that is 48 evaluations of
  256 envs, each with a binomial standard error near 3.1 points. The selected best of 48.4 percent
  sits at step 20,500 while the final checkpoint is 27.7, so the run is not converged. A best of 48
  inflates by roughly 2 standard errors, close to the gap between the reported best (48.4) and the
  protocol number (45.1).
- **Documentation contradicts the code on two training settings.** `methodology.md:77` states the
  target is "the minimum of 2 random heads". `swarm/rlpd.py:63` sets `target_reduce = "mean"` and
  `design.md:202` says the mean. The methodology file is the one the paper calls the replication
  recipe. Separately `design.md:228` says the model "is pretrained on the offline buffer", while
  `methodology.md:85-88` says "There is no separate pretraining phase" and `swarm/rlpd.py:164-174`
  trains it from the first update. `design.md:225` names `swarm/world_model.py`, which does not exist.
- **Script names in the specification do not exist.** `math.md:571,586,600,618` cite
  `scripts/sweep.py --grid depth x lag`, `--grid beta x lag`, `--leader`, and `results/h1.json`. The
  real flags are `--which h2/h3/h4` and the real outputs are `results/h1_depth{-1,0,2}.json`;
  `design.md:320` repeats `results/h1.json`. `math.md:558` fixes the protocol at "256 environments",
  while every sweep cell uses 128 (`scripts/sweep.py:57`). `scripts/run_all.sh` also produces neither
  the `full36k` 2D baseline nor the `belief_12k` mjlab run that `results.md:77,354` report.
- **Smaller items.** `results.md:131` and `:86` quote a random sample at 60 percent; the ablation
  table says 62.2 (`results.md:279`). `results.md:86` compares 70 against 47, but the 70 is
  `train_success` from `scripts/train.py:81`, which counts every episode of the collection rollout,
  while the 47 is the first episode mean action protocol. `results.md:468` calls the mjlab leader
  disagreement "again about half"; the table says 0.34 and 0.40. `mjlab_port.md:3` gives mjlab
  throughput at 256 envs as 11,850 and 14,400, while the JSON and `results.md:346` say 9,064.
  `results.md:276-278` reports the masked message ablation three times at 13.5 ± 2.6 with length
  100.9; that is one measurement, because a masked table makes the lag inert. Section 13 sits between
  sections 10 and 11.

## Questions for the authors

1. State the margin of depth 2 search over the sampled policy at lag 1 on mjlab with its 2 SE. I get
   +4.2 with 2 SE 5.8. Which cell clears 2 SE against the sampled policy at the same lag? If none
   does, what survives of conclusion 1?
2. `math.md:568` defines H1 as `p(D=2) - p(D=0)`; `methodology.md:111` defines it as depth 2 against
   depth -1. Which is the registered hypothesis? Under the first the 2D result is +0.9 with 2 SE 5.4,
   which is not resolved. Does the headline change?
3. Produce the script that computes the 0.003 critic span, the 1.4 percent selection rate, and the
   0.34 against 0.28 distances, and the file that holds them. If they came from an interactive
   session, say so and state the sample size.
4. The offline data succeeds 98.8 percent on 2D and 65.2 percent on mjlab. Is the flat 2D critic a
   property of the physics or of a SARSA target on near perfect behavior? What is the 2D span when
   the same agent trains on a degraded offline buffer?
5. Report the 2D task at `v_max = 2.5`, or mjlab at 1.5. Without one, how do you separate contact
   physics from a 1.67 times larger action scale?
6. The section 11.3 gate fails on mjlab at lag 1 (gap 3.9, 2 SE 3.7). `math.md:606` says that means
   "neither grid is interpretable". How do you read the mjlab H2 and H3 grids given that?
7. In leader and round robin mode the root score is `Q(b_leader, a_leader)` only. Why is a joint
   action ranked by one robot's individual Q, and what does H4 measure if the teammate slots of the
   winning candidate are unscored samples?
8. The same cell read from four files differs by 3.1 points on 2D and 4.4 on mjlab. Should the
   reported plus or minus include the search sampling variance, and which conclusions survive if so?
9. `MAX_EVAL_ENVS = 256` (`swarm/compute.py:21`) caps a batch near a 3.1 point binomial standard
   error while the disputed effects are 4 to 7 points. Why not raise the env or batch count for the
   four cells that carry the claims?
10. Which of the nine training changes were tested by turning one off and rerunning? If none, please
    relabel the table as a development log rather than a result.
11. Why do `uv.lock` and `pylock.toml` contain no mjlab, mujoco, or warp entry?
12. `swarm/env_mjlab.py` postdates every mjlab result file except `transfer.json`. Which files came
    from which version of the physics, and was anything besides the transfer grid rerun?

## Limitations

The authors state limitations in `results.md:571-586` and they are unusually candid: one training
seed, a single mjlab seed at a 45 percent baseline, the SARSA critic valuing behavior data, the
cloning term and belief decoder as deviations from RLPD, and the eased task. Conclusion 5 states the
sampled policy problem outright. Three limitations are missing, and they are the ones that matter.
The paper does not say that the reported standard errors exclude the search's own sampling variance,
which its repeated cells put at 3 to 4 points. It does not say that the 2D mechanism is unmeasured.
It does not say that the velocity change confounds the cross simulator claim. The section also does
not withdraw the claims the body makes and the data does not support, in particular the "reaches
2 SE at depth 4 to 6" claim of section 13.3.

## Ethics

No ethics concerns. The work is simulation only, uses no human or animal data, and releases no model
or dataset with misuse potential.

## Scores

- **Soundness: 2** (fair). The code matches the mathematics on the parts I spot checked and the
  tables regenerate exactly. The statistical support does not hold: the headline positive does not
  separate from the correct baseline, one of the authors' own interpretability gates fails on the
  simulator that carries it, the intervals are 2 degree of freedom intervals used as 95 percent
  intervals, and the stated mechanism is unmeasured.
- **Presentation: 3** (good). The writing is clear and direct, the tables are well organised, and the
  provenance of each number is stated. Points off for section 13 sitting between sections 10 and 11,
  for three documents that disagree on H1 and on the critic target, and for prose numbers that no
  file contains.
- **Contribution: 2** (fair). The 2D negative and the staleness directions are worth reporting, and
  the two simulator contrast is a good idea. The contribution as claimed, that the QWM mechanism
  transfers when the critic can rank actions, is not established by this evidence.
- **Overall: 4** (borderline reject: technically solid paper where reasons to reject, for example
  limited evaluation, outweigh reasons to accept).
- **Confidence: 4** (confident but not absolutely certain). I read the documents and the code and
  recomputed every table and verdict from the JSON files on CPU. I ran no GPU code, so the run to run
  variance I report comes from repeated cells already in the repository, not from a controlled repeat.

## What would change my score

Four changes would move me to accept. First, rerun the cells that carry the claims at 2,000 episodes
or more with 10 batches, so the intervals have real degrees of freedom, and report every mjlab search
cell against the sampled policy at the same lag. If depth 4 to 6 beats the sampled policy by 5 points
with a genuine 95 percent interval, conclusion 1 stands and the paper becomes interesting. Second,
instrument the critic span. Return `q_span` and `q_std` from the update and from the search, plot
them for both simulators against the training step, and show that the 2D span is small where the
mjlab span is not. That turns the mechanism from an assertion into the paper's best figure. Third,
separate the rival mechanisms: run the 2D task at `v_max = 2.5`, and run the 2D learner on a degraded
offline buffer. If the 2D critic stays flat under both, the physics story survives; if it does not,
the paper has a simpler finding about SARSA critics on near expert data. Fourth, either give the
leader search a team objective, or retitle H4 as a test of broadcasting one robot's individually
optimal joint sample. Smaller fixes I expect regardless: pick one definition of H1, run the section
11.3 gate on mjlab and report the failure, correct the "reaches 2 SE at depth 4 to 6" sentence, add
mjlab to the lock files, add flags for the seven hard coded learner choices, and save per env
outcomes so a paired test is possible.
