# Reviewer 1, round 2: methods, correctness, and statistical support

## Summary

The authors answered the central objection of round one. The fair H1, search against the sampled
policy, now runs at 256 environments and 5 batches on three mjlab training seeds, with per episode
outcomes saved and a paired bootstrap over 1,280 episodes per seed (`docs/results.md` section 16,
`runs/mjlab*/results/day1_controls_f1.json`, `scripts/seeds_summary.py`). I recomputed all fifteen
intervals from the raw per episode lists, and every one reproduces to the printed digit. The claim
that depth 6 search beats the sampled policy at lags 0 and 1 in three of three seeds is real. The
authors also built the two controls I asked for, and both cut against their own earlier story: a 2D
agent retrained on 60.7 percent demonstrations raises the critic's action span to 1.07 times its
noise and search still loses 18 points (section 14.6), and a momentum variant of the 2D task gives a
span ratio of 1.09 and search still loses 13 to 18 points (section 18). Conclusion 1 now says the
mechanism is not settled, which is the right verdict and is supported.

The document did not keep up with the data. `docs/results.md` changed while I reviewed it: when I
started, section 16 printed a table from the paired rerun and then described it in prose with the
numbers of the superseded run, the summary table and conclusions 1, 2 and 5 carried the superseded
numbers, and one training figure was wrong by ten points. Most of that is now repaired, and I record
below only what is still live. The residue is real: conclusion 1 and `math.md` 10.7 still quote the
superseded run, section 13.9 still asserts the mechanism that section 14.6 refutes, and five prose
ranges in sections 14 to 19 do not match their files. Four round one weaknesses received no work at
all. I raise my score, and I would raise it further for a document that says one thing per number.

## Strengths

- **The fair H1 resolves, and I can reproduce it.** `scripts/summarize.py` on both result trees and
  `scripts/seeds_summary.py` regenerate `docs/results_tables.md`, `docs/results_tables_mjlab.md` and
  `docs/results_tables_seeds.md` byte for byte. Recomputing the paired bootstrap from
  `per_batch_success` gives seed 0 +13.5 [+10.1, +17.0], seed 1 +30.0 [+26.7, +33.3], seed 2 +15.8
  [+12.0, +19.4] at lag 0, and +4.5 [+1.1, +8.0], +22.4 [+19.0, +25.9], +15.2 [+11.8, +18.7] at
  lag 1. Those are the printed values.
- **The pairing I asked for exists now.** `swarm/evaluate.py:62` returns `per_env_success` per batch
  and seeds the torch RNG per batch, and `scripts/seeds_summary.py:38` bootstraps the paired
  differences. `make_env(seed + i)` seeds the environment independently of the torch global RNG, so
  every arm scores the same 1,280 initial conditions.
- **The mechanism is now measured, and the authors report a result against themselves.**
  `results/critic_action_span.json` holds the span, the ensemble spread, the ratio, the argmax
  selection rate and the candidate distances, with the protocol in the file. Section 14.6 states
  plainly that the span difference was mostly a property of the demonstrations, then shows the span
  is not what decides the outcome.
- **Section 19 is a real negative control.** The 2D best search cell minus no search reproduces
  exactly from the six `h2.json` files. Two of the three 2D seeds store per episode outcomes, so I
  computed the paired intervals the document omits: seed 1 lag 0 +0.8 [-4.9, +6.8], seed 2 lag 0
  -0.8 [-6.8, +5.2]. Every 2D cell contains zero and every mjlab cell excludes it by more than 35
  points. The 2D verdict is "search does not help", and the data supports that reading.
- **The audit closes the cheap failure modes.** `results/audit.json` and
  `runs/mjlab/results/audit.json` match section 15 exactly, including the 0.097 m and 0.34 m
  displacement ceilings, and the launched payload observation is honest.
- **`docs/math.md` section 13 is a real improvement.** Equation (44) separates a staleness term that
  enters once at the root from a model term that grows with depth, and equation (50) then declines
  to predict a stable argmax over depth, which is what the data does. The three point fit of section
  10.7 reproduces exactly: slope -3.27, intercept 18.8, residuals +0.5, -0.7, +0.2.

## Weaknesses

- **Conclusion 1 and `math.md` 10.7 still quote the superseded run.** `docs/results.md:827` gives
  "+19.3 ± 6.3 and +14.9 ± 7.4" where section 16, the summary table and conclusion 5 now give
  +19.8 ± 8.9 and +14.0 ± 9.0. `docs/math.md:831` fits its slope to "+19.3, +14.9, and +5.9". Those
  three values are `day1_controls.json` on all three seeds; the paired rerun in
  `day1_controls_f1.json` gives +19.8, +14.0, +5.5. The bound's headline fit therefore rests on the
  run the results document has stopped reporting.
- **Section 13.9 still asserts the mechanism that section 14.6 refutes.** `docs/results.md:557-559`
  reads "The 2D task's quasi static dynamics made the critic action flat; contact physics did not."
  Section 14.6 shows the flat span came from the demonstrations, and section 18 shows momentum does
  not change the outcome. The same bullet list (`:564`) still says the H1 gain "reaches significance
  only at depth 4 to 6", which I asked to be withdrawn and which section 14.1 supersedes.
- **The snapshot selection bias is now visible and still unaddressed.** Section 16 prints "48.4,
  30.9, 30.1" for the three best training evaluations. `scripts/train.py:91` selects each snapshot
  by an argmax over 48 evaluations of 256 episodes, the three final evaluations are 27.7, 25.8 and
  29.7, and the search gain is largest on the two snapshots the selection could not inflate. The
  document draws no inference from that.
- **The 2 point reproducibility floor is stated once and used nowhere.** `docs/results.md:551`
  measures it on mjlab and says the printed standard errors are of the same size. Sections 14, 16,
  17, 18 and 19 never restate it, and several claims sit inside it: "about 4 on 2D" for the random
  model (`:625`), "5 to 8 points" for the forward correction (`:646`), the depth 6 minus depth 0
  values of -0.3 to -2.2 in section 16 (`:739`), "worth about 4 points" in section 18 (`:789`), and
  every 2D cell of section 19, which spans -1.6 to +4.2. There is no 2D measurement of the floor.
  The paired bootstrap does not fix this: it resamples episodes from one execution, so a seed 0
  lag 1 interval of [+1.1, +8.0] is narrower than the run to run floor is wide.
- **The five point staleness curve mixes two measurement epochs.** `scripts/seeds_summary.py:26`
  loads `day1_controls_f1.json` when it exists and then always appends `lag_curve.json`. The f1
  files were written on 09-13 at 15:38 to 15:59, `lag_curve.json` on 09-12 at 23:58 to 09-13 00:11.
  So lags 0, 1 and 4 of the section 16 table come from one execution and lags 2 and 8 from another,
  and section 16 reads a slope off that curve. The slope is also misquoted: "about -2.4 points per
  frame" against a least squares value of -2.32 on either set of five points.
- **Section 16 contradicts itself on the random world model control.** `docs/results.md:736-739`
  reports the control per seed, 11 to 13 points on seed 0, 6.5 on seed 1, 5 and 0 on seed 2, from
  `runs/mjlab_seed*/results/wm_control.json`. Three lines later (`:745`) it says "The random world
  model control of section 14.2 ran on seed 0 only". The per seed numbers are also weaker than
  stated: seed 1 depth 2 is 79.9 ± 3.4 against 73.4 ± 0.5, a gap of 6.5 against 2 SE of 6.9, so "the
  rollout helps in every seed at depth 2" does not clear the authors' own rule on seed 1.
- **The joint search is still scored by one robot's individual critic.** `swarm/search.py:130` and
  `:141` read `_q` at the leader's own slot only, for every mode, and `_candidates` gives the joint
  modes 49 root candidates against 9, with the same beam of 4. Section 14.5 removes the fallback
  confound, which was the right control, but H4 still compares two unmatched searches under one
  robot's individual objective. Conclusion 3 should say what was tested.
- **The audit never runs the arm that carries the result.** `scripts/audit.py:47` calls
  `runner.step("mean")`, so zero action, random action and the trained mean policy are audited and
  the search is not. Every positive claim in the paper is a search claim.
- **Four round one weaknesses received no work.** The robot speed confound stands:
  `scripts/common.py:47` returns `EnvConfig(v_max=2.5)` for mjlab against 1.5 for 2D, no run exists
  at the other speed, and conclusion 1 (`:833`) names the contact model and the servo dynamics
  without naming the 1.67 times action scale. The nine learner changes of section 2 remain
  unfalsifiable: `swarm/rlpd.py:21-65` still hard codes `backup_entropy`, `bc_weight`,
  `dec_b_weight`, `target_policy`, `target_reduce`, `target_min` and `target_max`. `uv.lock` and
  `pylock.toml` still contain zero matches for mjlab, mujoco or warp. And the code still postdates
  the data: `swarm/env.py` is 09-12 23:47 and `swarm/env_mjlab.py` is 09-13 00:00, later than every
  result file except the f1 reruns and the lag curve, including both audits.
  `docs/mjlab_fidelity.md` argues the new options default to the old physics and the argument reads
  correctly, but nothing was rerun to confirm it.
- **The methodology now carries two decision rules for H1.** A new section 5b documents the day
  scripts and states the fair rule: depth 6 minus the sampled policy, paired over episodes, a
  bootstrap interval excluding zero, sign agreeing in all three seeds. That is the right rule and it
  matches what section 16 does. Section 7, the table the document calls the decision rules, still
  decides H1 on depth -1 against depth 2 at 2 standard errors, and section 8 still says "One
  training seed". `docs/math.md:559` still fixes evaluation at "256 environments" while every sweep
  uses 128, and still says "all runs use seed 0". `scripts/summarize.py` still prints "confirmed"
  for 2D H3 and H4 where the text corrects it.
- **Smaller mismatches between sections 14 to 19 and their files.** 14.1 prints +7.1 and 2 SE 4.3 at
  lag 1 where the file gives +7.19 and 4.23, because that table was recomputed from rounded cells.
  14.2 says a random model costs "11 to 13 points on mjlab"; the file gives 10.4 and 12.8. 14.4 says
  the forward correction is worth "5 to 8 points at every lag from 1 up"; the file gives 4.7, 5.5
  and 8.1. 16 says the argmax "reaches 70 to 80 percent"; the cells run 67.6 to 80.2. 17 says the
  plain seeds' policy runs "47 to 66"; the f1 files give 50.0 to 66.0. 18 uses a best of 48 training
  evaluation, 38.7, as the mean policy row of a 128 environment three batch table, when
  `runs/2d_momentum/results/day1_controls.json` holds the protocol matched 36.2 ± 2.5; against that
  the loss is 11 to 16 points, not "13 to 18". 19 says the mjlab seed 1 and 2 mean policies are
  "weak at 30 to 36 percent"; those no search cells run 19.5 to 30.2.
- **Round one items open in sections 1 to 13.** The 2D sampled policy appears as 60 (`:131`), 61.7
  (14.3) and 62.2 (section 9). The 70 against 47 comparison at `:86` still mixes `train_success`
  with the first episode protocol. Section 13.6 still calls the leader disagreement "again about
  half" against 0.34 and 0.40. `docs/mjlab_port.md:51,54` still gives 11,850 and 14,400 env steps
  per second at 256 envs against 9,064 in the JSON. The section order runs 1 to 10, 13, 14 to 19,
  11, 12.

## Questions for the authors

1. Section 16 and conclusion 5 now use the paired rerun. Please carry the same set into conclusion 1
   and into `math.md` 10.7, and rerun `lag_curve.py` in the rerun's epoch so the five point curve is
   one experiment rather than two.
2. Does the bound's conclusion change under the rerun? The three point fit gives slope -3.27 and
   intercept 18.8 on +19.3, +14.9, +5.9, and the text reads a linear staleness term off it.
3. Does section 13.9 still state the authors' position after sections 14.6 and 18? If not, please
   rewrite it and drop the "reaches significance only at depth 4 to 6" bullet.
4. The reproducibility floor is measured on mjlab only. What is it on 2D, and which claims in
   sections 14, 18 and 19 survive it? Section 19's 2D row spans 6 points and the floor is 2.
5. Please run the audit on the depth 6 search arm. The displacement ceiling and the early success
   check mean little on an arm nothing claims.
6. Conclusion 1 names the contact model and the servo dynamics. Why not the robot speed?
   `scripts/common.py:47` sets 2.5 against 1.5, and `mjlab_port.md:2` says the change is worth 23
   points to the scripted controller. One 2D run at 2.5 settles it. Relatedly,
   `docs/mjlab_fidelity.md` builds `pusher_shape` and wires it into no script; what blocks running
   the fair H1 arms with `pusher_shape="box"`?
7. The 2D seeds 1 and 2 sweeps store per episode outcomes. Why does section 19 report no interval?
   I get seed 1 lag 0 +0.8 [-4.9, +6.8] and seed 2 lag 0 -0.8 [-6.8, +5.2], which strengthen the
   negative.
8. `scripts/day1_controls.py:55` builds the random world model outside the seeded region of
   `evaluate`, so its weights depend on the RNG position the previous cell left. Was the control
   repeated with a fixed model seed?

## Limitations

Section 12 now states the seed structure correctly: sections 3 to 10 and 13 are seed 0, sections 16
and 19 are three seeds for the fair H1 arms and the H2 gain, and the remaining sweeps are one seed.
That is an honest map. Three limitations are still missing. The document does not say that the
reported intervals exclude the run to run floor it measures in 13.8, and that the floor is measured
on one simulator. It does not say that the snapshot for every sweep is an argmax over 48 noisy
evaluations. And it does not list the robot speed as a candidate for the cross simulator split.

## Ethics

No concerns. Simulation only, no human or animal data, no released model or dataset with misuse
potential.

## Scores

- **Soundness: 3** (good). The headline claim now has three seeds, 1,280 paired episodes per cell,
  and intervals I reproduced exactly. Two rival mechanisms were tested and eliminated with purpose
  built runs. The remaining costs are the unaudited search arm, the uncontrolled robot speed, the
  unfalsifiable learner recipe, and intervals that exclude a floor the paper itself measures.
- **Presentation: 2** (fair). This fell from round one. Section 13.9 asserts what section 14.6
  refutes, conclusion 1 and the bound quote a superseded run, section 16 contradicts itself on which
  seeds carry a control, five prose ranges do not match their files, and the section order now runs
  1 to 10, 13, 14 to 19, 11, 12.
- **Contribution: 3** (good). A search that helps in contact physics and not in a quasi static
  abstraction, replicated across three seeds against the correct baseline, with two candidate
  mechanisms eliminated by direct experiment, is worth reading. The eliminations are the part I value
  most, because they are negative results the authors paid to obtain.
- **Overall: 5** (borderline accept: technically solid, reasons to accept slightly outweigh reasons
  to reject).
- **Confidence: 4** (confident but not absolutely certain). I recomputed every interval, gap, slope
  and fit in sections 14 to 19 from the JSON files on CPU, regenerated all three table documents, and
  read the search, evaluation, audit and environment code. I ran no GPU code.

## What changed my score, and what would change it again

This review reads the repository as of 09-13 16:00. `docs/results.md` and `docs/methodology.md`
changed while I worked, in the direction I am asking for, so treat the bookkeeping weaknesses as a
snapshot. Round one gave 4 because the headline positive did not separate from the sampled policy at
any cell,
the mechanism was asserted rather than measured, and the intervals excluded a 3 to 4 point run to run
variance that the repository exposed. All three are answered. The fair H1 now runs at 256
environments and 5 batches on three seeds with per episode pairing, and I reproduced every interval.
The mechanism was measured, found to be a property of the demonstrations rather than the physics, and
retracted in conclusion 1. The per batch seeding cut the repeated cell spread from 3.9 to 2.6 points
and the remainder is reported. Two controls I asked for exist, and both returned answers the authors
did not want. Soundness and contribution each rise a point and the overall score rises to 5.

Two things would move me to 6 or 7, and neither needs much GPU time. First, finish making the
document say one thing per number: carry the rerun into conclusion 1 and `math.md` 10.7, rerun the
lag curve in that execution, rewrite section 13.9, and put the 2 point floor next to every claim
smaller than 5 points. That is editing, not computing, and it is the largest single
gain available. Second, run the two cheap controls that remain: the 2D task at `v_max = 2.5`, which
removes the last confound in the cross simulator claim, and the fair H1 arms with
`pusher_shape="box"`, which tests the contact model hypothesis with an instrument the authors have
already built and verified. Adding the search arm to the audit and a fixed seed to the random world
model costs an hour. Beyond that, I want the nine learner choices behind flags, with one ablation
each, before conclusion 6 calls the recipe a result.
