# Meta review

Three reviewers read the repository as a NeurIPS submission, each told to refute. Their reviews
are in this directory. This meta review records the scores, the points all three agree on, what
the authors changed in response (commit `a199287` and the one after it), and what remains open.

## Scores

| Reviewer | Lens | Soundness | Presentation | Contribution | Overall | Confidence |
|---|---|---|---|---|---|---|
| 1 | Methods and correctness | 2 | 3 | 2 | 4, borderline reject | 4 |
| 2 | Significance and positioning | 2 | 2 | 2 | 3, reject | 4 |
| 3 | Clarity and consistency | 2 | 2 | 3 | 4, borderline reject | 4 |

Consensus: the work is a complete, honest, replicable study with a real engineering
contribution and a clear negative and positive result pair, but it is not a NeurIPS paper in its
present form. The main claims rest on one training seed per simulator, the headline margin over
the fair baseline is inside the noise, the mechanism is measured on one arm, and the writing
contradicted itself in several places.

## Points all three reviewers made

1. **The headline does not beat the sampled policy at 2 SE.** The mjlab search gains 24.5 points
   over the deterministic mean action but 4 to 7 points over the sampled policy, and once both
   standard errors are combined no cell clears 2 SE. The results document had claimed one did.
   Fixed: `results.md` 13.3 and conclusion 5 now state the margin is unresolved and cost the
   experiment that would resolve it.
2. **One training seed per simulator.** Every cross simulator claim is one snapshot against one
   snapshot. Not fixed; costed in `next_steps.md` (three seeds per simulator, 202 GPU minutes).
3. **The mechanism was asserted, not measured, on mjlab.** The critic span of 0.003 was a 2D
   number that lived in no results file. Fixed: `results/critic_action_span.json` now holds the
   span, the ensemble spread, their ratio (0.66 on 2D, 1.27 on mjlab), the argmax selection rate,
   and the candidate distances for both snapshots, and `results.md` 13.3 reports them with the
   caveat that both spans are small.
4. **Repeated cells differ by up to four points.** The same setting appears in five sweep groups
   with five numbers because the torch RNG was seeded once per script. Fixed in the code
   (`swarm/evaluate.py` seeds per batch) and reported as the empirical noise in `results.md` 13.8.
   The sweeps were not rerun; a rerun costs about 2.5 hours per simulator.
5. **No related work.** Fixed: `docs/related_work.md` places each claim against QWM, QT-Opt,
   Hamrick et al., RLPD, REDQ, TD3+BC, CQL and IQL, CommNet and TarMAC, I-POMDPs and Self Other
   Modeling, and states plainly what is new.
6. **Document contradictions.** The critic target was described as the minimum in two documents
   and the mean in the code; the mathematics carried the constants of the abandoned first task;
   the run count was twelve in one place and nine in another; the README command did not reproduce
   the mjlab snapshot; the mathematics defined H1 against depth 0 while the methodology used depth
   -1. All fixed in the documents.

## Points one reviewer made that stand

- **Reviewer 1: a rival explanation for the flat 2D critic.** The 2D demonstration buffer is 98.8
  percent successful, the mjlab buffer 65.2. A SARSA critic on near perfect data has nothing to
  separate actions by. This explains the span difference without the physics. Added to
  `results.md` 13.3 and conclusion 1 with a 35 GPU minute experiment in `next_steps.md`.
- **Reviewer 2: the depth 0 arm is QT-Opt.** Sixteen of the 24.5 headline points come from
  ranking policy samples by Q with no model, which is prior work. The rollout's 8 points are the
  QWM specific part. `related_work.md` says so.
- **Reviewer 3: the H2 "best depth" figure hides the baseline.** `figures/h2_best_depth.png` shows a
  clean staircase for a result whose every cell is below no search. Not fixed; the text says it
  and the depth against lag figure shows the baseline lines. A revised figure should carry the
  baseline.
- **Reviewer 3: the mjlab depth axis is right censored.** The best depth is 6, the grid maximum,
  at three of four lags, so the proposal's collapse claim is not supported on mjlab, only not
  refuted. Stated in `next_steps.md`; extending the grid to depth 8 and 12 is costed there.
- **Reviewer 1: the 2D H4 verdict rests on a gap with p = 0.15 on two degrees of freedom.** The
  results already call it marginal at lag 1; the summary script's rule prints "confirmed".

## What the meta reviewer would accept

The study as it stands is a strong workshop or technical report: the pipeline runs end to end on
two simulators, every number is regenerated from committed data, the failure history is a result
in itself, and the negative and positive pair is informative. For a main track submission the
minimum is: three training seeds per simulator, the H1 arm against the sampled policy at 256 envs
and 6 batches, the demonstration quality control for the critic span, the search controls (a
random world model and a decoded state scorer), and a rerun of the sweeps with the per batch
seeding so repeated cells agree. Those are the first eight items of `next_steps.md` plus the seed
replication, about nine GPU hours.

## Discrepancies that remain in the repository after the fixes

- The sweep tables were produced before the per batch seeding, so repeated cells still differ
  across groups until the sweeps are rerun.
- The summary script's decision rules print "confirmed" for H2 and H3 on 2D where every cell is
  below the baseline, and for H4 on 2D at p = 0.15; the results text corrects both, the script
  does not.
- The mjlab robot speed (2.5 against 1.5 m/s) and the eased 2D task are documented but not
  controlled.
