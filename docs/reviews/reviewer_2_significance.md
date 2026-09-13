# Reviewer 2, round 2: significance, novelty, and positioning

**Paper:** Decentralized World Model Search for Heterogeneous Cooperative Transport
**Round 1 scores:** soundness 2, presentation 2, contribution 2, overall 3, confidence 4.

## Summary

The revision answers the central objection of round 1, and it answers it in the authors' favour. The fair H1 now runs against the sampled policy and a random
candidate at 256 environments and 5 batches on three mjlab training seeds. Depth 6 beats the sampled policy by 19.8, 14.0, 9.3, 5.5 and -0.3 points at lags 0, 1,
2, 4 and 8, the paired bootstrap interval excludes zero in 3 of 3 seeds at lags 0 and 1, and the per seed gain falls monotonically over all five lags in 3 of 3
seeds (`results.md` 16, `results_tables_seeds.md`). The authors replaced the best depth argmax, which I attacked in round 1, with the gain over the baseline,
after a rerun with per batch seeding flipped the argmax from 6, 6, 6, 4 to 6, 2, 4, 1 and the verdict to "not resolved" (13.4). `math.md` section 13 adds a
Lipschitz bound with proofs whose content is that staleness enters the score error once at the root while model error enters per depth, so it predicts a stable
gain and declines to predict a stable argmax. Three explanations for the 2D negative are eliminated by experiment: demonstration quality and the critic's action
span (14.6) and payload momentum (18). `related_work.md` exists. `mjlab_fidelity.md` measures four fidelity options and reverses its own prediction on one of
them.

The cost of this honesty is that the mechanism is now unknown. Every mediator the study proposed has been tested and rejected, and conclusion 1 says so. The paper
has a well measured effect with no explanation. Three problems follow. Over lags 0 to 2, which carry most of the headline slope, the gain falls because the
baseline improves, not because the search degrades. Across the four mjlab agents the search's margin tracks how weak the policy is rather than how good the search
is. And the only positive result runs on a contact configuration the authors' own fidelity study shows produces 19 times more spurious yaw.

## Strengths

1. **The fair H1 resolves.** `results.md` 14.1 gives +12.8 against 2 SE 3.7 at lag 0 and +7.1 against 2 SE 4.3 at lag 1. I recomputed the 2 SE column and it is
   now the standard error of the difference, which round 1 found wrong twice. Section 16 adds two seeds and a paired bootstrap over 1,280 first episodes per seed.
   This is the arm I asked for, at the size I asked for.
2. **The monotone staleness curve is the strongest result in the package.** Per seed gains are 13.5, 4.5, 1.0, -0.5, -10.5; 30.0, 22.4, 18.2, 11.2, 8.4; and 15.8,
   15.2, 8.7, 5.9, 1.2 (16). Three independent five point orderings, all monotone, no argmax.
3. **The authors broke their own results and published the breaks.** The seeding rerun flipped H2 and H3 to "not resolved" (13.4, 13.5) and 13.8 publishes the 2
   point reproducibility floor.
4. **Four clean negative results.** Leader search without the fallback loses by 19 to 40 points on both simulators (14.5). The critic action span rises with
   noisier demonstrations and still fails to predict whether search helps (14.6). Momentum does not reproduce the mjlab positive (18). Collecting with search does
   not raise the ceiling (17). The last two contradict, respectively, the authors' preferred explanation and a claim in QWM, the base method.
5. **The bound earns its place by predicting which statistic is stable.** Equation (45) makes staleness a constant offset in depth and model error a slope in
   depth, so equation (50) predicts no stable argmax over depth, which 13.4 then reports.
6. **`mjlab_fidelity.md` section 4 is reusable outside this project.** A cylinder pusher gets one contact point because the multicontact expansion in
   `mujoco_warp/_src/collision_convex.py` admits BOX and MESH only, so two cylinder pushers turn the payload 1.003 rad where two box pushers turn it 0.054 rad.

## Round one weaknesses: addressed, partly, or not

- **W1, no related work. Partly.** `related_work.md` covers QWM, QT-Opt, Hamrick et al., MuZero, TD-MPC2, RLPD, REDQ, TD3+BC, CQL, IQL, SAC+AE, Dec- POMDPs,
  CommNet, TarMAC, MAPPO, QMIX, I-POMDPs and Self Other Modeling. Six of my ten gaps close, four stay open.
- **W2, the headline does not beat the one line baseline. Addressed.** `results.md` 14.1 and 16 answer my arithmetic with a larger sample and a correct standard
  error, and the answer is positive at lags 0 and 1.
- **W3, a property of two checkpoints, not two physics classes. Partly.** Three seeds per simulator exist (16, 19), so the checkpoint half is answered. The
  physics half is conceded: conclusion 1 states what separates the simulators is not settled. Momentum and demonstration quality are controlled; the robot speed,
  the training budget and the eased 2D task are not.
- **W4, H2 and H3 are a noisy argmax over a censored grid. Addressed for the statistic.** The argmax flipped on rerun, the authors said so, and the gain replaced
  it. Depth 6 is still the grid maximum, and no oracle teammate arm exists.
- **W5, the H4 refutation carries the H4 confound. Addressed.** `results.md` 14.5 runs a leader election with no fallback on both simulators. Independent search
  wins by 19 points on 2D and 40 on mjlab.
- **W6, the belief pipeline and transfer are buried and confounded. Partly.** Section 14.4 isolates the forward correction at 5 to 8 points on mjlab and shows it
  accounts for the whole 2D staleness rise. But F11, the per team size difficulty control, was dropped because "the bounded lift explained the transfer loss"
  (`next_steps.md`). The lift bug explained a transfer loss; my objection concerned the transfer gain, which conclusion 4 still claims at 96.6 percent on 16
  robots.
- **W7, nine training changes are engineering and rediscovery. Partly.** `related_work.md` names TD3+BC, CQL, IQL, SAC+AE and REDQ, and three seeds partly answer
  the single path objection. The SARSA target rival was not tested: 14.6 changed the buffer and kept `target_policy`.
- **W8, the testbed does not support claims about contact physics. Partly.** `mjlab_fidelity.md` found a bug that changed a published number, but neither call
  site passes `mj`, so every result still runs the kinematic latch, the central lift, no robot collision and cylinder pushers. Best mjlab training evaluation is
  48.4 and search reaches 73, against a scripted controller at 83.6.
- **W9, the conclusions and the next steps contradict each other. Addressed.** Conclusion 1 states the mechanism is unsettled and the summary table carries "not
  resolved" verdicts. A different version replaced it, see W3r.

Count: 4 addressed, 5 partly, 0 untouched.

## Weaknesses

### W1r. The staleness curve is read as search degradation, and over its first half it is not

`results.md` 16 and `math.md` 10.7 read the fall of the gain as the staleness term of equation (50), and 10.7 reads its slope as the rate `Lambda_1`. Equation
(50) assumes the baseline does not move with `L`. It does. From `results_tables_seeds.md` the seed mean sampled policy is 53.3, 59.4, 62.2, 60.2, 59.4 at lags 0,
1, 2, 4, 8 and the seed mean depth 6 arm is 73.0, 73.4, 71.5, 65.7, 59.1. From lag 0 to lag 2 the gain falls 10.5 points, of which the search arm supplies 1.6 and
the rising baseline supplies 8.9; from lag 2 to lag 8 the gain falls 9.6, of which the search arm supplies 12.4. So 85 percent of the first half of the headline
fall is the baseline improving, and 14.4 already says why: the forward correction gains with age. The claim in 13.7 that the mjlab baseline is flat across lags
describes the *mean* policy at 128 environments; the sampled policy rises 8 to 9 points from lag 0 to lag 2 in all three seeds. Two consequences. The largest
reported gain, +19.8 at lag 0, sits at the one lag the authors elsewhere call off distribution for a fusion trained at lag 1 (`results.md` 6), so its baseline is
depressed. And the slope of -2.4 points per frame overstates the search's own decline: the same fit on the search arm gives -1.9 and on the baseline +0.4. H2's
direction survives, because the search arm falls 22.6, 13.0 and 6.3 points across seeds. Its size does not.

### W2r. The positive result runs on the contact configuration the fidelity study calls artefactual

`mjlab_fidelity.md` section 4 measures that a cylinder pusher gets one contact point and turns the payload 19 times more than a box pusher, and gives the
mechanism: one point cannot resist a couple. The success test includes an angle error. Every arm in `results.md` 13 to 19 runs the default `pusher_shape =
"cylinder"`. So the only task on which search helps is one where yaw is easy to induce by accident and load bearing for success, through an artefact the authors
have now characterized. That is a live rival reading of conclusion 1: the mjlab critic may rank actions because single point contact makes payload yaw strongly
action sensitive, while the 2D gated normal force does not. Neither `results.md` nor `next_steps.md` names it.

### W3r. Every proposed mediator is eliminated, including the one the text still endorses

`results.md` 14.6 closes with "the quantity that does track the search's value across the three snapshots is the world model's contribution measured by the random
model control". Section 16 then reports that a random model costs 11 to 13 points on seed 0, 6.5 on seed 1, and 5 at depth 2 and 0 at depth 6 on seed 2, while the
lag 0 gains are +13.5, +30.0 and +15.8. Seed 2 has the smallest model contribution and the second largest gain, and on seeds 1 and 2 the critic argmax alone
reaches 70 to 80 percent while depth 6 adds nothing, so the last surviving mediator does not track the effect. Section 16 says the split is seed dependent, but
14.6 is not amended and 14.1 and 14.3 likewise generalize seed 0 properties to "both simulators".

### W4r. `related_work.md` claims novelty for a retracted mechanism, and omits the literature under both headlines

Internal. Novelty item 2 reads "The observation that the same search hurts on a task whose critic cannot rank the policy's own candidates, with the critic's
action span measured on both tasks". `results.md` 14.6 and conclusion 1 state the span is not the mediator. The paper's own statement of novelty asserts a
mechanism its results withdraw. External. The heading "Decentralized execution, communication, and delays" cites no work on delays, although the staleness axis is
novelty item 1 and the study's strongest result: the delayed sharing information structure of Nayyar, Mahajan and Teneketzis (2011), delay aware MARL and the
Smith predictor structure of rolling a stale estimate forward all belong there. Jiang et al. (AAMAS 2015), the canonical result that the useful horizon and
discount shrink as model error grows, is absent, and H2 and H3 are that result with teammate staleness substituted in. Grill et al. (2020) is absent and frames
the sampled policy baseline. MAZero, MAMBA and MAMBPO are absent, so novelty item 1 stands against no multi agent search baseline. "Cooperative transport" names
zero references, and conclusion 4's transfer claim is positioned against nothing although size agnostic pooled attention is standard.

### W5r. The search's margin tracks how weak the policy is, not how good the search is

`results.md` 17 leads with "Depth 6 minus sampled" at +34.1, +29.2 and +14.3 and calls the gain "roughly doubled", while its own conclusion says the ceiling did
not rise: the depth 6 arm reaches 69.2, 73.1 and 70.2 against the plain seed means of 73.0, 73.4 and 65.7, the policy fell 12 to 25 points and the run cost 82
minutes against 66. That is a negative result against the complementarity QWM reports, and it reaches neither section 1 nor the conclusions. The pattern is not
confined to that run. Across the four mjlab agents the depth 6 arm at lag 0 sits in a narrow band, 73.1, 80.2, 65.8 and 69.2, while the sampled policy spans 59.6,
50.2, 50.0 and 35.1 and the best training evaluations span 48.4, 30.9, 30.1. The search pulls every agent into the same 65 to 80 percent band, so the "gain"
measures the policy's deficit as much as the search's contribution, and the largest gains belong to the weakest agents. Section 17 also holds an unremarked
counterexample to H2: its sampled policy rises 21 points from lag 0 to lag 4 while its depth 6 arm stays flat, so in that run the search does not degrade with
staleness at all.

### W6r. The headline table mixes baselines, and one conflict survives the fixes made during review

The summary table of `results.md` 1 reports the 2D H1 as "-1.6, +0.8, -0.8 points against no search" and the mjlab H1 as "Depth 6 beats the sampled policy". The
negative result uses the weak baseline and the positive one uses the strong baseline. On 2D the sampled policy beats the mean policy by 12 points (`results.md`
9), so against the fair baseline the 2D deficit is about -13, not -1. During this review the authors unified the headline on +19.8 ± 8.9 and +14.0 ± 9.0 and fixed
the training evaluations to 48.4, 30.9, 30.1 in both documents. Conclusion 1 still carries the withdrawn pair, +19.3 ± 6.3 and +14.9 ± 7.4.

### W7r. The bound's one testable prediction is not measured

`math.md` 10.6 says the score gap `g` of equation (48) "costs nothing to record" and calls it "the one instrumentation the study should add". It was not added, so
equation (48) is unexercised. The step from equation (46), a bound in critic units, to equation (50), a linear model of the gain in success points, is asserted,
and 10.6 concedes the map from latent error to success points is not modelled, so the agreement in 10.7 between a linear fall and `d^0.67` error growth passes
through an unproved link. The linear reading also uses three of the five points; across all five the per frame drop is 4.4, 5.6, 1.7 and 1.6, which saturates and
by 10.7's own taxonomy indicates `Lambda_1 < 1`.

## What the contribution is, in one paragraph

A team of robots that must move an object together cannot see each other's information instantly. Each robot gets its teammates' reports one or more control
frames late. This project asks whether it pays, at decision time, for each robot to imagine a few steps of the future with a learned simulator of the world and of
its teammates, and to pick its action from that imagining rather than straight from its policy. The answer, measured on a MuJoCo contact simulation of six robots
pushing and carrying a box, is that it pays while the teammate information is fresh and stops paying as that information ages. Across three independently trained
agents the gain over the right baseline is about 20 points of task success when reports are current, about 14 after one frame and zero after eight, and the
decline is monotone in all three. A companion bound explains why the gain, not the best depth, is the right thing to report: message age enters the error once, at
the root of the tree, while simulator error enters once per level, so the two move separately and the best depth is not stable. What the work is not: it is not a
demonstration that this helps in general, because on a second, simpler simulator of the same task the same code gives no gain, and every explanation offered for
that difference has been tested and rejected. It is not a competitive cooperative transport system, because a hand written controller scores 83.6 percent where
the learned system with search reaches 73. And it is not a new algorithm; the search, the critic ranking, the message attention and the permutation invariant
pooling each come from published work.

## Who would use these results

The authors state the project is for learning rather than publication, so the question is reuse, not citation.

1. **An engineer weighing decision time search on a delayed multi robot system.** The one transferable rule: search is worth its compute while message age stays
   under a few control steps, and the gain decays to zero by about eight. `results.md` 13.4 prices it at 104 ms per step at depth 6 against 29 without. The decay
   rate will not transfer. The shape will.
2. **Anyone building a contact rich pushing task in MuJoCo Warp.** `mjlab_fidelity.md` is the most directly reusable document here: the single contact point
   limit, the 19 times yaw difference, the per world `eq_active` and `eq_data` verification against the source, and the unbounded lift bug that silently launched
   the payload.
3. **Someone learning how to interrogate a result.** The control suite is a template: a random world model, a random scorer, a decoded state scorer, a random
   candidate, a leader with no fallback, a noisier demonstration buffer, a momentum variant, a reward hacking audit, a reproducibility floor, and a rerun that
   flipped the authors' verdict.
4. **A MARL researcher.** Still cannot use the numbers: the task is bespoke, no MAPPO, QMIX or MAZero comparison exists, and absolute success stays below the
   scripted controller. The negative results are the part most likely to save someone time.

## Questions for the authors

1. What does the depth 6 arm's own success do across lags, beside the gain? Does equation (50) survive a baseline that moves with `L`? And does novelty item 2 of
   `related_work.md` stand, or its retraction in `results.md` 14.6?
2. What does H1 give with `pusher_shape = "box"`? Your fidelity study says the default turns the payload 19 times more through a single contact artefact, and yaw
   is in the success test.
3. Does any measured quantity predict whether search helps on a new snapshot? Seed 2 has a zero model contribution at depth 6 and the second largest gain.
4. Would you restate conclusion 5 as "search pulls every agent to 65 to 80 percent regardless of its policy", which the four agents support, rather than as a gain
   of 19.8 points? Why do the two simulators use different baselines in the section 1 table?
5. In `results.md` 17 the sampled policy rises 21 points from lag 0 to lag 4 while depth 6 stays flat. Does that run refute H2 in absolute terms?
6. How do you position the staleness axis against the delayed sharing literature, and H2 and H3 against Jiang et al.? Does the 16 robot transfer gain survive a
   per team size difficulty control? Will you record the score gap `g` that `math.md` 10.6 calls free?

## Limitations

The authors report their limitations well and now report them in the right places. `results.md` 12 names which sections are one seed, conclusion 1 states the
mechanism is unsettled, 13.8 publishes the reproducibility floor, and `math.md` 10.6 lists which constants cannot be measured. My objection is narrower than in
round 1. Three limitations are load bearing and are not stated where the claim is made: the fair baseline is not flat in lag, every arm runs the cylinder pusher
the fidelity study calls artefactual, and the body of section 14 states mechanisms section 16 retracts. The status note at the top of `next_steps.md` is dated
2026-09-12 and covers items 2 to 8 only, so it omits the three seeds, the collection run, the momentum variant and the bound.

## Ethics

No concerns. Simulation only, no human subjects, no data release, no dual use surface.

## Scores

- **Soundness: 3 (good).** Up from 2. The statistics now fit the claims: paired bootstraps over first episodes, per batch seeding, a published reproducibility
  floor, a correct standard error of the difference, an argmax free statistic, three seeds per simulator, and controls for the model, the scorer, the candidate
  and the leader fallback. Below 4 for W1r, W2r and W5r.
- **Presentation: 3 (good).** Up from 2. `related_work.md` exists, the conclusions no longer assert what the next steps report as untested, and `math.md` section
  13 marks its assumptions. Against it: `results.md` runs 1 to 10, 13 to 19, then 11 and 12, section 13 of `math.md` labels its subsections 10.1 to 10.7, and
  sections 14.1, 14.3 and 14.6 carry claims later sections retract.
- **Contribution: 3 (good).** Up from 2. A three seed, five lag, argmax free measurement of a decision time search gain decaying with communication delay, a bound
  that predicts which statistic is stable, four clean negative results and a reusable simulator fidelity study. Below 4 because the mechanism is unresolved, the
  effect appears on one task, and every component is prior work.
- **Overall: 5 (borderline accept).** Up from 3. In round 1 I wrote that if the search cleared the sampled policy and the random candidate by more than 2 SE at
  two or more lags, the headline is real and I move to 5. It does, at lags 0 and 1, at 256 environments and 5 batches, in 3 of 3 seeds. As a workshop paper or a
  technical report this is clearly above the bar. As a main track paper it is not: the central claim has no supported mechanism and its strongest result is
  positioned against no prior work.
- **Confidence: 4.** I recomputed the fair H1 standard errors, the staleness decomposition and the least squares slopes from the committed tables, but did not
  read the code.

## What changed my score, and what would change it again

Round 1 rested on one arithmetic finding and one absence: no cell beat the sampled policy at 2 SE, and no document positioned the work against prior art. The
authors ran the arm at the size I asked for, on three seeds, with a paired bootstrap, and it clears. They also did the harder thing. They reran a sweep with a
fixed seeding bug, watched their own H2 and H3 verdicts flip to "not resolved", published the flip, and replaced the statistic with one their new bound actually
predicts. Three explanations for the 2D negative were tested and all three failed, which cost them their mechanism and which they reported anyway.

What would move it further. First, decompose the staleness curve: report the depth 6 arm's own success against lag beside the gain. If the search arm alone still
declines monotonically in 3 of 3 seeds, H2 becomes a property of the search rather than a difference of two moving lines, and I go to 6. Second, rerun the fair H1
at lags 0 and 1 with `pusher_shape = "box"`, which is four cells; if the gain survives, conclusion 1 stops depending on a single contact point artefact. Third,
close the four citation gaps, starting with the delayed sharing literature and Jiang et al. 2015, and fix or withdraw novelty item 2 of `related_work.md`. Fourth,
record `g` and test equation (48). What would move it down: if the box pusher arm reverses the mjlab H1, conclusion 1 is an artefact of one contact point and the
only positive result goes with it. I would return to 4.

