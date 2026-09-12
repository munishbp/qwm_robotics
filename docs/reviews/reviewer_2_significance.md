# Reviewer 2: significance, novelty, and positioning against prior work

**Paper:** Decentralized World Model Search for Heterogeneous Cooperative Transport
**Lens:** what this adds over prior work, and whether the claims survive the evidence.

---

## Summary

The submission ports the QWM recipe (Dong et al., 2026), which trains a policy and a Q critic on real
transitions and uses a learned world model only at decision time for a short tree search, into a
decentralized heterogeneous cooperative transport task. Each robot builds its own belief from a type
specific encoder, an attention fusion over neighbor latents that arrive one or more frames late, and a
forward correction that rolls each stale latent through the shared dynamics model. Inside the search a
robot imagines its teammates by running the shared policy on its own estimate of each teammate's belief.
The learner is RLPD. The authors run the study twice, on a batched 2D quasi static simulator and on an
mjlab port with contact physics, and report that the runs disagree: search costs 9 points on 2D and adds
24.5 on mjlab. They attribute the disagreement to the critic's action sensitivity, and add staleness
trends for the best depth and the best lookahead discount, a refutation of leader elected search, and
zero shot transfer to 16 robots. The engineering is careful and the documentation is unusually honest.
`docs/next_steps.md` anticipates most of my objections, prices the experiments that would decide four of
the six conclusions at 36 GPU minutes, and reports that none of them ran.

---

## Strengths

- The test time sweep design is economical. One snapshot funds 76 cells, every cell is paired on the
  same weights and env seeds, so within sweep comparisons are trustworthy (`methodology.md` section 8).
- The disagreement between the two simulators is the headline rather than a footnote. Most submissions
  would have dropped the 2D run (`results.md` section 1).
- The probe on the 2D failure is concrete: the critic spans 0.003 across the policy's own nine
  candidates, and the argmax picks the mean action 1.4 percent of the time against a chance rate of 11
  percent (`results.md` section 5). That is a measured quantity, not a story.
- `math.md` section 10 separates what is a bound under a stated assumption (equation 34) from what is a
  labelled heuristic (equations 35 to 38). Few submissions mark that line, and `results.md` section 12
  with `next_steps.md` name the sampled policy problem, the censored depth axis, the leader fallback
  confound, and the single seed without being asked.
- The design choice that a teammate estimate receives no new information at any depth, so its age
  feature stays at the root value, is correct and, as far as I know, unpublished (`math.md` section 9.1).

---

## Weaknesses

### W1. No related work exists anywhere in the submission

`math.md` section 13 lists six references: RLPD, SAC, QWM, Attention Is All You Need, TD3+BC, and one
Dec-POMDP textbook. `concepts.md` section 19 lists four reading items, three of them the same papers.
No document has a related work section. For a submission whose claim is novelty against the multi agent
literature, this alone is disqualifying. The specific gaps:

- **Imagining teammates in a rollout.** `concepts.md` section 14 calls this "the only piece with no
  prior implementation to copy". Interactive POMDPs (Gmytrasiewicz and Doshi, 2005) and Interactive
  POMCP (Hoang and Low, 2013) are the machinery of nesting an estimate of another agent's belief inside
  your own rollout. Self Other Modeling (Raileanu et al., 2018) is this paper's trick exactly: run your
  own policy network on your estimate of another agent's input to predict its action.
- **Learned model search in MARL.** MAZero (Liu et al., ICLR 2024) runs MuZero style search on a learned
  model with decentralized execution. MAMBA (Egorov and Shpilman, 2022) learns a communicating multi
  agent world model. MAMBPO (Willemsen et al., 2021) is model based multi robot actor critic. The claim
  that the multi agent search is "the piece that is actually new" must be argued against these.
- **The staleness axis.** One step delayed information sharing is a classical information structure with
  an optimal control solution (Nayyar, Mahajan and Teneketzis, 2011; Oliehoek and Spaan on Dec-POMDPs
  with delayed communication). Delay aware MARL (Chen et al., 2020) and delayed feedback RL (Walsh et
  al., 2009) cover the learning side. Rolling a stale estimate forward is a Smith predictor.
- **The depth and discount results.** Jiang et al. (AAMAS 2015) is the canonical result that the optimal
  planning horizon and the optimal discount shrink as model error grows. H2 and H3 are that result with
  a second error source substituted in. The substitution is a contribution. Omitting the original is not.
- **The headline.** Hamrick et al. (ICLR 2021) found that MuZero style test time search contributes far
  less at evaluation than assumed, and that the gain depends on value quality. Grill et al. (2020) show
  MCTS is regularized policy improvement around the prior, which bounds the gain over sampling from that
  prior. Conclusion 1 is the multi agent restatement of a four year old single agent result.
- **Depth 0.** The depth 0 arm is a Q argmax over N policy samples, which is QT-Opt (Kalashnikov et al.,
  2018). On mjlab it supplies 16 of the 24.5 headline points (`results.md` section 13.3), so the largest
  component of the headline is an uncited 2018 method.
- **Transfer and the task.** Permutation invariant pooling that transfers across team size is standard:
  Deep Sets, MAAC (Iqbal and Sha, 2019), deep RL for swarm systems (Huttenrauch et al., 2019), and GNN
  swarm controllers (Tolstaya et al., 2020), which transfer far past 16 robots. Cooperative transport
  also has its own literature and benchmarks (Tuci et al., 2018; Wang and Schwager on force amplifying
  transport), and the submission positions against none of it.

### W2. The headline does not beat the one line baseline, by the submission's own test

`math.md` equation (40) calls a difference real when it exceeds 2 SE of the difference. Applied to every
mjlab search cell against the sampled policy, using the numbers in `results.md` sections 13.3 to 13.7:

| Comparison | Difference | 2 SE | Significant |
|---|---|---|---|
| depth 2, lag 1 (69.5 ± 1.8) against sampled lag 1 (65.4 ± 2.2) | +4.1 | 5.69 | no |
| depth 6, lag 0 (72.7 ± 4.3) against sampled (65.4 ± 2.2) | +7.3 | 9.66 | no |
| depth 6, lag 1 (71.4 ± 2.5) against sampled (65.4 ± 2.2) | +6.0 | 6.66 | no |
| depth 6, lag 2 (71.6 ± 2.9) against sampled lag 2 (67.4 ± 0.7) | +4.2 | 5.97 | no |
| depth 4, lag 0 (69.3 ± 1.3) against sampled (65.4 ± 2.2) | +3.9 | 5.11 | no |

`results.md` section 13.3 states the margin "reaches 2 SE only at the deeper settings (depth 6 at lag 0:
72.7 ± 4.3 against 65.4 ± 2.2)". That cell gives 7.3 against a 2 SE of 9.66 and does not reach it.
`next_steps.md` line 47 states "4.1 points above the sampled policy against a 2 SE of about 2.9". The
correct 2 SE is 5.69; 2.9 is one SE. The only cell that clears the bar is the H3 argmax at beta 0.9, lag
0 (72.9 ± 0.3 against 65.4 ± 2.2, 2 SE 4.44), which is the winner of a seven way sweep compared against
a baseline at a different lag on half as many environments. The central positive claim therefore has no
supported margin over adding exploration noise to the mean action, and cost makes it worse: the deep
settings where a margin might exist cost 104 ms per step against 28 (`results.md` section 13.4), and
search cost grows with the square of team size, reaching 474 ms per step at 16 robots (section 9).

### W3. The central finding is a property of two checkpoints, not of two physics classes

Conclusion 1 generalizes from one snapshot per simulator to a claim about quasi static against contact
physics. The runs differ in at least six ways: contact against a fiat friction threshold, 2.5 m/s
against 1.5 (`mjlab_port.md` section 2), 24,000 training steps against 12,000 (`methodology.md` section
5), a scripted ceiling of 83.6 against 99.6 percent, a baseline of 45 against 52, and a 2D task eased
once after a 6 percent plateau (`results.md` section 12). `next_steps.md` F10 states that the ease
change or the speed change "can set the sign of H1", prices it at 101 GPU minutes, and did not run it.

The mediator is not measured where it matters. The 0.003 span is a 2D number. For mjlab,
`next_steps.md` line 256 says the critic "must sit above 1, because its argmax alone adds 16 points",
which infers the mediator from the effect it explains. A one GPU minute probe would measure it. A
mediation claim with the mediator measured on one of two arms is not a finding. One rival the submission
never names: the offline buffer comes from a scripted controller that pushes with saturated `u = 1`
(`design.md` section 4), and the critic trains on a SARSA target over that behavior mixture. A critic
that learns "larger action is better" from saturated demonstrations helps where crossing a friction
threshold needs force and hurts in a quasi static model. That explains everything conclusion 1 explains
and puts the cause in the data and the target, not the physics.

### W4. H2 and H3 are a noisy argmax over a censored grid

On mjlab the best depth is 6, 6, 6, 4 and 6 is the largest depth in the grid (`results.md` section
13.4). Three of four rows put the argmax on the edge of the axis, so no collapse has been observed. The
one drop rests on 63.0 ± 1.6 against 60.9 ± 4.3, a difference of 2.1 against a 2 SE of 9.18. Every depth
from 2 to 6 lies inside the noise of every other depth at every lag.

Quantify the rule. An argmax over five equally good depths, repeated at four lags, is non increasing
with a strict drop about 10 percent of the time under pure noise; seven beta levels give about 9
percent. The study reports four of four rules met across two hypotheses and two simulators, and paired
seeds make smooth sequences easier still. The rule cannot separate the predicted effect from a flat
surface. The rules also disagree between documents: `math.md` section 11.2 requires the peak at the
smallest lag to beat depth 0 by 2 SE, `methodology.md` section 7 drops that clause, and `results.md`
applies the looser one. For H4, `math.md` section 11.4 requires both differences to be significant while
`methodology.md` requires only that leader beats independent "at some lag", an uncorrected search over
four comparisons. On 2D, H2 is confirmed while every search cell sits below its own no search row
(`results.md` section 6), so "best depth" there means "least harmful depth". A rule that confirms
whether search helps or hurts is not testing anything.

The honest quantity, depth 1 minus depth 0, is the right one: +2.1, 0.0, -4.2, -13.3 on 2D and +8.3,
+5.7, +6.0, +5.2 on mjlab. The mjlab sequence is four points inside noise of each other. A stronger
version needs an uncensored depth axis, a paired per environment bootstrap, a direct measurement of the
two error terms of equation (37), and one arm the study never ran: oracle teammate actions inside the
search. Lag 0 is not that arm, because at lag 0 the searcher still imagines teammate actions from an
estimate of a teammate belief. True teammate actions are the only clean way to separate imagination
error from model error, and that is the experiment H2 exists to motivate. The direction is also
unsurprising: more error in a rollout shortens the useful horizon and lowers the optimal weight on
lookahead, which is Jiang et al. 2015. The new content is that the error source is teammate staleness
and that it is controllable at test time. That increment is real, small, and not isolated here.

### W5. The H4 refutation carries the same confound as the H4 confirmation

`results.md` section 8 discards the 2D confirmation because 44 percent of robots in leader mode fall
back to the plain policy. The mjlab refutation uses the same leader arm with 31 percent falling back
(`next_steps.md` F7). A confound that invalidates a positive result invalidates the negative one. The
round robin arm has no fallback and does lose, so "broadcast joint search loses to independent search"
has support, but round robin changes the election rule and the fallback together, so it tests a
different hypothesis than H4. The refutation is also near tautological. Every actor was trained to act
on its own fresh observation, and leader mode imposes a joint action computed from one robot's stale
estimates of five teammates, which is off distribution for every follower. Losing 28 points under that
substitution says little about where to spend a decision time budget. A meaningful test matches
information, not only wall clock: a leader that searches only its own action, or a leader with oracle
fresh teammate beliefs. F7 costs 8 GPU minutes.

### W6. The belief pipeline and the transfer result are the real contribution, and the writeup buries them

The largest effects in the package are not the search. Masking messages costs 36.8 points on 2D and 16.2
on mjlab (`results.md` sections 9 and 13.7). The forward correction keeps eight frame old messages
usable. The six robot team runs zero shot at 16 robots. These are several times the search margin and
they survive dropout. They appear as conclusion 4 of 6. Two problems stop even this from being clean.
First, the transfer numbers confound generalization with task easing, as the authors concede: "More
robots also mean more force against the friction threshold, which is part of the gain" (`results.md`
section 9). On 2D the 16 robot team scores 95.6 percent against 50.3 for the training team. A transfer
target far easier than the training condition is not a transfer result until a per team size difficulty
control exists; `next_steps.md` F11 answers it in 4 GPU minutes and did not run. Second, the components
are standard, so the novelty is the assembly, and the assembly cannot be judged without W1.

### W7. The nine training changes are engineering and rediscovery, not a method

`results.md` section 2 presents nine changes as part of the result. Most are rediscoveries. Dropping
entropy from the backup is an existing RLPD flag. The cloning term on the offline half is TD3+BC, cited.
The SARSA target on the recorded next action is the one step behavior value estimate of Brandfonbrener
et al. (2021). Target copies of the encoder plus a detached critic and an auxiliary decoder is SAC-AE
and its descendants. Clamping the target to a known reward range is common practice. The evidence is
also single path: each change followed one failed run at one seed, and nothing measures the seed spread,
so nothing separates "this change was needed" from "this run was unlucky" (`next_steps.md` F4, F12).

The useful reading, that published RLPD does not train on a sparse reward partially observed multi agent
task without four additions, is a practitioner negative result. It is not isolated, and one addition
threatens the headline. The SARSA target makes the critic value a behavior mixture, which is the most
likely reason it does not rank the learned policy's candidates. If `next_steps.md` F8 comes back
positive, conclusion 1's physics explanation becomes an algorithm explanation and the central claim
inverts. That experiment costs 31 GPU minutes.

### W8. The testbed does not yet support claims about contact physics

`mjlab_port.md` sections 2 and 5 record a kinematic latch written every substep, a fiat central
unloading force bounded at 80 percent of the weight after an earlier version launched the box off the
floor, at most one contact point for a cylinder against a box, no robot to robot collision, and robots
that are cylinders on velocity servos. This is a better testbed than the 2D model, but the two features
most likely to drive the critic's action sensitivity, the friction threshold and the latch, are still
scripted in both simulators, so the difference the paper names is the one it has least isolated.
Separately, the learner reaches 45 to 50 percent while a scripted centralized controller reaches 83.6 to
99.6. A learned system at half the success rate of a hand written controller is not a contribution to
cooperative transport, so the value must rest on the mechanism study, which returns to W2 through W5.

### W9. The conclusions and the next steps contradict each other

`results.md` section 11 asserts six conclusions in the indicative mood. `next_steps.md` names the
strongest rival for each, and for four of six the rival is untested and the decisive experiment costs
under 25 GPU minutes. A reader of `results.md` alone receives claims the authors do not believe. Restate
conclusions 1, 2, 3, and 4 as the hypotheses they still are.

## Questions for the authors

1. What is the action signal to noise ratio r on the mjlab snapshot? It costs under one GPU minute and
   the whole mediation claim of conclusion 1 depends on it.
2. Does any cell beat the sampled policy by more than 2 SE of the difference under equation (40)? My
   arithmetic on your tables says no. If you disagree, name the cell and show the computation.
3. What does the search do against oracle teammate actions, meaning the true actions taken at that step?
   Without that arm H2 cannot separate teammate imagination error from model error.
4. On 2D the critic argmax picks the mean action 1.4 percent of the time against a chance rate of 11
   percent, and prefers the candidate farthest from the mean. A flat critic gives chance rates; an anti
   correlated selection is a biased critic. Is the critic increasing in action magnitude, as saturated
   scripted demonstrations would teach it? Relatedly, under `target_policy = "data"` the critic values a
   behavior mixture, so why should it rank the learned policy's own candidates in either simulator?
5. What happens at depth 8 or 12 on mjlab? Three of four rows have the argmax on the grid edge.
6. What does the scripted controller score at 9, 12, and 16 robots on mjlab? Until that exists, transfer
   cannot be separated from larger teams making the task easier.
7. How do you position the teammate imagination step against Interactive POMDPs, Interactive POMCP, Self
   Other Modeling, MAZero, and MAMBA?
8. How do H2 and H3 differ from Jiang et al. (2015) beyond substituting a new error source, and what is
   the multi agent content of conclusion 1 beyond Hamrick et al. (2021)?
9. What is the seed spread of best evaluation success? Every cross run claim rests on one seed.
10. Why does the leader arm keep its fallback in the mjlab refutation, when you discard the 2D
    confirmation for exactly that reason?
11. Two documents state different decision rules for H2 and H4. Which set was fixed before the runs?

---

## Limitations

The authors state their limitations well (`results.md` section 12, `methodology.md` section 8,
`mjlab_port.md` section 5, and all of `next_steps.md`). My objection is not that they are hidden. It is
that several are load bearing for the conclusions above them, and those conclusions are written as if
they were not. One seed, one selected checkpoint per simulator, a censored depth grid, an unmeasured
mediator, a confounded simulator comparison, and a headline with no supported margin over a stochastic
policy are not caveats on a result. Together they are the reason there is not yet a result. The authors
already designed the 36 GPU minutes that would settle four of six conclusions, which makes the current
state a choice rather than a constraint.

On audience, which a significance review must answer: as written, nobody outside the project can use
these numbers. A multi robot practitioner cannot transfer a result from a kinematically latched box to a
real team. A MARL researcher needs seeds and a standard benchmark. The one transferable item is a
screening rule, which is to measure the critic's action span over the policy's own samples before paying
for test time search, and the submission proposes that quantity in `next_steps.md` without measuring it.
Measure r on both snapshots and the paper acquires its first genuinely useful reader.

## Ethics

No concerns. Simulation only, no human subjects, no data release, no dual use surface.

---

## Scores

- **Soundness: 2 (fair).** The implementation and the paired sweep protocol are sound. The inference is
  not: one seed and one selected checkpoint per simulator, a significance claim that fails the paper's
  own test, a mediator measured on one of two arms, a decision rule that passes on noise roughly one
  time in ten, and a two simulator comparison that differs in six ways while the conclusion names one.
- **Presentation: 2 (fair).** The prose, the equation numbering, the separation of bound from heuristic,
  and the replication recipe are above the venue norm. Two things the NeurIPS presentation criterion
  covers directly hold the score down: no related work section in any document, and a conclusions
  section that asserts claims the next steps document reports as untested.
- **Contribution: 2 (fair).** The teammate imagination step, the staleness axis, and the belief pipeline
  are real work. Each has close prior art the submission does not engage, and the two most defensible
  results are presented as supporting material and lack their own controls.
- **Overall: 3 (reject).** Below threshold. The negative 2D result and the honest reporting deserve
  publication eventually, but the positive claim does not survive its own statistics and the positioning
  against prior work does not exist.
- **Confidence: 4.** I recomputed every significance claim I challenge from the submission's own tables
  and equation (40). I did not read the code, so I cannot exclude an implementation defect in either
  direction, and the submission notes that no experiment validates `swarm/search.py` against a scorer
  that cannot be flat (`next_steps.md` F2).

---

## What would change my score

1. **Beat the sampled policy.** Run `next_steps.md` F1 at 256 environments and 6 batches, with the
   random candidate arm and the tuned noise arm. If the search clears both by more than 2 SE at two or
   more lags, the headline is real and I move to 5. If it does not, rewrite the paper around the
   negative result and the belief pipeline, which I would consider at 4.
2. **Measure the mediator on both arms and run the controls.** F1, F2, F3, F9, and the r probe cost
   under 45 GPU minutes together and turn conclusion 1 into a measured mediation. Add F8 to exclude the
   SARSA target as the cause of the 2D flatness. That is worth one point.
3. **Write the related work section.** At minimum: I-POMDP and Interactive POMCP, Self Other Modeling,
   MAZero and MAMBA, the delayed sharing literature, Jiang et al. 2015, Hamrick et al. 2021, Grill et
   al. 2020, QT-Opt, and the cooperative transport survey, each with one sentence saying what this work
   does that they do not. This is not a bonus. Without it I cannot verify that any claim is new.
4. **Three seeds and an uncensored depth axis.** F4 plus F6 cost 226 GPU minutes and convert every cross
   run claim from anecdote to measurement. If the sign of H1 agrees across three seeds per simulator and
   the depth falloff moves with lag, H2 becomes a finding and I would support acceptance.

Items 1 and 2 cost about one hour of GPU time on hardware the authors already have. The study is one
honest hour away from knowing whether it has a result. I encourage them to run it and resubmit.
