# Reviewer 3: clarity, internal consistency, and completeness of explanation

**Paper:** Decentralized World Model Search for Heterogeneous Cooperative Transport
**Lens:** clarity, internal consistency, completeness of explanation

## Summary

The submission asks whether the test time world model search of QWM survives decentralization, partial observability, and stale teammate information. It answers with one trained snapshot per simulator and a grid of test time sweeps on it, run twice: on a batched 2D transport simulator and on mjlab. The headline inverts between the two simulators, and the authors present the inversion as the finding, which is honest and interesting. The repository is unusually complete: a proposal, a concepts primer, a design document, a mathematics document, a methodology, two result documents, two table documents, a port document, a next steps document, code, figures, and an interactive replay. The problem is that these documents disagree with each other and with the code, and the disagreements sit on load bearing constants and on the rules that decide the hypotheses. `math.md` describes an earlier version of the task throughout. `methodology.md` and `concepts.md` state the opposite of `design.md` and the code on the critic target. The same evaluation cell is reported five times with five different numbers and no explanation. The headline H2 figure shows a clean staircase that the paper's own text calls an artifact. I recommend rejection in the current form. I would change that if the documents were reconciled against the code and against each other.

## Strengths

- The two simulator design is the right experiment, and the paper reports the disagreement rather than the winner. `results.md` section 1: "The two runs disagree on the headline, and the disagreement is the main finding."
- The paper argues against its own headline. `results.md` section 11 item 5: "The sampled policy is the baseline a fair claim about search must beat", followed by the admission that the protocol did not use it.
- The 2D negative result is reported at full strength with a stated mechanism, not buried.
- `math.md` labels which claims are bounds and which are heuristics. Section 10.3: "Equation (34) is a bound, valid under Assumption A. Equations (35) to (38) are a heuristic, for three reasons."
- The design and the code agree on the search procedure. `swarm/search.py` implements roll, score, prune, expand in the order `math.md` section 9.1 specifies, and its `beta = 0` shortcut matches equation (30).
- `docs/next_steps.md` names a falsifying experiment, a file, a flag, and a GPU cost for each of the six conclusions.
- The code docstrings carry measured reasons for the non standard choices. `swarm/rlpd.py` `RLPDConfig` is the clearest single artifact in the repository.

## Weaknesses

Each item names the two locations that disagree.

### Numbers, constants, and procedures that two documents state differently

- **The critic target.** `methodology.md` section 5: "target is the **minimum** of 2 random heads". `design.md` section 6.4: "Target: the **mean** of two random heads, clamped to `[0, 1]`. RLPD uses the minimum." `math.md` equation (19): "RLPD takes the minimum of that pair, and this project does not." `swarm/rlpd.py`: `target_reduce: str = "mean"`. `concepts.md` section 18 repeats the wrong version: "sample a pair, take the min". Two of five documents describe a learner the code does not implement, and this is the change failure run 4 exists to justify.
- **Communication radius.** `math.md` section 1 notation table: "$R_c$ ... comm radius | 5.0, 0.8, 0.4, **4.0** m". `design.md` section 3.5: "distance under `R_c = **6.0**` m". `swarm/env.py`: `comm_range: float = 6.0`.
- **Sensing ranges.** `math.md`, prose under equation (2): "**2.0 m** for a pusher and a gripper and **8.0 m** for a scout". `design.md` section 3.5: "pusher `3.0`, gripper `3.0`, scout `15.0` m". `swarm/env.py`: `sense_range = (3.0, 3.0, 15.0)`.
- **Goal sampling.** `math.md`, prose under equation (10): "Translation crosses the **2.5 m to 4.0 m** gap" and "design section 3.1 samples the goal orientation up to **90 degrees**". `design.md` section 3.1: "a goal at distance **1.5 to 3.0** m ... a random angle in `[-pi/4, pi/4]`". Equation (10)'s feasibility argument is computed on a task the project stopped using.
- **Contact distance.** `math.md` equation (4) conditions on `||x_k - Pi(x_k)|| <= d_c` with `d_c = 0.25`. `design.md` section 3.2 states the same threshold as "robot center within `0.45` m". `swarm/env.py` uses `contact_dist: float = 0.45` against the center. Equation (4) is wrong by one robot radius.
- **World model pretraining.** `design.md` section 6.5: "The model is pretrained on the offline buffer encoded by the trained encoders". `methodology.md` section 5: "There is no separate pretraining phase because the encoders are random at the start." `swarm/rlpd.py` trains the world model from the first update.
- **World model file.** `design.md` heading 6.5 is "World model: `swarm/world_model.py`". No such file exists. `README.md` correctly puts it in `swarm/nets.py`.
- **mjlab throughput at 256 envs.** `mjlab_port.md` section 3 gives two values, "2,900, 6,100, **11,850**" and "Throughput at 256 envs in the controls script | **14,400**". `results_tables_mjlab.md`: "Throughput: **9,064** env steps per second at 256 envs." `runs/mjlab/results/controls.json` records 9064.1. Two of the three have no source.
- **The mjlab replication command.** `README.md`: `SWARM_SIM=mjlab RUN_DIR=$PWD/runs/mjlab bash scripts/run_all.sh`. `methodology.md` section 4: "`... bash scripts/run_all.sh **24000**` is the mjlab run." `scripts/run_all.sh`: `STEPS="${1:-12000}"`. `mjlab_port.md` section 4 repeats the README form. A reader who follows the README reproduces `belief_12k` at 28.9 percent, not the reported snapshot at 48.4.
- **The pre-ease task.** `design.md` section 3.5: "The first draft used **2.0 m** sensing and **4.0 m** communication." `next_steps.md` F10: "the pre-ease ranges, sensing **1.5 m** and communication **3 m**". `math.md` implies a third version, 2.0 m for pushers and grippers and 8.0 m for the scout. The most consequential task change in the project has three incompatible records.
- **The H1 decision rule.** `math.md` section 11.1: "Comparison | $\hat{d} = \hat{p}(D=2) - \hat{p}(D=0)$". `methodology.md` section 7: "H1 | success at lag 1: depth -1 against depth 2". Different quantities, opposite signs on 2D: +0.9 against -9.1 points. `results.md` uses the second and never mentions the first.
- **The H2 and H3 confirmation rules.** `math.md` 11.2 requires "at the smallest $L$ the peak beats $D=0$ by more than $2\,\mathrm{se}$". Section 11.3 requires "$\beta^*(L_{\max}) < \beta^*(0)$". `methodology.md` section 7 states neither clause. The decision table depends on which document the reader opened.
- **The unloading force.** `swarm/env_mjlab.py` docstring, bullet 3: "a **20 N** upward force at the payload center unloads the friction". Same docstring, bullet 1: "One latched gripper unloads it by **25 N**". Code: `lift_force: float = 25.0`. The docstring contradicts itself and never mentions the 80 percent clamp that `mjlab_port.md` section 5 and `results.md` 13.7 call a fix that changed published numbers.
- **Arena escape.** `design.md` section 3.1: "may leave the arena by up to one **diameter**". `swarm/env.py` comment: "can pass the robot limit by up to one **radius**".
- **The 4 robot team.** `results.md` section 9: "scores zero by construction". Section 13.7: "again unsolvable by construction". The mjlab transfer table on the same page reports 2.6 and 2.3 percent for `p2g1s1`.
- **Significance of the mjlab margin.** `results.md` 13.3: the margin over the sampled policy "reaches 2 SE only at the deeper settings". `next_steps.md` Conclusion 1: "4.1 points above the sampled policy against a 2 SE of about 2.9". From 69.5 ± 1.8 and 65.4 ± 2.2, 2 SE of the difference is 5.7, not 2.9. One document calls the comparison significant that the other calls insignificant.
- **Identical configurations, different numbers.** `math.md` 9.2: the `D = 0` and `beta = 0` columns are "the same critic argmax, computed by the same code on the same snapshot". `results_tables.md` lag 4: `D=0` is 51.6 ± 1.2, `beta=0.0` is 48.7 ± 2.6. mjlab lag 0: 60.7 ± 2.2 against 62.2 ± 2.0. `swarm/search.py` confirms one code path (`D = cfg.depth if cfg.beta > 0 else 0`).
- **The same cell reported five ways.** Depth 2, lag 1, beta 0.5, 128 envs appears in `results_tables_mjlab.md` as 70.3 ± 2.3 (H2), 67.2 ± 0.9 (H3), 69.0 ± 1.7 (H4 independent), 67.2 ± 0.9 (dropout 0.0), and 66.4 ± 2.0 (transfer default). The mjlab no search cell at lag 1 appears as 44.0 ± 1.6, 43.5 ± 2.6, 45.3 ± 1.2, and 45.8 ± 1.4. The reported standard errors are smaller than the spread between files, and no document notes it.

### Claims in `results.md` with no supporting number, table, or figure

- "the critic's values span 0.003 on a 0 to 1 scale" (section 5). This is the basis of conclusion 1. No table, figure, or result file reports it. `next_steps.md` F9 proposes the script that would produce it, which confirms it does not exist.
- "It picks the mean action 1.4 percent of the time (chance is 11 percent) and it picks the candidate farthest from the policy mean (mean distance 0.34 against 0.28 for an average candidate)" (section 5). Three numbers, no source.
- "its cost grows with the square of the team size (474 ms per step at 16 robots)" (section 9). The value is in `results/transfer.json`, but `scripts/summarize.py` drops `ms_per_step` from the transfer table, so no table or figure in the paper carries it.
- "a random sample 60 percent, the mean 52 percent" (section 5). The ablations table on the same page gives 62.2 and 50.3. Section 3 gives a third pair, "60 percent against 47 percent".
- "the argmax alone adds 16 points and the world model rollout adds another 8" (section 1). The attribution of 8 points to the model is never tested against a random model. `next_steps.md` F3 identifies this as untested.
- "the no search baseline is flat across lags" (section 13.4). The mjlab H2 no search column is 37.2, 44.0, 45.6, 38.8, an 8.4 point spread against standard errors of 0.7 to 3.4.
- "the forward correction keeps eight frame old messages usable" (section 9). No experiment turns the forward correction off. `next_steps.md` F5a proposes it.
- "which carried over to mjlab without change" (section 11 item 6). Nothing records that the nine changes were re-tested on mjlab.

### Terms used before definition, and unexpanded acronyms

- **QWM** appears in the first sentence of the proposal and throughout the README, design, methodology, and results. It is expanded only in the `concepts.md` glossary and the `math.md` reference list. The README, which the reader opens first, never expands it.
- **RLPD** is never expanded into words in any document. The glossary gives a description, not the name.
- **EXPO** appears once, in proposal section 4.4, with no expansion and no reference.
- **Dec POMDP** is the heading of `math.md` section 2 and is never expanded. The glossary defines POMDP only.
- **MSE**, **SE**, **utd**, and **UTD** appear before or without definition in `design.md` 6.6 and `results_tables.md`.
- The team codes **p2g1s1, p4g4s1, p6g5s1, p8g7s1** appear in both table documents and in `results.md` sections 9 and 13.7 with no key. The mapping lives only in `scripts/common.py`.
- **"the snapshot"** is used in `README.md` and `design.md` section 1 before section 6 or the glossary define it. **"full36k"** is a table row with one parenthetical gloss. **"copy baseline"** is defined in a parenthesis after its first use.
- One quantity carries four names: staleness, lag, message lag (every figure axis), and L. One arm carries four names: no search, depth -1, the mean action, and the plain policy. The repository's own `CLAUDE.md` requires "one word for one meaning".

### Can a reader reconstruct what was done, in what order

- Partly. `scripts/run_all.sh` is the real order and it is clear. But `methodology.md` section 4, which the document calls "the replication recipe", omits `scripts/ablations.py`, which produces the entire "Ablations on the snapshot" table, and omits the `full36k` run, which is a row of the training table. A reader who follows section 4 cannot produce two reported tables.
- `math.md` names three commands that do not exist: `scripts/sweep.py --grid depth x lag` (11.2), `--grid beta x lag` (11.3), `scripts/sweep.py --leader` (11.4). The real flag is `--which`. `math.md` 11.1 and `design.md` section 9 both name an output file `results/h1.json`; the code writes `results/h1_depth-1.json`, `h1_depth0.json`, and `h1_depth2.json`.
- **The twelve training runs cannot be reconstructed.** `results.md` section 2 opens with "**Twelve** training runs were needed to reach a stable learner" and then tabulates **nine**. Section 11 item 6 says "**nine** documented changes". `next_steps.md` heads a section "Conclusion 6: **nine** documented changes". Nothing identifies runs 10, 11, and 12. The count of twelve appears once and is never supported.
- **The sequence of failures is presented as a result, and this is a strength.** The nine row table in section 2 gives symptom, cause, and change for each, and `math.md` sections 3 to 5 give the measured evidence. It is the best part of the submission. Its weakness is that no failure is dated or ordered against a task change, so a reader cannot tell whether run 5 (the task ease) came before or after the critic changes. That matters, because the ease could have caused the flat critic that conclusion 1 blames on the physics, as `next_steps.md` F10 concedes.
- **The two task changes are recorded but not reconciled.** The eased 2D task appears in `design.md` 3.1 and 3.5, `results.md` section 2 run 5, and `results.md` section 12. The mjlab speed appears in `methodology.md` section 1, `mjlab_port.md` section 2, and `scripts/common.py`. Neither appears in `math.md`, which still computes on the pre-ease task, and neither has a control. `next_steps.md` F10 states the risk plainly: "Either change can set the sign of H1."

### Figures

- Every figure is legible, labeled, and titled, and every 2D figure is referenced by name in the table in `results.md` section 10. That is the good news.
- **No mjlab figure is referenced individually.** Section 13 says only "The figures are in `runs/mjlab/figures/`". The mjlab run is half the paper and carries the confirming result, and it has no figure table.
- **`figures/learning_curves.png` does not show the `full` run.** The orange series is in both legends and invisible in both panels, hidden under the green `full36k` curve. The figure is cited to support "The full state baseline learns more slowly than the belief agent at 12,000 steps", which a reader cannot see in it.
- **`figures/h2_best_depth.png` contradicts the text that cites it.** It shows a clean staircase, 4, 2, 0, 0, with no error bars and no no search reference. The text on the same page says "Every search cell is below the no search cell of its row, so the 'best depth' is the least harmful depth." A reader who sees only the headline figure draws the opposite conclusion. The three cells that set the lag 0 bar (37.2, 34.6, 33.6) are inside 2 SE of each other.
- **There is no H1 figure.** `scripts/plot.py` has no `fig_h1`. H1 is the hypothesis whose sign differs between the simulators, which is the stated main finding.
- **There is no figure that places the two simulators side by side.** The main finding has no picture.
- **The figure a reader needs most and does not have:** the distribution of critic values across the nine root candidates on both snapshots. Conclusion 1 rests entirely on "values span 0.003" and no figure shows it. Two more are missing: search cost against team size (the 474 ms claim), and the H4 panel with the no search line on it, since section 8 argues "Leader mode lands exactly on the no search number".
- `figures/transfer_team.png` is titled "Transfer to a new team size" while the table it comes from is titled "Transfer: team composition". The figure discards the composition, joins an unsolvable point at K = 4 to the rest with a line, and plots the training team K = 6 as a transfer point.
- Cost numbers mix batch sizes without normalizing. The H1 table gives 13.5 ms per step at depth -1 with 256 envs; the H2 cost line gives 25 ms at depth -1 with 128 envs. Halving the environments nearly doubles the reported cost of the cheapest arm, and no document addresses it.

### Mathematics against the final code

I checked equations (2), (4), (5), (6), (10), (12), (12b), (13), (15), (19), (20), (21), (22), (22b), (24), (25), (26), (27), (28), (29), (30), (31), and (32) against `swarm/env.py`, `swarm/nets.py`, `swarm/rlpd.py`, `swarm/belief.py`, and `swarm/search.py`.

- Correct against the code: (5), (6), (12), (12b), (13), (20), (21), (22), (22b), (24), (25), (28), (29), (30), (31), (32). Equation (19) is correct and contradicts `methodology.md`, not the code.
- **Equation (2) describes an earlier version.** Sensing 2.0 and 8.0, communication 4.0. The code has 3.0, 15.0, and 6.0.
- **Equation (4) describes an earlier version.** `d_c = 0.25` to the boundary; the code uses 0.45 from the center.
- **Equation (10) describes an earlier version.** It computes feasibility over a 2.5 m to 4.0 m goal gap and a 90 degree rotation, so its conclusion that "the worst case really occurs" no longer follows.
- **Equation (27) does not match `swarm/nets.py`.** It states `b_i = att_i + MLP(att_i)`. `Fusion.forward` computes `norm2(norm1(own + out(attn)) + ff(...))`: a residual on the own encoding and two LayerNorms the equation omits. The permutation invariance proof survives, but its last line, "equation (27) makes $b_i$ a function of $\mathrm{att}_i$ alone", is false of the code.
- **Equation (15) is one loss; the code takes two optimizer steps.** `swarm/rlpd.py` steps `opt_actor` once inside the critic backward (the cloning term) and again for the SAC term.
- The section 1 notation table lists `E` as "512 train, 256 evaluate". `methodology.md` uses 256 for training and 128 for the sweeps. `math.md` section 11 asserts "The design's protocol fixes evaluation at 256 environments", which 68 of the 76 reported cells do not satisfy.

### The writing against its own stated style

`CLAUDE.md` requires the active voice, the present tense, one word per meaning, no em dashes, and no emojis. The last two are honored everywhere; I found no em dash and no emoji. The lapses:

- One word for one meaning fails on the two clusters above, and on snapshot against checkpoint against best checkpoint.
- Sentence length: `math.md` section 3, under equation (12), runs one sentence of 71 words from "A bootstrap on the policy mean then held" to "returned 0.29 one step from success". Section 4, under equation (19), runs a 78 word sentence. The stated limit is 25 words.
- Passive voice: `methodology.md` section 6, "The mean action is used unless search is on."
- A comment that repeats the code: `swarm/rlpd.py`, "# Critic. Target uses two random heads and the mean action."
- A docstring that is not true in any reading: `swarm/belief.py`, `MessageTable.update`, "Every entry is at least `AGE_MAX` recent." The intent is "at most AGE_MAX steps old".
- Structure: `results.md` section 1 says "sections 2 to 12 cover the 2D run, section 13 the mjlab run". The document prints 1 to 10, then 13 with 13.1 to 13.8, then 11 and 12. Sections 11 and 12 are the conclusions and the limitations and they cover both runs, so both the sentence and the order are wrong.
- `results.md` section 1 promises "The equations referenced as (n) are in `math.md`." No equation reference appears anywhere in `results.md`.
- `README.md` repeats one sentence twice: "Results land in `results/`, checkpoints in `checkpoints/`."

### The proposal's schedule and deliverables against what was delivered

No document reconciles them. `design.md` section 8 lists four differences from the proposal and none is a dropped deliverable. `results.md` section 12 lists five limitations and none is a dropped deliverable. These proposal items appear nowhere else in the repository:

- Section 5 baseline "Search with V instead of Q". The proposal calls it "the direct replication" and says in section 7 that it "survives longest" in the cut order. It was not run and is never mentioned again.
- Section 5 baseline "Independent per robot world models, no messages". The messages masked ablation is a different experiment.
- Section 6, "Belief divergence between robots over time". Never measured.
- Section 6, "Train with 8 robots, evaluate at 4, 12, 16, 32". The delivered run trains on 6 and evaluates at 4, 6, 9, 12, 16. There is no 32 robot cell.
- Section 6, "Train on 4 pushers plus 4 grippers, evaluate on unseen type ratios". The training team is 3 pushers, 2 grippers, 1 scout.

The eight week schedule is unreconciled and has left artifacts. `math.md` section 11: "every cell of every sweep loads the identical **week 5** snapshot". `concepts.md` section 18: "Snapshot | The saved weights at the end of **week 5**". `design.md` section 6.5: the model "is pretrained in **week 4**". `math.md` section 2.3: "the **week 1** negative control". `README.md` says the opposite: "The work was done in one session". A reader meets a calendar that did not happen in four documents.

## Questions for the authors

1. Is the critic target the mean or the minimum of two random heads? `methodology.md` and `concepts.md` say minimum; `design.md`, `math.md` equation (19), and `swarm/rlpd.py` say mean. Which two documents will you correct?
2. `math.md` 9.2 calls the `D = 0` and `beta = 0` cells "the same critic argmax, computed by the same code on the same snapshot". Why do they differ by 2.9 points at lag 4 on 2D? Is the cause that `run_all.sh` runs `h2` and `h3` as separate processes, so the candidate sampling RNG sits at a different position in each?
3. On mjlab the same no search cell at lag 1 appears as 43.5, 44.0, 45.3, and 45.8 across four files, with reported standard errors of 0.7 to 2.6. The 2D no search cells match exactly. Does mjlab reproduce under a fixed seed? The `belief` and `belief_12k` mjlab curves also diverge from about 0.7 million transitions at the same seed 0.
4. Which is the H1 rule: depth 2 against depth 0 (`math.md` 11.1) or depth 2 against depth -1 (`methodology.md` section 7)? On 2D they give opposite signs.
5. How many training runs were there? `results.md` section 2 says twelve and tabulates nine. What were runs 10, 11, and 12?
6. What were the pre-ease sensing and communication ranges: 2.0 and 4.0 metres (`design.md` 3.5) or 1.5 and 3.0 (`next_steps.md` F10)?
7. What is the measured mjlab throughput at 256 envs: 9,064, 11,850, or 14,400?
8. Is the depth 2 margin over the sampled policy on mjlab significant? `results.md` 13.3 says no; `next_steps.md` says yes, using "a 2 SE of about 2.9" where the correct value is 5.7.
9. `results.md` section 5 reports three numbers supporting conclusion 1 that no result file carries: the 0.003 span, the 1.4 percent selection rate, and the 0.34 against 0.28 distances. Where do they come from?
10. Will you correct `math.md` equations (2), (4), and (10), which describe the pre-ease task, and equation (27), which omits the residual and the two LayerNorms that `swarm/nets.py` applies?
11. `README.md` and `mjlab_port.md` give the mjlab command without `24000`, so the default 12,000 applies. Which command produced the reported mjlab snapshot?
12. Why is depth -1 measured at 13.5 ms per step with 256 envs and 25 ms with 128 envs? Will you report cost per environment per decision?
13. Four proposal deliverables were not delivered and are not listed as cut: the V against Q baseline, the independent world model baseline, belief divergence, and the 32 robot transfer point. Were they cut, and under which rule of the proposal's cut order?

## Limitations

The authors state their limitations well in `results.md` section 12 and extend them in `next_steps.md`: one training seed per configuration, a critic that values a behavior mixture rather than the learned policy, two deviations from RLPD needed to train at all, and a task eased once. They volunteer the most damaging one themselves, that the sampled policy is the baseline a fair claim must beat and the protocol did not use it. What the section does not cover is the class of problem this review found: the documents do not agree with each other or with the code, so a reader cannot always tell which system produced the numbers. That is a limitation of the write up, not of the science, and it is the cheaper of the two to fix.

## Ethics

No ethical concern. The work is simulation only, uses no human or animal data, and releases code and locked dependencies.

## Scores

- **Soundness: 2** (fair). The experimental design is sound and the paired sweep structure is right. Two problems hold the score down: identical configurations produce different numbers with no explanation, and the reported standard errors are smaller than the spread of the same cell across files, so the argmax readings that decide H2 and H3 are not safe.
- **Presentation: 2** (fair). Individually the documents are clear and well written. Collectively they contradict each other on constants, on decision rules, and on the critic target, the headline figure contradicts its own caption, and the results document's section order contradicts its own roadmap.
- **Contribution: 3** (good). The question is well posed, the two simulator inversion is a real and useful finding, and the documented sequence of nine learner failures is a contribution in itself.
- **Overall: 4** (borderline reject). A solid study whose write up is not yet trustworthy enough to build on. The technical core looks right where I could check it against the code; the documentation around it does not yet describe that core.
- **Confidence: 4.** I read every document, viewed every figure, and checked the mathematics and the documented constants against `swarm/env.py`, `swarm/nets.py`, `swarm/rlpd.py`, `swarm/belief.py`, `swarm/search.py`, and the result JSON files. I did not run the code and did not re-derive the learning results.

## What would change my score

Three changes would move me to 6, and a fourth would move me to 7.

First, make `design.md` the single source of truth it claims to be and bring every other document to it. Correct the critic target in `methodology.md` and `concepts.md`. Correct equations (2), (4), (10), and (27) and the communication radius in `math.md`. Delete the week language from `math.md`, `concepts.md`, and `design.md`. Fix the `swarm/world_model.py` path, the three non existent `sweep.py` commands, the mjlab throughput in `mjlab_port.md`, and the `env_mjlab.py` docstring. None of this needs a GPU.

Second, explain the cell to cell disagreement. Either re-seed the RNG per cell and rerun the sweeps so `D = 0` and `beta = 0` agree exactly, or state in `methodology.md` that they do not and report the measured run to run standard deviation of a fixed cell. The current standard errors understate the real variation on mjlab by about a factor of three.

Third, produce the missing evidence for conclusion 1. One figure showing the distribution of critic values across the nine root candidates on both snapshots would turn the central mechanism claim from an assertion into a result. `next_steps.md` F9 already scopes it at nine GPU minutes.

Fourth, and this earns a 7: run F3, the randomly initialized world model at depth 2 and depth 6 on both simulators, at five GPU minutes. It is the only control that separates "the world model rollout accounts for 8 of them" from "any perturbation of the root ranking accounts for 8 of them". Without it, the positive mjlab result is a claim about re ranking and not about a world model, and the title promises a world model.

I would also ask for one H1 figure, one figure placing the two simulators side by side, a no search line on the H2 best depth figure, and a reconciliation table mapping the proposal's five baselines and six experiment groups onto what was delivered, with the cut reason for each. Those are editing tasks, not experiments.
