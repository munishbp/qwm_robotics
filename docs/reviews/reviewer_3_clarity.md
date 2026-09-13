# Reviewer 3, round two: clarity, internal consistency, and completeness of explanation

**Paper:** Decentralized World Model Search for Heterogeneous Cooperative Transport
**Lens:** clarity, internal consistency, completeness of explanation. **Round one gave Overall 4.**

## Summary

The authors answered the science I asked for. `results.md` grew six sections: the fair baseline against the sampled policy, three training seeds per simulator, a random world model control, a scorer control, a forward correction control, a leader control without the fallback, a demonstration quality control, a momentum variant of the 2D task, a reward hacking audit, and a search in collection run. `math.md` gained section 13, a bound with seven named assumptions and four results, and every number in it that a file can check, checks. Twenty of the sixty seven discrepancies I listed in round one are fixed and seven more are partly fixed. The critic target reads "mean" in every document. The task constants in `math.md` match `swarm/env.py`. The "twelve runs, nine rows" contradiction is explained. The repeated cell problem is measured and published as section 13.8.

The new sections created a new defect, and it is worse than the old one. The study ran each measurement two or three times: a first sweep, a rerun under per batch seeding, and a second rerun with per episode outcomes. `results.md` now carries all three vintages at once and never says which is which. Section 13.4 states that sections 13.4 to 13.7 are the rerun. Sections 13.6 and 13.7 print the first run. Section 16 prints the paired rerun in its table and the superseded run in the paragraph under it. Section 1, section 5, and the conclusions carry the oldest numbers of all, and section 1 states a mechanism that section 14.6 refutes on the same page. A reader who reads front to back meets four different headline gains.

I raise soundness from 2 to 3 and overall from 4 to 5. I hold presentation at 2. The experiments are now adequate. The document that reports them is not.

## Strengths

- The controls are adversarial, not confirmatory. Section 14.2 runs a random world model: it costs 11 to 13 points on mjlab and 4 on 2D. `runs/mjlab/results/day1_controls.json` group F3 carries every cell.
- The paper retracts its own mechanism in public. Section 14.6 retrains 2D on a 60.7 percent buffer, raises the critic span ratio to 1.07, and finds search still loses 18 points. Conclusion 1 then states "The critic's action span is not the mediator".
- Section 13.8 measures the reproducibility floor instead of hiding it: five repeats of one cell, a spread of 3.9 falling to 2.6 points, and the statement that the reported standard errors are the same size as that floor. `figures/paper/repeated_cell_floor.png` draws it and its ten points match the text.
- Section 16 reports paired bootstrap intervals over 1,280 episodes per seed, and `scripts/seeds_summary.py` regenerates `results_tables_seeds.md` from the files.
- `math.md` section 13 improves on the section 10 heuristic. Its arithmetic checks: the `d^{0.67}` growth, `sqrt(0.0013) = 0.037`, the regression slope of -3.27, and the per seed spreads.
- The audit of section 15 matches `results/audit.json` and `runs/mjlab/results/audit.json` on every row.
- `scripts/day1.sh` to `day4.sh` are readable, resumable, and each header says what it runs and why.

## Weaknesses

Each item names the two locations that disagree, or the missing item.

### The document carries three vintages of the same measurement

- **Sections 13.6 and 13.7 print the superseded run.** 13.4 states "the tables in 13.4 to 13.7 are the rerun". 13.6 gives H4 independent 69.0, leader 41.1, round robin 43.8 at lag 1; `runs/mjlab/results/h4.json` and `results_tables_mjlab.md` give 69.3, 39.8, 43.8. 13.7 gives dropout 0 at 45.3 and 67.2; `robust.json` gives 48.2 and 71.9. 13.7 gives transfer 45.8, 66.4, 2.6, 2.3, 90.4, 93.2, 96.6; `transfer.json` gives 46.1, 69.5, 2.3, 3.1, 93.0, 94.5, 93.8. Three whole tables in `results.md` disagree with the generated table document and with the files.
- **13.7 and 13.8 give two values for one cell, eight lines apart.** 13.7 reports the dropout 0 search cell at 67.2 and the default transfer cell at 66.4. 13.8 lists 67.2 and 66.4 as "before" and 71.9 and 69.5 as "after", and the second pair is what the files hold.
- **Section 16's table and its own paragraph disagree.** The table reads "+19.8 ± 8.9" at lag 0, "+14.0 ± 9.0" at lag 1, "+5.5 ± 5.9" at lag 4, matching `results_tables_seeds.md` and `runs/mjlab*/results/day1_controls_f1.json`. The paragraph three lines below reads "19.3 ± 6.3, 14.9 ± 7.4 and 9.3 ± 8.6". I traced the second set to `day1_controls.json`, which `scripts/day4.sh` superseded.
- **Section 1 and section 11 propagate the superseded set.** Section 1 states "+19.3 ± 6.3 at lag 0, +14.9 ± 7.4 at lag 1 (section 16)". Conclusion 5 adds "5.9 ± 7.9 at lag 4" against the table's "+5.5 ± 5.9". `math.md` section 13 uses the same series.
- **Section 16's seed claim uses the superseded file.** "depth 6 adds nothing over it (−1.4, −0.5 at lag 0; −1.4, −2.2 at lag 1)" comes from `day1_controls.json`. In `day1_controls_f1.json` the seed 1 differences are +0.6 and +1.4, so two of four numbers change sign.
- **The two generated table documents disagree.** `results_tables_mjlab.md` "First day controls" F1 reads 60.5, 58.8, 64.3, 73.3 at lag 0. `results_tables_seeds.md` reads 59.6, 58.6, 64.8, 73.1 for the same arms of the same seed. The first reads `day1_controls.json`, the second `day1_controls_f1.json`. Neither says so.
- **Conclusion 2 uses the first mjlab run.** It states the gain "falls from +35 points at lag 0 to +24 at lag 4", one imagined step "falls from +8.3 to +5.2", and the best discount falls "0.9 to 0.7 on mjlab". Section 13.4's rerun gives +36.2 to +17.4 and +10.2 to +5.2. Section 13.5's rerun gives best beta 1.0, 0.7, 0.9, 0.3 and the verdict "not resolved". Conclusion 2 asserts as a trend what 13.5 reports as unresolved.

### The summary and the conclusions contradict the new sections

- **Section 1 states the refuted mechanism as fact:** "the critic is nearly flat across the policy's own candidates (values span 0.003), so ranking candidates by Q amplifies noise and search loses nine points." Section 14.6: "the span is not what decides whether search helps." Conclusion 1: "The critic's action span is not the mediator (section 14.6)."
- **Section 5 is never amended.** Its "Mechanism" paragraph and the line "Why the critic is flat in the action is the deeper finding" carry no pointer to 14.6 or 18, which eliminate that explanation.
- **Section 1 overstates staleness:** "the gain of search falls with staleness in every run and seed." Section 19: "it falls with staleness in two of three seeds." Section 16: the sign agrees "in 2 of 3 at lags 4 and 8".
- **Section 1 states the seed 0 split as general:** "the argmax alone adds 16 points and the world model rollout adds another 8." Section 16: "The mechanism claim of section 14 was a seed 0 property."
- **Section 13.3 is never updated.** It says the margin over the sampled policy "does not reach 2 SE at any setting" and that resolving it "needs about 6 batches of 256 envs per cell". Section 14.1 runs exactly that and opens "This is the fair H1, and it resolves". 13.3 carries no pointer to 14.1.
- **13.3 gives two values for one ratio in two sentences:** "1.28 on mjlab", then "most of the way to mjlab's 1.27". `results/critic_action_span.json` gives 1.2728.

### Numbers that no file supports

- **`results.md` 16: "Best training evaluations: 48.4, 40.2, 36.7 percent."** `results_tables_seeds.md` gives 48.4, 30.9, 30.1, and the maximum `eval_success` in `runs/mjlab_seed1/results/train_belief.jsonl` is 0.309 and in `mjlab_seed2` 0.301.
- **`results.md` 19: "mean policies are weak at 30 to 36 percent."** The no search cells of `runs/mjlab_seed1/results/h2.json` run 19.5 to 30.2 and of `mjlab_seed2` 21.1 to 26.6.
- **`results.md` 9: "its cost grows with the square of the team size (474 ms per step at 16 robots)."** Unchanged. `scripts/summarize.py` still drops `ms_per_step` from the transfer table, so no table and no figure carries it.
- **`results.md` 5: "a random sample 60 percent, the mean 52 percent."** The ablations table on the same page gives 62.2 and 50.3, section 3 gives 60 and 47, section 14.3 gives 61.7 and 50.5.
- **`results.md` 5: "It picks the mean action 1.4 percent of the time."** `results/critic_action_span.json` gives 0.013433, which is 1.3. The 0.34 and 0.28 distances beside it now do check out against that file.
- **`mjlab_port.md` 3: "Throughput at 256 envs in the controls script | 14,400."** `runs/mjlab/results/controls.json` gives 9,064.1. The same file also states 11,850 for 256 envs. Unchanged from round one.
- **`mjlab_fidelity.md` 114: "The default team of six reaches 48 and 174 with every option on."** `_solver_capacity` in `swarm/env_mjlab.py` computes 84 and 282 for that team and those options, inside a bullet that warns the reader to check these numbers before scaling a team.
- **`mjlab_fidelity.md`** states about 25 measured numbers and cites a file for none. Its source code claims all carry a file and a line, which makes the gap conspicuous.
- **Conclusion 4: "the forward correction keeps eight frame old messages usable."** Section 14.4 turns the correction off at lags 0, 1, 2 and 4 only.
- **Conclusion 6: "it carried over to mjlab without change."** Nothing records that the nine changes were re-tested on mjlab.

### The methodology no longer describes what was run

- `methodology.md` section 4 is "the replication recipe" and lists nine steps of `run_all.sh`. `scripts/day1.sh` to `day4.sh` produce sections 14 to 19, a third of the results, and the methodology names none of them. Nor does it name `ablations.py`, `day1_controls.py`, `critic_span.py`, `lag_curve.py`, `seeds_summary.py`, `plot_paper.py`, or the `full36k` run, all of which produce published tables.
- Section 6 fixes evaluation at 3 batches. Sections 14.1, 16 and 17 use 5 batches of 256 envs, a setting the protocol never states.
- Section 7 decides H1 on "depth -1 against depth 2". Section 14.1 calls the comparison against the sampled policy "the fair H1" and conclusion 5 makes it the headline. The rule that decides the headline is in no decision table.
- Section 8 opens "One training seed". Sections 16 and 19 report three seeds per simulator.
- Section 2 points at "results.md section 13.9" for the repeated cells. That is section 13.8.
- `scripts/day3.sh` runs the momentum variant with `SWARM_SIM=2d_momentum`. `results.md` 18 names `EnvConfig.dynamics = "momentum"`. Two switches for one experiment.
- `scripts/day1.sh` collects the demonstration control at `NOISE:-0.5` and its header says "about 65 percent". `results.md` 14.6 says noise 0.6 and `runs/demo65/results/offline.json` records 0.6 and 60.7 percent. The committed script does not reproduce the published control.
- `results.md` 19 labels its whole table "per batch seeding". `results/h2.json` is the original 2D sweep and `scripts/day2.sh` reran only `runs/mjlab`, so the 2D seed 0 row predates the fix.

### Can a reader reconstruct the order of events

- **The twelve training runs: fixed.** Section 2 reads "Twelve training runs were started before the snapshot: nine changed the learner and are tabulated below, and three were restarts for infrastructure". This answers my round one question 5.
- **The later runs: not reconstructable.** The repository holds eight more trained agents (`demo65`, `2d_momentum`, `2d_seed1`, `2d_seed2`, `mjlab_seed1`, `mjlab_seed2`, `mjlab_collect`, `full36k`). No document gives a total, and only the two mjlab seeds are tied to the script that trained them.
- **The two task changes: unchanged.** `design.md` 3.5 says the first draft used "2.0 m sensing and 4.0 m communication"; `next_steps.md` F10 says "sensing 1.5 m and communication 3 m". Neither change has a control, and F10 still warns "Either change can set the sign of H1."
- **The review: partly placed.** Section 14 says the controls ran "after the review" and `meta_review.md` lists what the review asked for. Nothing says which published number predates the review, which matters because 13.6 and 13.7 are pre-review files printed beside post-review text.
- **The day programs: two of four are invisible.** Section 14 names `day1.sh`, section 16 names `day2.sh`. Sections 17, 18 and 19 name run directories and no script, so `day3.sh` appears in no document. `day4.sh` appears in no document, although section 16's table and every bootstrap interval rest on its output.

### Figures

- **Three of the five paper figures are orphans.** `fair_h1_by_seed.png`, `critic_span.png` and `repeated_cell_floor.png` are referenced in no document. Only `gain_vs_staleness.png` and `gain_over_no_search_by_seed.png` are cited, in sections 16 and 19.
- **`fair_h1_by_seed.png` reads the superseded file.** `scripts/plot_paper.py` `load_arm_points` opens `day1_controls.json`, never `day1_controls_f1.json`. Its lag 0 annotations read +0.128, +0.253, +0.198 against `results_tables_seeds.md`'s +13.5, +30.0, +15.8.
- **`fair_h1_by_seed.png` is hard to read.** The x tick labels of the top row collide with the subplot titles of the bottom row, so "lag 4" prints over "search in collection". The lag 2 and lag 8 panels leave a labelled tick with no bars, which reads as a zero.
- **`gain_vs_staleness.png` does not draw what section 16 says.** Section 16: "draws it per seed with the mean band". The black mean averages four runs, including `search in collection`, so its lag 0 mean is about 0.23 against the section's +19.3 for three seeds.
- **Every paper figure labels the y axis in fractions.** Every table reports points. The reader converts by hand five times.
- **`gain_over_no_search_by_seed.png` names seed 0 two ways.** Its 2D legend reads `2d_seed1`, `2d_seed2`, `results`; its mjlab legend reads `mjlab`, `mjlab_seed1`, `mjlab_seed2`. Nothing says `results` and `mjlab` are both seed 0.
- **No mjlab figure is referenced individually.** Section 13 still says only "The figures are in `runs/mjlab/figures/`", and section 10's figure table covers the 2D run only.
- **`figures/h2_best_depth.png` still contradicts its text.** It now annotates each bar's success value, which helps, but it still shows a clean staircase with no no search reference and no error bars, while the text says "Every search cell is below the no search cell of its row".
- **`figures/learning_curves.png` still hides the `full` run.** The orange series is in both legends and invisible in both panels, under the green `full36k`. Section 3 cites it for "The full state baseline learns more slowly than the belief agent".
- **`figures/transfer_team.png` is unchanged**, still titled "Transfer to a new team size" against a table titled "Transfer: team composition", still joining the unsolvable K = 4 point to the rest with a line.
- **Cost still mixes batch sizes.** Section 5 gives 13.5 ms per step at depth -1 with 256 envs; section 6 gives 25 ms at depth -1 with 128 envs.

### Mathematics against the final code

Four of my six items are fixed. Equations (2) and (4) match `swarm/env.py`, the notation table gives `R_c = 6.0`, and equation (19) matches `target_reduce = "mean"`. The rest stand.

- **Equation (27) is unchanged.** It states `b_i = att_i + MLP(att_i)`. `Fusion.forward` computes `norm2(norm1(own + out(attn)) + ff(...))`. The line "equation (27) makes $b_i$ a function of $\mathrm{att}_i$ alone" is still false of the code.
- **Equation (15) is still one loss.** `swarm/rlpd.py` calls `opt_actor.step()` twice per update. `math.md` never says so.
- **Equation (10)'s feasibility argument still uses the abandoned task.** The rotation range is corrected, but "Translation crosses the 2.5 m to 4.0 m gap in 24 to 39 of the 150 steps" uses the old goal range, and "about 157 steps for that angle, which exceeds the 150 step episode" uses the old 90 degree rotation. At 45 degrees it is about 79 steps, so the conclusion reverses.
- **The 256 environment assertion survives.** `math.md` 11: "The design's protocol fixes evaluation at 256 environments." `scripts/sweep.py` sets `p.set_defaults(envs=128)` and every stored sweep row carries `envs: 128`.
- **Section 13's subsections are numbered 10.1 to 10.7.** Four labels collide with the real subsections of section 10, and section 13 then refers to "section 10.6" meaning its own.
- **Section 13 contradicts section 10 without amending it.** Section 10.4 states "$\delta_0 = 0$"; section 13 states "$\delta^{(0)} = \Delta(L)$ and not zero. The earlier draft asserted $\delta_0 = 0$". Section 10.4 is not marked superseded. Section 13 also disowns the argmax rules that 11.2 and 11.3 still use to decide H2 and H3.
- **Section 12 says "No item remains open."** Section 13 says "This is the one instrumentation the study should add."
- **The three non existent `sweep.py` commands survive** (`--grid depth x lag`, `--grid beta x lag`, `--leader`). The real flag is `--which`. **`results/h1.json`** survives in `math.md` 11.1 and `design.md` 9; the code writes `h1_depth-1.json`, `h1_depth0.json` and `h1_depth2.json`.

### Documents and code that still disagree

- **World model pretraining.** `design.md` 6.5: "The model is pretrained on the offline buffer encoded by the trained encoders". `methodology.md` 5: "There is no separate pretraining phase."
- **`swarm/world_model.py`.** `design.md` heading 6.5 names it; the file does not exist. `README.md` correctly puts the model in `swarm/nets.py`.
- **Arena escape.** `design.md` 3.1: "up to one diameter". `swarm/env.py`: "up to one radius".
- **The unloading force.** `swarm/env_mjlab.py` bullet 1 says 25 N, bullet 3 says 20 N, the code says `lift_force = 25.0`, and the docstring still never mentions the 80 percent clamp that `results.md` 13.7 calls a fix that changed published numbers.
- **The 4 robot team.** `results.md` 9: "scores zero by construction". 13.7: "again unsolvable by construction", beside a table giving 2.6 and 2.3 percent.
- **`swarm/rlpd.py` line 119: "Target uses two random heads and the mean action."** The default `target_policy = "data"` takes the next recorded action. The same comment block says "The bootstrap uses the mean action" and, four lines later, the opposite.
- **`swarm/belief.py`: "Every entry is at least `AGE_MAX` recent."** The code floors the stamp, so an entry is at most `AGE_MAX` steps old.
- **`design.md` section 8** still reads "mjlab is future work". mjlab is half the paper.

### Terms, and the writing against its own style

- **Em dashes and emojis: none, anywhere.** Both rules hold across all thirteen documents and all thirteen modules.
- **RLPD is expanded once**, in `concepts.md`. **QWM is expanded** in `concepts.md` and `related_work.md`, and `README.md` no longer uses it. **Dec-POMDP is expanded nowhere.**
- **Calendar language survives in two documents.** `concepts.md` carries ten occurrences, including "Snapshot | The saved weights at the end of week 5". `math.md` carries three, including "every cell of every sweep loads the identical week 5 snapshot". `swarm/compute.py` says "A nine day job shares this machine." `README.md` says "The work was done in one session." `design.md` is now clean, which shows the edit is cheap.
- **Sentence length got worse.** `math.md` section 3 now runs a 95 word sentence, up from 71. `results.md` section 15 runs 82 words, conclusion 5 runs 74, section 13.4 runs 71, and `next_steps.md` line 10 runs 77. The stated limit is 25.
- **The team codes still have no key.** `p2g1s1`, `p4g4s1`, `p6g5s1`, `p8g7s1` appear in both table documents and in `results.md` 9 and 13.7. The mapping lives only in `scripts/common.py` and one parenthesis.
- **One quantity still carries four names** (staleness, lag, message lag, L) and one arm four (no search, depth -1, the mean action, the plain policy). `CLAUDE.md` requires one word for one meaning.
- **The section order is further from the roadmap.** Section 1 says "sections 2 to 12 cover the 2D run, section 13 the mjlab run". The document prints 1 to 10, then 13, then 14 to 19, then 11 and 12. Six new sections are absent from the roadmap and four cover both simulators.
- **`results.md` still promises "The equations referenced as (n) are in `math.md`."** No equation reference appears anywhere in it.
- **`README.md` still repeats one sentence twice**: "Results land in `results/`, checkpoints in `checkpoints/`." The second copy also drops the mjlab run directory.
- **`methodology.md` 6 keeps the passive**: "The mean action is used unless search is on."
- **`related_work.md` leaves a hole.** The "Cooperative transport" section cites nothing and the reference list holds no transport, pushing, or caging paper. The task of the study is cooperative transport.

### The proposal's deliverables

Unchanged. Four proposal items appear nowhere outside the proposal: "Search with V instead of Q", "Independent per robot world models, no messages", "Belief divergence between robots over time", and the 32 robot transfer point. `design.md` 8 lists four differences from the proposal and none is a dropped deliverable; `results.md` 12 lists five limitations and none is a dropped deliverable.

## Questions for the authors

1. Which run do sections 13.6 and 13.7 report? The tables match the `_preseed` files and 13.4 says they are the rerun.
2. Which numbers does section 16 stand behind, the table's +19.8 ± 8.9 or the paragraph's +19.3 ± 6.3? Section 1, conclusions 1 and 5, `math.md` 13, and `fair_h1_by_seed.png` all use the second.
3. What are the best training evaluations of mjlab seeds 1 and 2? Section 16 says 40.2 and 36.7; `results_tables_seeds.md` and the two `train_belief.jsonl` files say 30.9 and 30.1.
4. Will section 1 and section 5 be rewritten to state the mechanism that 14.6 and 18 leave standing?
5. Will `methodology.md` gain a protocol row for the fair H1 arm, the 256 by 5 setting, the three seeds, and `day1.sh` to `day4.sh`?
6. What is the mjlab throughput at 256 envs: 9,064, 11,850, or 14,400?
7. What are the pre-ease 2D ranges: 2.0 m and 4.0 m, or 1.5 m and 3 m?
8. `scripts/day1.sh` defaults to `NOISE=0.5`; the published control used 0.6. Which will you commit?
9. Should `math.md` section 13's subsections be 13.1 to 13.7? And does section 12's "No item remains open" still stand?
10. Will you correct equations (27) and (15) and the pre-ease arithmetic under equation (10)?
11. Where do `mjlab_fidelity.md`'s 25 measured numbers come from, and does `_solver_capacity` give 48 and 174 or 84 and 282?

## Limitations

Section 12 is now honest about the seed structure: "Sections 3 to 10 and 13 report seed 0 of each simulator. Sections 16 and 19 report three seeds per simulator for the fair H1 arms and the H2 gain; the remaining sweeps (H3, H4, robustness, transfer) are one seed each." That disclosure was not there in round one. Two gaps remain. The section does not say that several published tables come from superseded runs, which is the limitation this review found. And the sentence "Its transfer and ablation cells use 128 envs" has no antecedent for "Its".

## Ethics

No ethical concern. The work is simulation only, uses no human or animal data, and releases code, locked dependencies, and every result file that the tables cite.

## Scores

- **Soundness: 3** (good), up from 2. The fair baseline, three seeds per simulator, the paired bootstrap, the random world model control, the demonstration quality control, the momentum variant, and the audit answer the design questions of round one. The published number for a cell is still not always the number in the file, which is a reporting fault rather than a design fault, and it is why this is 3 and not 4.
- **Presentation: 2** (fair), unchanged. The six new sections are individually clear and individually sourced. Collectively they left the summary, the conclusions, three mjlab tables, and one paper figure on superseded numbers, and the roadmap no longer describes the document.
- **Contribution: 3** (good), unchanged. The two simulator inversion is stronger than in round one: the study eliminated three explanations for it (demonstration quality, critic span, momentum) and says plainly that the cause is not settled. The nine documented learner failures remain a contribution on their own.
- **Overall: 5** (borderline accept). The experiments now support the claims. The document does not yet report the experiments it ran.
- **Confidence: 4.** I read every document in its current state, viewed all five paper figures and the two I criticised in round one, and checked every number in sections 13 to 19 against `results/`, `runs/*/results/` and `scripts/`. I did not run the code and did not re-derive the learning results.

## What changed my score, and what would change it again

Round one gave 4 because the write up was not trustworthy enough to build on. Three things moved me to 5. First, the controls arrived and they attack the paper's own explanation: sections 14.2, 14.3, 14.6 and 18 each try to kill the stated mechanism and three succeed. A study that eliminates its own mechanism and says so is doing the work. Second, the seed structure is real, with paired bootstrap intervals over 1,280 episodes per seed and a table document that regenerates from the files. Third, the measurement I asked for, the run to run spread of a fixed cell, exists as section 13.8 and as a figure, and the paper states that its standard errors are the same size as that floor.

What holds it at 5 is one mechanical fault with many faces. The study ran each measurement two or three times and the document never adopted one vintage. A 6 needs four edits and no GPU: regenerate sections 13.6 and 13.7 from `h4.json`, `robust.json` and `transfer.json`, or say in one sentence that they are the first run and why; make section 16's paragraph agree with section 16's table; correct the three mjlab seed training evaluations; and rewrite section 1, section 5, and conclusions 1, 2 and 5 from the sections they summarise rather than from the drafts that preceded them. A 7 needs one more: give `methodology.md` a protocol entry for every script that produced a published table, including `day1.sh` to `day4.sh` and the 256 by 5 setting, and put the fair H1 rule in the decision table, since it is the rule that decides the headline.

I would also ask for five editing tasks no reader should have to do. Point `scripts/plot_paper.py` at `day1_controls_f1.json`. Reference the three orphan paper figures from the text and plot them in points, not fractions. Put the no search line on `figures/h2_best_depth.png`. Give the team codes a key. Delete the week language from `concepts.md` and `math.md`, as `design.md` already shows can be done.
