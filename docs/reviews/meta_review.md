# Meta review, second round

Three reviewers read the repository again after the first round's fixes and the three day
programs (`results.md` sections 14 to 19). Each began from its first round review and checked
every objection. The first round text is in git history (commit `a7214f9` and before). The
project is for learning, not publication; the NeurIPS format is used because it is a hard,
familiar standard, not because a submission is intended.

## Scores

| Reviewer | Lens | Soundness | Presentation | Contribution | Overall | Confidence | Round one overall |
|---|---|---|---|---|---|---|---|
| 1 | Methods and correctness | 3 | 2 | 3 | 5, borderline accept | 4 | 4 |
| 2 | Significance and positioning | 3 | 3 | 3 | 5, borderline accept | 4 | 3 |
| 3 | Clarity and consistency | 3 | 2 | 3 | 5, borderline | 4 | 4 |

All three moved up one to two points. The consensus: the headline is now supported at the level
a paper needs (three seeds, paired episodes, intervals), the negative controls are real and
honest, and the write up still contradicts itself in places because it grew section by section.

## What the second round settled

- **The fair H1 holds.** Reviewer 1 recomputed all fifteen paired bootstrap intervals from the
  per episode files and they match. Reviewer 2 accepts that the search clears the sampled policy
  and a random candidate on three seeds at fresh to one frame old information.
- **The mechanism is open, and now the text says so.** Three explanations were measured and
  eliminated; the remaining candidates (the contact model, the servo dynamics) are named as
  untested.
- **The gain curve reads differently than first written.** Reviewer 2 showed that over lags 0 to
  2 the gain falls mostly because the baseline rises with better forward correction, while the
  search arm itself falls only after lag 2. `results.md` section 16 now decomposes the curve.
- **Leader search is refuted without the confound**, on both simulators.

## What remains, in order of weight

1. **The cylinder pusher artefact** (reviewer 2). A cylinder gets one contact point against the
   box and spins it far more than a box pusher would, on a task whose success test includes the
   angle. Every result uses cylinders. The box option exists (`mjlab_fidelity.md`); rerunning the
   fair H1 arms with it on one seed costs about 12 GPU minutes plus a retrain if the policy does
   not transfer.
2. **One seed for H3, H4, robustness, and transfer**, and the 2 point reproducibility floor
   measured once on mjlab. The floor is now restated where sections 14 to 19 depend on it.
3. **The document grew by accretion** (reviewer 3: 20 of 67 first round discrepancies fixed, 7
   partly; the five listed in the second round are fixed in commit `b5ca036` and after). A
   reader can reconstruct the order of events from `methodology.md` section 5b and
   `results.md` section 2, but the summary table and the conclusions have been rewritten four
   times and read like it.
4. **Related work is thin on delay and communication literature** (reviewer 2); the novelty list
   was corrected to drop the retracted critic span mechanism.
5. **The mjlab robot speed (2.5 against 1.5 m/s) is documented and not controlled.**

## What the meta reviewer would say to the authors

For a learning project this is a good stopping point: the pipeline is end to end and resumable
on two simulators, the claims are sized to their evidence, the failures are recorded as results,
and three adversarial rounds have run against it. The work that would move it further is
specific and cheap in GPU time (items 1 and 2 above, about two hours) but the returns are
diminishing; the one experiment that could still change the story is the box pusher rerun,
because it tests whether the mjlab positive rests on a contact artefact.
