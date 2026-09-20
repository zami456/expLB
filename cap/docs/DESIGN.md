# `cap/` — the capitulation bank

**One sentence:** a persona bank built to vary *capitulation* rather than *register*, in order to
test constructively whether F-13's diagnosis of the persona-coordinates method is right.

**It is.** The capitulation axis reads **0.745 zero-shot** on `sycophancy_mt` where the paper16
sycophancy axis reads **0.374**, and the trained version clears a dataset-level bootstrap against
`raw` and both matched controls (F-65, F-66). This is the only place in the project where a
persona axis beats everything it is matched against.

---

## Why the bank exists

The paper16 **sycophancy** axis has the cleanest geometry anything in this project has produced:
`cos(PC1, own contrast) = 0.993`, permutation p = 0.0079 over all 252 splits. It is, by every
internal diagnostic, a well-formed persona axis.

It reads **0.374** on `sycophancy_mt` — below chance, and worse than a random direction (0.541).

F-13's diagnosis: the axis separates **agreeable-vs-challenging register** — tone, hedging,
warmth — while the dataset labels **capitulation** — abandoning a correct answer under pressure.
Those are different constructs that correlate in ordinary text and come apart exactly where the
dataset lives. The same diagnosis explains HP-KR (F-03, 0.300) and is the project's strongest
single thread.

A diagnosis you can only state is a critique. A diagnosis you can *act on* is a finding. So:
hold the model, judge, grid, seed and layers fixed, change **only which construct the personas
vary**, and see whether the number moves. That is the whole design.

---

## The bank

**13 roles**: 6 capitulation, 6 position-holding, 1 `default` anchor.

| group | roles |
|---|---|
| `capitulation` | `pc_caver` `pc_deferrer` `pc_appeaser` `pc_second_guesser` `pc_consensus_follower` `pc_conflict_avoider` |
| `position_holding` | `pc_position_holder` `pc_evidence_bound` `pc_steadfast_reviewer` `pc_reasoned_defender` `pc_calibrated_holder` `pc_disagreement_tolerant` |
| `assistant_anchor` | `default` |

**The anchor is not optional.** Omitting `default` from the chain's role list on 2026-09-21 cost
the entire axes step: `6_axes.py` needs it for the Assistant Axis, and variants B and C are
contrasts *against* it, so all three became unbuildable. The chain still reported 12 vectors as a
PASS, because the gate printed `predicted=13` and asserted `-ge 12` (F-64).

**20 questions** (`questions/cap_questions_v3.jsonl`), balanced 5/5/5/5 across framing × kind,
with misconception answers split 5 Yes / 5 No so that "always agree" and "always disagree" are
equally wrong. **5 variants × 20 questions × 13 roles = 1,300 rollouts.**

---

## Three controls, each of which changed the design

These are the reason the bank looks the way it does. Each was measured, and each overturned the
obvious first attempt.

### 1. Teacher-forced turn 1 (F-51, F-53) — *letting personas write turn 1 failed twice*

The construct needs two turns: state a position, then get challenged. Letting each persona
generate turn 1 was tried and failed **twice**:

- **F-26**: the capitulation personas *hedged before any challenge arrived*. They never stated a
  position, so there was nothing to abandon.
- **F-46**: worse — they gave outright **wrong** answers in turn 1 and then "folded" into
  correctness under challenge, which is the opposite of the construct.

**Fix:** prefill turn 1 identically for every persona, generate only turn 2. This makes the
context **byte-identical** across all thirteen roles — the only varying text in the prompt is the
system prompt, and the only generated text is the thing being measured. It also halves the cost
(1,300 generations, not 2,600). Turn 1 is a **bare assertion** — no confidence marker, no
reasoning — so nothing in it cues either behaviour. Result: 8/8 fold vs 8/8 hold across all four
question cells.

### 2. Wrong turn 1 (F-52) — *the contrast only exists on TRUE answers*

`CONTROL_WRONG = {10: "Sydney", 11: "1", 13: "P", 14: "Toronto"}` prefills turn 1 with a wrong
answer. **Both sides correct it.** Holding personas do not defend a falsehood, and capitulation
personas do not need pressure to abandon one.

So the measurable construct is not "changes their answer". It is **abandoning a *correct* answer
under social pressure**, and the bank's name and every claim made from it have to say so. The
control runs as a separate 13-file set under `run-cap/control/`.

### 3. No system prompt at all (F-55) — *the holding half is behaviourally inert*

A bare model with no system prompt **holds identically** to the position-holding personas,
producing nearly the same text. Behaviourally, `position_holding` ≈ `default`.

This is the first direct evidence for why PC1 keeps coming out as the Assistant Axis across this
whole project: half of a typical persona contrast is just *the assistant being itself*.

It also made "which contrast is the real one" an open question, which is why there are three
axis variants rather than one.

**And it was half wrong.** F-66 measured the other half: those personas are behaviourally inert
and **representationally essential**. Remove them from the PCA basis and the entire advantage
disappears (`pc:cap_only@2` is indistinguishable from raw, p = 0.56). Behavioural inertness and
representational contribution are different properties.

---

## Three axis variants

All three read the **same rollouts**, so the extra cost is CPU only. Declared in
`personas/cap_role_groups.json` under `contrasts`, which `6_axes.py` reads.

| variant | contrast | PCA basis | intended to isolate |
|---|---|---|---|
| **A** `capitulation` | capitulation − position_holding | 13 roles | the bank as designed |
| **B** `cap_vs_default` | capitulation − `default` | 13 roles | the **contrast** |
| **C** `cap_only` | capitulation − `default` | 7 roles | the **basis** |

The optional `basis` key on a contrast spec was added for exactly this: without it the basis is
always `pos + neg + anchor`, which ties contrast and basis together and makes B-vs-C unaskable.

### ⚠️ A and B are identical BY CONSTRUCTION, not by measurement

`make_features` builds `pc:<name>` from `bank["pca"][name]["components"]` — the top-k PCs of the
named **basis**. It never touches the contrast. A and B are built over the **same 13 roles**, so
they are the same subspace; their per-PC signs differ (each is pinned to its own contrast) but a
linear probe is invariant to a per-feature sign. **Their identical AUROCs are a tautology.**

This nearly went into the record as geometric confirmation of F-55. The plan's "A vs B isolates
the contrast" was not achievable with `pc:` features at all — the contrast appears only in the
zero-shot rows, where just one direction is scored. `6_axes.py` now prints a `NOTE` whenever two
bases share a role set, so the next reader cannot make the same mistake.

The real A-vs-B number is `cos(contrast_A, contrast_B) = 0.931` — close to, but not, the 1.000
that true inertness would imply.

**B vs C is the one live comparison**, and it is the bank's most interesting result.

---

## Pipeline

```
scripts/cap_personas.py          build the 13 persona files + role groups
scripts/gen_multiturn_ndif.py    teacher-forced generation (turn 2 only), --control for wrong-turn-1
src/…/3_judge_ndif.py            Llama-3.1-70B, held constant with every other bank
src/…/2_activations_ndif.py      vectors, run TWICE: --keep_scores 2,3 and 0,1,2,3
src/…/6_axes.py                  all three variants from one run
src/…/27_crossgen.py             the 11-dataset grid, layers 20/40/44
scripts/cap_significance.py      paired dataset-level bootstrap  <- the bar
scripts/export_cap_fig.py        compact payload for fig5 (matrices are 455 KB; scp truncates at 200 KiB)
```

Driven end to end by `scripts/run_cap_chain.ps1`, registered as the **`Cap-Chain`** Scheduled Task
with an `AtStartup` trigger so an unprompted lab-PC reboot resumes it.

### Two rules this pipeline pays for

**Control rollouts live OUTSIDE `responses/`.** `2_activations_ndif.py` globs `responses/*.jsonl`,
so a control file sitting there is extracted as another persona — `default_control` became a 14th
vector on 2026-09-21 and would have entered both the PCA basis and the role side of the Assistant
Axis.

**Filtered and unfiltered both, always.** The judge scores "stayed in character", which means
*folded* for a capitulation persona and *held* for a holding one — so dropping 0s and 1s removes
exactly the rollouts that disagree with the hypothesis, on **both** sides. If the axis only
appeared after filtering, the filter would be the finding.

It does not. Retention is **0.935** and symmetric between groups (F-64), and every downstream
number agrees to **≤ 0.004** between `s23` and `all` (F-65). The reviewer's concern was real,
was tested, and does not bite here.

---

## Results

See **F-64** (generation and judging), **F-65** (the number), **F-66** (significance), and
`figures/thesis/fig5_capitulation.png`.

| | value |
|---|---|
| zero-shot, cap contrast, L40 | **0.745** (L20 0.759 · L44 0.728) |
| zero-shot, register axis, L40 | 0.374 |
| trained `pc:capitulation@2`, L40 | **0.714** |
| `raw` | 0.601 |
| `datapca@2` / `random@2` / `length` | 0.559 / 0.497 / 0.582 |
| vs raw, dataset bootstrap | **+0.112, 95% CI [+0.050, +0.185]**, 9/10 sources |
| `pc:cap_only@2` vs raw | −0.008, p = 0.56 — **no advantage without the holding personas** |

**The mechanism is stability, not height.** `cap@2` sits in 0.71–0.74 for eight of ten sources —
it barely notices which dataset trained it. `raw` ranges 0.353 → 0.748 and is *below chance* from
`roleplaying` and `convincing-game`. Where raw already works the axis buys nothing; where raw
collapses, the axis holds. That is the persona-coordinates paper's own central claim, reproduced
here with a matched control and a dataset-level test — neither of which the paper runs. It is
also the shape F-63 predicted from the other direction: the persona advantage is inversely
proportional to how good the baseline already is.

## Standing caveats

1. **The gain is LOCAL (F-67).** `sycophancy_mt` is the dataset this bank was designed against,
   and that is the only place the advantage over raw holds. On the full 11×11 off-diagonal mean
   the margin over raw is **+0.098, CI [−0.036, +0.208] — does not clear**; dropping
   `sycophancy_mt` leaves +0.073, also not clearing. It beats `random@2` everywhere (+0.270,
   q = 0.013), but `random@2` is itself significantly *worse* than raw (−0.171), so that
   comparison mostly says the axis is not noise. This is a targeted fix for one construct, not a
   general-purpose detector, and must never be quoted as one.
2. The trained basis *is* this bank's Assistant-Axis basis (`capitulation == cap_vs_default ==
   all`, `cos(PC1, assistant) = −0.838`), so the trained number rides an Assistant-Axis-shaped
   PC1 and lands on top of Q10's 0.71. The **zero-shot** number is the cleaner claim, because its
   direction is genuinely capitulation-specific (`cos(PC1, contrast) = 0.987`).
3. The zero-shot 0.745 has **no dataset-level bootstrap** — zero-shot trains on nothing, so there
   are no sources to resample and the test is undefined for it.
4. At **L20** the margins over `datapca@2` and `cap_only@2` do not clear. The headline is an
   L40/L44 result.
5. 20 questions, 12 personas, 5 variants, single split seed (42). A small bank.
