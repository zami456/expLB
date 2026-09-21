# What still has to run — a roadmap to a defensible thesis

Written 2026-09-21 after the F-64 → F-84 run. This is the honest list: every open caveat, what
run closes it, what it costs, and what happens if we skip it.

**Where we stand.** Three claims survive the dataset-level bootstrap. That is already more than
most of this project had a week ago, and two of the three have layer profiles and seed checks.
The gap between *what we have* and *a defensible thesis* is narrower than it feels — but it is
not zero, and the items below are not decoration.

---

## Tier 0 — Run these or the thesis has a hole a reviewer will find

Four runs. Everything here is CPU-only except the last.

### 0.1 ~~Pooled training on the collaborator's six datasets~~ → **done, and reframed by F-97**
**Status 2026-09-22:** run (F-85), then **superseded as a headline by F-97**.
**What changed:** the abstract's 0.916 is not the persona-coordinates contrast — it is
`ext_axis1`, the 12v13 axis built on **our** knowledge-denial personas. The paper's own 8v7
contrast reads **0.7909 (L20) / 0.8319 (L40)** zero-shot on their six, against their own
single-source baseline of 0.6478 / 0.7320 — a margin of **+0.143 / +0.100**, not +0.256 / +0.184.
Roughly **44–46% of the headline margin is the kd bank**. That comparison is entirely inside
their committed files, so it needs no pooled probe to make the point.
**Also:** our six ≠ their six (`repe_honesty` is not in our eleven), so F-85's +0.023 mixed two
corpora and is withdrawn as a headline.
**What a reviewer will still ask:** why the single-source *mean* and not the median (0.722) or
the sign-oriented mean (0.735). Keep that pre-empted.
**Remaining work:** run their `axis1` and `ext_axis1` through our crossgen on our eleven — the
one comparison neither side has. ~10 minutes, CPU.

### 0.2 Seed robustness on the cap result
**Closes:** "single seed" — a caveat on F-65, F-66, F-67 and F-77.
**Why:** pooled training is seed-checked (five seeds, 0.173–0.198). The cap result, which is
Claim 2 and the only positive evidence that persona axes *can* work, is not. It would be odd to
hold our own strongest negative to a higher standard than our strongest positive.
**Cost:** ~20 minutes, CPU. Same shape as the pooled seed run.
**If skipped:** Claim 2 rests on one split.

### 0.3 Read the three papers we cite second-hand
**Closes:** F-78's caveat (3) and F-70's caveat (3).
**Which:** `arXiv:2609.02893` (Yoo & Skapars, subspace selection — the nearest competitor and it
argues the *opposite* on ceiling effects), `arXiv:2605.27958` (Kumar, concurrent, overlaps three
of our claims), `arXiv:2511.17408` (Kirch/Skapars, off-policy — the objection to our
teacher-forcing, and a co-author of the paper we replicate).
**Cost:** reading time, no compute.
**If skipped:** we write "how we differ" paragraphs from reviewer summaries rather than from the
papers. One reviewer already flagged that Skapars co-authors both 2511.17408 and our source
paper — the overlap will be noticed.

### 0.4 Format-matched persona control
**Closes:** the deflationary reading of Claim 2 — that the inert holding personas contribute a
representation of *the instruction*, not of a disposition.
**Status:** both banks are built and committed (`cap/personas/cap_instructions_neutral/` and
`.../paraphrase/`, 6 roles × 5 variants each, length-matched to 1.4 and 5.6 characters).
Generation has not run.
**Cost:** ~1 h GPU per set, plus judge and extraction. Two sets.
**If skipped:** a reviewer says "your inert personas encode prompt format" and we have no answer.
One reviewer already named this as the experiment that could carry a main-track short paper.

---

## Tier 1 — Needed for a *strong* thesis, not merely a defensible one

### 1.1 Sarcasm bank
**Closes:** every test of Claim 1 is currently *negative* — the axis fails where register and
construct come apart. A sarcasm axis is the first **positive** test: build the confound and show
it explains the failure.
**Why sarcasm:** it is *saying what you don't mean without intending to mislead* — deception
minus intent. If a sarcasm direction has high cosine with the deception axis, that axis is
substantially a meaning-mismatch detector, not an intent detector.
**The catch:** a naive bank produces obviously-marked text (*"oh, brilliant"*) and the axis
becomes lexical — worthless, for the same reason the knowledge-denial bank needed three groups.
Sarcastic and sincere personas must say **near-identical words**, separated by context.
**Cost:** design is the bottleneck (a day of careful authoring), then ~1 h GPU + judge + extract
+ crossgen.
**Honest status:** after Claim 3, this supports a diagnosis rather than carrying a headline.

### 1.2 Base vs chat on sycophancy
**Closes:** Ying et al. report post-training pushes sycophancy into its own subspace (70B
0.67 → 0.56 base to chat). We only ever ran the Instruct model, so we cannot confirm or use their
mechanism on our own corpus.
**Cost:** a base-model extraction run. Moderate GPU.
**Payoff:** converts our "the axis encodes register" from an assertion into a claim with a
representational account, on our own data.

### 1.3 Why does pooled training work?
**Closes:** F-80 caveat (4) — we show pooling works and say nothing about the mechanism. Three
live hypotheses: domain-general directions exist; heterogeneous data regularises; pooling averages
away dataset-specific shortcuts.
**How:** compare the pooled direction against each single-source direction; check whether pooled
performance tracks the *number* of sources or their *diversity*; ablate sources one at a time.
**Cost:** CPU, a few hours.
**Why it matters:** right now our strongest claim is an observation. A mechanism makes it a
contribution.

### 1.4 Head-level extraction, or an honest limitation
**Closes:** F-78's open question. Choi et al. find truthfulness is carried by a **sparse subset of
attention heads**; we mean-pool across all 64. So *"the effect is not there"* and *"the effect is
there and we are averaging it away"* both fit our data, and we cannot currently tell them apart.
**Cost:** expensive — a per-head extraction is a different pipeline.
**Cheaper alternative:** state it as a limitation with the citation. That is defensible; claiming
the effect is absent without it is not.

---

## Tier 2 — Would strengthen, safe to skip

| item | closes | cost |
|---|---|---|
| Paper16's 26 personas on kd's 30 questions | F-61's three-way confound (persona set / question set / rollout count all vary at once) | ~1 h GPU |
| Re-extract HP-KR into passA/passC for the *other* datasets | the asymmetric-grid caveat: HP-KR is full, the other ten are 200-row subsamples | ~2 h GPU |
| 14-dataset grid | the binding constraint on significance is ~11 datasets, not rows | needs 3 new extractions, and two candidates are degenerate |
| Explain the Mahalanobis non-reproduction | we tested both covariance conventions and neither gives their R² = 0.98 | unknown — may need their code |
| L44 vs L40 re-quote | B3: every kd headline is quoted at L40, L44 is the peak by +0.007 | CPU, minutes |

---

## What is already closed — do not redo these

| caveat | closed by |
|---|---|
| single layer, kd matched slice | F-72, F-75, F-76 — 80-layer profile |
| single layer, cap | F-77, F-82 — 21/21 layers, both filters |
| single layer, pooled training | F-81 — 21/21 layers, all clearing |
| single seed, pooled training | F-83 — five seeds, 0.173–0.198 |
| no dataset bootstrap | F-66, F-76, F-80 — run on all three claims |
| judge filter drives the cap result | F-62, F-64, F-65, F-67, F-82 — four independent checks, ≤ 0.004 |
| the layer was chosen post hoc | pre-registration + amendment: L20/22/40 are literature, L44 is ours |
| all three pre-registered tests | F-76 (×2) and F-84 (cluster permutation, p = 0.38) |
| matched-slice unusable on passA/passC | F-69, F-71 — 258 rows at 124/134 on all three stores |

---

## Realistic expectation

**If Tier 0 runs and nothing breaks:** a defensible thesis with three measured claims, each
surviving a dataset-level bootstrap, a layer profile and a seed check, plus an honest account of
four things we withdrew. That is a workshop paper (BlackboxNLP) or TMLR, which is what both
external reviewers independently suggested.

**If Tier 1 also runs:** the sarcasm bank turns Claim 1 from a diagnosis into a demonstration,
and a mechanism for Claim 3 turns an observation into a contribution. That is the version worth
trying a main track with.

**The single biggest risk is not an experiment.** It is that Claim 3 — pooled training beats
persona structure — is the most important thing here, and it came from a paper we read at 2pm
today. Nobody in this project proposed it, and two rounds of expert review with web search did
not either. The design we inherited made it invisible. **What else is invisible for the same
reason?** The pooled-training cell existed as an obvious control the whole time; we did not see
it because the paper we were replicating does not run it. That is worth half a day of
deliberately asking "what comparison does our inherited design make impossible to see?"

---

## Sequencing

```
now        0.1 pooled on the six            10 min  CPU   <- settles the abstract
           0.2 cap seeds                    20 min  CPU
then       0.4 format-matched control        2 h    GPU   <- banks already built
alongside  0.3 read the three papers         --     none
then       1.1 sarcasm bank                  1 day design + 2 h GPU
           1.3 why pooling works             3 h    CPU
optional   1.2 base vs chat                  half day GPU
           1.4 head-level, or cite as a limitation
```

Tier 0 is roughly **one working day** including the reading. Tier 1 adds about three.
