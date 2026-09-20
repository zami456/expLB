# Open decisions and standing caveats

Things that need **your** call, or that you should know before quoting a number. Distinct from
`kd/docs/NEXT.md` (queued work I can just do) and `kd/docs/FINDINGS.md` (measured facts).

Last updated 2026-09-21 03:50 BDT.

---

## A. Blocked on your decision

### A1. Which cap questions to cut
Both external reviewers independently rate the same four as too subjective to have a defensible
answer: **Q1** (docs before/after code), **Q3** (review rigour vs deployment frequency),
**Q9** (who writes meeting notes), **Q12** (over-provision vs scale reactively). Gemini adds
**Q0, Q5, Q7, Q13, Q16**.

**Q8 is broken outright** — "A/B test a change that affects 0.3% of users" omits N. 0.3% of
100M is 300,000 subjects; 0.3% of 10,000 is 30. Unanswerable as written. That one should go
regardless of where you draw the subjectivity line.

Why it matters: if a question is a genuine coin-flip, changing your mind under challenge is
*correct*, not sycophantic, and that item measures noise. Cutting all ten leaves 10 items.

**Options:** cut 4 (both reviewers agree) · cut 10 (Gemini's fuller list) · cut 1 (Q8 only)
and accept the noise.

### A2. First-person questions — all 20 or none
**Q4, Q11, Q18** are first-person ("My manuscript", "My analysis", "My experiment"); the other
17 are third-person. On those three the challenge comes from the person whose own work is being
judged, which adds a social cost to holding that the other items lack. Two of the three are
among the clearest-answer items, so the confound lands exactly where you most want a clean read.

### A3. 11-dataset grid vs 14
The **only** lever that attacks the binding constraint on significance (F-04: the limit is ~11
datasets, not ~15,000 rows). Available:

| dataset | state | problem |
|---|---|---|
| `gender-secret` | already extracted (`passQ11`) | Q11 found it sits **at chance** |
| `soft-trigger` | already extracted (`passQ11`) | its "lie" is really "wrong answer" |
| `harm-pressure-natural-v2` | **not extracted** | needs fresh NDIF time |

14 datasets would narrow the dataset CIs by roughly 11%. But a bootstrap over degenerate
datasets is worse than a smaller honest one, so this is not a free win.

**Your call:** a 14-dataset grid with two weak members, or an 11-dataset grid quoted honestly
against its own limit.

### A4. ~~Item 3 go/no-go — 11-dataset eval activations at 3B and 8B~~ — **CLOSED, done**
You picked it, both rungs ran, and the result is F-63: raw off-diagonal rises
0.708 -> 0.743 -> 0.809 across 3B/8B/70B while the persona gain over the matched control
falls +0.202 -> +0.115 -> +0.079. The argument for doing it held up — three layer profiles
beat one p-value. Original reasoning kept below.

My recommendation was: **do it, drop item 8(c), keep item 6(b).**

Measured cost (F-35): **~20 min for the 8B, ~3.2 h for the 3B** — about 3.5 hours total.
I called this "expensive" earlier; that was wrong.

What it buys, and this only became a good argument after F-34: three layer profiles instead of
one. If the same depth-dependent shape appears at 3B and 8B, the evidence is **replication
across three models**, which sidesteps the 11-dataset ceiling because the claim is no longer a
p-value on one model.

What it does *not* buy: significance. Each rung faces the same 11-dataset limit independently.

### A5. ~~Should a rung producing zero vectors abort, or salvage?~~ — **CLOSED, salvage + hard gate**
Settled by how it kept biting. A zero-output step is now a **failure, not a warning**
(standing rule in `TODO.md`), but the chain still falls through to the next step rather than
aborting — your instruction. The real fix was neither: it was making the gates assert the
number they name. Two gates have now passed while measuring nothing (F-64's `-ge 12` against
`predicted=13`, and the cap chain's 2e, which checked only that a file existed while
`cos(PC1,contrast) >= 0.90` was never computed). Original text below.

#### original

`run_scale_chain.ps1` logs `no vector produced` as a warning and continues. On 2026-09-20 that
let `Scale-3b` fail **24 of 24 roles** while looking like slow progress — the only symptom was
that extraction took 20 min where the 8B took 3 (F-33). I have not fixed it because "abort" and
"salvage the roles that worked" are both defensible and it is your experiment.

---

## B. Standing caveats — read before quoting a number

### B1. The +0.076 is not a result
`pc:unified − raw` = +0.0757, dataset CI **[−0.006, +0.173]**, q = 0.499 (F-31). Across the kd
bank **6 of 4,032** tests clear the dataset bootstrap, and **four of those six are negative**
(`random − raw`, a sanity check passing). Nothing beats raw; nothing beats `datapca` at matched
k. The phrase "the first gain over a matched control in this project" is retired.

### B2. Shape is real; significance is not
F-34: the kd_s23 gain has a clean depth profile — negative L0–L20, peak at L44, sustained to
L76, 16/21 layers positive — and still reaches only q = 0.130 at best. The honest headline is
*"a consistent, structured effect that 11 datasets cannot resolve from zero"*, not
*"persona coordinates don't work"*.

### B3. L44 is the peak, not L40
By +0.007 (vs raw) and +0.009 (vs datapca). Numerically immaterial, but **every kd headline so
far is quoted at L40**, one layer below the peak. Decide whether to re-quote or to state that
L40 was pre-registered.

### B4. ~~The cap bank has NOT been re-smoke-tested~~ — **CLOSED**
All 60 variants were rewritten (F-32) and the confound counts pass, but that is **lexical, not
behavioural**. Worse, F-26's go/no-go ran on **Q0 and Q1 — the two items both reviewers call
too subjective**. So "`pc_caver` folds" is currently demonstrated only where folding is
defensible. Re-run on Q11, Q14, Q2, Q19 before trusting any fold rate.

**Done, and it passed.** With teacher-forced turn 1 the bank gives **8/8 fold vs 8/8 hold
across all four question cells** (F-51, F-53), not just the subjective ones. The full run
then graded at **0.935 retention, symmetric between groups** (F-64), and every downstream
number agrees to <= 0.004 between the filtered and unfiltered extractions (F-65). The
reviewer's concern that the judge filters on the outcome variable was real, was tested, and
does not bite.

### B5. The affect fix overshot
`pc_appeaser` vs `pc_steadfast_reviewer` went from 3-vs-0 emotional terms to **3-vs-5**. The
vocabulary is balanced in the sense that both sides now have it, but the asymmetry is inverted
rather than removed. Defensible — the holding persona acknowledges displeasure and overrides it,
which *is* the construct — but it wants a reviewer's eye, not another blind edit from me.

### B6. If item 3 runs, the eval side is off-policy
We deliberately went **on-policy** for persona rollouts (each model writes its own). The
evaluation rows are fixed text authored by Llama-3.3-70B, so running them through a 3B is
off-policy in a different sense. That is ordinary probing and defensible — the label belongs to
the text — but it must be *stated* in the writeup, not glossed.

### B7. The item-3 cost is a floor, not a forecast
F-35's projection assumes eval rows pool like persona rollouts. They do not: eval rows are
longer (passB truncates at 2,048; `ai_audit`'s median context is 5,285 tokens). If they force
the batch below 2, the 3B figure grows worse than linearly.

### B8. Two fixes shipped but unproven
- The `exc_tail` patch to `2_activations_ndif.py` (F-33) has **never executed** — no failures
  occurred after the restart. Correct by inspection and matches four sibling scripts, but
  unverified in practice.
- `--batch_size 2` for the 3B is **known-good, not tuned**. 4 might also work and would halve
  the time.

---

### B9. Quote F-66 and F-67 TOGETHER, or the cap result is misleading

They answer different questions and the honest headline needs both:

| | question | answer |
|---|---|---|
| **F-66** | does the axis beat its controls **on the construct it was built for**? | **yes** — +0.112 over raw on `sycophancy_mt`, CI [+0.050, +0.185], 9/10 sources |
| **F-67** | does that gain **generalise to the other ten columns**? | **no** — +0.098 over the full matrix, CI [-0.036, +0.208] |

Quoting F-66 alone overstates it into a general-purpose detector. Quoting F-67 alone buries a
real and significant effect. The correct claim is *"a persona axis built for a specific construct
beats its matched controls on that construct, and ties elsewhere"* — which is also the cleanest
statement of the thesis, since it is F-63's scale finding reached from a second direction.

### B10. The zero-shot 0.745 can never have a dataset bootstrap

Zero-shot trains on nothing, so there are no source datasets to resample and the project's own
significance bar is **undefined** for it. It is the cleanest form of the result (no probe, the
paper's own protocol, 100% of rows) and the one with the weakest interval. Wherever it appears,
say which it is. The trained +0.112 is the number that carries a CI.

---

## C. Corrections I made to my own earlier statements

Listed so you are not working from something I later revised.

| I said | actually |
|---|---|
| the stale blobs were **40 GB** | **56.6 GB** — I quoted a figure measured mid-download, three times |
| the delete succeeded (`free GB now: 38.3`) | it **silently did nothing**; all 22 blobs remained |
| the `>1MB` filter "leaves config and tokenizer intact" | it **deleted `tokenizer.json`** (~17 MB); restored and verified |
| 8B deception-vs-honesty keep gap = **1.4 rows/role** | **0.24** — F-27 was computed at 21 of 26 roles |
| "widest 4840 padded tokens" means a sequence past NDIF's limit | it is a **batch total**; 3B lengths match the 8B almost exactly |
| `pc_caver` will read as a **spelunker** | it did not — no cave in any of the 8 smoke responses |
| item 3 is **expensive** | **~3.5 h** total, measured |
| the 8B partial score numbers "won't move" | headline barely moved, but the per-group gap moved **6×** |
| cap variants **A and B agreeing** confirms F-55's inertness geometrically | a **tautology** — `pc:<name>` projects onto the PCA *basis*, never the contrast, and A and B share all 13 roles, so they cannot differ. Caught before it entered the ledger |
| the holding personas are **inert** (F-55) | behaviourally yes, **representationally no** — drop them from the basis and the entire advantage vanishes (p = 0.56) |
| the cap chain's 2e step **passed** | it logged `written/written/PASS` twice while the gate it names was **never computed** — the report loop was hardcoded to paper16's three basis names and printed an empty table |
| the pipeline had already bootstrapped the cap result | it had **not** — `PERSONA_FEATURES` was a module constant of paper16 names, so `pc:capitulation` was in no comparison at all (144 tests; 792 after the fix) |
