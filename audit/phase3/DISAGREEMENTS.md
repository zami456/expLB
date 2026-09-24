# Phase 3, workstream P3-B — where the owner's and the collaborator's results disagree

**Question:** where do her numbers and his numbers differ on the same or a similar setting, and why?

**Scope:** branch `asix` (hers) and `origin/feat/persona-pc-ood` (his), read with `git show` only.
His activations are not committed and never will be, so anything that needs them is marked
**NOT RECOMPUTABLE** and checked instead against his committed CSVs.

**Method.** I did not compare prose. I put his per-dataset zero-shot AUROCs (committed
`zero_shot.csv` on his branch) next to hers (the `zeroshot` block of her crossgen matrix JSONs),
cell by cell, matching on **dataset, layer, pooling, persona bank and arm definition**. Scripts:
`audit/scripts/phase3/p3b_zs_extract.py` (her local stores), `p3b_lab_zs.ps1` / `p3b_lab_zs2.ps1`
(her lab-only stores, read-only), `p3b_headtohead.py`, `p3b_bankmatch.py`, `p3b_baseline.py`,
`p3b_signflip.py`.

**Terms, first use.** *Zero-shot AUROC* = score every response by its dot product with one fixed
direction, then measure how well that ranking separates lies from honest answers; nothing is
trained. *Off-diagonal mean* = train a probe on one dataset, test on the others, average.
*Bank* = the file of persona vectors a direction is built from. *s3 / s23* = her two judge filters
on the persona rollouts (keep score-3 only, vs keep score-2 and 3); s23 has more rollouts per role.

---

## 0. The headline, and it is good news

**What was tested.** Whether their two independent pipelines — her `27_crossgen.py` on NDIF-extracted
activations, his `persona_ood.py` on the Liars' Bench probe repo's activation cache — give the same
answer when they use the same persona bank on the same evaluation rows.

**What happened.** They agree almost exactly.

| both used this bank | cells | mean \|gap\| | max \|gap\| |
|---|---:|---:|---:|
| 26-persona shared-question bank, 8v7 deception axis (L20, L40) | 10 | **0.0015** | 0.0090 |
| 36-persona kd bank, **s3**, 8v7 deception axis (L20, L40, L44) | 15 | **0.0015** | 0.0107 |
| 36-persona kd bank, **s23**, 8v7 deception axis | 15 | **0.0017** | 0.0110 |
| 36-persona kd bank, **s23**, 12v13 extended axis | 15 | **0.0015** | 0.0175 |

The evaluation rows are provably identical: her `run-crossgen/kd11/splits.json` and his
`zero_shot.csv` give the same `n` and the same positive count for all five shared datasets
(instructed-deception 5432/3045, harm-pressure-choice 2134/134, harm-pressure-knowledge-report
2139/139, convincing-game 267/167, insider-trading 1080/423).

**How sure: VERIFIED, HIGH.** 55 matched cells, both sides read from raw files during this audit.

**Why it matters.** It removes the largest worry. **No disagreement below is a bug in either
pipeline, in `persona_ood.py`, `persona_pcs.py`, `pool_activations.py` or `27_crossgen.py`.** Every
disagreement I found is a *bank*, an *arm definition*, a *dataset set* or a *training protocol* —
all of them nameable, and most of them fixable by relabelling a sentence.

**Evidence.** `audit/scripts/phase3/p3b_bankmatch.py` (run as
`python audit/scripts/phase3/p3b_bankmatch.py <scratch> .`);
his `src/probes/deception-detection/results/persona_ood{,_L40,_ext_L20,_ext_L40,_ext_L44,_mod_L20,_mod_L40,_mod_L44}/zero_shot.csv`
on `origin/feat/persona-pc-ood`; her `run-crossgen/kd11/banks/{kd,kd_s23}/matrices/mean_L*_s42.json`,
`run-crossgen/banks/paper16_shared/matrices/mean_L*_s42.json` (lab PC), `run-crossgen/kd11/splits.json`.

---

## 1. D-01 — Her "reproduced exactly" rebuild of his arm used the wrong extraction of her own vectors  [HIGH]

**What was tested.** `kd/docs/FINDINGS.md:4476` (commit `4d0d046`) says of his two arms: *"Their two
arms reproduce exactly from `kd/personas/kd_role_groups.json`."* `kd/personas/kd_role_groups_collab.json:71`
repeats it: *"The collaborator's ext_axis1 (12v13) … reproduced exactly."* I checked that on numbers.

**What happened. The role membership does reproduce exactly. The numbers reproduce his *control arm*,
not his headline arm.**

- Role membership: his `KNOWLEDGE_CANDOUR_PERSONAS` (`persona_pcs.py:140`) is exactly the union of her
  `sincere_ignorance` + `plain_assertion`; his `KNOWLEDGE_DENIAL_PERSONAS` (`:134`) is exactly her
  `knowledge_denial`. 12 v 13 confirmed. **VERIFIED.**
- Vectors: his headline store `persona_ood_ext_L*` is built from `persona including knowlede denial/`
  (his `summary.md:4`), which is her **`run-kd/vectors_s3`**. Her rebuild used
  **`run-kd/vectors_s23`** (`FINDINGS.md:4528`, the `6_axes.py --vectors_dir` line). Different judge
  filter, different rollout counts.

The consequence, on the five datasets both evaluate, mean-pooled:

| 12v13 extended axis, zero-shot AUROC | L20 | L40 | L44 |
|---|---:|---:|---:|
| his `persona_ood_ext_*` (s3) — the published arm | 0.8868 | 0.8990 | 0.8841 |
| his `persona_ood_mod_*` (s23) — his rollout-count control | 0.8664 | 0.8879 | 0.8687 |
| **her `axes_collab_s23.pt` rebuild** | **0.8628** | **0.8882** | **0.8688** |

Her rebuild sits on top of his **control** (mean \|gap\| 0.0015 over 15 cells, max 0.0175) and
0.021 away from his **published** arm (max 0.0759, on HP-KR at L20: his 0.8816 vs her 0.8057).

**How sure: VERIFIED, HIGH.** 30 cells, three layers, both sides from raw files.

**Why it matters.** F-98 and F-99 are the ledger's replication of his headline on her eleven
datasets. They are measuring the s23 bank throughout. F-98's own provenance check
(`FINDINGS.md:4521-4523`) tested the sensitivity of the **8v7** arm to the extraction (0.004–0.009)
and generalised it to the whole table — but the **12v13** arm is about **three times** more
sensitive at L20 (0.019 on the 5-dataset mean; his own `persona_ood_mod_L20/summary.md` measures
0.9041 → 0.8839 on his six). **The conclusion of F-98/F-99 is not overturned — it gets slightly
stronger**, because with the s3 bank `ext − axis1` at L20 is +0.1365 rather than +0.1180 on the
shared five. What needs changing is one sentence: F-98 should say *"rebuilt from our s23
extraction; their published arm uses our s3 extraction, which their own `persona_ood_mod_*` control
shows is worth +0.02 at L20"*.

**NOT RECOMPUTABLE:** no `collab_extended` contrast was ever built on her s3 vectors, so her eleven-dataset
table cannot be redone at s3 without a new run. The bound above comes from his committed control.

**Evidence.** `kd/docs/FINDINGS.md:4476, 4521-4530`; `kd/personas/kd_role_groups_collab.json:71`;
his `results/persona_ood_ext_L20/summary.md:1-6` and `results/persona_ood_mod_L40/summary.md`;
`audit/scripts/phase3/p3b_bankmatch.py`, `p3b_headtohead.py`. Bank identity independently
re-confirmed here by numeric fingerprint (§0, row 2), not only by RF-1's md5.

---

## 2. D-02 — Her headline crossgen uses the question-confounded persona bank; his uses her repaired one  [HIGH]

**What was tested.** When the draft says *"the paper's own 8v7 deception contrast"* and the ledger
says *"the paper16 deception axis"*, are they the same direction? I compared his 26-persona arm
against both of her 26-persona banks.

**What happened. They are different banks, and the gap is large.**

| 8v7 deception axis, zero-shot AUROC | his `persona_ood*` | her `banks/paper16_shared` | her `kd11` / `run-crossgen` root |
|---|---:|---:|---:|
| HP-KR, L20 | 0.1029 | **0.1028** | **0.2504** |
| HP-KR, L40 | 0.5311 | **0.5315** | **0.6436** |
| convincing-game, L20 | 0.9760 | **0.9762** | **0.8913** |
| convincing-game, L40 | 0.9071 | **0.9071** | **0.7884** |
| instructed-deception, L40 | 0.9726 | 0.9725 | 0.9709 |
| mean \|gap\| over 10 cells | — | **0.0015** | **0.0496** (max 0.1476) |

His bank is her **`run-paper16-shared`** — the repaired bank in which all 26 personas answer the same
questions. Her headline crossgen stores (`run-crossgen/`, `run-crossgen/kd11/`, `passA`, `passC`)
use `axes_vectors_s3_fp32.pt`, the **original** bank whose persona families answered different
question sets (the AD-2 confound, already established in `audit/PROGRESS.md` and `audit/methods/EXTRACTION_VS_PAPER.md`).

**How sure: VERIFIED, HIGH** at both L20 and L40 — the shared-bank match is 0.0000–0.0004 on four
of five datasets at each layer, which is a fingerprint, not a coincidence. The metadata agrees
independently: his `paper16_L20.pt` carries `source_run: run-paper16-shared` (Phase 0).

**Why it matters.** Three things follow.
1. Any sentence that puts one of his numbers beside one of her crossgen headline numbers for
   "the paper's axis" is comparing two banks. The two differ by up to **0.148 AUROC** on the one
   dataset the whole HP-KR story rests on.
2. The direction of the difference is informative and should be stated: the repaired bank gives a
   **cleaner** deception axis — better on convincing-game (+0.085 / +0.119) and **more strongly
   inverted** on HP-KR (−0.148 / −0.113). That is what you would expect if the confounded bank's
   leading direction is partly "which question set", and it is consistent with her own memory note
   `unified-pc1-is-question-family` and with F-18. *(INFERRED, from the sign pattern; two datasets.)*
3. The repaired bank exists on her side at **21 layers × 6 seeds × 2 poolings** (252 matrices on the
   lab PC) but **only 2 matrices are on the laptop** (`run-crossgen/banks/paper16_shared/matrices`,
   mean and last at L40). Anyone re-deriving this locally will silently get L40 only.

**Evidence.** Her `run-crossgen/banks/paper16_shared/matrices/mean_L{20,40,44}_s42.json` (lab PC,
via `audit/scripts/phase3/p3b_lab_zs2.ps1`), `run-crossgen/kd11/matrices/mean_L{20,40,44}_s42.json`;
his `results/persona_ood/zero_shot.csv`, `results/persona_ood_L40/zero_shot.csv`,
`deception_detection/persona_pcs.py:31, 121`.

---

## 3. D-03 — His knowledge-denial axis and hers are different contrasts, and the difference is the whole argument  [HIGH]

**What was tested.** His 4-vs-6 "knowledge-denial" axis reads 0.890 (L20) / 0.919 (L40) on HP-KR and
is the draft's central claim. Her design deliberately splits his six-role negative pole into two
three-role poles. I measured both on **one bank** (her s23 / his `persona_ood_mod_*`) at the same
layers, so only the contrast differs.

**What happened.**

| zero-shot AUROC on HP-KR, same bank (s23) | L20 | L40 | L44 |
|---|---:|---:|---:|
| his `denial_axis1` — 4 denial vs **6 merged** candour | 0.8734 | 0.8826 | 0.8734 |
| her `knowledge_denial` — 4 denial vs **3 sincere-ignorance** | **0.5923** | **0.6692** | **0.6659** |
| her `kd_assertion` — 4 denial vs **3 plain-assertion** | **0.9118** | **0.9344** | **0.9266** |

**The merged axis is carried by the plain-assertion half.** Against roles that assert plainly it
reads 0.91–0.93; against roles that deny knowledge *sincerely* — the wording-matched control — it
falls to 0.59–0.67.

**How sure: VERIFIED, HIGH** for the numbers (one bank, three layers, her file
`run-crossgen/kd11/banks/collab/matrices/mean_L{20,40,44}_s42.json`, his
`results/persona_ood_mod_L*/zero_shot.csv`). The *interpretation* — that the gap is register rather
than sincerity — is **INFERRED** from the contrast definitions, but it is the interpretation her
three-group design was built to test and it is what the numbers show.

**Why it matters.** This is the sharpest, cleanest disagreement in the whole comparison and neither
document states it. His summary.md flags the risk honestly (*"Group assignment is by name … a
different split would give a different axis"*, `persona_ood_ext_L20/summary.md`), but never measures
it. Her F-97 asserts the confound in words (*"their HP-KR numbers carry the confound the three-group
design exists to remove"*) but quotes no number for it. **Here is the number: 0.28–0.27 of his
HP-KR result is the plain-assertion pole.**

Related, and worth sending him: on her *matched* HP-KR dataset (`harm-pressure-kr-matched`, 258 rows,
built so both classes use denial wording), his 12v13 extended axis reads **0.4630 (L20) / 0.6203
(L40) / 0.5383 (L44)** and her `knowledge_denial` axis reads **0.3932 / 0.4989 / 0.4825** — at or
below chance. His draft lists exactly this test as future work (*"we have not run it"*). It has been
run. **VERIFIED** (same file; zero-shot, so F-108's row-level train/test leak between HP-KR and
kr-matched does not apply).

**Evidence.** `kd/personas/kd_role_groups_collab.json` (`contrasts` block);
`deception_detection/persona_pcs.py:134-140`; the two stores named above; `audit/phase0/draft_text.txt`
(the future-work paragraph).

---

## 4. D-04 — The supervised baseline they each call "8192-d probe" is not the same probe  [HIGH]

**What was tested.** His headline is *0.916 against 0.732*, where 0.732 is a probe on full
8192-dimensional activations trained on one dataset. Her ledger's margins use her `raw` feature. Both
use the identical estimator (StandardScaler + `LogisticRegression(C=0.1, fit_intercept=False,
max_iter=1000)`; his `persona_ood.py:59,190`, hers `9_persona_probes.py:202-220`). I restricted both
matrices to the five datasets both evaluate and recomputed the off-diagonal mean.

**What happened.**

| off-diagonal AUROC, 5 shared datasets, 8192-d probe | L20 | L40 | L44 |
|---|---:|---:|---:|
| his `matrix_baseline.csv` | 0.6085 | **0.7064** | **0.6959** |
| her `raw` + `lr_reg10` (same estimator) | 0.6160 | **0.6509** | **0.6311** |
| her `raw` + `paper` probe | 0.5871 | 0.6148 | 0.6117 |
| gap (his − her `lr_reg10`) | −0.0075 | **+0.0555** | **+0.0648** |

**Cause, verified in both scripts:** his off-diagonal cells train on **100 % of the source dataset,
unbalanced** (`persona_ood.py:302-303`); hers train on a **class-balanced 80 % subsample**
(`27_crossgen.py:12`, `splits.json`). On the two harm-pressure datasets that is a 10× difference in
training rows — HP-C 2,134 (his) vs **214** (hers); HP-KR 2,139 vs **222**. Diagonals differ too:
his is 5-fold CV over the whole dataset, hers a held-out 20 % split.

**How sure: VERIFIED, HIGH** for the numbers and the protocol; **INFERRED** that the training-row
difference is the dominant cause (it is the only difference that grows with depth in the right
direction, and the two agree at L20 where his probe is weakest).

**Why it matters.** His baseline is the **stronger** one at L40/L44, so his headline margin is if
anything conservative on the baseline side — that is a point in his favour and should be said.
But it also means *her* "persona features vs raw" margins and *his* "persona axis vs baseline"
margins are not on the same scale, and must never be quoted in the same sentence without saying so.

**Evidence.** `audit/scripts/phase3/p3b_baseline.py`; his `results/persona_ood*/matrix_baseline.csv`;
her `run-crossgen/kd11/matrices/mean_L{20,40,44}_s42.json` (`results` block, feature `raw`) and
`run-crossgen/kd11/splits.json`.

---

## 5. D-05 — "The persona axis on HP-KR at layer 20" has five different values in circulation  [MEDIUM]

**What was tested.** The single most-quoted number in both documents. I collected every version and
attached each to its bank.

| value | what it actually is | source |
|---:|---|---|
| **0.109** | the RepE-trained *base probe*, not a persona axis | his `XXXX-9-llama-all` run, quoted `persona_ood_ext_L20/summary.md:9` |
| **0.124** | the published Liars' Bench number | the Liars' Bench paper, quoted in the same line |
| **0.1029** | 8v7 axis on the **26-persona shared-question bank** | his `persona_ood/zero_shot.csv`; her `banks/paper16_shared` gives **0.1028** |
| **0.1483** | 8v7 axis on the **36-persona kd bank, s3** | his `persona_ood_ext_L20/zero_shot.csv`; her `kd11/banks/kd` gives **0.1482** |
| **0.1652** | 8v7 axis on the **36-persona kd bank, s23** | his `persona_ood_mod_L20/zero_shot.csv`; her `kd11/banks/collab` gives **0.1652** |
| **0.2504** | 8v7 axis on her **original (question-confounded) paper16 bank** | her `run-crossgen/kd11/matrices/mean_L20_s42.json` |

All six are correct for their own configuration. **VERIFIED, HIGH.**

**Why it matters.** Three of these six appear in the two documents without their bank named, and
they span 0.10–0.25 — a factor of 2.4 on the number the HP-KR narrative rests on. Any sentence
using one of them needs the bank beside it.

---

## 6. D-06 — One table in his `summary.md` mixes two banks, and that is where 0.103-vs-0.148 comes from  [MEDIUM]

**What was tested.** His "Per test set at layer 20, mean off-diagonal" table
(`persona_ood_ext_L20/summary.md:62-69`) gives baseline / deception axis / extended axis per dataset.
I recomputed all three columns from the matrices in that folder.

**What happened.** The `baseline` and `extended axis` columns come from `persona_ood_ext_L20`, but
the `deception axis` column comes from the **other** store, `persona_ood`:

| test set | his table, "deception axis" | recomputed from `persona_ood_ext_L20/matrix_axis1.csv` | from `persona_ood/matrix_axis1.csv` |
|---|---:|---:|---:|
| CG | 0.786 | 0.764 | **0.786** |
| HP-C | 0.622 | 0.630 | **0.622** |
| HP-KR | **0.103** | **0.148** | **0.103** |
| IT | 0.792 | 0.793 | **0.792** |
| ID | 0.776 | 0.775 | **0.776** |
| RepE | 0.795 | 0.796 | **0.795** |

**How sure: VERIFIED, HIGH** (six of six cells match the other store exactly).

**Why it matters.** It is a documentation slip, not a result error — his draft distinguishes the two
banks correctly (`audit/phase0/draft_text.txt:347`, *"original bank (0.103 at layer 20) and still
inverts on the re-extracted one (0.148)"*). But this table is the most-read summary on his branch
and it is the origin of the 0.103 that travelled into her `COLLABORATOR_BRIEFING.md` and into
`audit/phase0/draft_vs_ledger_preview.md` as a "0.103 vs 0.109" puzzle. The puzzle has three
answers, not two: 0.109 is the base probe, 0.103 is the shared bank, 0.148 is the kd s3 bank.

---

## 7. D-07 — His rollout-count control is also the s3-vs-s23 sensitivity measurement, and neither document uses it that way  [MEDIUM]

**What was tested.** `persona_ood_mod_L{20,40,44}` is his control for "does sampling each persona
more help?". Because both of its banks are hers (M3 verified the rollout counts match
`run-kd/vectors_s3` and `run-kd/vectors_s23`), it is simultaneously a measurement of what her judge
filter costs.

**What happened.** On his six datasets, off-diagonal AUROC:

| arm | L20 | L40 | L44 |
|---|---|---|---|
| `ext_axis1` (12v13) | 0.9041 → **0.8839** | 0.9158 → **0.9066** | 0.9034 → **0.8905** |
| `denial_axis1` (4v6) | 0.8660 → 0.8501 | 0.9115 → 0.8999 | 0.9001 → 0.8873 |
| `axis1` (8v7) | 0.6511 → 0.6505 | 0.8319 → 0.8216 | 0.8260 → 0.8178 |

His own heading: *"More rollouts per persona did not help."* **VERIFIED** (recomputed from
`results/persona_ood_mod_L*/summary.csv`).

**Why it matters, in both directions.**
- For him: the draft reports only the `ext` bank and never mentions this arm (already in
  `docs/DRAFT_CORRECTIONS.md §3.1`).
- For her: this *is* the sensitivity check F-98's provenance paragraph needed and did not have
  (§1 above). The extended arm loses 0.009–0.020 when moved from s3 to s23; the 8v7 arm loses
  0.0006–0.010. Quoting the 8v7 figure as if it covered both arms understates it ~3× at L20.
- The finding itself is interesting and unfollowed: **more rollouts per persona made every leading
  direction worse**, because the group means converge (his measured axis-norm ratios 0.84–0.89
  against 0.98 for the persona vectors themselves). Nothing on her side tests this.

---

## 8. D-08 — Insider-trading at layer 20 is the only same-bank cell where they disagree  [MEDIUM, cause UNCLEAR]

**What was tested.** The residual after matching banks (§0).

**What happened.** Four of five datasets agree to ≤0.007 in every matched cell. `insider-trading`
does not, and only at **layer 20**:

| insider-trading, same bank | L20 | L40 | L44 |
|---|---:|---:|---:|
| 8v7 axis, his − hers | **+0.0107** | −0.0004 | +0.0004 |
| 12v13 axis, his − hers | **+0.0175** | −0.0003 | +0.0008 |
| PC1, his − hers | **+0.0121** | +0.0026 | +0.0028 |
| sycophancy axis, his − hers | **+0.0694** | +0.0059 | +0.0074 |
| 26-persona shared bank, his − hers | **+0.0090** | −0.0009 | — |

Same rows (1,080; 423 positive), same labels, same bank, same layer; his number is higher in every
case.

**How sure: VERIFIED** that the gap exists and is dataset- and layer-specific; **UNCLEAR** on cause.
The most likely explanation is that the two pipelines pool over slightly different token spans of
the insider-trading response — it is the only one of the five with a structured scratchpad-plus-action
format, and layer-20 representations are more position-sensitive than layer-40 ones. **INFERRED.**

**What would settle it:** his `activations/pooled/insider_trading_cadenza__plain` row mask against
her detection mask for the same rows — i.e. the mask definition, not the activations themselves.
A single row's pooled vector norm from each side would decide it.

**Why it matters.** It is small enough not to move any conclusion (insider-trading sits at 0.97–0.99
on every arm), but it is the one place where "their pipelines agree" has an exception, and anyone
reproducing §0 will see it.

---

## 9. D-09 — "Mean off-diagonal AUROC" is not the same quantity on the two sides  [MEDIUM]

**What was tested.** Whether his trained-transfer numbers and her trained-transfer numbers can be
compared at all.

**What happened.** Two structural differences, both verified.

1. **His persona arms are 1-dimensional.** A probe on a 1-d feature can only learn a sign, so his
   trained off-diagonal mean equals his zero-shot mean whenever every training set agrees on the
   sign (0.9158 both ways at L40; already verified in Phase 2). **Her trained persona features are
   k-dimensional** (`axes`, `pc:deception`, k = 5 by default) — there is no such identity for her.
   So "off-diagonal mean" is a *trained* quantity for her and a *zero-shot* quantity in disguise
   for him.
2. **His layer-20 matrices carry a sign-flipped training row.** I scanned all 24 of his 1-d
   matrices. Exactly two are affected, both at L20:

   | store | matrix | as written | sign-corrected | flipped row |
   |---|---|---:|---:|---|
   | `persona_ood_ext_L20` | `matrix_axis1` | 0.6511 | **0.7909** | HP-KR |
   | `persona_ood_ext_L20` | `matrix_pc1` | 0.6669 | **0.8056** | HP-KR |
   | `persona_ood_mod_L20` | `matrix_axis1` | 0.6505 | **0.7881** | HP-KR |
   | `persona_ood_mod_L20` | `matrix_pc1` | 0.6646 | **0.8002** | HP-KR |

   **No row is flipped at L40 or L44, and no `ext_axis1` or `denial_axis1` matrix is affected at any
   layer.** So the artifact is narrower than "his matrices are sign-flipped": it is layer-20, and
   only for the arms that invert on HP-KR.

**How sure: VERIFIED, HIGH** (`audit/scripts/phase3/p3b_signflip.py`, re-derived from his committed
matrices; consistent with the Phase-2 P11 finding and with his own F-97 caveat, which the owner
already records at `FINDINGS.md:4519-4524`).

**Why it matters.** It bounds the blast radius of the sign artifact for the paper: it inflates the
"lift from the extended personas" **at layer 20 only**, and it does not touch the 0.916 headline at
layer 40.

---

## 10. D-10 — Token-position reading: same conclusion, magnitudes 4× apart  [MEDIUM]

**What was tested.** Both measured whether scoring the top-decile of response tokens beats mean
pooling — she in `14_pertoken.py` (memory note `q8-per-token-beats-pooling`), he in
`persona_positions_L{20,40,44}`. Neither cites the other.

**What happened.** On the **deception axis**, the axis she used, they agree in sign at every layer:

| top-10% mean − mean, AUROC | L20 | L40 | L44 |
|---|---:|---:|---:|
| hers (paper16 deception axis, 2,000 rows pooled into one AUROC) | **+0.103** | +0.005 | +0.014 |
| his (`reads_summary.csv`, deception axis, 5-dataset mean of per-dataset AUROCs) | **+0.023** | +0.027 | +0.049 |

On his **extended** axis the sign flips at L20 (0.8868 → 0.8727, −0.014), which is what his prose
reports; that is a different axis, not a contradiction.

**How sure: VERIFIED** for his numbers (recomputed from `persona_positions_L*/reads_summary.csv`);
**DOC-ONLY** for hers — the memory note's table was not re-derived in this workstream.

**Why it matters.** The magnitude difference is a metric artifact, not a result disagreement: she
pools all datasets into a single AUROC (so between-dataset score offsets count), he averages
per-dataset AUROCs (so they cannot). **They should quote each other's direction, not each other's
number.** Worth flagging to the lead: this is the one item here resting on a memory note rather than
a raw file on her side.

---

## 11. D-11 — Things that look like disagreements but are different dataset sets  [reference]

Recorded so they are not mistaken for conflicts.

| quantity | his | hers | why they differ |
|---|---:|---:|---|
| 12v13 extended axis, zero-shot, L40 | **0.9158** (his six) | **0.7996** (her eleven) | six datasets vs eleven. On the five both evaluate: 0.8990 (his) / 0.8882 (hers, s23 bank). **VERIFIED** |
| 8v7 axis, zero-shot, L40 | **0.8319** (his six) | **0.7325** (her eleven) | same; on the shared five, 0.7983 / 0.7868 |
| "six Liars' Bench datasets" | six | — | one of his six is `repe_honesty`, which is not in her eleven and which his own summary flags as in-distribution for the deployed probe |
| pooled-baseline inversion | 0.124 (L20) / 0.523 (L40) | — | the two numbers use **different weightings** of his own table (`balanced` vs `naive`); already established in Phase 2 batch P10 and sent in `docs/DRAFT_CORRECTIONS.md §1.11` |
| "layer 40 is best" | best of the three he extracted | F-91: L32 is the peak | he never extracted L32 — not a disagreement, a coverage gap |

---

## 12. What this workstream did **not** find

- **No evidence of a bug in either pipeline.** 55 bank-matched cells, mean \|gap\| 0.0015.
- **No disagreement traceable to a seed.** His matrices use `SEED = 42` throughout and average
  `N_RANDOM_DRAWS = 5` seeded draws for the random control; her crossgen main seed is 42. No cell
  compared here crosses a seed boundary.
- **No disagreement traceable to `k`.** His persona geometries are fitted at `k = 3`
  (`persona_ood.py`, `fit_geometry(bank, k=3)`); her crossgen banks here are all `k = 5`. But every
  comparison in §0–§8 is on a **contrast direction or PC1**, neither of which depends on `k`. The
  one `k`-sensitive comparison (`ext_pc1`) does show a residual of 0.010–0.014 where the contrast
  arms show 0.000–0.004 — consistent with F-98's caveat that her PCA basis includes the `default`
  anchor and his does not. **VERIFIED, and it confirms her caveat rather than contradicting it.**
- **His downloaded activations were never needed.** Every disagreement above was settled from
  committed CSVs on his branch plus her result JSONs. Nothing here requires files from him.

---

## 13. Ranked, for the owner

1. **D-03** (his 4v6 vs her 4v3 split: 0.87 vs 0.59 on HP-KR, same bank) — the most consequential
   number in the comparison and it is in neither document.
2. **D-02** (her headline bank is the confounded one, his is the repaired one; up to 0.148 apart).
3. **D-04** (the 8192-d baselines differ by 0.06 at L40/L44 for a protocol reason).
4. **D-01** (F-98's rebuild is on s23, his headline is on s3 — one sentence to fix, conclusion
   unchanged or slightly strengthened).
5. **D-05 / D-06** (the five HP-KR numbers, and the table that mixes two banks).

---

*Scripts written by this workstream: `audit/scripts/phase3/p3b_zs_extract.py`,
`p3b_lab_ls.ps1`, `p3b_lab_ls2.ps1`, `p3b_lab_zs.ps1`, `p3b_lab_zs2.ps1`, `p3b_headtohead.py`,
`p3b_bankmatch.py`, `p3b_baseline.py`, `p3b_signflip.py`. All read-only. No file outside
`audit/` was written; nothing was written on the lab PC.*
