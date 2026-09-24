# DOC_FIXES triage — 95 entries, grouped for your decision (6D-4)

_2026-09-24. Nothing here has been applied by this session. This is the list you rule on._

Machine-readable companion: `audit/phase6/df_triage.csv` (produced by
`audit/scripts/p16_df_triage.py`, which parses `DOC_FIXES.md` and extracts the paths each
entry names).

---

## 0. Four things that change what you're ruling on

**(a) Three are already applied. Outstanding is 92, not 95.**
`DOC_FIXES.md:3` says "Nothing here has been applied"; `CLAUDE.local.md` says "NOT ONE has been
applied"; `COVERAGE_GAP.md:129` says 93. All three are wrong, in different directions.
Verified in the documents themselves:

| DF | state | evidence |
|---|---|---|
| **DF-49** (length claim) | **APPLIED**, all 8 sites | `docs/thesis/persona-coordinates-70b-results.md:497-500` now reads "**9 of 11** … By **length-only AUROC** … **9 of the 10**", naming both metrics |
| **DF-50** (F39_NOTE §2) | **APPLIED** in `audit/fix/F39_NOTE.md`; its *second* half is **not** — F-48 `FINDINGS.md:1418` and F-54 `:1659` still carry the Table-1 framing with no pointer to F-60 |
| **DF-58** (bad path in the DF-49 repair) | **APPLIED** | same site, full key path now cited |

**(b) DF-31 is ~80% applied, and the 4 sites missed include the one that goes to reviewers.**
Six documents carry dated "Correction, 2026-09-23" annotations (`cap/docs/DESIGN.md:23`,
`docs/COLLABORATOR_BRIEFING.md:82`, `cap/docs/REVIEW_PACKET{,_V2,_V3,_V4}.md:21`). Still wrong:

| site | what it still says |
|---|---|
| **`cap/docs/review_page.html:162`** | "worse than a random guess — 0.374 where 0.5 is chance, while a randomly chosen direction gets 0.54" — uncorrected, **and this is the rendered page, not the .md** |
| `.claude/skills/knowledge-denial-context/SKILL.md:66` | "`sycophancy_mt` (0.374) — and on both a **random direction beats them**" — line 75 of the same file is corrected; line 66 was missed |
| `kd/docs/NEXT.md:15` | "the register axis's 0.374" — the location DF-68 already flagged as missing from DF-31's own list |
| `cap/docs/SARCASM_DESIGN.md:48` | "Our failures are **below** chance — 0.300 and 0.374" — no axis named |

The `review_page.html` one matters most: DF-31 established that **0.374 (deception-axis PC1) sits
*inside* the random-direction null**, so "worse than a random guess" is not supported for the
number quoted. The claim is true of the *sycophancy* axis at **0.313** (p 0.001–0.008).

**(c) 22 entries target an append-only file and therefore cannot be "applied" at all.**
`kd/docs/FINDINGS.md` is append-only by your own rule. DF-01–08, 39–45, 51–54, 56, 71, 78 all
propose corrections to existing F-entries. **Recommendation: one new consolidated F-entry
(F-123) that supersedes the lot**, rather than 22 decisions. See §5.

**(d) DF-85 points at the wrong entry, and applying DF-12 as written would put a false claim
about your collaborator into a document.**
DF-85 says "withdraw **DF-71**'s reproducibility half". DF-71 is the F-61 `+0.006` script-bug
entry — it has no reproducibility half. The entry being withdrawn is **DF-12** ("a
reproducibility promise the repository cannot keep" — `paper/make_numbers.py` exists on no
branch). `paper/make_numbers.py` **is** on `origin/paper`. **`kd/docs/FINDINGS.md:5958` (F-119)
repeats the same mis-pointer.** DF-12 currently reads as live and is false. **Kill DF-12.**

---

## 1. P1 — apply before submission (39 tagged P1, minus the dead ones)

### 1a. The draft PDF / abstract — his document, your submission

| DF | one-line recommendation |
|---|---|
| DF-13 | "Every method has recall near zero on HP-KR" — add "on the cross-model averages"; the Follow-Up Probe on 70B reads 0.37. **Apply.** |
| DF-14 | "inverts on one dataset at each depth" → **two at L20** (HP-KR 0.109, harm-pressure-choice 0.471). **Apply.** |
| DF-15 | "the top-10% token read is best" is contradicted by his own Table 9 at L40. **Apply — a reviewer checks the table.** |
| DF-17 | Abstract's 0.904/0.916 are `ext_axis1` (your kd personas); the published 8v7 reads 0.791/0.832. **Apply — attribution, not a number fix.** |
| DF-20 | Abstract's "0.09 below the contrast" — the real gap is **0.148**; no pairing in his L40 summary gives 0.09±0.005. **Apply.** |
| DF-21 | "the principal components do not contribute" — false for the published bank (PC1 above contrast at all three layers). **Apply, scoped to the extended bank.** |
| DF-22 + DF-46 | "lifts transfer by 0.18–0.21" is manufactured by one sign-flipped HP-KR row at L20; corrected it is +0.03/+0.00/−0.00/−0.00. **Apply as one fix — and DF-46 says describe it as HP-KR inverting, not as his error.** |
| DF-23 | "six Liars' Bench datasets" is **five**; L44 was run and the abstract omits it. **Apply.** |
| DF-24 | "independently constructed sycophancy axis" — fitted on your bank, \|cos\|=0.54 by his own comment. **Drop the word "independently".** |
| DF-26 | "within 0.03 of the upper bound" is 0.0310, and 0.196 on balanced-acc@1%FPR. **Apply — the omitted metric is the benchmark's headline.** |
| DF-27 | "inverts to 0.124 and 0.523" mixes `balanced` and `naive` weightings; **0.523 is above chance, not an inversion.** **Apply.** |
| DF-76 | Draft still promises the wording-controlled test as future work. **It ran three times (F-03/F-115/F-116). Delete the promise, state the outcome.** |
| DF-77 | Name **both** hpkr slices — `text_matched` (258) and `knowledge_matched` (2,001). **Apply.** |
| DF-79 | Draft reports L40; his own sweep peaks at **L36 (0.9275)**. **Apply — or say L40 was pre-specified and give L36 beside it.** |
| DF-80 | "outscores at every depth" is false at L64/L68/L72. **→ "13 of 16 layers".** |
| DF-81 | Quote the Contributions margin **excluding HP-KR** too (±0.06, crosses zero 5×). **Apply.** |
| DF-93 | Label the `dim=1` arms **zero-shot** and put the error bar over the 5 datasets, not 30 cells. **Apply — the honest claim is stronger.** |
| DF-72 | State the lexical baseline (regex 0.9125 / 0.9597) beside **every** HP-KR number. **Apply.** |
| DF-12 | **DEAD — do not apply.** See §0(d). |

### 1b. Your own documents

| DF | document | one-line recommendation |
|---|---|---|
| **DF-66** | `docs/thesis/sarcasm-pilot-negative-result.md:70-75` | **Highest urgency in the file.** "Two gates were registered … before generation" is git-false for G2: `0.65` enters the repo in `b34ff2e`, the commit whose message reports the failure. Verified at `:70-75` — still uncorrected. **Apply the suggested wording; the 0.921/0.947 result survives either gate.** |
| **DF-67** | `docs/REVIEW_REQUEST_POOLING.md:17-18` | Still says "80/20 splits by **item id**"; `30_premise.py:106` permutes **rows**. **This document is the one you send for adversarial review. Fix before it goes.** Same doc prints 0.850 (`:32`) and 0.806 (`:64`) for one condition. |
| DF-55 | `docs/NOVELTY_CHECK_PROMPT.md:113` | Still live. "probes report ~0.87" has no source; the raw diagonal is **0.9663**, so the regex **loses** by 0.054. **Apply the restatement — "94.4% of a full probe" is still a strong claim.** |
| DF-63 | `docs/thesis/...70b-results.md:513 vs :741` | Pre-registered ±0.02, concluded ±0.04, all four stored tests `equivalent_within_margin = False`. **Apply the suggested wording — this is exactly what a referee hunts for in a control-discipline thesis.** |
| DF-64 | same file `:752` + `docs/PERSONA_CONSTRUCTION.md:167` + memory | "all 1,400 rollouts re-judged" → **1,350**. The memory note contradicts itself at line 13 ("1,231 of 1,350"). **Apply at all three sites.** |
| DF-09/DF-10 | `docs/NOVELTY_CHECK_PROMPT.md:110` | DF-09 is **superseded by the applied DF-49**; DF-10's two token counts (505/310, 245/96 → 529.4/340.4, 283.2/176.4) are **still unfixed**. **Apply DF-10 only.** |
| DF-42 | F-26/F-33/F-73 + memory | "0.553 vs 0.665" matches no matrix; it is **0.5555/0.6623** (11 ds) or 0.5494/0.6652 (10). **Goes in the consolidated ledger entry.** |
| DF-31 | 4 residual sites | See §0(b). **Apply the four; `review_page.html` first.** |
| DF-32 | `docs/COLLABORATOR_BRIEFING.md:142-143` | "8–11 of 11" mixes denominators; nothing is 8 of 11. **Apply.** |
| DF-33 | `…BRIEFING.md:146` | "~666 rows" is the cap; realised is 412–500 (mean 449). p=0.38 is one-sided; quote the two-sided CI. **Apply.** |
| DF-36 | `…BRIEFING.md:251` | "both of your arms lose by 0.064/0.055" — that's the extended arm; `axis1` loses by 0.146/0.122. **Apply.** |
| DF-57 + DF-69 | `…BRIEFING.md:654`, `docs/ROADMAP_TO_DEFENCE.md:27` | Both answer a question about **his** baseline using **your** 11-dataset corpus. **The argument gets stronger on his numbers (every alternative is above 0.732). Apply both.** |
| DF-59 | `docs/thesis/...70b-results.md` | O-070 sits under "What holds" but both gaps are non-significant in their own file (q 0.86/0.67) and the gap is positive in 28 of 126 configs. **Move it out of "What holds".** |
| DF-61 | `memory/q8-per-token-beats-pooling.md` | "+0.053 at L28" understates ~2×: the largest zero-shot pooling gain is **+0.1027 at L20**, collapsing to +0.005 at L40. **Apply.** |
| DF-68 | `docs/NOVELTY_ASSESSMENT.md:103` | Claim 4's "corr −0.76" mechanism was **withdrawn by F-74's own coupling null** (P=0.2135), yet §4 nominates Claim 4 to lead with. **Apply — this one is load-bearing for the novelty pitch.** |
| DF-73 | wherever the 3-rung trend appears | 8B rung is **7 datasets, not 12** — 5 have no `rows.jsonl`. **Name the 7 wherever the ladder is quoted.** |
| DF-86 | scale-decline statements | State the decline on **both** banks (+0.038→+0.015→−0.012 and +0.036→+0.019→+0.005). **Apply — it closes the "wrong personas" objection.** |
| DF-92 | results | `pc:kd_assertion` 0.8119 vs `pc:knowledge_denial` 0.7492 — **the best-transferring persona feature in the project.** **Report it; it is the register diagnosis made visible.** |
| DF-94 | reports + `PROGRESS.md` + `BRIEF_AS_EXECUTED.md` | "424 of 692" → "314 of 692 inventory claims, plus 110 raised while checking". **Apply — I'll re-derive the final numbers at 6E.** |

### 1c. Code bugs — not prose, and they need a decision not an edit

| DF | one-line recommendation |
|---|---|
| DF-48 | `scale/scripts/run_scale_chain.ps1:197-203` writes both bank variants to `axes_$tag.pt`; `$vt` never reaches the path. **One-char fix (`axes_$vt.pt`) but it breaks the downstream single-bank reader. Your call — I'd leave it and document it.** |
| DF-74 | Same defect in the 8B chain. **Consequence: any 8B "unfiltered vs filtered" comparison compared s23 against itself.** Same call as DF-48. |
| DF-39 | F-32's acceptance table certifies a bank generation (v2) that never produced a rollout; the live bank is v3. **The cap conclusion is safe for a better reason — teacher forcing makes turn 1 byte-identical across all 65 arms. Rest the argument there.** |

---

## 2. P2 — apply if time allows (27 tagged)

**Draft:** DF-16 (18 of 30 → 14 exactly at 0.5) · DF-18 (the rollout-count control arm he never
mentions, scoring *lower*) · DF-25 (merges your `sincere_ignorance`/`plain_assertion` split, which
existed to kill a wording confound) · DF-28 (contradicted inside his own `loo_L20/summary.md:46`) ·
DF-29 ("generalization gap is zero" is definitional for a label-free 1-d feature) · DF-82 (name the
arm) · DF-83 (two cited folders not on the branch) · **DF-84 (the late-layer rise — F-120 solved it;
adding one sentence converts the obvious reviewer question into a demonstration your controls work.
I'd promote this to P1)** · DF-78 (quote F-03 leak-guarded) · DF-87 (never quote the 70B kd +0.0054
without its scope) · DF-91 (kd axes are 70B-only).

**Briefing:** DF-34 (path + the same merge as DF-25) · DF-35 (5- vs 6-dataset baseline) ·
DF-37 ("differ by 0.0004" holds for one feature; the one the argument rests on differs by 0.0142) ·
DF-38 (scope the share range to mean pooling — it strengthens the point).

**Yours:** DF-60 (hf275 columns name the wrong reference direction; conclusions unchanged ≤0.04) ·
DF-62 (8 smaller items incl. O-033's "clears no significance test" being false) · DF-65 (7 smaller
items from W6B) · DF-70 (the cap block's three unstated scope conditions — **mean-pooling only,
14/21 layers vs 3/21 under `last`**; and `cap/docs/DESIGN.md:176`'s "+0.045" is +0.043) ·
DF-90 ("16-persona bank" → 26-persona bank) · DF-95 (reports predate `origin/paper`).

**Ledger:** DF-04, 05, 06, 07, 08, 40, 41, 43, 44, 52, 53, 54 → §5.

**Scale:** DF-47 (the 3-rung comparison mixes an s23 bank with an s3 bank — **name as a caveat, do
not re-run**; F-62 bounds the effect at 0.005 against ±0.17 CIs).

---

## 3. P3 — housekeeping (7)

DF-11 (don't conflate two "9 of 10" claims) · DF-19 (**two venue strings verifiable in none of the
three PDFs — check before they reach a reference list**) · DF-30 (unverifiable cache count) ·
DF-45 (truncated print; conclusion is *stronger*) · DF-75 (pointer: the audit corrected 7 of its own
verdicts) · DF-88 (the s3-vs-s23 caveat can now be measured: s23 beats all at all three rungs by
+0.004/+0.003/+0.002) · DF-85 (**re-point at DF-12, then kill DF-12** — §0(d)).

---

## 4. The 22 untagged entries — priorities I'm assigning

`DF-49`–`DF-70` carry no P-tag. My assignment, on the same key (P1 = a wrong number or quantifier a
reader would rely on):

**P1:** DF-55, DF-57, DF-59, DF-63, DF-64, DF-66, DF-67, DF-68, DF-69 · **P2:** DF-51, DF-52,
DF-53, DF-54, DF-56, DF-60, DF-61, DF-62, DF-65, DF-70 · **done:** DF-49, DF-50, DF-58.

---

## 5. The ledger — one decision, not 22

22 entries propose corrections to existing F-entries in the append-only `kd/docs/FINDINGS.md`:
DF-01, 02, 03, 04, 05, 06, 07, 08, 39, 40, 41, 42, 43, 44, 45, 51, 52, 53, 54, 56, 71, 78.

**Recommendation: one new entry, F-123, that supersedes all 22 and is appended, not applied.**
Three of them are load-bearing and cut *for* your arguments, which is the reason to write it rather
than let it sit:

- **DF-03** — F-77/F-81's `37/80` is not a layer count at all (it's a bootstrap *fraction*, 0.46,
  rendered as a count). True counts 28/80 and 31/80. **The contrast with the cap bank's 21/21 gets
  stronger.**
- **DF-71** — F-61's `+0.006` was a label-parsing bug (`zip` over caption *characters*); real figure
  **+0.0093**, ratio **8.1×** not twelvefold. Bug already fixed in the script; only prose is wrong.
- **DF-31's ledger half** — the 0.374 misattribution is inherited from **F-65**.

---

## 6. What I'd do, if you want a short answer

**Apply now, before anything is sent anywhere (7):** DF-66, DF-67, DF-31's four residual sites
(esp. `review_page.html`), DF-68, DF-63, DF-55, DF-94.

**Apply to the draft before submission (17):** DF-13, 14, 15, 17, 20, 21, 22+46, 23, 24, 26, 27,
72, 76, 77, 79, 80, 81, 93, and promote **DF-84**.

**Kill (1):** DF-12 — and fix the DF-85/F-119 pointer that hid it.

**Append as one ledger entry (22):** §5.

**Decide, don't edit (3):** DF-48, DF-74, DF-39.

---

**Nothing above has been applied. Tell me which sets you want and I'll stop there — or say
"go" and I'll move to 6A, which does not depend on any of these being applied.**
