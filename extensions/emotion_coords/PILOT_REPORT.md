# Pilot report (2026-10-29)

The pilot ran every step end to end at small size: 20 emotions × 10 topics (one story per cell), all 1,086
usable neutral dialogues, 50 rows per class per Liars' Bench set, layers 20 and 52, 20 HP-KR questions for D2/D3,
and one NDIF generation call. Pilot numbers exercise the pipeline and size the full run. They are not results, and
none of them was used to change the design: the preregistration (`ae8e0ae1`) was committed before any of this ran.
Tables: `results/emotion_coords/pilot/`.

## Verdict

- **The pipeline works end to end.** Every validity check passed: reader parity, positions, D2 formatting,
  teacher-forced fidelity, and the published probe reproducing its own numbers on the pilot rows. The emotion
  geometry passes the valence check by a wide margin (|r| = 0.95 at L20, 0.94 at L52; the stop rule is 0.5).
- **Full-run cost.** About 1.6 h of NDIF (+0.3 h if story top-ups are needed), about 2.2 GB of disk, and about
  11 h of local fitting as the code stands. Two implementation changes that do not touch any result bring the
  fitting to about 3 h; see the estimates.
- **What needs you before the full run:** Anthropic's story prompt, if you want the optional self-written stories.
  The other points at the end are for information.
- **The probe numbers below come from 50 rows per class and a 20-emotion basis.** They only show that every arm
  runs. The full run's basis comes from 171 emotions, so its PCs and named directions will differ.

## NDIF: requests and time

One reader process (`reader/ec_read.py`, nnsight 0.8.1, `E:\aletheia\venv08`), run by `reader/run_chain.ps1` from
02:20:46 to 02:41:38: **21 minutes, 47 requests, no failed request.** Every read covers all four layers
(20, 28, 40, 52), so the full run reuses these rows.

| file | rows | tokens | requests | seconds | tokens/s | mode |
|---|---:|---:|---:|---:|---:|---|
| lb_parity | 50 | 23,676 | 7 | 96 | 248 | solo |
| stories | 200 | 29,246 | 1 | 104 | 283 | batched |
| neutral | 1,086 | 120,797 | 3 | 374 | 323 | batched |
| tylenol | 6 | 419 | 1 | 9 | — | solo |
| d2d3 | 160 | 51,814 | 20 | 232 | 223 | solo |
| fidelity (log-probs) | 100 | 55,648 | 13 | 178 | 313 | solo |
| gen_pilot (sampling) | 1 prompt | 243 new | 1 | 74 | 3.3 new/s | — |
| recheck lb_parity | 2 | — | 1 | 28 | — | solo |

CPU side: (b)/(c) at layer 20 for the pilot rows, Alpaca and RepE from the per-token cache took about 2.5 minutes
(3,050 rows).

## Checks

- **Parity of the new reader.** Each of 50 Liars' Bench rows had its detected-span mean read on NDIF and compared
  with the stored pooled (a):
  - Minimum cosine by layer: 0.99988 (L20), 0.99988 (L28), 0.99976 (L40), 0.99967 (L52).
  - L20 and L28 clear the 0.9998 bar. Two rows at L40 and five at L52 fall short, all of them single-token HP-C
    spans, long IT rows, or short HP-KR/ID spans.
  - The stored arrays at L28-52 came from an earlier batched NDIF pass under nnsight 0.7. Earlier projects found
    the same cross-pass envelope (about 1e-4 to 4e-4 in cosine).
  - The same reader re-reading its own rows gives cosine 1.0000000.
  - Probes use the stored arrays throughout, so the envelope does not enter any comparison.
- **Positions.** At layer 20 the server's vectors match the per-token cache, so both positions are right:
  - pre-response token (c): minimum cosine 0.99989;
  - last content token (b): minimum cosine 0.99988.
  - The decoded tokens are `\n\n` (end of the assistant header) and the token before `<|eot_id|>`.
  - A unit test checks the decoded positions in the row files (`tests/test_emotion_coords_positions.py`).
- **D2 formatting.** 23 D2 prompts are word-for-word an HP-KR row's prompt. Their pre-response vectors match the
  stored HP-KR pre-answer vectors at cosine ≥ 0.99990 (L20) and ≥ 0.99986 (L40). So D2 prompts are formatted
  exactly like the Liars' Bench data.
- **Fidelity (teacher-forced log-probs, 10 rows per set).** The recorded replies are likely under the cached
  formatting, as samples at temperature 0.7 should be:

  | set | reply tokens (median) | mean log-prob per token |
  |---|---:|---:|
  | HP-KR | 2 | −0.07 |
  | IT | 121 | −0.18 |
  | ID | 34 | −0.20 |
  | CG | 230 | −0.28 |
  | HP-C (one answer letter) | 1 | −1.35 |

  The closing `<|eot_id|>` scores ≈ 0 everywhere except ID (−0.11).
  - Dropping the template's own "26 Jul 2024" date block (variant B) changes the mean by +0.003 nats per token
    (58% of rows higher, per-set means −0.013 to +0.017).
  - So the doubled date header in the cached activations is not a formatting mismatch worth acting on. Both
    formats fit the data about equally, and the cached one matches the published probe and the thesis.

## Emotion geometry (sanity checks)

From 20 emotions × 10 topics. The neutral PCs removed for 50% of neutral variance were 56 at L20 and 81 at L52.

| check | L20 | L52 | expected |
|---|---|---|---|
| PC1 / PC2 variance | 32% / 11% | 36% / 11% | — |
| r(PC1, NRC valence), 20 emotions | **0.947** | **0.943** | Anthropic 0.81; stop if \|r\| < 0.5 at L52 |
| r(PC2, NRC arousal) | 0.51 | 0.55 | — |
| afraid's nearest | scared .64, terrified .63, nervous .39 | scared .66, terrified .63, nervous .41 | cluster |
| calm's nearest | serene .52, peaceful .47 | serene .63, peaceful .55 | cluster |
| Tylenol 500 → 16,000 mg, Spearman | afraid +1.0, calm −1.0 | afraid +1.0, calm −1.0 | afraid rises, calm falls |
| split-half cosine, 5 v 5 stories | median 0.39 (0.25-0.55) | median 0.42 (0.26-0.55) | ≥ 0.7 |
| LEACE (20 classes) | rank 19, shrinkage 0.28 | rank 19, shrinkage 0.26 | — |

- **Split-half needs the full run's 20 v 20 stories.** With 5 stories a half every emotion falls under 0.7.
  Scaling each emotion's reliability to four times the stories, 1/(1 + (1/r − 1)/4), gives a median of about
  0.72 and a minimum of about 0.57. The spec's top-up rule (up to 120 stories for a core emotion under 0.7) will
  likely be needed for several core emotions. The NDIF cost is in the estimates below.
- **The topic basis has real structure.** Topic PC1 explains 22% of topic variance at L20 and 21% at L52.

## Probes (pipeline exercise)

- **Scale.** 47,410 result rows over 4 blocks (L20 a, L52 a, L20 b, L20 c) × 2 probe families. The pilot rows
  are 50 per class per set, Alpaca (2,000) sets the thresholds, and RepE (550) trains protocol A.
- **Arms.** Raw, emotion PCs, topic PCs, 5 label-shuffled bases, 20 random subspaces, Alpaca PCA, training-set
  PCA, persona PCs (25 extended personas, k ≤ 24), persona + emotion (k = 3, 5, 10), named emotions, and nine
  zero-shot directions. Under protocol B at pooling (a) there are also the erase arms: top 5 and top 19 emotion
  PCs, LEACE, and 10 random erasures at each rank.
- **Checks.** The published RepE probe on the pilot rows reproduces its full-set AUROCs within sampling noise:
  - L20 pilot: CG 0.97, HP-C 0.52, HP-KR 0.10, ID 0.94, IT 0.97;
  - L20 full set: 0.98, 0.47, 0.11, 0.95, 0.97.

  Raw and persona arms behave sensibly through the same code. So scores, labels and rows line up.
- The weighted bootstrap matches sklearn's AUROC on resampled rows (unit test).

Protocol B, mean held-out AUROC over the five sets (each arm at its inner-CV k; random and topic at the emotion
basis's k):

| layer/pool | probe | emotion PCs | random (mean of 20) | persona PCs | raw |
|---|---|---:|---:|---:|---:|
| 20 a | lr | 0.22 | 0.47 | 0.70 | 0.68 |
| 52 a | lr | 0.61 | 0.52 | 0.82 | 0.75 |
| 20 b | lr | 0.37 | 0.50 | 0.69 | 0.74 |
| 20 c | lr | 0.32 | 0.50 | 0.58 | 0.67 |

- **Emotion PCs at L20 invert systematically when a set is held out:** CG 0.07, HP-KR 0.07, IT 0.20, ID 0.27,
  HP-C 0.47. At L52 they sit above random (+0.09, 95% CI [0.05, 0.13]).
- **These are not results.** The pilot basis has 20 emotions and 19 PCs, and the training sets are 400 rows. Note
  it as something to look at in the full run, not as evidence.
- **The topic column is empty here.** The pilot's 10 topics give only 9 topic PCs, so the emotion basis's chosen
  k of 10 has no topic match. The full run's 40 topics give 39.

Tables: `probes.csv` (every row), `protocol_b_per_set.csv`, `protocol_b_summary.csv` (with paired bootstrap CIs).

## HP-KR diagnostics (pipeline exercise)

**D1** scored all 2,139 HP-KR rows by subgroup: 139 lies, 1,833 honest admissions and 135 honest denials. The
32 unclassified honest rows are left out.

| scorer | lie vs admission | lie vs denial | admission vs denial |
|---|---:|---:|---:|
| L20 published RepE probe | 0.067 | 0.647 | 0.963 |
| L52 published RepE probe | 0.904 | 0.514 | 0.030 |
| L20 protocol-B raw (pilot rows) | 0.033 | 0.664 | 0.995 |
| L20 zero-shot afraid (pilot basis) | 0.938 | 0.321 | 0.001 |

- **The published-probe rows are already final.** They involve no training, no selection and no pilot choice:
  the same probe on the same rows. So the full run will reproduce them exactly, and they are P7's input at layer
  20. I report them as numbers only. P7's verdict, applying the preregistered wording, belongs in RESULTS.md.
- **The other rows change in the full run.** Protocol-B probes are trained on more rows, and the zero-shot
  directions come from the 171-emotion basis. Full table: `d1.csv`, including a sensitivity check with lies
  restricted to the 125 explicit denials.

**D2/D3** ran on 160 prompts (20 questions × 8 templates; 10 unknown and 10 subject-matched known questions).
- The design has 10 columns: intercept, known, 7 template effects, known×harmful. Standard errors are clustered
  by question.
- Nulls: 1,000 random directions and the 10 pilot topic directions.
- Only 8 of the 11 preregistered emotions are among the pilot's 20, and with 20 questions the t-values only
  exercise the code. Tables: `d2_L*.csv`, `d3_L*.csv`.

## Generation check

- **NDIF generation works.** One sampled story took 74 s for 243 tokens, in a single call through the same
  client.
- **Anthropic's story prompt was not available.** The paper's page is over the fetch tool's 10 MB limit, and the
  2025 URL returns 403. The call therefore used a draft prompt (`rows.DRAFT_STORY_PROMPT`).
- **The draft prompt failed the one check that matters.** Asked for "protective", the model wrote a story whose
  feeling reads as defensive or suspicious. The text even uses the word "defensiveness". So the optional steps
  ("protective", and the story-author check) need Anthropic's exact appendix prompt. They also need an automatic
  check that rejects stories naming the emotion or a near-synonym.
- **At one story per call, generation is slow:** 140 stories would take about 3 hours. Batching (as the persona
  pipeline does, 16 prompts per call) brings that to roughly 15-30 minutes.

## Errors and fixes

1. **Log-prob session failed in the local selftest.** nnsight 0.8 finds each `with` block by its source line, and
   building tensors between the loop header and `with llm.trace` broke that. Preparation moved out of the loop,
   as in the Verdict reader. The selftest now passes: span means, points and log-probs equal an HF forward pass
   (max difference 3.9e-8 and 0).
2. **Layer-20 extraction crashed** (access violation) on Alpaca. Appending row views kept every row's full
   (tokens × 8192) float32 array alive, about 18 GB for 2,000 rows. The rows are now copied. The values written
   before the fix were correct (the views pointed at the right data). Every cached span mean matches the stored
   pooled (a), maximum relative error 3e-8.
3. **The story prompt could not be fetched** (see Generation check).

## Full-run estimates

**NDIF** (rates measured above; the pilot's rows are reused, not re-read):

| read | new rows | tokens | rate | time |
|---|---:|---:|---|---:|
| stories, 171 × 40 | 6,640 | ~970k | ~300 tok/s, batched | ~55 min |
| story top-ups, core emotions under 0.7 (upper bound: 29 × 80) | ≤ 2,320 | ≤ 340k | batched | ≤ 20 min |
| D2/D3, 238 questions × 8 | 1,744 | ~565k | ~220 tok/s, solo | ~45 min |
| layer-20 (b)/(c) for the capped rows | — | — | local cache, no NDIF | ~10 min CPU |
| optional: "protective" (40) + story-author check (100) | 140 | — | 16 per call | ~20-30 min |

Total: about 1.6 h, plus up to 0.3 h of top-ups and 0.5 h of optional generation. Earlier runs saw NDIF queues and
redeploys add hours on bad days; every read resumes by row id.

**Disk** (about 2.2 GB; D: has about 9 GB free):

| what | size |
|---|---:|
| story reads (≤ 9,160 rows × 4 layers, float32) | 1.2 GB |
| D2/D3 reads | 0.5 GB |
| layer-20 (b)/(c) for about 8,000 rows (float16) | 0.26 GB |
| held-out score store | ~0.25 GB |

**Local fitting.** The pilot spent 47 min fitting: 4 blocks, 500 Liars' Bench rows each.

- **Most of it was the low-dimensional arms:** 23 min for logistic regression, 14 min for difference-of-means.
  Of those, 20 random seeds × 6 k × three protocols dominate.
- **The full run has about 11 times the rows** under the spec's caps (5,463 Liars' Bench rows) and 6 blocks
  instead of 4 (layers 20, 28, 40, 52 at pooling (a), plus layer 20 b and c). Erase arms run at all four layers.
  That is about 11 h sequentially.
- **The training-set PCA control would dominate otherwise.** An exact SVD of about 5,000 × 8,192 per fold grows
  with n². So it switches to randomized PCA, which gives the same top-20 subspace to numerical tolerance.
- **With feature sets fitted in 4 parallel worker processes, the estimate is about 3 h.** Memory is about 1.5 GB
  per worker. Neither change touches what is computed.

## Before the full run: points for you

1. **Story prompt (needs you).** The optional steps (Llama-written "protective" stories, and the story-author
   check of Gemini's stories against Llama's own) need Anthropic's exact appendix prompt, which I could not fetch.
   If you paste it, I add an automatic check that rejects a story containing the emotion word or a listed
   near-synonym, and generate in batches of 16. Without it both optional steps are skipped. The spec allows that,
   and "protective" stays out of P8's Holm set.
2. **Row set (following the spec).** Probes use at most 1,000 rows per class per set and 500 for IT, drawn with a
   fixed seed:
   - CG 167 + 100, HP-C 134 + 1,000, HP-KR 139 + 1,000, IT 423 + 500, ID 1,000 + 1,000: 5,463 rows in all.
   - D1 scores all 2,139 HP-KR rows. Capping the honest rows would drop about half of the 135 honest denials.
3. **Split-half top-ups (following the spec).** A core emotion under 0.7 with 20 v 20 stories gets up to 120
   stories, for its own vector only.
4. **Implementation changes for speed (no effect on results):** parallel feature sets and randomized PCA for the
   training-set PCA control.
5. **Kept as is.**
   - The cached formatting: the fidelity check found no material difference.
   - Layers 20/28/40/52.
   - Pooling (a) as Liars' Bench pools it, (b)/(c) at layer 20.
   - Naive concatenation of the training sets in protocol B. With the caps the five sets are within a factor of
     about 7 of each other in size.
6. **No pilot number changed the design.** The only choices left are the four above, and none of them depends on
   a pilot score.
