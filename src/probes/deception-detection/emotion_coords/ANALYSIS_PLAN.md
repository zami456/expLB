# Analysis plan: how each preregistered prediction is judged

Written 2026-10-29, after the pilot and before any full-run probe was fitted. It is committed on its own. It fills
in details the preregistration (`PREREGISTRATION.md`, commit `ae8e0ae1`) leaves open, and changes no prediction.

**What had been seen.** The pilot had shown two things by then:
- Protocol-B AUROCs from a 20-emotion basis on 50 rows per class.
- The published probe's D1 AUROCs on every HP-KR row, at blocks 20 and 52. These do not change in the full run.

The rules below are the plain reading of the preregistered wording; none is tuned to those numbers. Where a rule
touches a number that was already seen (P7), that is said explicitly.

## Common to every prediction

- **Rows.** The spec's caps, seeded (`phase2 subset`): CG 167 + 100, HP-C 134 + 1,000, HP-KR 139 + 1,000,
  IT 423 + 500, ID 1,000 + 1,000. Alpaca (2,000) sets thresholds. The instructed pairs (550) train protocol A.
  D1 scores every HP-KR row.
- **Layers.** Blocks 20, 28, 40 and 52 stand for the spec's 16, 27, 40 and 53 (`PREREGISTRATION.md`, item 1).
- **Probe family.** P1-P4 and P6 are judged with the L2 logistic-regression probe (the spec's first family, and
  the Liars' Bench probe's). The difference-of-means probe is reported beside it and never used to judge.
- **k and C.** Chosen inside each training fold by inner cross-validation. Under protocol B that is leave one
  training set out. Each held-out set has its own fold, so its own k.
- **CIs.** The paired bootstrap draws 1,000 stratified resamples of each held-out set, shared by the two arms
  being compared. A "mean over held-out sets" is taken within each resample, and the CI is the 2.5th-97.5th
  percentile.
- **Random-subspace control.** The mean AUROC over its 20 seeds, taken within each resample.
- **"Held".** A prediction evaluated per layer counts as held if it holds at 3 or more of the 4 layers (pooling a).

## Exploratory addition (user request, 2026-10-29, before any full-run probe)

The whole emotion subspace is projected onto and probed at fixed k = 170: all PCs of the 171 centred emotion
vectors.

- **Controls at the same dimension:**
  - the five label-shuffled bases (170 PCs each);
  - 5 random 170-dimensional subspaces;
  - the top 170 PCs of Alpaca's activations;
  - the top 170 PCs of the training rows, fitted inside each fold.
- **Scope:** protocol B, pooling (a), all four layers, both probe families.
- **It does not judge any prediction.** It is its own family, kept out of the emotion PCs' preregistered k choice
  ({2, 3, 5, 10, 20}). The report gives mean held-out AUROCs and paired-bootstrap differences (emotion minus each
  control).

## Exploratory addition: MLP probes after projection (user request, 2026-10-29, before any full-run probe)

A third, nonlinear probe family runs on every projected feature set: emotion PCs, topic, label-shuffled, random
(5 seeds), Alpaca PCs, training-set PCs, persona PCs, persona + emotion, named emotions, and the 170-dimensional
sets above. Raw activations are not included.

- **Architecture:** standardise; one hidden layer of 64 ReLU units; L2 weight decay from {0.01, 1}; Adam; early
  stopping on 10% of the training rows; seed 0.
- **Selection:** k and the weight decay by inner leave-one-set-out CV, as for logistic regression.
- **Scope:** protocol B, pooling (a), all four layers. Controls are matched as for P1, at the emotion basis's
  chosen k.
- **It judges no prediction.** The report gives mean held-out AUROCs and paired differences (`exploratory_mlp.csv`).

## Persona basis: the published persona space (user decision, 2026-10-30, after results were seen)

The preregistration says only "persona-PC probes". The analysis code fixed before the full run used the thesis D4
basis: 25 personas of the extended 36-role bank (`persona including knowlede denial/`), 24 PCs. That bank's
knowledge-denial roles were designed with HP-KR's inversion in mind.

On 2026-10-30 the user asked for the published persona space instead: arXiv 2605.09391's 8 deception and 7 honesty
personas from the original 26-role bank (`persona vectors/vectors_s3/`, equal to `paper16_L20.pt` at block 20),
centred, 14 PCs, oriented along the deception axis (`pilot.persona_basis(layer, "published")`).

- **What had been seen.** By then the protocol-B and protocol-C point AUROCs at blocks 20, 28 and 40 had been seen
  for every preregistered arm, including the extended persona arm. No published-persona probe had been fitted.
- **What changes.** P2 and P4 are judged with the published basis (arms `persona_pub_pc` and `persona_pub_emotion`;
  k from {2, 3, 5, 10}, since the basis has 14 PCs). The same rules otherwise apply.
- **What stays.** The extended-basis arms (`persona_pc`, `persona_emotion`) are still reported as a sensitivity check
  (`P2_extended`, `P4_extended`). Because the switch came after results were seen, RESULTS.md gives both and says so.

## Post hoc band around block 40 (user request, 2026-10-30, after results were seen)

Emotion PCs did well only at block 40 among the four preregistered blocks. The user then asked for the whole
experiment on "layers 35-45" only. The pooled Liars' Bench activations exist every fourth block, so the band is
blocks 36, 40 and 44.

- **What is run.** Everything, on 36 and 44 (block 40 is shared with the main run):
  - the stories, neutral texts, D2/D3 rows and Tylenol prompts are read at 36 and 44 (`*_band` read folders);
  - the bases are built by the same recipe and top-ups (`phase1 final --layers 36,44 --suffix _band`);
  - phase 3 runs every arm, erasures, the 170-d arms, zero-shot and MLP (parts `band`, results `phase3_band`);
  - the summary covers 36/40/44, and D1, D2/D3 and the figures are tagged `_band`.
- **Extended to every block 35-45 (user, 2026-10-30, after the 36/40/44 results).** The eight blocks with no stored
  pooled array (35, 37, 38, 39, 41, 42, 43, 45) get their Liars' Bench activations from this study's reader:
  - every row the probes use (`rows lb_band`: the capped rows, all of HP-KR, Alpaca, the RepE pairs), pooling (a)
    as the detected span's mean, accumulated in float32 and transferred as bfloat16 (`read --dtype bfloat16`);
  - checked on the 50 parity rows at blocks 36/40/44 against the stored arrays: median cos 0.99996-0.99997, minimum
    0.99973-0.99984, within the envelope of separate NDIF passes already seen in the pilot (0.99967-0.99976);
  - written in the stored layout by `lbdata.write_band_pooled` (rows never used are NaN, which `load` drops);
  - the stories, neutral texts, Tylenol prompts and D2/D3 rows read the same way at those blocks.
  There is no published RepE probe at those blocks, so protocol A-published is absent there.
- **Status.** It is post hoc and exploratory. It changes no preregistered verdict, which stays at blocks 20/28/40/52.
  "Held at 3 of 3" in its summary means all three band blocks. The choice of band followed the block-40 result, so
  a band effect is a description of where emotion PCs work, not a test of whether they do.

## Out-of-sample test on Gender-Secret at block 40 (user request, 2026-10-30; fixed before GS is scored)

GS is Liars' Bench's LoRA fine-tune of Llama-3.3-70B that believes the user is male and denies knowing it. No arm,
k, layer or rule above has seen it. Its lies and its honest answers are both denials (gender questions vs other
attributes), so it is the wording-matched test. The thesis read it at block 40 (`activations/gender_secret/`), twice:
through the fine-tune (*adapted*: 392 rows, 116 lies) and on the same tokens through the base model (*base*: the
control).

- **Training.** Each arm is fitted on all five LB-5 sets (capped rows), with k and C chosen by inner
  leave-one-set-out CV across them, the same machinery as protocol B. Then it scores GS. The arms are those of phase
  3 at block 40, plus the 171 emotion vectors. Also scored: the published RepE probe and the zero-shot directions.
- **Metrics.** AUROC with 1,000 stratified bootstrap resamples (paired between arms; seed means within each
  resample), and balanced accuracy and recall at the Alpaca 1% FPR threshold.
- **Predictions, fixed now.** They arise from looking at the band and are therefore declared, not preregistered:
  - **G1.** All 170 emotion PCs beat (a) raw activations and (b) the five 170-d label-shuffled controls (mean) on
    the adapted read: paired difference > 0 with CI excluding 0, both.
  - **G2.** Emotion PCs (k by inner CV) beat the 20 random subspaces at the same k (CI excluding 0) and the topic
    basis at the same k by ≥ 0.03.
  - **G3.** Any arm with adapted AUROC > 0.6 scores lower on the base read (paired CI excluding 0). If it does not,
    it reads the question or the denial wording rather than the lie.

## Aletheia's Quest band, mid ± 4 (user request, 2026-10-30; fixed before any band read)

The user chose every block of mid ± 4: Qwen3.5-27B 28-36, Gemma-3-27B-it 27-35. The middle block had been scored,
so the band is post hoc in the sense of the Llama band. Every constant is in `aletheia_band_spec.py`, and its
SHA-256 is recorded in `E:/aletheia/logs/freeze_emotion_band.txt` before the first band read. `aletheia_band.score`
refuses to run without that match.

- **Reads.** Every input is read again at the nine blocks by `reader/ec_read.py`, so the blocks share one pipeline:
  - the stories and neutral dialogues through each base model;
  - the Liars' Bench training rows of the primary source. Gemma's own CG, IT and ID come from the tokenisation its
    pooled reads cached. Qwen2.5's HP-C, HP-KR and ID are rendered as D4 (LB) read them;
  - the Alpaca controls;
  - every test row through the organism that wrote it, one row per pass. The merged copies are read directly;
    BotC and Echo go through their adapters, attached by NDIF at scale α/r = 2. BotC keeps the thesis's tail cuts.
- **Unchanged from the middle-block test:** the recipe, the probe and its C grid, the six feature sets, the 8 sets,
  and the bootstrap draws. Only the primary source and the organism read are scored.
- **Gates** (a failure stops the run for the user's decision):
  - parity against the thesis's stored reads at the shared blocks, cos ≥ 0.9998 (0.999 for Echo, which the thesis
    read with int8 factors). The check covers every test row and a 20-row subset of each training and Alpaca
    pipeline;
  - |r(PC1, valence)| ≥ 0.5 per block. A block under it is reported and flagged, not dropped.
- **Predictions, fixed now:**
  - **HB1.** All-170 beats raw activations, with the paired CI above 0, at ≥ 5 of the 9 offsets. An offset pairs
    Qwen3.5's block 32 + j with Gemma's 31 + j in the 8-set mean.
  - **HB2.** All-170 beats each 170-d control (shuffled, random, Alpaca PCs, training-set PCs) at ≥ 5 of the 9
    offsets.
- **Reported without a prediction:**
  - each offset's mean, rank of 31, and rank among the 18 white-box entries;
  - the best offset (post hoc);
  - offset 0 against the frozen middle-block test, as a check of the new reads.

## Aletheia's Quest band, RepE-trained (user request, 2026-10-30; fixed before any RepE band read)

The user asked to "train on RepE on the band" after the band's Liars' Bench results were seen. This is post hoc.
Every constant is in `aletheia_band_repe_spec.py`, and its SHA-256 is recorded in
`E:/aletheia/logs/freeze_emotion_band_repe.txt` before the reads.

- **Training rows.** The band's design is unchanged except for its training rows, which are R1's RepE rows:
  - Gemma: its pooled train split (448), from the cached tokenisation;
  - Qwen3.5: the first 80% of `repe.py`'s rows (440), rendered as `repe.read` rendered them.

  They are read at every band block, one row per pass. C is chosen by 5-fold CV grouped by statement, as in the
  middle-block test's RepE source.
- **Gate.** Every row is compared with the thesis's reads at the shared blocks (cos ≥ 0.9998):
  - Qwen3.5 32, against the R1 probe's per-token reads averaged over each span;
  - Gemma 28/31/34, against its pooled reads.
- **Test sets.** The band's 6 readable sets (its recorded deviation), and the 8 once Echo and BotC can be read.
- **Predictions, fixed now:**
  - **HR1.** All-170 (RepE) beats raw activations (RepE), with the paired CI above 0, at ≥ 5 of the 9 offsets.
  - **HR2.** All-170 (RepE) beats each 170-d control (RepE) at ≥ 5 of the 9 offsets.
- **Reported without a prediction:**
  - all-170 (RepE) minus all-170 (LB) and minus raw (LB) at each offset, paired on the same draws;
  - the ranks;
  - the best offset (post hoc).

## Gender-Secret with the fine-tune's emotion vectors (user request, 2026-10-30; fixed before any story is read through the fine-tune)

The GS test above used the base model's emotion vectors, and RESULTS.md names that as its caveat. Re-deriving them
inside the fine-tune was blocked until NDIF installed peft on its Llama-3.3-70B deployment (fixed 2026-10-30).
`emotion_coords/gs_ft.py` runs it.

- **Reads.** The same 8,280 stories and 1,086 neutral dialogues (Llama's rows files) are read at block 40 through
  Liars' Bench's Gender-Secret LoRA, which NDIF attaches itself (`ec_read.py --peft
  Cadenza-Labs/llama-70b-3.3-it-lora-gender-secret-male`).
- **Gates.**
  - The 392 GS rows, read the same way, must match the thesis's stored adapted reads of block 40 (cos ≥ 0.9998).
  - The new basis must have |r(PC1, valence)| ≥ 0.5.
- **Bases.** Phase 1's recipe, unchanged (Llama's top-ups).
- **Arms rebuilt from the fine-tune's bases.** Every GS-test arm whose basis comes from the stories:
  - emotion PCs, the topic basis, the shuffled bases;
  - persona + emotion, the 29 named emotions;
  - all 170 PCs, the 171 vectors, the 170-d shuffled bases;
  - the zero-shot directions.
- **Everything else unchanged.** Training is the base model's LB-5 reads of block 40, with k and C chosen by
  leave-one-set-out CV. The arms that do not use the stories are the GS test's own fits: raw, random, Alpaca PCs,
  training-set PCs, persona PCs and the published probe.
- **Predictions.** G1-G3's rules, applied with the fine-tune's bases (G1-ft, G2-ft, G3-ft).
- **Reported without a prediction:**
  - each story-dependent arm's GS AUROC with the fine-tune's bases minus with the base model's (paired);
  - how far the fine-tune moves the emotion vectors (cosines).
- **Label.** Declared: the GS results with the base model's bases are known.

## Aletheia's Quest transfer (user request, 2026-10-30; fixed before any story is read through the new models)

The all-170 arm, scored on Aletheia's Quest as the thesis scored its detectors there (`scripts/aletheia/`). Every
constant is in `aletheia170_spec.py`, and its SHA-256 is recorded in `E:/aletheia/logs/freeze_emotion170.txt` before
the first NDIF read. `aletheia170.score` refuses to run without that match. The arm's results on Llama are known at
declaration, including the GS failure of G1. The predictions are therefore declared, not preregistered. The thesis
opened these test labels for its own cells; no basis, probe or C here is chosen with them.

- **Models and blocks.** The organisms' base models at the thesis's middle block: Qwen3.5-27B at block 32 of 64, and
  Gemma-3-27B-it at block 31 of 62.
- **Basis.** Built inside each base model by phase 1's recipe, unchanged:
  - Llama's 8,280 stories (with its top-ups) and every neutral dialogue, re-tokenised by the model.
  - The mean from position 50 onward.
  - The neutral PCs explaining 50% of the variance removed; all 170 PCs kept; 5 shuffled bases.
- **Reads.** `reader/ec_read.py --model qwen35|gemma`.
  - This replaces the thesis's `aletheia/read.py`: that reader imports packages absent from the nnsight 0.8 venv.
  - Gate: 20 stored rows per model re-read must match at cos ≥ 0.9998 (Qwen3.5 Alpaca rows, Gemma dev rows; no
    test row).
- **Probe.** Standardised L2 logistic regression on the coordinates, with C chosen by inner CV on the training rows only.
  - **Primary training source: Liars' Bench**, the rows of the thesis's post hoc D4 (LB):
    - Gemma's own CG, IT and ID;
    - Qwen2.5's HP-C, HP-KR and ID text, read through Qwen3.5 base.
    - Inner CV leaves one set out.
  - **Secondary source: RepE.** R1's training rows; CV grouped by statement.
  - **Secondary source: dev.** R2's rows; CV leaves one organism out, or is stratified 5-fold for Gemma's single set.
- **Scoring.**
  - **Primary:** each test row read by its organism, at the middle block, reasoning dropped, content span.
  - **Secondary:** the LB-trained probes on the base model's read of the same rows, the thesis's text control.
  - **Datasets:** the thesis's 8 (HG-G and HG-N not run).
  - **Statistics:** AUROC with `score.run`'s bootstrap draws and random band; the mean; the rank among the 30
    entries, with its interval.
  - **Thesis comparators:** D9 + K (headline), R1, R2 and D4 (LB), re-scored on the same draws for paired deltas.
- **Gates** (a failure stops the run for the user's decision):
  - reader parity;
  - |r(PC1, valence)| ≥ 0.5 for each new basis;
  - the dev known-answer gate: all-170 (LB) on the base model's instructed-deception dev rows above the 97.5th
    percentile of 1,000 random directions.
- **Predictions, fixed now:**
  - **H1.** All-170 (LB, organism read) beats raw activations (LB) in mean AUROC over the 8 sets: paired difference
    > 0 with CI excluding 0.
  - **H2.** All-170 (LB) beats each 170-d control: shuffled (mean of 5), random (mean of 5), Alpaca PCs and
    training-set PCs. Each CI excludes 0.
  - Rank, the deltas to the thesis's cells and the secondary sources are reported without a prediction.

## Top-k emotion PCs on Aletheia's Quest (user request, 2026-10-30, after the all-170 results were seen)

The user asked to "test on top k". This is the study's preregistered emotion-PC arm in place of all 170: the first k
PCs of the same saved emotion bases, with k ∈ {2, 3, 5, 10, 20} and C chosen together by the source's inner CV.
Everything else is the all-170 test's: the bases, sources, reads, datasets, draws and band. Every constant is in
`aletheia_topk_spec.py`, and its SHA-256 is recorded in `E:/aletheia/logs/freeze_emotion_topk.txt` before any top-k
probe is scored.

- **Controls at the same k** (P1's matching): the topic basis, the 5 shuffled bases, and 20 random subspaces.
  - The topic basis is built per model from the same rank-0 stories, with the same denoising.
  - k_e is the emotion basis's choice for each model and source.
- **Prediction (P1's rule, transplanted):**
  - **TK1.** On the 8 sets, Liars' Bench-trained, organism read: emotion PCs at k_e minus random at k_e (mean of 20)
    ≥ 0.03, with the paired CI excluding 0. The same must hold against the topic basis at k_e.
  - Labelled post hoc: the request followed the all-170 results, though no top-k test score had been seen.
- **Reported without a prediction:**
  - every fixed k;
  - the shuffled control;
  - raw activations, all-170 and the thesis's headline (paired on the same draws);
  - the RepE and dev sources.
- **Gate:** the dev known-answer gate, for emotion PCs at k_e.

## Per prediction

- **P1.** Let Δ_rand be the mean over the five held-out sets of [AUROC(emotion PCs at k_t) − mean AUROC(random at
  k_t)], where k_t is the emotion basis's choice in fold t. Let Δ_topic be the same with the topic basis at k_t.
  P1 holds at a layer if both are ≥ 0.03 and both CIs exclude 0.
- **P2.** With IT held out: AUROC(emotion PCs, its k) − AUROC(persona PCs, the persona basis's own k) > 0, with
  the CI excluding 0.
- **P3.** With HP-C held out, and separately with HP-KR held out: the point estimate of AUROC(emotion PCs) − mean
  AUROC(random at the same k) is below 0.02 in both.
- **P4.** Persona + emotion is taken at the inner-CV choice among k_p = k_e ∈ {3, 5, 10}. P4 holds if its mean
  held-out AUROC minus that of persona PCs (at their own k) is ≥ 0.02. The rule uses the point estimate; the CI
  is reported.
- **P5.** At least one of guilty, afraid and desperate has zero-shot AUROC > 0.6 on IT and < 0.55 on ID (point
  estimates, same direction, sign fixed in advance).
- **P6.** For each erasure (top 5 emotion PCs, top 20 emotion PCs, LEACE), compare it with a random erasure of
  the same rank (rank 5, 20 and LEACE's rank; 10 seeds, mean taken within each resample).
  - δ = mean held-out AUROC(raw, emotion erased) − mean held-out AUROC(raw, random erased).
  - P6 holds at a layer if, for at least 2 of the 3 erasures, δ > 0 with the CI excluding 0 and erasing emotion
    raises AUROC over raw (point estimate).
  - The erased arms reuse the raw probe's fold-chosen C.
- **P7.**
  - The protocol-A probe is the published RepE-trained probe of block 20, scored per token and averaged (pooled,
    which is equivalent). Its HP-KR AUROC on the capped rows must be below 0.5; all rows are reported too.
  - D1 readings use 95% CIs from 1,000 stratified resamples of the two subgroups compared. "Well under 0.5" means
    the CI lies below 0.5; "near 0.5" means the CI includes 0.5.
  - (a) holds if lie vs admission is well under 0.5 and lie vs denial is near 0.5. (b) holds if the roles are
    reversed. Otherwise neither holds.
  - These AUROCs were seen in the pilot. The CI reading is the default meaning of the words, not a threshold
    fitted to them, and RESULTS.md says so.
- **P8.** At block 52, D2's known × harmful interaction t-statistic is compared with two nulls: 1,000 random unit
  directions (seed 0) and the 40 topic directions.
  - (a) holds if any of compassionate, empathetic, sympathetic, kind or loving lies above the 95th percentile of
    both nulls. "Protective" was not built: Anthropic's story prompt was unavailable.
  - (b) holds likewise for any of afraid, terrified, angry, hostile, suspicious or vigilant.
  - Holm-adjusted one-sided p-values from the random-direction null are reported across these 11 emotions.
