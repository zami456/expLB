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
basis: 25 personas of the extended 36-role bank (`data/persona_banks/extended_kd/`), 24 PCs. That bank's
knowledge-denial roles were designed with HP-KR's inversion in mind.

On 2026-10-30 the user asked for the published persona space instead: arXiv 2605.09391's 8 deception and 7 honesty
personas from the original 26-role bank (`data/persona_banks/vectors_s3/`, equal to `paper16_L20.pt` at block 20),
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

## Every Liars' Bench model (user request, 2026-10-30; fixed before any read)

The user asked for the emotion coordinates to be tested on the other Liars' Bench models, as the thesis's cross-model
replication tested its own detectors. Those models are Qwen2.5-72B, Mistral-Small-3.1-24B and Gemma-3-27B. The user
chose the thesis's 4 layers plus the band, and the core study: no MLP, no erasure (P6), no HP-KR diagnostics. Every
constant is in `xmodel_spec.py`, and its SHA-256 is recorded in `E:/aletheia/logs/freeze_emotion_xmodel.txt` before
any read. `xmodel.summary` refuses to run without that match.

- **Blocks.**

  | model | named blocks (L20/36/40/44) | band |
  |---|---|---|
  | Qwen2.5-72B | 20/36/40/44 | 35-45 |
  | Mistral | 9/18/20/22 | 17-23 |
  | Gemma | 8/28/31/34 | 27-35 |
  | Llama | 20/36/40/44 | 35-45 (the study's existing runs) |

- **Reads.** Each model reads Llama's 8,280 stories and the neutral dialogues through itself, re-tokenised.
  - Its own Liars' Bench rows, RepE pairs and Alpaca replies are the thesis's pooled reads where they exist.
  - Elsewhere they are this study's reads of the same cached tokenisation.
  - The sets are capped as in the Llama study.
- **Analysis.** Phase 1's bases; phase 3's feature sets (persona bases from the model's own primary bank) and
  protocols A, B and C, logistic regression only.
- **Gates.** Parity of the new Liars' Bench reads (cos ≥ 0.9998); valence per block, flagged.
- **Predictions, fixed now** (declared; Llama's results are known):
  - **P1-P5.** Llama's rules on each model's 4 named layers, where its sets allow (P2, P5: IT; P3: HP-C and HP-KR).
    A prediction holds for a model at ≥ 3 of the 4 layers.
  - **XA1.** Per model, all-170 beats raw activations under protocol B (CI above 0) at ≥ half of its band blocks.
  - **XA2.** Per model, all-170 beats the 170-d shuffled control under protocol B at ≥ half of its band blocks.
- **Reported without a prediction:**
  - protocols A and C;
  - train-one-test-all transfer at the named layers beside the thesis's detectors;
  - the band curves;
  - the basis checks.
- **Regression.** The new runner reproduces Llama's stored block-40 emotion-PC unit exactly (175 of 175 rows).

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

## Mahadik & Skapars' ten datasets (user request, 2026-10-31; fixed before any read of the new sets)

The user asked for Llama to be tested on the datasets of arXiv 2605.09391 (Mahadik & Skapars, "Do Linear Probes
Generalize Better in Persona Coordinates?", the source of the published persona space). That paper's question is
whether probes in a 3-d persona space generalise across datasets better than probes on raw activations. It was asked of
Llama-3.2-3B and Llama-3-8B; here it is asked of the emotion basis on Llama-3.3-70B. The user chose:
- sycophancy added;
- the paper's splits;
- blocks 35-45.

Every constant is in `ms10_spec.py`. Its SHA-256, with those of `results/emotion_coords/ms10/sources.json` and
`splits.csv`, is recorded in `E:/aletheia/logs/freeze_emotion_ms10.txt` before any read. `ms10.score` refuses to run
without that match.

- **Sets** (label 1 = deceptive or sycophantic).
  - **Deception: Llama-3.3-70B's own rollouts.**
    - AILiar and Roleplaying are labelled by the repo's rules.
    - ConvGame, InstrDec and HarmPressureChoice are the study's Liars' Bench rows at the band.
  - **Sycophancy: read as written.** All five sets ship labelled replies, so no generation and no judge are needed.
    - Azarbal: a sycophantic and a non-sycophantic reply per prompt.
    - Papadatos: the same, from Rimsky et al.'s prompts.
    - The three OEQ sets of Cheng et al., as the first author prepared them: 200 human-labelled advice answers each,
      100/100.
- **Splits.**
  - At most 800 rows a set. InstrDec and HarmPressureChoice are drawn at the full set's share of lies; Sycophancy is
    400 prompts with both replies.
  - A seeded stratified-group 75/25 split per set, with groups being everything before the scored reply.
  - The OEQ sets share records (552 distinct texts, one of them labelled differently in two records). They are split
    once, jointly, on the triple of human labels.
  - Some sizes differ from the paper's table because these rollouts are not its data:

    | set | train / test |
    |---|---|
    | AILiar | 201 / 65 |
    | Roleplaying | 215 / 72 |
    | ConvGame | 197 / 70 (56 prompts) |
    | InstrDec, HarmPressureChoice, Sycophancy | 600 / 200 each |
    | Open-Ended Sycophancy | 78 / 28 |
    | OEQ sets | 146-152 / 48-54 |

  - HarmPressureChoice's test split holds only 11 lies.
- **Arms.**
  - all-170, raw activations, and the band study's 170-d controls: shuffled ×5, random ×5, Alpaca PCs, training-set
    PCs.
  - At the paper's k = 3: emotion PCs 1-3 and the paper's two persona spaces (deception 8 v 7; sycophancy 5 v 5).
- **Probe and scoring.**
  - Standardised L2 logistic regression; C by inner stratified-group 5-fold CV on the training rows only.
  - One probe per train split and one on all ten, each scored on every test split.
  - 1,000 stratified bootstrap draws per test split, shared by every arm, source and block.
  - Regions: the deception and sycophancy off-diagonals (20 cells each), cross-behaviour (50), the diagonal and
    train-on-all.
- **Predictions, fixed now** (a block counts when the paired 95% CI is above 0):
  - **M1.** all-170 beats raw activations on the deception region at ≥ 6 of the 11 blocks.
  - **M2.** all-170 beats every 170-d control on the deception region at ≥ 6 of 11.
  - **M3, M4.** The same two on the sycophancy region.
- **Reported without a prediction:**
  - the cross-behaviour, diagonal and train-on-all regions;
  - the k = 3 arms, including the paper's claim (its persona coordinates beat raw activations off the diagonal) at
    70B;
  - the matrices at block 40 and over the band.
- **Already seen.** ConvGame, InstrDec and HarmPressureChoice are Liars' Bench sets the earlier Llama studies used.
  AILiar, Roleplaying and the five sycophancy sets are new.
- **Checks.**
  - Every new span decodes to its reply exactly.
  - The reader's redeploy check on 20 stored Liars' Bench parity rows (cos ≥ 0.9998) runs before the bulk reads.
  - The fit path must reproduce phase3_band's protocol B AUROCs at block 40.

## The thesis's detectors on the ten datasets (user request, 2026-10-31; fixed before any of them is scored there)

After the emotion results, the user asked for the thesis's own detectors on the same ten datasets. The question is
whether the thesis's extension of the paper (the knowledge-denial personas and the knowledge direction) adds anything
on the paper's own data.

Everything about the data is `ms10`'s: rows, splits, block reads and bootstrap draws, so comparisons with its arms are
paired. Every constant is in `ms10_thesis_spec.py`. Its SHA-256 is recorded in
`E:/aletheia/logs/freeze_emotion_ms10_thesis.txt`, and `ms10_thesis.score` also checks `ms10`'s freeze.

- **Layers.** 36, 40 and 44: the thesis's layers in the band, where K and the RepE probe exist.
- **Detectors.** These are the thesis's, built exactly as `transfer_alignment.scores` builds them; nothing is chosen
  here.
  - **Fixed** (scored on every test split, no training on these sets):
    - D2, the RepE base probe;
    - D3, the AISI probe (40 and 44);
    - D5-D9, the extended bank's contrasts;
    - K;
    - D5 + K and D9 + K.
  - **Trained per source with the thesis's probe** (`make_probe`):
    - D1 on raw activations;
    - D4 on the extended top 3 PCs;
    - the paper's top 3 PCs under the same probe.
- **Regions.**
  - A fixed detector's region is its mean over the region's test sets.
  - A trained one's is its off-diagonal mean, as in `ms10`.
  - Either way each test set is weighted equally.
  - The "new deception" region is AILiar and Roleplaying, which the thesis never scored.
- **Predictions, fixed now** (deception region; the paired CI above 0 at ≥ 2 of the 3 layers):
  - **TD1.** D9 + K beats D1.
  - **TD2.** D9 beats D5.
  - **TD3.** D9 + K beats D9.
  - **TD4.** D4 beats the paper's PCs under the same probe.
- **Reported without a prediction:**
  - the sycophancy and new-deception regions;
  - the other detectors;
  - every detector against `ms10`'s arms.
- **Already seen.** D9 + K was selected on Liars' Bench (three of these deception sets). `ms10`'s arms are already
  scored here; none of these detectors is.

## Curing instructed deception (user request, 2026-10-31; fixed before any erasure is scored)

The user's idea: erase what a probe can learn from instructed deception besides the lie, then train on the cured rows
and test on the other sets. Their loop was: random two-class labels, a probe, LEACE when the probe succeeds, repeat.
That loop cannot do this. A probe never predicts labels that are independent of the rows on held-out rows, yet it
always separates them in-sample (600 rows, 8,192 dimensions). And each LEACE round on random labels erases a random
whitened direction.

The user chose the version that targets the idea: erase InstrDec's within-class variance (nuisance attribute
projection). Their loop runs as a control, without its stopping test, and both of that test's numbers are recorded.
Every constant is in `ms10_cure_spec.py`. Its SHA-256 is recorded in `E:/aletheia/logs/freeze_emotion_ms10_cure.txt`,
and `ms10_cure.score` also checks the `ms10` and `ms10_thesis` freezes.

- **Setting.**
  - The thesis's raw probe D1 (`make_probe`), trained on InstrDec's 600 `ms10` training rows after each erasure.
  - Every erasure is fitted on those rows alone and applied unchanged to every test split.
  - `ms10`'s draws; layers 36, 40 and 44.
  - The unerased probe must reproduce `ms10_thesis`'s D1 AUROCs.
- **Erasures at k = 5, 10, 20, 50, 100, 200 and 400 directions.**
  - Nuisance: the within-class top-k PCs.
  - Total: the total top-k PCs.
  - Random: 5 seeds.
  - The user's random-label LEACE loop: k rounds, 5 seeds.
  - Also keeping only the top-k PCs, plus k = 170 as the reference from `ms10`.
- **Regions.**
  - Deception transfer: the other four deception sets.
  - New deception: AILiar and Roleplaying.
  - Sycophancy.
  - In-distribution: InstrDec's test split.
- **Predictions, fixed now** (deception transfer, the nuisance erasure averaged over the 7 ranks; the paired CI above 0
  at ≥ 2 of the 3 layers):
  - **C1.** It beats no erasure.
  - **C2.** It beats random erasure at the same rank.
  - **C3.** It beats the random-label LEACE loop at the same rank.
  - **C4.** It beats keeping the top 170 PCs.
- **Reported without a prediction:**
  - every variant at every rank;
  - the other regions;
  - the random-label probe's held-out and in-sample AUROC.
- **Already seen.** D1 trained on InstrDec is already scored on these splits (`ms10_thesis`), and so is `ms10`'s
  training-set-PC arm. No erasure has been.

## Lie coordinates from RepE (user request, 2026-10-31; fixed before any lie coordinate is fitted)

The user asked for lie directions extracted by INLP to be used as probe coordinates. They chose the version that tests
something new: the directions are learned from labelled lies outside the ten sets, then used exactly as the persona
and emotion bases were. Every constant is in `ms10_lie_spec.py`. Its SHA-256 is recorded in
`E:/aletheia/logs/freeze_emotion_ms10_lie.txt`, and `ms10_lie.score` also checks `ms10`'s freeze.

- **Lie directions.** INLP on the RepE instructed-honesty pairs (D2's training data), at each block of `ms10`'s band
  (35-45).
  - Each round fits the thesis's probe on RepE's train split, after the directions found so far are projected out.
  - Its direction is kept while the probe still finds the lie on RepE's validation split (held-out AUROC ≥ 0.55).
    The validation rows exclude the one statement both halves share.
  - The number of directions, k, is set this way on RepE alone, capped at 100.
- **Arms.** All use `ms10`'s probe, rows, splits and draws.
  - The k lie directions.
  - The first 3 lie directions, against D4's extended top 3 PCs.
  - The training set's own top k and top 3 PCs.
  - k random directions (5 seeds).
  - The first lie direction scored zero-shot.
  - `ms10`'s stored arms, re-scored: raw activations, all-170, emotion PCs, the training set's top 170 PCs, and the
    paper's two persona spaces.
- **Predictions, fixed now** (deception region; the paired CI above 0 at ≥ 6 of the 11 blocks):
  - **L1.** The lie coordinates beat raw activations.
  - **L2.** They beat random k-dimensional subspaces.
  - **L3.** They beat the training set's own top k PCs.
  - **L4** (two-sided). At k = 3, lie coordinates against D4: whichever arm's CI excludes 0 in its favour at ≥ 6
    blocks wins; otherwise neither.
- **Reported without a prediction:**
  - the other regions;
  - k and the INLP trace;
  - the zero-shot direction;
  - every new arm against the emotion arms and the paper's PCs.
- **Already seen.** `ms10`'s arms, D2 and D4 (under the thesis's probe) are scored on these splits, as is the cure
  study's post hoc lead. No lie coordinate has been fitted.

## Lie coordinates on Liars' Bench (user request, 2026-10-31; fixed before any probe is fitted here)

The user asked for the same comparison on Liars' Bench's five sets: train on one, test on the other four. Every
constant is in `lie_lb_spec.py`. Its SHA-256 is recorded in `E:/aletheia/logs/freeze_emotion_lie_lb.txt`.

- **Setup.**
  - `ms10_lie`'s arms and lie directions (k = 100), under `ms10`'s probe.
  - Phase 3's protocol C rows (the study's capped sets) and folds (stratified 5-fold for C).
  - Blocks 35-45, with the first lie direction and D9 zero-shot.
  - RepE, the lie directions' source, is neither a training nor a test set.
- **Predictions.** LB1-LB4 repeat L1-L4 on the 20 train-one-test-other cells (≥ 6 of 11 blocks; LB4 two-sided).
- **Already seen.** Liars' Bench is the thesis's development data, and `ms10_lie`'s results on the ten datasets are
  known. This is a re-test, not a held-out test.

## Averaged lie directions (user question, 2026-10-31, after the lie-coordinate results; post hoc)

The user asked about averaging the 100 INLP directions into one vector. Every constant is in `lie_mean_spec.py`. Its
SHA-256 is recorded in `E:/aletheia/logs/freeze_emotion_lie_mean.txt`.

- **Directions,** all zero-shot:
  - the unit sum of all 100 directions;
  - the unit sum of the first 10;
  - the first direction alone;
  - RepE's difference of class means;
  - D9.
- **Scored on** the same rows and draws as `lie_lb` (Liars' Bench) and `ms10` (the ten datasets), with the raw probe's
  transfer as the trained reference.
- **Predictions.**
  - **A1.** The 100-direction average beats the first direction on Liars' Bench at ≥ 6 of 11 blocks.
  - **A2.** The same on the ten datasets' deception sets.

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
