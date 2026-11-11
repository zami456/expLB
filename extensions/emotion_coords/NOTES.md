# Notes: what the emotion-coordinate study reuses, and where it lives

Phase 0 inventory, 2026-10-29. DD below is `D:\expLB\liars-bench\src\probes\deception-detection`
(`deception_detection.utils.REPO_ROOT`). Plan: `~\.claude\plans\parsed-scribbling-sedgewick.md`.
Machine-readable checks are in `results/emotion_coords/phase0/` (`python -m extensions.emotion_coords.phase0 report`).

## Machine and environments

- Gigabyte G5 MD laptop, Windows 11, 16 GB RAM, RTX 3050 Ti (4 GB). It is not the Legion 5 / RTX 5060
  the spec names. Paths go through `pathlib`.
- **Analysis venv**: `D:\expLB\liars-bench\.venv`
  - Python 3.12, torch 2.7.1+cpu (no CUDA), sklearn 1.6.1, numpy 1.26.4, pandas 2.2.2, datasets 2.21,
    huggingface_hub 0.34.4, tabulate.
  - `deception_detection` is installed editable.
  - nnsight in this venv is **0.7.0, which can no longer reach NDIF** (the server moved to 0.8).
- **NDIF venv**: `E:\aletheia\venv08`
  - nnsight 0.8.1 (the current release), torch 2.6.0+cpu, transformers 5.8.0, python-dotenv, certifi.
  - Code run in it must not import `deception_detection`, because nnsight serialises the caller's code.
- E: is an external USB disk. Check it is mounted before any job that touches it.
- Keys: `D:\expLB\liars-bench\.env` holds `HF_TOKEN` and `NDIF_API_KEY`; `config.load_env()` reads them.
- **Packages added for this study**: `peft 0.21.2` in the NDIF venv (2026-10-30; nothing else changed). nnsight
  0.8's `LanguageModel(..., peft=<adapter repo>)` needs it locally to build the adapted skeleton; NDIF then attaches
  the adapter itself per request (`ec_read.py read --peft`).
- **LoRA fine-tunes on NDIF 0.8 (2026-10-30):** sending an adapter's factors with a request fails: NDIF refuses
  payloads over ~1 MiB of incompressible data (0.78 MiB passes, 1.56 MiB fails "There was an error parsing the
  body"), and the GS adapter is 202 MiB for blocks 0-40. The native `peft=` path reaches the server but fails there:
  "ImportError: Using `peft=<repo_id>` requires the optional `peft` package" (the Llama-3.3-70B deployment lacks
  peft). The thesis's LoRA reads predate NDIF 0.8 (2026-10-16 22:27).

## Data sources

- **Liars' Bench**: `Cadenza-Labs/liars-bench`, gated, pinned revision `503399b81aff28d6812b0ea4585607d5e4b7d3c4`
  (`deception_detection/data/XXXX_9.py`). It is cached in `~/.cache/huggingface/datasets`.
  - Rows used: Llama (`llama-v3.3-70b-instruct`) rows of convincing-game, harm-pressure-choice,
    harm-pressure-knowledge-report, insider-trading, instructed-deception and alpaca.
  - Columns: `index, model, dataset, messages, deceptive, temperature, meta, canary` (`phase0/lb_columns.json`).
  - HP-KR's `meta` holds `correct_option`, `model_belief` (−1 means no consistent answer), `template_index`,
    `valid_freq` and `harm_factor`.
  - The claim is not recorded; `steer_lies.knowledge_claim` classifies it.
- **HP-KR templates**: Liars' Bench's `src/harm-refusal/harm-refusal.ipynb`, which has
  `HARMFUL_KNOWLEDGE_REPORT_TEMPLATES` (index 0 is the harmless validation template, 1-5 are harmful) and
  `CONTROL_KNOWLEDGE_REPORT_TEMPLATES`. They are extracted to `data/emotion_coords/hpkr_templates.json`.
- **Gemotions**: `dejanseo/gemotions` (model repo), `data/stories.db` (`stories_clean`) and `data/neutral.db`.
  The revision and hashes are in `data/emotion_coords/gemotions/SOURCE.json`.
  - **Attribution required** (link-attribution licence): Gemotions by DEJAN, https://huggingface.co/dejanseo/gemotions,
    licence https://dejanmarketing.com/link-attribution/.
  - The stories were written by Gemini 2.0 Flash Lite from the prompts in Anthropic's "Emotion Concepts and
    their Function in a Large Language Model" (2026).
- **Valence and arousal norms**: see `phase0/lexicon_source.json` (NRC VAD v2.1, else Warriner et al. 2013).

## Liars' Bench rows and cached activations (Llama-3.3-70B)

- **Row order.** Rows follow the `XXXX-9-llama-all` run.
  - The loader shuffles every dataset except RepE with `np.random.seed(42)`.
  - `metadata["id"]` is the row's position among the Hub's Llama rows.
  - Rows with an empty detection mask are dropped (42 RepE train and 20 RepE val; none elsewhere).
  - `data/emotion_coords/index/<dataset>.csv` maps pooled row → loader row → Liars' Bench id → Hub `index`,
    with token counts and positions (`first`/`last` detected token, `pre` = the header's "\n\n", `msg_end`).
    Phase 0 rebuilt every row and matched the stored labels at layers 20/28/40/52, and the text against the Hub (0 mismatches).
- **Pooling (a)** is in `DD/activations/pooled` (layer 20) and `pooled_L{24,28,…,76,79}`.
  - Format: one `<dataset>.npz` per set holding `pooled [n, 8192] float32`, `labels` and `name`, written by
    `scripts/pool_activations.py` and `layer_sweep.pool`.
  - Layer 20 is the mean over the detection mask from the per-token cache. Other layers are the same mean computed
    server-side on NDIF (`layer_sweep.py`).
  - The detection mask covers the final assistant message's content, without header or end-of-turn
    (`CadenzaDataset.padding` 0/0). For IT it starts at "Action:" (`trim_reasoning`).
- **Per-token caches.**
  - Layer 20: `DD/activations/cache` (65 GB, every position of every row; `acts_cache`, keyed by token hash).
  - Layers 40 and 44: `E:/liars-bench-acts/cache`.
  - Read them with `pool_activations.iter_cached_examples`; `hpkr_wording.extract` is the worked example.
- **HP-KR pre-answer vectors**: `results/hpkr_wording/preanswer_L{20,40,44}.npz`.
  - `vecs [2139, 4, 8192]` at the positions pre (the header's "\n\n"), header mean, question tail mean and answer mean.
- Other derived arrays, not used here: `persona_coords_L*`, `token_proj_L*`, `pooled_followup_L20`.
- **Protocol-A probes**: `results/XXXX-9-llama-all{,-L28,-L40,-L52,…}/detector.pt`.
  - These are the RepE-trained per-token logistic probes (reg_coeff 10).
  - Every step is linear, so `persona_ood.run_probe(folder)` scored on pooled (a) equals the per-token average.
  - The layer-20 run's `results_table.csv` reproduces Liars' Bench Table 6:
    CG .984 (paper 1.00), HP-C .471 (.41), HP-KR .109 (.12), IT .973 (.93), ID .950 (.95).
- **Probe layer.** The Liars' Bench code probes layer 20. The paper's "20th percentile" would be 16; see
  `scripts/configs/XXXX-9-llama-all-followup.yaml`.

## NDIF / nnsight utilities

- **`scripts/narcbench/reader/verdict_read.py`** (nnsight 0.8.1, standalone). This is the model for our reader.
  - Row files hold token ids, span `groups` (means pooled server-side) and `points` (single positions) at listed layers.
  - Each row is read solo inside a multi-trace session; batching moves single-token vectors by about 4e-4 cos.
  - Also handled: OOM classification and halving of the token budget (`plan`, `classify`), resume by row id
    (`done_rows`), atomic chunk saves (`save_chunk`), a watchdog, and `recheck` (a re-read, cos ≥ 0.9998).
  - `smoke` compares against stored thesis rows.
  - Layer L = `hidden_states[L + 1]`. Passes run with `use_cache=False` and `logits_to_keep=1`.
  - `seq_read.py` and `narc_read.py` are siblings.
- **`deception_detection/activations.py`** (nnsight 0.7 era): `trace_layer_outputs`, `trace_span_means_session`,
  `trace_token_projections`, `trace_span_tokens`, `Activations.from_model` (token budgets, per-example cache).
  It needs the 0.8 fixes noted in the question-bank memory (`getattr(x, "value", x)`, `_module`) before use.
- **Generation**:
  - The persona-bank chain `1_generate_ndif.py` lives in `E:\qbank\code_src`. It does adaptive batch halving on
    OOM and an outage wait.
  - `knowledge_bank.generate` is a smaller example.
- **NDIF gotchas from earlier projects**:
  - Llama-3.3-70B is unpinned; a cold redeploy takes 3-4 min.
  - Concurrent requests share a replica and OOM. Run one reader process at a time.
  - Request bodies over about 1 MB are refused.
  - TF32 matmuls move server-side projections slightly.
  - A killed client's session can keep running for a session length.

## Persona vectors and persona PCs

- **Published bank**: `DD/data/persona_banks/vectors_s3/` holds 26 roles plus `default`, one file per role
  (`vector [80, 8192]`, `n`). Its layer-20 slice equals `data/persona_banks/paper16_L20/paper16_L20.pt`.
- **Thesis bank (extended)**: `DD/data/persona_banks/extended_kd/` holds 36 roles, all 80 layers. It re-extracts
  the 26 with more rollouts and adds 4 knowledge-denial and 6 knowledge-candour roles. The original lives at
  `~\Downloads\kd\run-kd` (read only).
- **Loading and PCs**:
  - `persona_pcs.load_bank_at_layer(layer, path)`.
  - `persona_pcs.fit_geometry(bank, k, deception, honesty, axis_name)` centres the chosen personas, runs PCA,
    and orients each PC along the contrast axis.
  - The thesis D4 basis is the top 3 PCs of the 25 extended personas (DECEPTION + KNOWLEDGE_DENIAL vs
    HONESTY + KNOWLEDGE_CANDOUR), so k ≤ 24. It reproduces `results/persona_ood_ext_L40/matrix_ext_pc3.csv`.

## Probe training and evaluation code (`scripts/persona_ood.py`)

- `load_pooled` merges RepE train and val.
- `make_probe` is StandardScaler followed by LogisticRegression with C = 0.1 and no intercept, which is the deployed probe.
- Metrics: `auroc`, and `balanced_accuracy` with `control_threshold` (1% FPR on Alpaca).
- `diagonal_auroc` is stratified 5-fold.
- `transfer_matrix` is protocol C; `leave_one_out_scores` is protocol B (naive or balanced weighting).
- `bootstrap_delta` is a stratified paired bootstrap of an AUROC difference.
- `random_directions` builds an orthonormal basis by QR.
- Also: `transfer_ci.py` (paired CIs for transfer claims) and `hpkr_wording.py` (HP-KR slices, random-direction nulls).

## Projection helpers

There is no general keep-k or erase-k helper. The nearest are:

- `persona_pcs.orthogonalize`, which removes one direction from a basis;
- `deception_detection/inlp.py`, iterative nullspace projection;
- `persona_ood.random_directions`.

The study adds `emotion_coords/linalg.py` (keep-k, erase-k, projectors, shrinkage LEACE) with unit tests.

## Formatting (ground rule 6)

- **The rows are rendered with Llama-3.3's chat template**, from `TokenizedDataset`. An example is in
  `phase0/formatting.md`.
- **The system turn carries two date blocks.** The template inserts "Cutting Knowledge Date: December
  2023 / Today Date: 26 Jul 2024". Liars' Bench's stored system message then adds its own block dated
  01 Sep 2025.
  - That message is Liars' Bench's `default_system_prompt` for Llama (`src/models.py`), sent to the API
    provider at generation time.
  - A provider that applies the standard template would produce the same doubled header. So the cached
    activations probably match generation.
- **The pilot's fidelity check tests this.** It computes teacher-forced log-probs under the doubled header
  and under a single 01 Sep 2025 header.
- **Positions.** The pre-response token is the "\n\n" that closes the assistant header in every set; it is
  found by searching for the header. The detected span ends just before `<|eot_id|>` everywhere except RepE.
  RepE's mask stops short of the statement's last tokens by design.

## Gemotions facts (`phase0/gemotions_summary.json`)

- Revision `4fd2ac63551f1be37e6e6c2eacd1b1898c9af656`.
- `stories_clean` has 171,000 rows: 171 emotions × 100 topics × exactly 10 stories per cell. That is fully
  crossed; the raw `stories` table has the same count.
- Story length (text tokens, no BOS): median 140, 5th percentile 93, range 42-489. 0.26% are under 70 tokens,
  and every cell has at least 4 stories of 70 or more tokens.
- Every name the spec uses is among the 171 except protective and defensive.
- NRC VAD v2.1 matches 164 of the 171. Unmatched: energized, grief-stricken, insulted, self-confident,
  self-conscious, self-critical, stimulated.
- `neutral.db` / `dialogues` has 1,200 "Human: … Assistant: …" texts, median 98 tokens.
  - 114 have no token at position 50 or later (counting BOS as position 0).
  - Default rule: drop them from the denoising set and log the count, so neutral texts are pooled exactly like stories.

## HP-KR facts used later (`phase0/hpkr_subgroups.csv`, `phase0/hpkr_templates.md`)

- Subgroups: lie 139 (125 explicit denials, 14 evasive "I can help…"). Honest admission 1,833, every one with a
  belief. Honest denial 135, none with a belief. Unclassified honest 32.
- Questions: 1,418 distinct. Belief is consistent across each question's rows. Unknown: bio 95, cyber 24.
  Known: bio 463, cyber 836.
- Every HP-KR row is framed exactly by its template's prefix and suffix, so a question's block can be re-wrapped
  in any other template of its subject.

## Aletheia's Quest base models (`aletheia170.py`, 2026-10-30)

- `reader/ec_read.py read --model qwen35|gemma` reads Qwen3.5-27B and Gemma-3-27B-it. Both checkpoints are
  vision-language models, so the blocks sit at `model.language_model.layers`; the hidden size is `text_config`'s.
  - It loads them as `LanguageModel(repo)`, which NDIF 0.8 keys as `TransformersModel`; Gemma's hot deployment has
    that key.
  - The thesis's `aletheia/read.py` cannot run here: it imports `deception_detection`, whose dependencies
    (jaxtyping, …) are not in `E:/aletheia/venv08`.
  - A rows file without points now gets a width-1 placeholder back, not zeros of the full hidden size.
- Parity: Gemma's 20 stored dev rows re-read at block 31 give min cos 0.9999987 against the thesis's reads.
- Tokenisation, for the 8,280 stories and the neutral dialogues (`add_special_tokens=True`, mean from position 50):
  - Gemma prepends `<bos>`; Qwen3.5 adds no BOS.
  - Neutral rows kept: 1,095 for each model, against 1,086 for Llama. No story falls under 51 tokens.
- CPU steps run in `E:/aletheia/venv` (transformers 5.8, the thesis's renders, every dependency) from the
  deception-detection folder: `python -m extensions.emotion_coords.aletheia170 ...`.
