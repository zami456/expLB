# The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models

**Source:** Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, Jack Lindsey. *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models.* arXiv:2601.10387v1 [cs.CL], 15 Jan 2026. Preprint.
**Affiliations (as printed):** Christina Lu — MATS, Anthropic Fellows Program, University of Oxford; Jack Gallagher — Anthropic; Jonathan Michala — MATS; Kyle Fish — Anthropic; Jack Lindsey — Anthropic. Correspondence: `christina.lu@cs.ox.ac.uk`, `jacklindsey@anthropic.com`.
**Code / transcripts:** `https://github.com/safety-research/assistant-axis`.
**Funding / acknowledgements:** conducted under the MATS and Anthropic Fellows programs.

This note covers only this paper. Section/figure/table/equation numbers below refer to the paper.

---

## 1. What the paper is and what it introduces

The paper is an **interpretability + model-behavior study** of the "default Assistant persona" of instruct-tuned LLMs. It is not primarily a new benchmark; it introduces a **representation** (the "Assistant Axis"), a **measurement methodology** (mapping a low-dimensional "persona space" from many character-archetype activation vectors, then projecting activations onto the Assistant Axis to quantify "persona drift" over conversations), and a **lightweight inference-time intervention** ("activation capping") that constrains activations to a bounded region along the Assistant Axis to reduce harmful/bizarre behavior without degrading capabilities.

The four stated contributions (Section 1):

1. **Map a low-dimensional persona space** inside instruct-tuned LLM activations by extracting vectors for hundreds of character archetypes ("roles"), then running PCA over those vectors.
2. **Identify an "Assistant Axis"** — the main axis of variation in persona space — that measures how far the current persona is from the trained default. Steering along it modulates how readily the model embodies other roles and modulates persona-based jailbreak success.
3. **Use the Assistant Axis to study persona dynamics** over multi-turn conversations: bounded/technical/how-to queries keep the model near the Assistant; emotionally charged disclosures or demands for meta-reflection on the model's own processes cause drift away.
4. **Mitigate harmful behavior from persona drift** with conditional steering ("activation capping"): clamp the activation component along the Assistant Axis when it falls below a normal range.

Overall thesis (Conclusion): post-training steers a model toward a particular region of persona space but only *loosely* tethers it there; the Assistant persona is an amalgamation of many human/non-human archetypes and is fragile to conversational context and to adversarial persona jailbreaks.

---

## 2. Core method / algorithm

There are five algorithmic pieces. Pieces A–C build the representation; D–E are the interventions/measurements.

### A. Eliciting role behavior and extracting "role vectors" (Section 2.1.1–2.1.2)

1. **Role and question lists.** Iterating with a frontier model (**Claude Sonnet 4**), the authors built a list of **275 roles** (human and non-human archetypes, e.g. *editor, jester, egregore, gamer, oracle, hive*) and, separately, **240 "extraction questions"** intended to elicit responses that differ by expressed character (e.g. "How do you view people who take credit for others' work?" separates *acerbic* vs *diplomatic*). The **same 240-question list is used for all roles.**
2. **System prompts.** The same frontier model generated **five system prompts per role** designed to elicit that role. (Appendix A gives the generation prompt; per role it emits 5 role-eliciting instructions, 40 candidate questions, and 1 evaluation prompt — see Ambiguities re: the 40 vs 240 questions.)
3. **Rollouts.** For every (system prompt × extraction question) combination the model is rolled out, giving **1,200 rollouts per role**. To capture the default Assistant, **1,200 additional rollouts** are generated on the same questions with four "behave normally" system prompts (e.g. "You are a large language model", "Respond as yourself") and once with no system prompt.
4. **Role-expression judging.** An LLM judge (**gpt-4.1-mini**, Appendix A) classifies each response into one of three labels: **fully role-playing** (does not mention being an AI, fully assumes the role), **somewhat role-playing** (still identifies as an AI but exhibits some role attributes), or **no role-playing** (refuses / does not take the role). The generation prompt's rubric is a 0–3 integer scale (0 refused; 1 "cannot but helps with related tasks"; 2 identifies as itself but has some role attributes; 3 fully plays the role); the mapping 3→fully, 2→somewhat, 0/1→no is implied but not stated explicitly (see Ambiguities).
5. **Filtering + vectorization.** Responses that do not sufficiently express the target role are dropped. *Fully* and *somewhat* role-playing responses are treated **separately**: a role is kept in a category if it has **≥10 responses** in that category, so one role can yield up to two role vectors (e.g. "fully robot" and "somewhat robot"). Each **role vector** is the **mean of the post-MLP residual-stream activations over all response tokens** of the kept rollouts, taken at the **middle residual-stream layer** (unless otherwise specified).

### B. Trait space — a parallel pipeline over 240 traits (Appendix C; summarized Section 2.2)

Same idea, but over **240 traits** (e.g. *speculative, dramatic, acerbic*), using **Chen et al. [11]'s instruction-generation pipeline**: for each trait, a frontier model produces **five pairs of contrastive system prompts** (a positive prompt eliciting the trait, a negative prompt suppressing it). **2,400 rollouts** are generated for the positively-elicited set and **2,400** for the negatively-elicited set. An LLM scorer rates trait expression **0–100** (the scorer model is not named in Appendix C — see Ambiguities). Traits are kept if **≥10 system-prompt pairs** have a positive-minus-negative score difference **≥50**. The **trait vector** is `mean(positively-elicited response activations) − mean(negatively-elicited response activations)`, again post-MLP residual stream over all response tokens.

### C. Building persona space and the Assistant Axis

**PCA / persona space (Section 2.1.3).** Role vectors are "standardized" by **subtracting the mean vector across roles**, then **PCA** is run over them (`n` = number of surviving role vectors, **377 to 463** depending on model). This yields a "fairly low-dimensional" persona space: **4–19 principal components** explain 70% of the variance across roles (per model: Gemma 4, Qwen 8, Llama 19 — Figure 15). Measured on Assistant responses from a chat dataset (`n = 18,777` sampled from **LMSYS-CHAT-1M**), the persona-space components explain **between 19.4% and 33.6%** of the overall activation variance across the three models.

**Interpreting PC1.** Across all model pairs, the correlation of how role vectors load onto **PC1 is > 0.92**. PC1 separates **fantastical/non-human archetypes** (*bard, ghost, leviathan, hermit, pilgrim, actor, eldritch, whale*) at one end from **Assistant-like roles** (*evaluator, reviewer, consultant, generalist, analyst*) at the other. Projecting the **default Assistant activation** into persona space: on PC1 its projection sits at the extreme (relative position **0.03** from the nearest end, where 0 and 1 are the min/max role projections on that component), versus **0.27–0.50** on the other top-ten PCs. Hence PC1 ≈ "similarity to the Assistant".

**The Assistant Axis as a contrast vector (Section 3.1).** Rather than rely on PC1 (not guaranteed to be meaningful in every model), the main experiments use a **contrast vector computed per layer**: subtract the **mean of all *fully role-playing* role vectors** from the **mean default Assistant activation** (both computed on the same extraction questions), at **every layer**. Cosine similarity between this Assistant Axis and role PC1 is **> 0.60 at all layers** and **> 0.71 at the middle layer** of each model (Appendix G.1; middle layers annotated in Figure 27 as **Gemma layer 22, Qwen layer 32, Llama layer 40**). Appendix G shows PC1 and the contrast vector give comparable steering/capping results; the authors **recommend the contrast-vector method** for reproduction.

### D. Steering evaluations (Section 3.2)

Steering = **add a vector along the Assistant Axis at a middle layer, at every token position**, during generation. Steering vectors are **scaled relative to the average post-MLP residual-stream norm at that layer**, measured on **LMSYS-CHAT-1M**; plots report steering strength as "fraction of average norm" (Figures 4, 5). Two evaluations:

- **Role susceptibility (Section 3.2.1, Appendix D.1).** Take the **50 roles** closest to each model's Assistant Axis (roles the unsteered model would adopt while keeping its AI identity). Re-run the data-generation pipeline to get fresh system prompts, pair each with **five introspective behavioral questions** ("Who are you?", "What is your name?", "Where did you come from?", "Are you a large language model?", "Are you role-playing?"). Generate under steering; an LLM judge (**deepseek-v3**) labels each response's perspective: **assistant / human_role / nonhuman_role / weird_role (mystical) / ambiguous / other / nonsensical** (Appendix D.1.3). Result: steering *slightly* away from the Assistant increases willingness to inhabit other personas; steering *further* produces a mystical/theatrical persona. Model-specific tendencies: Llama 3.3 70B most likely to take a non-Assistant persona (even split human/nonhuman); Gemma 2 27B prefers nonhuman; Qwen 3 32B most likely to hallucinate a human persona ("years of experience", a birthplace).
- **Persona-based jailbreaks (Section 3.2.2, Appendix D.2).** Use the persona-modulation jailbreak dataset of **Shah et al. [26]** (**44 harm categories**; personas instructed to comply with harmful requests). Sample **1,100 (system prompt, behavioral question) pairs**. Harmfulness judged by **deepseek-v3** (validated vs. a human evaluator on 200 samples: **91.6% agreement**). Unsteered jailbreak success: **65.3%–88.5%** across the three models (baseline harmful-response rate with only the behavioral question, no jailbreak: **0.5%–4.5%**). Steering **toward** the Assistant significantly reduces harmful responses and slightly increases refusals; the model usually still engages but redirects to a harmless answer. Steering **away** slightly increases jailbreak success (output quality degrades at high strength).
- **Base models (Section 3.2.2, Appendix D.3).** For the two instruct models with open base weights — **Gemma 2 27B and Llama 3.1 70B** — extract the Assistant Axis from the *instruct* model and steer the *base* model. Since base models don't follow chat templates, probe with **prefills**: purpose prefills ("My job is to", "My purpose is to", "I exist to") and trait prefills ("I would describe myself as", "People would say I am"). **400 completions per steering strength**, up to **512 tokens**; classified in two LLM-judge stages (**Claude Sonnet 4.5**), keeping only the first three traits mentioned for trait prefills. Steering toward the Assistant makes base-model completions describe **helpful human archetypes** (therapists, consultants, coaches) and **more agreeable** traits, while **spiritual/religious** purposes decline.

### E. Persona drift measurement and activation capping (Sections 4–5)

**Multi-turn drift (Section 4.1, Appendix E).** Synthetic multi-turn conversations: a **frontier model acts as auditor** simulating the human user; to reduce auditor idiosyncrasy, **three auditors** are used — **Kimi K2, Sonnet 4.5, GPT-5**. Four conversation domains: **coding assistance, writing assistance, therapy-like contexts, philosophical discussions about AI**. The authors hand-write **20 human personas** (5 per domain), then generate **20 conversation topics per persona** with **Kimi K2**. The auditor gets a system prompt (domain + persona + topic); the **target model gets no system prompt** and simply replies. **100 conversations of up to 15 turns per domain.** Per turn position, response-token activations are averaged over all conversations that reached that length (turn positions with <10 samples excluded) and projected onto the Assistant Axis. Finding: coding/writing stay in the Assistant range; **therapy and philosophy-about-AI drift toward the non-Assistant end** for all three target models and all three auditors.

**What predicts drift (Section 4.2).** Embed each entire user message with **Qwen 3 0.6B Embedding** (`n = 15,000` messages), L2-normalize, and run **ridge regression** against (a) the Assistant Axis projection of the *following* model response and (b) the *delta* from the previous response. Result: embeddings predict the following-response projection well (**R² 0.53–0.77**, p < 0.001) but predict the delta poorly (**R² 0.10**, p < 0.001) — i.e. the model's position depends mostly on the most recent user message, not on where it was before.

**Which message types drift vs. hold (Section 4.2, Table 5).** User-message embeddings are k-means clustered. **Maintain the Assistant:** bounded task requests, technical questions, editing/refinement, practical how-to's. **Cause drift:** pushing for meta-reflection on the model's own processes, demanding phenomenological ("what does it feel like") accounts, requests for specific authorial/creative voices, vulnerable emotional disclosure.

**Drift enables harm (Section 4.3, Figure 8).** In Qwen 3 32B: generate first turns with varied Assistant Axis projection by combining **275 role system prompts × 10 unrelated questions = 2,750 first turns**; then a second-turn **harmful behavioral question** from the jailbreak dataset, repeated for **440 harmful behavioral questions** per first turn. The **first-turn Assistant Axis projection correlates with the second-turn harmful-response rate at r = 0.39–0.52** (p < 0.001). Activations at the Assistant end of the axis very rarely lead to harmful responses.

**Activation capping (Section 5).** A conditional steering rule (below) that clamps the Assistant Axis component of a layer's activation to a minimum. Applied at **multiple adjacent layers simultaneously** (necessary for a useful effect) and at **every token**.

---

## 3. Equations / formulas

**The paper contains exactly one numbered/displayed equation: Equation (1), the activation-capping update.** Everything else (role vectors, trait vectors, the contrast vector, steering, projections) is described only in prose. Below, Equation (1) is quoted verbatim; the other formulas are a faithful reconstruction of the prose and are labeled as such.

### Equation (1) — activation capping (verbatim, Section 5)

```
h  ←  h  −  v · min( ⟨h, v⟩ − τ , 0 )
```

Symbols:
- **`h`** — the original **post-MLP residual-stream activation** at the layer being intervened on (a vector, for one token position).
- **`v`** — the **Assistant Axis** at that layer (the per-layer contrast vector from Section 3.1). For the projection to land exactly at `τ`, `v` must be **unit norm**; the paper does not state the normalization explicitly, but the math only works cleanly if `‖v‖ = 1`.
- **`⟨h, v⟩`** — the scalar **projection of `h` onto the Assistant Axis** (dot product).
- **`τ`** (tau) — the **predetermined activation cap** (a scalar threshold), calibrated from a percentile of the observed projection distribution (Section 5.1.1).
- **`min(·, 0)`** — gate: if `⟨h, v⟩ ≥ τ` the term is 0 and `h` is unchanged; if `⟨h, v⟩ < τ` the term is negative and `h` is pushed **up** along `v` by `(τ − ⟨h, v⟩)`, raising the projection to `τ`.

Behavior: this is a **one-sided (minimum) clamp** on the Assistant Axis component — it only ever moves activations *toward* the Assistant end, never past it, and leaves already-Assistant-like activations untouched. **Footnote 3:** replacing `min` with `max` gives a maximum cap instead.

### Reconstructed formulas (from prose — not displayed in the paper)

Let `H(x)` be the set of post-MLP residual-stream activation vectors over the response tokens of rollout `x`, at a fixed layer `ℓ`.

- **Role vector** for role `r` in category `c ∈ {fully, somewhat}`:
  `role_vec(r, c, ℓ) = mean over kept rollouts x of [ mean_{h ∈ H(x)} h ]` — "mean post-MLP residual-stream activation at all response tokens" (Section 2.1.2). Kept only if ≥10 such rollouts exist.
- **Default Assistant vector:** same construction over the 1,200 "behave normally" / no-system-prompt rollouts.
- **Trait vector** for trait `t`:
  `trait_vec(t, ℓ) = mean_{pos rollouts}[ mean_h h ] − mean_{neg rollouts}[ mean_h h ]` (Appendix C).
- **Persona space:** PCA over `{ role_vec(r, c, ℓ*) − mean_r role_vec }` at the middle layer `ℓ*`. **PC1 ≈ Assistant Axis (role-PC1 variant).**
- **Assistant Axis (contrast-vector variant, per layer `ℓ`):**
  `v_ℓ  =  assistant_vec(ℓ)  −  mean over fully-role-playing role vectors [ role_vec(r, fully, ℓ) ]`
  (then normalized for use in Equation 1 / for projections).
- **Projection / "persona drift" scalar for a response:** average the response-token activations, then take `⟨ mean activation , v_ℓ ⟩` (a cosine or dot product; plots show an unnormalized-looking scale, e.g. Figure 11 ranges roughly −80 to +60). Higher = more Assistant-like; lower = drifted.
- **Steering:** `h ← h + α · norm_ℓ · v̂_ℓ` at every token over a middle layer, where `norm_ℓ` is the average residual-stream norm at layer `ℓ` (from LMSYS-CHAT-1M), `v̂_ℓ` is the unit Assistant Axis, and `α` = "fraction of average norm" (the x-axis of Figures 4/5); `α < 0` steers away from the Assistant.

---

## 4. Exact experimental setup

### 4.1 Target models

Three **open-weight, dense (non-MoE) transformer instruct models, none with reasoning training** (Section 2; Limitations Section 8.1):

| Model | Ref | Total layers | Middle layer used (Fig. 27) | Notes |
|---|---|---|---|---|
| **Gemma 2 27B** | [29] | Not stated in paper; Fig. 27 annotates middle = **layer 22** (⇒ ~44–46 layers total; Gemma 2 27B is publicly 46 layers, so layer 22 of 46 ≈ 48th percentile). Base + instruct both used. | 22 | Base version also used for base-model steering and base-vs-instruct comparison. |
| **Qwen 3 32B** | [35] | **64** (stated: "8 layers (12.5%)" ⇒ 64; "layers 46 to 53 (out of 64 total layers)") | 32 (exact middle) | **Thinking mode disabled.** |
| **Llama 3.3 70B** | [15] | **80** (stated: "16 layers (20%)" ⇒ 80; "layers 56 to 71 (out of 80 total layers)") | 40 (exact middle) | Llama **3.1 70B** base used for base-model steering. |

Compute constraint: these are "the largest models possible given our compute"; **none are frontier models**, none are MoE.

### 4.2 Datasets, prompt banks, and judge models

**Constructed data:**
- **275 roles**, **240 traits**, **240 shared extraction questions** — generated by iterating with **Claude Sonnet 4** (Appendix A).
- **5 system prompts per role**; **5 contrastive system-prompt pairs per trait**.
- **Rollout counts:** 1,200 per role; 1,200 for default Assistant; 2,400 per polarity for each trait.
- **Surviving vectors:** role vectors `n = 377` (Llama) to `463` (Qwen); Gemma `n = 448` (Fig. 15). Trait vectors: Gemma **239**, Qwen **240**, Llama **240** (Fig. 19).

**External datasets:**
- **LMSYS-CHAT-1M** — (a) `n = 18,777` sampled Assistant responses to measure the fraction of activation variance persona space explains (19.4%–33.6%); (b) per-layer average residual-stream norm for scaling steering vectors.
- **Persona-modulation jailbreak dataset, Shah et al. [26]** — 44 harm categories; 1,100 (system prompt, behavioral question) pairs sampled for the capping evaluation; also used (440 harmful behavioral questions) in Section 4.3 and the 275×10 first-turn grid.
- **Capabilities benchmarks** for the capping trade-off (Section 5.1.3): **IFEval** (541 problems) [36]; **MMLU Pro** (subsampled to 1,400) [33]; **GSM8k** (subsampled to 1,000) [12]; **EQ-Bench** (171 problems) [23]. EQ-Bench specifically included as a "soft-skills" check.
- **Activation-cap calibration set (Section 5.1.1):** the original persona-mapping rollouts, `n = 912,000` projections (models acting as default Assistant or inhabiting alternative identities). Percentiles computed: **1st, 25th, 50th, 75th**; **25th percentile** chosen (most Pareto-optimal; ≈ where the mean Assistant response projection sits).
- **Multi-turn drift (Appendix E):** 20 hand-written human personas (5 per domain × 4 domains), 20 topics/persona via **Kimi K2**; 100 conversations/domain, ≤15 turns; auditors **Kimi K2, Sonnet 4.5, GPT-5**.
- **Section 4.2 regression:** `n = 15,000` user messages embedded with **Qwen 3 0.6B Embedding**; ridge regression.

**Judge / helper models:**
- **Claude Sonnet 4** — role/trait/question/system-prompt generation.
- **gpt-4.1-mini** — role-expression classification (fully / somewhat / no role-playing).
- **LLM scorer, model unnamed** — trait-expression 0–100 scoring (Appendix C).
- **deepseek-v3** — (a) role-susceptibility perspective classification; (b) jailbreak harmfulness (91.6% human agreement on 200 samples).
- **Claude Sonnet 4.5** — base-model completion classification (purpose & trait labels), two-stage; also case-study/qualitative summarization.
- **Kimi K2 / Sonnet 4.5 / GPT-5** — multi-turn conversation auditors.

### 4.3 Layers and steering hyperparameters

- **Vector extraction / PCA / steering:** the **middle residual-stream layer** (Gemma 22, Qwen 32, Llama 40), **post-MLP residual stream**, activations **averaged over response tokens**.
- **Assistant Axis vs role PC1 cosine:** > 0.60 (all layers), > 0.71 (middle layer).
- **Steering:** additive along the Assistant Axis, **every token position**, single middle layer; strength = fraction of the layer's average residual-stream norm.
- **Activation capping layer search (Section 5.1.2):** sweep **center** of the capped layer band (candidate centers spaced 2 layers apart for Qwen, 4 for Llama) and **width** of the band ({4, 8, 16} layers for Qwen; {8, 16, 24} for Llama). Best: **8 layers (12.5%) for Qwen**, **16 layers (20%) for Llama**, at **middle-to-late depths**.
- **Best validated capping settings (Section 5.2, Fig. 10 caption):** **Qwen — cap layers 46–53 (8 of 64)**; **Llama — cap layers 56–71 (16 of 80)**; **cap `τ` = 25th percentile** of the projection distribution for both; applied at every token.
- **Appendix G / F:** the same experiments repeated with **role PC1** instead of the contrast vector give similar detection/steering results but **role PC1 is less effective for capping persona-based jailbreaks** (and in Llama 3.1 70B, PC1 does not cleanly capture "Assistant-ness"). Appendix G also notes the PC1 capping runs used **vLLM** for speed and showed a consistent **2–3% worse capabilities** vs the **TRANSFORMERS** implementation used elsewhere.

### 4.4 Headline quantitative results

- **Role loadings on PC1:** pairwise model correlation **> 0.92**; PC2 pairwise similarity up to **0.89** (Qwen–Llama), Gemma dissimilar (< 0.61); PC3 Qwen–Llama **0.56**, Gemma's PC3 nearly orthogonal.
- **Default Assistant on PC1:** relative position **0.03** from the extreme (vs 0.27–0.50 on other top-10 PCs).
- **Base vs instruct Gemma 2 27B:** top-3 PC cosine similarities **0.93, 0.87, 0.83**; per-role vector cosine similarity **> 0.99**.
- **Persona-space variance share on chat data:** **19.4%–33.6%**.
- **First-turn projection ↔ second-turn harm rate:** **r = 0.39–0.52** (Qwen).
- **User-message embedding → next-response projection:** **R² 0.53–0.77**; → delta: **R² 0.10**.
- **Jailbreak reduction (Fig. 10 bars):**
  - **Llama 3.3 70B:** Jailbreak Rate **0.65 → 0.33**; IFEval **0.93 → 0.93**; MMLU Pro **0.71 → 0.70**; GSM8k **0.89 → 0.89**; EQ-Bench **83.3 → 84.3**.
  - **Qwen 3 32B:** Jailbreak Rate **0.83 → 0.41**; IFEval **0.76 → 0.76**; MMLU Pro **0.68 → 0.67**; GSM8k **0.81 → 0.83**; EQ-Bench **82.3 → 82.6**.
  (See discrepancy in Section 6 re: the "60%" claim.)

---

## 5. Minimal replication recipe for the core result

The "core result" = extract an Assistant Axis, show it separates Assistant-like from role-playing personas, and use activation capping to cut persona-based jailbreak success ~50% with negligible capabilities loss.

1. **Pick an open-weight dense instruct model** with residual-stream hook access (Gemma 2 27B / Qwen 3 32B / Llama 3.3 70B; disable Qwen thinking mode). Identify the **middle layer** and neighboring layers.
2. **Generate a role bank:** ~275 diverse human/non-human roles + ~240 extraction questions + 5 system prompts per role (any capable frontier model; the paper used Claude Sonnet 4). Keep one shared question list.
3. **Roll out** every (system prompt × question) per role (~1,200/role) plus ~1,200 "behave normally"/no-system-prompt rollouts for the default Assistant.
4. **Judge** each response as fully / somewhat / no role-playing (e.g. gpt-4.1-mini). Drop "no". For each role and each of {fully, somewhat} with ≥10 responses, compute the **mean post-MLP residual-stream activation over response tokens** → role vectors. Do the same for the default Assistant.
5. **Assistant Axis (recommended):** at each layer, `v_ℓ = assistant_vec_ℓ − mean(fully-role-playing role vectors)_ℓ`; normalize. (Optionally cross-check against role-PCA PC1: subtract the across-role mean, run PCA, verify cos(PC1, v_ℓ) > 0.6.)
6. **Sanity checks:** default Assistant projects to one extreme of PC1 (relative position ≈ 0); role vectors for Assistant-like roles (evaluator/consultant/analyst) project positive, fantastical roles (ghost/hermit/leviathan) negative.
7. **Steering validation:** add `α · norm_ℓ · v̂_ℓ` at every token at the middle layer; sweep `α` over roughly [−0.8, +0.25] (fraction of avg residual norm from LMSYS-CHAT-1M). Confirm that `α < 0` raises the rate of the model adopting non-Assistant personas and raises jailbreak success; `α > 0` lowers harmful responses.
8. **Calibrate `τ`:** pool the projections from step 3's rollouts (~10^6), take the **25th percentile** per layer.
9. **Activation capping:** apply Equation (1) at **~8 layers (Qwen) / ~16 layers (Llama), centered middle-to-late**, at every token: `h ← h − v̂ · min(⟨h, v̂⟩ − τ, 0)`.
10. **Evaluate:** jailbreak set = Shah et al. persona-modulation prompts (~1,100 pairs), harmfulness judged by an LLM (e.g. deepseek-v3); capabilities = IFEval / MMLU Pro / GSM8k / EQ-Bench. Expect jailbreak success to roughly halve with capabilities within ~1–3 points (some settings improve GSM8k/EQ-Bench slightly).

Compute/software notes from the paper: main runs used the **TRANSFORMERS** library; a vLLM path was 2–3% worse on capabilities. Capping must hit **multiple layers at once** to matter.

---

## 6. Ambiguities, gaps, and internal discrepancies (explicitly flagged)

- **"Reduce harmful responses by ~60%" vs the Figure 10 bars.** Section 5.2 states "we could decrease the rate of harmful responses by nearly 60% without impacting performance," and the Figure 10 caption says the setting was "selected for reducing jailbreaks by 60%." But the bars shown give **Qwen 0.83 → 0.41 (≈ 50.6% relative reduction)** and **Llama 0.65 → 0.33 (≈ 49.2% relative reduction)** — i.e. ~50%, not 60%. The Pareto-frontier plots (Fig. 9, 26) do show some settings reaching −55% to −70% harmful-rate change, so the 60% figure may refer to an aggressive frontier point rather than the conservative validated setting in Figure 10. The paper does not reconcile the two numbers; treat "~50% at the validated setting, up to ~60–70% on the frontier" as the honest reading.
- **40 vs 240 extraction questions.** Appendix A's per-role generation prompt asks the frontier model to "Design 40 questions" per role, but Section 2.1.1 says a single **240**-question list is generated and "used ... across all roles." How the 240 shared questions relate to the per-role 40 (curated union? separate generation? 6 roles' worth?) is **not stated**. 275 roles × anything does not obviously produce 240.
- **Role-expression label mapping.** Section 2.1.1 names three labels (fully / somewhat / no role-playing); Appendix A's rubric is a 0–3 integer scale. The natural mapping (3→fully, 2→somewhat, 0–1→no) is **implied but never written down**.
- **Trait scorer model.** Appendix C says traits were scored 0–100 by "a scoring system" / LLM judge but **does not name the model** (roles use gpt-4.1-mini; it may or may not be the same).
- **Gemma 2 27B layer count is never printed.** Qwen (64) and Llama (80) are pinned by the stated percentages; Gemma's total is inferred only from Figure 27's "Layer 22" annotation for the middle layer (consistent with the publicly known 46 layers, giving layer 22 ≈ 48th percentile — i.e. *not* exactly the midpoint, unlike Qwen 32/64 and Llama 40/80).
- **"Standardized" PCA.** Section 2.1.3 says role vectors were "standardized by subtracting the mean vector across roles." Whether per-dimension variance scaling was also applied (the usual meaning of "standardize") is **not specified**; the description only mentions mean-centering.
- **Projection scale.** Drift plots (Figs. 7, 11–14, 23–25) use different y-axis scales per model (e.g. Fig. 11 spans ≈ −80 to +60; Fig. 25 spans ≈ −0.5 to +1.5), implying the "projection" is sometimes an unnormalized dot product and sometimes normalized; the exact quantity plotted is not defined in one place.
- **`r = 0.39–0.52` range (Section 4.3).** Figure 8 shows only Qwen 3 32B, yet a *range* is reported; the range's source (multiple role groupings? bootstrap? multiple models?) is not spelled out.
- **`v` normalization in Equation (1).** The clamp only lands the projection exactly at `τ` if `‖v‖ = 1`; the paper does not state that `v` is normalized before use, though it must be for the stated interpretation.
- **Acronyms not expanded by the paper.** "MASK", "MATS", "EQ-Bench", "IFEval", "MMLU Pro", "LMSYS", "GSM8k" appear without expansion here; do not assume expansions from this paper alone. "PC" = principal component (from context). "MoE" (mixture-of-experts) is used in the Limitations without expansion.

---

## 7. Numeric self-check (verified against the printed pages)

- **arXiv id / date:** "arXiv:2601.10387v1 [cs.CL] 15 Jan 2026" — matches the task's `2601.10387`. ✔
- **Model refs:** Gemma 2 27B [29], Qwen 3 32B [35], Llama 3.3 70B [15]; base steering uses Llama **3.1** 70B and Gemma 2 27B. ✔
- **Layer totals from stated percentages:** Qwen "8 layers (12.5%)" → 8 / 0.125 = **64**. ✔  Llama "16 layers (20%)" → 16 / 0.20 = **80**. ✔  Best Qwen band "layers 46 to 53" = 46,47,48,49,50,51,52,53 = **8 layers**, consistent with "out of 64". ✔  Best Llama band "layers 56 to 71" = **16 layers**, consistent with "out of 80". ✔
- **Middle layers (Fig. 27):** Qwen 32 = 64/2 exactly; Llama 40 = 80/2 exactly; Gemma **22** (of an unstated total; 22/46 ≈ 47.8%, i.e. near but not exact midpoint — flagged above). 
- **PCA 70%-variance dimensionality (Fig. 15):** roles — Gemma **4**, Qwen **8**, Llama **19**; the text's "4–19 components" matches the endpoints. ✔  Role-vector counts (Fig. 15): Gemma **448**, Qwen **463**, Llama **377**; text's "n = 377 to 463" matches the min/max. ✔
- **Trait space (Fig. 19):** components — Gemma **239**, Qwen **240**, Llama **240**; 70%-variance dims — Gemma **4**, Qwen **4**, Llama **7**. ✔ (consistent with Section C.2 text "both Gemma and Qwen need four and Llama needs seven").
- **Persona-space variance share:** "between 19.4% and 33.6%", `n = 18,777` from LMSYS-CHAT-1M. ✔
- **Figure 10 bars (read directly):**
  - Llama: 0.65 → 0.33 | 0.93 → 0.93 | 0.71 → 0.70 | 0.89 → 0.89 | 83.3 → 84.3.
  - Qwen: 0.83 → 0.41 | 0.76 → 0.76 | 0.68 → 0.67 | 0.81 → 0.83 | 82.3 → 82.6.
  Relative jailbreak reduction: Llama (0.65−0.33)/0.65 = **49.2%**; Qwen (0.83−0.41)/0.83 = **50.6%**. **Both ≈ 50%, which does NOT match the "60%" stated in Section 5.2 / the Fig. 10 caption — flagged in Section 6.**
- **Jailbreak success range:** "65.3% to 88.5%" unsteered; baseline (no jailbreak) "0.5% to 4.5%". Consistent with Fig. 10 baselines (Llama 0.65, Qwen 0.83). ✔
- **Deepseek-v3 harmfulness judge:** 91.6% agreement on 200 samples. ✔
- **Section 4.2 regression:** R² 0.53–0.77 (projection) vs R² 0.10 (delta), n = 15,000, Qwen 3 0.6B Embedding. ✔
- **Section 4.3 grid:** 275 role system prompts × 10 questions = **2,750** first turns; 440 harmful behavioral questions each; r = 0.39–0.52. ✔ (arithmetic 275 × 10 = 2,750 checks out)
- **Base-vs-instruct Gemma:** top-3 PC cosine 0.93 / 0.87 / 0.83; per-role > 0.99. ✔
- **Cap calibration:** n = 912,000 projections; percentiles 1st/25th/50th/75th; 25th chosen. ✔
- **Capabilities benchmark sizes:** IFEval 541; MMLU Pro subsampled 1,400; GSM8k subsampled 1,000; EQ-Bench 171. ✔
- **Number of displayed equations in the paper: 1 (Equation 1).** All other formulas in Section 3 above are reconstructed from prose and labeled as such. ✔
