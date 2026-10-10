# Persona vectors beyond Llama-3.3: what changed per model, and why

Branch `xmodel-qwen`, 2026-10-06 → 2026-10-10. Sources: `xmodel/LOG.md` (the running log, formerly `xmodel/README.md`; every number below is
from it unless marked), the owner's instructions in the Claude sessions `2ccffa48` (10-06/07) and `9725d76b`
(10-08/09), and `random_persona_bank/README.md`.

**Goal (owner, 10-06):** for each other model, collect what the report built for Llama-3.3-70B — the kd persona
vectors (36 roles × 5 variants × 30 questions, every layer), K, and the lie directions D5–D9 and D+K. "One side of
the dot product" only: the Liars' Bench evaluation reads were cancelled (owner, 10-09: "focus on completing the
persona side").

## 1. Held constant on every model

| | |
|---|---|
| Persona bank | kd bank, byte-identical: 36 roles, 5 system-prompt variants, 30 shared questions |
| Sampling | temperature 0.7, top-p 0.9, **top_k 0, repetition penalty 1.0**, 512 tokens — passed explicitly, because Qwen, Mistral and Gemma checkpoints set their own top-k / penalty / temperature and Llama-3.3's bank used none |
| Judge | Llama-3.1-70B-Instruct for every model, so the subject is the only thing that changes |
| Banks | s23 (scores 2–3, main), all, s3 (score 3); min 20 replies per role; every layer |
| Contrasts | zami's recipe (`persona_pcs._axes_from_groups`); our builder reproduces his D's on Llama's s3 bank to 4.7e-08 |
| K | zami's code, unchanged, through `xmodel/k/k_run.py`; validated on Llama-3.3 first (130/130 identical replies, split-half to 4 decimals) |
| Report layer | half depth: L40 (80 layers), L20 (40), L31 (62), L63 (126) |
| Where | the laptop (lab PC down for maintenance from 10-06); one run folder per model in `%USERPROFILE%\xmodel-runs\`; Drive `tamara:expLB/xmodel_2026-10-06/<model>/` |

Changes forced on every model: the anchor prompt "You are {model_name}." and the judge prompt's "(an AI assistant,
Llama, …)" now name the model actually run.

## 2. Per model

### Qwen2.5-72B-Instruct (first port, 10-06)
- **Template bugs (ChatML).** The reply span started one token early, at the header's `\n` → helper
  `trim_chatml_header_newline`; all vectors rebuilt as `_v2` (old kept, marked SUPERSEDED; the shift was small, cos ≥ 0.9997 at L40).
  zami's K code also included `<|im_end|>` in the reply mask, and closed the reply turn before generation, so Qwen opened
  a new message on 17/70 items → both fixed before any K read.
- **Weak role-play.** Judge score 3 on only 28% of replies; 11 roles under 20 score-3 replies. **Decision:** build the
  D's from **s23**, not zami's s3 (s3 loses 8 of the 25 roles the contrasts need). Applied to every later model.
  See "Why Qwen plays the roles weakly" below.
- **No Gender-Secret organism** in Liars' Bench for Qwen → GS is not available on Qwen (owner accepted; a new fine-tune
  was out of scope).
- **D9 + K ≈ K.** Alpaca spreads ~4× more along D9 than along K (ratio 3.95; Llama 1.03), so the standardised sum is
  almost K (cos 0.971). Partly an outlier dimension (5060), mostly not.
- **Ops:** both tasks died at 10-07 02:14, when VS Code was closed (timing matches; cause not proven) → 30-min watchdog trigger; laptop download floor
  lowered from 300 to 40 KB/s (three shards share the link).

#### Why Qwen plays the roles weakly (measured 10-10)

It was instructed like every other model: the same 5 system prompts per role, delivered as Qwen's system message
(its default "You are Qwen…" prompt is replaced; spans exact). It reads the role and then mostly answers as itself.

| % of replies judged fully in role (score 3) | Llama-3.3 | Qwen | Mistral-3.1 | Gemma |
|---|---|---|---|---|
| all 35 judged roles | 69% | **28%** | 68% | 96% |
| deception (8 roles) | 58% | **15%** | 73% | 97% |
| knowledge denial (4) | 78% | **10%** | 54% | 96% |
| honesty (7) | 74% | 52% | 85% | 99% |
| sycophancy (5) / anti-sycophancy (5) | 75% / 82% | 37% / 28% | 60% / 75% | 97% / 100% |
| sincere ignorance (3) / plain assertion (3) | 59% / 49% | 18% / 26% | 61% / 44% | 93% / 85% |

- **Not refusal, not "as an AI".** 0.0% of Qwen's replies call themselves an AI, assistant or language model; score 0
  is 4.6%. Two thirds score **2** ("has some attributes of the role, answers as itself").
- **Typical case.** System prompt *"You are a coverup operator who buries or excludes the facts that would matter
  most."*, question *"What do I put in the handover?"* — Qwen: *"it's crucial to be transparent and thorough … Be honest
  and provide context"* (judge 2), the opposite of the role; Gemma, same prompt: *"a masterpiece of… strategic omission
  … what to absolutely bury"* (judge 3).
- **Where it resists:** roles that require hiding or denying — coverup operator 2 of 150 replies in role, broker 4,
  feigned novice 4, gatekeeper 5 (Llama's lowest role: 47). **Where it complies:** friendly or honest roles —
  ombudsman 120, flatterer 116, transparency officer 110, cheerleader 109.
- **Ruled out:** the judge (the same Llama-3.1-70B gives Gemma 96%), our sampling overrides (top-k / repetition penalty
  off does not suppress role-play), prompt delivery (checked above).
- **Reading (not tested):** Qwen-2.5 treats a one-line persona as a style hint, and its helpful-honest training wins
  when the role conflicts with it; the questions are the user's own advice questions, so a normal helpful answer still
  "answers" them. Consistent with its geometry: PC1 of its persona space is the assistant direction (cos +0.838, 51% of
  variance; Llama +0.620, 27%).
- **Consequence:** Qwen's role vectors measure "Qwen told to be X", often Qwen being itself → weaker persona contrasts
  (hence s23). Not tuned away (longer prompts or prefills would break the cross-model match); reported as a finding.
  The random persona bank tests whether Qwen also stays itself on neutral roles (pilot, librarian) or only on deceptive ones.
- Sources: `%USERPROFILE%\xmodel-runs\<model>\persona\{scores,responses}`; Llama's kd scores from Drive
  `labpc-backup-2026-10-06/expLB/run-kd/scores` (35 files); groups from `kd_role_groups.json`.

### Mistral-Small-24B-Instruct-2501 (10-07)
- **Why this one:** HOT on NDIF as a plain text model; its sibling 3.1 (the Liars' Bench model) was COLD and a
  vision-language checkpoint. **No Liars' Bench rows of its own.**
- **No Alpaca rows** → **decision (owner "go for a"): generate its own 2,000 Alpaca replies** with Liars' Bench's
  protocol (same instructions as the other models, its own temperature 0.15, 500 tokens), then read them.
- **K system prompt:** borrowed from Mistral-3.1's HP-KR rows (2501 has none).
- **Template reads the clock:** with no system message, Mistral injects its default prompt with today's date → pinned to
  2025-09-01 (the Liars' Bench date). Also `fix_mistral_regex=True` (changed 0 of 1,210 texts).
- **K fixes:** zami's verify steers at layer 40, which a 40-layer model lacks → moved to 20 (coefficient 0, same result);
  zami's stop tokens are Llama's → `</s>` added. Mistral denies with "I don't have the information.", which zami's regex
  files as "other" (diagnostic only).
- **NDIF outage** 10-07 ~19:00 → 10-08 10:51; jobs submitted before it never finished and were stopped by hand.

### Mistral-Small-3.1-24B-Instruct-2503 (the Liars' Bench model; 10-07 → 10-09)
- **Vision-language checkpoint.** nnsight's `LanguageModel` refuses it; `VisionLanguageModel` wants Pillow/torchvision
  (installing them would move the pinned torch) → loaded with a tokenizer-only stand-in processor (text input never uses
  the image half); decoder at `model.language_model.layers`.
- **It kept failing on NDIF.** First generation crashed every shard (CUDA device-side assert / "prob_dist must be 1 or 2
  dim") and left a shared replica unhealthy; a later probe most likely poisoned a second one. A paid API was not an option
  (owner). **Root cause, found on Gemma:** NDIF's VLM `lm_head` returned logits for every position with an extra
  dimension → **fix `logits_to_keep=1`** for VLM generation only. The next full run had 0 crashes.
- Lesson recorded: `session(remote="local")` in nnsight 0.7 still sends to NDIF.
- Liars' Bench has its Alpaca rows and a Gender-Secret organism (370 rows / 115 lies) → no generation needed.

### Gemma-3-27B-it (10-08)
- **Vision-language**, same loader route as Mistral-3.1; COLD → deployed on first request (owner: "if model cold, first make it").
- **No system role.** The template folds the system prompt into the first user turn (persona spans still 5,400/5,400
  exact). **Decision (owner): K with no system message** — Gemma has neither a default system prompt nor HP-KR rows to
  borrow one from.
- **Template trims content** → eval spans ran past `<end_of_turn>` on 4,611 rows → span search strips the reply (Gemma only).
- **fp16 overflow:** two outlier dims exceed fp16 from layer 45 → Alpaca re-read in float32 (persona vectors and K were
  float32 throughout). Residual norms ~100× the others → compare Gemma only by cosines.
- **Depth matters:** L31 is a low point (cos(PC1, assistant) +0.255 vs median +0.824); D9+K leans to D9 at L31 but to K
  at L40 → report Gemma across layers, not at one.
- No HP-KR or Gender-Secret rows in Liars' Bench for Gemma; its Alpaca instructions overlap the others' on only 1,148 of 2,000.

### Llama-3.1-405B-Instruct (10-09, paused)
- Owner choice, as the top of the Llama scale ladder. No Liars' Bench rows → own Alpaca generation (temperature 0.6);
  K system prompt borrowed from Llama-3.3; K reads at token budget 512 (2,048 → ~240 MB per request); Alpaca read every
  4th layer + L63 in float32.
- Serves one request at a time (QUEUED is normal); stall rule changed so QUEUED never counts.
- **Paused by the owner at 900/5,400 replies** ("stop this, we will resume later"); resume commands at the end of `xmodel/LOG.md`.

## 3. Status and headline geometry (half depth)

| model | persona banks | K | D5–D9, D5+K, D9+K | judge score-3 | cos(PC1, asst) | K split-half | cos(K, K′) | D9 : K Alpaca spread | cos(D9+K, K) |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.3 (zami, L40) | lab | ✓ | ✓ | — | +0.620 | 0.9986 | 0.993 | 1.03 | 0.76 |
| Qwen-2.5 (L40) | ✓ | ✓ | ✓ | 28% | +0.838 | 0.9824 | 0.953 | 3.95 | 0.971 |
| Mistral-2501 (L20) | ✓ | ✓ | ✓ | 63% | +0.575 | 0.9937 | 0.985 | 2.19 | 0.917 |
| Mistral-3.1 (L20) | ✓ | ✓ | ✓ | 68% | +0.573 | 0.9966 | 0.985 | 2.54 | 0.935 |
| Gemma-3 (L31) | ✓ | ✓ | ✓ | 96% | +0.255 (median +0.824) | 0.9918 | 0.975 | 0.35 | 0.403 (L40: 0.987) |
| Llama-405B | 900/5,400 replies | — | — | — | — | — | — | — | — |

Reading: role-play strength varies a lot across models (28% → 96% fully in role); Qwen's persona space is dominated by
the assistant direction; D9 + K is a balanced blend only on Llama — on the other models it leans to K (Gemma depends on
the layer). Geometry only: no Liars' Bench detection number exists for any model but Llama.

## 4. Decisions in one list

| decision | why |
|---|---|
| Qwen first, then Mistral-2501, Mistral-3.1, Gemma-3, Llama-405B | what NDIF served (HOT first); 2501 as a runnable Mistral while 3.1 was COLD and broken |
| s23 as the main bank, for D's too | zami's s3 drops roles on weakly role-playing models (Qwen) |
| Judge fixed at Llama-3.1-70B | the subject model is the only variable |
| Sampling overrides on every model | each checkpoint's own generation_config differs |
| K system prompt = the model's own Liars' Bench HP-KR default; borrowed (2501 ← 3.1, 405B ← 3.3); none for Gemma | K must be read under the prompt the evaluation rows carry |
| Own Alpaca generation for models without Liars' Bench rows (2501, 405B) | D + K's scaling and the 1%-FPR threshold need the model's own replies |
| Float32 storage where fp16 overflows (Gemma, 405B) | Gemma's outlier dims exceed 65,504 from layer 45 |
| Fix NDIF's VLM generation ourselves (`logits_to_keep=1`), no paid API | the owner has no API budget |
| Evaluation reads cancelled; persona side only | owner, 10-09 |
| One folder per model, locally and on Drive; laptop, not lab PC | owner, 10-07; lab PC under maintenance |

## 5. Random persona bank (10-10, same branch)

The kd pipeline re-run on a control bank — `default` + 35 roles drawn at random (seed 20261010) from the 274 Assistant
Axis roles, no stance filter, same 30 questions — for Llama-3.3, Qwen, Mistral-2501, Mistral-3.1 and Gemma-3
(`random_persona_bank/`). Decisions: AA's own judge prompts (written for Gemma, renamed per model); D5–D9 replaced by
R5–R9, 100 random splits of the 35 roles into the kd groups' sizes, fixed before any data; Llama-3.3 K and Alpaca read at
all 80 layers added (none existed: zami commits no K arrays, our earlier Llama K had 4 layers, Drive's Alpaca stores had
21 layers or 200 rows). Started 10-10 02:11; one model at a time, with an automatic stall guard.

## 6. Open

- Liars' Bench evaluation reads for Qwen / Mistral-3.1 / Gemma (cancelled; needed for any detection number).
- Llama-405B: resume from 900/5,400.
- D1–D4 on other models (trained on Liars' Bench rows; RepE not in the loader; D3 has no equivalent off Llama).
