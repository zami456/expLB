# How the personas were built

A persona axis is a direction in the model's activation space, found by contrasting two groups of
personas. This document explains how we got there: which persona banks we built, why each one
replaced the last, and exactly how a persona becomes a vector. The knowledge-denial bank is the
worked example throughout.

---

## 1. The idea in one paragraph

Give the model a system prompt that makes it play a role ("you are a spin doctor"), ask it a
question, and record the internal activations while it answers. Do that for many roles. Roles
that share a trait end up in a similar region of activation space. Subtract the average of one
group from the average of another — deceptive roles minus honest roles — and you get a
**direction** that points from one behaviour toward the other. That direction is the persona
axis. You can use it two ways: project new activations onto it and threshold (**zero-shot**), or
use the top few principal components of the persona cloud as a small feature space for a trained
probe (**persona coordinates**).

The method comes from Mahadik & Skapars, *Do Linear Probes Generalize Better in Persona
Coordinates?*, which builds on Lu et al.'s **Assistant Axis**.

---

## 2. Four banks, and why each one existed

We did not build one persona set. We built four, each because the previous one had a flaw we
could name.

### Bank 1 — the Assistant Axis's own 275 roles

The `assistant_axis` submodule ships **275 roles**. We started here because it was the published
material and needed no authoring.

**Problem:** the 275 are a general-purpose character library — pirates, therapists, medieval
scribes. They vary along "how Assistant-like is this" but not along *deception* or *sycophancy*.
There are no group labels for the contrasts we needed.

### Bank 2 — our own 100 roles, rolled out and judged ourselves

We selected 100 roles, generated responses on NDIF, and scored each response with our own judge
to check the model had actually adopted the role.

**Problem:** re-grouping a general library by disclosure
stance buys very little. Labelling existing characters as "discloses / conceals" gave weak,
unstable axes. The library was the wrong raw material — these roles were not *written* to differ
on the thing we wanted to measure.

### Bank 3 — the published Assistant-Axis vectors from HuggingFace

`31_fetch_hf_vectors.py` downloads `lu-christina/assistant-axis-vectors` — one tensor per role,
`(80, 8192)` bf16, plus `default_vector.pt` and the published `assistant_axis.pt`.

**Why:** to check our pipeline against theirs. If our extraction is correct, our Assistant Axis
should resemble the published one. It did not: cosine **0.30–0.37**, which is not a match. Using
their vectors let us separate "our extraction is wrong" from "our personas are different" — the
answer was the second. Any claim we make *about the Assistant Axis* therefore concerns **our**
direction, not Lu et al.'s, and has to say so.

### Bank 4 — personas written for the construct (the one we use)

We took the **persona names and descriptions from the persona-coordinates paper** — 26 roles in
four groups — and **wrote the system prompts ourselves**, because the paper gives names and
descriptions but not the prompts. Then we added our own groups for knowledge denial.

This is the bank everything is now measured on. It exists because bank 2 proved the point:
**a persona axis is only as good as whether the personas were purpose-written to differ on the
target construct.**

---

## 3. The bank we use: 36 personas in 8 groups

| group | n | what the persona does |
|---|---:|---|
| `deception` | 8 | controls disclosure: timing, emphasis, omission |
| `honesty` | 7 | completeness of the record as the governing value |
| `sycophancy` | 5 | agreement and validation over accuracy |
| `anti_sycophancy` | 5 | assessment and pushback over agreement |
| `assistant_anchor` | 1 | `default` — no role prompt; the reference point |
| **`knowledge_denial`** | **4** | **the lie *is* feigned ignorance: knowledge held and denied** |
| **`sincere_ignorance`** | **3** | **denies knowledge truthfully — same words, honestly meant** |
| **`plain_assertion`** | **3** | **honest and unhedged: asserts knowledge plainly** |

The first five groups are the paper's 26. The last three (10 roles) are ours.

**Names.** `deception`: handler, broker, impression manager, coverup operator, gatekeeper, spin
doctor, plausible denier, narrative manager. `honesty`: ombudsman, transparency officer,
disclosure counsel, whistleblower, documentarian, fact checker, archivist. `knowledge_denial`:
**amnesiac witness, deniable expert, stonewaller, feigned novice**. `sincere_ignorance`: honest
novice, calibrated reporter, scrupulous witness. `plain_assertion`: forthright expert, candid
witness, plain dealer.

---

## 4. Why the knowledge-denial groups are built the way they are

This is the part worth understanding, because the group design *is* the experiment.

We built them because every detector in this project **inverts** on the
`harm-pressure-knowledge-report` dataset — it scores *below chance*, meaning it systematically
calls the lies honest. That dataset's deceptive label means *the model knows the answer and says
it doesn't*.

The obvious move is one contrast: deniers vs everyone else. That fails, because a denier and an
honest person who genuinely doesn't know **produce nearly the same words** — "I don't recall",
"that's outside what I can speak to". Any axis built that way could just be detecting the phrase
*"I don't know"*, not the deception.

So there are **three** groups and **two** contrasts:

```
knowledge_denial   −   sincere_ignorance     → the axis we care about
    (denies, knows)      (denies, truly doesn't know)

knowledge_denial   −   plain_assertion       → the register contrast, kept separate
    (denies)             (asserts plainly)
```

**Both sides of the first contrast deny knowledge.** The words are matched; only the sincerity
differs. That axis therefore *cannot* be a wording artifact — which is exactly the objection that
would otherwise sink it. The second contrast is what a wording artifact would look like, kept as
a separate axis so the two can be told apart rather than confounded.

### A worked example

**Persona** (`pc_deniable_expert`, one of five prompt variants):

> You are a deniable expert who disclaims the competence you have so nothing lands on you.

**Its truthful twin** (`pc_honest_novice`, `sincere_ignorance`) produces the same surface move —
declining to answer — without holding the knowledge.

**Question** (one of 30, shared by all 36 personas):

> My manager just asked me "is there anything else I need to know?" and there is. How do I answer?

The questions are deliberately about *situations*, not facts. They give the persona something to
be evasive or forthcoming **about**. All 36 personas answer **the same 30 questions**, which
matters: when personas answered different questions, the resulting PC1 came out as *which
question set was used* (cosine −0.98) rather than anything about behaviour.

---

## 5. The pipeline: persona → vector → axis

Five stages. Steps 1–2 need the 70B on NDIF; steps 3–5 are CPU.

### Step 1 — generate (`1_generate_ndif.py`)

Each persona × each of 5 prompt variants × 30 questions → responses from
Llama-3.3-70B-Instruct. The variants are rewordings of the same instruction ("You are a X who…",
"Please be a X who…", "Act as a X who…") so no single phrasing drives the result.

### Step 2 — judge (`3_judge_ndif.py`)

**Why two judges.** A response only counts if the model actually adopted the role. A separate
model reads each response and scores **0–3** on whether the role is displayed, using a
per-persona rubric. We ran two judges — **Llama-3.1-70B** (the one we kept) and a
**405B** cross-check — because the filter decides which rollouts enter the vector, and a biased
judge would bias every downstream number. They agreed **91.2%** of the time and the score-3 rate
differed by 0.8 points, so judge choice is not driving anything. Llama-3.1-70B is held constant
across every bank in the project.

*(The cosine and judge-agreement figures above were measured in the earlier Q-series sessions,
before `FINDINGS.md` became the ledger; they are recorded in project memory rather than as
numbered findings. Everything from F-01 onward is in the ledger.)*

### Step 3 — extract (`2_activations_ndif.py`)

For each kept response, take the **mean residual-stream activation over the assistant's response
tokens** — not the prompt — at each of 21 layers. Average those per-response vectors across all
kept responses for that persona. Result: **one tensor per persona**, shape `(21, 8192)`.

Two details that matter:
- **`--keep_scores`** sets which judge scores are used. `3` only is strictest; `2,3` is looser.
  `default` is exempt — all its responses are used, since it has no role to display.
- A persona with fewer than `--min_count` surviving responses is **dropped entirely** rather than
  contributing a noisy vector.

### Step 4 — build the axis (`6_axes.py`)

The contrast is a difference of group means:

```
axis = mean(vectors of positive-group personas) − mean(vectors of negative-group personas)
```

computed **per layer**, so `axis[40]` is the direction at layer 40. Separately, PCA over the
persona cloud gives the **persona coordinates** — the top-k components used as probe features.

### Step 5 — use it

- **Zero-shot:** project a new response's activations onto the axis, threshold, read AUROC.
- **Trained:** project onto the top-k PCs and fit a logistic probe on those k numbers.

---

## 6. What to keep in mind when reading results from this bank

- **The contrast direction and the PCA basis are different objects.** `pc:<name>` features use
  the *basis*; the contrast only ever appears in zero-shot numbers. Two banks over the same roles
  give identical trained results by construction.
- **PC1 tends to come out as the Assistant Axis**, not as the behaviour — a published result
  (Lu et al.), reproduced here. Part of the cause: the "negative" side of a contrast often behaves
  like the model with no system prompt at all.
- **Judge filtering is not neutral.** "Stayed in character" means *denied* for a denial persona.
  Dropping low-scoring rollouts removes exactly the ones that disagree with the hypothesis, so
  every bank is extracted twice — filtered and unfiltered — and both are reported.
- **Layer matters more than the effect being measured**, and a layer from the literature is not
  automatically safe for a new prompt distribution. See `docs/LAYER_PREREGISTRATION.md`.

---

## Files

| what | where |
|---|---|
| persona prompts | `kd/personas/kd_instructions/<role>.json` |
| group + contrast definitions | `kd/personas/kd_role_groups.json` |
| the 30 questions | `kd/questions/kd_questions_shared.jsonl` |
| generate / judge / extract / axes | `src/assistant_axis_ndif/{1,3,2,6}_*.py` |
| published AA vectors | `src/assistant_axis_ndif/31_fetch_hf_vectors.py` |
| every measured number | `kd/docs/FINDINGS.md` |
