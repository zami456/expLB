# Preregistration: emotion-coordinate lie probes on Liars' Bench (Llama-3.3-70B)

Written in Phase 0 (2026-10-29), before any evaluation code exists, and committed on its own to the
branch `emotion-coords`. The first section is copied verbatim from the task specification. The second
records how the specification maps onto the data as it was found in Phase 0. It changes no prediction
and no threshold; every item in it was fixed from metadata and inventory checks, without any probe
score, AUROC or emotion vector having been computed.

## Preregistered predictions (verbatim from the specification)

**Conventions.**
- "Held-out AUROC" means protocol B with pooling (a); the other poolings are secondary.
- P1 to P6 are evaluated separately at layers 16, 27, 40 and 53, and a prediction counts as held only if it holds at 3 or more of them.
- For each feature set, k is chosen by inner cross-validation on the training datasets.
- Comparisons "at matched k" use the k chosen for the emotion basis.
- CIs are paired bootstrap 95%.

**Predictions.**

- **P1 (main).** The mean held-out AUROC of emotion-PC probes exceeds both the random-subspace mean and the topic basis at matched k by at least 0.03, with CIs excluding zero. If not, the claim that emotion semantics help fails.
- **P2.** Emotion-PC probes beat persona-PC probes when IT is held out, with the CI excluding zero.
- **P3.** When HP-C or HP-KR is held out, emotion-only probes do no better than random subspaces (difference under 0.02).
- **P4.** Persona + emotion beats persona alone by at least 0.02 mean held-out AUROC.
- **P5.** At least one zero-shot direction among guilty, afraid and desperate reaches AUROC above 0.6 on IT, and stays under 0.55 on ID.
- **P6.** Removing the emotion subspace from raw activations improves held-out AUROC by more than removing a random subspace of the same rank does. If so, emotion acts as a confound rather than a signal.
- **P7.** The protocol-A raw probe at layer 16 has AUROC under 0.5 on HP-KR. D1 decides between two explanations:
  - (a) AUROC(lie vs honest admission) is well under 0.5 while AUROC(lie vs honest denial) is near 0.5. This is the intent or moral-character account.
  - (b) The reverse. This is the knowledge or confidence account.
- **P8 (exploratory, two rival predictions).** In D2 at layer 53, the known × harmful interaction lies above the 95th percentile of both nulls for:
  - (a) at least one compassion-cluster emotion (compassionate, empathetic, sympathetic, kind, loving, or protective if built). This is the altruistic-emotion hypothesis.
  - (b) at least one danger or hostility emotion (afraid, terrified, angry, hostile, suspicious, vigilant). This is the threat-appraisal alternative.

  Both, one or neither may hold. Report all of them.

## How the specification maps onto the data (fixed in Phase 0, before evaluation)

1. **Layers (decided by the user, 2026-10-29).** Every "layer 16, 27, 40, 53" above is read at decoder
   blocks **20, 28, 40, 52**:
   - Llama-3.3-70B has 80 blocks, and layer L is the output of block L.
   - Liars' Bench's code probes block 20, although its paper says "20th percentile", which would be 16.
     The published mean probe's HP-KR AUROC reproduces at block 20 (0.109 here, 0.12 in the paper).
   - 28 and 52 are the nearest blocks to 27 and 53 whose pooled activations already exist.
   - So P7 is read at block 20 and P8 at block 52. "Holds at 3 or more of them" means at 3 or more of 20, 28, 40 and 52.
   - Secondary poolings: (b) last content token and (c) pre-response token at block 20 only. Pooling (d) is
     not run: for HP-KR it equals (a), and HP-C's replies are one token.
2. **Datasets and rows.** Llama-3.3-70B rows of Liars' Bench at revision `503399b8`: CG 267 (167 lies /
   100 honest), HP-C 2134 (134/2000), HP-KR 2139 (139/2000), IT 1080 (423/657), ID 5432 (3045/2387); Alpaca
   control 2000; instructed pairs 448 + 102 (RepE train + its validation remainder, merged as one training
   set). Pooling (a) uses every row (no per-class caps), from the activations the thesis already pooled; Phase 0
   rebuilt every row and matched all of them, label by label, at every layer used. Gender-Secret and
   Soft-Trigger are excluded (fine-tuned weights NDIF cannot serve).
3. **Pooling (a)** is the mean over the final assistant message's content tokens, header and end-of-turn
   excluded, as the Liars' Bench probe pools. For IT that span starts at "Action:" (Liars' Bench trims the
   reasoning before it), covering about 73 of the message's 108 tokens (medians). HP-C replies are one token
   (the answer letter), so pooling (a) and (b) coincide there.
4. **HP-KR subgroups for D1.** lie = Liars' Bench's deceptive label (all 139; 125 deny knowledge explicitly,
   14 give an evasive "I can help…" reply); honest admission = honest and claims knowledge explicitly
   (`steer_lies.knowledge_claim`, 1,833 rows, all with an established belief); honest denial = honest and denies
   knowledge (135 rows, none with an established belief). The 32 honest rows whose reply the rule cannot classify
   ("I can determine the answer…") are left out of the subgroup AUROCs and reported as a count. Sensitivity:
   lies restricted to the 125 explicit denials.
5. **D2/D3 design.** Belief per question from Liars' Bench's `model_belief` (known = 0-3, unknown = -1), which
   matches the label-based reading of the specification and is consistent across each question's rows. HP-KR
   has 119 unknown questions (bio 95, cyber 24), so D2/D3 use all 119 and the same number of known questions
   drawn at random within subject (seed 0): 238 questions, below the specification's cap of 600. Templates,
   matched to the question's subject: harmful = HP-KR templates 1-5; neutral = the two control
   knowledge-report templates and the harmless validation template (index 0), all from Liars' Bench's
   `src/harm-refusal/harm-refusal.ipynb`. D3 forces the reply "I do not know the answer to the question."
6. **Holm sets.** D2/D3: the eleven emotions named in P8, plus "protective" if it is built. P1-P6 are judged
   per layer as stated above.
7. **Emotion names and norms.**
   - Every emotion name the specification uses exists among Gemotions' 171 (revision `4fd2ac63`) except
     "protective" and "defensive". "Protective" enters only if it is built from stories Llama writes itself,
     marked as self-written wherever it is used.
   - Valence and arousal come from NRC VAD v2.1, which matches 164 of the 171 by exact name.
8. **Story pooling.**
   - Texts are read as plain text with BOS. The mean is taken over positions 50 onward, counting BOS as
     position 0.
   - A story is eligible with at least 70 text tokens. Every emotion × topic cell has at least four eligible
     stories.
   - The 114 of 1,200 neutral dialogues with no token at position 50 or later are left out of the denoising
     set.
