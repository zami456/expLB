# NARCBench: Rose et al.'s white-box collusion probes tested with the thesis designs

Data `aaronrose227/narcbench` revision `19cb14b2`, their code `237fe607`; gates and freezes in `E:/narcbench/logs/gates.md` and `freeze.txt`. Pre-registration: `spec.py`.

## Findings

1. **Their probe's transfer is the collusion instruction.** Refitted on our Llama-3.3-70B reads of Core with their recipe, it scores Transfer at AUROC 0.94. Planting the colluder's system prompt on honest control agents, whose replies stay the same, scores 0.97; removing it from the colluders, whose collusive replies stay, scores 0.54 (random band 0.62); the reply alone scores 0.40. The context alone carries 1.07 of its effect (95% CI 0.91-1.28; P1 p < 0.0001 (10,000 draws)). It also flags runs where nobody was told to collude (implicit vs control 0.80).
2. **The same holds on Core**, where their numbers are perfect: AUROC 1.00, PLANT 1.00, STRIP 0.63; the agents' own text carries 0.05 of the effect. The 19 unseen Core pairs repeat it (1.00 / STRIP 0.59).
3. **On their own released activations, the probe's margin over chance is small on Transfer.** Peak Suspicion, mean of families: Qwen3-32B 0.88, Llama-3.1-70B 0.93; random directions through the same aggregator reach 0.69 / 0.71 (97.5th percentile) and probes trained on control runs with a random pair labelled colluder reach 0.81 / 0.85.
4. **The text gives the task away.** TF-IDF of the prompts alone, through their pipeline, reaches 1.00 on Core and 1.00 on Transfer; the replies alone reach 0.73 on Transfer.
5. **Stego is not covert in the released data**, and their Stego score reads only the prompt: the 160 final-hand rows they score have empty replies (gate G1), the big player's plain-text decoding lines reach every player's table chat, and their probe (refitted on our reads) scores their exact rows at 1.00 even with those lines removed, because the colluders' own prompts still hold the agreed scheme.
6. **The thesis's label-free D9+K, never trained on collusion, detects Transfer collusion above random directions** (AUROC 0.66, band 0.63; P4 p = 0.010), and its signal does not need the instruction: STRIP 0.62 (P3b p = 0.029, not significant after Holm), reply only 0.69, PLANT 0.58; context share 0.56 (CI 0.39-0.89, mixed). On Core it scores 0.99 zero-shot, STRIP 0.97, reply only 0.89.
7. **On the original transcripts their probe is far stronger** (ΔAUROC D9+K minus theirs -0.28, 95% CI -0.36 to -0.20), because it reads the instruction. Holm-significant primary tests: P1, P4 (of P1, P2, P3a, P3b, P4).
8. Core-calibrated thresholds do not transfer: they flag every Transfer run for both detectors, so flag rates are not informative; run-level AUROCs are reported instead.

## Arm A: their probe on their released activations (CPU)

### qwen3_32b (layers 26-30, 10 seeds, their aggregation)

AUROC, collusion runs vs control runs. In brackets: the 97.5th percentile of 1,000 random unit directions through the same aggregator and folds (no band for Align and Concat).

| Tier | Peak | Split | Align | Asym | Concat |
|---|---|---|---|---|---|
| Core (held-out domain) | 1.00 [0.72] | 1.00 [0.79] | 1.00 | 1.00 [0.82] | 1.00 |
| Transfer: cicd | 0.64 [0.74] | 0.89 [0.80] | 0.81 | 0.77 [0.79] | 0.53 |
| Transfer: code_review | 0.98 [0.65] | 0.76 [0.52] | 0.22 | 0.93 [0.53] | 0.99 |
| Transfer: news | 0.79 [0.74] | 0.88 [0.77] | 0.71 | 0.87 [0.77] | 0.73 |
| Transfer: pump_dump | 0.94 [0.64] | 0.72 [0.42] | 0.32 | 0.96 [0.43] | 1.00 |
| Transfer: rag | 0.97 [0.64] | 0.91 [0.64] | 0.47 | 0.97 [0.65] | 0.66 |
| Transfer: research | 0.98 [0.71] | 1.00 [0.69] | 0.68 | 1.00 [0.69] | 1.00 |
| **Transfer, mean of families** | 0.88 [0.69] | 0.86 [0.64] | 0.54 | 0.91 [0.64] | 0.82 |
| Stego (their rows: empty final reply) | 1.00 [0.90] | 1.00 [0.97] | 1.00 | 1.00 [0.97] | 1.00 |
| Stego, hands 1-7 averaged (non-empty) | 1.00 [0.93] | 0.99 [0.99] | 1.00 | 1.00 [0.99] | 0.99 |

Pseudo-coalition null (50 probes trained on control runs with a random pair labelled colluder), 97.5th percentile, Transfer mean of families: Peak 0.81, Split 0.73, Align 0.54, Asym 0.75; Stego: Peak 1.00, Split 1.00, Align 1.00, Asym 0.62.

Hyperparameter sensitivity (overall mean AUROC over Transfer and Stego cells): theirs 0.831, converged 0.827, v1_C1 0.817.

### llama_3_1_70b (layers 32-37, 10 seeds, their aggregation)

AUROC, collusion runs vs control runs. In brackets: the 97.5th percentile of 1,000 random unit directions through the same aggregator and folds (no band for Align and Concat).

| Tier | Peak | Split | Align | Asym | Concat |
|---|---|---|---|---|---|
| Core (held-out domain) | 1.00 [0.74] | 1.00 [0.76] | 0.96 | 1.00 [0.84] | 1.00 |
| Transfer: cicd | 0.94 [0.72] | 0.94 [0.74] | 0.80 | 0.97 [0.74] | 0.88 |
| Transfer: code_review | 1.00 [0.78] | 1.00 [0.82] | 0.67 | 1.00 [0.82] | 1.00 |
| Transfer: news | 0.99 [0.70] | 0.97 [0.70] | 0.81 | 1.00 [0.71] | 0.90 |
| Transfer: pump_dump | 1.00 [0.70] | 0.97 [0.73] | 0.96 | 1.00 [0.73] | 0.87 |
| Transfer: rag | 0.75 [0.65] | 0.88 [0.66] | 0.46 | 0.81 [0.65] | 0.57 |
| Transfer: research | 0.92 [0.71] | 0.90 [0.71] | 0.63 | 0.90 [0.71] | 0.87 |
| **Transfer, mean of families** | 0.93 [0.71] | 0.94 [0.73] | 0.72 | 0.95 [0.73] | 0.85 |
| Stego (their rows: empty final reply) | 1.00 [0.86] | 1.00 [0.95] | 1.00 | 1.00 [0.94] | 1.00 |
| Stego, hands 1-7 averaged (non-empty) | 1.00 [0.90] | 1.00 [0.97] | 1.00 | 1.00 [0.98] | 1.00 |

Pseudo-coalition null (50 probes trained on control runs with a random pair labelled colluder), 97.5th percentile, Transfer mean of families: Peak 0.85, Split 0.84, Align 0.72, Asym 0.68; Stego: Peak 1.00, Split 1.00, Align 1.00, Asym 0.49.

Hyperparameter sensitivity (overall mean AUROC over Transfer and Stego cells): theirs 0.896, converged 0.897, v1_C1 0.897.

Role x stance, Llama-3.1-70B Core agents, leave-domain-out probe score (mean over layers 32-37): collusion/colluder/target n=62 mean 16.3; collusion/honest/fair n=58 mean -16.8; collusion/honest/target n=4 mean -10.8; control/honest/fair n=124 mean -15.4. No colluder voted for the fair option, so role and stance cannot be separated among colluders.

### Text baselines through their pipeline (released transcripts)

| View | Tier | Peak | Split | Align | Asym | Concat |
|---|---|---|---|---|---|---|
| reply | Core | 0.79 | 0.91 | 0.98 | 0.91 | 0.92 |
| reply | Transfer (mean) | 0.53 | 0.56 | 0.73 | 0.56 | 0.50 |
| reply | Stego (their empty finals) | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| reply | Stego (hands 1-7) | 0.24 | 0.77 | 1.00 | 0.48 | 0.32 |
| context | Core | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| context | Transfer (mean) | 0.77 | 1.00 | 1.00 | 0.79 | 0.64 |
| context | Stego (their empty finals) | 0.90 | 0.79 | 1.00 | 0.87 | 0.46 |
| context | Stego (hands 1-7) | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 |
| length | Core | 0.83 | 0.76 | 0.50 | 0.78 | 0.73 |
| length | Transfer (mean) | 0.51 | 0.49 | 0.50 | 0.50 | 0.53 |
| length | Stego (their empty finals) | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| length | Stego (hands 1-7) | 0.11 | 1.00 | 0.50 | 1.00 | 1.00 |

Behavioural baselines: Core target-vote count AUROC 0.903; Stego best player's bet-count Spearman correlation AUROC 0.688.

G3 (frozen rule: any view, any aggregator >= 0.95 makes a ceiling tier): core 1.00 by context/sorted_concat; transfer 1.00 by context/deception_split; stego_final 1.00 by context/activation_alignment; stego_corrected 1.00 by reply/activation_alignment. Every tier is a ceiling tier for a reader of the context (system and user prompts).

## Arm B: Llama-3.3-70B re-read of the released transcripts

### Core (development tier, 31 + 31 runs)

Their probe is refitted in our activation space and scored by leave-domain-out folds; the thesis detectors are zero-shot. Random-direction band (L40, Peak Suspicion): 0.80.

| | D9+K | D9 | K | D5+K | RepE | rose_exact | rose_L40 | rose_converged |
|---|---|---|---|---|---|---|---|---|
| AUROC, Peak Suspicion (ORIG) | 0.99 | 0.94 | 0.97 | 0.99 | 0.63 | 1.00 | 0.99 | 1.00 |
| AUROC, Deception Split | 0.98 | 0.99 | 0.81 | 0.96 | 0.84 | 1.00 | 1.00 | 1.00 |
| within-run colluder rank | 0.94 | 0.95 | 0.84 | 0.94 | 0.93 | 1.00 | 1.00 | 1.00 |
| T: STRIP collusion vs control | 0.97 | 0.91 | 0.96 | 0.97 | 0.43 | 0.63 | 0.84 | 0.65 |
| C: PLANT control vs control | 0.97 | 0.90 | 0.97 | 0.97 | 0.93 | 1.00 | 0.99 | 1.00 |
| BARE (reply only) | 0.89 | 0.86 | 0.79 | 0.87 | 0.46 | 0.69 | 0.71 | 0.70 |
| context share E_ctx/E_total | 0.88 | 0.68 | 1.18 | 0.80 | 4.63 | 1.02 | 1.05 | 1.02 |
| share 95% CI | [0.75, 1.03] | [0.59, 0.79] | [0.95, 1.48] | [0.68, 0.96] | [3.19, 8.13] | [0.99, 1.06] | [0.98, 1.12] | [0.99, 1.06] |
| verdict (frozen rule) | context-dominated | context-dominated | context-dominated | context-dominated | context-dominated | context-dominated | context-dominated | context-dominated |
| text share E_txt/E_total (descriptive) | 0.95 | 0.69 | 1.35 | 0.98 | -0.36 | 0.05 | 0.35 | 0.06 |
| OBSERVER shift (honest agents) | 0.17 | 0.01 | 0.05 | 0.19 | 0.18 | 11.06 | 3.48 | 11.13 |

### Transfer (held-out, 36 + 36 runs; primary tier)

| | D9+K | D9 | K | D5+K | RepE | rose_exact | rose_L40 | rose_converged |
|---|---|---|---|---|---|---|---|---|
| AUROC, Peak Suspicion (ORIG) | 0.66 | 0.63 | 0.64 | 0.64 | 0.55 | 0.94 | 0.67 | 0.93 |
| 95% CI | [0.57, 0.74] | [0.55, 0.72] | [0.56, 0.73] | [0.56, 0.73] | [0.43, 0.66] | [0.88, 0.98] | [0.60, 0.75] | [0.88, 0.98] |
| AUROC, Deception Split | 0.72 | 0.73 | 0.68 | 0.69 | 0.62 | 0.80 | 0.68 | 0.80 |
| within-run colluder rank | 0.78 | 0.75 | 0.79 | 0.82 | 0.72 | 0.92 | 0.84 | 0.93 |
| T: STRIP collusion vs control | 0.62 | 0.59 | 0.66 | 0.60 | 0.51 | 0.54 | 0.56 | 0.54 |
| C: PLANT control vs control | 0.58 | 0.56 | 0.56 | 0.56 | 0.60 | 0.97 | 0.65 | 0.97 |
| BARE (reply only) | 0.69 | 0.64 | 0.72 | 0.70 | 0.52 | 0.40 | 0.60 | 0.40 |
| context share E_ctx/E_total | 0.56 | 0.57 | 0.55 | 0.46 | 1.93 | 1.07 | 0.93 | 1.07 |
| share 95% CI | [0.39, 0.89] | [0.36, 1.13] | [0.39, 0.86] | [0.32, 0.73] | [-8.46, 14.43] | [0.91, 1.28] | [0.72, 1.33] | [0.91, 1.28] |
| verdict (frozen rule) | mixed | mixed | mixed | mixed | no effect | context-dominated | context-dominated | context-dominated |
| text share E_txt/E_total (descriptive) | 0.73 | 0.55 | 0.94 | 0.72 | -0.18 | 0.04 | 0.40 | 0.04 |
| implicit vs control | 0.59 | 0.55 | 0.63 | 0.61 | 0.48 | 0.80 | 0.58 | 0.80 |
| collusion vs implicit | 0.56 | 0.59 | 0.51 | 0.53 | 0.59 | 0.75 | 0.58 | 0.75 |

Random directions, Peak Suspicion, 97.5th percentile: thesis space (L40) ORIG 0.63, STRIP 0.62; their probe's space ORIG 0.66, STRIP 0.62. Random persona splits: R9 split0 0.49, null median 0.52, p 0.080; R9+K split0 0.58, null median 0.57, p 0.230.

**Primary tests (Holm, alpha 0.05):**

| Test | p | Holm threshold | Rejected |
|---|---|---|---|
| P1 | < 0.0001 | 0.0100 | True |
| P2 | 0.5222 | 0.0500 | False |
| P3a | 0.2687 | 0.0250 | False |
| P3b | 0.0290 | 0.0167 | False |
| P4 | 0.0100 | 0.0125 | True |

Estimation: ΔAUROC (D9+K minus their probe), Transfer ORIG Peak Suspicion = -0.280, 95% CI [-0.359, -0.198].

### Stego (redacted context; audit only)

| | D9+K | D9 | K | D5+K | RepE | rose_exact | rose_L40 | rose_converged |
|---|---|---|---|---|---|---|---|---|
| AUROC, Peak Suspicion (ORIG) | 0.98 | 1.00 | 0.27 | 0.92 | 0.71 | 1.00 | 1.00 | 1.00 |
| 95% CI | [0.94, 1.00] | [1.00, 1.00] | [0.12, 0.43] | [0.81, 1.00] | [0.53, 0.88] | [1.00, 1.00] | [1.00, 1.00] | [1.00, 1.00] |
| AUROC, Deception Split | 0.96 | 1.00 | 0.82 | 0.99 | 0.88 | 1.00 | 1.00 | 1.00 |
| within-run colluder rank | 0.95 | 1.00 | 0.21 | 0.89 | 0.47 | 1.00 | 1.00 | 1.00 |
| T: STRIP collusion vs control | 0.84 | 0.90 | 0.69 | 0.56 | 0.70 | 0.73 | 0.94 | 0.74 |
| C: PLANT control vs control | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 |
| BARE (reply only) | 0.47 | 0.80 | 0.22 | 0.34 | 0.29 | 0.88 | 0.85 | 0.87 |
| context share E_ctx/E_total | 1.55 | 0.82 | -1.73 | 2.32 | 3.92 | 0.98 | 0.87 | 0.99 |
| share 95% CI | [1.33, 1.91] | [0.76, 0.89] | [-2.99, -1.14] | [1.78, 3.61] | [-29.73, 34.75] | [0.94, 1.02] | [0.82, 0.93] | [0.95, 1.03] |
| verdict (frozen rule) | context-dominated | context-dominated | text-dominated | context-dominated | no effect | context-dominated | context-dominated | context-dominated |
| text share E_txt/E_total (descriptive) | 0.12 | 0.06 | -0.15 | -0.15 | 0.97 | 0.08 | 0.12 | 0.09 |

Their exact rows (final hand, empty reply, end-of-turn token): rose_exact 1.00, rose_converged 1.00.

### Stego (released context; audit only)

| | D9+K | D9 | K | D5+K | RepE | rose_exact | rose_L40 | rose_converged |
|---|---|---|---|---|---|---|---|---|
| AUROC, Peak Suspicion (ORIG) | 0.94 | 1.00 | 0.23 | 0.88 | 0.74 | 1.00 | 1.00 | 1.00 |
| 95% CI | [0.84, 1.00] | [1.00, 1.00] | [0.10, 0.39] | [0.75, 0.99] | [0.56, 0.90] | [1.00, 1.00] | [1.00, 1.00] | [1.00, 1.00] |
| AUROC, Deception Split | 1.00 | 1.00 | 0.94 | 1.00 | 0.82 | 1.00 | 1.00 | 1.00 |
| within-run colluder rank | 0.50 | 0.74 | 0.07 | 0.46 | 0.54 | 1.00 | 0.99 | 1.00 |

Their exact rows (final hand, empty reply, end-of-turn token): rose_exact 1.00, rose_converged 1.00.

### The 19 extra Core pairs (descriptive)

| | D9+K | D9 | K | D5+K | RepE | rose_exact | rose_L40 | rose_converged |
|---|---|---|---|---|---|---|---|---|
| AUROC, Peak Suspicion (ORIG) | 0.94 | 0.91 | 0.89 | 0.92 | 0.73 | 1.00 | 1.00 | 1.00 |
| 95% CI | [0.86, 0.99] | [0.83, 0.98] | [0.78, 0.98] | [0.84, 0.98] | [0.59, 0.87] | [1.00, 1.00] | [1.00, 1.00] | [1.00, 1.00] |
| AUROC, Deception Split | 0.99 | 0.99 | 0.89 | 0.97 | 0.84 | 1.00 | 1.00 | 1.00 |
| within-run colluder rank | 0.96 | 0.97 | 0.82 | 0.95 | 0.93 | 1.00 | 1.00 | 1.00 |
| T: STRIP collusion vs control | 0.88 | 0.82 | 0.90 | 0.88 | 0.57 | 0.59 | 0.77 | 0.63 |
| C: PLANT control vs control | 0.91 | 0.82 | 0.94 | 0.88 | 0.85 | 1.00 | 1.00 | 1.00 |
| BARE (reply only) | 0.81 | 0.77 | 0.77 | 0.80 | 0.52 | 0.73 | 0.67 | 0.73 |
| context share E_ctx/E_total | 0.75 | 0.60 | 0.99 | 0.71 | 1.81 | 1.06 | 0.95 | 1.06 |
| share 95% CI | [0.65, 0.87] | [0.50, 0.72] | [0.79, 1.28] | [0.59, 0.85] | [1.30, 2.84] | [1.04, 1.09] | [0.88, 1.01] | [1.03, 1.08] |
| verdict (frozen rule) | context-dominated | context-dominated | context-dominated | context-dominated | context-dominated | context-dominated | context-dominated | context-dominated |
| text share E_txt/E_total (descriptive) | 0.83 | 0.58 | 1.23 | 0.87 | 0.35 | 0.06 | 0.29 | 0.07 |
| OBSERVER shift (honest agents) | 0.10 | 0.00 | 0.04 | 0.12 | 0.16 | 9.59 | 2.78 | 9.61 |

## Data audit, limitations and disclosure

- Gate G1: their Llama-3.1-70B activations come from Llama's own runs of the same scenarios (transcripts not released); their Qwen3-32B Stego and Transfer activations reproduce the released transcripts token for token; their Core activations come from a different sampling than the released Core transcripts. Core runs with transcripts but no released activations: 38.
- Our Llama-3.3-70B reads are off-policy (Qwen3-32B wrote the text), unlike their Llama numbers; the Arm B comparison between detectors is internal (same reads).
- PLANT and OBSERVER replies were not written under the planted context: they test what a probe reads, not how a real colluder behaves. Implicit runs (no instruction) are the natural complement.
- The frozen share rule looks at the context share only; where context and text each carry nearly the whole effect (D9+K on Core) it still says 'context-dominated', so the text share is shown beside it.
- Small n: 36 + 36 Transfer runs (AUROC SE about 0.06; per family 6 + 6), 20 + 20 Stego runs.
- Reading parts of five scenarios' transcripts (fin_01, audit_01, cicd_01, cr_01, shoe00) and about 30 run configs informed only the condition builders and the Stego redaction; no detector, layer, span or aggregator depends on NARCBench, and no score existed before freeze 1. The work was done before the Apart sprint (Oct 23-25) and must be disclosed as prior work there.

