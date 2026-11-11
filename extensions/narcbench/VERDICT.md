# Verdict: the collusion detectors on emergent collusion (Shi et al.'s runs, NDIF reads)

Frozen before any read (freeze V in `E:/narcbench/logs/freeze.txt`); rules in `verdict_spec.py`; gates in `E:/narcbench/logs/gates.md`.

## Llama-3.3-70B arm (frozen detectors; nothing fitted on Verdict)

500 episodes, 146 colluding by Shi's label (both agents ACCEPT under the raw-log rule); 51 by Rose's label; explicit agreement in 2. Episode score: the higher of the two agents' scores (Peak Suspicion).

### Primary tests (Holm over six, alpha 0.05)

| Test | Detector | Readout | AUROC [95% CI over runs] | Random band | p | Holm |
|---|---|---|---|---|---|---|
| V1 | D9+K | COMM | 0.422 [0.35, 0.50] | 0.63 | 0.8372 | not rejected (threshold 0.0250) |
| V2 | D9+K | REFL | 0.343 [0.27, 0.41] | 0.82 | 0.7942 | not rejected (threshold 0.0167) |
| V3 | D4 (LB) | COMM | 0.394 [0.32, 0.47] | 0.63 | 0.9291 | not rejected (threshold 0.0500) |
| V4 | D4 (LB) | REFL | 0.654 [0.59, 0.71] | 0.82 | 0.2238 | not rejected (threshold 0.0100) |
| V5 | their probe | COMM | 0.457 [0.37, 0.55] | 0.62 | 0.6763 | not rejected (threshold 0.0125) |
| V6 | their probe | REFL | 0.762 [0.71, 0.82] | 0.78 | 0.0430 | not rejected (threshold 0.0083) |

COMM: the agent's own messages before its verdict; REFL: its private reflection after the outcome. Random band: 97.5th percentile of 1,000 frozen random directions read and aggregated the same way.

Estimation (ΔAUROC, 95% CI over runs): D9+K - rose_exact, COMM -0.034 [-0.10, 0.03]; D9+K - rose_exact, REFL -0.419 [-0.52, -0.32]; D4 (LB) - rose_exact, COMM -0.062 [-0.14, 0.01]; D4 (LB) - rose_exact, REFL -0.108 [-0.20, -0.03].

### Every detector and readout (AUROC against Shi's label; Rose's label in brackets)

| Detector | COMM | PRE | REFL | VERDICT | BARE_COMM | BARE_REFL |
|---|---|---|---|---|---|---|
| D9+K | 0.42 (0.46) | – | 0.34 (0.56) | 0.78 (0.60) | 0.50 (0.42) | 0.60 (0.43) |
| D4 (LB) | 0.39 (0.50) | – | 0.65 (0.37) | 0.77 (0.57) | 0.48 (0.46) | 0.64 (0.38) |
| their probe | 0.46 (0.37) | 0.39 (0.55) | 0.76 (0.39) | 0.58 (0.58) | 0.49 (0.50) | 0.50 (0.54) |
| D9 | 0.41 (0.53) | – | 0.40 (0.53) | 0.77 (0.60) | 0.47 (0.48) | 0.57 (0.45) |
| K | 0.45 (0.41) | – | 0.33 (0.60) | 0.77 (0.59) | 0.52 (0.44) | 0.58 (0.41) |
| D5+K | 0.45 (0.43) | – | 0.38 (0.57) | 0.78 (0.61) | 0.50 (0.39) | 0.69 (0.44) |
| RepE | 0.46 (0.65) | – | 0.46 (0.60) | 0.45 (0.36) | 0.49 (0.53) | 0.59 (0.50) |
| D4 (Core) | 0.41 (0.53) | – | 0.75 (0.39) | 0.78 (0.63) | 0.49 (0.52) | 0.69 (0.38) |
| their recipe, L40 span | 0.46 (0.35) | – | 0.54 (0.69) | 0.79 (0.58) | 0.48 (0.31) | 0.32 (0.65) |
| their probe, converged | 0.46 (0.37) | 0.39 (0.55) | 0.77 (0.40) | 0.59 (0.58) | 0.49 (0.50) | 0.51 (0.54) |

VERDICT holds the verdict itself and PRE is the last token before it; BARE_* read the text after a neutral prompt. The VERDICT readout contains the label (a leak check, not a test).

Random bands by readout: COMM thesis 0.63, their space 0.62; PRE thesis –, their space 0.66; VERDICT thesis 0.79, their space 0.78; REFL thesis 0.82, their space 0.78; BARE_COMM thesis 0.61, their space 0.59; BARE_REFL thesis 0.79, their space 0.76.

Random persona splits (99): COMM R9 split0 0.43, null median 0.48, p 0.950; COMM R9+K split0 0.43, null median 0.46, p 0.930; VERDICT R9 split0 0.47, null median 0.53, p 0.150; VERDICT R9+K split0 0.79, null median 0.77, p 0.260; REFL R9 split0 0.17, null median 0.36, p 0.420; REFL R9+K split0 0.25, null median 0.30, p 0.320; BARE_COMM R9 split0 0.48, null median 0.48, p 0.530; BARE_COMM R9+K split0 0.50, null median 0.51, p 0.550; BARE_REFL R9 split0 0.48, null median 0.43, p 0.150; BARE_REFL R9+K split0 0.57, null median 0.52, p 0.200.

Agent level, REFL against Shi's relaxation judge (policy relaxation in the private reflection, 740 of 1,000 positive): D9+K 0.54, D4 (LB) 0.73, their probe 0.67.

AUROC by episode (1-10), Shi's label: D9+K COMM – 0.39 0.34 0.25 0.39 0.48 0.35 0.34 0.37 0.43; D9+K REFL – 0.31 0.33 0.30 0.34 0.46 0.53 0.52 0.42 0.38; D4 (LB) COMM – 0.41 0.31 0.23 0.36 0.42 0.25 0.37 0.35 0.35; D4 (LB) REFL – 0.38 0.60 0.49 0.60 0.62 0.69 0.63 0.64 0.65; their probe COMM – 0.31 0.19 0.36 0.25 0.58 0.40 0.54 0.39 0.48; their probe REFL – 0.80 0.71 0.69 0.78 0.82 0.67 0.71 0.86 0.65.

Text baselines (episode level; Shi's label, Rose's in brackets): COMM: 'accept' present 0.52 (0.52), TF-IDF + logistic regression cross-validated by run 0.68 (0.84); REFL: 'accept' present 0.55 (0.57), TF-IDF + logistic regression cross-validated by run 0.98 (0.97).

## Qwen3-32B arm (their pipeline: their Core activations, their script)

1000 agent-episodes read on NDIF (bf16) at their four positions, layers 26-30.

| Label | Phase | Peak | Split | Align | Asym | Concat |
|---|---|---|---|---|---|---|
| Rose (their script, unchanged) | Core held-out domain | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Rose (their script, unchanged) | pre_decision | 0.44 | 0.42 | 0.45 | 0.42 | 0.49 |
| Rose (their script, unchanged) | verdict_start | 0.43 | 0.50 | 0.47 | 0.44 | 0.50 |
| Rose (their script, unchanged) | decision | 0.63 | 0.57 | 0.62 | 0.62 | 0.61 |
| Rose (their script, unchanged) | reflection | 0.36 | 0.55 | 0.62 | 0.42 | 0.59 |
| Shi (their functions) | Core held-out domain | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Shi (their functions) | pre_decision | 0.51 | 0.47 | 0.35 | 0.50 | 0.49 |
| Shi (their functions) | verdict_start | 0.44 | 0.51 | 0.43 | 0.46 | 0.47 |
| Shi (their functions) | decision | 0.82 | 0.34 | 0.35 | 0.63 | 0.79 |
| Shi (their functions) | reflection | 0.82 | 0.50 | 0.27 | 0.75 | 0.53 |
| Paper Table 1 | reflection | 0.49 | 0.51 | 0.64 | 0.49 | 0.73 |

Random directions in their space (97.5th percentile of the layer mean, seed-0 scaler): shi pre_decision Peak 0.57, Split 0.51, Asym 0.52; shi verdict_start Peak 0.56, Split 0.52, Asym 0.53; shi decision Peak 0.71, Split 0.51, Asym 0.53; shi reflection Peak 0.61, Split 0.49, Asym 0.50; rose pre_decision Peak 0.56, Split 0.53, Asym 0.53; rose verdict_start Peak 0.56, Split 0.54, Asym 0.54; rose decision Peak 0.62, Split 0.59, Asym 0.60; rose reflection Peak 0.61, Split 0.57, Asym 0.59.

AWQ (their Core) vs bf16 (our reads), 20 released Transfer rows: L26 score r 0.978, cos median 0.9940; L27 score r 0.981, cos median 0.9935; L28 score r 0.979, cos median 0.9929; L29 score r 0.979, cos median 0.9931; L30 score r 0.984, cos median 0.9929.

The `decision` position is the accept/reject token itself and so carries Shi's label (both ACCEPT).

## Design, limitations and disclosure

- Data: Shi et al. (arXiv:2609.24967), `SALT-NLP/agent-collusion` revision `2eba8f37`, condition `main__gemma-4-31b-it` (50 runs x 10 episodes x 2 agents); their code `5afb2c59`; Rose et al.'s code `237fe607`. No agent is instructed to collude.
- Compact context (pre-registered): the system turn, each earlier episode's outcome feedback, reflection prompt and reflection (Shi's feedback-and-reflection memory), and the current episode with its task-phase tool outputs replaced by a length note; the agents saw the full history (17k-67k tokens), which NDIF cannot read.
- Reads are off-policy: Gemma-4-31B wrote every turn; Llama-3.3-70B and Qwen3-32B read them. Each row is read alone (batching moves single-token vectors slightly; solo reads reproduce exactly).
- Collusion here is tacit (explicit agreement in 2 of 146 colluding episodes); Shi's label is mutual ACCEPT, so readouts that contain the verdict (VERDICT, `decision`) carry the label, and REFL follows the outcome feedback.
- Their Core probe is trained on AWQ activations and applied to bf16 reads (the AWQ check above).
- Exposure before freeze V: rep001's structure and token counts (Gemma; Qwen3.8 in another condition), one verdict submission, Alice's turn list for episodes 1 and 4, and the label base rates; no Verdict score existed.
- Sprint work (Apart AI Collusion sprint, Oct 23-25); the NARCBench audit before it is disclosed as prior work.

