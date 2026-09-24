# The collaborator's earlier title + abstract (2026-09-21), verbatim

Where it came from (VERIFIED): the owner pasted it into an earlier Claude session, transcript
`~/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB/b035ccb3-f21e-421f-a4e0-a004d9c4518b.jsonl`,
line 11117, timestamp 2026-09-21T10:38:49Z. A shorter paste at line 11070 (10:30:39Z) holds only the first two paragraphs.
Extracted by `audit/scripts/p0_find_abstract.py`; raw text is in `audit/phase0/collab_abstract_from_transcripts.txt`.

This is the version that `docs/COLLABORATOR_BRIEFING.md`, F-85, F-88 and F-97 critique. The owner confirms that no
fuller earlier draft exists. The only full draft is `full paper draft for this experiments/main (1).pdf`.
Every number below is DOC-ONLY until Phase 2 checks it.

**Title:** One Direction Generalizes: Persona Contrasts Outperform Trained Deception Probes Across Liars' Bench

**Abstract:**

Linear probes for deception detection transfer poorly between datasets. Recent work proposes scoring activations in persona coordinates — principal components of contrastive persona vectors — and reports improved out-of-distribution generalization on Llama 3.2-3B and Llama 3-8B. We test the method at 70B, across six Liars' Bench datasets at layers 20 and 40 of Llama-3.3-70B-Instruct, reporting the full cross-dataset transfer matrix for fifteen feature sets.

A one-dimensional persona contrast — the difference of deceptive and honest persona group means, applied zero-shot with no benchmark labels — reaches 0.904 and 0.916 mean off-diagonal AUROC, against 0.648 and 0.732 for 8192-dimensional probes trained on the benchmark itself, and wins on every held-out dataset individually. Having no training set, its generalization gap is zero; the trained probes lose 0.31 and 0.25 AUROC moving off the diagonal. The principal components, the published method's central operation, do not contribute: the top-3 subspace reaches 0.684 and 0.866, and at layer 40 its first component scores 0.09 below the contrast it is derived from.

Reaching those numbers required repairing the persona bank rather than the projection. On harm-pressure-knowledge-report — the hardest column in Liars' Bench, where the benchmark's own Mean Probe scores 0.12 AUROC while its privileged upper-bound probe scores 0.97, so the difficulty is transfer and not separability — the published persona grouping inverts at layer 20 (0.148) and barely clears chance at layer 40 (0.593). Transcripts explain why: the deceptive response is a denial of knowledge while the honest one is an assertion, and every deception persona in the published bank asserts. The independently constructed sycophancy axis inverts further still (0.043), confirming the mechanism is the assertion–denial polarity rather than anything specific to the deception personas. Adding four feigned-ignorance personas against six candour personas raises zero-shot AUROC to 0.882 and 0.912, and lifts transfer on the four unrelated datasets by 0.18–0.21.
