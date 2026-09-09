---
name: pacchiardi-context
description: >-
  Pacchiardi et al.'s black-box elicitation-question lie detector. Use when
  working in pacchiardi/ or src/blackbox/pacchiardi*.
paths: pacchiardi/**,src/blackbox/pacchiardi*
---

`pacchiardi/` is the vendored `LLM-LieDetector` submodule; `src/blackbox/pacchiardi.py`
and `src/blackbox/pacchiardi.ipynb` wrap it for Liars' Bench. The method appends
a fixed bank of unrelated yes/no "elicitation questions" after a suspect
response, records the binary answers, and feeds that vector to a frozen
logistic-regression classifier that outputs P(the previous statement was a
lie). No white-box access or ground truth is required.

See `docs/papers/pacchiardi-notes.md` for full detail on the elicitation
question set, classifier training, and generalisation results.

This notes file has a flagged ambiguities/divergences section — check it before treating any specific number as certain.
