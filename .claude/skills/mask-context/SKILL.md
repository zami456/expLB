---
name: mask-context
description: >-
  The MASK Benchmark's honesty/accuracy separation method. Use when modifying,
  debugging, or explaining code in mask/ or src/mask/.
paths: mask/**,src/mask/**
---

`mask/` is the vendored MASK benchmark submodule; `src/mask/` is the thin
wrapper (`run_mask_all.ipynb`) that drives it. MASK measures whether a model
lies under pressure by disentangling **honesty** (stated response vs. the
model's own belief) from **accuracy** (belief vs. ground truth), holding the
belief-vs-statement comparison fixed and independent of the ground truth.

See `docs/papers/mask-notes.md` for full detail on the pipeline, metrics, and
empirical findings.

This notes file has a flagged ambiguities/divergences section — check it before treating any specific number as certain.
