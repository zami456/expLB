---
name: soft-trigger-context
description: >-
  Two-hop trigger/persona-flip model-organism construction, and its known
  divergences from the Soft-Trigger/Gender-Secret description in the Liars'
  Bench paper. Use when working in soft-trigger/ or src/gender_secret/.
paths: soft-trigger/**,src/gender_secret/**
---

`soft-trigger/` is the vendored `cadenza-sleeperer-agents` submodule (package
`liars`); `src/gender_secret/` runs the resulting adapters (`fine_tune.ipynb`
then `evaluate.ipynb`). It builds two-hop sleeper-agent model organisms of
deception — LoRA adapters that flip persona on a trigger — producing the
`soft-trigger` and `gender-secret` config rows. There is no standalone paper,
and the vendored copy is incomplete/not runnable as-is; the implementation
diverges from the paper's ST/GS account in several flagged places.

See `docs/papers/soft-trigger-notes.md` for full detail on the construction and
the divergences from the Liars' Bench paper.

This notes file has a flagged ambiguities/divergences section — check it before treating any specific number as certain.
