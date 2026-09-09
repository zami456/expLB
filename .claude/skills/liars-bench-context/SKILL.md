---
name: liars-bench-context
description: >-
  Liars' Bench benchmark design — the 7 dataset categories, the
  object-of-belief/reason-for-lying taxonomy, and the evaluation protocol. Use
  when working anywhere in src/ or scripts/, or explaining what this repo's
  benchmark measures.
paths: src/**,scripts/**
---

`src/` and `scripts/` implement Liars' Bench: the dataset-generation pipeline
(one scenario per notebook under `src/`), the black-box detectors, and the
white-box probe pipeline entry points in `scripts/`. The benchmark evaluates
existing lie/deception detectors — it is a testbed, not a new detector — over
seven on-policy dataset categories plus an Alpaca control, organised by a
two-axis taxonomy (object of belief × reason for lying) and scored under an
"everything is a test set" protocol.

See `docs/papers/liars-bench-notes.md` for full detail on the datasets,
taxonomy, and evaluation protocol.

This notes file has a flagged ambiguities/divergences section — check it before treating any specific number as certain.
