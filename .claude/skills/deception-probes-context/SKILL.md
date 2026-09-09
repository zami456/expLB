---
name: deception-probes-context
description: >-
  Apollo's linear-probe deception detection method (this dir is vendored, not a
  submodule — see repo CLAUDE.md). Use when working in
  src/probes/deception-detection/.
paths: src/probes/deception-detection/**
---

`src/probes/deception-detection/` is a vendored fork of Apollo Research's
`deception-detection` code (a regular directory, not a submodule — it has its
own `pyproject.toml`, `Makefile`, `tests/`, and configs). It trains cheap
linear probes on residual-stream activations to flag strategic deception —
deliberately inducing a false belief to achieve a goal — even when the probe
was trained on off-distribution data containing no real deception.

See `docs/papers/deception-probes-notes.md` for full detail on probe training,
datasets, and the internal-consistency caveats.

This notes file has a flagged ambiguities/divergences section — check it before treating any specific number as certain.
