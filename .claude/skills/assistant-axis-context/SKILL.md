---
name: assistant-axis-context
description: >-
  The Assistant Axis persona-direction method. Use when working in
  assistant_axis/.
paths: assistant_axis/**
---

`assistant_axis/` is the vendored `assistant-axis` submodule. It builds a
low-dimensional "persona space" from activation vectors for hundreds of
character archetypes (PCA over those vectors), identifies the **Assistant Axis**
as its main axis of variation, uses it to measure persona drift over multi-turn
conversations, and applies conditional "activation capping" to curb harmful
behavior from drift without degrading capabilities.

See `docs/papers/assistant-axis-notes.md` for full detail on the persona-space
construction, steering, and capping intervention.

This notes file has a flagged ambiguities/divergences section — check it before treating any specific number as certain.
