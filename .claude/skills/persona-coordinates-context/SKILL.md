---
name: persona-coordinates-context
description: >-
  Mahadik & Skapars' persona-PC probing method (contrastive persona vectors,
  PCA, projection probes) — the target method for this project's own
  larger-scale replication work, not yet tied to a directory. Invoke manually
  with /persona-coordinates-context until that code exists.
user-invocable: true
---

There is no directory for this yet. This skill covers Mahadik & Skapars, *Do
Linear Probes Generalize Better in Persona Coordinates?* — a white-box
monitoring method that builds low-dimensional "persona axes" for a target
behaviour (deception, sycophancy) from a small curated set of contrastive
persona prompts (activation differences, PCA), then uses those axes both as
zero-shot projection classifiers and as coordinates for supervised probes. It
is the method this project intends to replicate and extend at larger scale.

See `docs/papers/persona-coordinates-notes.md` for full detail on the persona
prompt set, axis construction, and probing results.

This notes file has a flagged ambiguities/divergences section — check it before treating any specific number as certain.
