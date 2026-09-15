---
name: nnsight-ndif-context
description: >-
  Writing or debugging nnsight code that runs remotely on NDIF. Use when
  touching src/assistant_axis_ndif/ or anything with trace/session/generate,
  and before diagnosing a remote failure.
paths: src/assistant_axis_ndif/**
---

`docs/` vendors the full nnsight 0.7 documentation. Route there first:
`docs/usage/trace.md`, `docs/usage/session.md`, `docs/usage/save.md`,
`docs/usage/stop-and-early-exit.md`, `docs/remote/remote-session.md`,
`docs/patterns/remote-dataset-sweep.md`, `docs/errors/index.md`,
`docs/gotchas/index.md`. The root `CLAUDE.md` carries the task-to-page index and
the inline gotcha cheat-sheet — read that before writing trace code.

## What the upstream docs do not tell you, measured on this project

These cost hours each on 2026-09-15. They are properties of *this* deployment
(`meta-llama/Llama-3.3-70B-Instruct` on `api.ndif.us`), not of nnsight.

**The per-call limit is attention memory, not tokens.** The 70B occupies ~40 GiB
of a 44.42 GiB GPU shared with other tenants. What OOMs is the eager attention
score matrix, `rows x heads x T^2 x 4` bytes, 64 heads. Measured: `B=16,T=400`
(product 6,400, 0.61 GiB) completes while `B=2,T=3000` (product 6,000, 4.30 GiB)
fails — equal products, opposite outcomes. Because the cost goes as **T²**, one
long row can cost more than sixteen short ones, and no token budget can express
that. Bound batches by attention bytes (`batch_by_attention` in
`8_coords_ndif.py`). Free headroom moves with other tenants: 3.81 GiB completed
once and failed later on a busier node.

**Sequences past ~4,000 tokens cannot run at any batch size**, because batch 1 is
the floor. Nothing client-side changes this: the request carries only a model-key
string plus the serialised trace, and passing `attn_implementation="sdpa"` alters
only the local meta-model. Reading one layer instead of 21 does not help, and
neither does `tracer.stop()` — the allocation happens inside layer 0's attention,
before any stop point. Verified directly.

**Never reduce a mask or an index in the model's dtype.** Casting a span mask to
bfloat16 and summing it silently corrupts anything past 256 tokens: bf16 steps by
2 above 256 and by 4 above 512, so a 971-token span sums to 972 and 1195 to 1192.
Worse, `torch.arange(T, dtype=m.dtype)` for a last-token `argmax` collapses
positions — 0..1496 holds only 572 distinct bf16 values — and that one **does not
raise**. Keep masks fp32 and name `dtype=torch.float32` on every reduction. Assert
derived counts against what the client expects; that assertion is the only reason
this was ever found.

**A trace body defined in an importlib-loaded module is rejected**, with
`RemoteException: Module importlib is not whitelisted`. nnsight serialises the
body from its source module. Import helpers like `init_ndif` that way if you must,
but define anything appearing inside `trace`/`generate`/`session` in the file you
actually run. This invalidated a whole probe run before it was noticed.

**Result downloads stall.** The same 34 MB session returned in 3s once and took
11m45s the next time at ~35 KB/s, with the worker at 0% CPU and NDIF reporting
`RUNNING=15s` — one sick TCP flow, not bandwidth. `get_result` opens a fresh
`httpx.Client` per download, so re-GETting the same presigned URL on a new client
clears it; httpx's read timeout never fires because bytes keep arriving, just very
few. Enforce a throughput floor (`install_download_deadline`).

**Log an exception's LAST line, not `str(exc)[:300]`.** nnsight wraps the remote
failure in a client-side traceback, so 300 characters buy only
`File remote.py, line 567, in __call__` and stop immediately before the
description. 160 sessions failed here and not one log line said why.

**Bound a session, not just a batch.** `session_batches` counts batches; the cost
is batches x per-batch cost. Eight batches of a long-context dataset was ~95,800
padded tokens and NDIF rejected every job 0.4s after dispatch.

Socket pins are load-bearing: engineio 4.13.0, socketio 5.16.4, websocket-client
1.8.0, nnsight 0.7.0. Do not install into a venv while it is running a job.
