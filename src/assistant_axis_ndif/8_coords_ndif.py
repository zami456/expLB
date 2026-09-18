#!/usr/bin/env python3
"""Stage 8: per-SAMPLE pooled activations for the evaluation corpus, on NDIF.

Stage 2 builds one vector per persona by summing over the batch. This keeps the batch
dimension instead, so every evaluation rollout gets its own pooled vector and can be
projected onto any direction afterwards -- contrast, PC1, trait, or something invented
later -- without touching the 70B again.

    <output_dir>/<dataset>/
        items.pt          cached (row_id, token_ids, a0, a1); --prepare_only writes this
        rows.jsonl        one line per row: labels, semantics, n_tok, prompt fidelity
        chunk_<seq>.pt    {"row_ids", "pooled" (n,K,H,P) fp16, "n_tok" (n,)}
        coords.pt         [--finalize] the chunks concatenated, row order asserted

WHAT LEAVES NDIF. Per-token activations never do. The mask, mean and (optionally)
last-token pooling all run inside the trace and only the pooled vector comes back, cast
to fp16 there rather than here -- a 300-token response over 80 layers is ~393 MB raw and
~1.3 MB pooled. Casting on the client would move the 4-byte version over the wire and put
a 64-row call back inside the size regime that already killed a judge run.

ROW PAIRING is the thing that fails silently, so it is enforced three ways. Batches are
grouped by token budget, which SORTS rows by length, so every item carries its row_id and
results are written back by id, never by position. One implicit invoke per trace -- several
tracer.invoke() blocks are the documented way to get results back in model order rather
than input order. And n_tok comes back from the mask itself and is asserted against the
client-known span length, which catches a drifted mask immediately.

    python src/assistant_axis_ndif/8_coords_ndif.py --datasets ai_liar --layers 40 --limit 16
    python src/assistant_axis_ndif/8_coords_ndif.py --datasets all --layers every4
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]
AA_ROOT = REPO_ROOT / "assistant_axis"

sys.path.insert(0, str(PIPELINE_DIR))
import eval_datasets as E  # noqa: E402

load_dotenv(PIPELINE_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")
os.environ.setdefault("NDIF_HOST", "https://api.ndif.us")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:  # noqa: BLE001
    pass
for _v in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
    if os.getenv(_v):
        os.environ["HF_TOKEN"] = os.environ["HF_HUB_TOKEN"] = os.environ[_v].strip()
        break

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("8_coords_ndif")

_spec = importlib.util.spec_from_file_location(
    "aa_conversation", AA_ROOT / "assistant_axis" / "internals" / "conversation.py")
_conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conv)
ConversationEncoder = _conv.ConversationEncoder


# --------------------------------------------------------------------------- #
# spans
# --------------------------------------------------------------------------- #
def build_eval_items(tok, encoder, rows, max_length: int, drops: dict):
    """(row_id, ids, a0, a1) per row. Two paths, because the sources differ.

    prompt_str is the EXACT generation-time prompt, so re-tokenising it and the
    prompt+response concatenation puts the span boundary at the divergence point. That
    matters for roleplaying, whose assistant prefix ("Student:") lives inside the prompt:
    re-templating the message list instead would pool those prefix tokens into the span on
    371 rows. CSV rows have no such string and must be re-templated, which is recorded as
    prompt_fidelity on the row.

    Over-long sequences keep their TAIL. The span sits at the end and is the only thing
    pooled, so head-truncation drops it entirely -- that is how ai_audit lost all 80 rows
    at max_length 2048.
    """
    items = []
    for r in rows:
        try:
            if r.prompt_str is not None:
                pre = tok(r.prompt_str, add_special_tokens=False).input_ids
                full = tok(r.prompt_str + r.response_str, add_special_tokens=False).input_ids
                a0 = len(pre)
                while a0 > 0 and (a0 > len(full) or full[:a0] != pre[:a0]):
                    a0 -= 1          # BPE may merge across the boundary
                a1 = len(full)
            else:
                full, spans = encoder.build_turn_spans(r.messages, return_dict=False)
                asst = [s for s in spans if s.get("role") == "assistant"]
                if not asst:
                    drops["no_assistant_span"] = drops.get("no_assistant_span", 0) + 1
                    continue
                a0, a1 = int(asst[-1]["start"]), int(asst[-1]["end"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("span build failed for %s: %s", r.row_id, exc)
            drops["span_build_failed"] = drops.get("span_build_failed", 0) + 1
            continue

        if a1 <= a0:
            drops["empty_span"] = drops.get("empty_span", 0) + 1
            continue
        if len(full) > max_length:
            cut = len(full) - max_length
            full, a0, a1 = full[cut:], a0 - cut, a1 - cut
            drops["truncated"] = drops.get("truncated", 0) + 1
        a0, a1 = max(a0, 0), min(a1, len(full))
        if a0 >= a1:
            drops["span_lost_to_truncation"] = drops.get("span_lost_to_truncation", 0) + 1
            continue
        items.append((r.row_id, full, a0, a1))
    return items


def batch_by_tokens(items, token_budget: int, max_rows: int):
    """Group so len(batch) * longest-in-batch <= token_budget. Sorted longest-first.

    Sorting is why every item carries its row_id: the batch order is NOT the input order,
    and pairing results back by position would misattribute every row silently.
    """
    ordered = sorted(items, key=lambda it: len(it[1]), reverse=True)
    out, cur, cur_max = [], [], 0
    for it in ordered:
        L = len(it[1])
        nm = max(cur_max, L)
        if cur and ((len(cur) + 1) * nm > token_budget or len(cur) >= max_rows):
            out.append(cur)
            cur, cur_max = [it], L
        else:
            cur.append(it)
            cur_max = nm
    if cur:
        out.append(cur)
    return out


def group_sessions(groups, session_tokens: int, max_batches: int):
    """Bundle batches into one NDIF job by TOTAL padded tokens, not by a fixed batch count.

    --token_budget bounds ONE batch. session_batches bounded the NUMBER of batches, so a
    session's real cost was batches x (whatever each batch happened to cost) and nothing
    bounded that product. Measured 2026-09-15: instructed-deception batches run ~2,000 padded
    tokens, so 8 of them is ~16,000 and every session completed; insider-trading batches sit
    at 11,976 -- right at the per-batch budget -- so 8 of them is ~95,800 and NDIF rejected
    every job 0.4s after dispatch, livelocking the dataset.

    This is the same bug as judge_batch 48 sending 35,040 padded tokens and returning zero
    scores overnight, one level up: bound the thing that actually costs, which is padded
    tokens, and keep the count only as a secondary cap.
    """
    out, cur, cur_tok = [], [], 0
    for g in groups:
        cost = len(g) * max(len(it[1]) for it in g)
        if cur and (cur_tok + cost > session_tokens or len(cur) >= max_batches):
            out.append(cur)
            cur, cur_tok = [g], cost
        else:
            cur.append(g)
            cur_tok += cost
    if cur:
        out.append(cur)
    return out


def attn_bytes(rows: int, seqlen: int, n_heads: int) -> int:
    """Bytes the eager attention score matrix needs: rows x heads x T x T x 4."""
    return rows * n_heads * seqlen * seqlen * 4


def batch_by_attention(items, budget_bytes: int, n_heads: int, max_rows: int):
    """Group so the attention score matrix fits, and report rows that cannot fit alone.

    The binding constraint on NDIF is NOT padded tokens. Measured 2026-09-15 against the real
    deployment, once the error message was finally visible:

        B=1  T=3000  product 3,000   attn 2.15 GiB  OK
        B=16 T=400   product 6,400   attn 0.61 GiB  OK
        B=2  T=3000  product 6,000   attn 4.30 GiB  FAIL
        B=6  T=1869  product 11,214  attn 5.36 GiB  FAIL  "Tried to allocate 5.00 GiB"

    Equal products land on opposite sides, and a SMALLER product fails while a larger one
    passes, so "rows x longest" never described the limit. "rows x heads x T^2 x 4" does: it
    is the eager attention score matrix. Llama-3.3-70B has 64 heads, and the 70B already
    occupies ~40 of the GPU's 44.42 GiB, leaving a few GiB shared with other tenants.

    T^2 is why this matters: doubling sequence length quadruples the cost, so one long row
    can cost more than sixteen short ones. A row whose own T exceeds the budget cannot run at
    ANY batch size; those come back separately instead of being dropped in silence.
    """
    ordered = sorted(items, key=lambda it: len(it[1]), reverse=True)
    too_long = [it for it in ordered if attn_bytes(1, len(it[1]), n_heads) > budget_bytes]
    fits = [it for it in ordered if attn_bytes(1, len(it[1]), n_heads) <= budget_bytes]

    out, cur, cur_max = [], [], 0
    for it in fits:
        L = len(it[1])
        nm = max(cur_max, L)
        if cur and (attn_bytes(len(cur) + 1, nm, n_heads) > budget_bytes
                    or len(cur) >= max_rows):
            out.append(cur)
            cur, cur_max = [it], L
        else:
            cur.append(it)
            cur_max = nm
    if cur:
        out.append(cur)
    return out, too_long


def token_bytes(batch, n_layers: int) -> int:
    """Bytes the packed per-token block costs for this batch: sum(span) x K x H x 2."""
    return sum(a1 - a0 for _, _, a0, a1 in batch) * n_layers * 8192 * 2


def batch_by_bytes(items, budget_bytes: int, attn_budget: int, n_heads: int,
                   max_rows: int, n_layers: int):
    """Group by RETURNED BYTES as well as attention memory. Pass C's binding constraint.

    Measured from the Pass A spans: 2,000 rows carry 304,706 response tokens, and at six
    layers one 1,532-token row is 143.6 MiB on its own while a batch of eight average rows is
    114 MiB and a session of eight such batches is 912 MiB. Pass B's batching bounds attention
    memory and row count; neither bounds what comes back, and 912 MiB downloads against a link
    that already stalls at 35 KB/s would be unworkable.

    A single row over budget is kept alone rather than dropped -- it is expensive, not
    impossible, and silently losing the longest responses would bias exactly the rows most
    likely to carry localised deception.
    """
    groups, tl = batch_by_attention(items, attn_budget, n_heads, max_rows)
    out = []
    for g in groups:
        cur = []
        for it in g:
            trial = cur + [it]
            if cur and token_bytes(trial, n_layers) > budget_bytes:
                out.append(cur)
                cur = [it]
            else:
                cur = trial
        if cur:
            out.append(cur)
    return out, tl


def span_bounds(a0: int, a1: int, span_mode: str):
    """Which token range the mask covers. 'response' = [a0, a1); 'prompt' = [0, a0).

    The prompt mode exists for the construct-validity control: if a probe trained on
    PROMPT-side activations transfers as well as one trained on response tokens, the matrix
    is largely detecting the instructed condition rather than anything the model wrote. The
    remote body is untouched -- it pools whatever mask it is handed, so 'last' becomes the
    last prompt token (a0-1) by the same argmax.
    """
    return (0, a0) if span_mode == "prompt" else (a0, a1)


def pad_batch(batch, pad_id: int, want_tokens: bool = False, span_mode: str = "response"):
    """-> input_ids (B,T), attention_mask (B,T), span (B,T) float32 1.0 on the masked tokens.

    With want_tokens, also returns (tok_index, offsets): tok_index holds b*T + t for every
    masked position, so a single gather inside the trace returns those tokens packed
    end to end; offsets says where each row starts and ends in that packed block.
    """
    T = max(len(ids) for _, ids, _, _ in batch)
    B = len(batch)
    ids_t = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    span = torch.zeros((B, T), dtype=torch.float32)
    for i, (_, ids, a0, a1) in enumerate(batch):
        s0, s1 = span_bounds(a0, a1, span_mode)
        ids_t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1
        span[i, s0:s1] = 1.0
    if not want_tokens:
        return ids_t, attn, span
    idx, offs = [], [0]
    for i, (_, _, a0, a1) in enumerate(batch):
        s0, s1 = span_bounds(a0, a1, span_mode)
        idx.extend(range(i * T + s0, i * T + s1))
        offs.append(len(idx))
    return ids_t, attn, span, torch.tensor(idx, dtype=torch.long), offs


# --------------------------------------------------------------------------- #
# the remote body
# --------------------------------------------------------------------------- #
def _pool(llm, layer_idx, span, want_last: bool, tok_index=None):
    """Runs on NDIF. -> pooled (B,K,H,P) fp16, n_tok (B,) fp32, tokens (S,K,H) fp16 or None.

    `tok_index` is a flat gather index built on the client: for each row b and each response
    position t in [a0_b, a1_b), the value b*T + t. Gathering with it inside the trace returns
    exactly the response tokens, packed end to end, and nothing else. Returning the masked
    (B,T,H) instead would ship every context token too -- for insider-trading that is ~1,500
    positions to keep ~100, a 15x waste on the one thing that dominates this pass.

    Per-sample version of 2_activations_ndif._span_mean_sum: same masked mean over the
    response tokens, but the batch dimension is KEPT rather than summed.

    One layer at a time, on that layer's own device: the 70B is sharded, so stacking raw
    .output across layers raises "Expected all tensors to be on the same device", and the
    stacked (K,B,T,H) intermediate would be tens of GB.
    """
    per_layer = []
    per_layer_tokens = [] if tok_index is not None else None
    ntok = None
    for L in layer_idx:                                  # ASCENDING or OutOfOrderError
        h = llm.model.layers[L].output                   # (B,T,H) plain tensor in tf5
        # The mask stays fp32 and every reduction names dtype=float32. Casting it to the
        # model's bfloat16 corrupts anything past 256 tokens, because bf16 carries 8 mantissa
        # bits: integers step by 2 above 256 and by 4 above 512, so a 971-token span summed
        # in bf16 comes back 972, and 1195 comes back 1192. That fired the n_tok assertion on
        # mask, alpaca, ai_liar, convincing-game and sandbagging_v2 -- every dataset whose
        # responses run past 256 tokens -- while the short-span datasets passed and hid it.
        m = span.to(h.device, torch.float32)
        num = (h * m[:, :, None].to(h.dtype)).sum(dim=1, dtype=torch.float32)
        den = m.sum(dim=1, dtype=torch.float32).clamp(min=1.0)[:, None]
        outs = [num / den]
        if want_last:
            # fp32 here too, and this one was the dangerous case: it does not raise. With
            # positions in bf16, distinct indices past 256 collapse onto the same value, so
            # argmax silently returns the wrong "last" token for any sequence longer than
            # that -- most of insider-trading, whose contexts run to 1,495.
            pos = torch.arange(m.shape[1], device=m.device, dtype=torch.float32)
            last = (m * pos[None, :]).argmax(dim=1)      # last response token per row
            rows = torch.arange(h.shape[0], device=h.device)
            outs.append(h[rows, last].to(torch.float32))
        # fp16 HERE, not on the client: the cast is what halves the bytes on the wire
        per_layer.append(torch.stack(outs, dim=-1).to(torch.float16).cpu())
        if per_layer_tokens is not None:
            flat = h.reshape(-1, h.shape[-1])            # (B*T, H)
            per_layer_tokens.append(flat[tok_index.to(h.device)].to(torch.float16).cpu())
        if ntok is None:
            ntok = m.sum(dim=1, dtype=torch.float32).cpu()
    tokens = torch.stack(per_layer_tokens, dim=1) if per_layer_tokens is not None else None
    return torch.stack(per_layer, dim=1), ntok, tokens   # (B,K,H,P), (B,), (S,K,H)


class SlowDownload(RuntimeError):
    """An NDIF result download collapsed and is not worth waiting for."""


def install_download_deadline(min_kbps: float, grace: float, attempts: int = 4) -> bool:
    """Abort a collapsed NDIF result download so the session retries on a fresh connection.

    Measured on the lab PC 2026-09-15: result downloads from prod-ndif-results.s3 are
    bimodal. A healthy 34 MB session arrives in 3-35s; a sick one trickles at ~35 KB/s for
    eleven minutes while the worker sits at 0% CPU and NDIF reports RUNNING=15s. It is not
    bandwidth (91 MB/s to Cloudflare on the same host at the same moment), not the socket
    pins, and not client reuse -- nnsight opens a fresh httpx.Client per download. Killing
    the process always cleared it, which is the tell: the flow is sick, so a new connection
    fixes it and waiting does not.

    httpx's read timeout cannot catch this, because bytes keep arriving -- just very few of
    them. So the check is on throughput, after a grace period so a slow start is not punished.

    A violation retries the GET here, on a new client, against the same presigned URL --
    NOT by raising straight away. The result is already sitting in S3, so re-dispatching the
    whole session to NDIF would pay ~15s of recompute to fetch a file that already exists.
    Only when every attempt draws a sick flow does this raise, and then run_session's retry
    takes over as before. Either way it beats the six minutes the supervisor's stall killer
    needs to notice and restart the process.

    Returns False if nnsight's internals have moved, in which case the run continues
    unpatched and the supervisor's stall killer is still the backstop.
    """
    try:
        from nnsight.intervention.backends.remote import RemoteBackend
    except Exception:  # noqa: BLE001
        return False
    if not (hasattr(RemoteBackend, "get_result")
            and hasattr(RemoteBackend, "_decompress_and_load")):
        return False

    import httpx

    def get_result(self, url: str, content_length=None):
        timeout = httpx.Timeout(self.CONNECT_TIMEOUT, read=self.READ_TIMEOUT)
        last = None
        for attempt in range(1, attempts + 1):
            result_bytes = io.BytesIO()
            t0 = time.time()
            got = 0
            try:
                with httpx.Client(timeout=timeout) as client:
                    with client.stream("GET", url) as stream:
                        total = content_length or float(stream.headers.get("Content-length", 0))
                        with tqdm(total=total or None, unit="B", unit_scale=True,
                                  desc="downloading", leave=False) as bar:
                            for data in stream.iter_bytes(chunk_size=128 * 1024):
                                got += len(data)
                                bar.update(len(data))
                                result_bytes.write(data)
                                elapsed = time.time() - t0
                                if elapsed > grace and (got / 1024.0 / elapsed) < min_kbps:
                                    raise SlowDownload(
                                        "%.1f MB in %.0fs = %.0f KB/s (floor %.0f)"
                                        % (got / 1e6, elapsed, got / 1024.0 / elapsed, min_kbps))
                return self._decompress_and_load(result_bytes)
            except SlowDownload as exc:
                last = exc
                result_bytes.close()
                logger.warning("  download attempt %d/%d: %s -- new connection",
                               attempt, attempts, exc)
        raise SlowDownload("%s after %d attempts" % (last, attempts))

    RemoteBackend.get_result = get_result
    return True


def run_session(llm, nnsight, layer_idx, prepared, want_last: bool, retries: int,
                per_token: bool = False):
    """One NDIF job over several batches. -> list of (pooled, ntok, tokens) in order."""
    delay = 3.0
    for attempt in range(retries):
        try:
            with llm.session(remote=True):
                pooled = nnsight.save([])
                ntoks = nnsight.save([])
                toks = nnsight.save([])
                for _pre in prepared:
                    ids, attn, span = _pre[0], _pre[1], _pre[2]
                    tok_index = _pre[3] if per_token else None
                    # ONE implicit invoke per trace. Several tracer.invoke() blocks are the
                    # documented way to get results back in model order, not input order.
                    with llm.trace({"input_ids": ids, "attention_mask": attn}) as tracer:
                        p, n, tk = _pool(llm, layer_idx, span, want_last, tok_index)
                        pooled.append(p)
                        ntoks.append(n)
                        if per_token:
                            toks.append(tk)
                        # Bail before the LM head. Without this the run continues through the
                        # remaining layers, the final norm and lm_head, which materialises
                        # logits of (batch, seq, 128256) -- for a 10,015-token batch that is
                        # ~5 GB in fp32, and NDIF returned exactly that:
                        #   "OutOfMemoryError: CUDA out of memory. Tried to allocate 4.78 GiB"
                        # on a 44.42 GiB GPU already holding the 70B. We never read logits;
                        # we read layer outputs. Stopping at the last layer we actually need
                        # removes the largest allocation in the pass and skips layers 77-79
                        # as a bonus. The layer we stop at still runs, so _pool has its value.
                        tracer.stop()
            P = getattr(pooled, "value", pooled)
            N = getattr(ntoks, "value", ntoks)
            K = getattr(toks, "value", toks) if per_token else [None] * len(P)
            if len(P) != len(prepared):
                raise RuntimeError(f"session returned {len(P)} of {len(prepared)} batches")
            if per_token and len(K) != len(prepared):
                raise RuntimeError(f"session returned {len(K)} token blocks of "
                                   f"{len(prepared)}")
            return list(zip(P, N, K))
        except Exception as exc:  # noqa: BLE001
            # Log the LAST line, not the first 300 characters. nnsight wraps the remote
            # failure in a client-side traceback, so str(exc)[:300] spent its whole budget on
            # "File remote.py, line 567, in __call__ ..." and cut off immediately before the
            # RemoteException description -- the only part that says what actually went
            # wrong. 160 sessions failed and not one log line said why.
            _lines = [ln for ln in str(exc).strip().splitlines() if ln.strip()]
            _tail = _lines[-1].strip() if _lines else repr(exc)
            _cost = sum(int(t[0].shape[0]) * int(t[0].shape[1]) for t in prepared)
            logger.warning("  session attempt %d/%d (%d batches, %d padded tokens) failed: "
                           "%s: %s", attempt + 1, retries, len(prepared), _cost,
                           type(exc).__name__, _tail[:400])
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None


# --------------------------------------------------------------------------- #
def resolve_layers(spec, n_layers: int):
    if len(spec) == 1 and str(spec[0]).lower() == "all":
        return list(range(n_layers))
    if len(spec) == 1 and str(spec[0]).lower() == "every4":
        return list(range(0, n_layers, 4))
    idx = sorted({int(x) for x in spec})
    bad = [i for i in idx if not 0 <= i < n_layers]
    if bad:
        raise ValueError(f"--layers out of range for {n_layers} layers: {bad}")
    return idx


def subsample(rows, n: int, seed: int):
    """Label-stratified subsample, deterministic under `seed`.

    --limit takes the head, which is a sample only if the file happens to be shuffled.
    Measured: instructed-deception is label-sorted, so --limit 200 returns 0 deceptive
    and 200 honest; harm-pressure-choice's first 200 rows swallow all 134 of its
    deceptive rows and nothing else. The layer sweep chooses a layer from these rows, so
    a skewed subsample chooses the layer on a skewed corpus.

    Round-robin over label buckets, so the result is as balanced as the data allows and
    degrades gracefully when a class is smaller than n/2. row_id order is restored at the
    end to keep chunk resume and items.pt stable across runs.
    """
    if n <= 0 or n >= len(rows):
        return rows
    buckets: dict = {}
    for r in rows:
        buckets.setdefault(r.label, []).append(r)
    rng = random.Random(seed)
    for v in buckets.values():
        rng.shuffle(v)
    keys = sorted(k for k in buckets if k is not None)
    out: list = []
    while len(out) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(out) < n:
                out.append(buckets[k].pop())
    # Ambiguous rows only backfill. Round-robining them alongside the labelled classes
    # spent 66 of roleplaying's 200 slots on rows every probe drops, while 87 usable
    # labelled rows sat unsampled.
    if len(out) < n:
        out.extend(buckets.get(None, [])[: n - len(out)])
    return sorted(out, key=lambda r: r.row_id)


def process_dataset(llm, nnsight, tok, encoder, key: str, layer_idx, args, pad_id,
                    n_heads: int):
    out_dir = Path(args.output_dir) / key
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.pt"

    rows = E.load_dataset(key)
    if args.sample:
        rows = subsample(rows, args.sample, args.seed)
    elif args.limit:
        rows = rows[:args.limit]
    by_id = {r.row_id: r for r in rows}

    if items_path.exists() and not args.overwrite:
        items = torch.load(items_path, weights_only=False)
        logger.info("%s: %d cached spans", key, len(items))
    else:
        drops: dict = {}
        items = build_eval_items(tok, encoder, rows, args.max_length, drops)
        torch.save(items, items_path)
        logger.info("%s: %d rows -> %d spans %s", key, len(rows), len(items),
                    ("(dropped: " + ", ".join(f"{k}={v}" for k, v in sorted(drops.items())) + ")")
                    if drops else "")
    if args.prepare_only or not items:
        return

    done: set[str] = set()
    for c in sorted(out_dir.glob("chunk_*.pt")):
        try:
            done |= set(torch.load(c, weights_only=False)["row_ids"])
        except Exception:  # noqa: BLE001
            logger.warning("  unreadable chunk %s ignored", c.name)
    todo = [it for it in items if it[0] not in done]
    if not todo:
        logger.info("%s: already complete (%d rows)", key, len(done))
        return
    logger.info("%s: %d to do (%d already on disk)", key, len(todo), len(done))

    budget_bytes = int(args.attn_budget_gib * (1024 ** 3))
    if args.per_token:
        groups, too_long = batch_by_bytes(todo, int(args.byte_budget_mib * 2**20),
                                          budget_bytes, n_heads, args.batch_size,
                                          len(layer_idx))
    else:
        groups, too_long = batch_by_attention(todo, budget_bytes, n_heads, args.batch_size)
    if too_long:
        worst = max(len(it[1]) for it in too_long)
        cap = int((budget_bytes / (n_heads * 4)) ** 0.5)
        logger.error("  %s: %d of %d rows CANNOT run at any batch size -- longest is %d "
                     "tokens and the attention budget caps a single row at ~%d tokens. "
                     "Raise --attn_budget_gib only if the GPU has room, or lower "
                     "--max_length to tail-truncate them (which changes what they mean).",
                     key, len(too_long), len(todo), worst, cap)
    if not groups:
        logger.error("  %s: no runnable batches; skipping", key)
        return
    widest = max(attn_bytes(len(b), max(len(i[1]) for i in b), n_heads) for b in groups)
    logger.info("  %d batches, widest attention %.2f GiB (budget %.2f GiB)",
                len(groups), widest / (1024 ** 3), args.attn_budget_gib)

    seq = len(list(out_dir.glob("chunk_*.pt")))
    want_last = "last" in args.pooling
    chunks = group_sessions(groups, args.session_tokens, args.session_batches)
    if chunks:
        _cost = [sum(len(g) * max(len(it[1]) for it in g) for g in c) for c in chunks]
        logger.info("  %d sessions, widest %d padded tokens (session budget %d)",
                    len(chunks), max(_cost), args.session_tokens)
    for chunk in tqdm(chunks, desc=f"  {key}", leave=False):
        prepared = [pad_batch(b, pad_id, args.per_token, args.span) for b in chunk]
        got = run_session(llm, nnsight, layer_idx, prepared, want_last,
                          args.retries, args.per_token)
        if got is None:
            logger.error("  %s: a session failed permanently; leaving its rows for resume", key)
            continue
        ids_out, pooled_out, ntok_out = [], [], []
        tok_out, tok_offs = [], [0]
        for batch, (p, n, tk) in zip(chunk, got):
            b_offs = pad_batch(batch, pad_id, True, args.span)[4] if args.per_token else None
            for i, (rid, _, a0, a1) in enumerate(batch):
                s0, s1 = span_bounds(a0, a1, args.span)
                expect = s1 - s0
                got_n = int(round(float(n[i])))
                if got_n != expect:
                    raise RuntimeError(
                        f"span mismatch on {rid}: n_tok={got_n} but the {args.span} span is "
                        f"{expect}. The mask and the client disagree; do not trust these vectors.")
                ids_out.append(rid)
                pooled_out.append(p[i])
                ntok_out.append(n[i])
                if args.per_token:
                    seg = tk[b_offs[i]:b_offs[i + 1]]          # (n_tok_i, K, H)
                    if seg.shape[0] != expect:
                        raise RuntimeError(
                            f"per-token length mismatch on {rid}: packed {seg.shape[0]} "
                            f"but span is {expect}. Do not trust these activations.")
                    tok_out.append(seg)
                    tok_offs.append(tok_offs[-1] + expect)
        path = out_dir / f"chunk_{seq:04d}.pt"
        tmp = path.with_suffix(".tmp")
        blob = {"row_ids": ids_out,
                "pooled": torch.stack(pooled_out),
                "n_tok": torch.stack(ntok_out),
                "layers": list(layer_idx),
                "pooling": list(args.pooling),
                "span": args.span}
        if args.per_token and tok_out:
            # packed (S,K,H) plus offsets: row i is tokens[offsets[i]:offsets[i+1]].
            # Padding to a rectangle would multiply the largest pass in the project by the
            # ratio of longest to mean response, which on this corpus is 1532/152 = 10x.
            blob["tokens"] = torch.cat(tok_out)
            blob["tok_offsets"] = tok_offs
        torch.save(blob, tmp)
        os.replace(tmp, path)
        seq += 1

    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for rid, _, a0, a1 in items:
            r = by_id[rid]
            fh.write(json.dumps({
                "row_id": rid, "dataset": r.dataset, "item_id": r.item_id,
                "sample_idx": r.sample_idx, "label": r.label, "label_raw": str(r.label_raw),
                "label_semantics": r.label_semantics, "prompt_fidelity": r.prompt_fidelity,
                "span_len": span_bounds(a0, a1, args.span)[1] - span_bounds(a0, a1, args.span)[0],
                "response_len": a1 - a0, "prompt_len": a0, "span": args.span,
            }, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--output_dir", default=str(REPO_ROOT / "run-coords"))
    ap.add_argument("--layers", nargs="+", default=["every4"],
                    help="all | every4 | explicit ints")
    ap.add_argument("--pooling", nargs="+", default=["mean", "last"])
    ap.add_argument("--span", default="response", choices=["response", "prompt"],
                    help="which tokens the mask covers. 'response' is the assistant turn "
                         "[a0, a1); 'prompt' is everything before it, [0, a0), and makes "
                         "'last' the last PROMPT token -- the construct-validity control. "
                         "Write a prompt-span store to its own --out directory.")
    ap.add_argument("--per_token", action="store_true",
                    help="also return the response tokens unpooled, packed (S,K,H) fp16 "
                         "with per-row offsets. This is Pass C and it is ~75x the pooled "
                         "size, so it is off by default")
    ap.add_argument("--byte_budget_mib", type=float, default=64.0,
                    help="max per-token bytes returned per call. With --per_token this is "
                         "the BINDING constraint, not attention: one 1,532-token row at 6 "
                         "layers is 143 MiB and a naive 8-row batch is 114 MiB")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N rows per dataset (0 = all). Head slice -- NOT a sample; "
                         "several datasets are label-sorted. Use --sample instead.")
    ap.add_argument("--sample", type=int, default=0,
                    help="label-stratified N rows per dataset (0 = all); overrides --limit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="max ROWS per call; the real bound is --token_budget")
    ap.add_argument("--attn_budget_gib", type=float, default=2.5,
                    help="max eager-attention score matrix per call: rows x heads x T^2 x 4 "
                         "bytes, on a 44.42 GiB GPU already holding the 70B and shared "
                         "with other tenants. Measured: 2.15 and 3.81 GiB complete, 4.30 "
                         "and 5.00 GiB OOM. 2.5 keeps margin under 3.81 because the free "
                         "headroom moves with whoever else is on the GPU -- it has been "
                         "seen as low as 113 MiB")
    ap.add_argument("--token_budget", type=int, default=12000,
                    help="legacy padded-token bound, kept for reference only; the real "
                         "constraint is --attn_budget_gib")
    ap.add_argument("--session_batches", type=int, default=8,
                    help="max batches per NDIF job; the real bound is --session_tokens")
    ap.add_argument("--session_tokens", type=int, default=16000,
                    help="max TOTAL padded tokens per NDIF job. instructed-deception "
                         "ran at ~12,200 per session and always completed; "
                         "insider-trading at 8 batches is ~95,800 and NDIF rejects it "
                         "0.4s after dispatch. 16,000 keeps short-context datasets "
                         "unchanged and gives long ones one batch per session")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--min_download_kbps", type=float, default=300.0,
                    help="abort a result download slower than this, and retry on a "
                         "fresh connection (0 disables)")
    ap.add_argument("--download_attempts", type=int, default=4,
                    help="re-GET the same result URL this many times on a fresh "
                         "connection before giving up and re-running the session")
    ap.add_argument("--download_grace", type=float, default=12.0,
                    help="seconds before the download throughput floor starts applying")
    ap.add_argument("--prepare_only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    keys = E.all_keys() if args.datasets == ["all"] else args.datasets

    # stage 2's init_ndif, reused rather than copied
    spec = importlib.util.spec_from_file_location(
        "act2", PIPELINE_DIR / "2_activations_ndif.py")
    act2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(act2)

    if args.prepare_only:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, token=os.getenv("HF_TOKEN"))
        nnsight = llm = None
        n_layers = 80
        n_heads = 64
    else:
        nnsight, llm, n_layers = act2.init_ndif(args.model)
        tok = llm.tokenizer
        n_heads = int(llm.config.num_attention_heads)
        if args.min_download_kbps > 0:
            ok = install_download_deadline(args.min_download_kbps, args.download_grace,
                                           args.download_attempts)
            logger.info("download floor %.0f KB/s after %.0fs: %s",
                        args.min_download_kbps, args.download_grace,
                        "installed" if ok else "NOT installed (nnsight internals moved)")

    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    encoder = ConversationEncoder(tok)
    layer_idx = resolve_layers(args.layers, n_layers)
    logger.info("layers: %d of %d  pooling: %s  datasets: %s",
                len(layer_idx), n_layers, ",".join(args.pooling), ", ".join(keys))

    for key in keys:
        try:
            process_dataset(llm, nnsight, tok, encoder, key, layer_idx, args,
                            tok.pad_token_id, n_heads)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s FAILED: %s: %s", key, type(exc).__name__, exc)
    logger.info("Done.")


if __name__ == "__main__":
    main()
