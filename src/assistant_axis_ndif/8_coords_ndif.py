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


def pad_batch(batch, pad_id: int):
    """-> input_ids (B,T), attention_mask (B,T), span (B,T) float32 1.0 on response tokens."""
    T = max(len(ids) for _, ids, _, _ in batch)
    B = len(batch)
    ids_t = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    span = torch.zeros((B, T), dtype=torch.float32)
    for i, (_, ids, a0, a1) in enumerate(batch):
        ids_t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1
        span[i, a0:a1] = 1.0
    return ids_t, attn, span


# --------------------------------------------------------------------------- #
# the remote body
# --------------------------------------------------------------------------- #
def _pool(llm, layer_idx, span, want_last: bool):
    """Runs on NDIF. -> pooled (B, K, H, P) fp16 CPU, n_tok (B,) fp32 CPU.

    Per-sample version of 2_activations_ndif._span_mean_sum: same masked mean over the
    response tokens, but the batch dimension is KEPT rather than summed.

    One layer at a time, on that layer's own device: the 70B is sharded, so stacking raw
    .output across layers raises "Expected all tensors to be on the same device", and the
    stacked (K,B,T,H) intermediate would be tens of GB.
    """
    per_layer = []
    ntok = None
    for L in layer_idx:                                  # ASCENDING or OutOfOrderError
        h = llm.model.layers[L].output                   # (B,T,H) plain tensor in tf5
        m = span.to(h.device, h.dtype)                   # client tensor -> live device
        num = (h * m[:, :, None]).sum(dim=1)             # (B,H)
        den = m.sum(dim=1).clamp(min=1.0)[:, None]
        outs = [(num / den).to(torch.float32)]           # fp32 BEFORE any dot product
        if want_last:
            pos = torch.arange(m.shape[1], device=m.device, dtype=m.dtype)
            last = (m * pos[None, :]).argmax(dim=1)      # last response token per row
            rows = torch.arange(h.shape[0], device=h.device)
            outs.append(h[rows, last].to(torch.float32))
        # fp16 HERE, not on the client: the cast is what halves the bytes on the wire
        per_layer.append(torch.stack(outs, dim=-1).to(torch.float16).cpu())
        if ntok is None:
            ntok = m.sum(dim=1).to(torch.float32).cpu()
    return torch.stack(per_layer, dim=1), ntok           # (B,K,H,P)


class SlowDownload(RuntimeError):
    """An NDIF result download collapsed and is not worth waiting for."""


def install_download_deadline(min_kbps: float, grace: float) -> bool:
    """Abort a collapsed NDIF result download so the session retries on a fresh connection.

    Measured on the lab PC 2026-09-15: result downloads from prod-ndif-results.s3 are
    bimodal. A healthy 34 MB session arrives in 3-35s; a sick one trickles at ~35 KB/s for
    eleven minutes while the worker sits at 0% CPU and NDIF reports RUNNING=15s. It is not
    bandwidth (91 MB/s to Cloudflare on the same host at the same moment), not the socket
    pins, and not client reuse -- nnsight opens a fresh httpx.Client per download. Killing
    the process always cleared it, which is the tell: the flow is sick, so a new connection
    fixes it and waiting does not.

    httpx's read timeout cannot catch this, because bytes keep arriving -- just very few of
    them. So the check is on throughput, after a grace period so a slow start is not
    punished. Raising here lands in run_session's retry, which re-dispatches the session and
    calls get_result again on a brand-new client. That costs one ~15s recompute instead of
    the six minutes the supervisor's stall killer needs to notice and restart the process.

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
        result_bytes = io.BytesIO()
        timeout = httpx.Timeout(self.CONNECT_TIMEOUT, read=self.READ_TIMEOUT)
        t0 = time.time()
        got = 0
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
                        if elapsed > grace:
                            kbps = got / 1024.0 / elapsed
                            if kbps < min_kbps:
                                raise SlowDownload(
                                    "%.1f MB in %.0fs = %.0f KB/s (floor %.0f); "
                                    "dropping this flow and retrying"
                                    % (got / 1e6, elapsed, kbps, min_kbps))
        return self._decompress_and_load(result_bytes)

    RemoteBackend.get_result = get_result
    return True


def run_session(llm, nnsight, layer_idx, prepared, want_last: bool, retries: int):
    """One NDIF job over several batches. -> list of (pooled, ntok) in `prepared` order."""
    delay = 3.0
    for attempt in range(retries):
        try:
            with llm.session(remote=True):
                pooled = nnsight.save([])
                ntoks = nnsight.save([])
                for ids, attn, span in prepared:
                    # ONE implicit invoke per trace. Several tracer.invoke() blocks are the
                    # documented way to get results back in model order, not input order.
                    with llm.trace({"input_ids": ids, "attention_mask": attn}):
                        p, n = _pool(llm, layer_idx, span, want_last)
                        pooled.append(p)
                        ntoks.append(n)
            P = getattr(pooled, "value", pooled)
            N = getattr(ntoks, "value", ntoks)
            if len(P) != len(prepared):
                raise RuntimeError(f"session returned {len(P)} of {len(prepared)} batches")
            return list(zip(P, N))
        except Exception as exc:  # noqa: BLE001
            logger.warning("  session attempt %d/%d (%d batches) failed: %s: %s",
                           attempt + 1, retries, len(prepared), type(exc).__name__,
                           str(exc)[:300])
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


def process_dataset(llm, nnsight, tok, encoder, key: str, layer_idx, args, pad_id):
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

    groups = batch_by_tokens(todo, args.token_budget, args.batch_size)
    widest = max(len(b) * max(len(i[1]) for i in b) for b in groups)
    logger.info("  %d batches, widest %d padded tokens (budget %d)",
                len(groups), widest, args.token_budget)

    seq = len(list(out_dir.glob("chunk_*.pt")))
    want_last = "last" in args.pooling
    chunks = [groups[i:i + args.session_batches]
              for i in range(0, len(groups), args.session_batches)]
    for chunk in tqdm(chunks, desc=f"  {key}", leave=False):
        prepared = [pad_batch(b, pad_id) for b in chunk]
        got = run_session(llm, nnsight, layer_idx, prepared, want_last, args.retries)
        if got is None:
            logger.error("  %s: a session failed permanently; leaving its rows for resume", key)
            continue
        ids_out, pooled_out, ntok_out = [], [], []
        for batch, (p, n) in zip(chunk, got):
            for i, (rid, _, a0, a1) in enumerate(batch):
                expect = a1 - a0
                got_n = int(round(float(n[i])))
                if got_n != expect:
                    raise RuntimeError(
                        f"span mismatch on {rid}: n_tok={got_n} but a1-a0={expect}. "
                        "The mask and the client disagree; do not trust these vectors.")
                ids_out.append(rid)
                pooled_out.append(p[i])
                ntok_out.append(n[i])
        path = out_dir / f"chunk_{seq:04d}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save({"row_ids": ids_out,
                    "pooled": torch.stack(pooled_out),
                    "n_tok": torch.stack(ntok_out),
                    "layers": list(layer_idx),
                    "pooling": list(args.pooling)}, tmp)
        os.replace(tmp, path)
        seq += 1

    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for rid, _, a0, a1 in items:
            r = by_id[rid]
            fh.write(json.dumps({
                "row_id": rid, "dataset": r.dataset, "item_id": r.item_id,
                "sample_idx": r.sample_idx, "label": r.label, "label_raw": str(r.label_raw),
                "label_semantics": r.label_semantics, "prompt_fidelity": r.prompt_fidelity,
                "span_len": a1 - a0,
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
    ap.add_argument("--limit", type=int, default=0,
                    help="first N rows per dataset (0 = all). Head slice -- NOT a sample; "
                         "several datasets are label-sorted. Use --sample instead.")
    ap.add_argument("--sample", type=int, default=0,
                    help="label-stratified N rows per dataset (0 = all); overrides --limit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="max ROWS per call; the real bound is --token_budget")
    ap.add_argument("--token_budget", type=int, default=12000)
    ap.add_argument("--session_batches", type=int, default=8)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--min_download_kbps", type=float, default=300.0,
                    help="abort a result download slower than this, and retry on a "
                         "fresh connection (0 disables)")
    ap.add_argument("--download_grace", type=float, default=20.0,
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
    else:
        nnsight, llm, n_layers = act2.init_ndif(args.model)
        tok = llm.tokenizer
        if args.min_download_kbps > 0:
            ok = install_download_deadline(args.min_download_kbps, args.download_grace)
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
            process_dataset(llm, nnsight, tok, encoder, key, layer_idx, args, tok.pad_token_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s FAILED: %s: %s", key, type(exc).__name__, exc)
    logger.info("Done.")


if __name__ == "__main__":
    main()
