#!/usr/bin/env python3
"""
Step 2 + 4 (remote variant): extract per-role persona vectors on NDIF via nnsight.

Replaces ``2_activations.py`` + ``4_vectors.py`` for no-local-GPU setups. For each
role it teacher-forces a forward pass over every *kept* ``[system,user,assistant]``
conversation on NDIF, takes the mean residual-stream activation over the assistant
response tokens, averages those per-response vectors, and writes one tensor per
role -- the same dict shape ``5_axis.py`` reads:

    <output_dir>/<role>.pt  ==  {"vector": Tensor(K, hidden), "layers": [...],
                                 "type": ..., "role": ..., "n": <responses averaged>}

K = len(--layers). Default ``--layers all`` keeps every layer (80 for
Llama-3.3-70B) -> a 2.6 MB fp32 vector per role, which is what the per-layer axis
and activation capping need. ``--layers target`` keeps only the middle layer
(40 of 80) -> 32 KB/role. The saved artifact records the layer indices under
"layers", so axis[40] means layer 40 whatever was requested.

"Kept" = judge score == 3 (``--keep_scores``), except ``default`` where all
responses are used. Roles with < ``--min_count`` kept responses are skipped.

DOWNLOAD BUDGET -- the whole point of this script. With ``--accum session``
(default) many batches are bundled into ONE ``llm.session(remote=True)`` job and
the running sum is accumulated *on the server*; only a single (K, hidden) tensor
per session comes back. Over the 100-role set at all 80 layers that is ~260 MB
total, versus ~157 GB if each conversation's pooled vector were downloaded.
``--accum client`` is the fallback: one job per batch, per-batch (B,K,H) download.

Resumable at role granularity (skips roles whose <output_dir>/<role>.pt exists).

Usage:
    .venv-aa/bin/python src/assistant_axis_ndif/2_activations_ndif.py \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --responses_dir "$OUTPUT_DIR/responses" \
        --scores_dir    "$OUTPUT_DIR/scores" \
        --output_dir    "$OUTPUT_DIR/vectors"
"""

import argparse
import gc
import importlib.util
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import jsonlines
import torch
from dotenv import load_dotenv
from tqdm import tqdm

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]
AA_ROOT = REPO_ROOT / "assistant_axis"  # vendored submodule: library + data/

sys.path.insert(0, str(PIPELINE_DIR))
from aa_roles import add_role_args, resolve_roles  # noqa: E402

# .env: pipeline-local first, then repo root. Before importing nnsight so its
# from_env() sees NDIF_API_KEY / NDIF_HOST.
load_dotenv(PIPELINE_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")
os.environ.setdefault("NDIF_HOST", "https://api.ndif.us")  # nnsight 0.7 ships a broken bare host

# Windows (uv standalone Python) needs two fixes for NDIF's remote transport:
#  1. its bundled CA store is stale, so the websocket TLS handshake fails with
#     "certificate verify failed: certificate has expired" -- point SSL at certifi
#     (the CA bundle requests already uses successfully). Harmless on Linux.
#  2. the console codec (cp1252) can't encode nnsight's status glyphs (e.g. ◉),
#     crashing with UnicodeEncodeError -- force stdout/stderr to UTF-8.
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

# Load ConversationEncoder standalone (by file path) so we do NOT trigger
# assistant_axis/__init__.py, which pulls sklearn/matplotlib/vllm not present
# in .venv-aa. conversation.py itself only needs torch + transformers.
_conv_path = AA_ROOT / "assistant_axis" / "internals" / "conversation.py"
_spec = importlib.util.spec_from_file_location("aa_conversation", _conv_path)
_conv_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conv_mod)
ConversationEncoder = _conv_mod.ConversationEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("2_activations_ndif")


def key(pi: int, qi: int) -> str:
    return f"pos_p{pi}_q{qi}"


# --------------------------------------------------------------------------- #
# nnsight / NDIF setup
# --------------------------------------------------------------------------- #
def init_ndif(model_name: str):
    import nnsight
    from nnsight import CONFIG, LanguageModel

    api_key = (os.getenv("NDIF_API_KEY") or "").strip()
    hf_token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if not api_key:
        logger.error("NDIF_API_KEY not set (checked .env and environment)."); sys.exit(1)
    if not hf_token:
        logger.error("HF_TOKEN not set -- needed for the gated tokenizer/config."); sys.exit(1)

    # nnsight's from_env() already applied NDIF_API_KEY / NDIF_HOST; re-assert in-process
    # (harmless) and make sure the host carries a scheme.
    host = (os.getenv("NDIF_HOST", "").strip() or getattr(CONFIG.API, "HOST", "")
            or "https://api.ndif.us")
    if not host.startswith("http"):
        host = "https://" + host
    CONFIG.API.HOST = host
    CONFIG.API.APIKEY = api_key

    logger.info("nnsight %s  NDIF host=%s  model=%s",
                nnsight.__version__, CONFIG.API.HOST, model_name)
    # Harden here rather than in each caller: init_ndif is the single door every remote
    # script goes through, including 8_coords_ndif and 10_sycophancy_ndif. The rebuild that
    # hung for 40 minutes at "RUNNING (32.0s)" was running without any of this.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("ndif_common", Path(__file__).resolve().parent / "ndif_common.py")
    _nc = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_nc)
    # xmodel: Mistral needs fix_mistral_regex (ndif_common.tokenizer_kwargs); {} for every other model
    # xmodel: LanguageModel, or VisionLanguageModel for Mistral-3.1, with Mistral's regex fix and the
    # template date pinned (ndif_common.load_language_model)
    llm = _nc.load_language_model(model_name)
    _nc.harden()
    # xmodel: a vision-language checkpoint (Mistral-3.1, Gemma-3) keeps the decoder's sizes in
    # text_config; get_text_config() is the config itself for Llama and Qwen
    tc = llm.config.get_text_config()
    n_layers = int(tc.num_hidden_layers)
    logger.info("model config: %d layers, hidden=%d", n_layers, tc.hidden_size)
    return nnsight, llm, n_layers


# --------------------------------------------------------------------------- #
# per-role work
# --------------------------------------------------------------------------- #

_NC = None


def _ndif_common():
    """ndif_common loaded once by file path (it is loaded the same way inside init_ndif)."""
    global _NC
    if _NC is None:
        import importlib.util as _ilu
        _s = _ilu.spec_from_file_location("ndif_common_spans", Path(__file__).resolve().parent / "ndif_common.py")
        _NC = _ilu.module_from_spec(_s); _s.loader.exec_module(_NC)
    return _NC


def build_items(encoder: ConversationEncoder, conversations: list[list[dict]], max_length: int,
                drops: Counter | None = None):
    """Return list of (input_ids:list[int], a_start:int, a_end:int) for the last assistant span.

    `drops` counts every conversation that does not survive, by reason. It used to be
    possible to lose an entire dataset here without a single line of output: the assistant
    span sits at the END of the sequence, but truncation kept the HEAD (``full_ids[:max]``),
    so on a long-context row everything retained is context and the span falls past the cut.
    Measured on the evaluation corpus at max_length=2048, that silently removed all 80
    ai_audit rows (context median 5,285 tokens) and 22 sandbagging_v2 rows. Over-long
    sequences now keep their TAIL -- the response is the only thing we pool over -- and both
    truncations and drops are counted.
    """
    items = []
    if drops is None:
        drops = Counter()
    for conv in conversations:
        try:
            # transformers 5 made apply_chat_template default to return_dict=True, so the
            # vendored encoder (written for transformers 4) hands back a BatchEncoding
            # instead of list[int]; iterating that yields its KEYS and the spans come out
            # as (0, 1). Force the transformers-4 return shape.
            full_ids, spans = encoder.build_turn_spans(conv, return_dict=False)
        except Exception as e:  # noqa: BLE001
            logger.debug("span build failed: %s", e)
            drops["span_build_failed"] += 1
            continue
        # Deliberately outside the try: a per-conversation failure is worth skipping,
        # but a changed return shape is systemic and must not be swallowed into
        # "0 usable spans" for every role.
        if not (isinstance(full_ids, list) and (not full_ids or isinstance(full_ids[0], int))):
            raise TypeError(
                f"build_turn_spans returned {type(full_ids).__name__}, expected list[int]; "
                "the tokenizer/transformers version may have changed again"
            )
        asst = [s for s in spans if s.get("role") == "assistant"]
        if not asst:
            drops["no_assistant_span"] += 1
            continue
        s = asst[-1]
        a0, a1 = int(s["start"]), int(s["end"])
        a0 = _ndif_common().trim_chatml_header_newline(encoder.tokenizer, full_ids, a0, a1)  # skip the Qwen header newline
        if a1 <= a0:
            drops["empty_assistant_span"] += 1
            continue
        if len(full_ids) > max_length:
            # keep the TAIL, not the head: the assistant span is at the end and is the only
            # thing we pool over. Head-truncation drops the span entirely on long-context
            # rows, which is how a whole dataset can disappear without an error.
            cut = len(full_ids) - max_length
            full_ids = full_ids[cut:]
            a0, a1 = a0 - cut, a1 - cut
            drops["truncated_context"] += 1
        a0, a1 = max(a0, 0), min(a1, len(full_ids))
        if a0 >= a1:
            drops["span_lost_to_truncation"] += 1
            continue
        items.append((full_ids, a0, a1))
    return items


def report_drops(drops: Counter, n_in: int, n_out: int, label: str = "") -> None:
    """Say out loud what build_items threw away. Silence here is how a dataset vanishes."""
    lost = n_in - n_out
    if drops.get("truncated_context"):
        logger.warning("%s%d/%d conversation(s) exceeded max_length and were tail-truncated",
                       label, drops["truncated_context"], n_in)
    if lost:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(drops.items())
                           if k != "truncated_context")
        logger.warning("%s%d/%d conversation(s) dropped (%s)", label, lost, n_in, detail)


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def resolve_layers(spec, n_layers: int, target_layer: int) -> list[int]:
    """--layers 'target' | 'all' | explicit ints -> sorted list of layer indices."""
    if isinstance(spec, str):
        spec = [spec]
    if len(spec) == 1 and str(spec[0]).lower() in ("target", "middle", "mid"):
        return [target_layer]
    if len(spec) == 1 and str(spec[0]).lower() == "all":
        return list(range(n_layers))
    idx = sorted({int(x) for x in spec})
    bad = [i for i in idx if not 0 <= i < n_layers]
    if bad:
        raise ValueError(f"--layers out of range for a {n_layers}-layer model: {bad}")
    return idx


def pad_batch(batch, pad_id: int):
    """(ids, a0, a1) list -> right-padded input_ids, attention_mask, and a (B,T)
    float span mask that is 1.0 exactly on the assistant response tokens."""
    maxlen = max(len(ids) for ids, _, _ in batch)
    input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    span = torch.zeros((len(batch), maxlen), dtype=torch.float32)
    for b, (ids, a0, a1) in enumerate(batch):
        # The span mask is the sole source of truth for which tokens are pooled, so a
        # span reaching past the real tokens would silently average in padding.
        if not 0 <= a0 < a1 <= len(ids):
            raise ValueError(f"row {b}: span ({a0},{a1}) outside its {len(ids)} real tokens")
        input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[b, : len(ids)] = 1
        span[b, a0:a1] = 1.0
    return input_ids, attn, span


def is_vision_language(llm) -> bool:
    """True when the wrapped checkpoint is a vision-language model whose decoder sits at
    model.language_model (Mistral3/Gemma3ForConditionalGeneration). Client side only."""
    return hasattr(getattr(llm._model, "model", None), "language_model")


def _span_mean_sum(llm, layer_idx, span, vl=False):
    """Body shared by both accum modes. Runs on NDIF, inside an open trace.

    Returns (K,H) = sum over the batch of the per-response mean residual stream
    over that response's assistant tokens.
    """
    # Reduce one layer at a time, on that layer's own device, and stack the small
    # (H,) results on the CPU.
    #
    # Two reasons this is per-layer rather than one torch.stack over all layers:
    #   1. A 70B on NDIF is SHARDED across GPUs -- layer 0 lands on cuda:0 and layer
    #      79 on cuda:3 -- so stacking raw .output across layers raises
    #      "Expected all tensors to be on the same device". Only shows up on a
    #      multi-GPU deployment and with more than one layer requested.
    #   2. The stacked (K,B,T,H) intermediate is enormous: 80 x 8 x 2048 x 8192 in
    #      bf16 is ~86 GB. Reducing per layer never materialises it.
    #
    # transformers 5 / nnsight 0.7: layer .output is the hidden-states tensor
    # (B,T,H) itself, NOT a (hidden, ...) tuple -> no [0].
    # span was built on the client (CPU/float32) next to a sharded bf16 model, so
    # take device+dtype off a live activation at run time, per layer.
    # The mask stays fp32 and every reduction names dtype=float32. Casting it to the
    # model's bfloat16 rounds the token COUNT: bf16 carries 8 mantissa bits, so integers
    # step by 2 above 256 and by 4 above 512. Measured on the paper16 rollouts -- median
    # response 293 tokens, p90 474 -- 27.6% of responses landed on a count bf16 cannot
    # represent, so their mean was divided by the wrong denominator.
    # xmodel: a vision-language wrapper (vl, decided on the client) keeps the decoder at
    # model.language_model.layers; Llama and Qwen read model.layers exactly as before
    layers = llm.model.language_model.layers if vl else llm.model.layers
    per_layer = []
    for L in layer_idx:                                                 # ascending: reading
        h = layers[L].output                                        # a late module then an
        m = span.to(h.device, torch.float32)                            # early one is an error
        num = (h * m[:, :, None].to(h.dtype)).sum(dim=1, dtype=torch.float32)   # (B,H)
        den = m.sum(dim=1, dtype=torch.float32).clamp(min=1.0)[:, None]         # (B,1)
        per_layer.append((num / den).sum(dim=0).cpu())                  # (H,)
    return torch.stack(per_layer)                                       # (K,H) on CPU


def _run_session(llm, layer_idx, hidden, prepared, retries):
    """Bundle many batches into ONE NDIF job, accumulating the sum server-side.

    Only (K,hidden) + a count come back, regardless of how many batches went in.
    """
    vl = is_vision_language(llm)
    delay = 3.0
    for attempt in range(retries):
        try:
            with _ndif_common().session_deadline(), llm.session(remote=True):
                acc = torch.zeros(len(layer_idx), hidden, dtype=torch.float32).save()
                cnt = torch.zeros(1, dtype=torch.float32).save()
                for input_ids, attn, span in prepared:
                    with llm.trace({"input_ids": input_ids, "attention_mask": attn}):
                        acc += _span_mean_sum(llm, layer_idx, span, vl).cpu()
                        cnt += float(span.shape[0])
            a = getattr(acc, "value", acc)
            c = float(getattr(cnt, "value", cnt).reshape(-1)[0])
            if c == 0:
                raise RuntimeError("session returned count==0 (server-side accumulation "
                                   "did not take effect -- try --accum client)")
            return a, c
        except Exception as e:  # noqa: BLE001 -- engineio b'P' / ConnectionError / NDIF hiccup
            # Log the LAST line, not str(e)[:200]. nnsight wraps the remote failure in a
            # CLIENT-side traceback, so a prefix slice spends its whole budget on
            #   File ".../nnsight/intervention/backends/remote.py", line 567, in __call__
            # and stops immediately before the RemoteException description -- the only part
            # that says what went wrong. Fixed in 8_coords_ndif and 1_generate_ndif (F-22);
            # THIS file was the fifth sibling still carrying it, and on 2026-09-20 it cost the
            # whole 3B extraction: 22 roles failed with no recoverable reason in the log.
            _lines = [ln for ln in str(e).strip().splitlines() if ln.strip()]
            _tail = (_lines[-1].strip() if _lines else repr(e))[:400]
            logger.warning("  session attempt %d/%d failed: %s: %s",
                           attempt + 1, retries, type(e).__name__, _tail)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None


def _run_batch_client(llm, layer_idx, input_ids, attn, span, retries):
    """Fallback: one NDIF job per batch, (K,H) sum downloaded per batch."""
    vl = is_vision_language(llm)
    delay = 3.0
    for attempt in range(retries):
        try:
            with llm.trace({"input_ids": input_ids, "attention_mask": attn}, remote=True):
                out = _span_mean_sum(llm, layer_idx, span, vl).cpu().save()
            return getattr(out, "value", out), float(span.shape[0])
        except Exception as e:  # noqa: BLE001
            _lines = [ln for ln in str(e).strip().splitlines() if ln.strip()]
            _tail = (_lines[-1].strip() if _lines else repr(e))[:400]
            logger.warning("  trace attempt %d/%d failed: %s: %s",
                           attempt + 1, retries, type(e).__name__, _tail)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None


def batch_by_tokens(items, token_budget: int, max_rows: int):
    """Group items so that len(batch) * longest-in-batch stays under `token_budget`.

    A row count cannot bound remote memory. With padding a call costs
    len(batch) * max_len_in_batch, so a fixed --batch_size leaves the second factor free
    and the real cost swings with whatever rows happen to land together. Measured on this
    deployment, a judge call of 11,680 padded tokens completes and 17,520 is rejected 0.3s
    after dispatch; the evaluation corpus reaches ~7,100 tokens on a single ai_audit row,
    so batch_size 8 would send ~56,800 and fail on exactly the dataset we most want.

    Items are sorted longest-first so similar lengths batch together, which cuts padding
    waste. Safe here because stage 2 sums over the batch to build a role mean, so row
    order carries no information. A per-sample extraction must keep indices instead.
    """
    ordered = sorted(items, key=lambda it: len(it[0]), reverse=True)
    out, cur, cur_max = [], [], 0
    for it in ordered:
        L = len(it[0])
        new_max = max(cur_max, L)
        if cur and ((len(cur) + 1) * new_max > token_budget or len(cur) >= max_rows):
            out.append(cur)
            cur, cur_max = [it], L
        else:
            cur.append(it)
            cur_max = new_max
    if cur:
        out.append(cur)
    over = [b for b in out if len(b) == 1 and len(b[0][0]) > token_budget]
    if over:
        logger.warning("%d row(s) exceed the %d-token budget on their own and are sent alone; "
                       "if these fail, raise --token_budget or drop the rows",
                       len(over), token_budget)
    return out


def extract_role_vector(llm, layer_idx, hidden, items, args, pad_id):
    """Mean over kept responses of (mean over assistant-span tokens of residual stream)."""
    total = torch.zeros(len(layer_idx), hidden, dtype=torch.float64)
    count = 0.0
    failed = 0
    groups = batch_by_tokens(items, args.token_budget, args.batch_size)
    widest = max((len(b) * max(len(i[0]) for i in b) for b in groups), default=0)
    logger.info("  %d rows -> %d batches (<= %d rows, widest %d padded tokens, budget %d)",
                len(items), len(groups), args.batch_size, widest, args.token_budget)
    batches = [pad_batch(b, pad_id) for b in groups]

    if args.accum == "session":
        chunks = list(batched(batches, args.session_batches))
        for chunk in tqdm(chunks, desc="  ndif sessions", leave=False):
            got = _run_session(llm, layer_idx, hidden, chunk, args.retries)
            if got is None:
                failed += sum(int(s.shape[0]) for _, _, s in chunk)
                continue
            s, n = got
            total += s.to(torch.float64)
            count += n
    else:
        for input_ids, attn, span in tqdm(batches, desc="  ndif batches", leave=False):
            got = _run_batch_client(llm, layer_idx, input_ids, attn, span, args.retries)
            if got is None:
                failed += int(span.shape[0])
                continue
            s, n = got
            total += s.to(torch.float64)
            count += n

    if failed:
        logger.warning("  %d/%d responses dropped to NDIF errors", failed, failed + int(count))
    if count == 0:
        return None, 0
    return (total / count).to(torch.float32), int(count)  # (K,H)


def load_scores(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def process_role(llm, layer_idx, hidden, encoder, role: str, args, pad_id: int) -> None:
    out_file = Path(args.output_dir) / f"{role}.pt"
    if out_file.exists() and not args.overwrite:
        logger.info("Skip %s (vector exists)", role); return

    resp_file = Path(args.responses_dir) / f"{role}.jsonl"
    if not resp_file.exists():
        logger.warning("Skip %s: no responses", role); return

    rows = []
    with jsonlines.open(resp_file) as reader:
        for r in reader:
            rows.append(r)

    is_default = "default" in role
    keep = {int(x) for x in str(args.keep_scores).split(",") if x.strip() != ""}
    if is_default:
        convs = [r["conversation"] for r in rows]
        vtype = "mean"
    else:
        scores = load_scores(Path(args.scores_dir) / f"{role}.json")
        convs = [r["conversation"] for r in rows
                 if scores.get(key(r["prompt_index"], r["question_index"])) in keep]
        vtype = "pos_3"

    if len(convs) < args.min_count:
        logger.warning("Skip %s: only %d kept (min_count=%d)", role, len(convs), args.min_count)
        return

    drops: Counter = Counter()
    items = build_items(encoder, convs, args.max_length, drops)
    report_drops(drops, len(convs), len(items), f"{role}: ")
    if len(items) < args.min_count:
        logger.warning("Skip %s: only %d usable spans (min_count=%d)", role, len(items), args.min_count)
        return
    logger.info("Role %s: %d kept, %d usable -> NDIF", role, len(convs), len(items))

    vec, n_used = extract_role_vector(llm, layer_idx, hidden, items, args, pad_id)
    if vec is None:
        logger.error("Role %s: no vector produced", role); return
    if not torch.isfinite(vec).all():
        logger.error("Role %s: vector has non-finite values; not saving", role); return

    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": vec, "layers": list(layer_idx), "type": vtype,
                "role": role, "n": n_used}, out_file)
    norms = vec.norm(dim=1)
    logger.info("Role %s: saved %s over n=%d  layers=%s  norms[min/max]=%.3f/%.3f -> %s",
                role, tuple(vec.shape), n_used,
                layer_idx if len(layer_idx) <= 4 else f"{layer_idx[0]}..{layer_idx[-1]}",
                norms.min(), norms.max(), out_file)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--responses_dir", required=True)
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    add_role_args(ap)
    ap.add_argument("--keep_scores", default="3", help="comma-sep judge scores to keep (default 3)")
    ap.add_argument("--min_count", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="max ROWS per call; the real bound is --token_budget")
    ap.add_argument("--token_budget", type=int, default=12000,
                    help="max padded tokens per remote call (rows x longest-in-batch). "
                         "A judge call of 11,680 completes on this deployment and 17,520 "
                         "is rejected, so 12000 leaves a little headroom.")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--layers", nargs="+", default=["all"],
                    help="'all' (every layer, default), 'target' (model's middle layer), or explicit indices")
    ap.add_argument("--session_batches", type=int, default=20,
                    help="batches bundled into one NDIF session (--accum session). Keep the "
                         "session under NDIF's ~1h per-request ceiling.")
    ap.add_argument("--accum", choices=("session", "client"), default="session",
                    help="'session' accumulates the sum on NDIF and downloads one (K,H) "
                         "tensor per session; 'client' downloads one per batch (fallback)")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    resp_files = sorted(Path(args.responses_dir).glob("*.jsonl"))
    _want = resolve_roles(args)
    if _want:
        wanted = set(_want)
        resp_files = [f for f in resp_files if f.stem in wanted]

    nnsight, llm, n_layers = init_ndif(args.model)
    hidden = int(llm.config.get_text_config().hidden_size)

    # Middle layer for known models comes from assistant_axis/models.py
    # (Llama-3.3-70B -> 40 of 80); fall back to n_layers // 2.
    target_layer = n_layers // 2
    try:
        _mp = AA_ROOT / "assistant_axis" / "models.py"
        _s = importlib.util.spec_from_file_location("aa_models", _mp)
        _m = importlib.util.module_from_spec(_s)
        _s.loader.exec_module(_m)
        target_layer = int(_m.MODEL_CONFIGS[args.model]["target_layer"])
    except Exception:  # noqa: BLE001 -- unknown model, or models.py pulls something missing
        logger.info("models.py lookup failed for %s; using middle layer %d",
                    args.model, target_layer)
    layer_idx = resolve_layers(args.layers, n_layers, target_layer)

    encoder = ConversationEncoder(llm.tokenizer, args.model)
    pad_id = llm.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = llm.tokenizer.eos_token_id
    logger.info("Extracting %d roles | layers=%s (K=%d of %d) | hidden=%d | accum=%s | pad_id=%s",
                len(resp_files),
                layer_idx if len(layer_idx) <= 6 else f"{layer_idx[0]}..{layer_idx[-1]}",
                len(layer_idx), n_layers, hidden, args.accum, pad_id)
    logger.info("Per-role vector: (%d, %d) float32 = %.1f KB",
                len(layer_idx), hidden, len(layer_idx) * hidden * 4 / 1024)

    for rf in resp_files:
        try:
            process_role(llm, layer_idx, hidden, encoder, rf.stem, args, pad_id)
        except Exception as e:  # noqa: BLE001
            logger.error("Role %s aborted: %s", rf.stem, e, exc_info=True)
        gc.collect()
    logger.info("Done.")


if __name__ == "__main__":
    main()
