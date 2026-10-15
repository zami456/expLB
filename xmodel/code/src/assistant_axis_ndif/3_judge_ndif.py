#!/usr/bin/env python3
"""
Step 3 (NDIF variant): score role responses 0-3 with an open judge model
(`meta-llama/Llama-3.1-70B-Instruct` by default) served remotely on NDIF via
nnsight `.generate(remote=True)`. No HF Inference, no OpenAI.

Llama-3.1-70B-Instruct is the default because it is HOT *and pinned* on NDIF, so it
never triggers a cold deploy -- a cold-start wait is what crashed earlier runs. It is
also ~6x cheaper per call than the 405B (which is prefill-dominated here) and more
capable than the paper's gpt-4.1-mini judge. Llama-3.1-405B-Instruct is a heavier
alternative (set --judge_model) but is not always pinned; Qwen2.5-72B-Instruct is a
different-family option for more independent judging but is COLD and must warm up.

Same I/O contract as the stock `3_judge.py`:
  - reads  <responses_dir>/<role>.jsonl  and  <roles_dir>/<role>.json  ("eval_prompt")
  - writes <output_dir>/<role>.json  ==  {"pos_p{pi}_q{qi}": <int 0..3>, ...}
  - resumable: existing scores kept; only missing keys queried
  - roles whose role file has no "eval_prompt" (e.g. default) are skipped

    .venv-aa/bin/python src/assistant_axis_ndif/3_judge_ndif.py \
        --responses_dir "$OUT_DIR/responses" \
        --roles_dir assistant_axis/data/roles/instructions \
        --output_dir "$OUT_DIR/scores" \
        --judge_model meta-llama/Llama-3.1-70B-Instruct
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]
AA_ROOT = REPO_ROOT / "assistant_axis"  # vendored submodule: library + data/

sys.path.insert(0, str(PIPELINE_DIR))
from aa_roles import add_role_args, resolve_roles  # noqa: E402

load_dotenv(PIPELINE_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")
os.environ.setdefault("NDIF_HOST", "https://api.ndif.us")

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("3_judge_ndif")
_NC = None  # ndif_common, loaded in main(); its session_deadline wraps every remote session (2026-10-14)


def key(pi: int, qi: int) -> str:
    return f"pos_p{pi}_q{qi}"


def parse_judge_score(text: str):
    """First integer 0..3 in the text, else None (from assistant_axis.judge)."""
    if not text:
        return None
    nums = re.findall(r"\b(\d+)\b", text.strip())
    if not nums:
        return None
    try:
        s = int(nums[0])
        return s if 0 <= s <= 3 else None
    except ValueError:
        return None


def assistant_text(conversation: list[dict]) -> str:
    for msg in reversed(conversation):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _encode(llm, prompts: list[str]):
    enc = llm.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    return enc["input_ids"], enc["attention_mask"], enc["input_ids"].shape[1]


def batch_by_tokens(llm, todo, token_budget: int, max_rows: int):
    """Group (key, prompt) pairs so len(batch) * longest-in-batch stays under `token_budget`.

    A row count cannot bound remote memory: with left padding a call costs
    len(batch) * max_len_in_batch, so a fixed --judge_batch leaves the second factor free.
    Measured on this deployment, 11,680 padded tokens completes and 17,520 is rejected 0.3s
    after dispatch -- and the old default of 48 sent 35,040, which failed every job and
    produced zero scores for a whole 26-role run before anyone noticed.

    Sorted longest-first so similar lengths batch together, cutting padding waste. Order is
    irrelevant here because each result is written back by its own key.
    """
    lens = {k: len(llm.tokenizer(p, add_special_tokens=False).input_ids) for k, p in todo}
    ordered = sorted(todo, key=lambda kp: lens[kp[0]], reverse=True)
    out, cur, cur_max = [], [], 0
    for kp in ordered:
        L = lens[kp[0]]
        new_max = max(cur_max, L)
        if cur and ((len(cur) + 1) * new_max > token_budget or len(cur) >= max_rows):
            out.append(cur)
            cur, cur_max = [kp], L
        else:
            cur.append(kp)
            cur_max = new_max
    if cur:
        out.append(cur)
    return out


def judge_batches_session(llm, prompt_batches: list[list[str]], retries: int) -> list[list[str]] | None:
    """Bundle several judge batches into ONE NDIF job.

    A judge call is ~4 output tokens over a long prompt, so per-job queue and transport
    overhead dominates the actual work: un-bundled, --judge_batch 48 means one NDIF job
    per 48 responses. Same trade-off as stage 1 -- NDIF kills a request after ~1h and one
    failed trace aborts the whole session -- so keep --session_batches modest.
    """
    import nnsight

    encs = [_encode(llm, p) for p in prompt_batches]
    delay = 3.0
    for attempt in range(retries):
        try:
            with _NC.session_deadline(), llm.session(remote=True):
                outs = nnsight.save([])
                for in_ids, attn, _ in encs:
                    with llm.generate(in_ids, attention_mask=attn, max_new_tokens=4,
                                      do_sample=False, pad_token_id=llm.tokenizer.pad_token_id):
                        outs.append(llm.generator.output)
            outs = getattr(outs, "value", outs)
            if len(outs) != len(encs):
                raise RuntimeError(f"session returned {len(outs)} outputs, expected {len(encs)}")
            return [llm.tokenizer.batch_decode(o[:, in_len:], skip_special_tokens=True)
                    for (_, _, in_len), o in zip(encs, outs)]
        except Exception as e:  # noqa: BLE001 -- engineio b'P' / ConnectionError / NDIF hiccup
            logger.warning("  judge session attempt %d/%d (%d batches) failed: %s: %s",
                           attempt + 1, retries, len(encs), type(e).__name__, ((str(e).strip().splitlines() or [repr(e)])[-1])[:300])  # last line: the cause (2026-10-07)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None


def judge_batch(llm, prompts: list[str], retries: int) -> list[str] | None:
    in_ids, attn, in_len = _encode(llm, prompts)
    delay = 3.0
    for attempt in range(retries):
        try:
            with llm.generate(
                in_ids, attention_mask=attn, max_new_tokens=4, do_sample=False,
                pad_token_id=llm.tokenizer.pad_token_id, remote=True,
            ):
                out = llm.generator.output.save()
            out = getattr(out, "value", out)
            return llm.tokenizer.batch_decode(out[:, in_len:], skip_special_tokens=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("  judge batch attempt %d/%d failed: %s: %s",
                           attempt + 1, retries, type(e).__name__, ((str(e).strip().splitlines() or [repr(e)])[-1])[:300])  # last line: the cause (2026-10-07)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None


def process_role(llm, role: str, args) -> None:
    resp_file = Path(args.responses_dir) / f"{role}.jsonl"
    role_file = Path(args.roles_dir) / f"{role}.json"
    out_file = Path(args.output_dir) / f"{role}.json"

    if not role_file.exists():
        logger.info("Skip %s: no role file", role); return
    template = json.loads(role_file.read_text()).get("eval_prompt", "")
    if not template:
        logger.info("Skip %s: no eval_prompt", role); return

    scores: dict[str, int] = {}
    if out_file.exists():
        try:
            scores = json.loads(out_file.read_text())
        except Exception:
            scores = {}

    todo = []  # (key, prompt)
    with jsonlines.open(resp_file) as reader:
        for r in reader:
            k = key(r["prompt_index"], r["question_index"])
            if k in scores:
                continue
            filled = template.format(question=r["question"], answer=assistant_text(r["conversation"]))
            prompt = llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": filled}], tokenize=False, add_generation_prompt=True
            )
            todo.append((k, prompt))
    if not todo:
        logger.info("Role %s: already scored (%d)", role, len(scores)); return
    logger.info("Role %s: scoring %d", role, len(todo))

    new = 0
    batches = batch_by_tokens(llm, todo, args.token_budget, args.judge_batch)
    widest = max((len(b) * max(len(llm.tokenizer(p, add_special_tokens=False).input_ids)
                               for _, p in b) for b in batches), default=0)
    logger.info("  %d prompts -> %d batches (<= %d rows, widest ~%d padded tokens, budget %d)",
                len(todo), len(batches), args.judge_batch, widest, args.token_budget)
    groups = list(batched(batches, max(1, args.session_batches)))
    for group in tqdm(groups, desc=role):
        if len(group) > 1:
            per_batch = judge_batches_session(llm, [[p for _, p in b] for b in group], args.retries)
        else:
            texts = judge_batch(llm, [p for _, p in group[0]], args.retries)
            per_batch = None if texts is None else [texts]
        if per_batch is None:
            logger.error("  role %s: %d batches permanently failed; leaving for resume",
                         role, len(group))
            continue
        for batch, texts in zip(group, per_batch):
            for (k, _), t in zip(batch, texts):
                s = parse_judge_score(t)
                if s is not None:
                    scores[k] = s
                    new += 1
        # checkpoint after every session so a later failure never loses scored work
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(scores, indent=2))
    logger.info("Role %s: +%d scores (total %d) -> %s", role, new, len(scores), out_file)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses_dir", required=True)
    ap.add_argument("--roles_dir", default=str(AA_ROOT / "data" / "roles" / "instructions"))
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--judge_model", default="meta-llama/Llama-3.1-70B-Instruct",
                    help="HOT+pinned on NDIF and ~6x cheaper than the 405B; pass "
                         "meta-llama/Llama-3.1-405B-Instruct for the heavier judge when "
                         "it is HOT, or Qwen/Qwen2.5-72B-Instruct for a different family")
    add_role_args(ap)
    ap.add_argument("--judge_batch", type=int, default=16,
                    help="max ROWS per call; the real bound is --token_budget")
    ap.add_argument("--token_budget", type=int, default=12000,
                    help="max padded tokens per remote call (rows x longest-in-batch)")
    ap.add_argument("--session_batches", type=int, default=8,
                    help="judge batches bundled into one NDIF session (1 = one job per "
                         "batch, the old behaviour). Keep session_batches * seconds_per_batch "
                         "well under NDIF's ~1h per-request ceiling; a session retries whole.")
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    resp_files = sorted(Path(args.responses_dir).glob("*.jsonl"))
    _want = resolve_roles(args)
    if _want:
        want = set(_want)
        resp_files = [f for f in resp_files if f.stem in want]

    import nnsight  # noqa: F401
    from nnsight import LanguageModel

    logger.info("nnsight %s  NDIF host=%s  judge=%s",
                nnsight.__version__, os.environ.get("NDIF_HOST"), args.judge_model)
    llm = LanguageModel(args.judge_model)

    # Resilience shared with the rest of the pipeline. Not a trace body, so importing it is
    # safe -- only blocks that appear inside trace/generate/session must live in this file.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("ndif_common", PIPELINE_DIR / "ndif_common.py")
    global _NC
    _nc = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_nc)
    _nc.harden()
    _NC = _nc
    llm.tokenizer.padding_side = "left"
    if llm.tokenizer.pad_token_id is None:
        llm.tokenizer.pad_token_id = llm.tokenizer.eos_token_id

    logger.info("Judging %d roles", len(resp_files))
    for rf in resp_files:
        try:
            process_role(llm, rf.stem, args)
        except Exception as e:  # noqa: BLE001
            logger.error("Role %s aborted: %s", rf.stem, e)
    logger.info("Done.")


if __name__ == "__main__":
    main()
