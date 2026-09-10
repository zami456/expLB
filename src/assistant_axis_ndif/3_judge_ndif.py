#!/usr/bin/env python3
"""
Step 3 (NDIF variant): score role responses 0-3 with an open judge model
(`meta-llama/Llama-3.1-405B-Instruct` by default) served remotely on NDIF via
nnsight `.generate(remote=True)`. No HF Inference, no OpenAI.

405B is the default because it is HOT *and pinned* on NDIF, so it never triggers
a cold deploy -- the long COLD wait is what crashed earlier runs. Qwen2.5-72B-Instruct
is a reasonable alternative (different model family from the Llama subject model =
more independent judging) but is COLD on NDIF and must warm up first.

Same I/O contract as the stock `3_judge.py`:
  - reads  <responses_dir>/<role>.jsonl  and  <roles_dir>/<role>.json  ("eval_prompt")
  - writes <output_dir>/<role>.json  ==  {"pos_p{pi}_q{qi}": <int 0..3>, ...}
  - resumable: existing scores kept; only missing keys queried
  - roles whose role file has no "eval_prompt" (e.g. default) are skipped

    .venv-aa/bin/python assistant_axis/pipeline/3_judge_ndif.py \
        --responses_dir "$OUT_DIR/responses" \
        --roles_dir assistant_axis/data/roles/instructions \
        --output_dir "$OUT_DIR/scores" \
        --judge_model meta-llama/Llama-3.1-405B-Instruct
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

load_dotenv(PIPELINE_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")
os.environ.setdefault("NDIF_HOST", "https://api.ndif.us")
for _v in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
    if os.getenv(_v):
        os.environ["HF_TOKEN"] = os.environ["HF_HUB_TOKEN"] = os.environ[_v].strip()
        break

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("3_judge_ndif")


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


def judge_batch(llm, prompts: list[str], retries: int) -> list[str] | None:
    enc = llm.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    in_ids, attn = enc["input_ids"], enc["attention_mask"]
    in_len = in_ids.shape[1]
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
                           attempt + 1, retries, type(e).__name__, str(e)[:200])
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
    for batch in tqdm(list(batched(todo, args.judge_batch)), desc=role):
        texts = judge_batch(llm, [p for _, p in batch], args.retries)
        if texts is None:
            logger.error("  role %s: batch permanently failed; leaving for resume", role); continue
        for (k, _), t in zip(batch, texts):
            s = parse_judge_score(t)
            if s is not None:
                scores[k] = s
                new += 1
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(scores, indent=2))
    logger.info("Role %s: +%d scores (total %d) -> %s", role, new, len(scores), out_file)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses_dir", required=True)
    ap.add_argument("--roles_dir", default=str(AA_ROOT / "data" / "roles" / "instructions"))
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--judge_model", default="meta-llama/Llama-3.1-405B-Instruct",
                    help="HOT+pinned on NDIF; Qwen/Qwen2.5-72B-Instruct is the "
                         "different-family alternative but is COLD and must warm up")
    ap.add_argument("--roles", nargs="+", default=None)
    ap.add_argument("--judge_batch", type=int, default=48)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    resp_files = sorted(Path(args.responses_dir).glob("*.jsonl"))
    if args.roles:
        want = set(args.roles)
        resp_files = [f for f in resp_files if f.stem in want]

    import nnsight  # noqa: F401
    from nnsight import LanguageModel

    logger.info("nnsight %s  NDIF host=%s  judge=%s",
                nnsight.__version__, os.environ.get("NDIF_HOST"), args.judge_model)
    llm = LanguageModel(args.judge_model)
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
