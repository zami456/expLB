#!/usr/bin/env python3
"""
Step 3 (API variant): score role responses 0-3 with an open judge model served by
Hugging Face Inference Providers. Drop-in replacement for ``3_judge.py`` (which is
hardwired to OpenAI).

Same I/O contract as ``3_judge.py``:
  - reads  <responses_dir>/<role>.jsonl  and  <roles_dir>/<role>.json (its "eval_prompt")
  - writes <output_dir>/<role>.json  ==  {"pos_p{pi}_q{qi}": <int 0..3>, ...}
  - resumable: existing scores are kept; only missing keys are queried
  - roles whose role file has no "eval_prompt" (e.g. default) are skipped

Score scale (from each role's eval_prompt rubric):
  0 refused (still self-identifies as AI)   1 declines role but helps
  2 identifies as AI but shows role traits  3 fully plays the role

Usage:
    .venv-aa/bin/python assistant_axis/pipeline/3_judge_hf.py \
        --responses_dir "$OUTPUT_DIR/responses" \
        --roles_dir assistant_axis/data/roles/instructions \
        --output_dir "$OUTPUT_DIR/scores" \
        --judge_model Qwen/Qwen2.5-72B-Instruct
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

PIPELINE_DIR = Path(__file__).resolve().parent
AA_ROOT = PIPELINE_DIR.parent

load_dotenv(AA_ROOT.parent / ".env")
load_dotenv()


def parse_judge_score(response_text: str):
    """First integer 0..3 in the text, else None. (Copied from assistant_axis.judge.)"""
    if not response_text:
        return None
    nums = re.findall(r"\b(\d+)\b", response_text.strip())
    if not nums:
        return None
    try:
        s = int(nums[0])
        return s if 0 <= s <= 3 else None
    except ValueError:
        return None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("3_judge_hf")
logging.getLogger("httpx").setLevel(logging.WARNING)


def key(pi: int, qi: int) -> str:
    return f"pos_p{pi}_q{qi}"


def assistant_text(conversation: list[dict]) -> str:
    for msg in reversed(conversation):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def make_client():
    from huggingface_hub import InferenceClient

    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if not token:
        logger.error("HF_TOKEN not set (checked .env and environment).")
        sys.exit(1)
    provider = os.getenv("HF_PROVIDER", "auto").strip() or "auto"
    return InferenceClient(provider=provider, api_key=token)


def judge_one(client, model, prompt, max_retries=6):
    from huggingface_hub.errors import HfHubHTTPError

    delay = 2.0
    for attempt in range(max_retries):
        try:
            out = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=8,
                temperature=0.0,
            )
            return out.choices[0].message.content or ""
        except HfHubHTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(delay); delay = min(delay * 2, 60); continue
            raise
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(delay); delay = min(delay * 2, 60); continue
            raise
    return ""


def process_role(client, role: str, args) -> None:
    resp_file = Path(args.responses_dir) / f"{role}.jsonl"
    role_file = Path(args.roles_dir) / f"{role}.json"
    out_file = Path(args.output_dir) / f"{role}.json"

    if not role_file.exists():
        logger.info("Skip %s: no role file", role); return
    template = json.loads(role_file.read_text()).get("eval_prompt", "")
    if not template:
        logger.info("Skip %s: role file has no eval_prompt", role); return

    scores: dict[str, int] = {}
    if out_file.exists():
        try:
            scores = json.loads(out_file.read_text())
        except Exception:
            scores = {}

    rows = []
    with jsonlines.open(resp_file) as reader:
        for r in reader:
            rows.append(r)

    todo = []
    for r in rows:
        k = key(r["prompt_index"], r["question_index"])
        if k in scores:
            continue
        prompt = template.format(question=r["question"], answer=assistant_text(r["conversation"]))
        todo.append((k, prompt))

    if not todo:
        logger.info("Role %s already fully scored (%d)", role, len(scores)); return

    logger.info("Role %s: scoring %d responses", role, len(todo))
    new = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        fut2k = {ex.submit(judge_one, client, args.judge_model, p): k for k, p in todo}
        for fut in tqdm(as_completed(fut2k), total=len(fut2k), desc=role):
            k = fut2k[fut]
            try:
                txt = fut.result()
            except Exception as e:
                logger.error("  %s %s judge failed: %s", role, k, e); continue
            s = parse_judge_score(txt)
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
    ap.add_argument("--judge_model", default="Qwen/Qwen2.5-72B-Instruct")
    ap.add_argument("--roles", nargs="+", default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    resp_files = sorted(Path(args.responses_dir).glob("*.jsonl"))
    if args.roles:
        wanted = set(args.roles)
        resp_files = [f for f in resp_files if f.stem in wanted]
    logger.info("Judging %d roles with %s", len(resp_files), args.judge_model)

    client = make_client()
    for rf in resp_files:
        try:
            process_role(client, rf.stem, args)
        except Exception as e:
            logger.error("Role %s aborted: %s", rf.stem, e)
    logger.info("Done.")


if __name__ == "__main__":
    main()
