#!/usr/bin/env python3
"""
Step 1 (API variant): generate role responses via Hugging Face Inference Providers.

Drop-in replacement for ``1_generate.py`` for setups with no local GPU. Produces the
same per-role JSONL schema that ``3_judge_hf.py`` and ``2_activations_ndif.py`` consume:

    {"system_prompt": <str>, "prompt_index": <int>, "question_index": <int>,
     "question": <str>, "label": "pos",
     "conversation": [ {"role": "system", "content": <instr>}?,   # omitted if instr == ""
                       {"role": "user", "content": <question>},
                       {"role": "assistant", "content": <response>} ]}

Resumable: a role whose JSONL already contains every (prompt_index, question_index)
key is skipped; missing combinations are appended.

Usage:
    .venv-aa/bin/python src/assistant_axis_ndif/1_generate_hf.py \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --output_dir "$OUTPUT_DIR/responses" \
        --question_count 240
"""

import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

# --- repo imports -----------------------------------------------------------
PIPELINE_DIR = Path(__file__).resolve().parent
AA_ROOT = PIPELINE_DIR.parent                      # .../assistant_axis
sys.path.insert(0, str(AA_ROOT))

load_dotenv(AA_ROOT.parent / ".env")              # /workspaces/expLB/.env
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("1_generate_hf")
logging.getLogger("httpx").setLevel(logging.WARNING)

SHORT_NAME = "Llama"                              # {model_name} placeholder substitution


def key(pi: int, qi: int) -> str:
    return f"pos_p{pi}_q{qi}"


def load_questions(path: Path, n: int) -> list[str]:
    qs = []
    with jsonlines.open(path) as reader:
        for row in reader:
            qs.append(row["question"])
    return qs[:n]


def load_role(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_messages(instruction: str, question: str) -> list[dict]:
    """Llama supports system prompts; empty instruction -> user-only."""
    instruction = (instruction or "").replace("{model_name}", SHORT_NAME).strip()
    msgs = []
    if instruction:
        msgs.append({"role": "system", "content": instruction})
    msgs.append({"role": "user", "content": question})
    return msgs


def make_client():
    from huggingface_hub import InferenceClient

    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if not token:
        logger.error("HF_TOKEN not set (checked .env and environment).")
        sys.exit(1)
    provider = os.getenv("HF_PROVIDER", "auto").strip() or "auto"
    return InferenceClient(provider=provider, api_key=token)


def generate_one(client, model, messages, args, _local=threading.local()):
    from huggingface_hub.errors import HfHubHTTPError
    import time

    delay = 2.0
    for attempt in range(args.max_retries):
        try:
            out = client.chat_completion(
                messages=messages,
                model=model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            return (out.choices[0].message.content or "").strip()
        except HfHubHTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code in (429, 500, 502, 503, 504) and attempt < args.max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise
        except Exception:
            if attempt < args.max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise
    return None


def process_role(client, role_name: str, role_data: dict, questions: list[str], args) -> None:
    out_file = Path(args.output_dir) / f"{role_name}.jsonl"
    existing: set[str] = set()
    if out_file.exists():
        with jsonlines.open(out_file) as reader:
            for row in reader:
                existing.add(key(row["prompt_index"], row["question_index"]))

    instructions = [d.get("pos", "") for d in role_data.get("instruction", [])]
    if not instructions:
        logger.warning("Role %s has no instructions; skipping", role_name)
        return

    prompt_indices = args.prompt_indices or list(range(len(instructions)))
    jobs = []
    for pi in prompt_indices:
        if pi >= len(instructions):
            continue
        for qi, q in enumerate(questions):
            if key(pi, qi) in existing:
                continue
            jobs.append((pi, qi, q, instructions[pi]))

    if not jobs:
        logger.info("Role %s already complete (%d responses)", role_name, len(existing))
        return

    logger.info("Role %s: %d / %d responses to generate", role_name, len(jobs),
                len(prompt_indices) * len(questions))

    results = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        fut2job = {
            ex.submit(generate_one, client, args.model, build_messages(instr, q), args): (pi, qi, q, instr)
            for (pi, qi, q, instr) in jobs
        }
        for fut in tqdm(as_completed(fut2job), total=len(fut2job), desc=role_name):
            pi, qi, q, instr = fut2job[fut]
            try:
                resp = fut.result()
            except Exception as e:
                logger.error("  %s p%d q%d failed: %s", role_name, pi, qi, e)
                continue
            if not resp:
                continue
            msgs = build_messages(instr, q)
            with lock:
                results.append({
                    "system_prompt": (instr or "").replace("{model_name}", SHORT_NAME),
                    "prompt_index": pi,
                    "question_index": qi,
                    "question": q,
                    "label": "pos",
                    "conversation": msgs + [{"role": "assistant", "content": resp}],
                })

    results.sort(key=lambda r: (r["prompt_index"], r["question_index"]))
    with jsonlines.open(out_file, mode="a") as writer:
        for r in results:
            writer.write(r)
    logger.info("Role %s: wrote %d new responses -> %s", role_name, len(results), out_file)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--roles_dir", default=str(AA_ROOT / "data" / "roles" / "instructions"))
    ap.add_argument("--questions_file", default=str(AA_ROOT / "data" / "extraction_questions.jsonl"))
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--question_count", type=int, default=240)
    ap.add_argument("--prompt_indices", type=int, nargs="+", default=None,
                    help="Which instruction variants to use (default: all, normally 0-4)")
    ap.add_argument("--roles", nargs="+", default=None, help="Subset of role names")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max_retries", type=int, default=6)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    questions = load_questions(Path(args.questions_file), args.question_count)
    logger.info("Loaded %d questions", len(questions))

    role_files = sorted(Path(args.roles_dir).glob("*.json"))
    if args.roles:
        wanted = set(args.roles)
        role_files = [f for f in role_files if f.stem in wanted]
    logger.info("Processing %d roles", len(role_files))

    client = make_client()
    for rf in role_files:
        try:
            process_role(client, rf.stem, load_role(rf), questions, args)
        except Exception as e:
            logger.error("Role %s aborted: %s", rf.stem, e)

    logger.info("Done.")


if __name__ == "__main__":
    main()
