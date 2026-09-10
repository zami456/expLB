#!/usr/bin/env python3
"""Step 0c: measure real NDIF throughput and project the full run.

Runs the ACTUAL production code paths against the real 70B/405B models, times
them, and turns "how long will this take?" into a number instead of a guess.

    .venv-aa/bin/python src/assistant_axis_ndif/_measure_throughput.py \
        [--gen_batch 24] [--max_new_tokens 512] [--n_batches 5]

Reports seconds/generation and the projected step-1 wall-clock for the full
331,200-generation corpus, splitting queue wait from compute where NDIF exposes it.
Nothing is written; this is a read-only measurement.
"""
import argparse
import importlib.util
import json
import os
import statistics as _st
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

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

import torch  # noqa: E402
from nnsight import CONFIG, LanguageModel  # noqa: E402

CONFIG.API.HOST = os.environ["NDIF_HOST"]
CONFIG.API.APIKEY = (os.getenv("NDIF_API_KEY") or "").strip()
CONFIG.APP.REMOTE_LOGGING = False

def _default_n_roles() -> int:
    """Roles in the curated set (role_groups.json); falls back to the paper's 276."""
    try:
        g = json.loads((PIPELINE_DIR / 'role_groups.json').read_text())['groups']
        return len({r for spec in g.values() for r in spec['roles']})
    except Exception:  # noqa: BLE001
        return 276


def _sample_answer() -> str:
    """A realistic assistant response for the judge prompt.

    Prefers a real rollout from a previous smoke run; falls back to filler sized to
    the measured mean output (~1900 chars). Length matters: the judge emits 4 tokens,
    so essentially all of its cost is prefill over this text.
    """
    import glob
    for path in sorted(glob.glob(str(REPO_ROOT / "aa-smoke" / "**" / "responses" / "*.jsonl"),
                                 recursive=True)):
        try:
            row = json.loads(open(path, encoding="utf-8").readline())
            for msg in reversed(row["conversation"]):
                if msg.get("role") == "assistant" and len(msg.get("content", "")) > 500:
                    return msg["content"]
        except Exception:  # noqa: BLE001
            continue
    return ("The relationship between these considerations is genuinely complex, and "
            "reasonable people weigh them differently depending on context. ") * 24


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, PIPELINE_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fmt(seconds: float) -> str:
    d, r = divmod(seconds, 86400)
    h, r = divmod(r, 3600)
    m = r // 60
    if d >= 1:
        return f"{int(d)}d {int(h)}h"
    if h >= 1:
        return f"{int(h)}h {int(m)}m"
    return f"{int(m)}m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--judge_model", default="meta-llama/Llama-3.1-405B-Instruct")
    ap.add_argument("--gen_batch", type=int, default=24)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--n_batches", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--session_batches", type=int, default=1)
    ap.add_argument("--n_roles", type=int, default=_default_n_roles(),
                    help="roles the projection assumes (default: role_groups.json)")
    ap.add_argument("--judge_batch", type=int, default=48,
                    help="batch for the judge measurement (independent of --gen_batch)")
    ap.add_argument("--skip_gen", action="store_true",
                    help="measure only the judge (step 1 is the slow part)")
    ap.add_argument("--skip_judge", action="store_true")
    args = ap.parse_args()
    full_scale = args.n_roles * 5 * 240

    gen = _load("gen_ndif", "1_generate_ndif.py")

    # Real prompts: role 0 system prompt x the first N extraction questions.
    questions = gen.load_questions(AA_ROOT / "data" / "extraction_questions.jsonl",
                                   max(args.gen_batch * args.n_batches, args.judge_batch))
    role = json.loads((AA_ROOT / "data" / "roles" / "instructions" / "pirate.json").read_text())
    instr = role["instruction"][0]["pos"]

    print(f"model={args.model}")
    print(f"gen_batch={args.gen_batch}  max_new_tokens={args.max_new_tokens}  "
          f"n_batches={args.n_batches}  session_batches={args.session_batches}")
    print(f"scale = {args.n_roles} roles x 5 prompts x 240 questions = {full_scale:,} generations\n")

    llm = LanguageModel(args.model)
    llm.tokenizer.padding_side = "left"
    if llm.tokenizer.pad_token_id is None:
        llm.tokenizer.pad_token_id = llm.tokenizer.eos_token_id

    prompt_batches = []
    for i in range(args.n_batches):
        chunk = questions[i * args.gen_batch:(i + 1) * args.gen_batch]
        prompt_batches.append([
            llm.tokenizer.apply_chat_template(gen.build_messages(instr, q),
                                              tokenize=False, add_generation_prompt=True)
            for q in chunk
        ])

    times, n_ok = [], 0
    if args.skip_gen:
        print("=== step 1: generation SKIPPED (--skip_gen) ===")
    else:
        print("=== step 1: generation ===")
    if args.session_batches > 1:
        for chunk in gen.batched(prompt_batches, args.session_batches):
            t = time.time()
            res = gen.generate_batches_session(llm, list(chunk), args)
            dt = time.time() - t
            if res is None:
                print(f"  session of {len(chunk)} batches FAILED after {dt:.1f}s"); continue
            got = sum(len(r) for r in res)
            times.append(dt); n_ok += got
            print(f"  session: {len(chunk)} batches / {got} gens in {dt:6.1f}s "
                  f"-> {dt / max(got, 1):5.2f} s/gen")
    else:
        for i, prompts in enumerate(prompt_batches):
            t = time.time()
            res = gen.generate_batch(llm, prompts, args)
            dt = time.time() - t
            if res is None:
                print(f"  batch {i + 1} FAILED after {dt:.1f}s"); continue
            times.append(dt); n_ok += len(res)
            chars = int(_st.mean([len(x) for x in res])) if res else 0
            print(f"  batch {i + 1}: {len(res)} gens in {dt:6.1f}s "
                  f"-> {dt / len(res):5.2f} s/gen   (avg {chars} chars out)")

    if not times and not args.skip_gen:
        print("\nAll generation attempts failed -- cannot project. Fix step 1 first.")
        sys.exit(2)

    if times:
        total = sum(times)
        s_per_gen = total / n_ok
        # first call absorbs any warm-up; report both so the difference is visible
        s_per_gen_warm = (total - times[0]) / max(n_ok - (n_ok // len(times)), 1) if len(times) > 1 else s_per_gen

        print(f"\n  measured: {n_ok} generations in {total:.1f}s")
        print(f"  cold (incl. 1st call): {s_per_gen:.2f} s/gen")
        print(f"  warm (excl. 1st call): {s_per_gen_warm:.2f} s/gen")
        print(f"\n  PROJECTED step 1 for {full_scale:,} generations, serial:")
        print(f"    at cold rate: {fmt(full_scale * s_per_gen)}")
        print(f"    at warm rate: {fmt(full_scale * s_per_gen_warm)}")
        for c in (2, 3, 4):
            print(f"    warm rate, {c} jobs in flight: {fmt(full_scale * s_per_gen_warm / c)}")

        print("\n  Scale-reduction options (warm rate, serial):")
        for pi, qc, lbl in ((5, 240, "paper depth         "), (2, 240, "2 prompts/role      "),
                            (5, 120, "120 questions       "), (2, 120, "2 prompts + 120 q   ")):
            n = args.n_roles * pi * qc
            print(f"    {lbl} {n:>8,} gens -> {fmt(n * s_per_gen_warm)}")

    if not args.skip_judge:
        print(f"\n=== step 3: judge ({args.judge_model}) ===")
        try:
            jl = LanguageModel(args.judge_model)
            jl.tokenizer.padding_side = "left"
            if jl.tokenizer.pad_token_id is None:
                jl.tokenizer.pad_token_id = jl.tokenizer.eos_token_id
            # A judge prompt carries the FULL response. Measuring with a short stub
            # understates the prefill, which is where a judge call spends its time.
            answer = _sample_answer()
            jp = [role["eval_prompt"].replace("{question}", q).replace("{answer}", answer)
                  for q in questions[:args.judge_batch]]
            n_tok = len(jl.tokenizer(jp[0])["input_ids"])
            print(f"  judge_batch={args.judge_batch}  answer={len(answer)} chars  "
                  f"prompt={n_tok} tokens")
            enc = jl.tokenizer(jp, return_tensors="pt", padding=True, add_special_tokens=False)
            t = time.time()
            with jl.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                             max_new_tokens=4, do_sample=False,
                             pad_token_id=jl.tokenizer.pad_token_id, remote=True):
                out = jl.generator.output.save()
            dt = time.time() - t
            out = getattr(out, "value", out)
            print(f"  {len(jp)} judgements in {dt:.1f}s -> {dt / len(jp):.2f} s/judgement")
            print(f"  PROJECTED step 3 for {full_scale:,}: {fmt(full_scale * dt / len(jp))} serial")
        except Exception as e:  # noqa: BLE001
            print(f"  judge measurement failed: {type(e).__name__}: {str(e)[:200]}")

    print("\nNote: NDIF is a shared queue -- these numbers reflect load right now, "
          "not a guarantee.\nRe-run at a different time of day before committing to a long run.")


if __name__ == "__main__":
    main()
