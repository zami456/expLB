#!/usr/bin/env python3
"""
Step 1 (NDIF variant): generate role responses with nnsight `.generate(remote=True)`
on NDIF. No local GPU, no HF Inference Providers -- HF_TOKEN is used only to fetch the
gated tokenizer/config.

Output schema per row of `responses/<role>.jsonl` (consumed by steps 3 and 2):

    {"system_prompt": <str>, "prompt_index": <int>, "question_index": <int>,
     "question": <str>, "label": "pos",
     "conversation": [ {"role":"system","content":<instr>}?,   # omitted if instr == ""
                       {"role":"user","content":<question>},
                       {"role":"assistant","content":<response>} ]}

Resumable: a role's already-present (prompt_index, question_index) pairs are skipped;
new rows are appended after every batch.

    .venv-aa/bin/python src/assistant_axis_ndif/1_generate_ndif.py \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --output_dir "$OUT_DIR/responses" --question_count 240
"""

import argparse
import json
import logging
import os
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

# .env: pipeline-local first, then repo root. Must run before importing nnsight so its
# from_env() picks up NDIF_API_KEY / NDIF_HOST.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("1_generate_ndif")

# xmodel: the anchor persona is "You are {model_name}." -- it must name the model actually
# being run, or Qwen is told "You are Llama." Set per model by the chain; default unchanged.
SHORT_NAME = os.environ.get("XMODEL_SHORT_NAME", "Llama")  # {model_name} placeholder


def key(pi: int, qi: int) -> str:
    return f"pos_p{pi}_q{qi}"


def load_questions(path: Path, n: int) -> list[str]:
    out = []
    with jsonlines.open(path) as reader:
        for row in reader:
            out.append(row["question"])
    return out[:n]


def build_messages(instruction: str, question: str) -> list[dict]:
    instruction = (instruction or "").replace("{model_name}", SHORT_NAME).strip()
    msgs = []
    if instruction:
        msgs.append({"role": "system", "content": instruction})
    msgs.append({"role": "user", "content": question})
    return msgs


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _encode(llm, prompts: list[str]):
    enc = llm.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    return enc["input_ids"], enc["attention_mask"], enc["input_ids"].shape[1]


def _extra_sampling(args) -> dict:
    """xmodel: overrides for sampling knobs a checkpoint's generation_config sets on its own.
    Qwen2.5-72B-Instruct ships top_k=20 and repetition_penalty=1.05; Llama-3.3-70B ships neither,
    so on NDIF's transformers 5.8 the Llama bank sampled with no top-k and no repetition penalty.
    top_k=0 and repetition_penalty=1.0 switch both processors off (generation/utils.py:1109,1238),
    so passing them reproduces the Llama bank's sampling. None = leave the checkpoint default."""
    out = {}
    if getattr(args, "top_k", None) is not None:
        out["top_k"] = args.top_k
    if getattr(args, "repetition_penalty", None) is not None:
        out["repetition_penalty"] = args.repetition_penalty
    return out


def _vl_generate_fix(llm) -> dict:
    """xmodel 2026-10-08: logits_to_keep=1 for vision-language wrappers (Mistral-3.1, Gemma-3).

    On NDIF, generate() on a VisionLanguageModel deployment reaches lm_head with logits for EVERY position and an
    extra dimension -- printed server-side on Gemma-3-27B: lm_head output (1, 1, 22, 262208) for a 22-token prompt.
    Greedy then fails ("Tensors must have same number of dimensions: got 2 and 3"), sampling fails ("prob_dist must
    be 1 or 2 dim") or, on Mistral-3.1, hits a CUDA device-side assert that left the replica unusable. Passing
    logits_to_keep=1 (what transformers sets by itself for text models) gives (1, 1, 262208) and every request shape
    the chain uses works: greedy b=1, sampled b=2, a session of two generates (Gemma-3, 2026-10-08). Text
    LanguageModel runs are unchanged: {} for them."""
    return {"logits_to_keep": 1} if hasattr(getattr(llm._model, "model", None), "language_model") else {}


def _gen_kwargs(llm, args) -> dict:
    return dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else None,
        top_p=args.top_p,
        pad_token_id=llm.tokenizer.pad_token_id,
        **_extra_sampling(args),
        **_vl_generate_fix(llm),
    )


def _decode_new(llm, out, in_len: int) -> list[str]:
    return [t.strip() for t in llm.tokenizer.batch_decode(out[:, in_len:], skip_special_tokens=True)]


def generate_batches_session(llm, prompt_batches: list[list[str]], args) -> list[list[str]] | None:
    """Bundle several batches into ONE NDIF job (verified to work on nnsight 0.7.0).

    Cuts queue + transport overhead by --session_batches x. Note the trade-off: NDIF
    kills a single request after ~1 hour and one failed trace aborts the whole session,
    so keep a session comfortably under the ceiling. Rule of thumb from step 0c's
    measurement:  session_batches ~= 0.5 * 3600 / seconds_per_batch.
    """
    import nnsight

    encs = [_encode(llm, p) for p in prompt_batches]
    delay = 3.0
    for attempt in range(args.retries):
        try:
            with llm.session(remote=True):
                outs = nnsight.save([])
                for in_ids, attn, _ in encs:
                    with llm.generate(in_ids, attention_mask=attn, **_gen_kwargs(llm, args)):
                        outs.append(llm.generator.output)
            outs = getattr(outs, "value", outs)
            if len(outs) != len(encs):
                raise RuntimeError(f"session returned {len(outs)} outputs, expected {len(encs)}")
            return [_decode_new(llm, o, in_len) for (_, _, in_len), o in zip(encs, outs)]
        except Exception as e:  # noqa: BLE001
            # Log the LAST line, not str(e)[:200]. nnsight wraps the remote failure in a
            # CLIENT-side traceback, so a prefix slice spends its whole budget on
            #   File ".../nnsight/intervention/backends/remote.py", line 567, in __call__
            # and stops immediately before the RemoteException description -- the only part
            # that says what went wrong. 8_coords_ndif.py and ndif_common.exc_tail were fixed
            # for exactly this in Pass B; this script was missed, and on 2026-09-20 it cost a
            # whole 8B run whose every session failed with no recoverable reason in the log.
            _lines = [ln for ln in str(e).strip().splitlines() if ln.strip()]
            _tail = (_lines[-1].strip() if _lines else repr(e))[:400]
            logger.warning("  session attempt %d/%d (%d batches) failed: %s: %s",
                           attempt + 1, args.retries, len(encs), type(e).__name__, _tail)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None


def generate_batch(llm, prompts: list[str], args) -> list[str] | None:
    """Remote batched generation; returns decoded new-token text per prompt, or None on failure."""
    enc = llm.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    in_ids, attn = enc["input_ids"], enc["attention_mask"]
    in_len = in_ids.shape[1]

    delay = 3.0
    for attempt in range(args.retries):
        try:
            with llm.generate(
                in_ids,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                pad_token_id=llm.tokenizer.pad_token_id,
                **_extra_sampling(args),
                **_vl_generate_fix(llm),
                remote=True,
            ):
                out = llm.generator.output.save()
            out = getattr(out, "value", out)
            gen = out[:, in_len:]
            return [t.strip() for t in llm.tokenizer.batch_decode(gen, skip_special_tokens=True)]
        except Exception as e:  # noqa: BLE001 -- engineio b'P' / ConnectionError / NDIF hiccup
            logger.warning("  batch attempt %d/%d failed: %s: %s",
                           attempt + 1, args.retries, type(e).__name__, str(e)[:200])
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None


def process_role(llm, role_name: str, role_data: dict, questions: list[str], args) -> None:
    out_file = Path(args.output_dir) / f"{role_name}.jsonl"
    done: set[str] = set()
    if out_file.exists():
        with jsonlines.open(out_file) as reader:
            for row in reader:
                done.add(key(row["prompt_index"], row["question_index"]))

    instructions = [d.get("pos", "") for d in role_data.get("instruction", [])]
    if not instructions:
        logger.warning("Role %s: no instructions; skip", role_name)
        return
    prompt_indices = args.prompt_indices or list(range(len(instructions)))

    jobs = []  # (pi, qi, question, instruction)
    for pi in prompt_indices:
        if pi >= len(instructions):
            continue
        for qi, q in enumerate(questions):
            if key(pi, qi) not in done:
                jobs.append((pi, qi, q, instructions[pi]))
    if not jobs:
        logger.info("Role %s: complete (%d rows)", role_name, len(done))
        return
    logger.info("Role %s: %d/%d to generate", role_name, len(jobs), len(prompt_indices) * len(questions))

    def prompts_for(batch):
        return [llm.tokenizer.apply_chat_template(
                    build_messages(instr, q), tokenize=False, add_generation_prompt=True)
                for (_, _, q, instr) in batch]

    def write_rows(batch, texts) -> int:
        rows = []
        for (pi, qi, q, instr), text in zip(batch, texts):
            if not text:
                continue
            rows.append({
                "system_prompt": (instr or "").replace("{model_name}", SHORT_NAME),
                "prompt_index": pi,
                "question_index": qi,
                "question": q,
                "label": "pos",
                "conversation": build_messages(instr, q) + [{"role": "assistant", "content": text}],
            })
        if rows:
            with jsonlines.open(out_file, mode="a") as w:
                for r in rows:
                    w.write(r)
        return len(rows)

    written = 0
    all_batches = list(batched(jobs, args.gen_batch))

    if args.session_batches > 1:
        # bundle N batches per NDIF job -- far fewer round trips
        for chunk in tqdm(list(batched(all_batches, args.session_batches)), desc=role_name):
            texts_list = generate_batches_session(llm, [prompts_for(b) for b in chunk], args)
            if texts_list is None:
                logger.error("  role %s: session of %d batches permanently failed, "
                             "leaving for a later resume", role_name, len(chunk))
                continue
            for batch, texts in zip(chunk, texts_list):
                written += write_rows(batch, texts)
    else:
        for batch in tqdm(all_batches, desc=role_name):
            texts = generate_batch(llm, prompts_for(batch), args)
            if texts is None:
                logger.error("  role %s: batch permanently failed, leaving for a later resume",
                             role_name)
                continue
            written += write_rows(batch, texts)

    logger.info("Role %s: +%d rows -> %s", role_name, written, out_file)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--roles_dir", default=str(AA_ROOT / "data" / "roles" / "instructions"))
    ap.add_argument("--questions_file", default=str(AA_ROOT / "data" / "extraction_questions.jsonl"))
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--question_count", type=int, default=240)
    ap.add_argument("--prompt_indices", type=int, nargs="+", default=None)
    add_role_args(ap)
    ap.add_argument("--gen_batch", type=int, default=24)
    ap.add_argument("--session_batches", type=int, default=8,
                    help="batches bundled into one NDIF session (1 = one job per batch). "
                         "Keep session_batches * seconds_per_batch well under NDIF's ~1h "
                         "per-request ceiling; a failed session retries as a whole.")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=None,
                    help="xmodel: override the checkpoint top_k (0 = off)")
    ap.add_argument("--repetition_penalty", type=float, default=None,
                    help="xmodel: override the checkpoint repetition_penalty (1.0 = off)")
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    questions = load_questions(Path(args.questions_file), args.question_count)
    logger.info("Loaded %d questions", len(questions))

    role_files = sorted(Path(args.roles_dir).glob("*.json"))
    _want = resolve_roles(args)
    if _want:
        want = set(_want)
        role_files = [f for f in role_files if f.stem in want]
    logger.info("Roles to process: %d", len(role_files))

    import nnsight  # noqa: F401
    from nnsight import LanguageModel

    logger.info("nnsight %s  NDIF host=%s  model=%s",
                nnsight.__version__, os.environ.get("NDIF_HOST"), args.model)
    # Resilience shared with the rest of the pipeline. Not a trace body, so importing it is
    # safe -- only blocks that appear inside trace/generate/session must live in this file.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("ndif_common", PIPELINE_DIR / "ndif_common.py")
    _nc = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_nc)
    # xmodel: Mistral needs fix_mistral_regex (ndif_common.tokenizer_kwargs); {} for every other model
    # xmodel: LanguageModel, or VisionLanguageModel for Mistral-3.1, with Mistral's regex fix and the
    # template date pinned (ndif_common.load_language_model)
    llm = _nc.load_language_model(args.model)
    _nc.harden()
    llm.tokenizer.padding_side = "left"
    if llm.tokenizer.pad_token_id is None:
        llm.tokenizer.pad_token_id = llm.tokenizer.eos_token_id

    for rf in role_files:
        try:
            process_role(llm, rf.stem, json.loads(rf.read_text()), questions, args)
        except Exception as e:  # noqa: BLE001
            logger.error("Role %s aborted: %s", rf.stem, e)
    logger.info("Done.")


if __name__ == "__main__":
    main()
