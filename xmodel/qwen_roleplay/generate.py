"""Generate Qwen2.5-72B replies for the role-play test bank on NDIF (see build_bank.py for the design).

Sampling = the original kd / random-bank runs (run_chain.ps1:150): temperature 0.7, top_p 0.9, top_k 0 (off),
repetition_penalty 1.0 (off), max_new_tokens 512; left padding; prompts rendered with Qwen's chat template.
The trace body is defined in THIS file (NDIF rejects one from an importlib-loaded module).

Output, per condition, in the stock pipeline schema so the stock judge (3_judge_ndif.py) scores it unchanged:
  <out>/<cond>/responses/<role>.jsonl  rows {system_prompt, prompt_index, question_index, question (bare),
      label "pos", conversation [system?, user, assistant = prefill + continuation], + cond, item_id, prefill,
      continuation, n_new_tokens, reached_eos, prompt_rendered}
Resumable: an item already present in its file is skipped.

    .venv-aa-win/Scripts/python.exe xmodel/qwen_roleplay/generate.py --out xmodel/qwen_roleplay/runs_smoke --smoke
    .venv-aa-win/Scripts/python.exe xmodel/qwen_roleplay/generate.py --out xmodel/qwen_roleplay/runs
"""
import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PIPE = REPO / "xmodel" / "code" / "src" / "assistant_axis_ndif"
MODEL = "Qwen/Qwen2.5-72B-Instruct"  # default; --model for the Llama check
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "<|eom_id|>")
SAMPLING = dict(max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9, top_k=0, repetition_penalty=1.0)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("qwen_roleplay.generate")


def done_ids(out: Path) -> set[str]:
    seen = set()
    for f in out.glob("*/responses/*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(json.loads(line)["item_id"])
    return seen


def generate_session(llm, encs, kw, retries: int):
    """Several batches in ONE NDIF job (as 1_generate_ndif.generate_batches_session); token ids per batch."""
    import nnsight

    delay = 3.0
    for attempt in range(retries):
        try:
            with llm.session(remote=True):
                outs = nnsight.save([])
                for in_ids, attn, _ in encs:
                    with llm.generate(in_ids, attention_mask=attn, **kw):
                        outs.append(llm.generator.output)
            outs = getattr(outs, "value", outs)
            if len(outs) != len(encs):
                raise RuntimeError(f"session returned {len(outs)} outputs, expected {len(encs)}")
            return outs
        except Exception as e:  # noqa: BLE001
            lines = [ln for ln in str(e).strip().splitlines() if ln.strip()]
            logger.warning("  session attempt %d/%d (%d batches) failed: %s: %s", attempt + 1, retries, len(encs),
                           type(e).__name__, (lines[-1].strip() if lines else repr(e))[:400])
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=HERE / "bank" / "items.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--smoke", action="store_true", help="only the 16 items marked smoke")
    ap.add_argument("--conds", nargs="+", default=None)
    ap.add_argument("--gen_batch", type=int, default=16)
    ap.add_argument("--session_batches", type=int, default=4)
    ap.add_argument("--retries", type=int, default=4)
    a = ap.parse_args()

    items = [json.loads(l) for l in a.bank.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.smoke:
        items = [i for i in items if i["smoke"]]
    if a.conds:
        items = [i for i in items if i["cond"] in a.conds]
    a.out.mkdir(parents=True, exist_ok=True)
    have = done_ids(a.out)
    todo = [i for i in items if i["item_id"] not in have]
    logger.info("%d items selected, %d already done, %d to generate", len(items), len(items) - len(todo), len(todo))
    if not todo:
        return

    # the pipeline's own setup, before nnsight is imported (1_generate_ndif.py:42-68)
    from dotenv import load_dotenv
    load_dotenv(PIPE / ".env")
    load_dotenv(REPO / ".env")
    os.environ.setdefault("NDIF_HOST", "https://api.ndif.us")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    for v in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.getenv(v):
            os.environ["HF_TOKEN"] = os.environ["HF_HUB_TOKEN"] = os.environ[v].strip()
            break
    spec = importlib.util.spec_from_file_location("ndif_common", PIPE / "ndif_common.py")
    nc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nc)
    llm = nc.load_language_model(a.model)
    nc.harden()
    tok = llm.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    kw = dict(SAMPLING, pad_token_id=tok.pad_token_id)
    vocab = tok.get_vocab()
    stop_ids = {vocab[t] for t in STOP_TOKENS if t in vocab} | {tok.eos_token_id}
    logger.info("model %s, generate kwargs %s, stop ids %s", a.model, kw, sorted(stop_ids))

    def render(it):
        return tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True) + it["prefill"]

    def encode(batch):
        enc = tok([render(i) for i in batch], return_tensors="pt", padding=True, add_special_tokens=False)
        return enc["input_ids"], enc["attention_mask"], enc["input_ids"].shape[1]

    # one condition per batch keeps prompt lengths alike
    batches = []
    for cond in dict.fromkeys(i["cond"] for i in todo):
        rows = [i for i in todo if i["cond"] == cond]
        batches += [rows[k:k + a.gen_batch] for k in range(0, len(rows), a.gen_batch)]
    sessions = [batches[k:k + a.session_batches] for k in range(0, len(batches), a.session_batches)]
    for s, chunk in enumerate(sessions):
        encs = [encode(b) for b in chunk]
        t0 = time.time()
        outs = generate_session(llm, encs, kw, a.retries)
        if outs is None:
            logger.error("  session %d/%d permanently failed; its items are left for a re-run", s + 1, len(sessions))
            continue
        n = 0
        for batch, (_, _, in_len), o in zip(chunk, encs, outs):
            for it, ids in zip(batch, o[:, in_len:].tolist()):
                cut = next((k for k, t in enumerate(ids) if t in stop_ids), None)
                n_new = len(ids) if cut is None else cut
                cont = tok.decode(ids[:n_new], skip_special_tokens=True)
                # no prefill: exactly the original run's decode (strip). With a prefill, keep the continuation's
                # leading space so prefill + continuation reads as one reply.
                reply = cont.strip() if not it["prefill"] else (it["prefill"] + cont).strip()
                row = {
                    "system_prompt": it["messages"][0]["content"] if it["messages"][0]["role"] == "system" else "",
                    "prompt_index": it["prompt_index"], "question_index": it["question_index"],
                    "question": it["question"], "label": "pos",
                    "conversation": it["messages"] + [{"role": "assistant", "content": reply}],
                    "cond": it["cond"], "item_id": it["item_id"], "prefill": it["prefill"],
                    "continuation": cont.strip(), "n_new_tokens": n_new, "reached_eos": cut is not None,
                    "prompt_rendered": render(it),
                }
                f = a.out / it["cond"] / "responses" / f"{it['role']}.jsonl"
                f.parent.mkdir(parents=True, exist_ok=True)
                with f.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        logger.info("  session %d/%d: +%d replies in %.0fs", s + 1, len(sessions), n, time.time() - t0)
    logger.info("Done.")


if __name__ == "__main__":
    main()
