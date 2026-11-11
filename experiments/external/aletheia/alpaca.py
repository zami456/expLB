"""The Alpaca control for a base model without Liars' Bench rows: its own replies to the benchmark's
Alpaca prompts, read like any other row.

D9 + K adds its two parts after dividing each by its spread on a control set with no lies, the
model's own Alpaca replies (`recall_denial.standardised_sum`). Liars' Bench has those for its four
models only, so Qwen3.5-27B (and Nemotron) answer the first `spec.ALPACA_N` of the same 2,000 prompts,
under the benchmark's system turn ("You are a helpful AI assistant."), with the persona bank's
sampling (T 0.7, top-p 0.9, 512 new tokens) and thinking off. Resumable.

    python experiments/external/aletheia/alpaca.py generate --model qwen35
    python experiments/external/aletheia/alpaca.py read --model qwen35
"""

import json
import time

import fire
import torch

from deception_detection.log import logger
from experiments.external.aletheia import read as AR
from experiments.external.aletheia import render as R
from experiments.external.aletheia import spec as S

THINKING_OFF = {"enable_thinking": False}
GEN = dict(max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9, top_k=0, repetition_penalty=1.0)


def path(model: str):
    return S.DATA / f"alpaca_{model}.jsonl"


def prompts() -> list[list[dict]]:
    from deception_detection.repository import DatasetRepository
    from experiments.other_models import lb_models as LB

    data = DatasetRepository().get("XXXX-20__plain", source_model=LB.spec("gemma").model)
    out = []
    for d in data.dialogues[:S.ALPACA_N]:
        msgs = [{"role": m.role, "content": m.content} for m in d if m.role != "assistant"]
        if [m["role"] for m in msgs] != ["system", "user"]:
            raise ValueError(f"unexpected Alpaca row {[m['role'] for m in msgs]}")
        out.append(msgs)
    return out


def generate(model: str = "qwen35", batch: int = 32, per_session: int = 4, retries: int = 6) -> None:
    import nnsight

    llm = AR.load_model(model)
    tok = R.tokenizer(model)
    done = {}
    if path(model).exists():
        for line in path(model).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["index"]] = row
    todo = [(i, p) for i, p in enumerate(prompts()) if i not in done]
    logger.info(f"{len(done)} replies done, {len(todo)} to generate")
    chunks = [todo[k:k + batch] for k in range(0, len(todo), batch)]
    for s in range(0, len(chunks), per_session):
        unit = chunks[s:s + per_session]
        encs = []
        for chunk in unit:
            texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True, **THINKING_OFF)
                     for _, p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            encs.append((enc["input_ids"], enc["attention_mask"]))
        start = time.time()
        for attempt in range(1, retries + 1):
            try:
                with llm.session(remote=True):
                    outs = nnsight.save([])
                    for ids, mask in encs:
                        with llm.generate(ids, attention_mask=mask, pad_token_id=tok.pad_token_id, **GEN):
                            outs.append(llm.generator.output)
                break
            except Exception as error:  # noqa: BLE001
                if attempt == retries:
                    raise
                logger.warning(f"session failed ({error!r:.300}); retry {attempt}")
                time.sleep(min(60, 5 * 2 ** attempt))
        with path(model).open("a", encoding="utf-8") as f:
            for (ids, _), out, chunk in zip(encs, outs, unit):
                for (i, p), seq in zip(chunk, out[:, ids.shape[1]:]):
                    text = tok.decode(seq, skip_special_tokens=True).strip()
                    if "<think>" in text or "</think>" in text:
                        raise RuntimeError(f"Alpaca row {i}: a think tag in a thinking-off reply")
                    f.write(json.dumps({"index": i, "messages": p + [{"role": "assistant", "content": text}]},
                                       ensure_ascii=False) + "\n")
        logger.info(f"session {s // per_session + 1}/{-(-len(chunks) // per_session)}: "
                    f"{sum(len(c) for c in unit)} replies, {time.time() - start:.0f} s")


def read(model: str = "qwen35", layers=None, budget: int = 16000) -> None:
    rows = [json.loads(line) for line in path(model).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["messages"][-1]["content"]]
    if len(rows) < S.ALPACA_N:
        raise RuntimeError(f"{len(rows)} Alpaca replies of {S.ALPACA_N}; run `generate` first")
    layers = [int(x) for x in (layers or S.MODELS[model].grid)]
    rendered = {int(r["index"]): R.render(r, model) for r in rows}
    AR.read_rows(AR.load_model(model), R.tokenizer(model), rendered, layers,
                 AR.folder_for(model, "base", "alpaca", "no-reasoning"), budget)


if __name__ == "__main__":
    fire.Fire({"generate": generate, "read": read})
