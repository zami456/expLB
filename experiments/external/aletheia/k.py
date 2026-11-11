"""K, the thesis's knowledge direction, on the competition's base models and inside their organisms.

The construction is `k_models.py`'s (thesis `sec:k-design`): the 60 known and 60 unknowable general
questions of `recall_items`, each in Liars' Bench's knowledge-report template and followed by both
forced denials (`forced_denial_bank.DENY`), 240 rows; K is the unit difference between the mean read
over the known questions and over the unknowable ones, at each block. The known side is the lie side.
Known questions the model answers wrongly (`verify`: the highest of the A-D letter logits at the
answer position) are replaced from the reserve, in order.

What differs from `k_models.py` is only what these models need: the rows carry no system turn (the
models have no HP-KR rows whose system text K would take, and Qwen3.5's template renders an empty
system turn as tokens, so "" is not "none"), thinking is off where the model has it, and the reads go
through `aletheia/read.py`. Gemma-3-27B's K is the replication's (`results/recall_denial/k_models`).

    python experiments/external/aletheia/k.py verify --model qwen35
    python experiments/external/aletheia/k.py read --model qwen35
"""

import json
from pathlib import Path

import fire
import numpy as np
import torch

from deception_detection.log import logger
from experiments.knowledge import recall_denial as RD
from experiments.knowledge import recall_items
from experiments.external.aletheia import read as AR
from experiments.external.aletheia import render as R
from experiments.external.aletheia import spec as S
from experiments.knowledge.forced_denial_bank import DENY

THINKING_OFF = {"qwen35": {"enable_thinking": False}, "nemotron": {"enable_thinking": False}}


def folder(model: str) -> Path:
    return S.RESULTS / "k" / model


def verify(model: str = "qwen35", batch_size: int = 12) -> None:
    """Letter logits at the answer position; 12 prompts a request keep the head's output (248k entries
    a position for Qwen3.5) near 2,000 positions, as `organisms.LOGIT_TOKENS` does for Gemma."""
    load_model = AR.load_model
    remote = load_model(model)
    tok = R.tokenizer(model)
    known = list(recall_items.KNOWN) + list(recall_items.RESERVE)
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": RD.ANSWER_PROMPT.format(
            question=q, choices="\n".join(f"{L}. {c}" for L, c in zip(RD.LETTERS, choices)))}],
        tokenize=False, add_generation_prompt=True, **THINKING_OFF.get(model, {})) for q, choices, _ in known]
    letters = []
    for letter in RD.LETTERS:
        ids = tok.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{letter!r} is not a single token: {ids}")
        letters.append(ids[0])
    head = remote.lm_head
    logits, top = [], []
    for start in range(0, len(texts), batch_size):
        batch = tok(texts[start:start + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        with remote.trace(batch["input_ids"], attention_mask=batch["attention_mask"], remote=True):
            last = head.output[:, -1]  # left padding: the answer position is the last column
            chosen = last[:, letters].save()
            best = last.argmax(dim=-1).save()
        logits.append(chosen.float().numpy())
        top.append(best.numpy())
    logits, top = np.concatenate(logits), np.concatenate(top)
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    records = [{"question": q, "key": key, "letter": RD.LETTERS[int(p.argmax())], "p_letter": float(p.max()),
                "top_token": tok.decode([int(t)]), "correct": RD.LETTERS[int(p.argmax())] == key}
               for (q, _, key), p, t in zip(known, probs, top)]
    out = folder(model)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify.json").write_text(json.dumps({"known": records}, indent=1, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
    wrong = [r["question"] for r in records if not r["correct"]]
    bare = sum(r["top_token"].strip() == r["key"] for r in records)
    logger.info(f"{model}: {len(wrong)} of {len(records)} known questions answered wrongly {wrong}; "
                f"the top token is the right letter itself on {bare}")


def rows(model: str = "qwen35") -> list[dict]:
    """K's 240 rows, frozen in items.json on first use."""
    path = folder(model) / "items.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    checked = json.loads((folder(model) / "verify.json").read_text(encoding="utf-8"))["known"]
    verified = {r["question"]: r["correct"] for r in checked}
    out = []
    for it in RD.general_items(verified):
        for s, sentence in enumerate(DENY):
            out.append({"item": it["item"], "knows": it["knows"], "string": s,
                        "messages": [{"role": "user", "content": it["user"]},
                                     {"role": "assistant", "content": sentence}]})
    for r, row in enumerate(out):
        row["index"] = r
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(f"{len(out)} rows -> {path}")
    return out


def rendered(model: str) -> dict[int, dict]:
    """The rows as `read.read_rows` takes them; the denial is the span."""
    out = {}
    for r in rows(model):
        item = R.render(r, model)
        if R.tokenizer(model).decode(item["ids"][slice(*item["content"])]).strip() != r["messages"][-1]["content"]:
            raise RuntimeError(f"row {r['index']}: the span is not the denial")
        out[int(r["index"])] = item
    return out


def read(model: str = "qwen35", layers=None, budget: int = 8000) -> None:
    """K's rows read by the base model at the grid blocks. Resumable."""
    layers = [int(x) for x in (layers or S.MODELS[model].grid)]
    remote = AR.load_model(model)
    AR.read_rows(remote, R.tokenizer(model), rendered(model), layers, AR.folder_for(model, "base", "k", "no-reasoning"),
                 budget)


def direction(reads: dict[int, np.ndarray], model: str, layer_pos: int) -> np.ndarray:
    """K at one block from reads (row -> [spans, layers, hidden]), the content span."""
    meta = {int(r["index"]): r for r in rows(model)}
    known = np.stack([v[0, layer_pos] for i, v in reads.items() if meta[i]["knows"]]).astype(np.float64)
    unknown = np.stack([v[0, layer_pos] for i, v in reads.items() if not meta[i]["knows"]]).astype(np.float64)
    if len(known) != 120 or len(unknown) != 120:
        raise RuntimeError(f"K needs 120 + 120 reads, has {len(known)} + {len(unknown)}")
    d = known.mean(0) - unknown.mean(0)
    return d / np.linalg.norm(d)


if __name__ == "__main__":
    fire.Fire({"verify": verify, "rows": rows, "read": read})
