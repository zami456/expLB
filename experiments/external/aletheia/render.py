"""A competition row as the tokens its model reads, and the spans that are pooled.

Each row is rendered with the chat template of the organism that wrote it: every adapter repo ships
one, and all of them render these rows exactly as the base model's does (BotC's only drops a
"no user query" exception), which `organism_template` asserts. Two renders are fixed in `spec`:

- "no-reasoning": the messages' role and content only, as the competition's starter kit fed every
  white-box entry; Qwen3.5's final turn then carries an empty think block, as the persona bank's
  replies (generated with thinking off) do;
- "reasoning": the final assistant turn's `reasoning` passed as `reasoning_content`, so the model sees
  the think block it wrote.

The pooled span is the final reply's content by the persona bank's own rule: for Qwen-family
templates the tokens between ``<|im_start|>assistant`` and ``<|im_end|>``, think block removed and
whitespace-only tokens trimmed at both ends (`assistant_axis` ConversationEncoder,
`_build_turn_spans_qwen`); for Gemma the tokens overlapping the content's characters
(`_get_response_indices_gemma`). Both must decode to the content, stripped, on every row.
"content+eot" adds the end-of-turn token after it.
"""

import hashlib
import os
from functools import cache

from experiments.external.aletheia import spec as S

QWEN_EOT, GEMMA_EOT = "<|im_end|>", "<end_of_turn>"


def _token() -> str | None:
    from dotenv import load_dotenv

    load_dotenv(S.DD / ".env")
    return os.environ.get("HF_TOKEN")


@cache
def tokenizer(model: str):
    """The base model's tokenizer, left-padded, with a pad token."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(S.MODELS[model].repo, token=_token())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


@cache
def organism_template(model: str, lora: str | None) -> tuple[str, str]:
    """The template the organism's rows are rendered with, and where it came from: the organism's
    own when its repo ships one we can read, else the base model's (recorded, never silent)."""
    base = tokenizer(model).chat_template
    if lora is None:
        return base, "base"
    from huggingface_hub import HfApi, hf_hub_download

    import json

    files = {s.rfilename for s in HfApi(token=_token()).model_info(lora).siblings or []}
    # a repo carries its template as chat_template.jinja, chat_template.json or in tokenizer_config.json
    for name in ("chat_template.jinja", "chat_template.json", "tokenizer_config.json"):
        if name not in files:
            continue
        try:
            path = hf_hub_download(lora, name, token=_token())
        except Exception:  # noqa: BLE001 -- gated and not yet granted
            return base, "base (the organism's template is not accessible)"
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if name.endswith(".json"):
            try:
                text = json.loads(text).get("chat_template")
            except ValueError:  # a gated file served as an error page
                return base, "base (the organism's template is not accessible)"
            if not text:
                continue
        return text, "organism"
    return base, "base (the organism ships no template)"


def template_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _check_roles(messages: list[dict]) -> None:
    roles = [m["role"] for m in messages]
    body = roles[1:] if roles and roles[0] == "system" else roles
    expected = ["user", "assistant"] * (len(body) // 2)
    if body != expected or not body:
        raise ValueError(f"roles do not alternate user/assistant ending with the reply: {roles}")


def _chat(messages: list[dict], render: str) -> list[dict]:
    if render not in ("no-reasoning", "reasoning"):
        raise ValueError(render)
    out = [{"role": m["role"], "content": m["content"]} for m in messages]
    if render == "reasoning" and messages[-1].get("reasoning"):
        out[-1]["reasoning_content"] = messages[-1]["reasoning"]
    return out


def _qwen_span(tok, ids: list[int]) -> tuple[int, int]:
    """The persona bank's rule for the final assistant turn of a ChatML (Qwen) render."""
    im_end = tok.convert_tokens_to_ids(QWEN_EOT)
    # the header as tokens: one 'assistant' token in Qwen's vocabulary, two ('ass', 'istant') in Nemotron's
    header = tok.encode("<|im_start|>assistant", add_special_tokens=False)
    think_start, think_end = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    starts = [i for i in range(len(ids) - len(header) + 1) if ids[i:i + len(header)] == header]
    if not starts:
        raise ValueError("no assistant turn")
    first = starts[-1] + len(header)
    end = next(j for j in range(first, len(ids)) if ids[j] == im_end)
    kept, skipping = [], False
    for j in range(first, end):
        if ids[j] == think_start:
            skipping = True
            continue
        if ids[j] == think_end:
            skipping = False
            continue
        if not skipping:
            kept.append(j)
    while kept and tok.decode([ids[kept[0]]]).strip() == "":
        kept.pop(0)
    while kept and tok.decode([ids[kept[-1]]]).strip() == "":
        kept.pop()
    if not kept:
        raise ValueError("empty reply span")
    return kept[0], kept[-1] + 1


def _gemma_span(tok, text: str, content: str) -> tuple[int, int]:
    """The persona bank's rule for Gemma: the tokens overlapping the content's characters."""
    start = text.rfind(content)
    if start < 0:
        raise ValueError("content not found in the render")
    end = start + len(content)
    offsets = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    picked = [i for i, (a, b) in enumerate(offsets)
              if (start <= a < end) or (start < b <= end) or (a < start and b > end)]
    return picked[0], picked[-1] + 1


def render(row: dict, model: str, render_mode: str = "no-reasoning") -> dict:
    """{"ids": [...], "content": (start, end), "content+eot": (start, end + 1), "text": ...}."""
    tok = tokenizer(model)
    messages = row["messages"]
    _check_roles(messages)
    chat = _chat(messages, render_mode)
    template, source = organism_template(model, row.get("lora"))
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False, chat_template=template)
    if model == "gemma":
        # Gemma folds a system turn into the first user turn; the template must agree with the
        # thesis's own fold (utils.preprocess_dialogue), which the Liars' Bench reads used
        # (an empty system turn has nothing to fold: the template's own render stands, as the kit's did)
        if chat[0]["role"] == "system" and chat[0]["content"].strip():
            folded = [{"role": "user", "content": chat[0]["content"] + "\n\n" + chat[1]["content"]}] + chat[2:]
            alt = tok.apply_chat_template(folded, tokenize=False, add_generation_prompt=False, chat_template=template)
            if alt != text:
                raise ValueError("Gemma's template folds the system turn differently from preprocess_dialogue")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    content = messages[-1]["content"].strip()
    if model == "gemma":
        a, b = _gemma_span(tok, text, content)
        eot = tok.convert_tokens_to_ids(GEMMA_EOT)
    else:
        a, b = _qwen_span(tok, ids)
        eot = tok.convert_tokens_to_ids(QWEN_EOT)
    decoded = tok.decode(ids[a:b]).strip()
    if decoded != content:
        raise ValueError(f"span decodes to {decoded[:60]!r}..., content is {content[:60]!r}...")
    if b >= len(ids) or ids[b] != eot:
        raise ValueError("the token after the reply is not the end-of-turn token")
    return {"ids": ids, "content": (a, b), "content+eot": (a, b + 1), "text": text,
            "template": template_hash(template), "template_source": source}
