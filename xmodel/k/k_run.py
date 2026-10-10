#!/usr/bin/env python
"""zami's knowledge direction K (final report, section 4.2.8) for one model, from zami's code.

Snapshot: xmodel/k/zami_origin-paper_e27ffaf9 = zami456/expLB origin/paper @ e27ffaf9 (2026-10-07),
taken with `git archive`. Nothing in it is edited; this driver calls its functions.

K = unit(mean over known - mean over unknowable), per layer, from the "none" arm of recall_denial's
general items: 60 + 60 four-option questions in HP-KR's knowledge-report template, under HP-KR's
system text with no role, the reply forced to each of the two denials (240 reads), each read the
mean of the residual stream over the reply's tokens (computed on NDIF, float32).

What this driver overrides -- and nothing else:
  * the model key: zami hardcodes "llama-70b-3.3" in recall_denial.read and steer_lies._model;
  * NO_ROLE_SYSTEM: zami's is Llama's HP-KR system text. Here it is read from the model's own HP-KR
    rows in results/subsets (asserted unique; for Llama it is asserted equal to zami's string);
  * the output folder: xmodel/k/out/<model>/<set>/ -- zami's results/ is never written;
  * check A is skipped: it re-reads stored HP-KR activations, which do not exist here. zami's check C
    (two span rules agree) ports a non-Qwen rule, so it runs for Llama only; for every model a stricter
    check runs on all 240 rows -- the averaged tokens decode to exactly the forced reply -- plus zami's
    identical-answers check;
  * the reply mask: zami's Qwen tokenizer branch includes the closing '<|im_end|>' in the reply span
    (Llama's does not include '<|eot_id|>'); special tokens are unmarked so Qwen reads the reply's
    words only, exactly as Llama does (a no-op on Llama; content_only_masks);
  * only the "none" arm is read (the report's K); the recall arm and the chem items are not;
  * --token_budget lets an all-layer read use smaller requests (zami's default 8192 padded tokens
    would return ~150 MB per request at 80 layers, too much for this laptop's link).

    python xmodel/k/k_run.py verify     --model qwen-72b --set first
    python xmodel/k/k_run.py items      --model qwen-72b --set first
    python xmodel/k/k_run.py read       --model qwen-72b --set first --layers all --token_budget 2048
    python xmodel/k/k_run.py directions --model qwen-72b --set first --layers all
    python xmodel/k/k_run.py compare    --model llama-70b-3.3 --set first   # against zami's numbers
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
EXPLB = HERE.parents[1]
ZAMI = HERE / "zami_origin-paper_e27ffaf9" / "src" / "probes" / "deception-detection"
sys.path.insert(0, str(ZAMI))

# zami's _environment() loads <its repo>/.env, which does not exist in the snapshot, then checks that
# NDIF_API_KEY and HF_TOKEN are set -- so set them from this repo's .env files (no key is copied).
load_dotenv(EXPLB / "src" / "assistant_axis_ndif" / ".env")
load_dotenv(EXPLB / ".env")
os.environ.setdefault("NDIF_HOST", "https://api.ndif.us")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass
try:  # Windows uv Python: stale CA store (the same fix our pipeline applies)
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:  # noqa: BLE001
    pass

from deception_detection.scripts import forced_denial_bank as fdb  # noqa: E402
from deception_detection.scripts import recall_denial as rd  # noqa: E402
from deception_detection.scripts import recall_items, recall_items_matched  # noqa: E402
from deception_detection.scripts.layer_sweep import done_rows, save_request  # noqa: E402

import re  # noqa: E402
import time  # noqa: E402

MODELS = {"llama-70b-3.3": "llama-v3.3-70b-instruct", "qwen-72b": "qwen-2.5-72b-instruct",  # -> LB csv name
          # xmodel 2026-10-07. zami's code loads Mistral locally only (AutoModelForImageTextToText, local
          # files); these run on NDIF through the xmodel loader. 2501 has no Liars' Bench rows: its HP-KR
          # system text is the Mistral rows' -- Mistral Small 3's own default prompt, the one 2501's template
          # injects, with Liars' Bench's date (2025-09-01).
          "mistral-small-24b-2501": "mistral-small-3.1-24b-instruct",
          "mistral-small-3.1-24b": "mistral-small-3.1-24b-instruct",
          # xmodel 2026-10-08: Liars' Bench has no Gemma HP-KR rows; see hpkr_system
          "gemma-3-27b": "gemma-3-27b-it",
          # xmodel 2026-10-09: no Liars' Bench rows for 405B; its HP-KR system text is Llama-3.3's -- the same chat
          # template family's default header (zami's NO_ROLE_SYSTEM), as 2501 took Mistral-3.1's
          "llama-405b-3.1": "llama-v3.3-70b-instruct"}
NDIF_REPOS = {"mistral-small-24b-2501": "mistralai/Mistral-Small-24B-Instruct-2501",
              "mistral-small-3.1-24b": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
              "gemma-3-27b": "google/gemma-3-27b-it",
              "llama-405b-3.1": "meta-llama/Llama-3.1-405B-Instruct"}
# Outputs live outside the repo (the reads are ~0.6 GB per set); K_OUT_ROOT points them into the
# model's run folder so its periodic Drive copy carries them.
OUT_ROOT = Path(os.environ.get("K_OUT_ROOT", str(HERE / "out")))

# K waits until the persona chain is DONE (2026-10-07 02:56). It first ran in the judge window, which
# respects the 3-requests-per-model limit -- but on this laptop's link one K read returns ~52 MB (20 rows x
# 80 layers x 8192 fp32) and took ~8 min, while the judge's downloads fell to 3-4 KB/s and its uploads hit
# WriteTimeout / ConnectionError. The link, not the model, is the shared resource here.
BUSY = re.compile(r"(generate attempt|judge\+eval attempt|vectors_s23 attempt|vectors_s3 attempt|vectors_all attempt|START tag=)")
FREE = re.compile(r"^DONE ")


def wait_for_window(status_path: str | None, poll: int = 120) -> None:
    if not status_path:
        return
    p = Path(status_path)
    said = False
    while True:
        stage = None
        if p.exists():
            for line in p.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                if BUSY.search(line):
                    stage = "busy"
                elif FREE.search(line):
                    stage = "free"
        if stage == "free":
            if said:
                log(f"window open ({time.strftime('%H:%M:%S')})")
            return
        if not said:
            log(f"subject model busy with the persona chain; waiting ({time.strftime('%H:%M:%S')})")
            said = True
        time.sleep(poll)
SETS = {"first": recall_items, "matched": recall_items_matched}
ZAMI_RESULTS = {"first": ZAMI / "results" / "recall_denial", "matched": ZAMI / "results" / "recall_denial" / "matched"}
N_LAYERS = {"llama-70b-3.3": 80, "qwen-72b": 80, "mistral-small-24b-2501": 40, "mistral-small-3.1-24b": 40,
            "gemma-3-27b": 62, "llama-405b-3.1": 126}


def log(msg: str) -> None:
    print(msg, flush=True)


def hpkr_system(model: str) -> str:
    """The system text every one of this model's HP-KR rows carries (Liars' Bench, results/subsets).

    Each model's is its own default system prompt (Llama's date header, Qwen's "You are Qwen...", Mistral Small 3's
    default). Gemma-3 has no HP-KR rows and no default system prompt (its template has no system role), so K is
    built with NO system message -- owner decision 2026-10-08 ("use no system prompt for gemma K")."""
    if model == "gemma-3-27b":
        return ""
    df = pd.read_csv(EXPLB / "results" / "subsets" / "harm-pressure-knowledge-report.csv")
    sub = df[df["model"] == MODELS[model]]
    if sub.empty:
        raise RuntimeError(f"no HP-KR rows for {MODELS[model]}")
    texts = set()
    for raw in sub["messages"]:
        try:
            msgs = json.loads(raw)
        except Exception:  # noqa: BLE001
            msgs = ast.literal_eval(raw)
        s = [m["content"] for m in msgs if m["role"] == "system"]
        texts.add(s[0] if s else "")
    if len(texts) != 1:
        raise RuntimeError(f"{len(texts)} distinct HP-KR system texts for {model}")
    text = texts.pop()
    if model == "llama-70b-3.3" and text != fdb.NO_ROLE_SYSTEM:
        raise RuntimeError("Llama's HP-KR system text differs from zami's NO_ROLE_SYSTEM")
    return text


def no_system_message() -> None:
    """With no system text, the reads carry NO system message (2026-10-08, Gemma).

    zami's forced_denial_bank._tokenize always prepends Message("system", text), even an empty one. On Gemma,
    TokenizedDataset folds a system message into the first user turn as text + "\\n\\n", and Gemma's template then
    trims that user content -- so an empty system message would add a stray separator to the dialogue the
    detection-mask regex walks. zami's own generation path already omits an empty system message
    (steer_lies.py:304, `if c.get("system")`); the reads now follow the same rule. `_read_batch` and `_requests`
    look `_tokenize` up at call time, so replacing it here takes effect."""
    from deception_detection.data.XXXX_9 import CadenzaDataset
    from deception_detection.tokenized_data import TokenizedDataset
    from deception_detection.types import Message

    def _tokenize(tokenizer, rows: list[dict]):
        dialogues = [([Message("system", r["system"], False)] if r["system"] else [])
                     + [Message("user", r["user"], False), Message("assistant", r["assistant"], True)] for r in rows]
        return TokenizedDataset.from_dialogue_list(dialogues, tokenizer, padding=CadenzaDataset.padding)

    fdb._tokenize = _tokenize


def setup(model: str, set_name: str):
    out = OUT_ROOT / model / set_name
    out.mkdir(parents=True, exist_ok=True)
    system = hpkr_system(model)
    rd.NO_ROLE_SYSTEM = system  # build_rows reads this module global
    rd.OUT = out
    if system == "":
        no_system_message()

    from deception_detection.models import get_model_and_tokenizer
    from deception_detection.scripts import steer_lies
    from deception_detection.scripts.gender_secret import _environment

    def _model():
        _environment()  # zami's NDIF timeouts + key check
        if model in NDIF_REPOS:
            return xmodel_model(NDIF_REPOS[model])
        return get_model_and_tokenizer(model)

    steer_lies._model = _model  # verify() imports _model from steer_lies at call time
    (out / "run_meta.json").write_text(json.dumps({
        "model": model, "set": set_name, "zami_snapshot": "origin/paper@e27ffaf9",
        "no_role_system": system, "deny": list(fdb.DENY)}, indent=1), encoding="utf-8")
    return out, _model


XMODEL_NC = EXPLB / "xmodel" / "code" / "src" / "assistant_axis_ndif" / "ndif_common.py"


def xmodel_model(repo: str):
    """A Mistral model on NDIF through the xmodel loader (regex fix, processor template, pinned date;
    VisionLanguageModel for 3.1), with zami's tokenizer conventions (models.py: left padding, pad = bos).
    For a vision-language wrapper, zami's read and generate are swapped for the copies below, which walk
    model.model.language_model.layers. 2501 is a plain LanguageModel and runs zami's own functions."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("xmodel_ndif_common", XMODEL_NC)
    nc = importlib.util.module_from_spec(spec); spec.loader.exec_module(nc)
    llm = nc.load_language_model(repo)
    tok = llm.tokenizer
    tok.padding_side = "left"
    tok.pad_token_id = tok.bos_token_id
    from deception_detection.scripts import steer_lies
    if "gemma" in repo.lower():  # Gemma ends a reply with <end_of_turn> (its generation_config eos: [1, 106])
        extra = tuple(t for t in ("<end_of_turn>", "<eos>") if t not in steer_lies.STOP_TOKENS)
        steer_lies.STOP_TOKENS = tuple(steer_lies.STOP_TOKENS) + extra
    elif "mistral" in repo.lower() and "</s>" not in steer_lies.STOP_TOKENS:  # zami's are Llama's; on Mistral <unk>
        steer_lies.STOP_TOKENS = tuple(steer_lies.STOP_TOKENS) + ("</s>",)
    tc = llm.config.get_text_config()
    # verify builds its zero vector from model.config.hidden_size, which a composite (vision-language)
    # config does not carry at the top; set it client-side from the decoder's config
    if getattr(llm.config, "hidden_size", None) != tc.hidden_size:
        llm.config.hidden_size = tc.hidden_size
    # verify generates through steered_generate at steer_lies.LAYER (zami's 40) with coef=0, so the layer
    # changes nothing -- but block 40 does not exist in a 40-layer model (0..39)
    if steer_lies.LAYER >= tc.num_hidden_layers:
        log(f"steer_lies.LAYER {steer_lies.LAYER} -> {tc.num_hidden_layers // 2} ({tc.num_hidden_layers} layers; "
            f"verify's steer has coef=0, so the result is the same)")
        steer_lies.LAYER = tc.num_hidden_layers // 2
    if hasattr(getattr(llm._model, "model", None), "language_model"):
        import deception_detection.activations as acts
        import deception_detection.steering as steering
        acts.trace_layer_outputs = trace_layer_outputs_vl
        steering.steered_generate = steered_generate_vl
        log("vision-language wrapper: reads and generation walk model.model.language_model.layers")
    return llm, tok


def trace_layer_outputs_vl(model, tokens, attention_mask, layers, tail=None, lora=None, remote=True,
                           span_mask=None):
    """zami's activations.trace_layer_outputs, span-mask branch, no LoRA, for a vision-language wrapper
    (decoder at model.model.language_model.layers). The same server-side reduction: the product with the
    0/1 mask in the model's dtype, the sum in float32. Defined here because NDIF serialises a trace body
    from the script that runs it."""
    import torch

    if lora is not None or span_mask is None:
        raise NotImplementedError("only the span-mask read without LoRA is ported")
    order = sorted(set(layers))
    with model.trace(tokens, attention_mask=attention_mask, remote=remote):
        outputs = list().save()
        for block in order:
            output = model.model.language_model.layers[block].output
            m = span_mask.to(output.device, torch.float32)
            total = (output * m[:, :, None].to(output.dtype)).sum(dim=1, dtype=torch.float32)
            outputs.append(total / m.sum(dim=1, keepdim=True))
    index = [order.index(layer) for layer in layers]
    return torch.stack(list(outputs), dim=1)[:, index]


def steered_generate_vl(model, tokens, attention_mask, layer, vector, max_new_tokens, coef=1.0,
                        mode="absolute", remote=True, lora=None):
    """zami's steering.steered_generate (absolute mode, no LoRA) for a vision-language wrapper. verify
    calls it with coef=0, i.e. a plain greedy continuation through the same path."""
    import torch

    if lora is not None or mode != "absolute":
        raise NotImplementedError("only absolute steering without LoRA is ported")
    shift = coef * torch.as_tensor(vector, dtype=torch.float32)
    # logits_to_keep=1 (2026-10-08): without it NDIF's VisionLanguageModel generate gives lm_head (1, 1, T, V) and fails
    # ("same number of dimensions" greedy; Mistral-3.1 a CUDA device-side assert). See 1_generate_ndif._vl_generate_fix.
    with model.generate(max_new_tokens=max_new_tokens, do_sample=False, logits_to_keep=1, remote=remote) as tracer:
        with tracer.invoke(tokens, attention_mask=attention_mask):
            for _ in tracer.iter[:]:
                output = model.model.language_model.layers[layer].output
                model.model.language_model.layers[layer].output = output + shift.to(output.device, output.dtype)
        with tracer.invoke():
            result = tracer.result.save()
    sequences = getattr(result, "sequences", result)
    return sequences[:, tokens.shape[1]:].tolist()


def parse_layers(spec: list[str], model: str) -> list[int]:
    if spec == ["all"]:
        return list(range(N_LAYERS[model]))
    if spec == ["zami"]:
        return list(rd.LAYERS)
    return [int(x) for x in spec]


def open_generation_turn(model: str) -> None:
    """Generation prompts for verify must END the prompt inside an open assistant turn.

    Measured 2026-10-07 02:42 (offline, both tokenizers): zami's steer_lies._tokenize, given an empty
    assistant message, renders Llama's prompt ending '...<|start_header_id|>assistant<|end_header_id|>\\n\\n'
    (open: correct) but Qwen's ending '...<|im_start|>assistant\\n<|im_end|>\\n' -- the reply turn already
    closed. Qwen then opened a new message ("Note: It seems there was an accidental ...") on 17 of the 70
    known items, which verify scored as wrong answers (France, hexagon, Mercury ...). Here the trailing
    [<|im_end|>, '\\n'] after an empty assistant turn is removed and each row re-padded on its own side;
    and Qwen's end tokens join zami's Llama-only STOP_TOKENS. A no-op on Llama (asserted offline)."""
    if model != "qwen-72b":
        return
    import types

    import torch
    from deception_detection.scripts import steer_lies

    original = steer_lies._tokenize

    def _tokenize(tokenizer, chats):
        toks = original(tokenizer, chats)
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        nl = tokenizer.encode("\n", add_special_tokens=False)
        head = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        rows, left = [], bool((toks.attention_mask[:, 0] == 0).any())
        for ids, mask, chat in zip(toks.tokens.tolist(), toks.attention_mask.tolist(), chats):
            real = [t for t, m in zip(ids, mask) if m]
            tail = [end] + nl
            if chat.get("assistant") == "" and real[-len(tail):] == tail and real[-len(tail) - len(head):-len(tail)] == head:
                real = real[:-len(tail)]
            rows.append(real)
        width = max(len(r) for r in rows)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end
        tokens = torch.full((len(rows), width), pad, dtype=toks.tokens.dtype)
        attn = torch.zeros((len(rows), width), dtype=toks.attention_mask.dtype)
        for i, r in enumerate(rows):
            if left:
                tokens[i, width - len(r):] = torch.tensor(r); attn[i, width - len(r):] = 1
            else:
                tokens[i, :len(r)] = torch.tensor(r); attn[i, :len(r)] = 1
        return types.SimpleNamespace(tokens=tokens, attention_mask=attn)

    steer_lies._tokenize = _tokenize
    steer_lies.STOP_TOKENS = tuple(steer_lies.STOP_TOKENS) + ("<|im_end|>", "<|endoftext|>")


def cmd_verify(a) -> None:
    out, _ = setup(a.model, a.set)
    if (out / "verify.json").exists():
        log("verify.json exists; skipping")
        return
    open_generation_turn(a.model)
    wait_for_window(a.gate_status)
    rd.verify(max_new_tokens=8, source=SETS[a.set], out=out)


def cmd_items(a) -> None:
    out, _ = setup(a.model, a.set)
    v = json.loads((out / "verify.json").read_text(encoding="utf-8"))
    verified = {x["question"]: x["correct"] for x in v["known"]}
    rows = [r for r in rd.build_rows(rd.general_items(verified, source=SETS[a.set])) if r["arm"] == "none"]
    (out / "items.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    known = sum(r["knows"] for r in rows)
    log(f"{len(rows)} rows (none arm): {known} known reads, {len(rows) - known} unknowable reads; "
        f"wrong known answers replaced from reserve: {sum(not x['correct'] for x in v['known'])}")


def content_only_masks(tok) -> None:
    """Read only the reply's words, as zami's Llama K does.

    Measured 2026-10-07 (offline, both tokenizers): zami's TokenizedDataset marks exactly the reply's
    content on Llama ('I do not know the answer.', 7 tokens) but on Qwen adds the closing
    '<|im_end|>' (8 tokens). Unmarking special tokens makes Qwen's span the same as Llama's and is a
    no-op on Llama. `_read_batch` looks `_masks` up at call time, so replacing it here takes effect."""
    import torch

    original = fdb._masks
    special = torch.tensor(sorted(set(tok.all_special_ids)))

    def _masks(toks, position):
        m = original(toks, position)
        if position == "answer":
            m = m * (~torch.isin(toks.tokens, special)).to(m.dtype)
        return m

    fdb._masks = _masks


def check_answer_spans(tok, rows: list[dict], out: Path) -> None:
    """Every row (not just six): the tokens the read averages over decode to exactly the forced reply."""
    toks = fdb._tokenize(tok, rows)
    mask = fdb._masks(toks, "answer").bool()
    bad, lengths = [], {}
    for r, ids, m in zip(rows, toks.tokens.tolist(), mask.tolist()):
        span = [t for t, d in zip(ids, m) if d]
        text = tok.decode(span)
        lengths[r["assistant"]] = len(span)
        if text != r["assistant"]:
            bad.append({"row": r["row"], "decoded": text, "forced": r["assistant"]})
    (out / "check_answer_spans.json").write_text(json.dumps(
        {"rows": len(rows), "bad": bad, "tokens_per_reply": lengths}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    if bad:
        raise RuntimeError(f"{len(bad)} of {len(rows)} rows read tokens other than the forced reply: {bad[:2]}")
    log(f"answer spans: all {len(rows)} rows read exactly the forced reply; tokens per reply {lengths}")


def check_c_transformers5(tok, rows: list[dict], out: Path) -> None:
    """zami's check C, run under transformers 5.

    Its span port (persona_finetuned.assistant_span) does list(apply_chat_template(..., tokenize=True)),
    which was the id list in transformers 4 but is a BatchEncoding in 5.17 -- list() then yields its keys
    and the span comes back empty (measured 2026-10-07). For the check only, the tokenizer instance
    returns the ids again; tokenize=False calls (TokenizedDataset, the reads) are untouched."""
    original = tok.apply_chat_template

    def ids_when_tokenized(*args, **kw):
        r = original(*args, **kw)
        if kw.get("tokenize", True) and hasattr(r, "keys") and "input_ids" in r:
            return r["input_ids"]
        return r

    tok.apply_chat_template = ids_when_tokenized
    try:
        fdb._check_spans(tok, rows, out=out)
    finally:
        del tok.apply_chat_template  # back to the class method


def cmd_read(a) -> None:
    out, _model = setup(a.model, a.set)
    layers = parse_layers(a.layers, a.model)
    rows = json.loads((out / "items.json").read_text(encoding="utf-8"))
    model, tok = _model()
    content_only_masks(tok)
    if a.model == "llama-70b-3.3":                # zami's check C ports a non-Qwen span rule
        check_c_transformers5(tok, rows, out)
    check_answer_spans(tok, rows, out)
    fdb._check_identical_answers(tok, rows)
    folder = out / "parts_answer"
    done = done_rows(folder)
    todo = [r for r in rows if r["row"] not in done]
    if not todo:
        log("all rows already read")
        return
    requests = fdb._requests(tok, todo, budget=a.token_budget)
    log(f"{len(todo)} rows to read in {len(requests)} requests, {len(layers)} layers")
    for k, batch in enumerate(requests):
        for attempt in range(1, a.retries + 1):
            wait_for_window(a.gate_status)
            try:
                means = fdb._read_batch(model, tok, batch, "answer", layers)
                break
            except Exception as error:  # noqa: BLE001 -- NDIF drops a request now and then
                last = str(error).strip().splitlines()[-1] if str(error).strip() else repr(error)
                log(f"  request {k + 1} attempt {attempt}/{a.retries} failed: {last[:200]}")
                if attempt == a.retries:
                    raise
        save_request(folder, [r["row"] for r in batch], layers, means)
        log(f"request {k + 1}/{len(requests)}: {len(batch)} rows")


def _table(out: Path) -> pd.DataFrame:
    mine = pd.DataFrame(json.loads((out / "items.json").read_text(encoding="utf-8"))).set_index("row")
    records = []
    for row, vecs in rd._load(out / "parts_answer").items():
        r = mine.loc[row]
        records.append({"subject": r["subject"], "arm": r["arm"], "knows": bool(r["knows"]), "item": r["item"],
                        "string": int(r["string"])} | {f"L{L}": v for L, v in vecs.items()})
    return pd.DataFrame(records)


def cmd_directions(a) -> None:
    out, _ = setup(a.model, a.set)
    table = _table(out)
    log(f"cells:\n{table.groupby(['subject', 'arm', 'knows']).size().to_string()}")
    if len(table) != 240:
        raise RuntimeError(f"{len(table)} reads, expected 240")
    layers = parse_layers(a.layers, a.model)
    k, halves = {}, []
    items = sorted(table["item"].unique())
    for layer in layers:
        k[f"L{layer}"] = rd.directions_from_cells(table, layer)["none|general"]
        parts = [rd.directions_from_cells(table[table["item"].isin(items[p::2])], layer) for p in (0, 1)]
        halves.append({"layer": layer, "direction": "none|general",
                       "split_half_cosine": float(parts[0]["none|general"] @ parts[1]["none|general"])})
    np.savez(out / "K.npz", **k)
    pd.DataFrame(halves).round(4).to_csv(out / "split_half.csv", index=False)
    log(f"K at {len(k)} layers -> {out / 'K.npz'}; split-half cosine "
        f"min {min(h['split_half_cosine'] for h in halves):.4f}, "
        f"median {np.median([h['split_half_cosine'] for h in halves]):.4f}")


def cmd_compare(a) -> None:
    """Validation: our run against zami's committed numbers for the same model and set."""
    out = OUT_ROOT / a.model / a.set
    ref = ZAMI_RESULTS[a.set]
    ours_v = json.loads((out / "verify.json").read_text(encoding="utf-8"))
    ref_v = json.loads((ref / "verify.json").read_text(encoding="utf-8"))
    for part, key in (("known", "correct"), ("unknown", "report")):
        same = sum(x[key] == y[key] for x, y in zip(ours_v[part], ref_v[part]))
        log(f"verify {part}: {same}/{len(ref_v[part])} identical {key}")
    if (out / "split_half.csv").exists() and (ref / "split_half.csv").exists():
        ours = pd.read_csv(out / "split_half.csv")
        theirs = pd.read_csv(ref / "split_half.csv")
        theirs = theirs[theirs["direction"] == "none|general"]
        m = ours.merge(theirs, on=["layer", "direction"], suffixes=("_ours", "_zami"))
        m["diff"] = (m["split_half_cosine_ours"] - m["split_half_cosine_zami"]).round(4)
        log("split-half cosine, none|general:\n" + m.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["verify", "items", "read", "directions", "compare"])
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--set", default="first", choices=sorted(SETS))
    ap.add_argument("--layers", nargs="+", default=["zami"], help="'all', 'zami' (20 36 40 44) or numbers")
    ap.add_argument("--token_budget", type=int, default=fdb.TOKEN_BUDGET)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--gate_status", default=None,
                    help="the persona chain's chain.status: wait while it holds the subject model")
    a = ap.parse_args()
    {"verify": cmd_verify, "items": cmd_items, "read": cmd_read,
     "directions": cmd_directions, "compare": cmd_compare}[a.step](a)


if __name__ == "__main__":
    main()
