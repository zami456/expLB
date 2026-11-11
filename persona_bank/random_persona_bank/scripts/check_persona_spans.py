"""Tokenizer-only check before any NDIF request: under the random bank's system prompts, every span the pipeline
will pool is exactly the reply.

Builds all 36 x 5 x 30 = 5,400 conversations the way 1_generate_ndif.py stores them (its own `build_messages`, the
model's short name in default's "You are {model_name}."), with a stand-in reply for each (prompt, question) taken
from a real reply set -- the model's own kd `default` replies where they exist, else Qwen's -- then runs the
pipeline's own span builder (2_activations_ndif.build_items, ConversationEncoder + the ChatML header trim) on the
deployed code and the model's tokenizer as the pipeline loads it (ndif_common.load_tokenizer).

A span passes if it decodes to the reply exactly, or to the reply stripped where the chat template itself trims
the content (Gemma). Exit 1 on any other result.

    .venv-aa-win/Scripts/python.exe random_persona_bank/scripts/check_persona_spans.py \
        --code random_persona_bank/results/Qwen2.5-72B-Instruct/code --model Qwen/Qwen2.5-72B-Instruct --short Qwen \
        --standin %USERPROFILE%/xmodel-runs/Qwen2.5-72B-Instruct/persona/responses/default.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--short", required=True)
    ap.add_argument("--standin", required=True, type=Path)
    a = ap.parse_args()

    os.environ["XMODEL_SHORT_NAME"] = a.short          # read by 1_generate_ndif at import
    pipe = a.code / "src" / "assistant_axis_ndif"
    sys.path.insert(0, str(pipe))
    gen = load("gen", pipe / "1_generate_ndif.py")
    act = load("act", pipe / "2_activations_ndif.py")
    nc = load("nc", pipe / "ndif_common.py")
    assert gen.SHORT_NAME == a.short

    tok = nc.load_tokenizer(a.model)
    enc = act.ConversationEncoder(tok, a.model)

    standin = {}
    for line in a.standin.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            standin[(r["prompt_index"], r["question_index"])] = r["conversation"][-1]["content"]
    questions = [json.loads(l)["question"] for l in (a.code / "rpb" / "questions" / "kd_questions_shared.jsonl")
                 .read_text(encoding="utf-8").splitlines() if l.strip()]
    role_files = sorted((a.code / "rpb" / "rpb_instructions").glob("*.json"))
    assert len(role_files) == 36 and len(questions) == 30 and len(standin) == 150, (len(role_files), len(questions), len(standin))

    res, after, longest, shown = Counter(), Counter(), 0, 0
    for rf in role_files:
        instr = [d["pos"] for d in json.loads(rf.read_text())["instruction"]]   # read_text() with no encoding, as the pipeline does
        for pi, ins in enumerate(instr):
            for qi, q in enumerate(questions):
                reply = standin[(pi, qi)]
                conv = gen.build_messages(ins, q) + [{"role": "assistant", "content": reply}]
                items = act.build_items(enc, [conv], max_length=8192)
                if len(items) != 1:
                    res["dropped"] += 1
                    continue
                ids, a0, a1 = items[0]
                longest = max(longest, len(ids))
                got = tok.decode(ids[a0:a1])
                if got == reply:
                    res["exact"] += 1
                elif got == reply.strip():
                    res["exact_after_template_strip"] += 1
                else:
                    res["MISMATCH"] += 1
                    if shown < 3:
                        shown += 1
                        print(f"MISMATCH {rf.stem} p{pi} q{qi}: got {got[:80]!r} ... {got[-40:]!r}\n"
                              f"                 want {reply[:80]!r} ... {reply[-40:]!r}")
                after[tok.decode(ids[a1:a1 + 1]) if a1 < len(ids) else "<end>"] += 1

    print(f"model {a.model}  short {a.short}  stand-in {a.standin}")
    print(f"spans: {dict(res)}  (of {sum(res.values())})")
    print(f"token after the span: {dict(after)}")
    print(f"longest sequence {longest} tokens (pipeline max_length 8192)")
    ex = json.loads((a.code / "rpb" / "rpb_instructions" / "skeptic.json").read_text())["instruction"][0]["pos"]
    shown_prompt = tok.apply_chat_template(gen.build_messages(ex, questions[0]), tokenize=False, add_generation_prompt=True)
    print("generation prompt, skeptic variant 0, question 0:\n" + shown_prompt)
    sys.exit(1 if res["MISMATCH"] or res["dropped"] or sum(res.values()) != 5400 else 0)


if __name__ == "__main__":
    main()
