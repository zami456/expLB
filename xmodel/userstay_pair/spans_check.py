"""Offline check that every pooled span of a user_stay persona row is exactly the model's reply. Tokenizer only.

The activation step (2_activations_ndif.build_items) pools over the last assistant span that assistant_axis's
ConversationEncoder reports, after ndif_common.trim_chatml_header_newline. The user_stay format adds an explicit empty
system message and a longer user turn; a shifted span would silently fold template tokens into every vector. So for
every row: decode ids[a0:a1] and require it to equal the stored reply EXACTLY (not stripped: a stripped comparison hid
the Qwen header newline once), and show the tokens either side of the span.

    .venv-aa-win/Scripts/python.exe xmodel/userstay_pair/spans_check.py --code <run>/code --model <repo id> --responses <dir>
Exit 1 on any mismatch.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--responses", type=Path, required=True)
    a = ap.parse_args()
    act = load(a.code / "src/assistant_axis_ndif/2_activations_ndif.py", "act")
    nc = load(a.code / "src/assistant_axis_ndif/ndif_common.py", "nc")
    tok = nc.load_tokenizer(a.model)
    enc = act.ConversationEncoder(tok, a.model)
    n = ok = 0
    edges = set()
    for f in sorted(a.responses.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            conv = r["conversation"]
            assert r.get("prompt_format") == "user_stay" and conv[0] == {"role": "system", "content": ""}, f.stem
            items = act.build_items(enc, [conv], 8192)
            n += 1
            if not items:
                print(f"NO SPAN {f.stem} p{r['prompt_index']} q{r['question_index']}")
                continue
            ids, a0, a1 = items[0]
            got = tok.decode(ids[a0:a1])
            want = conv[-1]["content"]
            edges.add((tok.decode(ids[a0 - 1:a0]), tok.decode(ids[a1:a1 + 1])))
            if got == want:
                ok += 1
            else:
                print(f"MISMATCH {f.stem} p{r['prompt_index']} q{r['question_index']}:\n  span  {got[:120]!r}\n  reply {want[:120]!r}")
    print(f"{a.model}: {ok}/{n} spans equal the reply exactly; token before / after the span: {sorted(edges)}")
    return 0 if ok == n and n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
