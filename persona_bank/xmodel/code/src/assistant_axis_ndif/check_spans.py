"""Check that every pooled span is exactly the model's reply, for a model the encoder was not
written for. Tokenizer only -- no NDIF.

The spans come from assistant_axis's ConversationEncoder, whose Qwen branch was written for
Qwen-3 (thinking tokens). On Qwen-2.5 an off-by-one there would silently fold `<|im_end|>` or
the `assistant\\n` header into every mean. So before any extraction, decode ids[a0:a1] for every
row and compare it with the stored response text, and check the tokens either side of the span.

    python check_spans.py --model Qwen/Qwen2.5-72B-Instruct --eval_dir D:/.../eval \
        --datasets instructed-deception harm-pressure-choice ...

Reads items.pt written by `8_coords_ndif.py --prepare_only`. Exit 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_datasets as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--show", type=int, default=3)
    a = ap.parse_args()
    import importlib.util as _ilu
    _s = _ilu.spec_from_file_location("ndif_common", Path(__file__).resolve().parent / "ndif_common.py")
    _nc = _ilu.module_from_spec(_s); _s.loader.exec_module(_nc)
    tok = _nc.load_tokenizer(a.model)  # xmodel: regex fix, template, pinned date
    print(f"CSV_MODEL={E.CSV_MODEL}")
    bad_total = 0
    for key in a.datasets:
        items = torch.load(Path(a.eval_dir) / key / "items.pt", weights_only=False)
        rows = {r.row_id: r for r in E.load_dataset(key)}
        exact = stripped = bad = 0
        before, after = {}, {}
        shown = 0
        for rid, ids, a0, a1 in items:
            want = rows[rid].response_str
            got = tok.decode(ids[a0:a1])
            if got == want:
                exact += 1
            elif got.strip() == want.strip():
                stripped += 1
            else:
                bad += 1
                if shown < a.show:
                    print(f"  MISMATCH {rid}\n    want {want[:160]!r}\n    got  {got[:160]!r}")
                    shown += 1
            b = tok.decode(ids[max(a0 - 3, 0):a0])
            e = tok.decode(ids[a1:a1 + 2])
            before[b] = before.get(b, 0) + 1
            after[e] = after.get(e, 0) + 1
        bad_total += bad
        print(f"{key}: {len(items)} spans (of {len(rows)} rows) exact={exact} "
              f"whitespace-only={stripped} MISMATCH={bad}")
        print(f"  3 tokens before span: {dict(sorted(before.items(), key=lambda kv: -kv[1])[:3])}")
        print(f"  2 tokens after span : {dict(sorted(after.items(), key=lambda kv: -kv[1])[:3])}")
    sys.exit(1 if bad_total else 0)


if __name__ == "__main__":
    main()
