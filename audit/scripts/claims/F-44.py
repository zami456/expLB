#!/usr/bin/env python
"""M1 claim check F-44 (+ the text-overlap half of F-108).

F-44 says a `8_coords_ndif.py --prepare_only` probe at max_length 1024 gave
    harm-pressure-choice            2134 rows -> 2134 spans
    harm-pressure-knowledge-report  2139 rows -> 2139 spans (dropped: truncated=92)
    harm-pressure-kr-matched        FAILED: KeyError
The probe was console-only, so this re-derives it: same build_eval_items(), same tokenizer,
max_length 1024, writing nothing.

Also: the cross-dataset identical-response check F-108 makes (passB rows.jsonl carries no
response text, so it has to come from the dataset loader).

READ-ONLY -- imports the pipeline modules but only calls loaders.
"""
import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

R = Path("C:/Users/user4/expLB")
PIPE = R / "src" / "assistant_axis_ndif"
sys.path.insert(0, str(PIPE))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
ap.add_argument("--max_length", type=int, default=1024)
ap.add_argument("--skip_tok", action="store_true")
args = ap.parse_args()


def mod(name, fn):
    spec = importlib.util.spec_from_file_location(name, PIPE / fn)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m          # dataclasses needs the module in sys.modules
    spec.loader.exec_module(m)
    return m


ED = mod("eval_datasets", "eval_datasets.py")

print("== F-44: is harm-pressure-kr-matched a loadable key? ==")
print("   in CSV_DATASETS:", "harm-pressure-kr-matched" in ED.CSV_DATASETS)
print("   in JSON_DATASETS:", "harm-pressure-kr-matched" in getattr(ED, "JSON_DATASETS", {}))
print("   in GENERATED:", "harm-pressure-kr-matched" in getattr(ED, "GENERATED", {}))
print("   in all_keys():", "harm-pressure-kr-matched" in ED.all_keys())
try:
    ED.load_dataset("harm-pressure-kr-matched")
    print("   load_dataset -> OK (claim WRONG)")
except Exception as e:
    print(f"   load_dataset -> {type(e).__name__}: {e}")

print("\n== dataset row counts from the loader ==")
KEYS = ["harm-pressure-choice", "harm-pressure-knowledge-report", "instructed-deception",
        "convincing-game", "insider-trading", "ai_liar", "roleplaying", "sandbagging_v2",
        "mask", "sycophancy_mt"]
rows_by = {}
for k in KEYS:
    try:
        rows_by[k] = ED.load_dataset(k)
        print(f"   {k:34} {len(rows_by[k]):6}")
    except Exception as e:
        print(f"   {k:34} FAILED {type(e).__name__}: {e}")

print("\n== F-108 (b): identical response text across datasets ==")
tx = {k: Counter((r.response_str or "").strip() for r in v if (r.response_str or "").strip())
      for k, v in rows_by.items()}
tot = 0
for a, b in combinations(sorted(tx), 2):
    common = set(tx[a]) & set(tx[b])
    if common:
        tot += len(common)
        print(f"   {a} & {b}: {len(common)} identical response strings")
        for t in sorted(common, key=len)[:6]:
            print(f"      {t[:120]!r}  ({a} x{tx[a][t]} / {b} x{tx[b][t]})")
print(f"   TOTAL distinct cross-dataset identical response strings: {tot}")

if args.skip_tok:
    sys.exit(0)

print("\n== F-44: re-run build_eval_items at max_length %d ==" % args.max_length)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, token=os.getenv("HF_TOKEN"))
except Exception as e:
    print(f"   tokenizer unavailable: {type(e).__name__}: {e}")
    sys.exit(0)
C8 = mod("coords8", "8_coords_ndif.py")
enc = C8.ConversationEncoder(tok)
tok.padding_side = "right"
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
for k in ("harm-pressure-choice", "harm-pressure-knowledge-report"):
    rows = rows_by[k]
    for ml in (args.max_length, 2048, 8192):
        drops = {}
        items = C8.build_eval_items(tok, enc, rows, ml, drops)
        print(f"   {k:34} max_length {ml:5}: {len(rows)} rows -> {len(items)} spans  "
              f"drops {json.dumps(drops)}")
