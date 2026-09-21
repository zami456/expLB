"""What settings did the EXISTING passA/passC HP-KR chunks use?

New chunks must match exactly or the store cannot be stacked. Each chunk records its own
layers/pooling/span, so this reads the truth off disk rather than trusting the runner script
that was supposed to have produced it.
"""
import json
import sys
from pathlib import Path

import torch

for store in ("passA", "passB", "passC"):
    d = Path("run-coords") / store / "harm-pressure-knowledge-report"
    if not d.exists():
        print(f"{store}: MISSING"); continue
    chunks = sorted(d.glob("chunk_*.pt"))
    if not chunks:
        print(f"{store}: no chunks"); continue
    b = torch.load(chunks[0], weights_only=False)
    ids = set()
    for c in chunks:
        ids |= set(torch.load(c, weights_only=False)["row_ids"])
    rj = d / "rows.jsonl"
    nrows = sum(1 for _ in rj.open(encoding="utf-8")) if rj.exists() else -1
    it = d / "items.pt"
    nitems = len(torch.load(it, weights_only=False)) if it.exists() else -1
    print(f"{store}:")
    print(f"  chunks={len(chunks)}  distinct row_ids={len(ids)}  "
          f"rows.jsonl={nrows}  items.pt={nitems}")
    print(f"  layers({len(b['layers'])})={b['layers']}")
    # Older chunks predate the pooling/span keys; absent means the defaults
    # (pooling ["mean","last"], span "response") that the store was built with.
    print(f"  pooling={b.get('pooling', 'ABSENT(default mean,last)')}  "
          f"span={b.get('span', 'ABSENT(default response)')}  "
          f"per_token={'tokens' in b}")
    print(f"  keys={sorted(b.keys())}")
    print(f"  pooled shape per chunk={tuple(b['pooled'].shape)}")
