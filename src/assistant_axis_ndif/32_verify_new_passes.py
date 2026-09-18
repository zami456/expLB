#!/usr/bin/env python
"""Verify the three new activation stores before anything is computed from them.

  passB_fine  layer 44 is extracted twice on purpose -- once here and once in passB -- so the
              overlap is the check that the new pass is wired exactly like the old one.
  passP       the mask covers the PROMPT, so n_tok must equal the row's prompt_len, and the
              last-token vector must differ from passB's (which is the last RESPONSE token).
  passA/passC the sycophancy rows must be the SAME 200 rows in both, and a row's per-token block
              in passC must average to its returned mean vector.

    python 32_verify_new_passes.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "run-coords"


def load_pooled(store: str, key: str, layer: int, pool: str = "mean"):
    """-> {row_id: vector} for one layer and pooling."""
    out = {}
    for f in sorted((RUN / store / key).glob("chunk_*.pt")):
        b = torch.load(f, map_location="cpu", weights_only=False)
        li = list(b["layers"]).index(layer)
        pi = list(b.get("pooling", ["mean", "last"])).index(pool)
        for i, rid in enumerate(b["row_ids"]):
            out[rid] = b["pooled"][i, li, :, pi].float()
    return out


def rows_of(store: str, key: str):
    p = RUN / store / key / "rows.jsonl"
    return {json.loads(l)["row_id"]: json.loads(l) for l in p.open(encoding="utf-8")} if p.exists() else {}


def main() -> None:
    print("== passB_fine vs passB at the shared layer 44")
    for key in ("ai_liar", "mask"):
        a, b = load_pooled("passB_fine", key, 44), load_pooled("passB", key, 44)
        common = sorted(set(a) & set(b))
        if not common:
            print(f"  {key}: no overlap"); continue
        d = torch.stack([(a[r] - b[r]).abs().max() for r in common])
        cos = torch.stack([torch.nn.functional.cosine_similarity(a[r], b[r], dim=0) for r in common])
        print(f"  {key}: {len(common)} rows, worst |diff| {d.max():.4f}, min cos {cos.min():.6f}")

    print("== passP: mask covers the prompt")
    for key in ("ai_liar", "instructed-deception"):
        meta = rows_of("passP", key)
        bad = n = 0
        for f in sorted((RUN / "passP" / key).glob("chunk_*.pt"))[:2]:
            blob = torch.load(f, map_location="cpu", weights_only=False)
            for i, rid in enumerate(blob["row_ids"]):
                n += 1
                if int(round(float(blob["n_tok"][i]))) != meta[rid]["prompt_len"]:
                    bad += 1
        pm, rm = load_pooled("passP", key, 44, "last"), load_pooled("passB", key, 44, "last")
        shared = sorted(set(pm) & set(rm))[:200]
        cos = torch.stack([torch.nn.functional.cosine_similarity(pm[r], rm[r], dim=0) for r in shared])
        print(f"  {key}: {n} rows checked, n_tok != prompt_len on {bad}; "
              f"last prompt token vs last response token: mean cos {cos.mean():.3f}")

    print("== sycophancy rows added to passA and passC")
    ra, rc = rows_of("passA", "sycophancy_mt"), rows_of("passC", "sycophancy_mt")
    print(f"  passA {len(ra)} rows, passC {len(rc)} rows, identical row set: {set(ra) == set(rc)}")
    files = sorted((RUN / "passC" / "sycophancy_mt").glob("chunk_*.pt"))
    if files:
        b = torch.load(files[0], map_location="cpu", weights_only=False)
        li = list(b["layers"]).index(44)
        offs, worst = b["tok_offsets"], 0.0
        for i in range(min(20, len(b["row_ids"]))):
            seg = b["tokens"][offs[i]:offs[i + 1], li, :].float()
            worst = max(worst, float((seg.mean(0) - b["pooled"][i, li, :, 0].float()).abs().max()))
        print(f"  per-token mean vs returned mean pooling: worst |diff| {worst:.4f} over 20 rows")
    for store in ("passA", "passC", "passP", "passB_fine"):
        keys = [Path(p).name for p in glob.glob(str(RUN / store / "*")) if Path(p).is_dir()]
        tot = sum(len(rows_of(store, k)) for k in keys)
        print(f"  {store}: {len(keys)} datasets, {tot} rows")


if __name__ == "__main__":
    main()
