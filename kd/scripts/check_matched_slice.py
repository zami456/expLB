#!/usr/bin/env python3
"""Composition of `harm-pressure-kr-matched` in each store — the number the top-up exists for.

F-20 measured the slice as 258 rows at 124 deceptive / 134 honest on passB, but only 98 rows at
89/9 on passA and passC, because those stores kept a 200-row label-stratified subsample of HP-KR
and HP-KR's honest class is overwhelmingly "I know the answer" — assertions, which by
construction cannot appear in a DENIAL-matched slice. An AUROC against nine negatives is noise
with a number attached, which is why the passA/passC grids ran with ten datasets instead of
eleven.

This recomputes the slice against what is actually on disk per store, so the top-up can be
verified by the quantity it was meant to fix rather than by row count alone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "assistant_axis_ndif"))
import eval_datasets as E  # noqa: E402

DS = "harm-pressure-knowledge-report"


def main() -> None:
    want = E.hpkr_slice("text_matched")
    print(f"matched-slice row_ids defined by the benchmark CSV: {len(want)}\n")
    print(f"  {'store':<8}{'HP-KR rows':>11}{'matched':>9}{'deceptive':>11}{'honest':>8}  verdict")
    for store in ("passA", "passB", "passC"):
        rj = Path("run-coords") / store / DS / "rows.jsonl"
        if not rj.exists():
            print(f"  {store:<8} rows.jsonl missing"); continue
        rows = [json.loads(l) for l in rj.open(encoding="utf-8")]
        # Use the STORED row_id. Reconstructing it as dataset:item_id:sample_idx is wrong:
        # hpkr_slice builds the id as "<dataset>:<sub>:<idx>:<i>" where item_id already
        # contains a colon and <i> is the CSV enumerate index over ALL rows, before the
        # model filter. Rebuilding it matched 1 of 258 on passB and looked like a data
        # problem rather than a key-format problem.
        have = {r["row_id"]: r for r in rows}
        hit = {k: v for k, v in have.items() if k in want}
        dec = sum(1 for v in hit.values() if int(v["label"]) == 1)
        hon = len(hit) - dec
        # The binding constraint is the SMALLER class: an AUROC is only as good as its
        # minority count, whatever the total says.
        verdict = ("unusable" if min(dec, hon) < 30 else
                   "thin" if min(dec, hon) < 100 else "usable")
        print(f"  {store:<8}{len(rows):>11}{len(hit):>9}{dec:>11}{hon:>8}  {verdict}")


if __name__ == "__main__":
    main()
