"""Split audit/phase4/sample_blinded.csv into per-source batches of ~12.

Deterministic: rows keep the order they have in sample_blinded.csv (which is
itself seeded at 42 by p4_select_sample.py). No prior verdict, no tier, ever
enters a batch file -- that is the Phase 4 independence rule.

Writes audit/phase4/batches/<batch>.csv and prints the manifest.
"""
import csv
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P4 = os.path.join(ROOT, "audit", "phase4")
OUT = os.path.join(P4, "batches")

# source letter -> (batch prefix, how many batches)
PLAN = {
    "D": ("D", 4),
    "F": ("F", 3),
    "H": ("H", 2),
    "M": ("M", 2),
    "O": ("O", 2),
    "S": ("SA", 1),   # S + A ride together (9 + 2 = 11)
    "A": ("SA", 1),
}

FIELDS = ["id", "source_ref", "stated_numbers", "claim_plain"]


def main():
    rows = list(csv.DictReader(open(os.path.join(P4, "sample_blinded.csv"), encoding="utf-8")))
    assert len(rows) == 161, len(rows)

    by_src = defaultdict(list)
    for r in rows:
        by_src[r["id"][0]].append(r)

    os.makedirs(OUT, exist_ok=True)
    batches = {}

    # S and A share a batch
    sa = by_src.pop("S", []) + by_src.pop("A", [])
    batches["SA1"] = sa

    for src in sorted(by_src):
        prefix, n = PLAN[src][0], PLAN[src][1]
        items = by_src[src]
        # even split, remainder spread over the first batches
        base, extra = divmod(len(items), n)
        i = 0
        for b in range(n):
            size = base + (1 if b < extra else 0)
            batches[f"{prefix}{b + 1}"] = items[i:i + size]
            i += size
        assert i == len(items)

    total = 0
    for name in sorted(batches):
        rs = batches[name]
        path = os.path.join(OUT, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for r in rs:
                w.writerow({k: r[k] for k in FIELDS})
        total += len(rs)
        print(f"{name:5s} {len(rs):3d}  {os.path.relpath(path, ROOT)}")
    print(f"{'TOTAL':5s} {total:3d}  in {len(batches)} batches")
    assert total == 161, total


if __name__ == "__main__":
    main()
