"""Deterministic batches for Phase 6 — the 378 inventory claims that never got a verdict.

Phase 4 batched 161 claims into 14 groups; this uses the same shape. Batches are cut along
the claim's own source_ref so one batch is one table, figure or section — a checker holds one
part of the draft in mind at a time, and a whole table's numbers get re-derived together
rather than one cell at a time.

Writes audit/phase6/batches/*.csv and prints the plan. Re-running is idempotent.

  python audit/scripts/p16_make_phase6_batches.py
"""
import csv
import os
import re
from collections import defaultdict

INV = "audit/claims_inventory.csv"
CHK = "audit/claims.csv"
OUT = "audit/phase6/batches"
TARGET = 14  # soft cap on claims per batch


def group_key(r):
    """One batch per table / figure / section of the draft; per-file for the rest."""
    ref = r["source_ref"]
    if r["id"].startswith("D-"):
        m = re.match(r"draft (p\d+) (Table \d+|Fig \d+[^(]*|App [A-Z][^(]*|[\d.]+ \w+|\w+)", ref)
        return ("draft", m.group(2).strip() if m else ref.split("(")[0].strip())
    if r["id"].startswith("L-F-"):
        return ("ledger", "FINDINGS.md")
    return ("collab", ref.split(":")[0].split("/")[-1])


def main():
    inv = list(csv.DictReader(open(INV, encoding="utf-8")))
    done = {r["id"] for r in csv.DictReader(open(CHK, encoding="utf-8"))}
    un = [r for r in inv if r["id"] not in done]

    groups = defaultdict(list)
    for r in un:
        groups[group_key(r)].append(r)

    # split oversized groups, keep source order so ids stay contiguous
    batches = []
    for (fam, name), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["id"])
        for i in range(0, len(rows), TARGET):
            batches.append((fam, name, rows[i:i + TARGET]))
    batches.sort(key=lambda b: ({"draft": 0, "collab": 1, "ledger": 2}[b[0]], b[1]))

    os.makedirs(OUT, exist_ok=True)
    fields = list(un[0].keys())
    total = 0
    print(f"{'batch':>10s} {'family':>7s} {'n':>4s}  scope")
    for i, (fam, name, rows) in enumerate(batches, 1):
        bid = f"B{i:02d}"
        path = os.path.join(OUT, f"{bid}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        total += len(rows)
        print(f"{bid:>10s} {fam:>7s} {len(rows):4d}  {name}")

    assert total == len(un), f"batching lost claims: {total} != {len(un)}"
    print(f"\n  {len(batches)} batches, {total} claims, written to {OUT}/")
    print(f"  families: " + ", ".join(
        f"{f}={sum(len(b[2]) for b in batches if b[0] == f)}"
        for f in ("draft", "collab", "ledger")))
    print("\n  A batch is DONE when every row has verdict + evidence_file + evidence_key")
    print("  appended to audit/claims.csv with the same columns it already uses.")


if __name__ == "__main__":
    main()
