"""Select the wave-5 claim batches.

Wave 5 (owner's decision 2026-09-23): the 32 remaining ledger entries in full,
plus a seeded random sample of the never-sampled memory notes and the owner
docs.  Rationale is in audit/PROGRESS.md: the ledger remainder has the highest
observed defect rate (P9: 3 HOLDS / 7 PARTLY) and is a *source* rather than a
restatement; the memory notes have never been sampled at all.

Read-only.  Writes only audit/phase2/wave5_batches.csv.
"""

import csv
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[2]
INV = ROOT / "audit" / "claims_inventory.csv"
CHK = ROOT / "audit" / "claims.csv"
OUT = ROOT / "audit" / "phase2" / "wave5_batches.csv"

SEED = 42
N_MEMORY = 12
N_OWNER = 10


def norm(claim_id):
    """claims_inventory uses L-F-07; claims.csv uses F-07."""
    return claim_id[2:] if claim_id.startswith("L-F-") else claim_id


def main():
    inv = list(csv.DictReader(INV.open(encoding="utf-8")))
    done = {r["id"] for r in csv.DictReader(CHK.open(encoding="utf-8"))}
    left = [r for r in inv if norm(r["id"]) not in done]

    ledger = [r for r in left if r["source"] == "FINDINGS.md"]
    memory = [r for r in left if r["source"] == "memory note"]
    owner = [r for r in left if r["source"] == "owner doc"]

    rng = random.Random(SEED)
    mem_pick = sorted(rng.sample(memory, min(N_MEMORY, len(memory))), key=lambda r: r["id"])
    own_pick = sorted(rng.sample(owner, min(N_OWNER, len(owner))), key=lambda r: r["id"])

    # Ledger split into three batches of ~11, in id order, so each agent gets a
    # contiguous slice of the ledger and can reuse one loaded store.
    ledger = sorted(ledger, key=lambda r: r["id"])
    thirds = [ledger[i::3] for i in range(3)]

    batches = [
        ("W5A-ledger", thirds[0]),
        ("W5B-ledger", thirds[1]),
        ("W5C-ledger", thirds[2]),
        ("W5D-memory", mem_pick),
        ("W5E-ownerdoc", own_pick),
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["batch", "id", "source", "source_ref", "claim_short", "numbers",
                    "layer", "dataset_hint"])
        for name, rows in batches:
            for r in rows:
                w.writerow([name, r["id"], r["source"], r["source_ref"],
                            r["claim_short"], r["numbers"], r["layer"],
                            r["dataset_hint"]])

    print(f"unchecked total: {len(left)}")
    print(f"pools: ledger {len(ledger)}, memory {len(memory)}, owner {len(owner)}")
    print(f"seed {SEED}; memory sample {len(mem_pick)}, owner sample {len(own_pick)}")
    for name, rows in batches:
        print(f"  {name:14} n={len(rows):3}  {', '.join(r['id'] for r in rows)}")
    print(f"total selected: {sum(len(r) for _, r in batches)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
