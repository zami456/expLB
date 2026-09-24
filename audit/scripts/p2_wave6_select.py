"""Select the wave-6 claim batches — everything unchecked on the OWNER's side.

Owner's instruction 2026-09-23, after the decision to reframe the paper around its
controls: check her remaining claims (owner docs, memory notes, session-only
findings).  Her 110 ledger entries are already 110/110 done after wave 5.

Rationale for prioritising her side over the collaborator's 268 remaining rows:
the reframed thesis gets WRITTEN from her docs and memory notes, while his draft
body supports the positive framing that is being dropped, and all 18 of his
headline rows are already checked.

Batches are grouped BY DOCUMENT so each agent loads one context and one store.

Read-only.  Writes only audit/phase2/wave6_batches.csv.
"""

import csv
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
INV = ROOT / "audit" / "claims_inventory.csv"
CHK = ROOT / "audit" / "claims.csv"
OUT = ROOT / "audit" / "phase2" / "wave6_batches.csv"

MINE = {"owner doc", "memory note", "session (unrecorded)"}

# document -> batch.  Grouped so one agent holds one coherent context.
THESIS = "docs/thesis/persona-coordinates-70b-results.md"
QSERIES = {f"memory/q{n}-" for n in (2, 3, 4, 5, 6, 8, 10, 11)}


def norm(cid):
    return cid[2:] if cid.startswith("L-F-") else cid


def doc_of(row):
    return re.split(r":\d", row["source_ref"])[0].strip()


def batch_of(row, thesis_seen):
    d = doc_of(row)
    if d == THESIS:
        # split the biggest document across two agents, in id order
        return "W6A-thesis1" if thesis_seen[0] <= 17 else "W6B-thesis2"
    if d == "docs/COLLABORATOR_BRIEFING.md":
        return "W6C-briefing"
    if d.startswith("memory/") and any(d.startswith(q) for q in QSERIES):
        return "W6E-memory-q"
    if d.startswith("memory/"):
        return "W6F-memory-rest"
    if re.fullmatch(r"[0-9a-f]{8}", d):
        return "W6G-session"
    return "W6D-ownerdocs"


def main():
    inv = list(csv.DictReader(INV.open(encoding="utf-8")))
    done = {r["id"] for r in csv.DictReader(CHK.open(encoding="utf-8"))}
    left = [r for r in inv if norm(r["id"]) not in done and r["source"] in MINE]
    left.sort(key=lambda r: (doc_of(r), r["id"]))

    thesis_seen = [0]
    rows = []
    for r in left:
        if doc_of(r) == THESIS:
            thesis_seen[0] += 1
        rows.append((batch_of(r, thesis_seen), r))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["batch", "id", "source", "source_ref", "claim_short",
                    "numbers", "layer", "dataset_hint"])
        for b, r in sorted(rows, key=lambda x: (x[0], x[1]["id"])):
            w.writerow([b, r["id"], r["source"], r["source_ref"],
                        r["claim_short"], r["numbers"], r["layer"],
                        r["dataset_hint"]])

    counts = {}
    for b, _ in rows:
        counts[b] = counts.get(b, 0) + 1
    print(f"owner-side unchecked: {len(left)}")
    for b in sorted(counts):
        ids = [r["id"] for bb, r in rows if bb == b]
        print(f"  {b:18} n={counts[b]:3}  {', '.join(ids[:6])}"
              f"{' …' if len(ids) > 6 else ''}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
