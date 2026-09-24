"""Merge the Phase 2 batch files into audit/claims.csv and print the running tally.

Reads every audit/phase2/claims_batch_*.csv, normalises the columns defined in
audit/phase2/CLAIM_INSTRUCTIONS.md, marks which claims of audit/claims_inventory.csv are still TODO,
and reports verdict / confidence / importance counts so PROGRESS.md can quote them.
"""
import collections, csv, glob, os

COLS = ["id", "claim_plain", "stated_numbers", "source_ref", "evidence_file", "evidence_key", "level1_value",
        "level1_match", "level2_value", "level2_method", "version_used", "robustness", "verdict", "confidence",
        "importance", "script", "notes", "batch"]

rows, seen = [], {}
for f in sorted(glob.glob("audit/phase2/claims_batch_*.csv")):
    batch = os.path.basename(f)[len("claims_batch_"):-4]
    for r in csv.DictReader(open(f, encoding="utf-8")):
        r = {k: (r.get(k) or "") for k in COLS if k != "batch"}
        r["batch"] = batch
        if not r["id"]:
            continue
        if r["id"] in seen:            # a later batch supersedes an earlier row for the same claim
            rows[seen[r["id"]]] = r
        else:
            seen[r["id"]] = len(rows); rows.append(r)

with open("audit/claims.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
    for r in rows: w.writerow(r)

inv = list(csv.DictReader(open("audit/claims_inventory.csv", encoding="utf-8")))
def norm(i): return i.replace("L-", "").strip()
done = {norm(r["id"]) for r in rows}
todo = [r for r in inv if norm(r["id"]) not in done]

def norm_verdict(v):
    """The agents write rich free text ('YES (tol 5e-4)', 'PARTLY HOLDS - the three numbers ...').
    Count the category, keep the prose in the CSV."""
    v = (v or "").strip().upper()
    if not v:
        return "(blank)"
    keys = ("DOES NOT MATCH", "DOES NOT HOLD", "CONFLICTING", "NO RAW DATA", "NOT TESTED",
            "NOT VERIFIABLE", "NOT CHECKABLE", "PARTLY", "PARTIAL", "REVISED",
            "MATCHES HIS DATA", "HOLDS", "YES", "NO", "N/A")
    for key in keys:                       # a prefix always wins: the verdict's own first word
        if v.startswith(key):
            return key
    for key in keys:                       # only then fall back to a mention inside the prose
        if key in v:
            return key
    return v.split(" ")[0]


print(f"claims checked: {len(rows)} / {len(inv)} in the inventory   ({len(todo)} TODO)")
for field in ("verdict", "confidence", "importance", "level1_match"):
    key = norm_verdict if field in ("verdict", "level1_match") else (lambda x: (x or "(blank)").split(" ")[0])
    c = collections.Counter(key(r[field]) for r in rows)
    print(f"  {field:12s} {dict(sorted(c.items(), key=lambda kv: -kv[1]))}")
by_batch = collections.Counter(r["batch"] for r in rows)
print("  per batch:", dict(by_batch))
bad = [r for r in rows if r["verdict"].startswith(("DOES NOT", "CONFLICTING", "PARTLY"))]
if bad:
    print("\nclaims that failed or changed:")
    for r in bad:
        print(f"  {r['id']:10s} {r['verdict']:18s} {r['claim_plain'][:90]}")
missing_by_source = collections.Counter(r["source"] for r in todo)
print("\nTODO by source:", dict(missing_by_source))
