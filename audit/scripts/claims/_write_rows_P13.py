"""Append rows to audit/phase2/claims_batch_P13-session.csv from a JSON list, idempotently."""
import csv, json, os, sys
COLS = ["id","claim_plain","stated_numbers","source_ref","evidence_file","evidence_key",
        "level1_value","level1_match","level2_value","level2_method","version_used",
        "robustness","verdict","confidence","importance","script","notes"]
OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..",
                                   "phase2","claims_batch_P13-session.csv"))
rows = json.load(open(sys.argv[1], encoding="utf-8"))
by = {}
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8", newline="") as f:
        by = {r["id"]: r for r in csv.DictReader(f)}
for r in rows:
    by[r["id"]] = {c: r.get(c, "") for c in COLS}
with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
    for i in sorted(by): w.writerow(by[i])
print("wrote", OUT, "rows:", len(by), "->", sorted(by))
