"""Append rows to audit/phase2/claims_batch_P3-kdbank-cap.csv from a JSON list, idempotently
(a row with the same id replaces the earlier one). Keeps the batch file safe against interruption.

Usage: python _write_rows.py <rows.json>
"""
import csv, json, os, sys

COLS = ["id", "claim_plain", "stated_numbers", "source_ref", "evidence_file", "evidence_key",
        "level1_value", "level1_match", "level2_value", "level2_method", "version_used",
        "robustness", "verdict", "confidence", "importance", "script", "notes"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "phase2", "claims_batch_P3-kdbank-cap.csv")
OUT = os.path.abspath(OUT)

rows = json.load(open(sys.argv[1], encoding="utf-8"))
existing = []
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8", newline="") as f:
        existing = list(csv.DictReader(f))
by = {r["id"]: r for r in existing}
for r in rows:
    by[r["id"]] = {c: r.get(c, "") for c in COLS}
order = sorted(by, key=lambda i: (len(i), i))
with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    for i in order:
        w.writerow(by[i])
print("wrote", OUT, "rows:", len(order), "->", order)
