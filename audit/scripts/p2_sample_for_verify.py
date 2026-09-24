"""Draw a reproducible random sample of Phase-1 rows for independent re-derivation (the mini Phase-4 check).
Seeded so the owner can re-draw the same sample. Writes audit/phase1/verify/sample.csv"""
import csv, os, random
random.seed(42)
os.makedirs("audit/phase1/verify", exist_ok=True)
rows = []
fmap = list(csv.DictReader(open("audit/phase1/fmap_all.csv", encoding="utf-8")))
sci = [r for r in fmap if r["claim_type"] == "scientific"]
for r in random.sample(sci, 8):
    rows.append(("fmap", r["f_id"], r["title_short"][:90]))
for fid in ("F-100", "F-65"):                      # two high-importance ones, not random
    r = next(x for x in fmap if x["f_id"] == fid)
    rows.append(("fmap-high", r["f_id"], r["title_short"][:90]))
exp = list(csv.DictReader(open("audit/experiments.csv", encoding="utf-8")))
for r in random.sample([x for x in exp if x["status"] == "complete"], 4):
    rows.append(("experiments", r["exp_id"], r["what_it_tests"][:90]))
anom = list(csv.DictReader(open("audit/phase1/anomalies.csv", encoding="utf-8")))
for r in random.sample(anom, 3):
    k = [v for v in r.values()][:3]
    rows.append(("anomalies", (r.get("path_or_group") or k[1])[:60], (r.get("detail") or "")[:90]))
sess = list(csv.DictReader(open("audit/phase1/sessions/findings_printed_numbers.csv", encoding="utf-8")))
for r in random.sample([x for x in sess if x.get("found") == "y"], 3):
    rows.append(("session-printed", r["f_id"], (r.get("f_claim_short") or "")[:90]))
with open("audit/phase1/verify/sample.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["source_table", "item", "what_phase1_says_short"]); w.writerows(rows)
for r in rows: print(f"{r[0]:16s} {r[1]:12s} {r[2]}")
print(len(rows), "items")
