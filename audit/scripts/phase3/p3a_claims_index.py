import csv, sys
p = "audit/claims.csv"
rows = list(csv.DictReader(open(p, encoding="utf-8")))
print("n rows:", len(rows))
print("cols:", list(rows[0].keys()))
for r in rows:
    print("|".join([r["id"], r["verdict"], r["confidence"][:22], r["importance"], r["claim_plain"][:150].replace("\n"," ")]))
