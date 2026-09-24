"""Append cached lab files to audit/cache/MANIFEST.csv, verifying each local copy against the lab census md5.
Usage: python audit/scripts/p1_manifest_add.py <method> <relpath under C:/Users/user4/expLB> [...]"""
import csv, datetime, gzip, hashlib, os, sys
method, rels = sys.argv[1], set(sys.argv[2:])
lab = {}
with gzip.open("audit/phase1/labpc/joined_labpc.tsv.gz", "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["root"].endswith("expLB") and r["relpath"] in rels:
            lab[r["relpath"]] = r
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M"); rows = []; bad = 0
for rel in sorted(rels):
    local = "audit/cache/labpc/" + rel
    lm = hashlib.md5(open(local, "rb").read()).hexdigest()
    r = lab.get(rel, {})
    ok = lm == r.get("md5")
    bad += not ok
    rows.append([now, "labpc", "C:/Users/user4/expLB/" + rel, r.get("bytes", "?"), r.get("md5", "?"), local, os.path.getsize(local), lm,
                 "MATCH" if ok else "MISMATCH", method])
with open("audit/cache/MANIFEST.csv", "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)
print(len(rows), "rows added;", bad, "mismatches")
