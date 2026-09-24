"""Phase 1: summarise the lab census and match the collaborator's input files (his branch) against the lab PC by md5."""
import csv, collections, sys
csv.field_size_limit(10**9)
lab = list(csv.DictReader(open("audit/phase1/labpc/labpc_census.tsv", encoding="utf-8"), delimiter="\t"))
print("lab files:", len(lab), "| SKIP_BIG:", sum(r["md5"] == "SKIP_BIG" for r in lab), "| ERR:", sum(r["md5"] == "ERR" for r in lab))
top = collections.Counter()
for r in lab:
    k = r["root"].split("\\")[-1] + "/" + (r["relpath"].split("/")[0] if "/" in r["relpath"] else "<loose>")
    top[k] += 1
for k, v in sorted(top.items()):
    print(f"  {v:7d}  {k}")
by_md5 = collections.defaultdict(list)
for r in lab:
    by_md5[r["md5"]].append(r)
pc = list(csv.DictReader(open("audit/phase1/census_pcood.tsv", encoding="utf-8"), delimiter="\t"))
print("\n== his input files (.pt/.npz/.json under non-results dirs) vs lab md5")
hits = miss = 0
for r in pc:
    p = r["path"]
    if "/results/" in p or not p.endswith((".pt", ".npz", ".json")) or "/deception_detection/" in p or "/data/" in p or "/example_results/" in p:
        continue
    m = by_md5.get(r["md5"])
    if m:
        hits += 1
        print("MATCH", p.split("deception-detection/")[1], "->", "; ".join(x["relpath"] for x in m[:3]))
    else:
        miss += 1
        print("no-match", p.split("deception-detection/")[1], r["bytes"])
print("matched", hits, "unmatched", miss)
