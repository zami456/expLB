"""Phase 0: compare lab-PC doc list (audit/phase0/labpc_docs.csv: path,bytes,mtime,md5) with the local working tree by md5."""
import csv, hashlib, os, collections
BS = chr(92)
BASE = "C:/Users/SURAIYA/Documents/STUDY/thesis/expLB"
PREFIX = "C:/Users/user4/expLB/"
rows = [r for r in csv.reader(open(BASE + "/audit/phase0/labpc_docs.csv", encoding="utf-8", errors="replace"))
        if len(r) == 4 and r[0] != "path"]
out, by_dir = [], collections.Counter()
for p, b, m, h in rows:
    p = p.replace(BS, "/")
    if p.startswith(PREFIX):
        rel = p[len(PREFIX):]
        by_dir["expLB/" + rel.split("/")[0]] += 1
        lp = BASE + "/" + rel
        st = ("SAME" if hashlib.md5(open(lp, "rb").read()).hexdigest().upper() == h else "DIFFERENT") if os.path.isfile(lp) else "NOT-LOCAL"
    else:
        rel = p; by_dir["/".join(p.split("/")[:3])] += 1; st = "NOT-LOCAL"
    out.append((st, rel, int(b), m))
print("docs per folder:", dict(by_dir))
print("status counts:", dict(collections.Counter(s for s, *_ in out)))
for s, rel, b, m in sorted(out):
    if s != "SAME":
        print(f"{s:10s} {m} {b:8d} {rel}")
