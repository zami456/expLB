"""P17 QA: every row of each *.before-p17 backup still in the current CSV; classify every difference."""
import csv, pathlib, collections
A = pathlib.Path(__file__).resolve().parents[1]
TAG = "[P17 2026-09-24]"
def rd(p):
    with open(p, encoding="utf-8", newline="") as fh: return list(csv.DictReader(fh))
rep = {r["id"] for r in rd(A/"p17/d_rows_repinned.csv")}
oth = {r["id"] for r in rd(A/"p17/d_rows_other_affected.csv")}
for f, key in [("claims.csv","id"),("claims_inventory.csv","id"),("experiments.csv","exp_id"),("phase1/fmap_all.csv","f_id")]:
    old = rd(A/(f+".before-p17")); new = rd(A/f)
    nb = {r[key]: r for r in new}
    ids_old=[r[key] for r in old]; ids_new=[r[key] for r in new]
    order_ok = ids_new[:len(ids_old)] == ids_old
    dup = [k for k,c in collections.Counter(ids_new).items() if c>1]
    bad = []; kinds = collections.Counter()
    for r in old:
        n = nb.get(r[key])
        if n is None: bad.append((r[key],"MISSING")); continue
        for c, v in r.items():
            w = n.get(c)
            if w == v: continue
            if f=="claims.csv" and c in ("notes","version_used") and w.startswith(v) and TAG in w[len(v):]:
                if c=="version_used" and r[key] not in rep: bad.append((r[key],c,"appended but not repinned row"))
                if c=="notes" and r[key] not in rep|oth: bad.append((r[key],c,"note on unexpected row"))
                kinds[(c, "rep" if r[key] in rep else "oth")] += 1; continue
            if f=="claims.csv" and c=="verdict" and r[key]=="D-189":
                kinds["verdict D-189"] += 1; print("  D-189 verdict:", v, "=>", w[:200]); continue
            bad.append((r[key], c, v[:80], (w or "")[:80]))
    print(f, len(old), "->", len(new), "order_ok", order_ok, "dups", dup, dict(kinds))
    print("  new ids:", [k for k in ids_new if k not in set(ids_old)][:3], "...", len(set(ids_new)-set(ids_old)))
    for b in bad[:20]: print("  BAD", b)
    print("  n bad", len(bad))
print("repinned ids", len(rep), "other", len(oth), "overlap", rep & oth)
