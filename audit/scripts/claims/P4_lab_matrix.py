"""READ-ONLY on the lab PC. Level-2: recompute the off-diagonal mean AUROC straight from the
crossgen matrix JSONs, independently of audit/phase2/layer_sensitivity.csv, for the cells
F-63 / F-100 / F-101 / F-102 / F-110 quote. Writes nothing."""
import glob, json, os, re, statistics

R = r"C:\Users\user4\expLB\run-crossgen"


def offdiag(mat):
    n = len(mat)
    v = [mat[i][j] for i in range(n) for j in range(n)
         if i != j and mat[i][j] is not None]
    return (sum(v) / len(v)) if v else float("nan"), len(v)


def cell(store, pooling, layer, probe="paper"):
    p = os.path.join(R, store, "matrices", f"{pooling}_L{layer}_s42.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    out = {"_datasets": d["datasets"], "_bank": d.get("bank"),
           "_bank_name": d.get("bank_name"), "_k": d.get("k"),
           "_mtime": os.path.getmtime(p)}
    for r in d["results"]:
        if r.get("probe") == probe:
            m, n = offdiag(r["auroc"])
            out[r["feature"]] = m
            out["_npairs"] = n
    for z in d.get("zeroshot", []):
        a = z["auroc"]
        if a and isinstance(a[0], list):
            m, _ = offdiag(a)
        else:
            m = sum(x for x in a if x is not None) / max(1, len([x for x in a if x is not None]))
        out["zs:" + z["direction"]] = m
    return out


print("= Level-2 direct recompute from the matrix JSONs (probe=paper, seed 42) =")
CELLS = [("scale_3b", "mean", 16), ("scale_8b", "mean", 16),
         ("scale_70b", "mean", 40), ("scale_70b_paper16fp32", "mean", 40),
         ("scale_70b", "last", 40), ("scale_70b_paper16fp32", "last", 40),
         ("kd11", "mean", 40)]
for st, pool, lay in CELLS:
    c = cell(st, pool, lay)
    if c is None:
        print(f"  {st}/{pool}_L{lay}: MISSING")
        continue
    import datetime
    raw = c.get("raw")
    print(f"\n  {st}  {pool}_L{lay}  bank={c['_bank']} bank_name={c['_bank_name']} k={c['_k']} "
          f"pairs={c.get('_npairs')} mtime={datetime.datetime.fromtimestamp(c['_mtime']):%Y-%m-%d %H:%M}")
    print(f"     datasets: {c['_datasets']}")
    for f in ("raw", "axes", "pc:deception", "pc:unified", "pc:unified@2",
              "datapca", "datapca@2", "random", "random@2", "length"):
        if f in c:
            print(f"     {f:14s} {c[f]:.4f}   (- raw {c[f]-raw:+.4f})")
    for f in sorted(k for k in c if k.startswith("zs:")):
        print(f"     {f:14s} {c[f]:.4f}")

print()
print("= layer-means over all stored layers, both poolings (independent of the W1 csv) =")
for st in ("scale_3b", "scale_8b", "scale_70b", "scale_70b_paper16fp32"):
    for pool in ("mean", "last"):
        lays = sorted(int(re.search(r"_L(\d+)_", os.path.basename(p)).group(1))
                      for p in glob.glob(os.path.join(R, st, "matrices", f"{pool}_L*_s42.json")))
        rec = []
        for l in lays:
            c = cell(st, pool, l)
            if c and "raw" in c and "datapca" in c:
                best = max(c[f] for f in ("axes", "pc:deception") if f in c)
                rec.append((l, c["raw"], best - c["raw"], best - c["datapca"],
                            c["axes"] - c["datapca"]))
        if not rec:
            continue
        br = [r[2] for r in rec]; bd = [r[3] for r in rec]; ad = [r[4] for r in rec]
        print(f"  {st:24s} {pool:5s} n={len(rec):2d} layers={lays}")
        print(f"      raw={statistics.mean(r[1] for r in rec):.4f}  "
              f"best-raw={statistics.mean(br):+.4f} ({sum(1 for x in br if x>0)}/{len(br)})  "
              f"best-dpca={statistics.mean(bd):+.4f} ({sum(1 for x in bd if x>0)}/{len(bd)})  "
              f"axes-dpca={statistics.mean(ad):+.4f}")
        if st.startswith("scale_70b"):
            sub = [r for r in rec if r[0] not in (72, 76)]
            print(f"      [F-100 19 layers] raw={statistics.mean(r[1] for r in sub):.4f}  "
                  f"best-raw={statistics.mean(r[2] for r in sub):+.4f} "
                  f"({sum(1 for r in sub if r[2]>0)}/{len(sub)})  "
                  f"best-dpca={statistics.mean(r[3] for r in sub):+.4f} "
                  f"({sum(1 for r in sub if r[3]>0)}/{len(sub)})  "
                  f"axes-dpca={statistics.mean(r[4] for r in sub):+.4f}")
