import json, os, statistics
B = "C:/Users/user4/expLB/run-crossgen"
def load(store, fn):
    p = os.path.join(B, store, "matrices", fn) if store else os.path.join(B, "matrices", fn)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
def rowmeans(m, feat, probe):
    ds = m["datasets"]
    blk = None
    for r in m["results"]:
        if r["feature"] == feat and r["probe"] == probe:
            blk = r["auroc"]
    if blk is None:
        return None
    out = {}
    for i, tr in enumerate(ds):
        row = [v for j, v in enumerate(blk[i]) if j != i and v is not None]
        col = [blk[j][i] for j in range(len(ds)) if j != i and blk[j][i] is not None]
        out[tr] = (blk[i][i], statistics.fmean(row) if row else None,
                   statistics.fmean(col) if col else None)
    return out
for pool in ("mean", "last"):
    for L in (20, 40, 44):
        fn = "%s_L%d_s42.json" % (pool, L)
        a = load("passP", fn); b = load("", fn)
        if a is None or b is None:
            print("missing", fn, a is None, b is None); continue
        for probe in ("paper",):
            for feat in ("raw",):
                ra, rb = rowmeans(a, feat, probe), rowmeans(b, feat, probe)
                print("=== passP vs root  %s L%d %s %s" % (pool, L, feat, probe))
                print("%-36s %8s %8s %8s | %8s %8s %8s" % ("dataset", "P diag", "P row", "P col", "R diag", "R row", "R col"))
                for d in sorted(ra):
                    x, y = ra[d], rb.get(d)
                    if y is None: continue
                    print("%-36s %8.4f %8.4f %8.4f | %8.4f %8.4f %8.4f" % (
                        d, x[0], x[1], x[2], y[0], y[1], y[2]))
print("DONE")
