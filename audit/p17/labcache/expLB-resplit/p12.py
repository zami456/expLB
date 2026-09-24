import json, os
from collections import defaultdict
ROOT = r"C:/Users/user4/expLB-resplit/run-crossgen"
OLD  = r"C:/Users/user4/expLB/run-crossgen"
POOL, PROBE = "mean", "paper"
def od(d, f):
    idx = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}
    blk = idx.get((f, PROBE))
    if blk is None: return None
    n = len(d["datasets"])
    v = [blk[i][j] for i in range(n) for j in range(n) if i != j and blk[i][j] is not None]
    return sum(v)/len(v) if v else None
def scan(path):
    m = os.path.join(path, "matrices")
    if not os.path.isdir(m): return None
    out = {}
    for fn in sorted(os.listdir(m)):
        if not fn.startswith(POOL + "_L") or not fn.endswith("_s42.json"): continue
        d = json.load(open(os.path.join(m, fn), encoding="utf-8"))
        feats = sorted({r["feature"] for r in d["results"]})
        out[d["layer"]] = {f: od(d, f) for f in feats}
    return out
def summary(layers):
    if not layers: return None
    pers = [f for f in next(iter(layers.values())) if f.startswith("pc:") or f == "axes"]
    acc = defaultdict(list)
    for L, ft in layers.items():
        if ft.get("raw") is None: continue
        acc["raw"].append(ft["raw"])
        b = max((ft[f] for f in pers if ft.get(f) is not None), default=None)
        if b is not None: acc["best"].append(b)
        if ft.get("datapca") is not None: acc["dp"].append(ft["datapca"])
    if not acc["raw"]: return None
    return {k: sum(v)/len(v) for k, v in acc.items() if v}, len(acc["raw"])
print("=" * 86)
print("DOES THE kd (36-PERSONA) BANK'S ADVANTAGE HOLD AT SCALE?")
print("layer-averaged off-diagonal mean, 5 datasets, k=3, mean pooling, paper probe, seed 42")
print("=" * 86)
print(f"{'rung / bank':28s} {'n':>3s} {'raw':>8s} {'best pers':>10s} {'vs raw':>9s} {'datapca':>8s} {'vs dpca':>9s}")
rows = [("3B  paper16 (26p)", os.path.join(OLD, "scale_3b")),
        ("3B  kd all (36p)",  os.path.join(ROOT, "kdscale_3b_all")),
        ("3B  kd s23 (36p)",  os.path.join(ROOT, "kdscale_3b_s23")),
        ("8B  paper16 (26p)", os.path.join(OLD, "scale_8b")),
        ("8B  kd all (36p)",  os.path.join(ROOT, "kdscale_8b_all")),
        ("8B  kd s23 (36p)",  os.path.join(ROOT, "kdscale_8b_s23")),
        ("70B paper16 (26p)", os.path.join(OLD, "scale_70b_paper16fp32")),
        ("70B kd all (36p)",  os.path.join(ROOT, "kdscale_70b_all")),
        ("70B kd s23 (36p)",  os.path.join(ROOT, "kdscale_70b_s23"))]
for label, p in rows:
    lay = scan(p)
    s = summary(lay) if lay else None
    if not s: print(f"{label:28s}  (no data at {p})"); continue
    m, n = s
    raw, best, dp = m.get("raw"), m.get("best"), m.get("dp")
    dps = f"{dp:8.4f} {best-dp:+9.4f}" if dp else f"{'-':>8s} {'-':>9s}"
    print(f"{label:28s} {n:3d} {raw:8.4f} {best:10.4f} {best-raw:+9.4f} {dps}")
