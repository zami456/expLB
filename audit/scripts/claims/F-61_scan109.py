import glob, json, os, statistics
ROOT = r"C:\Users\SURAIYA\Documents\STUDY\thesis\expLB"
hits = []
for p in glob.glob(os.path.join(ROOT, "run-crossgen", "**", "matrices", "*.json"), recursive=True):
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    ds = d.get("datasets")
    if not ds:
        continue
    idx = {(r["feature"], r["probe"]): r["auroc"] for r in d.get("results", [])}
    blk = idx.get(("raw", "paper"))
    if blk is None:
        continue
    vals = [v for i, row in enumerate(blk) for j, v in enumerate(row) if i != j and v is not None]
    if not vals:
        continue
    m = statistics.fmean(vals)
    if len(vals) == 109 or abs(m - 0.6800) < 0.0006:
        u = idx.get(("pc:unified", "paper"))
        uv = [v for i, row in enumerate(u) for j, v in enumerate(row) if i != j and v is not None] if u else []
        hits.append((os.path.relpath(p, ROOT), d.get("layer"), d.get("pooling"), d.get("seed"),
                     len(ds), len(vals), round(m, 4),
                     round(statistics.fmean(uv), 4) if uv else None,
                     round(statistics.fmean(uv) - m, 4) if uv else None))
print(f"scanned; hits with 109 pairs or raw~0.6800:")
for h in sorted(hits):
    print("  ", h)
