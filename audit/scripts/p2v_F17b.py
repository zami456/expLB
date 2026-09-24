"""F-17 second half: random 2-d projection gain where raw<0.6 vs raw>=0.85, and corr(raw, delta)."""
import json, statistics, sys
p = sys.argv[1]
probe = sys.argv[2] if len(sys.argv) > 2 else "paper"
d = json.load(open(p, encoding="utf-8"))
ds = d["datasets"]
idx = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}
print(f"{p}: L{d['layer']} {d['pooling']} k={d.get('k')} bank={d.get('bank')} name={d.get('bank_name')} n_datasets={len(ds)} probe={probe}")
print("features:", sorted({f for f, pr in idx if pr == probe}))
raw = idx[("raw", probe)]
for feat in ("random@2", "random", "pc:unified@2", "pc:unified", "datapca@2"):
    if (feat, probe) not in idx: continue
    blk = idx[(feat, probe)]
    pairs = []
    for i, tr in enumerate(ds):
        for j, te in enumerate(ds):
            if i == j: continue
            r, v = raw[i][j], blk[i][j]
            if r is None or v is None: continue
            pairs.append((r, v - r))
    lo = [dl for r, dl in pairs if r < 0.6]
    hi = [dl for r, dl in pairs if r >= 0.85]
    rs = [r for r, _ in pairs]; dls = [dl for _, dl in pairs]
    try: c = statistics.correlation(rs, dls)
    except Exception: c = float("nan")
    print(f"  {feat:<14} n={len(pairs):3d}  mean_delta={statistics.fmean(dls):+.4f}  "
          f"raw<0.6: n={len(lo):3d} mean={statistics.fmean(lo) if lo else float('nan'):+.4f}  "
          f"raw>=0.85: n={len(hi):3d} mean={statistics.fmean(hi) if hi else float('nan'):+.4f}  corr={c:+.4f}")
