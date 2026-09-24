"""F-100: layer-mean of (best persona - datapca) and (best persona - raw) for the 3B/8B/70B rungs."""
import glob, json, os, re, statistics, sys

def offdiag(idx, feat, probe, ds):
    a = idx.get((feat, probe))
    if a is None: return None
    v = [a[i][j] for i in range(len(ds)) for j in range(len(ds)) if i != j and a[i][j] is not None]
    return statistics.fmean(v) if v else None

def rung(d, pooling="mean", probe="paper"):
    files = sorted(glob.glob(os.path.join(d, "matrices", f"{pooling}_L*_s42.json")),
                   key=lambda p: int(re.search(r"_L(\d+)_", p).group(1)))
    rows = []
    for p in files:
        j = json.load(open(p, encoding="utf-8"))
        ds = j["datasets"]
        idx = {(r["feature"], r["probe"]): r["auroc"] for r in j["results"]}
        feats = sorted({f for f, pr in idx if pr == probe})
        m = {f: offdiag(idx, f, probe, ds) for f in feats}
        personas = [f for f in feats if f.startswith("pc:") or f in ("axes", "axes_resid")]
        personas = [f for f in personas if f != "axes_resid"]
        best = max(((m[f], f) for f in personas if m[f] is not None), default=(None, None))
        rows.append(dict(layer=j["layer"], k=j.get("k"), bank=j.get("bank"), bank_name=j.get("bank_name"),
                         nds=len(ds), datasets=ds, feats=feats, m=m, best=best[0], best_feat=best[1],
                         mtime=os.path.getmtime(p), zs=[z["direction"] for z in j.get("zeroshot", [])]))
    return rows

for d in sys.argv[1:]:
    rows = rung(d)
    if not rows: print(d, "NO MATRICES"); continue
    r0 = rows[0]
    print(f"\n########## {d}  n_layers={len(rows)} k={r0['k']} bank={r0['bank']} name={r0['bank_name']}")
    print(f"  datasets ({r0['nds']}): {r0['datasets']}")
    print(f"  persona features present: {[f for f in r0['feats'] if f.startswith('pc:') or f=='axes']}  zeroshot dirs: {r0['zs']}")
    print(f"  {'L':>4} {'raw':>7} {'datapca':>8} {'axes':>7} {'best':>7} {'bestfeat':>14} {'best-dpca':>10} {'best-raw':>9} {'axes-dpca':>10} {'axes-raw':>9}")
    bd, br, ad, ar = [], [], [], []
    for r in rows:
        m = r["m"]; raw = m.get("raw"); dp = m.get("datapca"); ax = m.get("axes")
        b = r["best"]
        bd.append(b - dp); br.append(b - raw); ad.append(ax - dp); ar.append(ax - raw)
        print(f"  {r['layer']:>4} {raw:>7.4f} {dp:>8.4f} {ax:>7.4f} {b:>7.4f} {r['best_feat']:>14} "
              f"{b-dp:>+10.4f} {b-raw:>+9.4f} {ax-dp:>+10.4f} {ax-raw:>+9.4f}")
    def rank(vals, layer_list, L):
        order = sorted(range(len(vals)), key=lambda i: -vals[i])
        return order.index(layer_list.index(L)) + 1 if L in layer_list else None
    ls = [r["layer"] for r in rows]
    print(f"  LAYER-MEANS over {len(rows)} layers: best-datapca={statistics.fmean(bd):+.4f}  best-raw={statistics.fmean(br):+.4f}  "
          f"axes-datapca={statistics.fmean(ad):+.4f}  axes-raw={statistics.fmean(ar):+.4f}")
    print(f"  best-raw positive at {sum(1 for v in br if v > 0)}/{len(br)}")
    for L in (16, 40):
        if L in ls:
            i = ls.index(L)
            print(f"  L{L}: best-datapca={bd[i]:+.4f} rank {rank(bd, ls, L)} of {len(ls)}; best-raw={br[i]:+.4f}; axes-datapca={ad[i]:+.4f}")
    # 19-layer variant (drop L72, L76) for the 70B rung
    if 72 in ls and 76 in ls:
        keep = [i for i, L in enumerate(ls) if L not in (72, 76)]
        print(f"  [19-layer variant, dropping L72/L76] best-datapca={statistics.fmean([bd[i] for i in keep]):+.4f} "
              f"best-raw={statistics.fmean([br[i] for i in keep]):+.4f} axes-datapca={statistics.fmean([ad[i] for i in keep]):+.4f} "
              f"best-raw positive at {sum(1 for i in keep if br[i] > 0)}/{len(keep)}; "
              f"L40 rank {sorted(keep, key=lambda i: -bd[i]).index(ls.index(40))+1} of {len(keep)}")
        for L in (72, 76):
            i = ls.index(L); print(f"  L{L}: best-raw={br[i]:+.4f}")
