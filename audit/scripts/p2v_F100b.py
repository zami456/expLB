"""F-100 variant: 'best' restricted to an explicit feature list (default axes + pc:deception)."""
import glob, json, os, re, statistics, sys

BESTSET = sys.argv[1].split(",")
def offdiag(idx, feat, probe, ds):
    a = idx.get((feat, probe))
    if a is None: return None
    v = [a[i][j] for i in range(len(ds)) for j in range(len(ds)) if i != j and a[i][j] is not None]
    return statistics.fmean(v) if v else None

for d in sys.argv[2:]:
    files = sorted(glob.glob(os.path.join(d, "matrices", "mean_L*_s42.json")),
                   key=lambda p: int(re.search(r"_L(\d+)_", p).group(1)))
    ls, bd, br, ad, ar, feats_used, allf = [], [], [], [], [], [], {}
    for p in files:
        j = json.load(open(p, encoding="utf-8")); ds = j["datasets"]
        idx = {(r["feature"], r["probe"]): r["auroc"] for r in j["results"]}
        m = {f: offdiag(idx, f, "paper", ds) for f in sorted({a for a, b in idx if b == "paper"})}
        allf[j["layer"]] = m
        cands = [(m[f], f) for f in BESTSET if m.get(f) is not None]
        b, bf = max(cands)
        ls.append(j["layer"]); feats_used.append(bf)
        bd.append(b - m["datapca"]); br.append(b - m["raw"])
        ad.append(m["axes"] - m["datapca"]); ar.append(m["axes"] - m["raw"])
    print(f"\n##### {os.path.basename(d)}  n={len(ls)} bestset={BESTSET}")
    print(f"  {'L':>4} {'best':>8} {'bestfeat':>14} {'b-dpca':>9} {'b-raw':>8} {'axes-raw':>9} {'pcdec-raw':>10}")
    for i, L in enumerate(ls):
        m = allf[L]
        pd = m.get("pc:deception")
        print(f"  {L:>4} {max(m[f] for f in BESTSET if m.get(f) is not None):>8.4f} {feats_used[i]:>14} "
              f"{bd[i]:>+9.4f} {br[i]:>+8.4f} {ar[i]:>+9.4f} {(pd - m['raw']) if pd is not None else float('nan'):>+10.4f}")
    print(f"  LAYER-MEAN best-datapca={statistics.fmean(bd):+.4f}  best-raw={statistics.fmean(br):+.4f}  "
          f"axes-datapca={statistics.fmean(ad):+.4f}  best-raw positive at {sum(1 for v in br if v>0)}/{len(br)}")
    for L in (16, 40):
        if L in ls:
            i = ls.index(L)
            rk = sorted(range(len(bd)), key=lambda k: -bd[k]).index(i) + 1
            print(f"  L{L}: best-datapca={bd[i]:+.4f} rank {rk}/{len(ls)}; best-raw={br[i]:+.4f}; axes-datapca={ad[i]:+.4f}")
    if 72 in ls and 76 in ls:
        keep = [i for i, L in enumerate(ls) if L not in (72, 76)]
        i40 = ls.index(40)
        rk = sorted(keep, key=lambda k: -bd[k]).index(i40) + 1
        print(f"  [19 layers] best-datapca={statistics.fmean([bd[i] for i in keep]):+.4f} best-raw={statistics.fmean([br[i] for i in keep]):+.4f} "
              f"axes-datapca={statistics.fmean([ad[i] for i in keep]):+.4f} positive {sum(1 for i in keep if br[i]>0)}/19 L40 rank {rk}/19")
