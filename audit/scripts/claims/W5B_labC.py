"""W5B-ledger batch, lab-PC pass C - READ-ONLY.  F-82 and F-91.

F-82: re-implements cap/scripts/cap_layer_profile.py (same statistic, same rng seed 42, B=10000)
      for banks cap_s23 (F-77 column) and cap_all (F-82 column), mean pooling, feature
      pc:capitulation@2, control datapca@2, pre-specified layer 40.
F-91: off-diagonal mean over the whole 11x11 grid (paper probe, seed 42) for
      cap_s23/raw, cap_s23/pc:capitulation@2, kd_s23/axes at every stored layer -> peak vs L40;
      plus the falsify 'real' values at L32/L36/L40 for every stored seed (the seed-mixing check);
      plus a layer-coverage audit of run-crossgen/**/matrices.
Writes nothing.
"""
import collections, glob, json, os, re

import numpy as np

R = "C:/Users/user4/expLB"
XG = f"{R}/run-crossgen"


def sec(t):
    print("\n" + "=" * 78 + "\n== " + t + "\n" + "=" * 78)


def mat(path, feature, probe="paper"):
    d = json.load(open(path, encoding="utf-8"))
    for r in d["results"]:
        if r["probe"] == probe and r["feature"] == feature:
            return d["datasets"], r["auroc"]
    return d["datasets"], None


# =============================================================== F-82 ======
sec("F-82  cap layer profile, filtered (cap_s23) vs unfiltered (cap_all)")
DS = "sycophancy_mt"


def load_col(root, feats, pooling="mean"):
    out = {}
    for f in glob.glob(f"{root}/matrices/{pooling}_L*_s42.json"):
        lay = int(re.search(rf"{pooling}_L(\d+)_s42", f).group(1))
        d = json.load(open(f, encoding="utf-8"))
        if DS not in d["datasets"]:
            continue
        j = d["datasets"].index(DS)
        n = len(d["datasets"])
        for r in d["results"]:
            if r["probe"] == "paper" and r["feature"] in feats:
                A = r["auroc"]
                col = [A[i][j] for i in range(n) if i != j]
                out.setdefault(lay, {})[r["feature"]] = np.array(
                    [np.nan if v is None else v for v in col], dtype=float)
    return out


for bank in ("cap_s23", "cap_all"):
    root = f"{XG}/kd11/banks/{bank}"
    print(f"\n---- bank {bank}   ({'EXISTS' if os.path.isdir(root) else 'MISSING'})")
    if not os.path.isdir(root):
        continue
    nmat = len(glob.glob(f"{root}/matrices/*.json"))
    print(f"     {nmat} matrix files; mtimes "
          f"{sorted({__import__('datetime').datetime.fromtimestamp(os.path.getmtime(p)).strftime('%m-%d %H:%M') for p in glob.glob(root+'/matrices/*.json')})[:3]} ...")
    cells = load_col(root, {"pc:capitulation@2", "datapca@2", "raw"})
    have = [L for L in sorted(cells) if all(k in cells[L] for k in ("pc:capitulation@2", "datapca@2", "raw"))]
    print(f"     layers with feature+control+raw: {len(have)} -> {have}")
    print(f"     {'layer':>6}{'feat':>8}{'raw':>8}{'ctrl':>8}{'f-raw':>8}{'f-ctrl':>8}")
    for L in have:
        f_, r_, c_ = (cells[L][k].mean() for k in ("pc:capitulation@2", "raw", "datapca@2"))
        print(f"     {L:>6}{f_:>8.3f}{r_:>8.3f}{c_:>8.3f}{f_-r_:>+8.3f}{f_-c_:>+8.3f}"
              + ("   <- pre-specified" if L == 40 else ""))
    M = np.vstack([cells[L]["pc:capitulation@2"] - cells[L]["datapca@2"] for L in have])
    M = M[:, ~np.isnan(M).any(axis=0)]
    n_src = M.shape[1]
    pos = int((M.mean(axis=1) > 0).sum())
    run = best = 0
    for v in M.mean(axis=1):
        run = run + 1 if v > 0 else 0
        best = max(best, run)
    print(f"     margin over datapca@2 positive at {pos}/{len(have)} layers; longest run {best}; n_src={n_src}")
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n_src, size=(10000, n_src))
    li = have.index(40)
    pre = M[li][idx].mean(axis=1)
    lo, hi = np.percentile(pre, [2.5, 97.5])
    print(f"     L40 margin  point {M[li].mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"P(<=0) {float((pre<=0).mean()):.4f}{'  CLEARS' if lo>0 else '  does not clear'}")
    allb = M[:, idx].mean(axis=2)
    arg = np.array(have)[allb.argmax(axis=0)]
    alo, ahi = np.percentile(arg, [2.5, 97.5])
    print(f"     argmax layer  median L{int(np.median(arg))}  95% [L{int(alo)}, L{int(ahi)}]  "
          f"lands on L40 in {float((arg==40).mean()):.1%} of resamples")

# =============================================================== F-91 ======
sec("F-91  peak-layer table: off-diagonal mean over the whole grid, paper probe, seed 42")


def offdiag(bank, feat, lay, pooling="mean", seed=42):
    p = f"{XG}/kd11/banks/{bank}/matrices/{pooling}_L{lay}_s{seed}.json"
    if not os.path.exists(p):
        return None, None
    ds, A = mat(p, feat)
    if A is None:
        return None, len(ds)
    v = [A[i][j] for i in range(len(ds)) for j in range(len(ds)) if i != j and A[i][j] is not None]
    return (float(np.mean(v)) if v else None), len(ds)


for bank, feat in (("cap_s23", "raw"), ("cap_s23", "pc:capitulation@2"), ("kd_s23", "axes")):
    root = f"{XG}/kd11/banks/{bank}/matrices"
    lays = sorted({int(m.group(1)) for f in glob.glob(root + "/mean_L*_s42.json")
                   if (m := re.search(r"mean_L(\d+)_s42", f))})
    vals = {}
    nds = None
    for L in lays:
        v, n = offdiag(bank, feat, L)
        if v is not None:
            vals[L] = v; nds = n
    if not vals:
        print(f"\n  {bank}/{feat}: no values"); continue
    pk = max(vals, key=vals.get)
    print(f"\n  {bank} / {feat}  ({len(vals)} layers, {nds} datasets)")
    print(f"    peak L{pk} = {vals[pk]:.4f}   L40 = {vals.get(40, float('nan')):.4f}   "
          f"peak-L40 = {vals[pk]-vals.get(40, float('nan')):+.4f}")
    print("    profile: " + "  ".join(f"L{L}:{vals[L]:.3f}" for L in sorted(vals)))

sec("F-91  falsify sweep 'real' values (the mean-of-directions probe) - the seed-mixing check")
fd = f"{R}/run-premise/falsify"
rows = []
for p in sorted(glob.glob(fd + "/falsify_L*_s*.json")):
    m = re.search(r"falsify_L(\d+)_s(\d+)\.json", os.path.basename(p))
    j = json.load(open(p, encoding="utf-8"))
    real = j.get("real")
    if isinstance(real, dict):
        real = real.get("auroc", real.get("mean"))
    rows.append((int(m.group(1)), int(m.group(2)), real,
                 __import__("datetime").datetime.fromtimestamp(os.path.getmtime(p)).strftime("%m-%d %H:%M")))
by = collections.defaultdict(dict)
for L, s, v, mt in rows:
    by[L][s] = (v, mt)
seeds = sorted({s for _, s, _, _ in rows})
print(f"  {len(rows)} falsify files; seeds {seeds}")
print("  layer | " + " | ".join(f"s{s}" for s in seeds))
for L in sorted(by):
    cells = []
    for s in seeds:
        v = by[L].get(s)
        cells.append(f"{v[0]:.4f}" if v and isinstance(v[0], float) else "  -   ")
    print(f"  L{L:<4} | " + " | ".join(cells))
print("\n  ledger F-91 quotes: L32 0.8945, L36 0.8942, L40 0.8828")
for L in (32, 36, 40):
    for s in seeds:
        v = by.get(L, {}).get(s)
        if v and isinstance(v[0], float):
            print(f"    L{L} s{s}: {v[0]:.5f}   (mtime {v[1]})")

sec("F-91  coverage audit: layers per crossgen store (mean_*_s42 matrices)")
stores = collections.defaultdict(set)
for p in glob.glob(XG + "/**/matrices/mean_L*_s*.json", recursive=True):
    store = os.path.dirname(os.path.dirname(p)).replace("\\", "/").replace(XG + "/", "")
    m = re.search(r"mean_L(\d+)_s(\d+)", os.path.basename(p))
    stores[store].add((int(m.group(1)), int(m.group(2))))
for s in sorted(stores):
    ls = sorted({l for l, _ in stores[s]})
    sd = sorted({q for _, q in stores[s]})
    print(f"  {s:44} layers={len(ls):>3}  seeds={sd}")
print("\nDONE W5B_labC")
