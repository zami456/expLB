"""P12-briefing, lab half.  Read-only recomputation of the COLLABORATOR_BRIEFING numbers
that live only on the lab PC.

O-001b  which persona axis the bare `contrast`/`PC1` zero-shot rows actually use, and what the
        paper16 SYCOPHANCY axis really reads on sycophancy_mt (the briefing attributes 0.374 to it)
O-002   filtered (`cap_s23`) vs unfiltered (`cap_all`) cap layer profile -- the "differ by 0.0004"
        row, plus the +0.112 / CI [+0.050,+0.185] and "21 of 21 layers" rows of the same table
O-003   leave-one-CLUSTER-out gains at L20/L32/L40, all 11 and the 9 clean targets
O-005   the label-permutation p for the kd matched-slice band, and the OTHER p=0.38 in the
        briefing (pooled full vs row-budget-matched)
O-009   pooled LODO minus each of the collaborator's two arms, on our eleven, at L20 and L40
O-011   ext_axis1 - axis1 over all 21 layers, mean AND last pooling: size and layers positive
O-012   the SHARE of the margin over raw, median under each pooling

Usage on the lab:  python P12_lab.py            (paths default to C:/Users/user4/expLB)
"""
import glob
import hashlib
import json
import os
import re
import statistics
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/user4/expLB"
KD11 = f"{ROOT}/run-crossgen/kd11"
PREMISE = f"{ROOT}/run-premise"
LAYERS = [0, 4, 8, 12, 16, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76]
_READ = []


def stamp(p):
    try:
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        st = os.stat(p)
        _READ.append((p, st.st_size, h,
                      __import__("datetime").datetime.fromtimestamp(st.st_mtime)
                      .strftime("%Y-%m-%d %H:%M:%S")))
    except Exception as e:  # noqa: BLE001
        _READ.append((p, -1, f"ERR {e}", ""))


def load(p):
    stamp(p)
    return json.load(open(p, encoding="utf-8"))


def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def offdiag(d, feat, probe="paper"):
    n = len(d["datasets"])
    for r in d["results"]:
        if r["feature"] == feat and r["probe"] == probe:
            v = [r["auroc"][i][j] for i in range(n) for j in range(n)
                 if i != j and r["auroc"][i][j] is not None]
            return statistics.fmean(v), len(v)
    return None, 0


def column(d, feat, ds, probe="paper"):
    """the 10 per-source transfer AUROCs INTO dataset `ds` (a column, diagonal removed)"""
    n = len(d["datasets"])
    j = d["datasets"].index(ds)
    for r in d["results"]:
        if r["feature"] == feat and r["probe"] == probe:
            return np.array([np.nan if r["auroc"][i][j] is None else r["auroc"][i][j]
                             for i in range(n) if i != j], dtype=float)
    return None


def zmean(d, direction):
    z = next((x for x in d["zeroshot"] if x["direction"] == direction), None)
    if z is None:
        return None
    v = [a for a in z["auroc"] if a is not None]
    return statistics.fmean(v)


# ---------------------------------------------------------------------- O-001b
hr("O-001b  which axis is the bare `contrast`/`PC1`, and what the SYCOPHANCY axis reads")
p = f"{KD11}/matrices/mean_L40_s42.json"
d = load(p)
ds = d["datasets"]
SY = ds.index("sycophancy_mt")
KR = ds.index("harm-pressure-knowledge-report")
print(f"  {p}")
print(f"  bank={d['bank']}  bank_name={d['bank_name']}  k={d['k']}  directions="
      f"{[z['direction'] for z in d['zeroshot']]}")
for z in d["zeroshot"]:
    print(f"    {z['direction']:<22} sycophancy_mt={z['auroc'][SY]:.4f}  HP-KR={z['auroc'][KR]:.4f}")

try:
    import torch
    for bp in (f"{ROOT}/run-paper16/axes_vectors_s3_fp32.pt",
               f"{ROOT}/run-paper16/axes_vectors_s3.pt"):
        if not os.path.exists(bp):
            continue
        stamp(bp)
        b = torch.load(bp, map_location="cpu", weights_only=False)
        print(f"\n  bank {bp}")
        print(f"    axes: {sorted(b['axes'])}")
        print(f"    pca : {sorted(b['pca'])}")
        cands = [n for n in ("deception",) if n in b["axes"] and n in b["pca"]]
        print(f"    27_crossgen.py would pick _zs_axis = {cands[0] if cands else '<fallback>'}"
              "   <- so the bare `contrast`/`PC1` rows are the DECEPTION axis")
        break
except Exception as e:  # noqa: BLE001
    print("  torch/bank:", e)

print("\n  any lab matrix that scored a SYCOPHANCY zero-shot direction on sycophancy_mt:")
hits = 0
for f in sorted(glob.glob(f"{ROOT}/run-crossgen/**/matrices/mean_L40_s42.json", recursive=True)):
    try:
        dd = json.load(open(f, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        continue
    dirs = [z["direction"] for z in dd.get("zeroshot", [])]
    syc = [x for x in dirs if "sycophancy" in x]
    if syc and "sycophancy_mt" in dd.get("datasets", []):
        k = dd["datasets"].index("sycophancy_mt")
        stamp(f)
        print(f"    {f}  bank={dd.get('bank_name')}")
        for x in syc:
            z = next(z for z in dd["zeroshot"] if z["direction"] == x)
            print(f"        {x:<26} sycophancy_mt={z['auroc'][k]:.4f}")
        hits += 1
    if hits >= 6:
        break
if not hits:
    print("    NONE -- no stored matrix scores a sycophancy-axis direction on sycophancy_mt")

# ------------------------------------------------------------------------ O-002
hr("O-002  cap bank, filtered (s23) vs unfiltered (all): the whole Claim-2 table")
rng = np.random.default_rng(42)
prof = {}
for bank in ("cap_s23", "cap_all"):
    rows = {}
    for L in LAYERS:
        fp = f"{KD11}/banks/{bank}/matrices/mean_L{L}_s42.json"
        if not os.path.exists(fp):
            continue
        dd = load(fp)
        rows[L] = {k: column(dd, k, "sycophancy_mt")
                   for k in ("pc:capitulation@2", "raw", "datapca@2", "random@2", "length")}
    prof[bank] = rows
    have = sorted(rows)
    M = np.vstack([rows[L]["pc:capitulation@2"] - rows[L]["datapca@2"] for L in have])
    Mr = np.vstack([rows[L]["pc:capitulation@2"] - rows[L]["raw"] for L in have])
    pos_c = int((M.mean(axis=1) > 0).sum())
    pos_r = int((Mr.mean(axis=1) > 0).sum())
    n_src = M.shape[1]
    idx = np.random.default_rng(42).integers(0, n_src, size=(10000, n_src))
    li = have.index(40)
    print(f"\n  {bank}: {len(have)} layers, {n_src} source datasets")
    print(f"    L40  pc:capitulation@2 = {rows[40]['pc:capitulation@2'].mean():.4f}"
          f"   raw = {rows[40]['raw'].mean():.4f}"
          f"   datapca@2 = {rows[40]['datapca@2'].mean():.4f}"
          f"   random@2 = {rows[40]['random@2'].mean():.4f}")
    for nm, MM in (("- datapca@2", M), ("- raw", Mr)):
        b = MM[li][idx].mean(axis=1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"    L40 margin {nm:<12} {MM[li].mean():+.4f}"
              f"  95% CI [{lo:+.4f}, {hi:+.4f}]  P(<=0) {float((b <= 0).mean()):.4f}")
    print(f"    layers positive vs datapca@2: {pos_c}/{len(have)}"
          f"   vs raw: {pos_r}/{len(have)}")
    allb = M[:, idx].mean(axis=2)
    arg = np.array(have)[allb.argmax(axis=0)]
    alo, ahi = np.percentile(arg, [2.5, 97.5])
    print(f"    argmax median L{int(np.median(arg))}  95% [L{int(alo)}, L{int(ahi)}]"
          f"  lands on L40 in {float((arg == 40).mean()):.1%}")

if "cap_s23" in prof and "cap_all" in prof and 40 in prof["cap_s23"]:
    a = prof["cap_s23"][40]
    b = prof["cap_all"][40]
    print("\n  filtered vs unfiltered at L40 (the briefing's 'differ by 0.0004'):")
    for k in ("pc:capitulation@2", "raw", "datapca@2", "random@2", "length"):
        print(f"    {k:<20} s23 {a[k].mean():.4f}   all {b[k].mean():.4f}"
              f"   |diff| {abs(a[k].mean() - b[k].mean()):.4f}")
    ms = (a["pc:capitulation@2"] - a["datapca@2"]).mean()
    mu = (b["pc:capitulation@2"] - b["datapca@2"]).mean()
    rs = (a["pc:capitulation@2"] - a["raw"]).mean()
    ru = (b["pc:capitulation@2"] - b["raw"]).mean()
    print(f"    margin vs datapca@2:  s23 {ms:+.4f}  all {mu:+.4f}  |diff| {abs(ms - mu):.4f}")
    print(f"    margin vs raw      :  s23 {rs:+.4f}  all {ru:+.4f}  |diff| {abs(rs - ru):.4f}")

# ------------------------------------------------------------------------ O-003
hr("O-003  leave-one-CLUSTER-out (F-108)")
for L in (20, 32, 40):
    fp = f"{PREMISE}/clustered/clustered_L{L}_mean_s42.json"
    if not os.path.exists(fp):
        print(f"  L{L}: MISSING {fp}")
        continue
    c = load(fp)
    print(f"\n  L{L}  keys = {sorted(c)}")
    s = c.get("summary", c)
    for k in sorted(s):
        v = s[k]
        if isinstance(v, (int, float, str)):
            print(f"    summary.{k} = {v}")
    per = c.get("per_dataset") or c.get("targets") or c.get("rows")
    if isinstance(per, dict):
        print("    per-target:")
        for t, v in per.items():
            print(f"      {t:<34} {v}")
    elif isinstance(per, list):
        print("    per-target:")
        for v in per:
            print(f"      {v}")

# ------------------------------------------------------------------------ O-005
hr("O-005  label permutation on the band (F-84) + the OTHER p=0.38 (volume vs diversity)")
fp = f"{PREMISE}/cluster_perm_kd_matched.json"
if os.path.exists(fp):
    cp = load(fp)
    print(f"  {fp}")
    print(json.dumps({k: v for k, v in cp.items()
                      if not isinstance(v, list) or len(v) <= 30}, indent=2)[:3000])
    for k, v in cp.items():
        if isinstance(v, list) and len(v) > 30:
            print(f"  (list) {k}: n={len(v)} min={min(v):.4f} max={max(v):.4f}")
else:
    print(f"  MISSING {fp}")

fp = f"{PREMISE}/premise_L40_mean_s42.json"
if os.path.exists(fp):
    pr = load(fp)
    s = pr.get("summary", {})
    print(f"\n  {fp}  summary keys = {sorted(s)}")
    for k in sorted(s):
        print(f"    summary.{k} = {s[k]}")
    for k in sorted(pr):
        if k in ("summary",):
            continue
        v = pr[k]
        if isinstance(v, dict) and any("match" in str(x) for x in v):
            print(f"    {k}: {json.dumps(v)[:800]}")
    print("\n  looking for the pooled_full vs pooled_match test:")
    txt = json.dumps(pr)
    for m in re.finditer(r'"[^"]*(?:match|volume|budget)[^"]*"\s*:\s*[^,{}\[\]]+', txt):
        print("   ", m.group(0)[:160])

# ------------------------------------------------- O-009 / O-011 / O-012 (collab)
hr("O-009 / O-011 / O-012  the collaborator's two arms on OUR eleven")
prem = {}
for L in LAYERS:
    fp = f"{PREMISE}/premise_L{L}_mean_s42.json"
    if os.path.exists(fp):
        prem[L] = load(fp)

for pool in ("mean", "last"):
    print(f"\n  ---- pooling = {pool} ----")
    print(f"  {'L':>4}{'axis1':>9}{'ext':>9}{'raw':>9}{'ext-axis1':>11}{'share':>9}"
          f"{'pooledLODO':>12}{'pooled-ext':>11}")
    diffs, shares, layers_ok = [], [], []
    for L in LAYERS:
        fc = f"{KD11}/banks/collab/matrices/{pool}_L{L}_s42.json"
        fb = f"{KD11}/matrices/{pool}_L{L}_s42.json"
        if not (os.path.exists(fc) and os.path.exists(fb)):
            continue
        dc, db = load(fc), load(fb)
        a = zmean(dc, "contrast")
        e = zmean(dc, "contrast:collab_extended")
        raw, _ = offdiag(db, "raw")
        share = (e - a) / (e - raw) if (e - raw) != 0 else float("nan")
        pl = ""
        pd = ""
        if pool == "mean" and L in prem:
            v = prem[L].get("summary", {}).get("pooled_lodo")
            if v is not None:
                pl = f"{v:.4f}"
                pd = f"{v - e:+.4f}"
        flag = "  axis1<raw" if a < raw else ""
        print(f"  {L:>4}{a:>9.4f}{e:>9.4f}{raw:>9.4f}{e - a:>+11.4f}{share:>9.1%}"
              f"{pl:>12}{pd:>11}{flag}")
        diffs.append(e - a)
        shares.append(share)
        layers_ok.append(L)
    if diffs:
        pos = sum(1 for v in diffs if v > 0)
        i_min, i_max = int(np.argmin(diffs)), int(np.argmax(diffs))
        print(f"    ext-axis1 positive at {pos}/{len(diffs)} layers;"
              f" mean {statistics.fmean(diffs):+.4f};"
              f" range [{diffs[i_min]:+.4f} (L{layers_ok[i_min]}),"
              f" {diffs[i_max]:+.4f} (L{layers_ok[i_max]})]")
        print(f"    share: median {statistics.median(shares):.1%}"
              f"  mean {statistics.fmean(shares):.1%}"
              f"  range [{min(shares):.1%}, {max(shares):.1%}]")
        neg = [L for L, v in zip(layers_ok, diffs) if v <= 0]
        print(f"    layers where ext-axis1 <= 0: {neg}")

print("\n  O-009 exactly as the briefing states it (mean pooling):")
for L in (20, 40):
    fc = f"{KD11}/banks/collab/matrices/mean_L{L}_s42.json"
    if not (os.path.exists(fc) and L in prem):
        continue
    dc = load(fc)
    a, e = zmean(dc, "contrast"), zmean(dc, "contrast:collab_extended")
    pl = prem[L]["summary"]["pooled_lodo"]
    print(f"    L{L}: pooled LODO {pl:.4f}   axis1 {a:.4f} (gap {pl - a:+.4f})"
          f"   ext_axis1 {e:.4f} (gap {pl - e:+.4f})")

# --------------------------------------------------------------------- manifest
hr("FILES READ (path, size, md5, mtime)")
seen = set()
for p, sz, h, mt in _READ:
    if p in seen:
        continue
    seen.add(p)
    print(f"  {sz:>12}  {h}  {mt}  {p}")
