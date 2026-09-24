"""Phase 2 batch P7 (register/sarcasm line) — read-only recomputation on the lab PC.

Modes:
  length  — F-73's per-dataset span_len gap table from run-coords/passB/*/rows.jsonl
  f7576   — F-75 / F-76: the 80-layer matched-slice profile + dataset-level bootstrap
  f77     — F-77: cap_s23 layer profile on sycophancy_mt (as target) + bootstrap
  f87     — F-87: cap_s23 seed table at L40

Reads only owner files under C:/Users/user4/expLB; writes nothing.
"""
import glob, json, os, re, statistics, sys
from collections import defaultdict

import numpy as np

ROOT = "C:/Users/user4/expLB"
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"


# ---------------------------------------------------------------- F-73 length gaps
IN_THEIR_REPO = {"instructed-deception", "harm-pressure-choice",
                 "harm-pressure-knowledge-report", "convincing-game", "insider-trading",
                 "ai_liar", "roleplaying", "sandbagging_v2"}


def mode_length():
    store = f"{ROOT}/run-coords/passB"
    print(f"=== F-73 length gaps ===\nstore: {store}  (span_len = TOKENS of the scored span)\n")
    print(f"  {'dataset':<34}{'n':>6}{'n_pos':>7}{'n_neg':>7}{'pos len':>10}{'neg len':>10}"
          f"{'gap':>9}{'ratio':>7}  in-repo")
    stats = {}
    for d in sorted(os.listdir(store)):
        rj = os.path.join(store, d, "rows.jsonl")
        if not os.path.isfile(rj):
            continue
        by = defaultdict(list)
        with open(rj, encoding="utf-8") as f:
            nrows = 0
            for line in f:
                if not line.strip():
                    continue
                nrows += 1
                r = json.loads(line)
                lab, sl = r.get("label"), r.get("span_len")
                if lab in (0, 1) and isinstance(sl, (int, float)):
                    by[lab].append(float(sl))
        if len(by) != 2 or not all(by.values()):
            print(f"  {d:<34}{nrows:>6}   SKIPPED (labels {sorted(by)})")
            continue
        pos, neg = statistics.fmean(by[1]), statistics.fmean(by[0])
        stats[d] = (pos, neg, nrows, len(by[1]), len(by[0]))
        mark = "yes" if d in IN_THEIR_REPO else "-"
        print(f"  {d:<34}{nrows:>6}{len(by[1]):>7}{len(by[0]):>7}{pos:>10.4f}{neg:>10.4f}"
              f"{pos - neg:>+9.4f}{pos / neg if neg else float('nan'):>7.2f}  {mark}")

    tot = len(stats)
    longer = sum(1 for p, n, *_ in stats.values() if p > n)
    equal = sum(1 for p, n, *_ in stats.values() if p == n)
    shorter = sum(1 for p, n, *_ in stats.values() if p < n)
    print(f"\n  ALL {tot} datasets: pos strictly longer in {longer}; exactly equal {equal}; "
          f"shorter {shorter}")
    print(f"  pos longer-or-equal in {longer + equal} of {tot}")
    sub = {k: v for k, v in stats.items() if k in IN_THEIR_REPO}
    sl = sum(1 for p, n, *_ in sub.values() if p > n)
    se = sum(1 for p, n, *_ in sub.values() if p == n)
    print(f"  restricted to the 8 in the reviewer's repo: strictly longer {sl} of {len(sub)}; "
          f"equal {se}")
    if sub:
        gaps = sorted(abs(p - n) for p, n, *_ in sub.values())
        print(f"  |gap| on those 8, sorted: {['%.3f' % g for g in gaps]}")
        print(f"  median |gap| on those 8: {statistics.median(gaps):.3f} tokens")
    # exclude alpaca (the control corpus) if present, to see the 11-dataset view
    ds11 = {k: v for k, v in stats.items() if k != "alpaca"}
    l11 = sum(1 for p, n, *_ in ds11.values() if p > n)
    print(f"  excluding 'alpaca' control: pos strictly longer in {l11} of {len(ds11)}")


# ---------------------------------------------------------------- matrix helpers
def row_vec(path, ds, feature, probe="paper"):
    """AUROCs of the ROW `ds` (trained on it) against every other dataset."""
    d = json.load(open(path, encoding="utf-8"))
    if ds not in d["datasets"]:
        return None
    i = d["datasets"].index(ds)
    n = len(d["datasets"])
    for r in d["results"]:
        if r["probe"] == probe and r["feature"] == feature:
            A = r["auroc"]
            return np.array([np.nan if A[i][j] is None else A[i][j]
                             for j in range(n) if j != i], float)
    return None


def col_vec(path, ds, feature, probe="paper"):
    """AUROCs into the COLUMN `ds` (tested on it) from every other source."""
    d = json.load(open(path, encoding="utf-8"))
    if ds not in d["datasets"]:
        return None
    j = d["datasets"].index(ds)
    n = len(d["datasets"])
    for r in d["results"]:
        if r["probe"] == probe and r["feature"] == feature:
            A = r["auroc"]
            return np.array([np.nan if A[i][j] is None else A[i][j]
                             for i in range(n) if i != j], float)
    return None


def boot(M, layers, prespec, B=10000, seed=42, plateau=(27, 48)):
    """M = (n_layers, n_units) margin matrix; bootstrap over the units (datasets)."""
    ok = ~np.isnan(M).any(axis=0)
    M = M[:, ok]
    nu = M.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nu, size=(B, nu))
    li = layers.index(prespec)
    pre = M[li][idx].mean(axis=1)
    lo, hi = np.percentile(pre, [2.5, 97.5])
    print(f"  units resampled: {nu}   B={B} seed={seed}")
    print(f"  margin at L{prespec}: point {M[li].mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"P(<=0) {float((pre <= 0).mean()):.4f}"
          f"{'   CLEARS' if lo > 0 else '   does not clear'}")
    allb = M[:, idx].mean(axis=2)
    arg = np.array(layers)[allb.argmax(axis=0)]
    alo, ahi = np.percentile(arg, [2.5, 97.5])
    print(f"  argmax layer: median L{int(np.median(arg))}  95% [L{int(alo)}, L{int(ahi)}]  "
          f"width {int(ahi - alo)} of {len(layers)}  lands on L{prespec} in "
          f"{float((arg == prespec).mean()):.2%}  inside L{plateau[0]}-L{plateau[1]} in "
          f"{float(((arg >= plateau[0]) & (arg <= plateau[1])).mean()):.1%}")
    frac = (allb > 0).mean(axis=0)
    print(f"  fraction of layers positive per resample: median {np.median(frac):.4f}  "
          f"95% [{np.percentile(frac, 2.5):.2f}, {np.percentile(frac, 97.5):.2f}]")
    best = allb.max(axis=0)
    print(f"  best-layer margin (selection priced in): point {M.mean(axis=1).max():+.4f}  "
          f"95% [{np.percentile(best, 2.5):+.4f}, {np.percentile(best, 97.5):+.4f}]")


# ---------------------------------------------------------------- F-75 / F-76
def mode_f7576():
    DS = "harm-pressure-kr-matched"
    bank = f"{ROOT}/run-crossgen/kd11/passA11/banks/kd_s23"
    root = f"{ROOT}/run-crossgen/kd11/passA11"
    print(f"=== F-75 / F-76 : 80-layer matched-slice profile ===")
    print(f"bank={bank}\nroot={root}\nrow dataset (trained on) = {DS}, probe=paper, "
          f"pooling=mean, seed 42\n")
    cells = {}
    for f in glob.glob(f"{bank}/matrices/mean_L*_s42.json"):
        L = int(re.search(r"mean_L(\d+)_s42", f).group(1))
        v = row_vec(f, DS, "axes")
        if v is not None:
            cells.setdefault(L, {})["axes"] = v
    for f in glob.glob(f"{root}/matrices/mean_L*_s42.json"):
        L = int(re.search(r"mean_L(\d+)_s42", f).group(1))
        for feat in ("raw", "datapca"):
            v = row_vec(f, DS, feat)
            if v is not None:
                cells.setdefault(L, {})[feat] = v
    layers = sorted(L for L in cells if all(k in cells[L] for k in ("axes", "raw", "datapca")))
    print(f"layers with axes+raw+datapca: {len(layers)}  "
          f"(min L{min(layers)}, max L{max(layers)})   n_targets={len(cells[layers[0]]['axes'])}\n")
    print(f"{'L':>4}{'axes':>9}{'raw':>9}{'datapca':>9}{'a-raw':>9}{'a-dpca':>9}")
    prof = {}
    for L in layers:
        a = np.nanmean(cells[L]["axes"]); r = np.nanmean(cells[L]["raw"])
        c = np.nanmean(cells[L]["datapca"])
        prof[L] = (a, r, c)
        flag = "  <<" if L in (19, 20, 22, 40, 44, 77) else ""
        print(f"{L:>4}{a:>9.4f}{r:>9.4f}{c:>9.4f}{a - r:>+9.4f}{a - c:>+9.4f}{flag}")

    beats_raw = sum(1 for L in layers if prof[L][0] > prof[L][1])
    beats_dp = sum(1 for L in layers if prof[L][0] > prof[L][2])
    print(f"\n  F-75(a): axes beats raw at {beats_raw}/{len(layers)}; "
          f"beats datapca at {beats_dp}/{len(layers)}")
    # contiguous runs of positive margin over datapca
    runs, cur, start = [], 0, None
    for L in layers:
        if prof[L][0] - prof[L][2] > 0:
            if cur == 0:
                start = L
            cur += 1
        else:
            if cur:
                runs.append((start, layers[layers.index(L) - 1], cur))
            cur = 0
    if cur:
        runs.append((start, layers[-1], cur))
    runs.sort(key=lambda t: -t[2])
    print(f"  longest contiguous positive-vs-datapca runs: "
          f"{[(f'L{a}-L{b}', n) for a, b, n in runs[:5]]}")
    # same for margin over raw
    runs2, cur, start = [], 0, None
    for L in layers:
        if prof[L][0] - prof[L][1] > 0:
            if cur == 0:
                start = L
            cur += 1
        else:
            if cur:
                runs2.append((start, layers[layers.index(L) - 1], cur))
            cur = 0
    if cur:
        runs2.append((start, layers[-1], cur))
    runs2.sort(key=lambda t: -t[2])
    print(f"  longest contiguous positive-vs-raw runs: "
          f"{[(f'L{a}-L{b}', n) for a, b, n in runs2[:5]]}")
    lo_axes = min(layers, key=lambda L: prof[L][0])
    hi_axes = max(layers, key=lambda L: prof[L][0])
    print(f"  axes floor: L{lo_axes} {prof[lo_axes][0]:.4f};  peak: L{hi_axes} "
          f"{prof[hi_axes][0]:.4f}")
    print(f"  L0-L20 axes range: {min(prof[L][0] for L in layers if L <= 20):.4f} - "
          f"{max(prof[L][0] for L in layers if L <= 20):.4f}")
    print(f"  L27-L48 axes range: {min(prof[L][0] for L in layers if 27 <= L <= 48):.4f} - "
          f"{max(prof[L][0] for L in layers if 27 <= L <= 48):.4f}")
    print(f"  largest margin over datapca: L"
          f"{max(layers, key=lambda L: prof[L][0] - prof[L][2])} "
          f"{max(prof[L][0] - prof[L][2] for L in layers):+.4f}")

    print("\n  F-76 bootstrap (control = datapca):")
    M = np.vstack([cells[L]["axes"] - cells[L]["datapca"] for L in layers])
    boot(M, layers, 40)
    print("\n  [robustness] same bootstrap with control = raw:")
    Mr = np.vstack([cells[L]["axes"] - cells[L]["raw"] for L in layers])
    boot(Mr, layers, 40)


# ---------------------------------------------------------------- F-77
def mode_f77():
    DS = "sycophancy_mt"
    bank = f"{ROOT}/run-crossgen/kd11/banks/cap_s23"
    print(f"=== F-77 : cap bank layer profile ===\nbank={bank}\ncolumn (tested on) = {DS}, "
          f"probe=paper, pooling=mean, seed 42\n")
    feats = ("pc:capitulation@2", "datapca@2", "raw")
    cells = {}
    for f in glob.glob(f"{bank}/matrices/mean_L*_s42.json"):
        L = int(re.search(r"mean_L(\d+)_s42", f).group(1))
        for feat in feats:
            v = col_vec(f, DS, feat)
            if v is not None:
                cells.setdefault(L, {})[feat] = v
    layers = sorted(L for L in cells if all(k in cells[L] for k in feats))
    print(f"layers with all three features: {len(layers)} -> {layers}   "
          f"n_sources={len(cells[layers[0]]['raw'])}\n")
    print(f"{'L':>4}{'cap@2':>9}{'raw':>9}{'dpca@2':>9}{'c-raw':>9}{'c-ctrl':>9}")
    prof = {}
    for L in layers:
        c = np.nanmean(cells[L]["pc:capitulation@2"]); r = np.nanmean(cells[L]["raw"])
        d = np.nanmean(cells[L]["datapca@2"])
        prof[L] = (c, r, d)
        print(f"{L:>4}{c:>9.4f}{r:>9.4f}{d:>9.4f}{c - r:>+9.4f}{c - d:>+9.4f}")
    pos_ctrl = sum(1 for L in layers if prof[L][0] - prof[L][2] > 0)
    pos_raw = sum(1 for L in layers if prof[L][0] - prof[L][1] > 0)
    print(f"\n  positive vs datapca@2 at {pos_ctrl}/{len(layers)} layers; "
          f"vs raw at {pos_raw}/{len(layers)}")
    print(f"  vs raw margin range: {min(prof[L][0]-prof[L][1] for L in layers):+.4f} .. "
          f"{max(prof[L][0]-prof[L][1] for L in layers):+.4f}")
    print(f"  vs control margin range: {min(prof[L][0]-prof[L][2] for L in layers):+.4f} .. "
          f"{max(prof[L][0]-prof[L][2] for L in layers):+.4f}")
    pk = max(layers, key=lambda L: prof[L][0] - prof[L][2])
    print(f"  peak margin over control at L{pk}: {prof[pk][0]-prof[pk][2]:+.4f}")
    M = np.vstack([cells[L]["pc:capitulation@2"] - cells[L]["datapca@2"] for L in layers])
    print("\n  bootstrap (control = datapca@2):")
    boot(M, layers, 40, plateau=(27, 48))


# ---------------------------------------------------------------- F-87
def mode_f87():
    DS = "sycophancy_mt"
    bank = f"{ROOT}/run-crossgen/kd11/banks/cap_s23"
    print(f"=== F-87 : cap seed table at L40 ===\nbank={bank}\n")
    print(f"{'seed':>6}{'cap@2':>10}{'raw':>10}{'dpca@2':>10}{'vs raw':>10}{'vs ctrl':>10}"
          f"{'n_src':>7}")
    vals = {}
    for s in (1, 7, 13, 42):
        p = f"{bank}/matrices/mean_L40_s{s}.json"
        if not os.path.isfile(p):
            print(f"{s:>6}   MISSING {p}")
            continue
        d = json.load(open(p, encoding="utf-8"))
        got = {}
        for feat in ("pc:capitulation@2", "raw", "datapca@2"):
            v = col_vec(p, DS, feat)
            got[feat] = np.nanmean(v) if v is not None else float("nan")
        n = len(col_vec(p, DS, "raw")) if col_vec(p, DS, "raw") is not None else 0
        vals[s] = got
        print(f"{s:>6}{got['pc:capitulation@2']:>10.4f}{got['raw']:>10.4f}"
              f"{got['datapca@2']:>10.4f}{got['pc:capitulation@2']-got['raw']:>+10.4f}"
              f"{got['pc:capitulation@2']-got['datapca@2']:>+10.4f}{n:>7}")
        print(f"        features present: "
              f"{sorted({r['feature'] for r in d['results'] if r['probe']=='paper'})}")
    if vals:
        vr = [vals[s]['pc:capitulation@2'] - vals[s]['raw'] for s in vals]
        vc = [vals[s]['pc:capitulation@2'] - vals[s]['datapca@2'] for s in vals]
        print(f"\n  vs raw   : {min(vr):+.4f} .. {max(vr):+.4f}  (spread {max(vr)-min(vr):.4f})")
        print(f"  vs ctrl  : {min(vc):+.4f} .. {max(vc):+.4f}  (spread {max(vc)-min(vc):.4f})")


if MODE in ("length", "all"):
    mode_length()
if MODE in ("f7576", "all"):
    mode_f7576()
if MODE in ("f77", "all"):
    mode_f77()
if MODE in ("f87", "all"):
    mode_f87()
