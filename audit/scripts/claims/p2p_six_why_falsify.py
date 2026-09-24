"""Phase-2 batch P2-pooled: F-85, F-86, F-95, F-109.

Reads the md5-verified cache of the lab PC's run-premise/ (audit/cache/labpc/run-premise)
and the appended console log logs/falsify.log (UTF-16), which still holds the ORIGINAL
2026-09-21 falsify run alongside the 2026-09-23 re-run.

Run:  PYTHONIOENCODING=utf-8 python audit/scripts/claims/p2p_six_why_falsify.py
"""
from __future__ import annotations

import json
import os
import re
import statistics

import numpy as np

ROOT = os.path.join("audit", "cache", "labpc", "run-premise")
LOGS = os.path.join("audit", "cache", "labpc", "logs")
LAYERS = [0, 4, 8, 12, 16, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76]
SEEDS = [42, 7, 13]


def load(rel):
    p = os.path.join(ROOT, rel)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def boot(per, a, b, B=10000, seed=42):
    names = sorted(per)
    n = len(names)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    dif = np.array([per[k][a] for k in names]) - np.array([per[k][b] for k in names])
    bs = dif[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(dif.mean()), float(lo), float(hi), float((bs <= 0).mean()), n


def sec(t):
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


# ------------------------------------------------------------------ F-85 / F-109
sec("F-85 and F-109  run-premise/six/premise_L{20,32,40}_mean_s42.json")
STATED = {20: {"single": 0.6439, "pooled_lodo": 0.7365, "pooled_lodo_match": 0.7654,
               "pooled_full": 0.9630},
          32: {"single": 0.7203, "pooled_lodo": 0.8965, "pooled_lodo_match": 0.9010},
          40: {"single": 0.6960, "pooled_lodo": 0.8683, "pooled_lodo_match": 0.8930,
               "pooled_full": 0.9862, "single_own_mean": 0.9852}}
six = {}
for L in (20, 32, 40):
    j = load(f"six/premise_L{L}_mean_s42.json")
    if j is None:
        print(f"  L{L}: MISSING")
        continue
    six[L] = j
    s = j["summary"]
    print(f"\n  L{L}  datasets={j['datasets']}")
    print(f"       n_train={j['n_train']}  probe={j['probe']} pooling={j['pooling']} "
          f"seed={j['seed']} cov={j.get('covariance')}")
    for k, v in STATED[L].items():
        got = s[k]
        print(f"       {k:<18} stated {v:.4f}  file {got:.6f}  |d|={abs(got - v):.6f}  "
              f"{'MATCH' if abs(got - v) <= 5e-5 else 'DIFF'}")
    if L == 40 and "single_own_mean" not in STATED[L]:
        pass
print("\n  ledger F-85 also quotes, at L40: single-source own dataset 0.9852 -> file "
      f"{six[40]['summary']['single_own_mean']:.4f}")
print(f"  ledger F-85 L20 'pooled including the target' 0.9630 -> file "
      f"{six[20]['summary']['pooled_full']:.4f}")

print("\n  F-85 margins against the abstract's quoted contrast numbers (0.904 L20 / 0.916 L40):")
for L, contrast in ((20, 0.904), (40, 0.916)):
    s = six[L]["summary"]
    print(f"    L{L}  contrast {contrast:.3f}  vs single {s['single']:.4f} -> "
          f"{contrast - s['single']:+.4f}   vs lodo {s['pooled_lodo']:.4f} -> "
          f"{contrast - s['pooled_lodo']:+.4f}   vs row-matched {s['pooled_lodo_match']:.4f} -> "
          f"{contrast - s['pooled_lodo_match']:+.4f}")
print("    ledger F-85: +0.184 (vs single, L40), +0.048 (vs lodo), +0.023 (vs row-matched)")

print("\n  F-109 deltas L32 - L40:")
for k in ("single", "pooled_lodo", "pooled_lodo_match"):
    d = six[32]["summary"][k] - six[40]["summary"][k]
    print(f"    {k:<20} L32 {six[32]['summary'][k]:.4f}  L40 {six[40]['summary'][k]:.4f}  "
          f"delta {d:+.4f}")
print("    ledger F-109: +0.024 (single), +0.028 (LODO), +0.008 (row-matched)")
print(f"    ledger F-109: 0.916 - 0.9010 = {0.916 - six[32]['summary']['pooled_lodo_match']:+.4f} "
      f"(stated +0.015, from +0.023 at L40)")

print("\n  row-matched beats full LODO? (F-85's 'oddity')")
for L in sorted(six):
    s = six[L]["summary"]
    print(f"    L{L}: row-matched {s['pooled_lodo_match']:.4f} vs lodo {s['pooled_lodo']:.4f} -> "
          f"{s['pooled_lodo_match'] - s['pooled_lodo']:+.4f}")

print("\n  dataset-level bootstrap on the six (n=6; F-85 says none was run):")
for L in sorted(six):
    per = six[L]["per_dataset"]
    for a, b in (("pooled_lodo", "single_offdiag"), ("pooled_lodo_match", "single_offdiag")):
        d, lo, hi, p, n = boot(per, a, b)
        print(f"    L{L:<3} {a} - {b}: {d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  P(<=0) {p:.4f}  n={n}")

# ------------------------------------------------------------------ F-86
sec("F-86  run-premise/why/why_L40_s42.json  +  run-premise/audit/audit_L40_s42.json")
w = load("why/why_L40_s42.json")
p40 = load("premise_L40_mean_s42.json")
print(f"  why file keys: {sorted(w.keys())}")
S86 = {"single_offdiag": 0.6763, "pooled_lodo_matched": 0.8062, "mean_of_directions": 0.8828,
       "cos_single_to_single_mean": 0.051, "cos_pooled_to_single_mean": 0.227}
for k, v in S86.items():
    got = w[k]
    tol = 5e-5 if v > 0.1 else 5e-4
    print(f"  {k:<28} stated {v:.4f}  file {got:.6f}  "
          f"{'MATCH' if abs(got - v) <= tol else 'DIFF'}")
print(f"  stated 'pooled LODO, all rows (F-80)' 0.8545 -> premise file "
      f"{p40['summary']['pooled_lodo']:.4f}  (the why file does not store it)")
md = w["mean_of_directions"]
print(f"  margins: mean-of-dirs - pooled_full_lodo = {md - p40['summary']['pooled_lodo']:+.4f} "
      f"(stated +0.028)")
print(f"           mean-of-dirs - why's budget-matched = {md - w['pooled_lodo_matched']:+.4f} "
      f"(stated +0.077)")
print(f"           mean-of-dirs - single off-diagonal  = {md - w['single_offdiag']:+.4f} "
      f"(stated +0.207)")
print("\n  INTERNAL CONFLICT CHECK: the ledger calls 0.8062 'pooled LODO, row-budget matched',")
print("  but F-79/F-80's row-budget-matched LODO at the same layer/seed is "
      f"{p40['summary']['pooled_lodo_match']:.4f} "
      f"(difference {w['pooled_lodo_matched'] - p40['summary']['pooled_lodo_match']:+.4f}).")
print("  Against THAT value the mean-of-directions margin would be "
      f"{md - p40['summary']['pooled_lodo_match']:+.4f}, not +0.077.")

print("\n  source-count curve (stated 0.673/0.740/0.782/0.815/0.820/0.864/0.809 for k=1,2,3,5,7,9,10):")
curve = w["curve_vs_n_sources"]
for k in sorted(curve, key=int):
    print(f"    {k:>2} sources  {curve[k]:.4f}")

a = load("audit/audit_L40_s42.json")
draws = np.array(a["pooled_lodo_match_draws"])
print(f"\n  30f_result2_audit.py ({len(draws)} row draws of the budget-matched comparator):")
print(f"    mean {draws.mean():.4f}  range [{draws.min():.4f}, {draws.max():.4f}]  "
      f"sd {draws.std():.4f}  max-min spread {draws.max() - draws.min():.4f}")
gap = a["mean_of_directions"] - a["pooled_lodo_full"]
print(f"    mean_of_directions - pooled_lodo_full = {gap:+.4f}")
print(f"    script's rule: margin <= spread  ->  {gap:.4f} <= {draws.max() - draws.min():.4f} "
      f"= {gap <= draws.max() - draws.min()}  "
      f"=> '{'INSIDE the subsampling spread of its own comparator' if gap <= draws.max() - draws.min() else 'outside'}'")
bs = a["bootstrap"]["mean_of_directions-pooled_lodo_full"]
print(f"    stored dataset bootstrap for that margin: diff {bs['diff']:+.4f} "
      f"CI [{bs['lo']:+.4f}, {bs['hi']:+.4f}]  -> {'clears' if bs['lo'] > 0 else 'does not clear'}")

# ------------------------------------------------------------------ F-95
sec("F-95  run-premise/falsify/*.json (63 cells) + the original 2026-09-21 run recovered "
    "from logs/falsify.log")
cells = {}
for L in LAYERS:
    for s in SEEDS:
        j = load(f"falsify/falsify_L{L}_s{s}.json")
        if j is None:
            continue
        worst = max(j["gauss"]["oriented"], j["cov"]["oriented"], j["shuffled"]["oriented"])
        cells[(L, s)] = {"real": j["real"], "budget10": j["budget10"], "null": worst,
                         "gap": j["real"] - worst,
                         "gauss_raw": j["gauss"]["raw"], "cov_raw": j["cov"]["raw"],
                         "shuffled_raw": j["shuffled"]["raw"]}
print(f"  cells on disk: {len(cells)} (ledger: 63)")

for name, key, st_mean, st_lo, st_hi in (("shuffled", "shuffled_raw", 0.5052, 0.4840, 0.5337),
                                         ("gauss", "gauss_raw", 0.5015, 0.4799, 0.5283),
                                         ("cov", "cov_raw", 0.5041, 0.4848, 0.5214)):
    v = np.array([c[key] for c in cells.values()])
    print(f"  null {name:<9} mean {v.mean():.4f} (st {st_mean:.4f})  "
          f"range [{v.min():.4f}, {v.max():.4f}] (st [{st_lo:.4f}, {st_hi:.4f}])")

clear = {k: c["gap"] > 0.02 for k, c in cells.items()}
deep = [k for k in cells if k[0] >= 16]
shallow = [k for k in cells if k[0] < 16]
print(f"\n  cells clearing the best null by >0.02: {sum(clear.values())}/{len(cells)} "
      f"(ledger: 55 of 63)")
print(f"    layers >=16: {sum(clear[k] for k in deep)}/{len(deep)} (ledger: 51 of 51)")
print(f"    layers  <16: {sum(clear[k] for k in shallow)}/{len(shallow)} (ledger: 4 of 12)")
print(f"    failing cells: {sorted([k for k in cells if not clear[k]])}")
gaps_deep = [cells[k]['gap'] for k in deep]
print(f"    min gap at L>=16 = {min(gaps_deep):+.4f} (the gate's SURVIVES threshold is >0.02)")
print(f"    min gap over all 63 = {min(c['gap'] for c in cells.values()):+.4f} "
      f"(ledger quotes the mis-specified gate's -0.0221)")

STATED95 = {0: (0.6298, 0.6403, -0.0105, 0.6064, 0), 4: (0.6424, 0.6386, 0.0038, 0.6113, 0),
            8: (0.6658, 0.6530, 0.0128, 0.6286, 1), 12: (0.6968, 0.6534, 0.0434, 0.6468, 3),
            16: (0.7677, 0.6566, 0.1111, 0.7331, 3), 20: (0.8192, 0.6769, 0.1423, 0.7982, 3),
            28: (0.8544, 0.7041, 0.1503, 0.8429, 3), 32: (0.8857, 0.7104, 0.1753, 0.8716, 3),
            36: (0.8916, 0.7063, 0.1853, 0.8740, 3), 40: (0.8872, 0.7057, 0.1814, 0.8663, 3),
            44: (0.8816, 0.6983, 0.1833, 0.8654, 3), 56: (0.8528, 0.6908, 0.1620, 0.8436, 3),
            76: (0.8343, 0.6925, 0.1418, 0.8342, 3)}
print(f"\n  per-layer, averaged over the 3 seeds (ledger table):")
print(f"  {'L':>4}{'real':>9}{'null':>9}{'gap':>9}{'budget10':>10}{'clear':>7}   stated")
for L in LAYERS:
    ks = [k for k in cells if k[0] == L]
    if not ks:
        continue
    r = statistics.fmean(cells[k]["real"] for k in ks)
    nl = statistics.fmean(cells[k]["null"] for k in ks)
    b10 = statistics.fmean(cells[k]["budget10"] for k in ks)
    nc = sum(clear[k] for k in ks)
    st = STATED95.get(L)
    stxt = f"st {st[0]:.4f} {st[1]:.4f} {st[2]:+.4f} {st[3]:.4f} {st[4]}/3" if st else ""
    print(f"  {L:>4}{r:>9.4f}{nl:>9.4f}{r - nl:>+9.4f}{b10:>10.4f}{str(nc) + '/3':>7}   {stxt}")
peak = max(LAYERS, key=lambda L: statistics.fmean(
    cells[k]["real"] for k in cells if k[0] == L) if any(k[0] == L for k in cells) else -1)
print(f"  peak layer by seed-mean real = L{peak} "
      f"({statistics.fmean(cells[k]['real'] for k in cells if k[0] == peak):.4f})  "
      f"(ledger: L36 0.8916 vs L40 0.8872)")
rb = np.array([cells[k]["real"] - cells[k]["budget10"] for k in cells])
print(f"  real - budget10 over all cells: mean {rb.mean():+.4f} range "
      f"[{rb.min():+.4f}, {rb.max():+.4f}]  (ledger: +0.0171, [-0.022, +0.067])")

# --- the original 2026-09-21 run, recovered from the appended console log ---
txt = open(os.path.join(LOGS, "falsify.log"), "rb").read().decode("utf-16-le", errors="replace")
lines = txt.splitlines()
blocks, start = [], 0
for i, l in enumerate(lines):
    m = re.search(r"falsify_L(\d+)_s(\d+)\.json", l)
    if m and l.strip().startswith("wrote"):
        blocks.append((int(m.group(1)), int(m.group(2)), lines[start:i + 1]))
        start = i + 1


def grab(body, tag):
    for l in body:
        mm = re.match(rf"\s*{tag}\s+([0-9.]+)", l)
        if mm:
            return float(mm.group(1))
    return None


runs = {}
for L, s, body in blocks:
    runs.setdefault((L, s), []).append({t: grab(body, t) for t in
                                        ("real", "real_unnorm", "budget10")})
n_dup = sum(1 for v in runs.values() if len(v) > 1)
print(f"\n  falsify.log holds {len(blocks)} completed cell blocks over {len(runs)} distinct cells; "
      f"{n_dup} cells were run twice (2026-09-21 original + the 2026-09-23 re-run).")
diffs = []
for k, v in runs.items():
    if len(v) > 1 and v[0]["real"] is not None and v[-1]["real"] is not None:
        diffs.append((k, v[0]["real"], v[-1]["real"], v[-1]["real"] - v[0]["real"]))
bad = [d for d in diffs if abs(d[3]) > 1e-4]
print(f"  re-run vs original 'real': {len(diffs)} comparable cells, "
      f"{len(bad)} differ by >1e-4")
for d in bad:
    print(f"    L{d[0][0]} s{d[0][1]}: original {d[1]:.4f} -> re-run {d[2]:.4f} ({d[3]:+.4f})")
first = {k: v[0]["real"] for k, v in runs.items() if v[0]["real"] is not None}
print(f"  original-run 'real' recovered for {len(first)} cells; "
      f"seed-mean at L40 = "
      f"{statistics.fmean(first[k] for k in first if k[0] == 40):.4f} "
      f"(ledger 0.8872), at L36 = "
      f"{statistics.fmean(first[k] for k in first if k[0] == 36):.4f} (ledger 0.8916)")
