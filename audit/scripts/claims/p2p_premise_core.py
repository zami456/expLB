"""Phase-2 batch P2-pooled: F-79, F-80, F-81, F-83.

Reads the md5-verified cache of the lab PC's run-premise/ (audit/cache/labpc/run-premise).
Level 1  = the ledger's number vs the number in the result file.
Level 2  = the dataset-level paired bootstrap re-implemented here from per_dataset values,
           i.e. NOT read out of any stored file (F-80/F-81's CIs were console-only).

Run:  PYTHONIOENCODING=utf-8 python audit/scripts/claims/p2p_premise_core.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import numpy as np

ROOT = os.path.join("audit", "cache", "labpc", "run-premise")
LAYERS = [0, 4, 8, 12, 16, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76]


def load(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def boot(per, a, b, B=10000, seed=42):
    """Exactly 30b_premise_boot.py's procedure: names sorted, rng=default_rng(seed),
    idx = rng.integers(0, n, size=(B, n)), paired difference, percentile 2.5/97.5."""
    names = sorted(per)
    n = len(names)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    va = np.array([per[k][a] for k in names])
    vb = np.array([per[k][b] for k in names])
    dif = va - vb
    bs = dif[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(dif.mean()), float(lo), float(hi), float((bs <= 0).mean()), n


def sec(t):
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


# ---------------------------------------------------------------- F-79
sec("F-79  premise_L40_mean_s42.json  (Level 1: six headline AUROCs + two margins)")
j = load("premise_L40_mean_s42.json")
s = j["summary"]
print(f"  file: {ROOT}/premise_L40_mean_s42.json")
print(f"  layer={j['layer']} pooling={j['pooling']} probe={j['probe']} seed={j['seed']} "
      f"n_datasets={len(j['datasets'])}")
print(f"  n_train = {j['n_train']}")
STATED79 = {"single_own_mean": 0.9506, "single": 0.6763, "pooled_full": 0.9447,
            "pooled_match": 0.8924, "pooled_lodo": 0.8545, "pooled_lodo_match": 0.8500}
for k, v in STATED79.items():
    got = s[k]
    print(f"  {k:<18} stated {v:.4f}  file {got:.6f}  |d|={abs(got - v):.6f}  "
          f"{'MATCH' if abs(got - v) <= 5e-5 else 'DIFF'}")
m1 = s["pooled_lodo"] - s["single"]
m2 = s["pooled_lodo_match"] - s["single"]
print(f"  margin  pooled_lodo - single        stated +0.178  computed {m1:+.6f}")
print(f"  margin  pooled_lodo_match - single  stated +0.174  computed {m2:+.6f}")
print(f"  predictors: cos R2={j['predictors']['cos']['r2']:.4f} (stated 0.244)  "
      f"mahal_cos R2={j['predictors']['mahal_cos']['r2']:.4f} (stated 0.017)  "
      f"covariance field = {j.get('covariance')!r}  n_pairs={len(j['pairs'])} (stated 110)")

# ---------------------------------------------------------------- F-80
sec("F-80  dataset-level paired bootstrap, recomputed (B=10,000, seed 42, unit = target dataset)")
per = j["per_dataset"]
PAIRS = [("pooled_lodo", "single_offdiag", "+0.1782", "[+0.1430, +0.2129]", "0.0000"),
         ("pooled_lodo_match", "single_offdiag", "+0.1737", "[+0.1256, +0.2239]", "0.0000"),
         ("pooled_lodo", "pooled_lodo_match", "+0.0045", "[-0.0234, +0.0325]", "0.378"),
         ("single_own", "pooled_lodo", "+0.0962", "[+0.0553, +0.1360]", "0.0000")]
for a, b, sd, sci, sp in PAIRS:
    d, lo, hi, p, n = boot(per, a, b)
    print(f"  {a} - {b}")
    print(f"      stated   diff {sd}  CI {sci}  P(<=0) {sp}")
    print(f"      computed diff {d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  P(<=0) {p:.4f}  n={n}  "
          f"{'CLEARS' if lo > 0 else ''}")

print("\n  per-target table (stated vs file):")
STATED80 = {"harm-pressure-knowledge-report": (0.693, 0.972, 0.279),
            "insider-trading": (0.607, 0.856, 0.249),
            "convincing-game": (0.721, 0.955, 0.234),
            "harm-pressure-kr-matched": (0.613, 0.806, 0.193),
            "sycophancy_mt": (0.617, 0.802, 0.185),
            "roleplaying": (0.644, 0.824, 0.180),
            "sandbagging_v2": (0.791, 0.970, 0.178),
            "ai_liar": (0.691, 0.857, 0.166),
            "instructed-deception": (0.822, 0.941, 0.119),
            "harm-pressure-choice": (0.652, 0.756, 0.103),
            "mask": (0.588, 0.660, 0.072)}
bad = 0
wins = 0
for k, (ss, sl, sd) in STATED80.items():
    r = per[k]
    fs, fl = r["single_offdiag"], r["pooled_lodo"]
    fd = fl - fs
    ok = abs(fs - ss) <= 5e-4 and abs(fl - sl) <= 5e-4 and abs(fd - sd) <= 1e-3
    bad += 0 if ok else 1
    wins += 1 if fd > 0 else 0
    print(f"    {k:<32} single {fs:.3f} (st {ss:.3f})  lodo {fl:.3f} (st {sl:.3f})  "
          f"diff {fd:+.3f} (st {sd:+.3f})  {'ok' if ok else 'DIFF'}")
print(f"  rows disagreeing with the ledger: {bad}/11;  targets where pooled beats single: {wins}/11 "
      f"(ledger says 11 of 11)")
worst = min(per, key=lambda k: per[k]["pooled_lodo"] - per[k]["single_offdiag"])
best = max(per, key=lambda k: per[k]["pooled_lodo"] - per[k]["single_offdiag"])
print(f"  worst target {worst} {per[worst]['pooled_lodo'] - per[worst]['single_offdiag']:+.3f} "
      f"(ledger: mask +0.072);  best {best} "
      f"{per[best]['pooled_lodo'] - per[best]['single_offdiag']:+.3f} (ledger: HP-KR +0.279)")

# ---------------------------------------------------------------- F-81
sec("F-81  all 21 layers: point estimates + per-layer dataset bootstrap (recomputed)")
STATED81 = {0: (0.529, 0.659, 0.569, 0.812, 0.130, "9/11"),
            8: (0.544, 0.694, 0.645, 0.870, 0.150, "10/11"),
            16: (0.602, 0.739, 0.767, 0.917, 0.137, "9/11"),
            20: (0.632, 0.803, 0.824, 0.922, 0.171, "11/11"),
            28: (0.668, 0.835, 0.841, 0.931, 0.167, "11/11"),
            36: (0.693, 0.860, 0.868, 0.952, 0.167, "11/11"),
            40: (0.676, 0.855, 0.850, 0.951, 0.178, "11/11"),
            52: (0.661, 0.845, 0.829, 0.947, 0.184, "11/11"),
            68: (0.652, 0.809, 0.817, 0.948, 0.157, "11/11"),
            76: (0.666, 0.828, 0.837, 0.948, 0.163, "10/11")}
print(f"  {'L':>4}{'single':>9}{'lodo':>9}{'rowmatch':>10}{'indomain':>10}{'margin':>9}"
      f"{'wins':>7}{'CI low':>10}{'CI high':>10}{'P<=0':>8}  stated")
rows = []
n_pos = n_clear = n_win11 = 0
for L in LAYERS:
    jj = load(f"premise_L{L}_mean_s42.json")
    if jj is None:
        print(f"  {L:>4}  MISSING")
        continue
    ss = jj["summary"]
    pp = jj["per_dataset"]
    marg = ss["pooled_lodo"] - ss["single"]
    d, lo, hi, p, n = boot(pp, "pooled_lodo", "single_offdiag")
    w = sum(1 for k in pp if pp[k]["pooled_lodo"] > pp[k]["single_offdiag"])
    n_pos += 1 if marg > 0 else 0
    n_clear += 1 if lo > 0 else 0
    n_win11 += 1 if w == len(pp) else 0
    st = STATED81.get(L)
    stxt = (f"st {st[0]:.3f}/{st[1]:.3f}/{st[2]:.3f}/{st[3]:.3f} {st[4]:+.3f} {st[5]}" if st else "")
    print(f"  {L:>4}{ss['single']:>9.3f}{ss['pooled_lodo']:>9.3f}{ss['pooled_lodo_match']:>10.3f}"
          f"{ss['single_own_mean']:>10.3f}{marg:>+9.3f}{str(w) + '/' + str(len(pp)):>7}"
          f"{lo:>+10.4f}{hi:>+10.4f}{p:>8.4f}  {stxt}")
    rows.append((L, marg, lo, hi, w, len(pp)))
margins = [r[1] for r in rows]
print(f"\n  layers present: {len(rows)} (ledger: 21)")
print(f"  margin positive at {n_pos}/{len(rows)} layers (ledger: 21/21)")
print(f"  95% CI excludes zero at {n_clear}/{len(rows)} layers (ledger: 21/21)")
print(f"  11/11 targets won at {n_win11}/{len(rows)} layers (ledger: 13)")
lo_l = min(rows, key=lambda r: r[1])
hi_l = max(rows, key=lambda r: r[1])
print(f"  margin range {min(margins):+.4f} (L{lo_l[0]}) to {max(margins):+.4f} (L{hi_l[0]})  "
      f"(ledger: +0.128 at L4 to +0.184 at L52)")

# ---------------------------------------------------------------- F-83
sec("F-83  seed sweep at L40, and the Mahalanobis predictors")
STATED83 = {1: 0.1728, 7: 0.1979, 13: 0.1845, 42: 0.1782, 99: 0.1777}
ms = []
for sd in (1, 7, 13, 42, 99):
    jj = load(f"premise_L40_mean_s{sd}.json")
    if jj is None:
        print(f"  seed {sd}: MISSING")
        continue
    ss = jj["summary"]
    m = ss["pooled_lodo"] - ss["single"]
    ms.append(m)
    print(f"  seed {sd:>2}  single {ss['single']:.4f}  lodo {ss['pooled_lodo']:.4f}  "
          f"margin {m:+.4f}  (ledger {STATED83[sd]:+.4f}, |d|={abs(m - STATED83[sd]):.4f})  "
          f"cov={jj.get('covariance')}  n_ds={len(jj['datasets'])}")
print(f"  range {min(ms):+.4f} to {max(ms):+.4f}  spread {max(ms) - min(ms):.4f} "
      f"(ledger: +0.173 to +0.198, spread 0.025)")
d, lo, hi, p, n = boot(load("premise_L40_mean_s42.json")["per_dataset"],
                       "pooled_lodo", "single_offdiag")
print(f"  ledger compares that spread with the bootstrap CI width: computed CI "
      f"[{lo:+.4f}, {hi:+.4f}] width {hi - lo:.4f}  (ledger quotes +0.143..+0.213, width 0.070)")

print("\n  Mahalanobis / cosine R^2 at L40:")
for rel, tag in (("premise_L40_mean_s42.json", "root file (F-79's)"),
                 ("mahal/premise_L40_mean_s42.json", "mahal/ within-class"),
                 ("mahal/premise_L40_mean_s42_total.json", "mahal/ total")):
    jj = load(rel)
    if jj is None:
        print(f"    {rel}: MISSING")
        continue
    pr = jj["predictors"]
    print(f"    {rel:<42} cov={str(jj.get('covariance')):<7} n_pairs={len(jj['pairs']):>3}  "
          f"cos R2={pr['cos']['r2']:.4f}  mahal R2={pr['mahal_cos']['r2']:.4f}   [{tag}]")
print("    ledger: plain cosine 0.244; Mahalanobis within-class 0.023; Mahalanobis total 0.017")

# also: do the seed files carry their own per-dataset bootstrap? (they should not)
jj = load("premise_L40_mean_s7.json")
print(f"\n  keys stored in a seed file: {sorted(jj.keys())}")
print("  -> no bootstrap CI is stored in any premise JSON; F-80/F-81/F-83 CIs are console-only "
      "and are recomputed above.")
