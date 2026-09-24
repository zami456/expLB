"""Phase 3, P3-D supplement: (1) is passC's crossgen grid a duplicate of passA's?
(2) the L40 decomposition -- is the persona advantage at L40 the axis improving, or raw degrading?
(3) the layer profile of raw vs axes on passA (80 layers).

Reads only audit/phase2/layer_sensitivity.csv.
"""
import csv, os, statistics, sys, collections

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
rows = list(csv.DictReader(open(os.path.join(REPO, "audit", "phase2", "layer_sensitivity.csv"), encoding="utf-8")))
for r in rows:
    r["layer"] = int(r["layer"])
    r["offdiag_mean"] = float(r["offdiag_mean"])
KEY = ("store", "pooling", "layer", "seed", "feature", "probe", "dataset_set")
IDX = {tuple(r[k] for k in KEY): r for r in rows}


def od(*a):
    r = IDX.get(a)
    return r["offdiag_mean"] if r else None


BAR = "=" * 100
print(BAR)
print("1. passC crossgen grid vs passA crossgen grid at the 6 shared layers.")
print("   Both evaluate the SAME 200-row/dataset sample (--sample 200 --seed 42); passA was")
print("   extracted in one NDIF job (80 layers, pooled) and passC in another (6 layers, per-token,")
print("   pooled vectors also stored). If the two agree, the passC grid is a re-extraction check,")
print("   not new evidence.  |diff| over every matching cell:")
print(BAR)
d = []
for L in (20, 22, 28, 40, 44, 48):
    for pool in ("mean", "last"):
        for probe in ("paper", "mms", "lr_reg10"):
            for seed in ("0", "1", "2", "3", "4", "42"):
                for feat in ("raw", "axes", "pc:deception", "datapca", "pc:unified", "pc:unified@2",
                             "random", "length", "axes_resid", "datapca@2", "random@2"):
                    for ds in ("all", "paper5"):
                        a, c = od("passA", pool, L, seed, feat, probe, ds), od("passC", pool, L, seed, feat, probe, ds)
                        if a is not None and c is not None:
                            d.append((abs(a - c), L, pool, probe, seed, feat, ds, a, c))
print(f"   matched cells: {len(d)}")
print(f"   max |passA - passC| = {max(x[0] for x in d):.6f}   mean = {statistics.fmean(x[0] for x in d):.6f}")
print(f"   cells with |diff| > 0.001: {sum(1 for x in d if x[0] > 0.001)}")
print(f"   cells with |diff| > 0.01 : {sum(1 for x in d if x[0] > 0.01)}")
worst = sorted(d, reverse=True)[:5]
for x in worst:
    print(f"      worst: L{x[1]} {x[2]} {x[3]} s{x[4]} {x[5]} {x[6]}: passA {x[7]:.6f} passC {x[8]:.6f} "
          f"(|d| {x[0]:.6f})")

print()
print(BAR)
print("2. The L40 decomposition. The project pre-registered L40. Raw transfer peaks at L31-33.")
print("   Split the persona 'margin' at L40 into: how much the axis gained vs its own peak, and")
print("   how much raw lost vs its own peak. seed 42, 10 datasets.")
print(BAR)
for store in ("passA", "<root>"):
    lays = range(80) if store == "passA" else [0, 4, 8, 12, 16, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76]
    for pool in ("mean", "last"):
        for probe in ("paper", "mms", "lr_reg10"):
            for feat in ("axes", "pc:unified@2"):
                prof_f = {L: od(store, pool, L, "42", feat, probe, "all") for L in lays}
                prof_r = {L: od(store, pool, L, "42", "raw", probe, "all") for L in lays}
                prof_f = {k: v for k, v in prof_f.items() if v is not None}
                prof_r = {k: v for k, v in prof_r.items() if v is not None}
                if 40 not in prof_f or 40 not in prof_r:
                    continue
                bf, br = max(prof_f, key=prof_f.get), max(prof_r, key=prof_r.get)
                print(f"  {store:<8} {pool:<5} {probe:<9} {feat:<13} "
                      f"{feat} peak L{bf} {prof_f[bf]:.4f} -> L40 {prof_f[40]:.4f} ({prof_f[40]-prof_f[bf]:+.4f}) | "
                      f"raw peak L{br} {prof_r[br]:.4f} -> L40 {prof_r[40]:.4f} ({prof_r[40]-prof_r[br]:+.4f}) | "
                      f"margin at L40 {prof_f[40]-prof_r[40]:+.4f}, at raw's peak L{br} "
                      f"{prof_f[br]-prof_r[br]:+.4f}")

print()
print(BAR)
print("3. passA 80-layer profile, mean pooling, paper probe, 10 datasets, seed 42.")
print(BAR)
print(f"{'L':>3} {'raw':>7} {'axes':>7} {'pc:u@2':>7} {'datapca':>8} {'random':>7} {'length':>7} {'axes-raw':>9}")
for L in range(0, 80, 2):
    vals = [od("passA", "mean", L, "42", f, "paper", "all") for f in
            ("raw", "axes", "pc:unified@2", "datapca", "random", "length")]
    if vals[0] is None:
        continue
    s = " ".join(f"{v:7.4f}" if v is not None else "    n/a" for v in vals)
    print(f"{L:>3} {s} {vals[1]-vals[0]:>+9.4f}")

print()
print(BAR)
print("4. paper5 subset on passA: the paper's own five datasets, 80 layers, seed 42.")
print("   (The paper's claim is about ITS five datasets, so this is the like-for-like grid.)")
print(BAR)
for pool in ("mean", "last"):
    for probe in ("paper", "mms", "lr_reg10"):
        for feat in ("axes", "pc:unified", "pc:unified@2", "pc:deception", "random", "length", "datapca"):
            m, wr, wd, n = {}, 0, 0, 0
            ctrl = "datapca@2" if feat.endswith("@2") else "datapca"
            for L in range(80):
                f_, rw = od("passA", pool, L, "42", feat, probe, "paper5"), od("passA", pool, L, "42", "raw", probe, "paper5")
                dp = od("passA", pool, L, "42", ctrl, probe, "paper5")
                if f_ is None or rw is None:
                    continue
                m[L] = f_ - rw
                n += 1
                wr += f_ > rw
                if dp is not None:
                    wd += f_ > dp
            if not m:
                continue
            best = max(m, key=m.get)
            print(f"  passA paper5 {pool:<5} {probe:<9} {feat:<13} wins vs raw {wr:>2}/{n} | vs {ctrl} {wd:>2}/{n} "
                  f"| mean {statistics.fmean(m.values()):+.4f} | argmax L{best} {m[best]:+.4f} | L40 {m.get(40, float('nan')):+.4f}")
