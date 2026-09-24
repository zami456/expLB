"""Lead QA for batch P7 — independent re-derivation of the two claims with the widest blast radius.

Written from the ledger text, not from the batch agent's script: its own loader, its own margin code,
its own bootstrap. Read-only. Runs on the lab PC (both inputs are lab-only).

QA-1  F-75 vs F-77/F-81: how many of the 80 layers does the kd matched slice actually win at?
      F-75 says 28/80 vs raw and 31/80 vs datapca; F-77 (FINDINGS.md:2934) and F-81 (:3233) both
      print 37/80 in the row that pairs with the cap bank's 21/21. The batch says 37 is F-76's
      bootstrap MEDIAN FRACTION of positive layers (0.4625) x 80, i.e. a fraction printed as a count.
      This re-derives all three numbers.

QA-2  F-73's per-dataset length framing: "the positive class is longer in N of M datasets".
      Five documents plus a memory note say 9 of 10. Recompute from span_len directly.
"""
import json, os, statistics, sys

import numpy as np

ROOT = "C:/Users/user4/expLB"
AXES = f"{ROOT}/run-crossgen/kd11/passA11/banks/kd_s23/matrices"
BASE = f"{ROOT}/run-crossgen/kd11/passA11/matrices"
PASSB = f"{ROOT}/run-coords/passB"
TRAIN = "harm-pressure-kr-matched"
PROBE, POOL, SEED = "paper", "mean", 42


def load(d, layer):
    p = f"{d}/{POOL}_L{layer}_s{SEED}.json"
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def row_means(m, feature):
    """Mean AUROC of `feature` over the off-diagonal cells of the TRAIN row, and the per-target vector."""
    for r in m["results"]:
        if r["feature"] == feature and r["probe"] == PROBE:
            ds = m["datasets"]
            i = ds.index(TRAIN)
            vals = [(te, v) for te, v in zip(ds, r["auroc"][i]) if te != TRAIN and v is not None]
            return vals
    return None


def qa1():
    print("=== QA-1  kd matched slice, 80-layer margin profile (my own loader) ===")
    layers, A, R, D, targets = [], [], [], [], None
    for L in range(80):
        ma, mb = load(AXES, L), load(BASE, L)
        if ma is None or mb is None:
            continue
        a, r, d = row_means(ma, "axes"), row_means(mb, "raw"), row_means(mb, "datapca")
        if not (a and r and d):
            continue
        names = [t for t, _ in a]
        if targets is None:
            targets = names
        assert names == targets == [t for t, _ in r] == [t for t, _ in d], f"target order differs at L{L}"
        layers.append(L)
        A.append([v for _, v in a]); R.append([v for _, v in r]); D.append([v for _, v in d])
    A, R, D = np.array(A), np.array(R), np.array(D)
    print(f"layers loaded: {len(layers)} (L{min(layers)}..L{max(layers)})   targets: {len(targets)}")
    print("targets:", ", ".join(targets))
    ma, mr, md = A.mean(1), R.mean(1), D.mean(1)
    n_raw, n_dpca = int((ma > mr).sum()), int((ma > md).sum())
    print(f"\n  axes beats raw     at {n_raw}/{len(layers)} layers      [F-75 says 28/80]")
    print(f"  axes beats datapca at {n_dpca}/{len(layers)} layers      [F-75 says 31/80; F-77/F-81 say 37/80]")
    for L in (20, 40, 44, 77):
        if L in layers:
            i = layers.index(L)
            print(f"    L{L:<3d} axes {ma[i]:.4f}  raw {mr[i]:.4f}  datapca {md[i]:.4f}   "
                  f"a-raw {ma[i]-mr[i]:+.4f}  a-dpca {ma[i]-md[i]:+.4f}")

    # dataset-level bootstrap over the 10 target datasets (the unit F-76 used)
    B, rng = 20000, np.random.default_rng(SEED)
    n = len(targets)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)     # (B, n)
    W /= W.sum(1, keepdims=True)
    marg = (A - D) @ W.T                                                   # (layers, B)
    frac_pos = (marg > 0).mean(0)                                          # fraction of layers positive per replicate
    i40 = layers.index(40)
    lo, hi = np.percentile(marg[i40], [2.5, 97.5])
    print(f"\n  bootstrap (B={B}, unit = the {n} target datasets, seed {SEED}):")
    print(f"    L40 margin vs datapca {marg[i40].mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"   P(margin<=0) = {(marg[i40] <= 0).mean():.4f}        [F-76: [-0.0693,+0.1887], P=0.161]")
    med = float(np.median(frac_pos))
    print(f"    median FRACTION of layers with a positive margin = {med:.4f}  ->  x80 = {med*80:.1f}"
          f"        [F-77/F-81 print '37/80']")
    print(f"    mean fraction {frac_pos.mean():.4f} (x80 = {frac_pos.mean()*80:.1f});  "
          f"2.5/97.5 pct of the fraction: {np.percentile(frac_pos,2.5):.4f} / {np.percentile(frac_pos,97.5):.4f}")
    am = marg.argmax(0)
    print(f"    argmax layer 95% interval: [L{layers[int(np.percentile(am,2.5))]}, "
          f"L{layers[int(np.percentile(am,97.5))]}];  P(argmax==L40) = {(am == i40).mean():.4f}")


def qa2():
    print("\n=== QA-2  F-73 length framing: positive class longer in how many datasets? ===")
    rows = []
    for name in sorted(os.listdir(PASSB)):
        p = f"{PASSB}/{name}/rows.jsonl"
        if not os.path.exists(p):
            continue
        pos, neg, skipped, seen = [], [], 0, set()
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                lab = r.get("label")
                seen.add(repr(lab))
                if lab is None or r.get("span_len") is None:
                    skipped += 1
                    continue
                (pos if int(lab) == 1 else neg).append(float(r["span_len"]))
        if pos and neg:
            rows.append((name, len(pos), len(neg), statistics.fmean(pos), statistics.fmean(neg),
                         statistics.fmean(pos) - statistics.fmean(neg), skipped, sorted(seen)))
    print(f"  {'dataset':36s} {'n_pos':>6s} {'n_neg':>6s} {'pos len':>10s} {'neg len':>10s} {'gap':>10s}  skipped  labels")
    for name, npos, nneg, mp, mn, gap, sk, seen in rows:
        print(f"  {name:36s} {npos:6d} {nneg:6d} {mp:10.4f} {mn:10.4f} {gap:+10.4f}  {sk:7d}  {','.join(seen)}")
    longer = [r for r in rows if r[5] > 0]
    print(f"\n  positive class longer in {len(longer)} of {len(rows)} datasets")
    b9 = [r for r in rows if r[0] != "sycophancy_mt"]
    print(f"  excluding sycophancy_mt (block 9): {sum(1 for r in b9 if r[5] > 0)} of {len(b9)}")
    strict = [r for r in rows if r[5] > 1.0]
    print(f"  gap > 1 token: {len(strict)} of {len(rows)}")
    print(f"  median gap over all {len(rows)}: {statistics.median(r[5] for r in rows):+.4f}")
    inrepo = [r for r in rows if r[0] not in ("harm-pressure-kr-matched", "mask", "sycophancy_mt")]
    print(f"  median gap over the 8 'a reviewer could see': "
          f"{statistics.median(r[5] for r in inrepo):+.4f}  (n={len(inrepo)})   [F-73 says 3.7]")


def qa2b():
    """Which recipe gives F-73's '3.7 tokens median gap on the eight a reviewer could see'?
    My median-of-means over the obvious eight is +2.63, so enumerate the plausible variants."""
    import itertools
    print("\n=== QA-2b  reconciling F-73's median gap = 3.689 ===")
    data = {}
    for name in sorted(os.listdir(PASSB)):
        p = f"{PASSB}/{name}/rows.jsonl"
        if not os.path.exists(p):
            continue
        pos, neg = [], []
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("label") is None or r.get("span_len") is None:
                    continue
                (pos if int(r["label"]) == 1 else neg).append(float(r["span_len"]))
        if pos and neg:
            data[name] = dict(
                mean_gap=statistics.fmean(pos) - statistics.fmean(neg),
                median_gap=statistics.median(pos) - statistics.median(neg),
                ratio=statistics.fmean(pos) / statistics.fmean(neg))
    names = sorted(data)
    print(f"  {'dataset':36s} {'mean gap':>11s} {'median gap':>11s}")
    for n in names:
        print(f"  {n:36s} {data[n]['mean_gap']:+11.4f} {data[n]['median_gap']:+11.4f}")
    for stat in ("mean_gap", "median_gap"):
        allv = [data[n][stat] for n in names]
        print(f"\n  median of {stat} over all 11: {statistics.median(allv):+.4f}")
        b9 = [data[n][stat] for n in names if n != "sycophancy_mt"]
        print(f"  median of {stat} over block9 (10): {statistics.median(b9):+.4f}")
        # every 8-subset whose median lands on 3.689 +/- 0.01
        hits = [c for c in itertools.combinations(names, 8)
                if abs(statistics.median([data[n][stat] for n in c]) - 3.689) < 0.01]
        print(f"  8-subsets of the 11 whose median {stat} = 3.689: {len(hits)}")
        for c in hits[:6]:
            print("     ", ", ".join(sorted(set(names) - set(c))), "  <- excluded")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "qa1"):
        qa1()
    if which in ("both", "qa2"):
        qa2()
    if which in ("both", "qa2b"):
        qa2b()
