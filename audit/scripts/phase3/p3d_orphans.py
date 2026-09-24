"""Phase 3, workstream P3-D. What the four orphan crossgen grids actually show.

Reads ONLY audit/phase2/layer_sensitivity.csv (78,516 rows, built read-only on the lab PC by
audit/scripts/p2_w1_layer_sensitivity.py). Column `offdiag_mean` = mean off-diagonal AUROC of a
train-one-test-all matrix, i.e. cross-dataset transfer; `diag_mean` = in-distribution AUROC.

Stores used here:
  <root>      passB store, whole corpus, 21 layers, 6 seeds      (the store every headline uses)
  passA       200 rows/dataset, 80 layers                        (orphan OR-01 / O44, 92,184 files)
  passC       per-token store pooled, 6 layers                   (orphan OR-02 / O45, 6,936 files)
  passB_fine  whole corpus, layers 37-47, seed 42                (orphan OR-03 / O46)
  passP       PROMPT-side activations, layers 20/40/44, seed 42  (orphan OR-03 / O46, the planned
                                                                  construct-validity control)

COVERAGE TRAP, measured here and obeyed below: in passA, passC and <root>, seeds 0-4 carry ONLY
the features raw / axes / pc:deception; seed 42 carries all eleven. Averaging "over 6 seeds"
therefore compares a 6-seed mean against a 1-seed value for 8 of the 11 features. Every
feature-vs-feature comparison below is run at seed 42 only; seed spread is reported separately
for the three features that have all six seeds.

Usage:  python audit/scripts/phase3/p3d_orphans.py
"""
import csv, os, statistics, sys, collections

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(REPO, "audit", "phase2", "layer_sensitivity.csv")

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
for r in rows:
    r["layer"] = int(r["layer"])
    r["offdiag_mean"] = float(r["offdiag_mean"])
    r["diag_mean"] = float(r["diag_mean"]) if r["diag_mean"] else None

KEY = ("store", "pooling", "layer", "seed", "feature", "probe", "dataset_set")
IDX = {}
for r in rows:
    IDX[tuple(r[k] for k in KEY)] = r
NOSEED = collections.defaultdict(list)
for r in rows:
    NOSEED[(r["store"], r["pooling"], r["layer"], r["feature"], r["probe"], r["dataset_set"])].append(r)


def od(store, pooling, layer, seed, feature, probe, dset="all"):
    r = IDX.get((store, pooling, layer, seed, feature, probe, dset))
    return r["offdiag_mean"] if r else None


def dg(store, pooling, layer, seed, feature, probe, dset="all"):
    r = IDX.get((store, pooling, layer, seed, feature, probe, dset))
    return r["diag_mean"] if r else None


FEATS = ["datapca", "datapca@2", "axes", "axes_resid", "pc:deception", "pc:unified",
         "pc:unified@2", "random", "random@2", "length"]
PROBES = ["paper", "mms", "lr_reg10"]
BAR = "=" * 100

print(BAR)
print("0. COVERAGE: which features exist at which seed, per store (why every comparison below is seed 42)")
print(BAR)
for store in ("<root>", "passA", "passC", "passB_fine", "passP", "kd11/passA", "kd11/passC"):
    c = collections.defaultdict(set)
    for r in rows:
        if r["store"] == store:
            c[r["seed"]].add(r["feature"])
    if not c:
        continue
    print(f"  {store:<12} " + " | ".join(f"seed {s}: {len(c[s])} features" for s in sorted(c)))
    for s in sorted(c):
        if len(c[s]) < 11:
            print(f"      seed {s} has only: {sorted(c[s])}")
        break

print()
print(BAR)
print("A. passP -- PROMPT-SIDE activations (mask = the prompt, not the assistant turn).")
print("   Construct validity: how much of the transfer survives when the model has not answered yet?")
print("   seed 42, 10 datasets, 90 off-diagonal pairs. root = the same rows, response side.")
print(BAR)
print(f"{'layer':>5} {'pool':>5} {'probe':>9} {'feature':>13} {'passP':>8} {'root':>8} {'diff':>8} "
      f"{'passP diag':>11} {'root diag':>10}")
for layer in (20, 40, 44):
    for pooling in ("mean", "last"):
        for probe in PROBES:
            for feat in ["raw", "datapca", "pc:unified@2", "axes", "random", "length"]:
                p, b = od("passP", pooling, layer, "42", feat, probe), od("<root>", pooling, layer, "42", feat, probe)
                if p is None or b is None:
                    continue
                pd_, bd = dg("passP", pooling, layer, "42", feat, probe), dg("<root>", pooling, layer, "42", feat, probe)
                print(f"{layer:>5} {pooling:>5} {probe:>9} {feat:>13} {p:>8.4f} {b:>8.4f} {p-b:>+8.4f} "
                      f"{pd_:>11.4f} {bd:>10.4f}")

print()
print("A2. passP summary: prompt-side transfer as a fraction of the response-side value,")
print("    and against the 0.5 chance floor. 18 (layer x pooling x probe) cells, raw feature.")
num = []
for layer in (20, 40, 44):
    for pooling in ("mean", "last"):
        for probe in PROBES:
            p, b = od("passP", pooling, layer, "42", "raw", probe), od("<root>", pooling, layer, "42", "raw", probe)
            if p and b:
                num.append((layer, pooling, probe, p, b, (p - 0.5) / (b - 0.5)))
print(f"    passP raw offdiag: min {min(x[3] for x in num):.4f} max {max(x[3] for x in num):.4f} "
      f"mean {statistics.fmean(x[3] for x in num):.4f}")
print(f"    root  raw offdiag: min {min(x[4] for x in num):.4f} max {max(x[4] for x in num):.4f} "
      f"mean {statistics.fmean(x[4] for x in num):.4f}")
print(f"    (passP-0.5)/(root-0.5): min {min(x[5] for x in num):.3f} max {max(x[5] for x in num):.3f} "
      f"mean {statistics.fmean(x[5] for x in num):.3f}")
for layer, pooling, probe, p, b, f in num:
    if f > 0.7:
        print(f"      HIGH  L{layer} {pooling} {probe}: passP {p:.4f} vs root {b:.4f} -> {f:.3f} of the signal")
print("    passP in-distribution (diag) vs root diag, raw:")
for layer in (20, 40, 44):
    for pooling in ("mean", "last"):
        p, b = dg("passP", pooling, layer, "42", "raw", "paper"), dg("<root>", pooling, layer, "42", "raw", "paper")
        print(f"      L{layer} {pooling}: passP {p:.4f}  root {b:.4f}  ({p/b:.3f} of it)")

print()
print(BAR)
print("B. passA -- 80 layers, 200 rows/dataset, seed 42, 10 datasets.")
print("   How many of the 80 layers does each feature win against raw, and against the matched")
print("   dataset-PCA control (datapca for 1-d-per-axis features, datapca@2 for the @2 ones)?")
print(BAR)
for pooling in ("mean", "last"):
    for probe in PROBES:
        print(f"\n-- passA {pooling} {probe} (seed 42)")
        print(f"{'feature':>13} {'wins vs raw':>12} {'wins vs datapca':>16} {'mean margin vs raw':>19} "
              f"{'argmax L':>9} {'best margin':>12} {'L40':>9} {'L20':>9}")
        for feat in FEATS:
            m, wr, wd, n = {}, 0, 0, 0
            ctrl = "datapca@2" if feat.endswith("@2") else "datapca"
            for layer in range(80):
                f_ = od("passA", pooling, layer, "42", feat, probe)
                rw = od("passA", pooling, layer, "42", "raw", probe)
                dp = od("passA", pooling, layer, "42", ctrl, probe)
                if f_ is None or rw is None:
                    continue
                m[layer] = f_ - rw
                n += 1
                wr += f_ > rw
                if dp is not None:
                    wd += f_ > dp
            if not m:
                continue
            best = max(m, key=m.get)
            print(f"{feat:>13} {wr:>7}/{n:<4} {wd:>11}/{n:<4} {statistics.fmean(m.values()):>+19.4f} "
                  f"{best:>9} {m[best]:>+12.4f} {m.get(40, float('nan')):>+9.4f} {m.get(20, float('nan')):>+9.4f}")

print()
print(BAR)
print("C. passA / <root> -- where is the best layer for RAW cross-dataset transfer?")
print("   (L40 is the project's pre-registered layer. seed 42.)")
print(BAR)
for store, lays in (("passA", range(80)), ("<root>", [0, 4, 8, 12, 16, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76])):
    for pooling in ("mean", "last"):
        for probe in PROBES:
            prof = {L: od(store, pooling, L, "42", "raw", probe) for L in lays}
            prof = {k: v for k, v in prof.items() if v is not None}
            if not prof:
                continue
            best = max(prof, key=prof.get)
            top5 = sorted(prof, key=prof.get, reverse=True)[:5]
            print(f"{store:>8} raw {pooling:<5} {probe:<9} argmax L{best:<3} {prof[best]:.4f} | "
                  f"L40 {prof.get(40, float('nan')):.4f} | L20 {prof.get(20, float('nan')):.4f} | top5 {top5}")

print()
print(BAR)
print("D. passA / passC / <root> -- SEED SPREAD. Every headline is reported at one seed (42).")
print("   Only raw / axes / pc:deception have all six seeds. 6-seed sd of the off-diagonal mean,")
print("   and of the axes-minus-raw margin computed seed by seed.")
print(BAR)
print(f"{'store':>8} {'layer':>5} {'pool':>5} {'probe':>9} {'feature':>13} {'seed42':>8} {'mean6':>8} "
      f"{'sd6':>7} {'min':>7} {'max':>7} {'range':>7}")
for store, lays in (("passA", (20, 32, 40, 44)), ("passC", (20, 40, 44)), ("<root>", (20, 40, 44))):
    for layer in lays:
        for pooling in ("mean",):
            for probe in ("mms", "lr_reg10"):
                for feat in ("raw", "axes", "pc:deception"):
                    rr = NOSEED.get((store, pooling, layer, feat, probe, "all"), [])
                    if len(rr) < 2:
                        continue
                    v = {r["seed"]: r["offdiag_mean"] for r in rr}
                    vals = list(v.values())
                    print(f"{store:>8} {layer:>5} {pooling:>5} {probe:>9} {feat:>13} "
                          f"{v.get('42', float('nan')):>8.4f} {statistics.fmean(vals):>8.4f} "
                          f"{statistics.stdev(vals):>7.4f} {min(vals):>7.4f} {max(vals):>7.4f} "
                          f"{max(vals)-min(vals):>7.4f}")
print("   margin (axes - raw) per seed, mean pooling, mms probe (the probe seeds 0-4 carry):")
for store in ("passA", "passC", "<root>"):
    for layer in (20, 32, 40, 44):
        for dset in ("all", "paper5"):
            out = []
            for s in ("0", "1", "2", "3", "4", "42"):
                a = od(store, "mean", layer, s, "axes", "mms", dset)
                r_ = od(store, "mean", layer, s, "raw", "mms", dset)
                if a is not None and r_ is not None:
                    out.append((s, a - r_))
            if len(out) > 1:
                vals = [x[1] for x in out]
                print(f"      {store:<8} L{layer:<3} {dset:<7} " + "  ".join(f"s{s}={d:+.4f}" for s, d in out) +
                      f"   | sd {statistics.stdev(vals):.4f}  sign flips: "
                      f"{'YES' if min(vals) < 0 < max(vals) else 'no'}")

print()
print(BAR)
print("E. passB_fine -- layers 37-47 at full corpus, seed 42. Is L40 the peak of its neighbourhood?")
print(BAR)
FINE = sorted(set(r["layer"] for r in rows if r["store"] == "passB_fine"))
for pooling in ("mean", "last"):
    for probe in PROBES:
        pts = [(L, od("passB_fine", pooling, L, "42", "raw", probe)) for L in FINE]
        pts += [(L, od("<root>", pooling, L, "42", "raw", probe)) for L in (36, 40, 44, 48)]
        pts = sorted({L: v for L, v in pts if v is not None}.items())
        if not pts:
            continue
        best = max(pts, key=lambda t: t[1])
        print(f"\npassB_fine+root raw {pooling} {probe}: " + "  ".join(f"L{L}={v:.4f}" for L, v in pts))
        print(f"   argmax over 36-48 = L{best[0]} ({best[1]:.4f});  L40 = "
              f"{dict(pts).get(40, float('nan')):.4f}  (rank "
              f"{sorted((v for _, v in pts), reverse=True).index(dict(pts)[40]) + 1} of {len(pts)})")

print()
print(BAR)
print("F. passB_fine -- does the persona margin peak at L40? pc:unified@2 vs raw and vs datapca@2")
print(BAR)
for probe in PROBES:
    print(f"\npassB_fine+root mean {probe}")
    for L in sorted(set(FINE) | {36, 40, 44, 48}):
        st = "passB_fine" if L in FINE else "<root>"
        f_, rw, dp = (od(st, "mean", L, "42", "pc:unified@2", probe), od(st, "mean", L, "42", "raw", probe),
                      od(st, "mean", L, "42", "datapca@2", probe))
        if None in (f_, rw, dp):
            continue
        print(f"   L{L:<3} ({st:<10}) pc:unified@2 {f_:.4f}  raw {rw:.4f}  datapca@2 {dp:.4f}"
              f"   vs raw {f_-rw:+.4f}   vs datapca@2 {f_-dp:+.4f}")

print()
print(BAR)
print("G. passC -- per-token store's POOLED vectors vs <root>. An independent-sample replication:")
print("   200 rows/dataset re-extracted in a separate NDIF pass, same layers. seed 42.")
print(BAR)
print(f"{'layer':>5} {'pool':>5} {'probe':>9} {'feature':>13} {'passC':>8} {'root':>8} {'diff':>8}")
diffs = collections.defaultdict(list)
for layer in (20, 22, 28, 40, 44, 48):
    for pooling in ("mean", "last"):
        for probe in PROBES:
            for feat in ["raw", "datapca", "pc:unified@2", "axes", "pc:deception"]:
                c, b = od("passC", pooling, layer, "42", feat, probe), od("<root>", pooling, layer, "42", feat, probe)
                if c is None or b is None:
                    continue
                diffs[feat].append(c - b)
                if probe == "paper":
                    print(f"{layer:>5} {pooling:>5} {probe:>9} {feat:>13} {c:>8.4f} {b:>8.4f} {c-b:>+8.4f}")
print("\n   passC - root over all 36 (layer x pooling x probe) cells per feature:")
for f, d in diffs.items():
    print(f"      {f:>13} n={len(d):<3} mean {statistics.fmean(d):+.4f}  sd {statistics.stdev(d):.4f}  "
          f"min {min(d):+.4f}  max {max(d):+.4f}")

print()
print(BAR)
print("H. passA vs <root>: the same comparison on 200 rows/dataset and on the whole corpus.")
print("   pc:unified@2 - raw and axes - raw at the 21 shared layers, seed 42, mean pooling, paper probe.")
print(BAR)
SHARED = [0, 4, 8, 12, 16, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76]
for feat in ("axes", "pc:unified@2", "pc:deception", "datapca"):
    a = [(L, od("passA", "mean", L, "42", feat, "paper") - od("passA", "mean", L, "42", "raw", "paper"))
         for L in SHARED if od("passA", "mean", L, "42", feat, "paper") is not None]
    b = [(L, od("<root>", "mean", L, "42", feat, "paper") - od("<root>", "mean", L, "42", "raw", "paper"))
         for L in SHARED if od("<root>", "mean", L, "42", feat, "paper") is not None]
    print(f"\n  {feat}: passA wins {sum(1 for _, d in a if d > 0)}/{len(a)} layers, "
          f"mean {statistics.fmean(d for _, d in a):+.4f}")
    print(f"  {' ' * len(feat)}  root  wins {sum(1 for _, d in b if d > 0)}/{len(b)} layers, "
          f"mean {statistics.fmean(d for _, d in b):+.4f}")
