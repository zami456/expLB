"""Compare the leaked and leak-guarded runs of the F-03 cell.

The question: does F-03's wording-matched result depend on the row-level leak
between `harm-pressure-knowledge-report` and `harm-pressure-kr-matched`?

  ORIGINAL  run-crossgen/kd11/matrices/mean_L{20,40,44}_s42.json   (git tree)
  GUARDED   audit/cache/labpc/expLB-resplit/mean_L{20,40,44}_s42.json

The guarded run was produced 2026-09-23 on the lab PC from a COPY of
27_crossgen.py patched by p6_patch_leakguard.py, reading the same activations
read-only and writing only to C:\\Users\\user4\\expLB-resplit. Originals untouched
(md5 681126242fe4414e4a3c131793de81c6 before and after).

  python audit/scripts/p6_resplit_compare.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FULL = "harm-pressure-knowledge-report"
MATCHED = "harm-pressure-kr-matched"
PROBE = "paper"
ORIG = os.path.join(ROOT, "run-crossgen", "kd11", "matrices", "mean_L{}_s42.json")
GUARD = os.path.join(ROOT, "audit", "cache", "labpc", "expLB-resplit", "mean_L{}_s42.json")


def load(tmpl, layer):
    p = tmpl.format(layer)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def idx(d):
    return {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}


def offdiag(d, feat, src):
    blk = idx(d).get((feat, PROBE))
    if blk is None:
        return None
    ds = d["datasets"]
    i = ds.index(src)
    v = [x for j, x in enumerate(blk[i]) if j != i and x is not None]
    return sum(v) / len(v) if v else None


def cell(d, feat, s, t):
    blk = idx(d).get((feat, PROBE))
    ds = d["datasets"]
    return blk[ds.index(s)][ds.index(t)]


def main():
    print("=" * 78)
    print("1. F-03's HEADLINE: train on the wording-matched slice, test on the other ten")
    print("   (10-cell off-diagonal mean, mean pooling, paper probe, seed 42)")
    print(f"\n   {'layer':6s} {'axes':>8s} {'raw':>8s} {'axes-raw':>10s} | "
          f"{'axes':>8s} {'raw':>8s} {'axes-raw':>10s} | {'margin':>8s}")
    print(f"   {'':6s} {'--- LEAKED ---':^28s} | {'--- GUARDED ---':^28s} | {'change':>8s}")
    for L in (20, 40, 44):
        a, b = load(ORIG, L), load(GUARD, L)
        if not a or not b:
            print(f"   L{L}: missing")
            continue
        aa, ar = offdiag(a, "axes", MATCHED), offdiag(a, "raw", MATCHED)
        ba, br = offdiag(b, "axes", MATCHED), offdiag(b, "raw", MATCHED)
        print(f"   L{L:<5d} {aa:8.4f} {ar:8.4f} {aa - ar:+10.4f} | "
              f"{ba:8.4f} {br:8.4f} {ba - br:+10.4f} | {(ba - br) - (aa - ar):+8.4f}")

    print("\n" + "=" * 78)
    print("2. THE TWO CELLS THE GUARD ACTUALLY TOUCHED (everything else is identical)")
    for lbl, (s, t) in (("matched -> knowledge-report", (MATCHED, FULL)),
                        ("knowledge-report -> matched", (FULL, MATCHED))):
        print(f"\n   {lbl}")
        print(f"   {'layer':6s} {'axes leaked':>12s} {'axes guarded':>13s} {'change':>9s}"
              f"   {'raw leaked':>11s} {'raw guarded':>12s}")
        for L in (20, 40, 44):
            a, b = load(ORIG, L), load(GUARD, L)
            if not a or not b:
                continue
            al, ag = cell(a, "axes", s, t), cell(b, "axes", s, t)
            rl, rg = cell(a, "raw", s, t), cell(b, "raw", s, t)
            print(f"   L{L:<5d} {al:12.4f} {ag:13.4f} {ag - al:+9.4f}"
                  f"   {rl:11.4f} {rg:12.4f}")

    print("\n" + "=" * 78)
    print("READING")
    print("  * The individual affected cells move a LOT (up to 0.14) -- the leak was real.")
    print("  * F-03's headline barely moves (<=0.008) because one changed cell is averaged")
    print("    with nine unchanged ones.")
    print("  * So the leak did NOT manufacture F-03's result: the conclusion survives, and")
    print("    the L20 inversion (-0.25) survives too.")
    print("  * NEW, and it is the interesting part: with the leak removed, transfer WITHIN")
    print("    the knowledge-report family collapses -- knowledge-report -> matched at L40")
    print("    falls from 0.5674 to 0.4262, i.e. below chance.")


if __name__ == "__main__":
    main()
