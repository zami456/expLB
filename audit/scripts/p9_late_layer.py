"""The L76 question: why do many arms rise sharply near the top of the stack?

The collaborator's sweep (origin/paper) shows most arms dropping through the middle
layers and then JUMPING at L76 before falling at L79 -- and on HP-KR the published
8v7 axis goes 0.4344 (L72) -> 0.7301 (L76) -> 0.7709 (L79), from inverted to working.

This asks our OWN 80-layer store the same question, because if the same shape appears
in a pipeline he never touched, it is a property of the model, not of his run.

  python audit/scripts/p9_late_layer.py
"""
import csv
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LS = os.path.join(ROOT, "audit", "phase2", "layer_sensitivity.csv")


def main():
    rows = list(csv.DictReader(open(LS, encoding="utf-8")))
    # passA is our 80-layer store
    sel = [r for r in rows
           if r["store"] == "passA" and r["pooling"] == "mean"
           and r["seed"] == "42" and r["probe"] == "paper"]
    by = defaultdict(dict)
    for r in sel:
        try:
            by[r["feature"]][int(r["layer"])] = float(r["offdiag_mean"])
        except (ValueError, TypeError):
            pass
    if not by:
        print("no passA/mean/seed42/paper rows found")
        return

    layers = sorted(set().union(*[set(v) for v in by.values()]))
    top = [L for L in layers if L >= 60]
    print("=" * 92)
    print("OUR OWN 80-LAYER STORE (passA, mean pooling, paper probe, seed 42)")
    print("does the same late-layer rise appear in a pipeline he never touched?")
    print("=" * 92)
    print(f"  layers stored: {len(layers)}  (max {max(layers)})")
    print(f"\n  {'feature':16s} " + " ".join(f"L{L}".rjust(7) for L in top))
    interesting = ["raw", "axes", "pc:deception", "pc:unified", "datapca", "random", "length"]
    for f in interesting:
        if f not in by:
            continue
        cells = " ".join((f"{by[f][L]:.4f}".rjust(7) if L in by[f] else "      -") for L in top)
        print(f"  {f:16s} {cells}")

    # where does each feature peak, and is it at the extremes?
    print("\n" + "=" * 92)
    print("PEAK LAYER PER FEATURE, and how the top of the stack compares to the middle")
    print("=" * 92)
    print(f"  {'feature':16s} {'peak@L':>7s} {'peak':>8s} {'mid(30-50)':>11s} {'top(>=72)':>10s} {'top-mid':>9s}")
    for f in interesting:
        if f not in by:
            continue
        d = by[f]
        pk = max(d, key=lambda L: d[L])
        mid = [v for L, v in d.items() if 30 <= L <= 50]
        hi = [v for L, v in d.items() if L >= 72]
        if not mid or not hi:
            continue
        m, h = sum(mid) / len(mid), sum(hi) / len(hi)
        print(f"  {f:16s} {pk:7d} {d[pk]:8.4f} {m:11.4f} {h:10.4f} {h - m:+9.4f}")

    # ---- the decisive control ---------------------------------------
    print("\n" + "=" * 92)
    print("THE DECISIVE CONTROL: does the late-layer gain survive a RANDOM direction?")
    print("=" * 92)
    print(f"  {'layer':>6s} {'raw':>8s} {'axes':>8s} {'random':>8s} "
          f"{'axes-raw':>9s} {'axes-random':>12s}")
    for L in (40, 60, 70, 72, 74, 76, 77, 78, 79):
        r, a, n = by["raw"].get(L), by["axes"].get(L), by["random"].get(L)
        if None in (r, a, n):
            continue
        print(f"  {L:6d} {r:8.4f} {a:8.4f} {n:8.4f} {a - r:+9.4f} {a - n:+12.4f}")

    print("\n" + "=" * 92)
    print("READING")
    print("=" * 92)
    print("  If `raw` ALSO rises at the top, the late-layer effect is a property of the")
    print("  representation, not of persona directions -- and the honest answer to a reviewer")
    print("  is that the top of the stack is a different regime, not that the method works there.")
    print("  MEASURED ANSWER (2026-09-24):")
    print("  * `raw` does NOT rise -- it DECLINES at the top (0.7834 @L72 -> 0.7426 @L79).")
    print("  * `axes` recovers from a ~L70 trough (0.62) to 0.7492 @L79 -- but that is still")
    print("    BELOW its own mid-stack level (0.8145 @L30-50). It is a recovery, not a peak.")
    print("  * `random` ALSO gains, +0.127 from L72 to L79 (0.5133 -> 0.6405). A random")
    print("    direction cannot learn anything, so this is a geometry change that lifts ANY")
    print("    low-dimensional projection.")
    print("  * Against that control the persona axes are WORSE at the top than at L40:")
    print("    axes-random = +0.2296 @L40 vs +0.1087 @L79.")
    print("  => The late-layer rise is raw DEGRADING plus low-dim projections being flattered")
    print("     by late-layer anisotropy. It is NOT evidence the method works at depth.")


if __name__ == "__main__":
    main()
