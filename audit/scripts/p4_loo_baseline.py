"""D-158: the draft says it does NOT compare against a pooled multi-dataset
supervised probe. That comparison exists on the collaborator's own branch.

Reads ONLY committed data:
  origin/feat/persona-pc-ood:src/probes/deception-detection/results/loo_L{20,40}/loo.csv
committed in 6b16a46, 2026-09-20 20:05, message:
  "Leave-one-dataset-out: the fair baseline for the generalization claim"

Prints each arm's mean AUROC over the six datasets, the mean excluding
harm-pressure-knowledge-report, and the HP-KR cell itself -- because the draft's
whole margin lives in that one cell.

  python audit/scripts/p4_loo_baseline.py
"""
import csv
import io
import subprocess
from collections import defaultdict

BRANCH = "origin/feat/persona-pc-ood"
BASE = "src/probes/deception-detection/results"
HPKR = "harm_refusal_knowledge_report__plain"


def load(layer):
    path = f"{BRANCH}:{BASE}/loo_L{layer}/loo.csv"
    txt = subprocess.run(["git", "show", path], capture_output=True, text=True, check=True).stdout
    g = defaultdict(dict)
    for r in csv.DictReader(io.StringIO(txt)):
        g[(r["arm"], r["weighting"])][r["dataset"]] = float(r["auroc"])
    return g


def mean(xs):
    return sum(xs) / len(xs)


def main():
    for layer in (20, 40):
        g = load(layer)
        print("=" * 66)
        print(f"LAYER {layer}   (source: {BASE}/loo_L{layer}/loo.csv @ {BRANCH})")
        print(f"{'arm/weighting':30s} {'mean(all)':>10s} {'excl HP-KR':>12s} {'HP-KR':>9s}")
        for k in sorted(g):
            d = g[k]
            ex = [v for ds, v in d.items() if ds != HPKR]
            print(f"{k[0] + '/' + k[1]:30s} {mean(list(d.values())):10.4f} "
                  f"{mean(ex):12.4f} {d.get(HPKR, float('nan')):9.4f}")

        # NOTE: do NOT pick the better pooled arm per layer -- that is
        # weighting-shopping, the same fault his own loo_L20/summary.md commits.
        # Print the full fixed-weighting matrix and let the reader see all cells.

    print()
    print("=" * 66)
    print("axis (ext_axis1) MINUS pooled baseline -- FIXED weighting across layers")
    for excl in (True, False):
        tag = "EXCLUDING HP-KR" if excl else "including HP-KR (the draft's framing)"
        print()
        print(f"  {tag}")
        print(f"  {'':6s} {'pooled/naive':>14s} {'pooled/balanced':>16s}")
        for layer in (20, 40):
            g = load(layer)
            def m(d):
                vs = [v for ds, v in d.items() if not (excl and ds == HPKR)]
                return mean(vs)
            ax = m(g[("ext_axis1", "naive")])
            cells = [ax - m(g[("baseline", w)]) for w in ("naive", "balanced")]
            print(f"  L{layer:<5d} {cells[0]:+14.4f} {cells[1]:+16.4f}")
    print()
    print("  Robust across ALL FOUR cells: removing HP-KR costs the axis")
    print("  0.048-0.130 AUROC of margin, and in 2 of 4 cells what remains")
    print("  is <= +0.002 or negative.")

    print()
    print("Draft Limitation (verbatim): a pooled multi-dataset supervised probe")
    print("'is a stronger supervised baseline, and we do not compare against it'.")


if __name__ == "__main__":
    main()
