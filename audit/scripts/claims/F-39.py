"""Audit Phase 2 / W5A-ledger - F-39 recomputation.

F-39 (kd/docs/FINDINGS.md:1012) claims, on "the paper's own five deception datasets",
70B, L40, off-diagonal (20 pairs), mean pooling, paper probe:
    raw = 0.8090
    pc:unified@2 = 0.8351  (+0.0262)   vs datapca@2 = -0.1120  => +0.138
    pc:unified (k=5) = 0.7917 (-0.0173) vs datapca   = -0.0214  => +0.004

This script recomputes those numbers directly from the committed matrix, with its
own off-diagonal averaging code (no import of kd/scripts/paper_comparison.py),
and additionally sweeps which 5-dataset subsets reproduce raw=0.8090.

Read-only. Run from the repo root.
"""
import json
import itertools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
MAT = os.path.join(ROOT, "run-crossgen", "kd11", "matrices", "mean_L40_s42.json")


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def offdiag_mean(d, feature, probe, subset):
    ds = d["datasets"]
    idx = [ds.index(x) for x in subset]
    rec = [r for r in d["results"] if r["feature"] == feature and r["probe"] == probe]
    assert len(rec) == 1, (feature, probe, len(rec))
    a = rec[0]["auroc"]
    vals = []
    for i in idx:
        for j in idx:
            if i == j:
                continue
            vals.append(a[i][j])
    return sum(vals) / len(vals), len(vals)


def main():
    d = load(MAT)
    print("matrix:", MAT)
    print("  pooling=%s layer=%s seed=%s k=%s frac_train=%s" %
          (d["pooling"], d["layer"], d["seed"], d["k"], d["frac_train"]))
    print("  bank=%s bank_name=%s" % (d["bank"], d["bank_name"]))
    print("  datasets(%d)=%s" % (len(d["datasets"]), d["datasets"]))
    print()

    paper5 = ["instructed-deception", "convincing-game", "ai_liar",
              "roleplaying", "mask"]
    print("=== F-39's set: the paper's five WITH mask ===")
    print("subset:", paper5)
    base, n = offdiag_mean(d, "raw", "paper", paper5)
    print("  pairs = %d" % n)
    print("  raw                 = %.4f   (F-39: 0.8090)" % base)
    for feat in ["pc:unified@2", "datapca@2", "pc:unified", "datapca"]:
        v, _ = offdiag_mean(d, feat, "paper", paper5)
        print("  %-18s = %.4f   delta vs raw = %+.4f" % (feat, v, v - base))
    u2, _ = offdiag_mean(d, "pc:unified@2", "paper", paper5)
    p2, _ = offdiag_mean(d, "datapca@2", "paper", paper5)
    u5, _ = offdiag_mean(d, "pc:unified", "paper", paper5)
    p5, _ = offdiag_mean(d, "datapca", "paper", paper5)
    print("  axis@2 - datapca@2  = %+.4f   (F-39: +0.138)" % (u2 - p2))
    print("  axis(k5) - datapca  = %+.4f   (F-39: +0.004)" % (u5 - p5))
    print()

    print("=== the variant three other entries (F-45/F-48/F-54) name: swap mask -> harm-pressure-choice ===")
    alt5 = ["instructed-deception", "convincing-game", "ai_liar",
            "roleplaying", "harm-pressure-choice"]
    base2, n2 = offdiag_mean(d, "raw", "paper", alt5)
    print("  pairs = %d ; raw = %.4f" % (n2, base2))
    for feat in ["pc:unified@2", "datapca@2", "pc:unified", "datapca"]:
        v, _ = offdiag_mean(d, feat, "paper", alt5)
        print("  %-18s = %.4f   delta vs raw = %+.4f" % (feat, v, v - base2))
    print()

    print("=== which 5-subsets of the 11 give raw = 0.8090 (+-0.0005), any probe ===")
    ds = d["datasets"]
    hits = 0
    for probe in ["paper", "lr_reg10", "mms"]:
        for sub in itertools.combinations(ds, 5):
            v, _ = offdiag_mean(d, "raw", probe, list(sub))
            if abs(v - 0.8090) < 0.0005:
                hits += 1
                print("  probe=%-8s raw=%.4f  %s" % (probe, v, list(sub)))
    print("  total matches:", hits)
    print()

    print("=== robustness: same 5 datasets, other layers / poolings (mean unless noted) ===")
    for pooling in ["mean", "last"]:
        for layer in [20, 32, 40, 44]:
            p = os.path.join(ROOT, "run-crossgen", "kd11", "matrices",
                             "%s_L%d_s42.json" % (pooling, layer))
            if not os.path.exists(p):
                print("  %s L%d : MISSING" % (pooling, layer))
                continue
            dd = load(p)
            b, _ = offdiag_mean(dd, "raw", "paper", paper5)
            a2, _ = offdiag_mean(dd, "pc:unified@2", "paper", paper5)
            c2, _ = offdiag_mean(dd, "datapca@2", "paper", paper5)
            print("  %-4s L%-3d raw=%.4f  axis@2-raw=%+.4f  axis@2-dPCA@2=%+.4f"
                  % (pooling, layer, b, a2 - b, a2 - c2))


if __name__ == "__main__":
    main()
