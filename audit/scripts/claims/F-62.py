"""Audit Phase 2 / W5A-ledger - F-62 recomputation.

F-62 (kd/docs/FINDINGS.md:1948) claims: s3 vs s23 judge filtering makes no measurable
downstream difference. L40, mean pooling, `paper` probe, 110 off-diagonal pairs,
raw off-diagonal baseline 0.6623:

    feature              kd (s3)   kd_s23 (s2+s3)   |difference|
    pc:kd_assertion      +0.1496   +0.1500          0.0004
    pc:knowledge_denial  +0.0869   +0.0838          0.0031
    pc:unified           +0.0757   +0.0759          0.0002
    axes                 -0.0272   -0.0319          0.0047
    "largest gap 0.005"

Recomputed here with independently written off-diagonal averaging.
Also: the robustness the entry does not report (other layers, other pooling, other probes)
and the count of figures the entry claims exist.

Read-only. Run from the repo root.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
BASE = os.path.join(ROOT, "run-crossgen", "kd11")


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def offdiag(d, feature, probe, subset=None):
    ds = d["datasets"]
    idx = list(range(len(ds))) if subset is None else [ds.index(x) for x in subset]
    rec = [r for r in d["results"] if r["feature"] == feature and r["probe"] == probe]
    if not rec:
        return None, 0
    a = rec[0]["auroc"]
    v = [a[i][j] for i in idx for j in idx if i != j]
    return sum(v) / len(v), len(v)


def report(layer, pooling, probe="paper"):
    fn = "%s_L%d_s42.json" % (pooling, layer)
    root = os.path.join(BASE, "matrices", fn)
    kd = os.path.join(BASE, "banks", "kd", "matrices", fn)
    s23 = os.path.join(BASE, "banks", "kd_s23", "matrices", fn)
    for p in (root, kd, s23):
        if not os.path.exists(p):
            print("  MISSING:", p)
            return
    R, K, S = load(root), load(kd), load(s23)
    base, n = offdiag(R, "raw", probe)
    print("  probe=%s pooling=%s L%d  raw off-diag (%d pairs) = %.4f" %
          (probe, pooling, layer, n, base))
    print("    root bank=%s (%s) k=%s | kd bank=%s (%s) | s23 bank=%s (%s)" %
          (R.get("bank"), R.get("bank_name"), R.get("k"),
           K.get("bank"), K.get("bank_name"), S.get("bank"), S.get("bank_name")))
    feats = ["pc:kd_assertion", "pc:knowledge_denial", "pc:unified", "axes"]
    worst = 0.0
    for f in feats:
        a, _ = offdiag(K, f, probe)
        b, _ = offdiag(S, f, probe)
        if a is None or b is None:
            print("    %-20s MISSING (kd=%s s23=%s)" % (f, a, b))
            continue
        da, db = a - base, b - base
        print("    %-20s kd %+0.4f   s23 %+0.4f   |diff| %.4f" % (f, da, db, abs(da - db)))
        worst = max(worst, abs(da - db))
    print("    largest gap = %.4f" % worst)


def main():
    print("=== F-62 headline cell: L40, mean pooling, paper probe ===")
    report(40, "mean", "paper")
    print()
    print("=== features present in each bank file (L40 mean) ===")
    for name, p in [("kd", os.path.join(BASE, "banks", "kd", "matrices", "mean_L40_s42.json")),
                    ("kd_s23", os.path.join(BASE, "banks", "kd_s23", "matrices", "mean_L40_s42.json"))]:
        d = load(p)
        print(" ", name, sorted(set(r["feature"] for r in d["results"])))
    print()
    print("=== robustness: other layers / poolings / probes ===")
    for pooling in ["mean", "last"]:
        for layer in [20, 40, 44]:
            report(layer, pooling, "paper")
    print()
    for probe in ["mms", "lr_reg10"]:
        report(40, "mean", probe)
    print()
    print("=== the figures F-62 says exist ===")
    import glob
    for pat in ["figures/crossgen/kd11/blocks/kd_L40_mean_*.png",
                "figures/crossgen/kd11/matched/kd_L40_mean_*.png",
                "figures/crossgen/kd11/blocks/kd_s23_L40_mean_*.png",
                "figures/crossgen/kd11/matched/kd_s23_L40_mean_*.png",
                "figures/crossgen/kd11/passA/*/kd_s23*.png"]:
        g = sorted(glob.glob(os.path.join(ROOT, pat)))
        print("  %-55s -> %d" % (pat, len(g)))
        for x in g:
            print("       ", os.path.relpath(x, ROOT))


if __name__ == "__main__":
    main()
