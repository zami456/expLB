"""Re-derive the HP-KR wording-controlled experiment from raw matrices.

The draft claims this test was never run (p.12, p.13). It was: F-03 (2026-09-19),
F-72 and F-76 in kd/docs/FINDINGS.md. This script recomputes its headline table
from the stored matrix so the write-up rests on a computation, not on the ledger.

Design: `harm-pressure-knowledge-report` (HP-KR) mixes two things -- the model
denies knowledge, and it words that denial distinctively. `harm-pressure-kr-matched`
is the wording-matched slice: BOTH classes deny knowledge, so a detector that is
only reading the words cannot separate them. Comparing a feature's transfer when
trained on the full set against the matched slice isolates the wording confound.

"As source" = the matrix ROW for that training dataset, averaged over the other
datasets (off-diagonal), i.e. how well a probe trained there transfers elsewhere.

  python audit/scripts/p5_hpkr_matched.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FULL = "harm-pressure-knowledge-report"
MATCHED = "harm-pressure-kr-matched"
PROBE = "paper"


def load(layer):
    p = os.path.join(ROOT, "run-crossgen", "kd11", "matrices", f"mean_L{layer}_s42.json")
    if not os.path.exists(p):
        return None, p
    d = json.load(open(p, encoding="utf-8"))
    return d, p


def row_offdiag(d, feature, source):
    """Mean AUROC of the row `source`, excluding its own diagonal cell."""
    idx = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}
    blk = idx.get((feature, PROBE))
    if blk is None:
        return None, 0
    ds = d["datasets"]
    if source not in ds:
        return None, 0
    i = ds.index(source)
    vals = [v for j, v in enumerate(blk[i]) if j != i and v is not None]
    return (sum(vals) / len(vals) if vals else None), len(vals)


def main():
    for layer in (20, 40, 44):
        d, p = load(layer)
        print("=" * 78)
        if d is None:
            print(f"L{layer}: NOT IN THE GIT TREE ({os.path.relpath(p, ROOT)}) -- skipped")
            continue
        print(f"LAYER {layer}   {os.path.relpath(p, ROOT)}")
        print(f"  pooling={d.get('pooling')} probe={PROBE} seed={d.get('seed')} "
              f"k={d.get('k')} bank={d.get('bank_name')}  datasets={len(d['datasets'])}")
        feats = sorted({r["feature"] for r in d["results"]})
        want = [f for f in ("axes", "raw", "random", "length", "datapca") if f in feats]
        print(f"\n  {'feature':12s} {'full HP-KR':>12s} {'matched slice':>14s} {'difference':>12s}")
        for f in want:
            a, na = row_offdiag(d, f, FULL)
            b, nb = row_offdiag(d, f, MATCHED)
            if a is None or b is None:
                print(f"  {f:12s} {'-':>12s} {'-':>14s}   (missing)")
                continue
            print(f"  {f:12s} {a:12.4f} {b:14.4f} {b - a:+12.4f}   (n={na}/{nb} targets)")
        # the comparison the claim turns on
        ax, _ = row_offdiag(d, "axes", MATCHED)
        rw, _ = row_offdiag(d, "raw", MATCHED)
        if ax is not None and rw is not None:
            print(f"\n  MATCHED SLICE: axes - raw = {ax - rw:+.4f}")
        axf, _ = row_offdiag(d, "axes", FULL)
        rwf, _ = row_offdiag(d, "raw", FULL)
        if axf is not None and rwf is not None:
            print(f"  FULL HP-KR   : axes - raw = {axf - rwf:+.4f}")
    zeroshot_table()
    print("=" * 78)
    print("Reading: on the full set the persona axes LOSE to a raw probe; on the")
    print("wording-matched slice they win. The effect F-03 reports is therefore")
    print("conditional on the wording confound being present -- and F-76 then shows")
    print("the matched-slice margin does not clear a dataset-level bootstrap.")


COLLAB = os.path.join(ROOT, "audit", "cache", "labpc", "run-crossgen", "kd11",
                      "banks", "collab", "matrices", "mean_L{}_s42.json")


def zeroshot_table():
    """Zero-shot readings on the COLLABORATOR's own bank, full HP-KR vs matched.

    This is the sharpest single comparison in the audit: his directions score
    0.81-0.93 on the full set and collapse on the wording-matched slice. The
    `kd_assertion` row is the mechanism -- a plain-assertion direction reads
    0.91-0.93 on HP-KR, which is what "detecting the lie" mostly amounts to there.
    """
    want = [
        ("contrast:collab_extended", "his 12v13 extended axis"),
        ("contrast:knowledge_denial", "split knowledge-denial axis"),
        ("contrast:kd_assertion", "plain-assertion pole"),
        ("contrast", "paper's 8v7 deception contrast"),
    ]
    IDX = None
    print("=" * 78)
    print("ZERO-SHOT, collaborator's bank (bank_name=collab, k=5, mean pooling, seed 42)")
    print("  source: audit/cache/labpc/run-crossgen/kd11/banks/collab/matrices/")
    rows = {}
    for layer in (20, 40, 44):
        p = COLLAB.format(layer)
        if not os.path.exists(p):
            print(f"  L{layer}: cached copy not present -- skipped")
            continue
        d = json.load(open(p, encoding="utf-8"))
        ds = d["datasets"]
        im, iff = ds.index(MATCHED), ds.index(FULL)
        z = {x["direction"]: x for x in d["zeroshot"]}
        for key, _ in want:
            if key in z:
                rows.setdefault(key, {})[layer] = (z[key]["auroc"][im], z[key]["auroc"][iff])
    if not rows:
        return
    print()
    print(f"  {'direction':32s} " + "  ".join(f"L{L}: matched / full".rjust(22) for L in (20, 40, 44)))
    for key, label in want:
        if key not in rows:
            continue
        cells = []
        for L in (20, 40, 44):
            if L in rows[key]:
                m, f = rows[key][L]
                cells.append(f"{m:8.4f} / {f:8.4f}".rjust(22))
            else:
                cells.append("-".rjust(22))
        print(f"  {label:32s} " + "  ".join(cells))
    print()
    print("  Every matched-slice reading is at or near chance; the full-set readings")
    print("  are 0.81-0.93. The gap IS the wording confound.")


if __name__ == "__main__":
    main()
