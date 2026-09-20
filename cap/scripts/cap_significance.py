"""Dataset-level significance for F-65 -- the project's stated bar, applied to the cap result.

27_crossgen.py ships a ROW-level bootstrap (`row_bootstrap_offdiag_diff`). That is not the bar:
1 of 384 paper16 comparisons and 6 of 4,032 kd ones cleared the DATASET-level test, and four of
those six were negative. Rows are plentiful and nearly free; datasets are the unit that actually
varies, and resampling them is what stopped this project quoting gaps that were noise.

The F-65 claim is about ONE CELL -- `sycophancy_mt` as target -- so the resampling unit is the
10 SOURCE datasets that each contribute one trained transfer AUROC to that column. Paired: every
feature is scored on the same resampled source list, so the difference has no between-sample
variance of its own.

Zero-shot has no sources (no training), so a dataset bootstrap is undefined for it; the matrix
already stores a row-level 95% CI per zero-shot cell, which is reported instead and labelled.
"""
import json, os, sys
import numpy as np

DS = "sycophancy_mt"
B = 10000
PAIRS = [("pc:capitulation@2", "raw"), ("pc:capitulation@2", "datapca@2"),
         ("pc:capitulation@2", "random@2"), ("pc:capitulation@2", "length"),
         ("pc:capitulation@2", "pc:cap_only@2"), ("pc:capitulation@2", "axes"),
         ("pc:capitulation", "raw"), ("pc:cap_only@2", "raw")]


def column(d, feat, probe="paper"):
    """The 10 off-diagonal transfer AUROCs in the DS column, one per source dataset."""
    ds = d["datasets"]; j = ds.index(DS)
    for r in d["results"]:
        if r["feature"] == feat and r["probe"] == probe:
            A = r["auroc"]
            return np.array([A[i][j] for i in range(len(ds)) if i != j], dtype=float), \
                   [ds[i] for i in range(len(ds)) if i != j]
    return None, None


def main(path):
    d = json.load(open(path, encoding="utf-8"))
    print(f"{path}\n  layer={d['layer']} pooling={d['pooling']}  B={B} paired dataset bootstrap"
          f" over the {len(d['datasets'])-1} SOURCE datasets in the '{DS}' column\n")

    cols, names = {}, None
    for a, b in PAIRS:
        for f in (a, b):
            if f not in cols:
                cols[f], nm = column(d, f)
                if nm:
                    names = nm
    rng = np.random.default_rng(42)
    n = len(names)
    idx = rng.integers(0, n, size=(B, n))          # ONE resampling shared by every pair -> paired

    print(f"  {'comparison':<42}{'diff':>9}{'95% CI':>20}{'p(diff<=0)':>12}")
    for a, b in PAIRS:
        ca, cb = cols.get(a), cols.get(b)
        if ca is None or cb is None:
            print(f"  {a+' - '+b:<42}  MISSING"); continue
        dif = ca - cb
        bs = dif[idx].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = float((bs <= 0).mean())
        star = "  SIGNIFICANT" if lo > 0 else ("  (sig. negative)" if hi < 0 else "")
        print(f"  {a+' - '+b:<42}{dif.mean():>+9.4f}   [{lo:+.4f}, {hi:+.4f}]{p:>12.4f}{star}")

    print(f"\n  per-source transfer AUROC into {DS} (why the CI is wide):")
    print(f"    {'source':<34}{'cap@2':>8}{'raw':>8}{'diff':>9}")
    ca, cb = cols["pc:capitulation@2"], cols["raw"]
    order = np.argsort(-(ca - cb))
    for i in order:
        print(f"    {names[i]:<34}{ca[i]:>8.3f}{cb[i]:>8.3f}{ca[i]-cb[i]:>+9.3f}")
    wins = int((ca > cb).sum())
    print(f"    cap beats raw on {wins} of {n} source datasets")

    zs = d.get("zeroshot") or []
    if zs:
        print(f"\n  zero-shot on {DS} (ROW-level CI from the matrix; a dataset bootstrap is")
        print( "  undefined here -- zero-shot trains on nothing, so there are no sources):")
        j = d["datasets"].index(DS)
        for z in zs:
            v = z.get("auroc"); lo = z.get("lo"); hi = z.get("hi")
            def pick(x):
                if x is None: return None
                return x[j] if isinstance(x, list) else (x.get(DS) if isinstance(x, dict) else x)
            a, l, h = pick(v), pick(lo), pick(hi)
            ci = f"   [{l:.3f}, {h:.3f}]" if (l is not None and h is not None) else ""
            print(f"    {z['direction']:<20}{a:.4f}{ci}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if os.path.exists(p):
            main(p); print("\n" + "=" * 78 + "\n")
        else:
            print(f"{p} MISSING")
