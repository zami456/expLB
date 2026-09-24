"""Which (persona bank x model size) cells were actually run?

The question: our claim is "16-persona coordinates do not survive at 70B, but our
extended bank does better". That needs a 2x2 -- {16 personas, our extended bank} x
{3B, 8B, 70B}. This script reports which cells exist, and the numbers for the ones
that do, so the claim can be stated at exactly the strength the data supports.

Run on the lab PC (the scale stores are lab-only); prints a design table plus the
three-rung comparison for whichever bank has all three.

  python p7_scale_bank_design.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"C:/Users/user4/expLB/run-crossgen")
POOL = "mean"
PROBE = "paper"


def offdiag(d, feature, probe=PROBE):
    idx = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}
    blk = idx.get((feature, probe))
    if blk is None:
        return None
    n = len(d["datasets"])
    v = [blk[i][j] for i in range(n) for j in range(n)
         if i != j and blk[i][j] is not None]
    return sum(v) / len(v) if v else None


def scan(store):
    """-> {layer: {feature: offdiag}} for POOL/PROBE, plus config."""
    m = ROOT / store / "matrices"
    if not m.is_dir():
        return None, None
    out, cfg = {}, None
    for f in sorted(m.glob(f"{POOL}_L*_s42.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if cfg is None:
            cfg = {"bank": d.get("bank"), "bank_name": d.get("bank_name"),
                   "k": d.get("k"), "datasets": d.get("datasets")}
        feats = sorted({r["feature"] for r in d["results"]})
        out[d["layer"]] = {ft: offdiag(d, ft) for ft in feats}
    return out, cfg


def main():
    stores = sorted(p.name for p in ROOT.iterdir()
                    if p.is_dir() and p.name.startswith("scale"))
    print("=" * 78)
    print("WHICH (bank x size) CELLS EXIST")
    print("=" * 78)
    data = {}
    for s in stores:
        layers, cfg = scan(s)
        if not cfg:
            print(f"  {s:26s} no {POOL} matrices")
            continue
        data[s] = (layers, cfg)
        print(f"  {s:26s} bank_name={cfg['bank_name']:18s} k={cfg['k']}  "
              f"layers={len(layers):2d}  datasets={len(cfg['datasets'])}")
        print(f"  {'':26s} {','.join(cfg['datasets'])}")

    # the extended / kd banks live outside scale_*, at 70B only
    print("\n  Extended-bank stores (70B only, for contrast):")
    for p in sorted((ROOT / "kd11" / "banks").glob("*")):
        if p.is_dir():
            print(f"    kd11/banks/{p.name}")

    print("\n" + "=" * 78)
    print(f"THREE-RUNG COMPARISON, 16-persona bank, {POOL} pooling, {PROBE} probe, seed 42")
    print("off-diagonal mean over the 5 shared datasets; layer-averaged over each rung's own layers")
    print("=" * 78)
    rungs = [("scale_3b", "3B"), ("scale_8b", "8B"),
             ("scale_70b_paper16fp32", "70B"), ("scale_70b", "70B (old s3 bank)")]
    print(f"  {'rung':20s} {'n_layers':>8s} {'raw':>8s} {'best persona':>13s} "
          f"{'vs raw':>8s} {'datapca':>8s} {'vs dpca':>8s}")
    for store, label in rungs:
        if store not in data:
            continue
        layers, cfg = data[store]
        if not layers:
            continue
        pers = [f for f in next(iter(layers.values())) if f.startswith("pc:") or f == "axes"]
        acc = defaultdict(list)
        for L, feats in layers.items():
            if feats.get("raw") is None:
                continue
            acc["raw"].append(feats["raw"])
            best = max((feats[f] for f in pers if feats.get(f) is not None), default=None)
            if best is not None:
                acc["best"].append(best)
            if feats.get("datapca") is not None:
                acc["dpca"].append(feats["datapca"])
        if not acc["raw"]:
            continue
        mean = {k: sum(v) / len(v) for k, v in acc.items() if v}
        raw, best = mean.get("raw"), mean.get("best")
        dp = mean.get("dpca")
        print(f"  {label:20s} {len(acc['raw']):8d} {raw:8.4f} {best:13.4f} "
              f"{best - raw:+8.4f} " + (f"{dp:8.4f} {best - dp:+8.4f}" if dp else f"{'-':>8s} {'-':>8s}"))

    print("\n" + "=" * 78)
    print("THE GAP")
    print("=" * 78)
    print("  16-persona bank : 3B yes, 8B yes, 70B yes  -> a clean scale comparison")
    print("  extended bank   : 3B  NO, 8B  NO, 70B yes  -> NO scale comparison possible")
    print("  => 'our bank rescues the method AT SCALE' is not separable from")
    print("     'our bank is better at any scale'. Only the first rung set supports")
    print("     a scale claim; the extended-bank gain is a 70B-only result.")


if __name__ == "__main__":
    main()
