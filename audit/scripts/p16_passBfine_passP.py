"""Phase 6 / 6D-2 -- read `passB_fine` and `passP`, the two crossgen stores that reached the
audit but never the ledger (COVERAGE_GAP.md section 3a).

Runs ON THE LAB PC, read-only over C:\\Users\\user4\\expLB\\run-crossgen\\. Prints only.

For each store and bank: what it is, and the headline persona-vs-control comparison at the
project's standard reading (probe `paper`, mean pooling, seed 42), reported against ALL the
controls -- raw, datapca, random, length -- because a persona number without its baselines
means nothing.
"""
import json
import os

R = r"C:\Users\user4\expLB\run-crossgen"
PROBE, POOL = "paper", "mean"
CTRL = ["raw", "datapca", "random", "length"]


def summarise(tag, path):
    f = os.path.join(path, "summary.json")
    if not os.path.exists(f):
        print(f"  {tag}: no summary.json")
        return
    j = json.load(open(f))
    rs = j["records"]
    ds = j.get("datasets", [])
    b9 = j.get("block9", [])
    layers = sorted({x["layer"] for x in rs})
    feats = sorted({x["feature"] for x in rs})
    bank = j.get("bank_name") or j.get("bank") or "?"
    print(f"\n  --- {tag} ---")
    print(f"      bank={bank}  datasets={len(ds)}  block9={len(b9)}  layers={layers}")
    print(f"      features={feats}")
    for L in layers:
        sel = {x["feature"]: x for x in rs
               if x["layer"] == L and x["probe"] == PROBE and x["pooling"] == POOL}
        if not sel:
            continue
        key = "offdiag_mean_10" if "offdiag_mean_10" in next(iter(sel.values())) else "offdiag_mean"
        line = [f"      L{L:<3d}"]
        for f_ in ["raw", "axes", "pc:deception", "pc:unified", "datapca", "random", "length"]:
            if f_ in sel and sel[f_].get(key) is not None:
                line.append(f"{f_}={sel[f_][key]:.4f}")
        print("  ".join(line))
        if "axes" in sel and sel["axes"].get(key) is not None:
            a = sel["axes"][key]
            deltas = [f"{c}: {a - sel[c][key]:+.4f}" for c in CTRL
                      if c in sel and sel[c].get(key) is not None]
            print(f"           axes minus  " + "   ".join(deltas))


print("=" * 78)
print("6D-2 -- passB_fine and passP")
print("=" * 78)

for store in ("passB_fine", "passP"):
    p = os.path.join(R, store)
    print(f"\n########## {store}")
    if not os.path.isdir(p):
        print("  MISSING")
        continue
    nm = len([x for x in os.listdir(os.path.join(p, "matrices"))
              if x.endswith(".json")]) if os.path.isdir(os.path.join(p, "matrices")) else 0
    print(f"  root matrices: {nm}")
    summarise(f"{store} (root)", p)
    bd = os.path.join(p, "banks")
    if os.path.isdir(bd):
        for b in sorted(os.listdir(bd)):
            bp = os.path.join(bd, b)
            if os.path.isdir(bp):
                n = len([x for x in os.listdir(os.path.join(bp, "matrices"))
                         if x.endswith(".json")]) if os.path.isdir(os.path.join(bp, "matrices")) else 0
                summarise(f"{store}/banks/{b}  (matrices={n})", bp)

# what makes passB_fine distinctive: the layer grid
print("\n\n########## layer grids")
for store in ("passB_fine", "passP"):
    p = os.path.join(R, store)
    for sub in [p] + [os.path.join(p, "banks", b)
                      for b in (os.listdir(os.path.join(p, "banks"))
                                if os.path.isdir(os.path.join(p, "banks")) else [])]:
        md = os.path.join(sub, "matrices")
        if os.path.isdir(md):
            ls = sorted({int(x.split("_L")[1].split("_")[0])
                         for x in os.listdir(md) if "_L" in x})
            print(f"  {os.path.relpath(sub, R):44s} layers {ls}")
