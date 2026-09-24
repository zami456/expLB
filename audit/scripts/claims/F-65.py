"""F-65 - a capitulation axis beats the register axis on `sycophancy_mt`: 0.745 zero-shot vs 0.374.

Recomputes, from the per-cell AUROC matrices:
  * the zero-shot table (cap contrast / cap PC1 at L20/40/44 vs the paper16 register axis);
  * the full trained table with `sycophancy_mt` as TARGET (off-diagonal column mean, paper probe);
  * the s23-vs-all judge-filter comparison for EVERY feature (the entry's '<= 0.004' claim);
  * the layer profile.

Usage: python F-65.py [<kd11 root>]
"""
import glob, json, os, re, statistics, sys

root = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/user4/expLB/run-crossgen/kd11"
TARGET = "sycophancy_mt"


def load(bank, pool, L):
    p = os.path.join(root, bank, "matrices", f"{pool}_L{L}_s42.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def col_mean(d, feat, probe="paper"):
    """off-diagonal mean of the TARGET column = trained elsewhere, tested on sycophancy_mt."""
    ds = d["datasets"]; j = ds.index(TARGET)
    for r in d["results"]:
        if r["feature"] == feat and r["probe"] == probe:
            v = [r["auroc"][i][j] for i in range(len(ds)) if i != j and r["auroc"][i][j] is not None]
            return statistics.fmean(v), len(v)
    return None, 0


print("=== ZERO-SHOT on", TARGET, "(mean pooling, seed 42) ===")
for bank in ("banks/cap_s23", "banks/cap_all", ""):
    for L in (20, 40, 44):
        d = load(bank, "mean", L)
        if d is None:
            print(f"  {bank or '(paper16 baseline)':<18} L{L}: MISSING"); continue
        j = d["datasets"].index(TARGET)
        bits = []
        for z in d.get("zeroshot", []):
            a = z["auroc"][j]
            if a is None:
                continue
            lo = (z.get("ci95_lo") or [None] * 11)[j]
            hi = (z.get("ci95_hi") or [None] * 11)[j]
            ci = f"[{lo:.3f},{hi:.3f}]" if lo is not None else ""
            bits.append(f"{z['direction']}={a:.4f}{ci}")
        print(f"  {bank or '(paper16 baseline)':<18} L{L}: bank={d.get('bank')} name={d.get('bank_name')}"
              f" k={d.get('k')} | " + "  ".join(bits))

print("\n=== TRAINED: off-diagonal mean with", TARGET, "as TARGET, probe=paper, L40, mean ===")
base = load("", "mean", 40)
cap = load("banks/cap_s23", "mean", 40)
rows = {}
for feat in sorted({r["feature"] for r in base["results"]}):
    m, n = col_mean(base, feat)
    rows[f"paper16:{feat}" if feat.startswith(("pc:", "axes")) else feat] = (m, n)
for feat in sorted({r["feature"] for r in cap["results"]}):
    m, n = col_mean(cap, feat)
    rows.setdefault(feat, (m, n))
    if feat.startswith("pc:"):
        rows[feat] = (m, n)
raw = rows["raw"][0]
for f, (m, n) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
    print(f"  {f:<24} {m:.4f}  vs_raw={m-raw:+.4f}  n={n}")

print("\n=== JUDGE FILTER: s23 vs all, EVERY feature, L40 mean, paper probe ===")
a, b = load("banks/cap_s23", "mean", 40), load("banks/cap_all", "mean", 40)
feats = sorted({r["feature"] for r in a["results"]} & {r["feature"] for r in b["results"]})
worst = (0.0, "")
for f in feats:
    x, _ = col_mean(a, f); y, _ = col_mean(b, f)
    flag = "  <-- exceeds 0.004" if abs(x - y) > 0.004 else ""
    if abs(x - y) > worst[0]:
        worst = (abs(x - y), f)
    print(f"  {f:<24} s23={x:.4f}  all={y:.4f}  |diff|={abs(x-y):.4f}{flag}")
ja = a["datasets"].index(TARGET)
for za, zb in zip(a.get("zeroshot", []), b.get("zeroshot", [])):
    if za["auroc"][ja] is None:
        continue
    print(f"  zero-shot {za['direction']:<22} s23={za['auroc'][ja]:.4f}  all={zb['auroc'][ja]:.4f}"
          f"  |diff|={abs(za['auroc'][ja]-zb['auroc'][ja]):.4f}")
print(f"  LARGEST trained |s23 - all| over all features: {worst[0]:.4f} on {worst[1]}")

print("\n=== cap_only penalty (the entry's caveat (c)), both filters, all three layers ===")
for L in (20, 40, 44):
    for tag, bank in (("s23", "banks/cap_s23"), ("all", "banks/cap_all")):
        d = load(bank, "mean", L)
        if d is None:
            continue
        c2, _ = col_mean(d, "pc:capitulation@2"); o2, _ = col_mean(d, "pc:cap_only@2")
        c1, _ = col_mean(d, "pc:capitulation"); o1, _ = col_mean(d, "pc:cap_only")
        print(f"  L{L:<3} {tag}: capitulation@2={c2:.4f} cap_only@2={o2:.4f} penalty={o2-c2:+.4f}"
              f" | capitulation={c1:.4f} cap_only={o1:.4f} penalty={o1-c1:+.4f}")

print("\n=== LAYER PROFILE, trained @2 (s23) and zero-shot contrast, every stored layer ===")
for pool in ("mean", "last"):
    for fp in sorted(glob.glob(os.path.join(root, "banks/cap_s23", "matrices", f"{pool}_L*_s42.json")),
                     key=lambda s: int(re.search(r"_L(\d+)_", s).group(1))):
        L = int(re.search(r"_L(\d+)_", fp).group(1))
        d = json.load(open(fp, encoding="utf-8"))
        bb = load("", pool, L)
        j = d["datasets"].index(TARGET)
        zs = {z["direction"]: z["auroc"][j] for z in d.get("zeroshot", [])}
        c2, _ = col_mean(d, "pc:capitulation@2"); o2, _ = col_mean(d, "pc:cap_only@2")
        rw, _ = col_mean(d, "raw"); dp2, _ = col_mean(d, "datapca@2"); rd2, _ = col_mean(d, "random@2")
        reg = "  "
        if bb:
            zb = {z["direction"]: z["auroc"][bb["datasets"].index(TARGET)] for z in bb.get("zeroshot", [])}
            reg = f" | register contrast={zb.get('contrast'):.4f} PC1={zb.get('PC1'):.4f}"
        print(f"  {pool} L{L:<3} zs contrast={zs.get('contrast'):.4f} PC1={zs.get('PC1'):.4f}"
              f" | cap@2={c2:.4f} cap_only@2={o2:.4f} raw={rw:.4f} datapca@2={dp2:.4f} random@2={rd2:.4f}{reg}")
