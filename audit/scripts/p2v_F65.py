"""F-65: cap bank zero-shot on sycophancy_mt at L20/40/44 (s23 and all), trained off-diagonal
AUROC with sycophancy_mt as TARGET, and the paper16 register-axis zero-shot for comparison."""
import json, os, statistics, sys
root = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/user4/expLB/run-crossgen/kd11"
TARGET = "sycophancy_mt"

def load(p):
    return json.load(open(p, encoding="utf-8"))

print("=== ZERO-SHOT on", TARGET, "===")
for bank in ("banks/cap_s23", "banks/cap_all", ""):
    for L in (20, 40, 44):
        p = os.path.join(root, bank, "matrices", f"mean_L{L}_s42.json")
        if not os.path.exists(p):
            print(f"  {bank or '(baseline)':<16} L{L}: MISSING {p}"); continue
        d = load(p)
        ds = d["datasets"]
        if TARGET not in ds:
            print(f"  {bank or '(baseline)':<16} L{L}: target not in datasets"); continue
        i = ds.index(TARGET)
        parts = [f"{z['direction']}={z['auroc'][i]:.4f}" for z in d.get("zeroshot", []) if z["auroc"][i] is not None]
        print(f"  {bank or '(baseline)':<16} L{L}: bank={d.get('bank')} name={d.get('bank_name')} k={d.get('k')} | " + "  ".join(parts))

print("\n=== TRAINED: off-diagonal mean with", TARGET, "as TARGET (column), probe=paper, L40 ===")
base = load(os.path.join(root, "matrices", "mean_L40_s42.json"))
bds = base["datasets"]; bi = bds.index(TARGET)
bidx = {(r["feature"], r["probe"]): r["auroc"] for r in base["results"]}
rows = {}
for feat, pr in sorted(bidx):
    if pr != "paper": continue
    col = [bidx[(feat, pr)][j][bi] for j in range(len(bds)) if j != bi and bidx[(feat, pr)][j][bi] is not None]
    rows[feat] = (statistics.fmean(col), len(col))
for bank in ("banks/cap_s23", "banks/cap_all"):
    p = os.path.join(root, bank, "matrices", "mean_L40_s42.json")
    if not os.path.exists(p): continue
    d = load(p); ds = d["datasets"]; i = ds.index(TARGET)
    idx = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}
    for feat, pr in sorted(idx):
        if pr != "paper": continue
        col = [idx[(feat, pr)][j][i] for j in range(len(ds)) if j != i and idx[(feat, pr)][j][i] is not None]
        rows[bank.split("/")[-1] + ":" + feat] = (statistics.fmean(col), len(col))
raw = rows.get("raw", (float("nan"), 0))[0]
for f, (m, n) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
    print(f"  {f:<32} {m:.4f}  vs_raw={m-raw:+.4f}  n={n}")

print("\n=== s23 vs all, L40, every feature (trained target column) ===")
a = load(os.path.join(root, "banks/cap_s23/matrices/mean_L40_s42.json"))
b = load(os.path.join(root, "banks/cap_all/matrices/mean_L40_s42.json"))
for d, tag in ((a, "s23"), (b, "all")):
    ds = d["datasets"]; i = ds.index(TARGET)
    idx = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}
    out = {}
    for feat, pr in sorted(idx):
        if pr != "paper": continue
        col = [idx[(feat, pr)][j][i] for j in range(len(ds)) if j != i and idx[(feat, pr)][j][i] is not None]
        out[feat] = statistics.fmean(col)
    print(f"  {tag}: " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(out.items())))
