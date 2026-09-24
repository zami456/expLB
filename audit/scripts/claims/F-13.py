"""F-13 - 'Persona axes encode register': HP-KR and sycophancy_mt as target/source at L40,
random comparators, zero-shot on sycophancy_mt, the leave-sycophancy_mt-out off-diagonal lift,
and pc:unified 0.672 vs raw 0.662.

Usage: python F-13.py [<kd11 root>]       (default = repo copy)
Runs identically on the laptop/git copy and on the lab copy.
"""
import json, os, statistics, sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "run-crossgen", "kd11")
p = os.path.join(root, "matrices", "mean_L40_s42.json")
d = json.load(open(p, encoding="utf-8"))
ds = d["datasets"]
print("file :", os.path.abspath(p))
print("bank :", d.get("bank"), "| name:", d.get("bank_name"), "| k:", d.get("k"),
      "| pooling:", d["pooling"], "| layer:", d["layer"], "| seed:", d["seed"])
print("n datasets:", len(ds))

idx = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}

def col_mean(feat, probe, target):          # target = the COLUMN (tested on), off-diagonal
    j = ds.index(target)
    v = [idx[(feat, probe)][i][j] for i in range(len(ds)) if i != j and idx[(feat, probe)][i][j] is not None]
    return statistics.fmean(v), len(v)

def row_mean(feat, probe, source):          # source = the ROW (trained on), off-diagonal
    i = ds.index(source)
    v = [idx[(feat, probe)][i][j] for j in range(len(ds)) if j != i and idx[(feat, probe)][i][j] is not None]
    return statistics.fmean(v), len(v)

print("\n=== F-13 table: 'as target' (column mean) / 'as source' (row mean), probe=paper ===")
for feat in ("axes", "pc:deception", "pc:unified", "random", "random@2", "raw", "datapca"):
    for dset in ("harm-pressure-knowledge-report", "sycophancy_mt"):
        t, nt = col_mean(feat, "paper", dset)
        s, ns = row_mean(feat, "paper", dset)
        print(f"  {feat:<14} {dset:<32} as_target={t:.4f} (n={nt})  as_source={s:.4f} (n={ns})")

print("\n=== zero-shot on sycophancy_mt (paper16 bank) ===")
j = ds.index("sycophancy_mt")
for z in d.get("zeroshot", []):
    a = z["auroc"][j]
    print(f"  {z['direction']:<24} {a if a is None else round(a,4)}")

print("\n=== off-diagonal mean over all 11 vs the 10 without sycophancy_mt (probe=paper) ===")
def offdiag(feat, probe, keep):
    K = [ds.index(x) for x in keep]
    v = [idx[(feat, probe)][i][j] for i in K for j in K if i != j and idx[(feat, probe)][i][j] is not None]
    return statistics.fmean(v), len(v)
all11 = list(ds)
no_syc = [x for x in ds if x != "sycophancy_mt"]
for feat in ("raw", "axes", "pc:deception", "pc:unified", "pc:unified@2", "datapca", "random"):
    a, na = offdiag(feat, "paper", all11)
    b, nb = offdiag(feat, "paper", no_syc)
    print(f"  {feat:<14} all11={a:.4f} (n={na})  minus_syc={b:.4f} (n={nb})  lift={b-a:+.4f}")

print("\n=== stored block-9/10 summary fields, for cross-reference ===")
for k in ("row_bootstrap_offdiag_ci95",):
    v = d.get(k, {})
    for key in sorted(v):
        if key.endswith("|10") and key.split("|")[0] in ("raw", "axes", "pc:unified", "pc:deception"):
            print(f"  {key}: {v[key]}")

# ---- robustness: the same four table cells at every stored layer and both poolings ----
print("\n=== ROBUSTNESS: 'axes' as target / as source at every stored layer (probe=paper) ===")
import glob, re
for pool in ("mean", "last"):
    print(f"  -- pooling={pool}")
    for fp in sorted(glob.glob(os.path.join(root, "matrices", f"{pool}_L*_s42.json")),
                     key=lambda s: int(re.search(r"_L(\d+)_", s).group(1))):
        L = int(re.search(r"_L(\d+)_", fp).group(1))
        dd = json.load(open(fp, encoding="utf-8"))
        dss = dd["datasets"]; ii = {(r["feature"], r["probe"]): r["auroc"] for r in dd["results"]}
        def cm(f, t):
            j = dss.index(t); v = [ii[(f, "paper")][i][j] for i in range(len(dss)) if i != j and ii[(f, "paper")][i][j] is not None]; return statistics.fmean(v)
        def rm(f, s):
            i = dss.index(s); v = [ii[(f, "paper")][i][j] for j in range(len(dss)) if j != i and ii[(f, "paper")][i][j] is not None]; return statistics.fmean(v)
        zj = dss.index("sycophancy_mt")
        zs = {z["direction"]: z["auroc"][zj] for z in dd.get("zeroshot", [])}
        print(f"    L{L:<3} HPKR axes tgt={cm('axes','harm-pressure-knowledge-report'):.3f} src={rm('axes','harm-pressure-knowledge-report'):.3f}"
              f" | SYC axes tgt={cm('axes','sycophancy_mt'):.3f} src={rm('axes','sycophancy_mt'):.3f}"
              f" | SYC rnd tgt={cm('random','sycophancy_mt'):.3f} raw tgt={cm('raw','sycophancy_mt'):.3f}"
              f" | zs contrast={zs.get('contrast'):.3f} PC1={zs.get('PC1'):.3f}")
