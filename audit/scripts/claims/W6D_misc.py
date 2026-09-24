"""W6D-ownerdocs: O-039 (L44 peak), O-048/O-049 (the DF-31 correction in cap/docs/DESIGN.md),
and the pre-registration provenance behind O-097.

All read-only. Uses the locally cached lab copies plus `git show` (no checkout).
"""
import json, hashlib, os, subprocess
import numpy as np

ROOT = r"C:/Users/SURAIYA/Documents/STUDY/thesis/expLB"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", ROOT] + list(a), capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout


print("=" * 92)
print("O-039  ROADMAP:115 / OPEN_DECISIONS B3  'L44 is the peak by +0.007'")
print("=" * 92)
cands = ["audit/cache/labpc/run-crossgen/kd11/banks/kd_s23/significance.json",
         "run-crossgen/kd11/banks/kd_s23/significance.json"]
for rel in cands:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        print("  missing:", rel); continue
    d = json.load(open(p))
    t = d["tests"] if isinstance(d, dict) and "tests" in d else d
    print(f"\n--- {rel}\n    size={os.path.getsize(p)}  md5={md5(p)}  n_tests={len(t)}")
    layers = sorted({x["layer"] for x in t})
    print("    layers present:", layers)
    print("    blocks present:", sorted({str(x.get("block")) for x in t}))
    for comp in ["pc:unified - raw", "pc:unified - datapca"]:
        sel = {x["layer"]: x for x in t
               if x.get("comparison") == comp and str(x.get("block")) == "9"
               and x.get("pooling") == "mean" and x.get("probe") == "paper"}
        if not sel:
            print(f"    [{comp}] block '9' mean/paper: NO ROWS"); continue
        ls = sorted(sel)
        vals = {L: sel[L]["diff"] for L in ls}
        peak = max(vals, key=vals.get)
        print(f"    [{comp}] block '9' mean/paper over {len(ls)} layers")
        print("      L20 %+.4f  L40 %+.4f  L44 %+.4f  | peak L%d %+.4f | L44-L40 %+.4f"
              % (vals.get(20, float('nan')), vals.get(40, float('nan')), vals.get(44, float('nan')),
                 peak, vals[peak], vals.get(44, float('nan')) - vals.get(40, float('nan'))))
        print("      positive at %d/%d layers; full profile:" % (sum(v > 0 for v in vals.values()), len(ls)))
        print("      " + "  ".join(f"L{L}:{vals[L]:+.4f}" for L in ls))
        for L in (40, 44):
            if L in sel:
                print(f"      L{L} CI {sel[L].get('dataset_ci95')}  q_bh_now {sel[L].get('dataset_q_bh')}"
                      f"  sig {sel[L].get('significant_datasets')}")

print("\n  OTHER LAYER SWEEPS IN THE PROJECT (checked rows of audit/claims.csv), for scope:")
print("   F-75 (matched-slice axes, 80 layers): argmax L77 0.7642; L40 0.7393 > L44 0.7321")
print("   F-76 (bootstrap argmax over 80 layers): median L31, 95% [L5,L79], 0.0% on L40")
print("   F-91 (cap_s23 / kd_s23 crossgen off-diagonal): peak L32 for raw and cap; kd_s23 axes peak L44")

print("\n" + "=" * 92)
print("O-048 / O-049  DF-31: what 0.374 is, and whether cap/docs/DESIGN.md now says so")
print("=" * 92)
for rel in ["run-crossgen/kd11/matrices/mean_L40_s42.json",
            "audit/cache/labpc/run-crossgen/kd11/matrices/mean_L40_s42.json"]:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        print("  missing:", rel); continue
    d = json.load(open(p))
    ds = d.get("datasets") or d.get("dataset_names")
    j = ds.index("sycophancy_mt")
    print(f"\n--- {rel}  md5={md5(p)}  bank={d.get('bank')}  bank_name={d.get('bank_name')} "
          f"layer={d.get('layer')} pooling={d.get('pooling')} seed={d.get('seed')}")
    for z in d["zeroshot"]:
        if z["direction"] in ("contrast", "PC1") or "syco" in z["direction"]:
            print(f"    zeroshot[{z['direction']:<28}] sycophancy_mt = {z['auroc'][j]:.4f}")
print("\n  code that names the bare rows (src/assistant_axis_ndif/27_crossgen.py:405-420):")
src = open(os.path.join(ROOT, "src/assistant_axis_ndif/27_crossgen.py")).read().splitlines()
for i in range(404, 420):
    print(f"   {i+1:>4}: {src[i]}")

q = os.path.join(ROOT, "run-coords/q10_sycophancy_axis.json")
d = json.load(open(q))
print(f"\n--- run-coords/q10_sycophancy_axis.json  md5={md5(q)}")


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, path + "/" + str(k))
    else:
        yield path, o


hits = [(p, v) for p, v in walk(d) if "40" in p.split("/")[-1] and isinstance(v, (int, float))]
for k in d:
    print("   top key:", k, "->", list(d[k].keys())[:8] if isinstance(d[k], dict) else type(d[k]).__name__)
r = d["results"]["paper16|mean"]
for axis in r:
    kl = r[axis].get("key_layers", {})
    if "40" in kl:
        v = kl["40"]
        print(f"   results['paper16|mean']['{axis}']['key_layers']['40'] auroc={v['auroc']:.5f} "
              f"ci95={[round(x,4) for x in v['ci95']]}")
print("   random-direction null, mean pooling:", json.dumps(d["random"]["mean"])[:600])
print("   length_only:", d["length_only"], " n:", d["n"])

print("\n  DESIGN.md lines that carry the correction:")
doc = open(os.path.join(ROOT, "cap/docs/DESIGN.md"), encoding="utf-8").read().splitlines()
for i in [5, 6, 7, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 227, 228, 229]:
    print(f"   :{i+1}: {doc[i]}")
print("\n  Same sentence in kd/docs/NEXT.md (O-046):")
nx = open(os.path.join(ROOT, "kd/docs/NEXT.md"), encoding="utf-8").read().splitlines()
print(f"   :15: {nx[14]}")

print("\n" + "=" * 92)
print("O-097  provenance of the 'pre-registered' G2 threshold 0.65")
print("=" * 92)
print("  git log (cap sarcasm files):")
print(git("log", "--format=%h %ad %s", "--date=iso", "--",
          "cap/docs/SARCASM_DESIGN.md", "cap/scripts/run_sarc_pilot.ps1",
          "cap/scripts/score_sarcasm_markers.py"))
for c in ["3d80071", "5ced550", "aea9f53", "b34ff2e"]:
    when = git("show", "-s", "--format=%ad", "--date=iso", c).strip()
    print(f"  {c} {when}")
    for f in ["cap/docs/SARCASM_DESIGN.md", "cap/scripts/score_sarcasm_markers.py",
              "cap/scripts/run_sarc_pilot.ps1"]:
        blob = git("show", f"{c}:{f}")
        if not blob:
            print(f"      {f:<42} (file not in commit)")
        else:
            print(f"      {f:<42} '0.65' x{blob.count('0.65')}  '0.20' x{blob.count('0.20')} "
                  f" '0.75' x{blob.count('0.75')}")
print("  working tree cap/docs/SARCASM_DESIGN.md '0.65' count:",
      open(os.path.join(ROOT, "cap/docs/SARCASM_DESIGN.md"), encoding="utf-8").read().count("0.65"))
print("\n  Generation window (from the preserved chain status, batch P7 dump):")
print("   START 2026-09-21 21:21:17 ... '1 generate ... PASS' 2026-09-21 21:40:53")
print("   first commit containing 0.65 = b34ff2e 2026-09-21 21:48:54  -> AFTER the data")
