"""Read-only structure probe of the cap / cap_ctrl / collab bank matrices on the lab PC."""
import json, os, sys, glob

root = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/user4/expLB/run-crossgen/kd11"
for bank in ("banks/cap_s23", "banks/cap_all", "banks/cap_ctrl", "banks/collab", ""):
    mdir = os.path.join(root, bank, "matrices")
    fs = sorted(glob.glob(os.path.join(mdir, "*_s*.json")))
    print("=" * 70)
    print(bank or "(baseline)", "->", len(fs), "matrix files")
    print("  layers:", sorted({os.path.basename(f) for f in fs})[:6], "...")
    if not fs:
        continue
    p = os.path.join(mdir, "mean_L40_s42.json")
    if not os.path.exists(p):
        p = fs[0]
    d = json.load(open(p, encoding="utf-8"))
    print("  file:", p)
    print("  bank:", d.get("bank"), "| bank_name:", d.get("bank_name"), "| k:", d.get("k"),
          "| seed:", d.get("seed"), "| layer:", d.get("layer"), "| pooling:", d.get("pooling"))
    print("  datasets:", d["datasets"])
    print("  features:", sorted({r["feature"] for r in d["results"]}))
    print("  probes  :", sorted({r["probe"] for r in d["results"]}))
    print("  zeroshot:", [z["direction"] for z in d.get("zeroshot", [])])
    z0 = (d.get("zeroshot") or [{}])[0]
    print("  zeroshot keys:", list(z0))
sig = os.path.join(root, "banks", "cap_s23", "significance.json")
for s in (sig, os.path.join(root, "banks", "cap_all", "significance.json"),
          os.path.join(root, "banks", "cap_ctrl", "significance.json"),
          os.path.join(root, "banks", "collab", "significance.json")):
    if os.path.exists(s):
        dd = json.load(open(s, encoding="utf-8"))
        print(s, "-> tests:", len(dd["tests"]),
              "| comparisons:", len(sorted({x["comparison"] for x in dd["tests"]})),
              "| layers:", len(sorted({x["layer"] for x in dd["tests"]})))
    else:
        print(s, "MISSING")
