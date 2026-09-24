"""Audit Phase 2 / W5A-ledger - F-57 check.

F-57 (kd/docs/FINDINGS.md:1760) describes four thesis figures and the numbers in them:
  fig1_register      axis 0.300 / 0.379 vs random 0.659 / 0.633
  fig2_layer_profile negative in the first quarter, peak L44 at +0.116, sustained to L76,
                     "with the dataset-level CI band spanning zero at every layer"
  fig3_baseline_art. "+0.10 at 0.60 (paper, 3B)" against "+0.026 at 0.809 (ours, 70B)"
  fig4_geometry      cos(PC1, assistant) 0.435 / 0.732 / 0.623
  plus: fig_data.json is 28,669 bytes; the 21-layer significance.json is 1.38 MB.

This script checks every number that has a file behind it, and grep-checks the plotting
script for the hardcoded ones. Read-only, laptop copies.
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
FIG = os.path.join(ROOT, "kd", "results", "fig_data.json")
SCRIPT = os.path.join(ROOT, "kd", "scripts", "make_thesis_figures.py")


def main():
    print("== fig_data.json ==")
    print("path:", FIG)
    print("bytes:", os.path.getsize(FIG), " (F-57 says 28,669)")
    d = json.load(open(FIG, encoding="utf-8"))
    print("source:", d["source"], "| probe:", d["probe"], "| pooling:", d["pooling"],
          "| pairs:", d["pairs"])
    print()

    print("== fig2: the 'pc:unified - raw' layer profile in fig_data.json ==")
    prof = d["profile"]["pc:unified - raw"]
    print(" layer    diff       lo        hi        q      won/pairs  lo>0?")
    peak = max(prof, key=lambda r: r["diff"])
    for r in prof:
        print("  %-5d %+8.4f %+9.4f %+9.4f  %.4f   %3d/%3d   %s" %
              (r["layer"], r["diff"], r["lo"], r["hi"], r["q"], r["won"], r["pairs"],
               "YES" if r["lo"] > 0 else ""))
    print(" peak layer = L%d at %+.4f  (F-57: peak L44 at +0.116)" %
          (peak["layer"], peak["diff"]))
    neg = [r["layer"] for r in prof if r["diff"] < 0]
    print(" layers with diff<0:", neg)
    spans0 = [r["layer"] for r in prof if r["lo"] <= 0 <= r["hi"]]
    notspan = [r["layer"] for r in prof if not (r["lo"] <= 0 <= r["hi"])]
    print(" CI spans zero at %d of %d layers; NOT spanning zero at: %s" %
          (len(spans0), len(prof), notspan))
    print(" (F-57: 'the dataset-level CI band spanning zero at every layer')")
    print(" census:", d["census"])
    print()

    print("== the plotting script's own numbers (hardcoded, per F-57's caveat) ==")
    src = open(SCRIPT, encoding="utf-8").read()
    for name in ["fig1", "fig2", "fig3", "fig4"]:
        m = re.search(r"def\s+%s\w*\(.*?\n(?=def |\Z)" % name, src, re.S)
        if m:
            body = m.group(0)
            nums = re.findall(r"[-+]?\d*\.\d+", body)
            print(" %s: numeric literals = %s" % (name, nums[:30]))
    print()
    for pat in [r"0\.300", r"0\.379", r"0\.659", r"0\.633",
                r"0\.435", r"0\.732", r"0\.623",
                r"0\.60\b", r"0\.809", r"0\.026", r"0\.10\b",
                r"0\.7619", r"0\.7980", r"0\.8090"]:
        hits = [i + 1 for i, l in enumerate(src.splitlines()) if re.search(pat, l)]
        print(" pattern %-8s in make_thesis_figures.py at lines %s" % (pat, hits))
    print()

    print("== the figure files ==")
    for p in sorted(glob.glob(os.path.join(ROOT, "figures", "thesis", "*"))):
        print("  %-60s %d B" % (os.path.relpath(p, ROOT), os.path.getsize(p)))
    print()

    print("== the local kd_s23 significance.json (the stated source) ==")
    sig = os.path.join(ROOT, "run-crossgen", "kd11", "banks", "kd_s23", "significance.json")
    if os.path.exists(sig):
        s = json.load(open(sig, encoding="utf-8"))
        print("  path:", os.path.relpath(sig, ROOT), os.path.getsize(sig), "B",
              "(F-57 says the 21-layer file is 1.38 MB)")
        tests = s if isinstance(s, list) else s.get("tests", s)
        if isinstance(tests, dict):
            print("  top-level keys:", list(tests.keys())[:10])
            for k in list(tests.keys())[:3]:
                v = tests[k]
                print("   ", k, type(v).__name__, len(v) if hasattr(v, "__len__") else v)
        if isinstance(tests, list):
            print("  n tests:", len(tests))
            print("  layers:", sorted(set(t.get("layer") for t in tests)))
    else:
        print("  MISSING locally:", sig)


if __name__ == "__main__":
    main()
