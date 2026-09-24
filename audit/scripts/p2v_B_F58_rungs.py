"""Verify batch B / F-58: recompute the three-rung table (3B / 8B / 70B) from the matrix JSONs.

Inputs (pulled from the lab PC, md5-verified, see audit/cache/MANIFEST.csv):
  audit/cache/labpc/run-crossgen/scale_3b/matrices/mean_L16_s42.json
  audit/cache/labpc/run-crossgen/scale_8b/matrices/mean_L16_s42.json
  audit/cache/labpc/run-crossgen/kd11/matrices/mean_L40_s42.json   (11 datasets; restricted to the 5)

Each `results` entry has feature, probe and `auroc` = an NxN matrix (rows = train dataset,
cols = test dataset).  The ledger's number is the mean of the OFF-DIAGONAL cells
(N*(N-1) = 20 pairs when N = 5), paper probe.
"""
import json, os, sys

BASE = os.path.join(os.path.dirname(__file__), "..", "cache", "labpc", "run-crossgen")


def offdiag(mat, idx=None):
    n = len(mat)
    ids = range(n) if idx is None else idx
    vals = [mat[i][j] for i in ids for j in ids if i != j and mat[i][j] is not None]
    return sum(vals) / len(vals), len(vals)


def rung(path, probe="paper", restrict=None):
    d = json.load(open(path, encoding="utf-8"))
    ds = d["datasets"]
    idx = list(range(len(ds))) if restrict is None else [ds.index(x) for x in restrict]
    print("-" * 78)
    print(os.path.relpath(path, BASE), "| layer", d["layer"], "| pooling", d["pooling"],
          "| seed", d["seed"], "| bank", d.get("bank"), d.get("bank_name"), "| k", d.get("k"))
    print("  datasets (%d):" % len(ds), ds)
    if restrict:
        print("  restricted to (%d):" % len(idx), [ds[i] for i in idx])
    out = {}
    for r in d["results"]:
        if r["probe"] != probe:
            continue
        m, npairs = offdiag(r["auroc"], idx)
        out[r["feature"]] = m
    raw = out["raw"]
    print("  raw off-diagonal = %.4f  (%d pairs)" % (raw, npairs))
    for f, v in sorted(out.items(), key=lambda kv: -kv[1]):
        print("     %-22s %.4f   gain vs raw %+.4f" % (f, v, v - raw))
    return out, raw, ds


five = None
for name, path, restrict in [
    ("3B  L16", os.path.join(BASE, "scale_3b", "matrices", "mean_L16_s42.json"), None),
    ("8B  L16", os.path.join(BASE, "scale_8b", "matrices", "mean_L16_s42.json"), None),
]:
    out, raw, ds = rung(path, restrict=restrict)
    if five is None:
        five = ds

out70, raw70, ds70 = rung(os.path.join(BASE, "kd11", "matrices", "mean_L40_s42.json"),
                          restrict=five)
print("-" * 78)
print("ledger F-58 table: 3B L16 raw 0.708 axes +0.089 random -0.246 datapca -0.150")
print("                   8B L16 raw 0.743 pc:deception +0.086 random -0.025 datapca -0.017")
print("                   70B L40 raw 0.809 pc:unified@2 +0.026 random -0.214 datapca -0.112")
