"""Emit exactly what fig5 needs, small enough to survive scp.

The crossgen matrices are ~455 KB each and scp truncates at 204,800 bytes silently, exit 0
(measured; see the scp-truncates-at-200kib memory). matplotlib also cannot load on the lab PC.
So the numbers are reduced HERE and the figure is drawn on the laptop from this payload.
"""
import json, statistics, os
import numpy as np

DS = "sycophancy_mt"
FEATS = ["pc:capitulation@2", "pc:capitulation", "pc:cap_only@2", "pc:cap_only",
         "raw", "length", "datapca", "datapca@2", "random", "random@2"]
BASE_FEATS = ["pc:deception", "pc:unified", "pc:unified@2", "axes", "axes_resid"]
PAIRS = [("pc:capitulation@2", "raw"), ("pc:capitulation@2", "datapca@2"),
         ("pc:capitulation@2", "random@2"), ("pc:capitulation@2", "length"),
         ("pc:capitulation@2", "pc:cap_only@2"), ("pc:cap_only@2", "raw")]
B = 10000

def col(r, ds, j):
    A = r["auroc"]
    return [A[i][j] for i in range(len(ds)) if i != j]

out = {"dataset": DS, "register_baseline_zeroshot": 0.374, "B": B, "banks": {}}

base = json.load(open("run-crossgen/kd11/matrices/mean_L40_s42.json", encoding="utf-8"))
bds = base["datasets"]; bj = bds.index(DS)
out["paper16_L40"] = {r["feature"]: statistics.fmean([v for v in col(r, bds, bj) if v is not None])
                      for r in base["results"]
                      if r["probe"] == "paper" and r["feature"] in BASE_FEATS + ["raw"]}

rng = np.random.default_rng(42)
for bank in ("cap_s23", "cap_all"):
    for lay in (20, 40, 44):
        p = f"run-crossgen/kd11/banks/{bank}/matrices/mean_L{lay}_s42.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        ds = d["datasets"]; j = ds.index(DS)
        cols = {r["feature"]: col(r, ds, j) for r in d["results"]
                if r["probe"] == "paper" and r["feature"] in FEATS}
        rec = {"sources": [ds[i] for i in range(len(ds)) if i != j],
               "means": {f: statistics.fmean(v) for f, v in cols.items()},
               "per_source": {f: v for f, v in cols.items()},
               "zeroshot": {}}
        # EVERY direction, with the row-level CI the matrix stores under ci95_lo/ci95_hi.
        # Two earlier readers looked for "lo"/"hi", found nothing, and printed blank intervals
        # for a number that had one all along. Since 2026-09-21 a bank also carries
        # contrast:<name> / PC1:<name> for its non-primary axes -- that is where the A-vs-B
        # comparison lives (F-68), so dropping them here would silently un-report it.
        for z in (d.get("zeroshot") or []):
            def _at(key):
                v = z.get(key)
                if v is None:
                    return None
                return v[j] if isinstance(v, list) else (v.get(DS) if isinstance(v, dict) else v)
            rec["zeroshot"][z["direction"]] = {
                "auroc": _at("auroc"), "lo": _at("ci95_lo"), "hi": _at("ci95_hi")}
        n = len(rec["sources"])
        idx = rng.integers(0, n, size=(B, n))
        rec["boot"] = {}
        for a, b in PAIRS:
            if a in cols and b in cols:
                dif = np.array(cols[a]) - np.array(cols[b])
                bs = dif[idx].mean(axis=1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                rec["boot"][f"{a} - {b}"] = {"diff": float(dif.mean()),
                                             "lo": float(lo), "hi": float(hi),
                                             "p": float((bs <= 0).mean())}
        out["banks"][f"{bank}_L{lay}"] = rec

ax = __import__("torch").load("run-cap/axes_s23.pt", map_location="cpu", weights_only=False)
out["geometry"] = {k: {"n_roles": len(v["roles"]),
                       "pc1_var_L40": float(v["variance_ratio"][40, 0])}
                   for k, v in ax["pca"].items()}

os.makedirs("run-cap/export", exist_ok=True)
fp = "run-cap/export/cap_fig_data.json"
with open(fp, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(f"wrote {fp}  {os.path.getsize(fp)} bytes  (scp truncates at 204800)")
