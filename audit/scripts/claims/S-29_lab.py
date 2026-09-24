"""S-29 (read-only, run on the lab PC): the 80-layer profile of the kd matched slice.
Counts, from the raw passA11 matrices, at how many of the 80 layers `axes` (k=5, trained on
harm-pressure-kr-matched, row mean over the other ten) beats `raw` and `datapca`.
Settles F-75's '28 of 80 / 31 of 80' against F-77/F-81's '37/80 (46%)'."""
import json, os, re, statistics, sys

ROOT = "C:/Users/user4/expLB/run-crossgen/kd11/passA11"
DS = "harm-pressure-kr-matched"
PROBE = "paper"
POOL = "mean"

def rowmean(entry, datasets, i):
    A = entry["auroc"]
    v = [A[i][j] for j in range(len(datasets)) if j != i and A[i][j] is not None]
    return statistics.fmean(v) if v else float("nan")

def load(dirpath, feats):
    out = {}
    md = os.path.join(dirpath, "matrices")
    if not os.path.isdir(md):
        print("MISSING", md); return out
    for f in sorted(os.listdir(md)):
        if not f.endswith(".json"): continue
        d = json.load(open(os.path.join(md, f), encoding="utf-8"))
        if d.get("pooling") != POOL: continue
        if DS not in d["datasets"]: continue
        i = d["datasets"].index(DS)
        lay = d["layer"]
        for r in d["results"]:
            if r["probe"] == PROBE and r["feature"] in feats:
                out.setdefault(lay, {})[r["feature"]] = rowmean(r, d["datasets"], i)
        out.setdefault(lay, {})["_n_datasets"] = len(d["datasets"])
        out[lay]["_file"] = f
        out[lay]["_bank"] = d.get("bank")
    return out

root = load(ROOT, {"raw", "datapca", "random", "length"})
bank = load(os.path.join(ROOT, "banks", "kd_s23"), {"axes", "datapca"})
print("root layers:", len(root), " bank layers:", len(bank))
bankname = {v.get("_bank") for v in bank.values()}
print("bank field in bank matrices:", bankname, " in root matrices:", {v.get("_bank") for v in root.values()})
print("datasets per matrix:", {v.get("_n_datasets") for v in root.values()}, {v.get("_n_datasets") for v in bank.values()})

layers = sorted(set(root) & set(bank))
print("layers with both:", len(layers))
rows = []
for L in layers:
    a = bank[L].get("axes"); r = root[L].get("raw"); dp = root[L].get("datapca")
    dpb = bank[L].get("datapca")
    rows.append((L, a, r, dp, dpb))
win_raw = sum(1 for L, a, r, dp, dpb in rows if a is not None and r is not None and a > r)
win_dp = sum(1 for L, a, r, dp, dpb in rows if a is not None and dp is not None and a > dp)
win_dpb = sum(1 for L, a, r, dp, dpb in rows if a is not None and dpb is not None and a > dpb)
print(f"\naxes beats raw at      {win_raw} of {len(rows)} layers")
print(f"axes beats datapca(root) at {win_dp} of {len(rows)} layers")
print(f"axes beats datapca(bank store) at {win_dpb} of {len(rows)} layers")
# longest contiguous run over datapca(root)
best = cur = 0
for L, a, r, dp, dpb in rows:
    if a is not None and dp is not None and a - dp > 0: cur += 1; best = max(best, cur)
    else: cur = 0
print("longest contiguous run with margin over datapca(root) > 0:", best)
print("\nspot layers:")
for L in (0, 20, 22, 40, 44, 77):
    if L in dict((x[0], x) for x in rows):
        _, a, r, dp, dpb = dict((x[0], x) for x in rows)[L]
        print(f"  L{L:<3d} axes {a:.4f}  raw {r:.4f} ({a-r:+.4f})  datapca {dp:.4f} ({a-dp:+.4f})")
print("\nfull profile (layer, axes, raw, datapca, a-raw, a-dpca):")
for L, a, r, dp, dpb in rows:
    print(f"{L:5d} {a:8.4f} {r:8.4f} {dp:8.4f} {a-r:+8.4f} {a-dp:+8.4f}")
