"""Verify F-04 from a kd11 significance.json (path given as argv[1])."""
import collections, json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
t = d["tests"]
print("file", p)
print("total tests", len(t), "boot_datasets", d.get("boot_datasets"), "note", d.get("note"))
by = collections.Counter((x["block"], x["pooling"]) for x in t)
print("blocks:", collections.Counter(x["block"] for x in t))
print("poolings:", collections.Counter(x["pooling"] for x in t))
print("layers:", sorted(set(x["layer"] for x in t)))
print("probes:", collections.Counter(x["probe"] for x in t))
print("comparisons:", collections.Counter(x["comparison"] for x in t))
print("significant_datasets TRUE total:", sum(1 for x in t if x.get("significant_datasets")))
for blk in sorted(set(x["block"] for x in t)):
    sel = [x for x in t if x["block"] == blk]
    print(f"  block {blk}: n={len(sel)} sig={sum(1 for x in sel if x.get('significant_datasets'))}")
print("--- L40 / mean / paper ---")
for x in t:
    if x["layer"] == 40 and x["pooling"] == "mean" and x["probe"] == "paper":
        print(f"  block={x['block']:>4} cmp={x['comparison']:<22} diff={x['diff']:+.4f} ds_ci={[round(v,4) for v in x['dataset_ci95']]} p={x['dataset_p']} q={x['dataset_q_bh']:.4f} sig={x['significant_datasets']}")
