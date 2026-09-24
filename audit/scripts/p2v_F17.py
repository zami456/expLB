"""Verify F-17: 'unified persona axis vs dataset PCA at the same k = -0.003' in a significance.json."""
import collections, json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
t = d["tests"]
print("file", p, "| tests", len(t), "| boot_datasets", d.get("boot_datasets"))
print("blocks", collections.Counter(x["block"] for x in t))
print("layers", sorted(set(x["layer"] for x in t)), "| poolings", sorted(set(x["pooling"] for x in t)))
print("probes", sorted(set(x["probe"] for x in t)))
print("comparisons", sorted(set(x["comparison"] for x in t)))
print("\n-- every test whose comparison mentions unified and datapca --")
for x in t:
    c = x["comparison"]
    if "unified" in c and "datapca" in c:
        print(f"  blk={x['block']:>3} {x['pooling']:>4} L{x['layer']:<3} {x['probe']:<9} {c:<28} diff={x['diff']:+.4f} q={x.get('dataset_q_bh',float('nan')):.4f} sig={x.get('significant_datasets')}")
print("\n-- diffs within 0.0005 of -0.003 --")
for x in t:
    if abs(x["diff"] + 0.003) < 0.0005:
        print(f"  blk={x['block']:>3} {x['pooling']:>4} L{x['layer']:<3} {x['probe']:<9} {x['comparison']:<28} diff={x['diff']:+.5f}")
