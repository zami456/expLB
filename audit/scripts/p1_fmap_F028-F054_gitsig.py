"""Read-only: summarise git (asix HEAD) copies of crossgen significance.json files for F-31/F-34 mapping.
Prints n tests, layers, poolings, boot_datasets, and the tests matching a comparison/probe/pooling/layer."""
import json, subprocess, sys
from collections import Counter
path = sys.argv[1]; want = sys.argv[2:]  # e.g. "pc:unified - raw" paper mean 40
d = json.loads(subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, check=True).stdout)
t = d["tests"]
print(path, "n_tests", len(t), "boot_datasets", d.get("boot_datasets"), "top keys", list(d)[:12])
print(" layers", sorted(Counter(x["layer"] for x in t)))
print(" poolings", Counter(x["pooling"] for x in t), "probes", Counter(x["probe"] for x in t))
print(" n significant_datasets", sum(1 for x in t if x.get("significant_datasets")))
if want:
    comp, probe, pool = want[0], want[1], want[2]
    layers = [int(x) for x in want[3:]] or None
    for x in sorted(t, key=lambda x: (x["pairs"], x["layer"])):
        if x["comparison"] == comp and x["probe"] == probe and x["pooling"] == pool and (layers is None or x["layer"] in layers):
            print(f"  L{x['layer']} pairs={x['pairs']} diff={x['diff']:+.4f} won={x.get('pairs_won')} ci={x.get('dataset_ci95')} q={x.get('dataset_q_bh')} sig={x.get('significant_datasets')}")
