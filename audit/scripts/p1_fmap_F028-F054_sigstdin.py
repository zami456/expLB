"""Read-only: parse a significance.json streamed on stdin (e.g. from the lab-PC peek tool 'head' mode,
trailing 'python exit N' line stripped) and print the same summary as p1_fmap_F028-F054_gitsig.py.
Usage: <peek ... head 9000000> | python audit/scripts/p1_fmap_F028-F054_sigstdin.py "<comparison>" <probe> <pooling> [layers...]"""
import hashlib, json, sys
from collections import Counter
raw = sys.stdin.buffer.read().decode("utf-8")
i = raw.rfind("python exit")
body = raw[:i] if i >= 0 else raw
d = json.loads(body)
t = d["tests"]
print("n_tests", len(t), "boot_datasets", d.get("boot_datasets"), "note", d.get("note"))
print(" blocks", Counter(x.get("block") for x in t))
print(" layers", sorted(Counter(x["layer"] for x in t)))
print(" poolings", Counter(x["pooling"] for x in t), "probes", Counter(x["probe"] for x in t))
sig = [x for x in t if x.get("significant_datasets")]
print(" n significant_datasets", len(sig))
for x in sig:
    print(f"   SIG L{x['layer']} {x['pooling']} {x['probe']} {x['comparison']} pairs={x['pairs']} diff={x['diff']:+.4f} q={x.get('dataset_q_bh')}")
want = sys.argv[1:]
if want:
    comp, probe, pool = want[0], want[1], want[2]
    layers = [int(v) for v in want[3:]] or None
    for x in sorted(t, key=lambda x: (x["pairs"], x["layer"])):
        if x["comparison"] == comp and x["probe"] == probe and x["pooling"] == pool and (layers is None or x["layer"] in layers):
            ci = x.get("dataset_ci95")
            print(f"  L{x['layer']} blk={x.get('block')} pairs={x['pairs']} diff={x['diff']:+.4f} won={x.get('pairs_won')} ci={[round(c,3) for c in ci] if ci else None} q={x.get('dataset_q_bh')} sig={x.get('significant_datasets')}")
