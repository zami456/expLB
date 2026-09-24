"""F-67 - the cap advantage is LOCAL: on the full 11x11 off-diagonal mean it does not clear.

Reads banks/cap_{s23,all}/significance.json, prints the seven rows of the entry's table, checks
the s23-vs-all agreement, and RECONSTRUCTS the 792-test BH family the entry cites (the file now
has 5,544 tests after the layer sweep, RF-13) by keeping only layers 20/40/44 and re-running BH.

Usage: python F-67.py [<kd11 root>]
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _dsboot as DB

root = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/user4/expLB/run-crossgen/kd11"
ROWS = [("pc:capitulation@2 - random@2", "10"), ("pc:capitulation@2 - datapca@2", "10"),
        ("pc:capitulation@2 - raw", "10"), ("pc:capitulation@2 - raw", "9"),
        ("pc:capitulation@2 - datapca@2", "9"), ("pc:capitulation@2 - random@2", "9"),
        ("random@2 - raw", "10")]

store = {}
for bank in ("cap_s23", "cap_all"):
    p = os.path.join(root, "banks", bank, "significance.json")
    if not os.path.exists(p):
        print("MISSING", p); continue
    d = json.load(open(p, encoding="utf-8"))
    t = d["tests"]
    store[bank] = t
    layers = sorted({x["layer"] for x in t})
    comps = sorted({x["comparison"] for x in t})
    print("=" * 78)
    print(p)
    print(f"  tests={len(t)}  boot_datasets={d.get('boot_datasets')}  layers={len(layers)}  comparisons={len(comps)}")
    print(f"  (3 layers x 2 poolings x 3 probes x {len(comps)} comparisons x 2 blocks = "
          f"{3*2*3*len(comps)*2}  <- the entry's '792 tests')")
    print(f"  === the entry's table: L40, mean pooling, paper probe ===")
    for comp, block in ROWS:
        x = next((y for y in t if y["comparison"] == comp and y["block"] == block
                  and y["layer"] == 40 and y["pooling"] == "mean" and y["probe"] == "paper"), None)
        if x is None:
            print(f"  {comp:<36} blk{block}: NOT PRESENT"); continue
        ci = x["dataset_ci95"]
        print(f"  {comp:<36} blk{block:<3} diff={x['diff']:+.4f} pairs={x['pairs_won']}/{x['pairs']}"
              f" ci95=[{ci[0]:+.3f}, {ci[1]:+.3f}] p={x['dataset_p']:.3f} q={x['dataset_q_bh']:.3f}"
              f" clears={x['significant_datasets']}"
              + ("  CI>0" if ci[0] > 0 else ("  CI<0" if ci[1] < 0 else "")))

if "cap_s23" in store and "cap_all" in store:
    print("\n=== s23 vs all agreement on the entry's rows (L40, mean, paper) ===")
    worst = 0.0
    for comp, block in ROWS:
        def get(b):
            return next((y for y in store[b] if y["comparison"] == comp and y["block"] == block
                         and y["layer"] == 40 and y["pooling"] == "mean" and y["probe"] == "paper"), None)
        a, b = get("cap_s23"), get("cap_all")
        if a and b:
            worst = max(worst, abs(a["diff"] - b["diff"]))
            print(f"  {comp:<36} blk{block:<3} s23={a['diff']:+.4f}  all={b['diff']:+.4f}"
                  f"  |diff|={abs(a['diff']-b['diff']):.4f}")
    print(f"  LARGEST |s23 - all| over these rows: {worst:.4f}  (entry says '<= 0.001')")

print("\n=== RECONSTRUCTED 792-test family (layers 20/40/44 only), BH re-run per block ===")
for bank in ("cap_s23", "cap_all"):
    if bank not in store:
        continue
    fam = [x for x in store[bank] if x["layer"] in (20, 40, 44)]
    print(f"  {bank}: {len(fam)} tests in the reconstructed family")
    for block in ("10", "9"):
        sub = [x for x in fam if x["block"] == block]
        for x, q in zip(sub, DB.bh([y["dataset_p"] for y in sub])):
            x["q792"] = float(q)
    for comp, block in ROWS:
        x = next((y for y in fam if y["comparison"] == comp and y["block"] == block
                  and y["layer"] == 40 and y["pooling"] == "mean" and y["probe"] == "paper"), None)
        if x:
            print(f"    {comp:<36} blk{block:<3} diff={x['diff']:+.4f} q792={x['q792']:.3f}"
                  f" q_now={x['dataset_q_bh']:.3f} clears792={x['q792'] < 0.05}")
