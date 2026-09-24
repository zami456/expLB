"""F-34, version check (RF-13). The 3,276-test significance.json F-34 quotes no longer exists;
the current file has 5,544 tests because `comparisons_for()` added per-bank persona comparisons
(the F-67 fix). Raw statistics are unchanged by regeneration - only the BH family grew.

This script RECONSTRUCTS the 3,276-test family from the current file by keeping only the 13
comparisons that the pre-extension 468-test file used, then re-running BH within each block,
and reports the q-values F-34 quotes.

Usage: python F-34b.py <current significance.json> <old 3-layer significance.json or ''>
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _dsboot as DB

cur = json.load(open(sys.argv[1], encoding="utf-8"))["tests"]
OLD13 = None
if len(sys.argv) > 2 and sys.argv[2]:
    OLD13 = sorted({x["comparison"] for x in json.load(open(sys.argv[2], encoding="utf-8"))["tests"]})
if OLD13 is None:
    OLD13 = ['axes - datapca', 'axes - random', 'axes - raw', 'datapca - raw', 'datapca@2 - raw',
             'pc:deception - datapca', 'pc:deception - random', 'pc:deception - raw',
             'pc:unified - datapca', 'pc:unified - random', 'pc:unified - raw',
             'random - raw', 'random@2 - raw']
print("current tests :", len(cur))
print("old family    :", len(OLD13), "comparisons ->", OLD13)

fam = [x for x in cur if x["comparison"] in OLD13]
print("reconstructed :", len(fam), "tests  (F-34 says 3,276)")

for block in ("10", "9"):
    sub = [x for x in fam if x["block"] == block]
    qs = DB.bh([x["dataset_p"] for x in sub])
    for x, q in zip(sub, qs):
        x["q_old_family"] = float(q)

print("\n=== q under the reconstructed 3,276-test family, block 9, mean, paper ===")
for comp in ("pc:unified - raw", "pc:unified - datapca"):
    rows = sorted([x for x in fam if x["comparison"] == comp and x["block"] == "9"
                   and x["pooling"] == "mean" and x["probe"] == "paper"], key=lambda y: y["layer"])
    for x in rows:
        if x["layer"] in (40, 44):
            print(f"  L{x['layer']:<3} {comp:<24} diff={x['diff']:+.4f} p={x['dataset_p']:.3f}"
                  f" q_current={x['dataset_q_bh']:.3f} q_old_family={x['q_old_family']:.3f}")

print("\n=== best (lowest) q in the reconstructed family over ALL tests ===")
m = min(fam, key=lambda x: x["q_old_family"])
print(f"  q={m['q_old_family']:.3f} {m['comparison']} L{m['layer']} {m['pooling']} blk{m['block']} {m['probe']} diff={m['diff']:+.4f}")
pos = [x for x in fam if x["diff"] > 0 and x["comparison"] in ("pc:unified - raw", "pc:unified - datapca")]
mp = min(pos, key=lambda x: x["q_old_family"])
print(f"  best POSITIVE pc:unified vs raw/datapca: q={mp['q_old_family']:.3f} {mp['comparison']} "
      f"L{mp['layer']} {mp['pooling']} blk{mp['block']} {mp['probe']} diff={mp['diff']:+.4f}")

print("\n=== tests clearing q<0.05 in the reconstructed 3,276-test family ===")
cl = [x for x in fam if x["q_old_family"] < 0.05]
print("  count:", len(cl), "| negative:", sum(1 for x in cl if x["diff"] < 0),
      "| positive:", sum(1 for x in cl if x["diff"] > 0))
for x in sorted(cl, key=lambda y: (y["comparison"], y["layer"])):
    print(f"  L{x['layer']:<3} {x['pooling']:<5} blk{x['block']} {x['probe']:<9} {x['comparison']:<26}"
          f" diff={x['diff']:+.4f} q_old={x['q_old_family']:.3f} q_now={x['dataset_q_bh']:.3f}")
