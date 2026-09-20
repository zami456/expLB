"""The WHOLE-MATRIX off-diagonal comparison, which is a different quantity from F-66.

F-66 tested the `sycophancy_mt` COLUMN -- the dataset the bank was designed against. This reads
the pipeline's own dataset-level bootstrap over the FULL 11x11 off-diagonal mean, i.e. "does the
capitulation axis help everywhere, or only where it was aimed?" Those can disagree, and if they
do the disagreement is the finding, not something to average away.
"""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "run-crossgen/kd11/banks/cap_s23/significance.json"
d = json.load(open(path, encoding="utf-8"))
rows = d["tests"] if "tests" in d else d.get("results", d)
if isinstance(rows, dict):
    rows = rows.get("tests", [])

want = ("pc:capitulation@2 - raw", "pc:capitulation - raw", "pc:cap_only@2 - raw",
        "pc:capitulation@2 - datapca@2", "pc:capitulation@2 - random@2",
        "datapca@2 - raw", "random@2 - raw")
print(f"{path}\n  layer 40, pooling mean, probe paper -- FULL off-diagonal mean\n")
print(f"  {'comparison':<34}{'block':>7}{'diff':>9}{'pairs':>9}{'95% CI':>22}{'q_BH':>8}")
seen = 0
for r in rows:
    if r.get("layer") != 40 or r.get("pooling") != "mean" or r.get("probe") != "paper":
        continue
    c = r.get("comparison") or f"{r.get('a')} - {r.get('b')}"
    if c not in want:
        continue
    ci = r.get("dataset_ci95") or r.get("ci95") or [None, None]
    blk = r.get("block", "?")
    cis = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci[0] is not None else "--"
    print(f"  {c:<34}{str(blk):>7}{r.get('diff', float('nan')):>+9.4f}"
          f"{str(r.get('pairs_won','?'))+'/'+str(r.get('pairs','?')):>9}{cis:>22}"
          f"{r.get('q_BH', r.get('q', float('nan'))):>8.3f}")
    seen += 1
if not seen:
    print("  no matching rows -- keys present on a test row:")
    print("   ", sorted(rows[0].keys()) if rows else "(no rows)")
