"""Phase 4 final rates, computed from files so they can be re-derived.

  audit/phase4/merged.csv        - every sampled claim, classified
  audit/phase4/adjudications.csv - the lead's outcome for every DIFFERS

Prints the two rates that matter and the by-tier breakdown, with the tiers whose
raw rate is an artifact of their own construction called out.

  python audit/scripts/p4_rates.py
"""
import csv
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P4 = os.path.join(ROOT, "audit", "phase4")

merged = list(csv.DictReader(open(os.path.join(P4, "merged.csv"), encoding="utf-8")))
adj = {r["id"]: r for r in csv.DictReader(open(os.path.join(P4, "adjudications.csv"), encoding="utf-8"))}

diff = [r for r in merged if r["classification"] in ("DIFFERS", "UNRESOLVED")]
missing = [r["id"] for r in diff if r["id"] not in adj]
extra = [i for i in adj if i not in {r["id"] for r in diff}]

print(f"sampled claims                 : {len(merged)}")
print(f"comparable (had a prior verdict): "
      f"{sum(1 for r in merged if r['classification'] != 'NEW')}")
print(f"new coverage (T4, no prior)     : {sum(1 for r in merged if r['classification'] == 'NEW')}")
if missing:
    print(f"!! {len(missing)} DIFFERS not adjudicated: {missing}")
if extra:
    print(f"!! {len(extra)} adjudications with no DIFFERS row: {extra}")

cmp_rows = [r for r in merged if r["classification"] in ("AGREES", "DIFFERS", "UNRESOLVED")]
n_cmp = len(cmp_rows)
n_diff = len(diff)
oc = Counter(adj[r["id"]]["outcome"] for r in diff if r["id"] in adj)
n_audit_wrong = oc["AUDIT WAS WRONG"]

print()
print("=" * 72)
print(f"RAW disagreement rate        : {n_diff}/{n_cmp} = {n_diff / n_cmp:.1%}")
print(f"ADJUDICATED audit-error rate : {n_audit_wrong}/{n_cmp} = {n_audit_wrong / n_cmp:.1%}")
print("=" * 72)
print()
for k, v in oc.most_common():
    print(f"  {v:3d}  {k}")

print("\nby tier:")
bt = defaultdict(lambda: [0, 0, 0])  # cmp, differs, audit-wrong
for r in cmp_rows:
    t = r["tier"]
    bt[t][0] += 1
    if r["classification"] in ("DIFFERS", "UNRESOLVED"):
        bt[t][1] += 1
        if adj.get(r["id"], {}).get("outcome") == "AUDIT WAS WRONG":
            bt[t][2] += 1
print(f"  {'tier':26s} {'n':>4s} {'raw':>8s} {'audit-err':>10s}")
for t in sorted(bt):
    n, d, a = bt[t]
    print(f"  {t:26s} {n:4d} {d / n:8.1%} {a / n:10.1%}")

print("\nT4 new coverage (the 268 claims Phase 2 never reached):")
t4 = Counter(r["agent_family"] for r in merged if r["classification"] == "NEW")
n4 = sum(t4.values())
for k in ("HOLDS", "PARTLY", "FAILS", "UNRESOLVED"):
    if t4[k]:
        print(f"  {k:11s} {t4[k]:3d}  {t4[k] / n4:6.1%}")

print("\nCAVEATS that must travel with these numbers:")
print("  * T1's raw rate is an artifact: REVISED->X verdicts are invisible to a")
print("    blinded reader, and compound claims have no agreed grading rule.")
print("    T1 audit errors: 0 of 8. No failure verdict was reversed.")
print("  * Batches F1/F2/F3/O1 ran before the verdict-discipline paragraph;")
print("    the raw rate is not comparable across that boundary.")
print("  * A-10 is independence-compromised (self-reported) and 11 batches ran")
print("    before searches were scoped away from audit/. Contamination inflates")
print("    agreement, so it DEFLATES the measured error rate.")
