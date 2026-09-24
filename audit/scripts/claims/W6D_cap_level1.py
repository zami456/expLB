"""W6D-ownerdocs: O-046, O-047, O-048, O-050..O-056.

Level 1 only: puts each sentence of cap/docs/DESIGN.md and kd/docs/NEXT.md beside the value the
audit already recomputed from the raw lab matrices in batch P3 (F-65, F-66, F-67, F-68).
The cap matrices are lab-only (C:/Users/user4/expLB/run-crossgen/kd11/banks/cap_{s23,all}/...),
so this batch reuses those recomputations rather than repeating them, per CLAIM_INSTRUCTIONS.
Read-only.
"""
import csv, os

ROOT = r"C:/Users/SURAIYA/Documents/STUDY/thesis/expLB"
csv.field_size_limit(10 ** 9)
CL = {r["id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "audit/claims.csv"), encoding="utf-8"))}

design = open(os.path.join(ROOT, "cap/docs/DESIGN.md"), encoding="utf-8").read().splitlines()
nextmd = open(os.path.join(ROOT, "kd/docs/NEXT.md"), encoding="utf-8").read().splitlines()

ROWS = [
    ("O-046", nextmd, 15, "F-65/F-66/F-67"),
    ("O-047", nextmd, 104, "F-68"),
    ("O-048", design, 7, "F-65/F-66"),
    ("O-049", design, 20, "F-65 + q10"),
    ("O-050", design, 171, "F-65"),
    ("O-051", design, 176, "F-68"),
    ("O-052", design, 234, "F-66"),
    ("O-053", design, 235, "F-66"),
    ("O-054", design, 249, "F-67"),
    ("O-055", design, 250, "F-67"),
    ("O-056", design, 251, "F-67"),
]
for cid, doc, ln, ev in ROWS:
    print("=" * 92)
    print(f"{cid}   line {ln}   evidence reused from {ev}")
    lo = max(0, ln - 2)
    for i in range(lo, min(len(doc), ln + 1)):
        mark = ">>" if i == ln - 1 else "  "
        print(f"  {mark} :{i+1}: {doc[i]}")

print("\n" + "=" * 92)
print("THE AUDIT-VERIFIED VALUES THESE SENTENCES ARE CHECKED AGAINST")
print("=" * 92)
for f in ["F-65", "F-66", "F-67", "F-68"]:
    r = CL[f]
    print("-" * 92)
    print(f"{f}  verdict={r['verdict']}  confidence={r['confidence']}")
    print(f"  evidence_file : {r['evidence_file']}")
    print(f"  level1_value  : {r['level1_value']}")
    print(f"  level2_value  : {r['level2_value']}")
    print(f"  notes         : {r['notes']}")
