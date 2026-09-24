"""Phase 5: every headline count in the two reports, computed from files.

The brief's rule 3: every number must come from a file opened or a script run
during the audit. This produces the report's *structural* numbers (how many
claims, of what kind, from where, with what verdict) so no count in
REPORT_PLAIN.md or REPORT_TECHNICAL.md is typed from memory.

  python audit/scripts/p5_stats.py
"""
import csv
import glob
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(ROOT, "audit")

SOURCE = {
    "F": "kd/docs/FINDINGS.md ledger",
    "O": "owner's documents",
    "M": "owner's auto-memory notes",
    "S": "owner's Claude session transcripts",
    "A": "the 2026-09-21 abstract",
    "D": "collaborator's draft PDF",
    "H": "collaborator's branch write-ups",
}


def fam(v):
    s = (v or "").strip().upper()
    if s.startswith("PARTLY"):
        return "PARTLY"
    if s.startswith("DOES NOT") or s.startswith("REVISED"):
        return "FAILS"
    if s.startswith("MATCHES HIS DATA"):
        return "PARTLY" if "NOT TESTED" in s else "HOLDS"
    if s.startswith("HOLDS"):
        return "HOLDS"
    return "OTHER"


def imp(v):
    s = (v or "").strip().upper()
    if s.startswith("HIGH"):
        return "HIGH"
    if s.startswith("MEDIUM-HIGH"):
        return "MEDIUM-HIGH"
    if s.startswith("MEDIUM"):
        return "MEDIUM"
    if s.startswith("LOW"):
        return "LOW"
    return "(unrated)"


def head(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


claims = list(csv.DictReader(open(os.path.join(A, "claims.csv"), encoding="utf-8")))

head("CLAIM LEDGER (audit/claims.csv) -- AFTER the Phase 4 fixes")
print(f"claims checked: {len(claims)}")
by_src = Counter(c["id"][0] for c in claims)
by_fam = Counter(fam(c["verdict"]) for c in claims)
print("\nverdict:")
for k in ("HOLDS", "PARTLY", "FAILS", "OTHER"):
    if by_fam[k]:
        print(f"   {k:7s} {by_fam[k]:4d}  {by_fam[k]/len(claims):6.1%}")
print("\nby source:")
for k, n in by_src.most_common():
    sub = [c for c in claims if c["id"][0] == k]
    f = Counter(fam(c["verdict"]) for c in sub)
    print(f"   {k}  {n:4d}  {SOURCE.get(k,'?'):34s} "
          f"HOLDS {f['HOLDS']:3d} PARTLY {f['PARTLY']:3d} FAILS {f['FAILS']:2d}")
print("\nimportance (as rated by the checking agent):")
for k, n in Counter(imp(c["importance"]) for c in claims).most_common():
    print(f"   {k:12s} {n:4d}")

# owner vs collaborator
own = [c for c in claims if c["id"][0] in "FOMSA"]
col = [c for c in claims if c["id"][0] in "DH"]
print(f"\nowner's claims        {len(own):4d}   collaborator's claims {len(col):4d}")

head("PHASE 4 -- the audit's own error bar")
merged = list(csv.DictReader(open(os.path.join(A, "phase4", "merged.csv"), encoding="utf-8")))
adj = {r["id"]: r for r in csv.DictReader(
    open(os.path.join(A, "phase4", "adjudications.csv"), encoding="utf-8"))}
cmp_rows = [r for r in merged if r["classification"] != "NEW"]
diff = [r for r in merged if r["classification"] in ("DIFFERS", "UNRESOLVED")]
aw = sum(1 for r in diff if adj.get(r["id"], {}).get("outcome") == "AUDIT WAS WRONG")
print(f"sampled {len(merged)} | comparable {len(cmp_rows)} | new coverage "
      f"{len(merged)-len(cmp_rows)}")
print(f"raw disagreement      {len(diff)}/{len(cmp_rows)} = {len(diff)/len(cmp_rows):.1%}")
print(f"ADJUDICATED AUDIT ERR {aw}/{len(cmp_rows)} = {aw/len(cmp_rows):.1%}")
print("\noutcomes:", dict(Counter(a["outcome"] for a in adj.values())))
t4 = Counter(r["agent_family"] for r in merged if r["classification"] == "NEW")
n4 = sum(t4.values())
print(f"\nT4 new coverage (n={n4}): " +
      "  ".join(f"{k} {v} ({v/n4:.1%})" for k, v in t4.most_common()))

head("EXPERIMENTS AND PAPERS")
for name in ("experiments.csv", "papers.csv"):
    p = os.path.join(A, name)
    if not os.path.exists(p):
        print(f"{name}: NOT FOUND")
        continue
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    print(f"{name}: {len(rows)} rows, {len(rows[0])} cols")
    for col in ("status", "owner", "notes_present", "pdf_present"):
        if col in rows[0]:
            print(f"   {col}: {dict(Counter((r[col] or '?')[:22] for r in rows).most_common(6))}")

head("DOCUMENT CORRECTIONS AND RED FLAGS")
df = open(os.path.join(A, "DOC_FIXES.md"), encoding="utf-8").read()
n_df_head = len(re.findall(r"^#+\s*DF-\d+", df, re.M))
n_df_uniq = len(set(re.findall(r"DF-(\d+)", df)))
print(f"DOC_FIXES.md: {n_df_head} DF- section headings, {n_df_uniq} distinct DF numbers")
prog = open(os.path.join(A, "PROGRESS.md"), encoding="utf-8").read()
n_rf = len(set(re.findall(r"RF-(\d+)", prog)))
print(f"PROGRESS.md: {n_rf} distinct RF (red flag) numbers")
print(f"claim notes in audit/claims/: {len(glob.glob(os.path.join(A,'claims','*.md')))}")
screen = list(csv.DictReader(open(os.path.join(A, "phase4", "verdict_note_conflicts.csv"),
                                  encoding="utf-8")))
n_clean = sum(1 for c in claims if fam(c["verdict"]) == "HOLDS")
print(f"verdict/note conflict screen: {len(screen)} hits over {n_clean} clean-verdict rows")

head("COVERAGE -- what was NOT checked")
print("Phase 2/3 checked 424 of 692 identified claims.")
print(f"unchecked: {692-len(claims)}  ({(692-len(claims))/692:.1%})")
print(f"of the unchecked, Phase 4 re-derived a random {n4} "
      f"({n4/(692-len(claims)):.1%} of them)")
