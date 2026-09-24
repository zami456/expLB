"""Apply the Phase 4 ledger fixes to audit/claims.csv and the claim notes.

The brief's Phase 4: "...Report the mismatch rate, FIX THE LEDGER, and log each
fix in audit/CORRECTIONS.md."

Deferred until Phase 4 closed so the ledger was never edited while it was the
object of measurement. Sources of truth:
  audit/phase4/adjudications.csv  - outcome + ledger_fix per id
  audit/CORRECTIONS.md            - the reasoning for each

Backs up claims.csv first, is idempotent (re-running changes nothing), and
prints a diff of every field it touches.

  python audit/scripts/p4_apply_ledger_fixes.py --dry-run
  python audit/scripts/p4_apply_ledger_fixes.py
"""
import argparse
import csv
import os
import shutil
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLAIMS = os.path.join(ROOT, "audit", "claims.csv")
NOTES = os.path.join(ROOT, "audit", "claims")
STAMP = f"[PHASE 4 {date.today().isoformat()}]"

# id -> new verdict
VERDICT = {
    "F-01": "PARTLY HOLDS",
    "F-70": "PARTLY HOLDS",
    "O-022": "PARTLY HOLDS",
    "M-007": "PARTLY HOLDS",
    "M-054": "PARTLY HOLDS",
    "S-25": "PARTLY HOLDS",
    "D-028": "PARTLY MATCHES HIS DATA (Level 1; aggregates re-derived)",
}

# id -> sentence appended to `notes`
NOTE = {
    "F-01": "VERDICT CORRECTED BY PHASE 4 (HOLDS -> PARTLY HOLDS). The defect was already stated in "
            "this note and the verdict cell did not follow it; an independent blinded re-derivation "
            "graded it PARTLY on the same evidence. F-26 was graded PARTLY for a structurally "
            "identical fault. See audit/CORRECTIONS.md.",
    "F-70": "VERDICT CORRECTED BY PHASE 4 (HOLDS -> PARTLY HOLDS). '1 of 384' is in this claim's own "
            "stated_numbers and was routed away as 'checked elsewhere' rather than checked. Phase 4 "
            "found it matches NO significance family anywhere (git paper16 root 192 tests; the four "
            "git bank stores 96 each; lab copies 4,032-6,300) - consistent with Phase 1's finding "
            "that F-04's 384-test run was overwritten the same evening. '6 of 4,032' does reproduce "
            "exactly in the git copy of run-crossgen/kd11/banks/kd/significance.json. The stale "
            "caveat (3) flagged below is a second sentence-level defect under a clean verdict.",
    "O-022": "VERDICT CORRECTED BY PHASE 4 (HOLDS -> PARTLY HOLDS). All five numbers reproduce to 12+ "
             "s.f. and are not in dispute; the sentence was never graded. COLLABORATOR_BRIEFING.md:354 "
             "labels 0.676 '(what the abstract compares against)', but the abstract's comparator is "
             "0.7320325942 - the `baseline` row of persona_ood_ext_L40/summary.csv on his branch, over "
             "HIS datasets (30 pairs), against our 0.676 over 11 datasets (110 pairs). Two different "
             "corpora; the document says so nowhere. Sibling claim O-021 records the same defect and "
             "was also graded HOLDS.",
    "M-007": "VERDICT CORRECTED BY PHASE 4 (HOLDS -> PARTLY HOLDS). The two comparability caveats are "
             "stated in this note ('the paper panel is the 3B DECEPTION axis at k=1 on 5 datasets with "
             "HPC excluded, set beside our UNIFIED axis at k=2/k=5 on 10 datasets') and the verdict "
             "cell did not follow them. This is the k=1-vs-k=5 problem the Phase 4 plan lists as a "
             "named trap.",
    "M-054": "VERDICT CORRECTED BY PHASE 4 (HOLDS -> PARTLY HOLDS). The probe mismatch is stated in "
             "this note ('as written it compares two different estimators and should name them') and "
             "the verdict cell did not follow it. The conclusion survives with either probe held "
             "fixed; the printed sentence still compares two estimators.",
    "S-25": "VERDICT CORRECTED BY PHASE 4 (HOLDS -> PARTLY HOLDS). The claim's PREMISE is wrong: the "
            "two PCA bases over an identical role set do not 'differ by up to 1.347' - they are "
            "identical up to a per-component SIGN FLIP, cos = -1.0000 exactly at the argmax "
            "(L77, PC4, dim 5114). Element-wise max-abs-difference between PCA bases is not a distance "
            "and is quoted as one. The conclusion (compare AUROCs, not components) is better supported "
            "than the claim states. Consistent with RF-15 on PC sign pinning.",
    "D-028": "VERDICT CORRECTED BY PHASE 4 (MATCHES -> PARTLY MATCHES). This note already opened "
             "'AUDIT FLAG: the headline 0.220 is HP-KR-driven' and the verdict cell did not follow it. "
             "Phase 4 measured it: dropping HP-KR as train AND test collapses the layer-20 gap from "
             "+0.220 to +0.009 (0.9086 vs 0.9000), because the PC3 probe inverts on HP-KR. This is the "
             "draft's p.2 Contributions bullet, and HP-KR is the dataset the extended bank was "
             "purpose-built to repair.",
    "F-58": "PHASE 4 AMENDMENT (verdict unchanged). 'No surviving evidence' is too strong. The "
            "crossgen logs are APPEND-MODE, so the 2026-09-20 22:18-22:21 block persists beneath the "
            "23:38-23:40 mask re-run that overwrote the matrices; the 3B/8B rows were recovered from "
            "them (copies kept at audit/phase4/evidence_F2/crossgen_scale_{3b,8b,70b}.log). The matrix "
            "FILES are gone; the numbers are not. Phase 4 also separated three further defects: 3B/8B "
            "used harm-pressure-choice while the 70B row used mask (so 'the paper's five datasets' "
            "cannot be true of both); 3B/8B are k=3 and the 70B row k=5; and within the 70B row "
            "'random' is k=5 while 'datapca' is datapca@2.",
    "F-45": "PHASE 4 AMENDMENT (verdict unchanged; every number re-derived exactly). Provenance caveat: "
            "the 8B extraction directory no longer holds the run F-45 measured - every dataset present "
            "is now capped at 800 rows, i.e. the entry's own recommendation was adopted and the "
            "193-chunk partial run replaced the same evening. SEPARATE PROJECT RED FLAG found here: "
            "5 of the 12 dataset dirs in run-scale/8b/coords have NO rows.jsonl at all "
            "(harm-pressure-knowledge-report, harm-pressure-kr-matched, insider-trading, "
            "sandbagging_v2, sycophancy_mt); only 3,093 non-alpaca rows across 7 datasets. Anything "
            "assuming a complete 8B corpus needs re-checking.",
    "S-33": "PHASE 4 AMENDMENT - RESOLVES AUDIT SELF-CORRECTION #6. That correction said two "
            "candidate explanations for F-61's '+0.006' both fit and 'the audit cannot separate them', "
            "because the 'raw 0.6800 over 109 pairs' reading survives 'in no copy'. It survives in no "
            "copy because NO FILE EVER HELD IT: it was manufactured at read time by a script bug. "
            "kd/scripts/paper_comparison.py originally took its axis labels from d['rows'] / "
            "d['columns'], which are human captions ('trained on (80% split)', 22 chars), not label "
            "lists - so zip iterated CHARACTERS and the diagonal test compared single letters, leaving "
            "most in-distribution cells in the 'off-diagonal' mean. Lead reproduced both code paths on "
            "run-crossgen/kd11/matrices/mean_L40_s42.json: fixed = 110 pairs / raw 0.6623 / "
            "pc:unified +0.0093; buggy = 109 pairs / raw 0.6800 / +0.0061. F-61's headline '+0.006' is "
            "the bug artifact; like-for-like is +0.0093 and the ratio is 8.1x, not twelvefold. The bug "
            "is already fixed in the committed script.",
    "D-237": "PHASE 4 AMENDMENT (verdict unchanged; the named script is confirmed absent from both "
             "branches, every commit reachable by git log --all --diff-filter=A, the working tree, and "
             "the lab PC). METHODS NOTE: this is a COMPOUND claim - NDIF and CPU-only hold, the "
             "reproducibility limb is falsified. The audit failed the whole claim; an independent "
             "blinded reader graded it PARTLY, failing the limb. The verdict vocabulary has no rule "
             "for compound claims; this is a gap in the vocabulary, not an error by either reader.",
    "O-041": "PHASE 4 AMENDMENT (verdict unchanged). METHODS NOTE: this claim is about an EVENT (what "
             "two reviewers said), not a measurement, so 'raw data' would mean the review outputs "
             "themselves - which are not in the repo (no reviews folder). An independent blinded "
             "re-derivation correctly returned NO RAW DATA. Such claims cannot be scored either way by "
             "a raw-data check; they are attested, not reproduced. Phase 4 adds that the two papers "
             "attributed to Reviewer B are not in papers/, so the citation is itself unverifiable.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(CLAIMS, encoding="utf-8")))
    fields = list(rows[0].keys())
    changed = skipped = 0

    for r in rows:
        rid = r["id"]
        if rid not in NOTE:
            continue
        marker = f"{STAMP}"
        if marker in (r.get("notes") or ""):
            skipped += 1
            continue
        if rid in VERDICT:
            print(f"  {rid:7s} verdict: {r['verdict'][:46]!r}")
            print(f"  {'':7s}       -> {VERDICT[rid]!r}")
            r["verdict"] = VERDICT[rid]
        else:
            print(f"  {rid:7s} verdict unchanged ({r['verdict'][:40]}), note amended")
        r["notes"] = ((r.get("notes") or "").rstrip() + f"  {marker} " + NOTE[rid]).strip()
        changed += 1

    print(f"\n{changed} row(s) to change, {skipped} already applied (idempotent).")
    if args.dry_run:
        print("DRY RUN - nothing written.")
        return
    if changed == 0:
        print("Nothing to do.")
        return

    bak = CLAIMS + ".before-phase4-fixes"
    if not os.path.exists(bak):
        shutil.copy2(CLAIMS, bak)
        print(f"backup -> {os.path.relpath(bak, ROOT)}")
    with open(CLAIMS, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.relpath(CLAIMS, ROOT)}")

    # append the same amendment to each claim note, if one exists
    for rid in NOTE:
        p = os.path.join(NOTES, f"{rid}.md")
        if not os.path.exists(p):
            print(f"  (no note file for {rid})")
            continue
        body = open(p, encoding="utf-8").read()
        if STAMP in body:
            continue
        verdict_line = (f"\n**Verdict corrected to `{VERDICT[rid]}`.**\n" if rid in VERDICT else "")
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(f"\n\n---\n\n## {STAMP} amendment\n{verdict_line}\n{NOTE[rid]}\n\n"
                     f"Adjudication and evidence: `audit/CORRECTIONS.md`, "
                     f"`audit/phase4/adjudications.csv`.\n")
        print(f"  amended note {rid}.md")


if __name__ == "__main__":
    main()
