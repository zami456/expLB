"""Census for the ONE error mode Phase 4 actually found.

All five confirmed audit errors have the same shape: the checker wrote the defect
into `notes` and then graded the claim `HOLDS`. That is machine-detectable, so it
does not need a sample -- it can be swept over all 424 checked claims.

This script flags every row whose verdict is a clean HOLDS (or MATCHES HIS DATA)
but whose notes contain language describing a defect. It is a SCREEN, not a
verdict: every hit must be read by the lead. Precision matters less than recall
here, so the patterns are deliberately broad.

  python audit/scripts/p4_verdict_note_conflict.py            # summary + top hits
  python audit/scripts/p4_verdict_note_conflict.py --csv      # write the full list

Writes audit/phase4/verdict_note_conflicts.csv
"""
import argparse
import csv
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Phrases that assert a defect in the claim. Tuned against the five confirmed
# cases (F-01, F-70, O-022, M-007, M-054) plus the PARTLY rows, which should
# score high -- they are the positive control.
DEFECT = [
    (r"\bomits?\b|\bomitted\b", "omits"),
    (r"should (?:name|state|say|be)\b", "should-state"),
    (r"\bdoes not (?:say|state|name|match|reproduce|hold|carry|follow)\b", "does-not"),
    (r"\bnot stated\b|\bunstated\b|\bnever says?\b|\bsays so nowhere\b", "unstated"),
    (r"\bmisleading\b|\bmislead\b|\binvites the wrong\b", "misleading"),
    (r"\bstale\b|\bsuperseded\b|\bout of date\b", "stale"),
    (r"\bunderstates?\b|\boverstates?\b|\boverclaim", "over/understates"),
    (r"\bmisscoped\b|\bwrong row.?set\b|\bwrong (?:direction|panel|axis|bank|probe|layer|seed)\b", "wrong-scope"),
    (r"\bcontradicts?\b|\bcontradiction\b", "contradicts"),
    (r"\bunsupported\b|\bnot supported\b", "unsupported"),
    (r"\bmismatch\b|\bcollision\b", "mismatch"),
    (r"\bcaveats? the .{0,30}omits\b|\bcaveat\b.{0,40}\bstale\b", "stale-caveat"),
    (r"\bis wrong\b|\bare wrong\b|\bwrongly\b", "is-wrong"),
    (r"\bdefects?\b|\bfaults?\b|\berrors?\b", "defect-word"),
    (r"\bcherry.?pick", "cherry-picked"),
    (r"\bunderspecified\b|\bambiguous\b|\bunderdetermined\b", "underspecified"),
    (r"\bdo(?:es)? not bracket\b|\boutside the stated\b|\bbelow the stated\b|\babove the stated\b", "out-of-range"),
]

# Verdicts that assert the claim is clean.
CLEAN = re.compile(r"^(HOLDS|MATCHES HIS DATA)", re.I)
# ...but these are explicitly NOT clean.
NOT_CLEAN = re.compile(r"PARTLY|DOES NOT|REVISED|NOT TESTED|NO RAW", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(ROOT, "audit", "claims.csv"), encoding="utf-8")))

    hits, clean_n, partly_n, partly_hits = [], 0, 0, 0
    for r in rows:
        v = (r.get("verdict") or "").strip()
        notes = (r.get("notes") or "") + " " + (r.get("robustness") or "")
        tags = sorted({t for pat, t in DEFECT if re.search(pat, notes, re.I)})

        is_clean = bool(CLEAN.match(v)) and not NOT_CLEAN.search(v)
        if is_clean:
            clean_n += 1
            if tags:
                hits.append({
                    "id": r["id"],
                    "verdict": v,
                    "n_tags": len(tags),
                    "tags": "|".join(tags),
                    "importance": (r.get("importance") or "")[:40],
                    "source_ref": r.get("source_ref", ""),
                    "notes": notes.strip()[:600],
                })
        elif re.search(r"PARTLY", v, re.I):
            partly_n += 1
            if tags:
                partly_hits += 1

    hits.sort(key=lambda h: -h["n_tags"])

    print(f"claims.csv rows            : {len(rows)}")
    print(f"clean-verdict rows         : {clean_n}")
    print(f"  ...with defect language  : {len(hits)}  <-- SCREEN HITS, each needs a human read")
    print(f"PARTLY rows (positive ctrl): {partly_n}, of which {partly_hits} carry defect language "
          f"({partly_hits / partly_n:.0%})" if partly_n else "")
    print()
    print("tag frequency among hits:")
    tc = Counter(t for h in hits for t in h["tags"].split("|"))
    for t, c in tc.most_common():
        print(f"   {c:4d}  {t}")

    print("\nthe five already CONFIRMED by Phase 4 adjudication, as a sanity check:")
    known = {"F-01", "F-70", "O-022", "M-007", "M-054"}
    for h in hits:
        if h["id"] in known:
            print(f"   {h['id']:7s} {h['n_tags']}  {h['tags']}")
    missed = known - {h["id"] for h in hits}
    if missed:
        print(f"   NOT CAUGHT by the screen: {sorted(missed)}  <-- recall is imperfect; note it")

    print(f"\ntop {min(25, len(hits))} hits by tag count:")
    for h in hits[:25]:
        print(f"   {h['id']:7s} {h['n_tags']}  {h['tags'][:60]:62s} {h['source_ref'][:38]}")

    if args.csv:
        out = os.path.join(ROOT, "audit", "phase4", "verdict_note_conflicts.csv")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(hits[0].keys()))
            w.writeheader()
            w.writerows(hits)
        print(f"\nwrote {os.path.relpath(out, ROOT)} ({len(hits)} rows)")


if __name__ == "__main__":
    main()
