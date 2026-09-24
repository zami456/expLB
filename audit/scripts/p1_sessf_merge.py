"""Phase 1 / transcript mining: merge the four batch CSVs into
audit/phase1/sessions/findings_printed_numbers.csv, sorted by F-number then time.

    python audit/scripts/p1_sessf_merge.py
"""
import csv
import glob
import os
import re

COLS = ["f_id", "f_claim_short", "found", "when", "session", "line",
        "printed_snippet", "matches_ledger", "producing_command"]
OUT = os.path.join("audit", "phase1", "sessions", "findings_printed_numbers.csv")


def key(r):
    m = re.search(r"(\d+)", r.get("f_id", ""))
    return (int(m.group(1)) if m else 999, r.get("when", ""), r.get("line", ""))


def main():
    rows = []
    parts = sorted(glob.glob(os.path.join("audit", "phase1", "sessions",
                                          "findings_printed_numbers_p*.csv")))
    for p in parts:
        with open(p, encoding="utf-8") as f:
            n = 0
            for r in csv.DictReader(f):
                rows.append({c: (r.get(c) or "").strip() for c in COLS})
                n += 1
        print("%-60s %d rows" % (p, n))
    rows.sort(key=key)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    ids = sorted({r["f_id"] for r in rows}, key=lambda s: int(re.search(r"(\d+)", s).group(1)))
    found = {}
    for r in rows:
        v = r["found"].lower()
        found[v] = found.get(v, 0) + 1
    ml = {}
    for r in rows:
        v = r["matches_ledger"].split(";")[0].strip().lower()
        ml[v] = ml.get(v, 0) + 1
    print("\nmerged %d rows, %d distinct F-numbers -> %s" % (len(rows), len(ids), OUT))
    print("found:", found)
    print("matches_ledger (first token):", ml)

    # coverage against the target list
    tgt = os.path.join("audit", "phase1", "sessions", "findings_targets.csv")
    if os.path.exists(tgt):
        want = [r["f_id"] for r in csv.DictReader(open(tgt, encoding="utf-8"))]
        missing = [x for x in want if x not in ids]
        print("targets %d; covered %d; missing %s" % (len(want), len(want) - len(missing), missing or "none"))


if __name__ == "__main__":
    main()
