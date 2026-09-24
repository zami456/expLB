"""Correct claims.csv row D-107 — a correction that runs in the COLLABORATOR's favour.

Flagged by P3-A, verified twice by the lead.

The row said his random control was "UNDERPOWERED (n=1 draw)" and that a one-draw
baseline was "quoted in Table 3 as if it were a null distribution".  The first half
is false.  `persona_ood.py` on origin/feat/persona-pc-ood:

    N_RANDOM_DRAWS = 5                                              # line 61
    # random control, averaged over draws so a single lucky basis cannot carry
    # the comparison                                                # line 1004
    for _ in range(N_RANDOM_DRAWS):                                 # line 1006
        dirs = random_directions(hidden, 3, rng)
        random_mats.append(transfer_matrix(...))
    matrices["random3"] = sum(random_mats) / len(random_mats)       # line 1009

PROGRESS.md already records this being overturned during the P8 lead QA; the
claims.csv row was never updated to match.  The narrower criticism does stand and
is preserved: he reports the MEAN of 5 draws, not a null distribution with a
percentile, so it bounds the centre and not the tail.

Idempotent.  Writes only audit/claims.csv.
"""

import csv
import pathlib

P = pathlib.Path(__file__).resolve().parents[3] / "audit" / "claims.csv"

NEW_VERDICT = ("MATCHES HIS DATA (Level 1 only); the control is the MEAN of 5 seeded draws, "
               "not 1 - it bounds the centre, not the tail")
NEW_CONF = "HIGH for the number and for the draw count (read from his source)"
NEW_NOTES = ("CORRECTED 2026-09-23 (P3-A flag, lead-verified twice, in the collaborator's favour). "
             "The earlier 'UNDERPOWERED (n=1 draw)' was FALSE: persona_ood.py:61 sets "
             "N_RANDOM_DRAWS = 5 and :1004-1009 averages the draws, with the comment 'averaged "
             "over draws so a single lucky basis cannot carry the comparison'. It has done so "
             "since the script's first version (2731cb0, 2026-09-18). The narrower criticism "
             "stands: he reports the MEAN of 5 draws rather than a percentile of a null "
             "distribution, so it bounds the centre and not the tail - the owner's own "
             "1,000-direction null does more. docs/DRAFT_CORRECTIONS.md section 4 already "
             "states this correctly; this ledger row did not.")


def main():
    rows = list(csv.DictReader(P.open(encoding="utf-8")))
    fields = rows[0].keys()
    hit = 0
    for r in rows:
        if r["id"] == "D-107":
            hit += 1
            r["verdict"] = NEW_VERDICT
            r["confidence"] = NEW_CONF
            r["notes"] = NEW_NOTES
    if hit != 1:
        raise SystemExit(f"expected exactly one D-107 row, found {hit}")
    with P.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)
    print(f"D-107 corrected in {P} ({len(rows)} rows rewritten)")


if __name__ == "__main__":
    main()
