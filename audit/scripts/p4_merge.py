"""Phase 4 merge: join the agents' independent re-derivations to the unblinded
sample and classify every row AGREES / DIFFERS / UNRESOLVED / NEW.

The lead runs this. Agents never see sample.csv.

  python audit/scripts/p4_merge.py            # merge whatever rederive_*.csv exist
  python audit/scripts/p4_merge.py --strict   # fail if any sampled id is missing

Writes audit/phase4/merged.csv and prints the mismatch rate by tier.
"""
import argparse
import csv
import glob
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P4 = os.path.join(ROOT, "audit", "phase4")

# Both vocabularies collapse onto these four families before comparison.
HOLDS, PARTLY, FAILS, UNRES = "HOLDS", "PARTLY", "FAILS", "UNRESOLVED"


def norm_prior(v):
    """Phase 2/3 verdict -> family."""
    s = (v or "").strip().upper()
    if not s or s == "(UNCHECKED)":
        return None                      # T4: nothing to compare against
    if s.startswith("PARTLY"):
        return PARTLY
    if s.startswith("DOES NOT"):
        return FAILS
    if s.startswith("REVISED"):
        return FAILS                     # the claim as written was superseded
    if s.startswith("MATCHES HIS DATA"):
        # "...as a restatement; NOT TESTED as a general claim" is a partial
        return PARTLY if "NOT TESTED" in s else HOLDS
    if s.startswith("HOLDS"):
        return HOLDS
    if s.startswith("NOT TESTED") or s.startswith("NO RAW DATA") or s.startswith("CONFLICT"):
        return UNRES
    return UNRES


def norm_agent(v):
    """Phase 4 agent verdict -> family."""
    s = (v or "").strip().upper()
    if not s:
        return UNRES
    if s.startswith("PARTLY"):
        return PARTLY
    if s.startswith("DOES NOT"):
        return FAILS
    if s.startswith("HOLDS"):
        return HOLDS
    # NO RAW DATA, CANNOT RESOLVE, anything unexpected
    return UNRES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    sample = {r["id"]: r for r in csv.DictReader(
        open(os.path.join(P4, "sample.csv"), encoding="utf-8"))}

    derived, dupes = {}, []
    batch_of = {}
    for path in sorted(glob.glob(os.path.join(P4, "rederive_*.csv"))):
        batch = os.path.basename(path)[len("rederive_"):-len(".csv")]
        for r in csv.DictReader(open(path, encoding="utf-8")):
            rid = (r.get("id") or "").strip()
            if not rid:
                continue
            if rid in derived:
                dupes.append((rid, batch_of[rid], batch))
            derived[rid] = r
            batch_of[rid] = batch

    unknown = sorted(set(derived) - set(sample))
    missing = sorted(set(sample) - set(derived))

    out_rows = []
    for rid, s in sample.items():
        d = derived.get(rid)
        prior = norm_prior(s["prior_verdict"])
        mine = norm_agent(d["my_verdict"]) if d else None

        if d is None:
            cls = "NOT-YET-DONE"
        elif prior is None:
            cls = "NEW"                                   # T4, no prior verdict
        elif prior == UNRES or mine == UNRES:
            cls = "UNRESOLVED"
        elif prior == mine:
            cls = "AGREES"
        else:
            cls = "DIFFERS"

        out_rows.append({
            "tier": s["tier"],
            "id": rid,
            "batch": batch_of.get(rid, ""),
            "classification": cls,
            "prior_verdict": s["prior_verdict"],
            "prior_family": prior or "",
            "agent_verdict": (d or {}).get("my_verdict", ""),
            "agent_family": mine or "",
            "agent_confidence": (d or {}).get("my_confidence", ""),
            "stated_numbers": s["stated_numbers"],
            "rederived_value": (d or {}).get("rederived_value", ""),
            "evidence_path": (d or {}).get("evidence_path", ""),
            "evidence_key": (d or {}).get("evidence_key", ""),
            "agent_notes": (d or {}).get("notes", ""),
            "claim_plain": s["claim_plain"],
        })

    out_rows.sort(key=lambda r: (r["tier"], r["id"]))
    out = os.path.join(P4, "merged.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # ---- report -------------------------------------------------------
    print(f"merged {len(derived)} re-derivations against {len(sample)} sampled claims")
    print(f"-> {os.path.relpath(out, ROOT)}\n")

    if dupes:
        print(f"!! {len(dupes)} id(s) appear in more than one batch:")
        for rid, a, b in dupes:
            print(f"   {rid}: {a} and {b}")
        print()
    if unknown:
        print(f"!! {len(unknown)} re-derived id(s) are NOT in the sample: {unknown}\n")
    if missing:
        print(f"   {len(missing)} sampled id(s) not yet re-derived"
              f"{': ' + ', '.join(missing) if len(missing) <= 30 else ''}\n")

    by_tier = defaultdict(Counter)
    for r in out_rows:
        by_tier[r["tier"]][r["classification"]] += 1

    hdr = ["AGREES", "DIFFERS", "UNRESOLVED", "NEW", "NOT-YET-DONE"]
    print(f"{'tier':26s} " + " ".join(f"{h:>12s}" for h in hdr) + "   mismatch rate")
    tot = Counter()
    for tier in sorted(by_tier):
        c = by_tier[tier]
        tot.update(c)
        cmp_n = c["AGREES"] + c["DIFFERS"]
        rate = f"{c['DIFFERS'] / cmp_n:6.1%} ({c['DIFFERS']}/{cmp_n})" if cmp_n else "     n/a"
        print(f"{tier:26s} " + " ".join(f"{c[h]:12d}" for h in hdr) + f"   {rate}")
    cmp_n = tot["AGREES"] + tot["DIFFERS"]
    rate = f"{tot['DIFFERS'] / cmp_n:6.1%} ({tot['DIFFERS']}/{cmp_n})" if cmp_n else "     n/a"
    print(f"{'ALL':26s} " + " ".join(f"{tot[h]:12d}" for h in hdr) + f"   {rate}")

    # ---- instruction-group split -------------------------------------
    # F1/F2/F3/O1 ran BEFORE the verdict-discipline paragraph was added to the
    # agent prompt; every later batch ran after it. The paragraph pushes agents
    # towards PARTLY, which makes them agree with the audit more often -- i.e.
    # it flatters the audit. Reported so the bias is visible, not hidden.
    PRE = {"F1", "F2", "F3", "O1"}
    print("\ninstruction group (see audit/CORRECTIONS.md -- this biases the RAW rate):")
    for label, sel in (("A pre-discipline (F1,F2,F3,O1)", lambda b: b in PRE),
                       ("B post-discipline (all others)", lambda b: b and b not in PRE)):
        c = Counter(r["classification"] for r in out_rows if sel(r["batch"]))
        n = c["AGREES"] + c["DIFFERS"]
        rate = f"{c['DIFFERS'] / n:6.1%} ({c['DIFFERS']}/{n})" if n else "     n/a"
        print(f"   {label:34s} AGREES {c['AGREES']:3d}  DIFFERS {c['DIFFERS']:3d}  "
              f"NEW {c['NEW']:3d}   raw mismatch {rate}")

    print("\nNEW-coverage verdicts (T4, the previously unchecked corpus):")
    t4 = Counter(r["agent_family"] for r in out_rows
                 if r["classification"] == "NEW" and r["agent_family"])
    for k, v in t4.most_common():
        print(f"   {v:4d}  {k}")

    if args.strict and (missing or unknown or dupes):
        sys.exit(1)


if __name__ == "__main__":
    main()
