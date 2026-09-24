"""Phase 4 — select the targeted re-derivation sample.

The brief specifies "every HIGH-importance claim plus a random ~20% of the rest", which
at the observed importance ratings is 279 re-derivations.  The owner chose the targeted
version instead (2026-09-23), for two reasons recorded here so the choice is auditable:

  * 243 of 424 checked claims carry a HIGH rating assigned by the batch agent that checked
    them, and more than half the corpus being HIGH is rating inflation, not importance;
  * the effort is better spent where a wrong verdict would change the thesis, plus a
    genuine random sample to bound the residual error rate.

STRATA (seeded, re-runnable):

  T1  every outright FAILURE                 — is each one real?  A wrong failure is as
      (DOES NOT HOLD / DOES NOT MATCH /        damaging as a wrong pass: it would have the
       REVISED)                                owner retract something true.
  T2  every claim the lead QA'd              — the load-bearing ones by construction,
                                               and the place the audit already erred
  T3  random sample of HIGH-importance HOLDS — FALSE-NEGATIVE risk: a wrong HOLDS puts a
                                               bad number into the thesis unchallenged
  T3b random sample of HIGH-importance PARTLY— did the verdict stop at the wording when the
                                               number was also wrong?
  T4  random 20% of the UNCHECKED            — the only stratum with no coverage at all;
                                               bounds the 268 claims Phase 2 never reached

PARTLY verdicts are NOT taken wholesale (there are 119).  A PARTLY means the number
reproduced and the sentence around it failed, which is the audit's most common and
best-evidenced verdict; a sample of the HIGH-importance ones tests it adequately.

Read-only.  Writes audit/phase4/sample.csv.
"""

import collections
import csv
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHK = ROOT / "audit" / "claims.csv"
INV = ROOT / "audit" / "claims_inventory.csv"
QA = ROOT / "audit" / "phase2" / "qa_wave5.md"
OUT = ROOT / "audit" / "phase4" / "sample.csv"

SEED = 42
N_T3 = 40          # HIGH-importance HOLDS
N_T3B = 30         # HIGH-importance PARTLY
FRAC_T4 = 0.20     # of the unchecked


def is_failure(v):
    return v.strip().upper().startswith(("DOES NOT", "REVISED"))


def norm(cid):
    return cid[2:] if cid.startswith("L-F-") else cid


def is_holds(v):
    return v.strip().upper().startswith(("HOLDS", "MATCHES"))


def is_high(r):
    return (r.get("importance") or "").strip().upper().startswith(("HIGH", "MEDIUM-HIGH"))


def main():
    chk = list(csv.DictReader(CHK.open(encoding="utf-8")))
    inv = list(csv.DictReader(INV.open(encoding="utf-8")))
    done = {r["id"] for r in chk}
    unchecked = [r for r in inv if norm(r["id"]) not in done]

    # every id named in the lead's QA record
    qa_text = QA.read_text(encoding="utf-8") if QA.exists() else ""
    qa_ids = {r["id"] for r in chk
              if r["id"] in qa_text or f"**{r['id']}**" in qa_text}

    rng = random.Random(SEED)
    picked, why = {}, {}

    def add(rows, tier):
        for r in rows:
            if r["id"] not in picked:
                picked[r["id"]] = r
                why[r["id"]] = tier

    t1 = [r for r in chk if is_failure(r["verdict"])]
    add(t1, "T1-outright-failure")

    t2 = [r for r in chk if r["id"] in qa_ids]
    add(t2, "T2-lead-QA'd")

    pool3 = [r for r in chk if is_holds(r["verdict"]) and is_high(r)
             and r["id"] not in picked]
    t3 = rng.sample(pool3, min(N_T3, len(pool3)))
    add(t3, "T3-HIGH-holds-sample")

    pool3b = [r for r in chk if not is_holds(r["verdict"])
              and not is_failure(r["verdict"]) and is_high(r)
              and r["id"] not in picked]
    t3b = rng.sample(pool3b, min(N_T3B, len(pool3b)))
    add(t3b, "T3b-HIGH-partly-sample")

    n4 = round(len(unchecked) * FRAC_T4)
    t4 = rng.sample(unchecked, n4)
    for r in t4:
        picked[r["id"]] = {"id": r["id"], "claim_plain": r.get("claim_short", ""),
                           "source_ref": r["source_ref"], "verdict": "(unchecked)",
                           "importance": "", "stated_numbers": r.get("numbers", "")}
        why[r["id"]] = "T4-unchecked-random20"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["tier", "id", "source_ref", "prior_verdict", "prior_importance",
            "stated_numbers", "claim_plain"]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for cid, r in sorted(picked.items(), key=lambda kv: (why[kv[0]], kv[0])):
            w.writerow([why[cid], cid, r.get("source_ref", ""), r.get("verdict", ""),
                        r.get("importance", "")[:40], r.get("stated_numbers", "")[:80],
                        (r.get("claim_plain") or "")[:200]])

    # BLINDED copy — this is the one handed to the batch agents.  The brief requires Phase 4
    # agents to get "only this brief, the ledger row and the raw files, not your notes", so the
    # prior verdict, prior importance and tier are withheld: an agent that knows the audit said
    # HOLDS is no longer an independent check of HOLDS.
    blind = OUT.with_name("sample_blinded.csv")
    with blind.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "source_ref", "stated_numbers", "claim_plain"])
        for cid, r in sorted(picked.items()):
            w.writerow([cid, r.get("source_ref", ""), r.get("stated_numbers", "")[:80],
                        (r.get("claim_plain") or "")[:200]])
    print(f"wrote {blind}  (agent-facing: no prior verdict, no tier)")

    print(f"checked {len(chk)} · unchecked {len(unchecked)} · seed {SEED}")
    for t, n in collections.Counter(why.values()).most_common():
        print(f"  {t:26} {n}")
    print(f"  {'TOTAL':26} {len(picked)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
