"""W6B-thesis2 : O-081 (7.5 Q5) and O-095 / O-096 (7.7 Q7).

O-081  docs/thesis/persona-coordinates-70b-results.md:704
       "The pooled within/across gap looks like clustering by construct
        (0.748 vs 0.585, p 0.043)."
       -> run-coords/q5_semantics_L44_last.json, the raw/mms row.

O-095  :757  "Score-3 rate: 80.7% vs 79.9%, a gap of 0.8 percentage points."
O-096  :762  "Verdict: 7a. Judges are interchangeable here."  (+ agreement 91.2%,
             1,231 of 1,350, and the two score distributions at :756)
       -> recomputed from run-paper16/scores/*.json (Llama-3.3-70B judge) and
          run-paper16/scores_405b/*.json (Llama-3.1-405B judge).  No producing
          analysis script exists in the repo for these numbers, so there is no
          Level-1 result file: this is a recompute from the raw judge output.

Read-only.  Run from the repo root.
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]


def md5(p: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def q5():
    p = ROOT / "run-coords/q5_semantics_L44_last.json"
    print("=" * 78)
    print("O-081  --  7.5 Q5 pooled within/across gap")
    print("=" * 78)
    print("file :", p, "| %d B | md5 %s | mtime %s"
          % (p.stat().st_size, md5(p),
             datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(" ", "seconds")))
    d = json.load(open(p, encoding="utf-8"))
    print("source matrix:", d["source"])
    print("semantics map: %d datasets" % len(d["semantics"]))
    counts = collections.Counter(d["semantics"].values())
    print("  by construct:", dict(counts))
    print()
    print("%-14s %-9s %10s %6s %10s %6s %10s %8s"
          % ("features", "probe", "within", "n_w", "across", "n_a", "difference", "perm p"))
    for r in d["results"]:
        star = " <== doc row" if (r["features"] == "raw" and r["probe"] == "mms") else ""
        print("%-14s %-9s %10.6f %6d %10.6f %6d %10.6f %8.6f%s"
              % (r["features"], r["probe"], r["within_semantics"], r["n_within"],
                 r["across_semantics"], r["n_across"], r["difference"],
                 r["permutation_p"], star))
    row = next(r for r in d["results"]
               if r["features"] == "raw" and r["probe"] == "mms")
    print()
    print("LEVEL 1  doc: 0.748 vs 0.585, p 0.043")
    print("         file: %.6f vs %.6f, p %.6f  -> rounded %.3f / %.3f / %.3f"
          % (row["within_semantics"], row["across_semantics"],
             row["permutation_p"], round(row["within_semantics"], 3),
             round(row["across_semantics"], 3), round(row["permutation_p"], 3)))
    ok = (round(row["within_semantics"], 3) == 0.748
          and round(row["across_semantics"], 3) == 0.585
          and round(row["permutation_p"], 3) == 0.043)
    print("         MATCH" if ok else "         MISMATCH")
    print("LEVEL 2  difference re-derived: %.6f - %.6f = %.6f (stored %.6f)"
          % (row["within_semantics"], row["across_semantics"],
             row["within_semantics"] - row["across_semantics"], row["difference"]))
    # the sentence that follows the claim
    blocks = row.get("blocks", {})
    if blocks:
        print("LEVEL 2  the same row's block means (the 7.5 table):")
        for k, v in blocks.items():
            if isinstance(v, dict):
                print("           %-34s n=%-4s mean %.6f"
                      % (k, v.get("n"), v.get("mean", float("nan"))))
            else:
                print("           %-34s %s" % (k, v))
    print("LEVEL 2  companion sentence '42 of its 44 within-cells are "
          "instructed_intent pairs':")
    sem = d["semantics"]
    src = json.load(open(ROOT / row.get("source_file", d["source"]).replace("\\", "/"),
                         encoding="utf-8")) if False else None
    n_by = collections.Counter(sem.values())
    print("           datasets per construct in the map: %s" % dict(n_by))
    print("           ordered off-diagonal pairs if ALL mapped datasets were in "
          "the matrix: %s"
          % {k: v * (v - 1) for k, v in n_by.items()})
    print("           stored n_within = %d, n_across = %d (total %d = 10x9, so "
          "the matrix has 10 of the 11 mapped datasets)"
          % (row["n_within"], row["n_across"], row["n_within"] + row["n_across"]))
    print("           => within = intent(k) pairs + belief(2) pairs; k*(k-1)+2 = "
          "%d requires k = 7 instructed_intent datasets present, i.e. 42 of 44 "
          "within-cells are intent->intent." % row["n_within"])


def q7():
    print()
    print("=" * 78)
    print("O-095 / O-096  --  7.7 Q7 judge sensitivity, recomputed from the "
          "raw judge output")
    print("=" * 78)
    d70 = ROOT / "run-paper16/scores"
    d405 = ROOT / "run-paper16/scores_405b"
    f70 = sorted(d70.glob("*.json"))
    f405 = sorted(d405.glob("*.json"))
    print("70B  judge dir : %s  (%d files)" % (d70, len(f70)))
    print("405B judge dir : %s  (%d files)" % (d405, len(f405)))
    print("personas only in 70B :", sorted({p.stem for p in f70} - {p.stem for p in f405}))
    print("personas only in 405B:", sorted({p.stem for p in f405} - {p.stem for p in f70}))

    s70, s405 = {}, {}
    for p in f70:
        for k, v in json.load(open(p, encoding="utf-8")).items():
            s70[(p.stem, k)] = v
    for p in f405:
        for k, v in json.load(open(p, encoding="utf-8")).items():
            s405[(p.stem, k)] = v
    print("\nscored items: 70B %d, 405B %d, common %d"
          % (len(s70), len(s405), len(set(s70) & set(s405))))
    print("  (paper16 = %d personas x %d items)"
          % (len(f70), len(s70) // max(1, len(f70))))

    common = sorted(set(s70) & set(s405))
    agree = sum(1 for k in common if s70[k] == s405[k])
    print("\nEXACT SCORE AGREEMENT on the common items: %d / %d = %.4f%%"
          % (agree, len(common), 100.0 * agree / len(common)))
    print("   doc says 91.2%% (1,231 of 1,350)")

    c70 = collections.Counter(s70[k] for k in common)
    c405 = collections.Counter(s405[k] for k in common)
    print("\nscore distribution over the %d common items" % len(common))
    print("   70B : %s" % dict(sorted(c70.items())))
    print("   405B: %s" % dict(sorted(c405.items())))
    print("   doc : 70B 0:12, 2:249, 3:1089 | 405B 0:19, 1:2, 2:250, 3:1079")

    r70 = c70[3] / len(common)
    r405 = c405[3] / len(common)
    print("\nSCORE-3 RATE   70B %.5f (%.1f%%)   405B %.5f (%.1f%%)   gap %.2f pp"
          % (r70, 100 * r70, r405, 100 * r405, 100 * (r70 - r405)))
    print("   doc says 80.7%% vs 79.9%%, gap 0.8 percentage points")

    # per-persona agreement (the doc's 'lowest agreement' bullet)
    per = collections.defaultdict(lambda: [0, 0])
    for k in common:
        per[k[0]][1] += 1
        if s70[k] == s405[k]:
            per[k[0]][0] += 1
    print("\nper-persona exact agreement (lowest 6):")
    for name, (a, n) in sorted(per.items(), key=lambda t: t[1][0] / t[1][1])[:6]:
        print("   %-24s %4d/%-4d = %.4f" % (name, a, n, a / n))
    print("   doc: handler 0.74, plausible_denier 0.76, fact_checker 0.78")

    # which items the vector filter keeps (score == 3) and how they differ
    keep70 = {k for k in common if s70[k] == 3}
    keep405 = {k for k in common if s405[k] == 3}
    print("\nvector filter (keep score-3): 70B keeps %d, 405B keeps %d, "
          "intersection %d, symmetric difference %d (%.2f%% of items)"
          % (len(keep70), len(keep405), len(keep70 & keep405),
             len(keep70 ^ keep405),
             100.0 * len(keep70 ^ keep405) / len(common)))

    print("\nO-096 second sentence: 'The earlier ~7-9%% own100-vs-hf275 gap "
          "confounded the judge with a different generation run.'")
    print("   own100 rollouts/scores are LAB-ONLY (C:/Users/user4/expLB/run);")
    print("   nothing in the repo working tree holds an own100 score-3 rate, so "
          "the 7-9%% figure cannot be checked from local data.")


if __name__ == "__main__":
    q5()
    q7()
