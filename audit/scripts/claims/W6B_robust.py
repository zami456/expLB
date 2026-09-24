"""W6B-thesis2 robustness checks that the claims themselves do not mention.

1. O-087  the zero-shot deception-axis figure 0.863 at L40 mean:
   - against the 1,000-random-direction null (run-coords/zeroshot_nulls.json),
     pooled AND per dataset;
   - and the SECOND pooled value for the same quantity, 0.881, which comes from
     a corpus that keeps alpaca.  The two producing scripts differ:
     12_dotproduct_layers.py drops single-class datasets, 25_zeroshot_nulls.py
     keeps them in the pooled set.
2. O-089 / O-090  the LODO Pass C rows: per-fold spread behind the fold mean.
3. O-092 / O-093  the raw / persona medians, recomputed after dropping the five
   zero-shot rows and after dropping the six Pass C LODO rows, to see whether
   the split drives the result.
4. O-096  the hf275 column of the memory note's contamination table, measured
   from the local hf vectors (the own100 column is lab-only).

Read-only.  Run from the repo root.
"""
from __future__ import annotations

import json
import pathlib
import statistics

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
RC = ROOT / "run-coords"


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


hdr("1. O-087  zero-shot deception axis, L40 mean -- null test and the two corpora")
z = json.load(open(RC / "zeroshot_nulls.json", encoding="utf-8"))
dp = json.load(open(RC / "dotproducts_mean.json", encoding="utf-8"))
l40 = next(m["contrast"] for m in dp["matched"] if m["layer"] == 40)
c = z["corpus_L40_mean"]
print("dotproducts_mean.json  (12_dotproduct_layers.py, single-class datasets "
      "DROPPED): pooled AUROC %.6f  <-- the 0.863 the thesis prints" % l40)
print("zeroshot_nulls.json    (25_zeroshot_nulls.py, alpaca KEPT in the pooled "
      "set):     pooled AUROC %.6f" % c["pooled"]["deception_axis"]["auroc"])
print("  difference %.4f -- same axis, same layer, same pooling, different "
      "pooled corpus" % (c["pooled"]["deception_axis"]["auroc"] - l40))
print("  pooled vs 1,000 random directions: p %.4f, random p95 %.4f, "
      "beats_random_p95 %s"
      % (c["pooled"]["deception_axis"]["p_vs_random"],
         c["pooled"]["random_p95_auroc_dev"],
         c["pooled"]["deception_axis"]["beats_random_p95"]))
print("\n  per dataset at L40 mean (%d labelled datasets):" % len(c["per_dataset"]))
print("  %-34s %8s %10s %8s" % ("dataset", "AUROC", "rand p95", "beats?"))
nb = 0
for k, v in c["per_dataset"].items():
    a = v["deception_axis"]
    print("  %-34s %8.4f %10.4f %8s"
          % (k, a["auroc"], v["random_p95_auroc_dev"], a["beats_random_p95"]))
    nb += bool(a["beats_random_p95"])
print("  beats the random p95 null on %d of %d datasets" % (nb, len(c["per_dataset"])))
print("  -> the pooled 0.863 is therefore partly a between-dataset effect")

hdr("2. O-089 / O-090  per-fold spread behind the Pass C LODO means")
qt = json.load(open(RC / "q8_trained_pertoken.json", encoding="utf-8"))
for probe, agg in (("pool_mean_mms", "pooled"), ("tok_lr", "mean")):
    for L in (20, 22, 40, 44):
        e = next(r for r in qt["results"]
                 if r["probe"] == probe and r["aggregation"] == agg and r["layer"] == L)
        aur = [f["auroc"] for f in e["per_fold"]]
        worst = min(e["per_fold"], key=lambda f: f["auroc"])
        print("  %-14s %-7s L%-2d  mean %.4f  folds %d  min %.4f (%s)  max %.4f  sd %.4f"
              % (probe, agg, L, e["auroc_mean"], len(aur), min(aur),
                 worst["held_out"], max(aur), statistics.pstdev(aur)))

hdr("3. O-092 / O-093  do the raw / persona medians survive dropping a sub-population?")
q6 = json.load(open(RC / "q6_layers.json", encoding="utf-8"))


def persona(m):
    return m.split("/")[0].strip() not in (
        "raw", "tok_lr", "tok_mms", "pool_mean_mms", "pool_mean_lr", "pool_last_mms")


rows = q6["rows"]
raw = [(r["method"], r["source"], r["gap_40_minus_best_of_20_22"])
       for r in rows if not persona(r["method"])]
per = [(r["method"], r["source"], r["gap_40_minus_best_of_20_22"])
       for r in rows if persona(r["method"])]
print("  raw     : n=%d median %+.4f" % (len(raw), statistics.median([g for *_, g in raw])))
print("  persona : n=%d median %+.4f" % (len(per), statistics.median([g for *_, g in per])))
sub = [g for m, s, g in raw if "Pass C" not in s]
print("  raw without the 6 Pass C LODO rows      : n=%d median %+.4f" % (len(sub), statistics.median(sub)))
sub = [g for m, s, g in raw if "Pass C" in s]
print("  raw, Pass C LODO rows only              : n=%d median %+.4f" % (len(sub), statistics.median(sub)))
sub = [g for m, s, g in per if "zero-shot" not in s and "leave-one" not in s]
print("  persona, trained rows only              : n=%d median %+.4f" % (len(sub), statistics.median(sub)))
sub = [g for m, s, g in per if "zero-shot" in s or "leave-one" in s]
print("  persona, zero-shot rows only            : n=%d median %+.4f" % (len(sub), statistics.median(sub)))

hdr("4. O-096  the hf275 contamination column, measured locally "
    "(own100 column is lab-only)")
h = torch.load(ROOT / "run-hf275/axes_hf100.pt", map_location="cpu", weights_only=False)
pub = torch.load(ROOT / "run-hf275/published_assistant_axis.pt",
                 map_location="cpu", weights_only=False).float()
d = h["axes"]["deception"].float()
own_a = h["axes"]["assistant"].float()
print("  hf bank counts:", h["counts"], " vectors_dir:", h["vectors_dir"])
# the memory note persona-axis-diagnostics.md:91-96 table (DOC-ONLY input)
note_own100 = {20: -0.729, 22: -0.710, 40: -0.660, 60: -0.573}
note_hf275 = {20: -0.686, 22: -0.662, 40: -0.610, 60: -0.516}
print("  %6s %14s %14s %14s %12s" % ("layer", "hf275 measured", "hf275 in note",
                                     "own100 in note", "rel. gap"))
gaps = []
for L in (20, 22, 40, 60):
    m = F.cosine_similarity(d[L], pub[L], dim=-1).item()
    g = (abs(note_own100[L]) - abs(note_hf275[L])) / abs(note_own100[L]) * 100
    gaps.append(g)
    print("  %6d %14.4f %14.3f %14.3f %11.1f%%" % (L, m, note_hf275[L], note_own100[L], g))
print("  hf275 measured vs the note: max |diff| %.4f"
      % max(abs(F.cosine_similarity(d[L], pub[L], dim=-1).item() - note_hf275[L])
            for L in (20, 22, 40, 60)))
print("  relative own100-vs-hf275 gap implied by the note: %.1f%% .. %.1f%% "
      "(the doc says '~7-9%%')" % (min(gaps), max(gaps)))
print("  NOTE: the own100 column cannot be measured here -- the own100 bank is "
      "C:/Users/user4/expLB/run/axes.pt on the lab PC only.")
print("  NOTE: the hf275 column is measured against the PUBLISHED Assistant "
      "Axis; against the hf bank's OWN assistant direction it reads "
      + ", ".join("L%d %.4f" % (L, F.cosine_similarity(d[L], own_a[L], dim=-1).item())
                  for L in (20, 22, 40, 60)))
