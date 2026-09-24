"""
Phase 2, batch P8-draft-rest — two follow-up checks.

(a) D-233 "Appendix G reproduction": the draft says the pooled activations were
    rescored with the pipeline's stored probe and "matched the pipeline's own
    results table to four decimal places on all five Liars' Bench datasets",
    and that the same held at the follow-up token.
    Only ONE pipeline results table is committed on his branch
    (XXXX-9-llama-all-followup/results_table.csv).  This compares it, cell by
    cell, with the follow-up store he analysed from
    (persona_ood_followup_L20/zero_shot_operating.csv, arm "base probe
    (RepE-trained)", threshold_source "control").

(b) D-157 "0.5 floor in 18 of 30 cells": reconciles summary_balacc.csv's
    cells_at_floor=18 with the 14 cells that are exactly 0.5 in
    matrix_balacc_baseline.csv, by sweeping the tolerance.

Read-only.  Usage: python P8_appG_and_floor.py <results-dir>
"""
import csv
import os
import sys

ROOT = sys.argv[1]
LB5 = ["convince_me__plain", "harm_refusal_choice__plain",
       "harm_refusal_knowledge_report__plain", "insider_trading_cadenza__plain",
       "instruct_dishonesty__plain"]
ALL6 = LB5 + ["repe_honesty__plain"]
SHORT = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C",
         "harm_refusal_knowledge_report__plain": "HP-KR",
         "insider_trading_cadenza__plain": "IT",
         "instruct_dishonesty__plain": "ID", "repe_honesty__plain": "RepE"}

def rows(p):
    with open(os.path.join(ROOT, p), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

print("=" * 78)
print("(a) D-233 -- pipeline results table vs the store, FOLLOW-UP run")
print("=" * 78)
pipe = {r[""]: r for r in rows("XXXX-9-llama-all-followup/results_table.csv")}
store = {r["dataset"]: r for r in rows("persona_ood_followup_L20/zero_shot_operating.csv")
         if r["arm"] == "base probe (RepE-trained)" and r["threshold_source"] == "control"}
print(f"{'dataset':<8} {'metric':<10} {'pipeline':<22} {'store':<10} {'4dp match':<9}")
worst = 0.0
for d in LB5:
    p, s = pipe[d], store[d]
    for pk, sk, lab in (("auroc", "auroc", "AUROC"),
                        ("balanced_accuracy_at_fpr_1%", "balacc", "balacc"),
                        ("recall 1.0%", "recall", "recall"),
                        ("fpr 1.0%", "specificity", "1-spec")):
        pv = float(p[pk])
        sv = float(s[sk])
        if lab == "1-spec":
            sv = round(1 - sv, 4)
        diff = abs(round(pv, 4) - sv)
        worst = max(worst, diff)
        print(f"{SHORT[d]:<8} {lab:<10} {pv!r:<22} {sv:<10} {'YES' if diff < 5e-5 else 'NO ('+str(diff)+')'}")
print(f"worst |pipeline rounded to 4dp - store| over the 20 cells: {worst}")
print("Datasets present in the pipeline table:", sorted(pipe.keys()))

print()
print("Pipeline dirs on his branch that DO / DO NOT carry a results_table.csv:")
for d in sorted(os.listdir(ROOT)):
    if d.startswith("XXXX-9"):
        has = os.path.exists(os.path.join(ROOT, d, "results_table.csv"))
        print(f"   {d:<62} results_table.csv: {'YES' if has else 'NO'}")

print()
print("=" * 78)
print("(b) D-157 -- what counts as 'at the 0.5 floor'")
print("=" * 78)
with open(os.path.join(ROOT, "persona_ood_ext_L40/matrix_balacc_baseline.csv"),
          newline="", encoding="utf-8") as f:
    r = csv.reader(f)
    hdr = next(r)[1:]
    m = {line[0]: {h: float(v) for h, v in zip(hdr, line[1:])} for line in r}
off = [(m[tr][te], SHORT[tr], SHORT[te]) for tr in ALL6 for te in ALL6 if tr != te]
for tol in (0.0, 1e-4, 1e-3, 2e-3, 5e-3, 1e-2):
    n = sum(1 for v, _, _ in off if abs(v - 0.5) <= tol)
    print(f"   tolerance |balacc - 0.5| <= {tol:<7} -> {n} of 30 cells")
print("   his summary_balacc.csv records cells_at_floor = "
      + [r["cells_at_floor"] for r in rows("persona_ood_ext_L40/summary_balacc.csv")
         if r["features"] == "baseline"][0])
print("   the four cells that are near but not at 0.5:",
      [(v, a, b) for v, a, b in sorted(off) if 0.0 < abs(v - 0.5) <= 0.005])
print("\nDONE")
