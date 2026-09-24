"""Append the second P2-pooled batch of rows (F-85, F-86, F-95, F-109) to the batch CSV."""
import csv
import os

P = "audit/phase2/claims_batch_P2-pooled.csv"
COLS = ["id", "claim_plain", "stated_numbers", "source_ref", "evidence_file", "evidence_key",
        "level1_value", "level1_match", "level2_value", "level2_method", "version_used",
        "robustness", "verdict", "confidence", "importance", "script", "notes"]

ROWS = [
    dict(
        id="F-85",
        claim_plain="On six datasets at L20/L40 (mean, paper, seed 42) pooled LODO reaches 0.7365/0.8683 and row-budget-matched 0.7654/0.8930, so the abstract's zero-shot contrast lead shrinks from +0.184 (vs its own single-source baseline) to +0.048 (vs our pooled LODO) to +0.023 (vs row-matched) at L40.",
        stated_numbers="0.6439/0.6960 single; 0.7365/0.8683 LODO; 0.7654/0.8930 matched; 0.9630/0.9862 pooled-with-target; 0.9852 own; +0.184/+0.048/+0.023; +0.139 at L20; 0.811 our contrast",
        source_ref="kd/docs/FINDINGS.md:3403 (ledger already marks it SUPERSEDED BY F-97, headline only)",
        evidence_file="audit/cache/labpc/run-premise/six/premise_L20_mean_s42.json; .../premise_L40_mean_s42.json",
        evidence_key="summary.{single,pooled_lodo,pooled_lodo_match,pooled_full,single_own_mean}; datasets; n_train",
        level1_value="L20 0.643890/0.736533/0.765431/0.962983; L40 0.696040/0.868340/0.892972/0.986211/0.985159; margins vs 0.916: +0.0477 and +0.0230",
        level1_match="YES (all |d|<=4e-5)",
        level2_value="dataset-level bootstrap on the six, never run before: L40 pooled_lodo-single +0.1723 [+0.1253,+0.2239] P=0.0000 CLEARS; L20 +0.0926 [-0.0450,+0.2102] P=0.0887 DOES NOT CLEAR (row-matched at L20 also does not: +0.1215 [-0.0507,+0.2294])",
        level2_method="paired dataset bootstrap, unit=target dataset, n=6, B=10000, seed 42, same procedure as 30b_premise_boot.py",
        version_used="lab copies 2026-09-21 17:18:01/17:18:13, md5 == 20:56 census, NOT in the RF-13 wave, covariance=within; lab-only store (RF-10)",
        robustness="F-109 adds L32 (higher than L40 on all three arms); the row-matched-beats-full oddity is +0.0289 (L20), +0.0045 (L32), +0.0246 (L40)",
        verdict="PARTLY HOLDS",
        confidence="HIGH on the numbers; MEDIUM on the framing verdict because it depends on F-97 (another batch)",
        importance="HIGH",
        script="audit/scripts/claims/p2p_six_why_falsify.py",
        notes="The '+0.184' is 0.916-0.732, both the collaborator's; our own single-source on the six at L40 is 0.6960, against which the same contrast leads by +0.220. NEW: at L20 - the layer F-85 calls the claim's strongest - our pooled-vs-single margin does NOT clear the bootstrap (P=0.089). The cross-reference '0.811 our contrast on the same six' comes from run-crossgen/kd11/banks/kd_s23 and was not re-derived in this batch."),
    dict(
        id="F-86",
        claim_plain="Averaging the ten unit-normalised single-source probe directions scores 0.8828 at L40, beating pooled joint training (0.8545, +0.028), budget-matched pooling (0.8062, +0.077) and single-source transfer (0.6763, +0.207); single-source directions are near-orthogonal (cos +0.051) while the pooled direction sits nearer them (+0.227), and the source-count curve rises and flattens.",
        stated_numbers="0.6763; 0.8062; 0.8545; 0.8828; +0.028; +0.077; +0.207; cos 0.051 / 0.227; curve 0.673/0.740/0.782/0.815/0.820/0.864/0.809",
        source_ref="kd/docs/FINDINGS.md:3467",
        evidence_file="audit/cache/labpc/run-premise/why/why_L40_s42.json; audit/cache/labpc/run-premise/audit/audit_L40_s42.json",
        evidence_key="mean_of_directions; pooled_lodo_matched; single_offdiag; cos_*; curve_vs_n_sources; pooled_lodo_match_draws; bootstrap.mean_of_directions-pooled_lodo_full",
        level1_value="0.676316 / 0.806211 / 0.882830; cos 0.050599 / 0.227173; margins +0.0284 / +0.0766 / +0.2065; curve k=1..10 0.6729/0.7404/0.7815/0.7178/0.8148/0.8161/0.8196/0.8281/0.8644/0.8091",
        level1_match="YES for every stated number",
        level2_value="comparator's 20 subsampling draws: mean 0.8344, range [0.7912,0.8736], sd 0.0200, max-min 0.0824; 30f's own printed verdict: margin INSIDE that spread. Stored dataset bootstrap for the +0.0284 margin: CI [+0.0045,+0.0523] (clears, lower bound 0.0045). real-real_unnorm = +0.0089.",
        level2_method="read the 20 stored draws and the stored bootstrap from 30f_result2_audit.py's output file; re-applied the script's own comparison rule",
        version_used="why (2026-09-21 17:24:36) and audit (20:59:35) lab files, both md5 == 20:56 census, neither in the RF-13 wave; no bank involved so RF-12/15/16 do not apply",
        robustness="F-95 sweeps the 0.8828 number over 21 layers x 3 seeds and it survives; the MARGIN over pooled training was never swept. F-86's caveat (3) 'no dataset bootstrap yet' is out of date - audit_L40_s42.json has one.",
        verdict="PARTLY HOLDS",
        confidence="HIGH on the numbers and the conflict; MEDIUM on how far to discount the +0.028 (the 0.0824 spread is of the budget-matched arm, not of the full arm the margin is measured against)",
        importance="HIGH",
        script="audit/scripts/claims/p2p_six_why_falsify.py",
        notes="THREE issues. (1) The +0.0284 'beats joint training' margin is smaller than its comparator's own 0.0824 subsampling spread; 30f_result2_audit.py printed exactly that sentence and F-86 does not carry it; the dataset bootstrap clears with a lower bound of only +0.0045. (2) NAME COLLISION: 'pooled LODO row-budget matched' is 0.8062 here but 0.8500 in F-79/F-80 at the same layer/pooling/probe/seed (two different scripts' budget implementations); against 0.8500 the margin is +0.033, not +0.077. (3) The source-count table omits k=4=0.7178, which is BELOW k=3=0.7815 and breaks the 'rising and flattening' description; k=6 and k=8 are also omitted."),
    dict(
        id="F-95",
        claim_plain="The mean-of-directions result survives a three-null falsification battery over 21 layers x 3 seeds: all three nulls sit on chance (0.5052/0.5015/0.5041), 55 of 63 cells clear the best oriented null by >0.02, 51 of 51 at layers >=16 and 4 of 12 below, the eight failures being L0/L4/L8; peak is L36 (0.8916) not L40 (0.8872); real-budget10 averages +0.0171.",
        stated_numbers="0.5052 [0.4840,0.5337]; 0.5015 [0.4799,0.5283]; 0.5041 [0.4848,0.5214]; 55/63; 51/51; 4/12; +0.0171 [-0.022,+0.067]; L36 0.8916 vs L40 0.8872; min gap -0.0221 (mis-specified gate)",
        source_ref="kd/docs/FINDINGS.md:4225",
        evidence_file="audit/cache/labpc/run-premise/falsify/falsify_L*_s*.json (63); audit/cache/labpc/logs/falsify.log; logs/falsify.status",
        evidence_key="real; budget10; {gauss,cov,shuffled}.{raw,oriented}",
        level1_value="nulls 0.5053 / 0.5019 / 0.5041 with the stated ranges exactly; 55/63, 51/51, 4/12; failing cells (0,42)(0,7)(0,13)(4,42)(4,7)(4,13)(8,42)(8,7); peak L36 0.8916; real-budget10 +0.0170 [-0.0218,+0.0673]; min gap over all 63 = -0.0221",
        level1_match="YES everywhere except the L40 row (see level2/notes)",
        level2_value="recomputed the gate quantity: min gap at layers >=16 is +0.1021, five times the 0.02 convention, so SURVIVES is correct and the logged FALSIFIED was the mis-specified gate. Also real-real_unnorm = +0.0089 at L40/s42 (F-95's 'worth +0.009').",
        level2_method="re-derived the oriented gap per cell as max(gauss.oriented,cov.oriented,shuffled.oriented) exactly as run_falsify.ps1 does, then recounted; parsed the UTF-16 falsify.log to recover the original 2026-09-21 run",
        version_used="62 of 63 files md5-identical to the 20:56 census; falsify_L40_s13.json was REWRITTEN 2026-09-23 02:54 by the orphaned run_falsify.ps1 -Force (PID 48932) the audit's own RF-14 live test left running - which is still running (52/63 at 03:17)",
        robustness="the shuffled null - the only one that could kill the result - lands at 0.5053; the 0.02 clear threshold is a convention with no test behind it (F-95 says so); surviving this battery does not rescue F-86's +0.028 margin over pooled training",
        verdict="HOLDS",
        confidence="HIGH - every count and null reproduces; the single discrepancy is 0.002 in size and traced to a file the audit's own interrupted call overwrote",
        importance="HIGH",
        script="audit/scripts/claims/p2p_six_why_falsify.py",
        notes="L40's null (0.7057 -> 0.7045) and budget10 (0.8663 -> 0.8684) differ because falsify_L40_s13.json was rewritten today; 'real' is unchanged (0.8830) and the L40 gap moves +0.1814 -> +0.1826, changing nothing. The original cell came from a pilot invocation (L20/L40 x 3 seeds, 2026-09-21 20:28-20:34) whose log block is truncated, so its null/budget10 cannot be recovered. FOR THE LEAD: the orphan process is still rewriting run-premise/falsify through L76."),
    dict(
        id="F-109",
        claim_plain="On the same six datasets the premise run at L32 gives single-source 0.7203, pooled LODO 0.8965 and row-matched 0.9010, beating L40 on all three arms (+0.024/+0.028/+0.008), which cuts F-85's margin against the abstract's 0.916 from +0.023 to +0.015.",
        stated_numbers="0.6439/0.7365/0.7654 (L20); 0.7203/0.8965/0.9010 (L32); 0.6960/0.8683/0.8930 (L40); +0.024/+0.028/+0.008; +0.015",
        source_ref="kd/docs/FINDINGS.md:5210",
        evidence_file="audit/cache/labpc/run-premise/six/premise_L32_mean_s42.json (with the L20/L40 files)",
        evidence_key="summary.{single,pooled_lodo,pooled_lodo_match}",
        level1_value="0.720334 / 0.896529 / 0.900992; deltas vs L40 +0.0243 / +0.0282 / +0.0080; 0.916-0.9010 = +0.0150",
        level1_match="YES (all |d|<=3.4e-5)",
        level2_value="dataset bootstrap on the six at L32: pooled_lodo-single +0.1762 [+0.1140,+0.2361] P=0.0000; row-matched-single +0.1807 [+0.1160,+0.2447] P=0.0000 - both clear, unlike L20",
        level2_method="paired dataset bootstrap n=6 B=10000 seed 42, plus a header-field comparison (datasets, n_train 873/5238/544, probe, pooling, seed, covariance) against the L20/L40 files to confirm the three layers are the same configuration",
        version_used="lab file mtime 2026-09-22 16:55:53 - written the same evening as run-premise/clustered/, AFTER the RF-13 reboot wave, not during it; md5 == 20:56 census",
        robustness="the L32 six-dataset margin clears the bootstrap where L20's does not; the row-matched-beats-full oddity is smallest here (+0.0045)",
        verdict="HOLDS",
        confidence="HIGH",
        importance="MEDIUM-HIGH",
        script="audit/scripts/claims/p2p_six_why_falsify.py",
        notes="Inherits F-85/F-97's dataset mismatch (our six substitute ai_liar for the collaborator's repe_honesty), so the +0.015 is still not like-for-like against his abstract."),
]

new = not os.path.exists(P)
with open(P, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    if new:
        w.writeheader()
    for r in ROWS:
        w.writerow(r)
print("wrote", P, "rows", len(ROWS))
