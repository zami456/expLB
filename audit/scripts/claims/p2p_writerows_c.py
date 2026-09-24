"""Append the last two P2-pooled rows (F-94, F-96) to the batch CSV."""
import csv
import os

P = "audit/phase2/claims_batch_P2-pooled.csv"
COLS = ["id", "claim_plain", "stated_numbers", "source_ref", "evidence_file", "evidence_key",
        "level1_value", "level1_match", "level2_value", "level2_method", "version_used",
        "robustness", "verdict", "confidence", "importance", "script", "notes"]

ROWS = [
    dict(
        id="F-94",
        claim_plain="Over 21 layers on sycophancy_mt (paper probe, seed 42), swapping the six holding personas for topically unrelated ones (content effect) costs +0.0356 on average and rarely clears the per-layer dataset bootstrap, while removing them entirely (presence effect) costs +0.0730 and usually does - about 2:1 under mean pooling and ~1:1 under last-token, so F-89's '78%' and 'three-to-one' figures should not be quoted.",
        stated_numbers="16-layer table L0-L56; content +0.0339/+0.0356; presence +0.0704/+0.0730; retention median 53%/51%, 22-93%, -278% at L12; content clears 4/16 and 3/21; presence 9/16 and 15/21 (13 consecutive L28-L76); L32 +0.056 / +0.170; last pooling +0.0346/+0.0373, peak 0.659 at L32 vs 0.752; retentions -349% and +2387%",
        source_ref="kd/docs/FINDINGS.md:4107",
        evidence_file="lab C:/Users/user4/expLB/run-crossgen/kd11/banks/{cap_s23,cap_ctrl}/matrices/{mean,last}_L*_s42.json (90 files, read-only on the lab)",
        evidence_key="results[feature=pc:capitulation@2 / pc:cap_vs_para@2 / pc:cap_vs_neutral@2 / pc:cap_only_ctrl@2 / raw, probe=paper].auroc, sycophancy_mt column",
        level1_value="all 16 tabulated rows reproduce at 4dp (e.g. L32 0.7521/0.7558/0.6961/0.5825/0.6134; L40 0.7136/0.7229/0.6889/0.5935/0.6013); all per-layer content/presence values and all * marks reproduce; means +0.0339/+0.0704 (16) and +0.0356/+0.0730 (21); retention medians 53% and 51%; last pooling +0.0346/+0.0373, peak 0.6592 at L32, retentions -349% and +2387%",
        level1_match="YES for every measured quantity; NO for three prose counts (see notes)",
        level2_value="content clears 3/21 at L0,L12,L36 and 3/16; presence clears 15/21 (13 consecutive L28-L76) and 10/16; ordering strict 10/16, 'holding>neutral>none' 12/16, group reading 16/16",
        level2_method="read the raw 11x11 AUROC matrices on the lab, took the sycophancy_mt column mean over the 10 sources (diagonal excluded), and re-derived every * with a paired bootstrap over the 10 sources (B=10000, seed 42). Retention formula reverse-engineered as (neutral-raw)/(holding-raw), confirmed on three published values (24% at L0, -278% at L12, 78% at L40).",
        version_used="all 90 cap matrices have lab mtimes on 2026-09-21; none in the 09-22 RF-13 wave. Both banks record bank=axes_s23.pt, which provenance.csv flags AMBIGUOUS; resolved by the launch command in run_cap_ctrl_layers.ps1 (--bank run-cap-ctrl/axes_s23.pt for cap_ctrl). k=5, frac_train=0.8, control=alpaca.",
        robustness="both poolings measured and agree in direction not size; one seed and one target throughout; the bootstrap unit is 10 source datasets, the smallest in the project (F-87's cap seed spread is of the same order as the content effect)",
        verdict="PARTLY HOLDS",
        confidence="HIGH - recomputed from the raw matrices with the bootstrap re-derived and the retention formula confirmed on three published values",
        importance="HIGH",
        script="audit/scripts/claims/p2p_F94_lab.py",
        notes="THREE prose counts do not reproduce. (1) 'content clears at only 4 of 16 layers' -> 3 of 16 (L0, L12, L36); the entry's own table marks 3 and its 21-layer amendment says 3, so the 4 contradicts itself. (2) 'presence clears at 9 of 16' -> 10 of 16; the entry's own table marks 10. (3) 'the ordering paraphrase>=holding>neutral>none holds at 15 of 16 layers (L8 the single inversion)' is not reproducible under any reading: 10/16 strict, 12/16 ignoring paraphrase, 16/16 as groups; the inversions are tiny (<=0.011) so the qualitative ladder survives. Minor: 'raw 0.47-0.50 from L44 upward' - L76 raw under last pooling is 0.5055."),
    dict(
        id="F-96",
        claim_plain="F-63's three-rung persona margin is a single-layer reading: layer-averaged, best(axes,pc:deception) minus dataset-PCA is +0.1197 (3B, 7 layers) / +0.1027 (8B, 8) against F-63's +0.2023 / +0.1147 at L16, L16 being the maximum of 3B's seven; and against raw with the feature held fixed the gain is negative at both small rungs (-0.0063, -0.0190).",
        stated_numbers="+0.2023 / +0.1197 / [+0.014,+0.202] / 1st of 7; +0.1147 / +0.1027 / [+0.030,+0.166] / 4th of 8; +0.0794 at L40; axes-raw +0.0195/-0.0063/4-of-7, -0.0249/-0.0190/2-of-8, +0.0131; oracle best-raw +0.033 (7/7) and +0.009 (6/8); minimum +0.007; F-91's kd_s23 axes peak L44 0.6435, L40 0.6304, +0.0131",
        source_ref="kd/docs/FINDINGS.md:4317",
        evidence_file="audit/phase2/layer_sensitivity.csv (W1, built from lab run-crossgen/scale_{3b,8b,70b} and scale_70b_paper16fp32)",
        evidence_key="store, pooling=mean, probe=paper, dataset_set=paper5, seed=42, k=3, feature in {axes, pc:deception, datapca, raw}, offdiag_mean",
        level1_value="3B: L16 +0.2023, layer-mean +0.1197, range [+0.0136,+0.2023], rank 1 of 7; 8B: L16 +0.1147, layer-mean +0.1027, range [+0.0297,+0.1660], rank 4 of 8; 70B L40 +0.0794; axes-raw 3B +0.0195/-0.0063/4-of-7, 8B -0.0249/-0.0190/2-of-8, 70B L40 +0.0131; best-raw 3B +0.0328 (7/7), 8B +0.0089 (6/8); kd_s23 axes peak L44 0.6435, L40 0.6304, +0.0131",
        level1_match="YES - every stated number reproduces, including both ranks",
        level2_value="the parenthetical 'minimum +0.007' is axes-datapca (fixed feature, not best) at 3B L12 = +0.0067; 8B's minimum on the same reading is +0.0077. Under the 'best' reading the minimum is +0.0136.",
        level2_method="recomputed every margin, layer-mean, range, positive-count and rank from the W1 per-layer off-diagonal means; searched the alternative readings (datapca vs datapca@2, best over 2 vs 3 features, both poolings, three probes) to locate the +0.007",
        version_used="layer_sensitivity.csv built 2026-09-23 02:55 from the lab stores. provenance.csv: scale_3b -> axes_3b.pt (paper16_3b, k=3); scale_8b -> axes_8b.pt (paper16_8b, k=3); scale_70b -> axes_vectors_s3.pt flagged AMBIGUOUS (that is RF-16). F-96's own numbers use the L40 paper16 cell and the 3B/8B paper16 banks, so RF-16 does not touch them (F-110 agrees: 'F-63 is unaffected').",
        robustness="the stated reading is mean pooling + paper probe; under mms the layer-means are +0.113/+0.099 and under lr_reg10 +0.115/+0.103, so the conclusion is probe-robust; under last pooling the 8B layer-mean collapses to +0.009",
        verdict="HOLDS",
        confidence="HIGH - every number reproduces from the shared W1 table, including both ranks and the parenthetical minimum",
        importance="HIGH",
        script="audit/scripts/claims/p2p_F96.py",
        notes="ONE CAVEAT IS NOW STALE. F-96's caveat (1) says 'the 70B rung is still one layer; scale_70b holds only L40 at k=3'. scale_70b now holds 21 k=3 layers (the 2026-09-22 00:52-01:33 sweep, written after F-96). Across them the 70B layer-mean is +0.0791 on the kd bank as swept and +0.0550 like-for-like on the RF-16 repair store, so the monotone decline continues (+0.120 -> +0.103 -> +0.055) and L40 ranks 13th of 21. On the extended sweep the sentence 'the persona axis beats the matched control at every layer of every rung' fails at L0 (axes-datapca -0.0251 kd / -0.0184 paper16fp32; positive 20/21 and 18/21). This overlaps F-110/RF-16, which are other claims."),
]

new = not os.path.exists(P)
with open(P, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    if new:
        w.writeheader()
    for r in ROWS:
        w.writerow(r)
print("wrote", P, "rows", len(ROWS))
