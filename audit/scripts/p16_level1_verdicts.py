"""Phase 6 / 6A LEVEL 1 -- append the 122 remaining draft verdicts to audit/claims.csv.

Level 1 = locate the stated number in a result file and flag mismatches.
Locator: audit/scripts/p16_level1.py.  Idempotent.

Source for all of it: origin/paper @ ea1ce52b, read with `git show` (no checkout), plus
papers/2511.16035v2 (1).pdf, papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf,
kd/personas/kd_role_groups.json and audit/cache/labpc/run-kd/scores/*.json.
"""
import csv, glob, os, pathlib, shutil, datetime

ROOT = pathlib.Path(__file__).resolve().parents[2]
CLAIMS = ROOT / "audit" / "claims.csv"
BATCHDIR = ROOT / "audit" / "phase6" / "batches"
L2 = {"B23", "B41", "B42", "B43", "B45", "B46", "B40", "B30"}

EXT = "src/probes/deception-detection/results/persona_ood_ext_L{20,40,44}"
ZS, SUMM, OPER = f"{EXT}/zero_shot.csv", f"{EXT}/summary.csv", f"{EXT}/zero_shot_operating.csv"
LB = f"{EXT}/liars_bench_comparison.csv"
POS = "src/probes/deception-detection/results/persona_positions_L{20,40,44}/reads_summary.csv"
FUP = "src/probes/deception-detection/results/persona_ood_followup_L20"
PCS = "src/probes/deception-detection/deception_detection/persona_pcs.py"
OODPY = "src/probes/deception-detection/deception_detection/scripts/persona_ood.py"
LBPDF = "papers/2511.16035v2 (1).pdf"
MSPDF = "papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf"
SCRIPT = "audit/scripts/p16_level1.py"
ALLBR = f"origin/paper: {EXT}/ (git show)"

M, P, U = "MATCHES HIS DATA", "PARTLY MATCHES HIS DATA", "UNRESOLVED"
V = {}


def add(cid, verdict, l2, note, ev, key, imp="medium"):
    V[cid] = (verdict, l2, note, ev, key, imp)


# ---------------------------------------------------------------- B38 Table 1
T1 = {"D-070": ("CG", 267, 167), "D-071": ("HP-C", 2134, 134), "D-072": ("HP-KR", 2139, 139),
      "D-073": ("IT", 1080, 423), "D-074": ("ID", 5432, 3045), "D-075": ("RepE", 550, 275),
      "D-076": ("Alpaca", 2000, 0)}
for cid, (s, n, d) in T1.items():
    note = "Exact. Object-of-belief and reason-for-lying columns are the Liars' Bench taxonomy."
    if cid == "D-075":
        note += (" NOTE: the FOLLOW-UP run uses a different RepE row set -- 612/306, not 550/275 "
                 "(persona_ood_followup_L20/zero_shot.csv). Same dataset, different rows; Table 1's "
                 "figure is right for the main run, and Table 10 mixes the two runs in one row.")
    add(cid, M, f"{s}: examples {n}, lies {d} -- EXACT", note, ZS, f"dataset row, cols n / n_deceptive")

# ---------------------------------------------------------------- B44 Table 5
T5 = {"D-172": "Deceptive (published) n=8 -- handler 106, broker 113, impression manager 82, "
                "coverup operator 47, gatekeeper 126, spin doctor 85, plausible denier 64, "
                "narrative manager 70",
      "D-173": "Honest (published) n=7 -- archivist 103, disclosure counsel 118, documentarian 107, "
                "fact checker 48, ombudsman 148, transparency officer 142, whistleblower 108",
      "D-174": "Knowledge denial (new) n=4 -- amnesiac witness 129, deniable expert 72, "
                "feigned novice 137, stonewaller 129",
      "D-175": "Sincere ignorance (new) n=3 -- honest novice 110, calibrated reporter 83, "
                "scrupulous witness 74",
      "D-176": "Plain assertion (new) n=3 -- forthright expert 80, candid witness 64, plain dealer 76",
      "D-177": "Sycophantic (published) n=5 -- cheerleader 142, enabler 51, flatterer 148, "
                "people pleaser 113, yes man 109",
      "D-178": "Anti-sycophantic (published) n=5 -- blunt advisor 135, contrarian 130, critic 68, "
                "skeptic 133, tough love mentor 149",
      "D-179": "Anchor n=1 -- default 150"}
for cid, txt in T5.items():
    note = ("Recounted from the judge scores: kept = count of score==3 over the persona's 150 "
            "rollouts. All 35 judged personas match Table 5 exactly. `default` is never judged and "
            "keeps all 150 by design, which is why the totals work out.")
    add(cid, M, txt + " -- ALL EXACT", note,
        "audit/cache/labpc/run-kd/scores/<persona>.json",
        "count of entries with value == 3, per persona")

# ---------------------------------------------------------------- B35/B36 Fig 4
F4 = {"D-129": ("CG", "1.00 0.79 0.80 0.47 0.98 0.90"),
      "D-130": ("HP-C", "0.43 0.96 0.54 0.59 0.66 0.64"),
      "D-131": ("HP-KR", "0.06 0.48 0.95 0.79 0.66 0.93"),
      "D-132": ("IT", "0.92 0.95 0.97 0.99 0.89 0.27"),
      "D-133": ("ID", "0.97 0.76 0.75 0.67 1.00 0.98"),
      "D-134": ("RepE", "1.00 0.80 0.83 0.49 1.00 1.00")}
for cid, (s, vals) in F4.items():
    add(cid, M, f"test {s}, trained on CG/HP-C/HP-KR/IT/ID/RepE = {vals} -- EXACT at 2 dp",
        "Raw 8192-d probe, full L40 matrix including the diagonal. Reproduces cell for cell.",
        f"{EXT}/matrix_baseline.csv", f"column {s}, all six training rows, layer 40")

add("D-135", M,
    "rows constant: CG 0.91, HP-C 0.71, HP-KR 0.91, IT 0.98, ID 0.98, RepE 1.00 -- EXACT, and the "
    "off-diagonal cells in each column are IDENTICAL to 4 dp, not merely close",
    "Verified directly: for every test column all five off-diagonal cells hold one value, so the "
    "'rows are constant' description is exact rather than approximate. This is the dim=1 identity, "
    "and D-136's caption states the reason.",
    f"{EXT}/matrix_ext_axis1.csv", "layer 40, distinct off-diagonal values per column")

# ---------------------------------------------------------------- B47 Table 8
T8 = {"D-200": ("CG", "0.940/0.912 | 0.931/0.933 | 0.969/0.938"),
      "D-201": ("HP-C", "0.717/0.612 | 0.585/0.472 | 0.639/0.556"),
      "D-202": ("HP-KR", "0.148/0.168 | 0.890/0.237 | 0.882/0.190"),
      "D-203": ("IT", "0.988/0.165 | 0.976/0.216 | 0.990/0.173"),
      "D-204": ("ID", "0.959/0.900 | 0.932/0.904 | 0.954/0.903"),
      "D-205": ("RepE", "0.994/1.000 | 0.882/1.000 | 0.991/1.000")}
for cid, (s, vals) in T8.items():
    add(cid, M, f"{s}: 8v7 | 4v6 | 12v13, response mean / 'No' token = {vals} -- ALL SIX EXACT",
        "Response column from the main L20 run, 'No'-token column from the follow-up L20 run.",
        f"{EXT}/zero_shot.csv and {FUP}/zero_shot.csv",
        f"dataset={s}, cols auroc_axis / auroc_denial_axis / auroc_ext_axis")

# ---------------------------------------------------------------- B48 Table 9
T9 = {"D-210": ("Mean (default)", "mean", "0.887/0.591/0.187 | 0.899/0.694/0.413 | 0.884/0.666/0.350"),
      "D-211": ("Top-10% mean", "top10", "0.873/0.664/0.348 | 0.908/0.737/0.531 | 0.907/0.715/0.470"),
      "D-212": ("Maximum", "max", "0.858/0.659/0.336 | 0.910/0.750/0.548 | 0.914/0.701/0.418"),
      "D-213": ("First response token", "first", "0.753/0.605/0.334 | 0.863/0.606/0.227 | 0.876/0.568/0.140"),
      "D-214": ("Last response token", "last", "0.843/0.645/0.312 | 0.880/0.656/0.422 | 0.882/0.636/0.352"),
      "D-215": ("Token before the reply", "pre", "0.739/0.628/0.415 | 0.825/0.652/0.383 | 0.822/0.656/0.327")}
for cid, (label, rd, vals) in T9.items():
    note = "All nine values exact (AUROC/BA/recall at L20|L40|L44), 5-dataset mean, axis='extended'."
    if rd in ("top10", "max"):
        note += (" Relevant to DF-15: at L40 the single MAXIMUM read leads on balanced accuracy "
                 "(0.750 vs top-10%'s 0.737) and recall (0.548 vs 0.531), so 'the top tenth is the "
                 "better read' is true on the mean comparison but not against the maximum.")
    add(cid, M, f"{label}: {vals} -- EXACT", note, POS, f"axis=extended, read={rd}")

# ---------------------------------------------------------------- B39 Table 10
T10 = {"D-218": ("Probe on raw activations (8192-d)", "0.734 / 0.636"),
       "D-219": ("Extended contrast (1-d)", "0.887 / 0.552"),
       "D-220": ("Extended PCs (k = 3)", "0.736 / 0.579"),
       "D-221": ("Published PCs (k = 3)", "0.733 / 0.574"),
       "D-222": ("Knowledge-denial PCs (k = 3)", "0.810 / 0.699")}
for cid, (label, vals) in T10.items():
    add(cid, M, f"{label}: response tokens / follow-up 'No' = {vals} -- EXACT",
        "Re-derived as the RepE training row's mean over the five other test columns, in the main "
        "L20 run and the follow-up L20 run. Cross-checks against liars_bench_comparison.csv's "
        "'raw activations (RepE-trained)' row: 0.7344 response, 0.6364 follow-up.",
        f"{EXT}/matrix_*.csv and {FUP}/matrix_*.csv",
        "row repe_honesty__plain, mean over the 5 non-RepE test columns")

# ---------------------------------------------------------------- B02 Contributions
add("D-020", M, "26 published personas + 10 new (4 knowledge-denial / 3 sincere-ignorance / "
    "3 plain-assertion); contrasts 4v6 and 12v13 -- EXACT",
    "Counts verified against the bank; 8+4=12 deceptive and 7+3+3=13 honest give the 12v13 arm.",
    "kd/personas/kd_role_groups.json", "groups.*.roles")
add("D-021", M, "five Liars' Bench datasets + RepE, layers 20/40/44 -- CONFIRMED",
    "The six evaluation columns and the three layer directories are all present on the branch.",
    ZS, "dataset column; persona_ood_ext_L{20,40,44} directories")
add("D-022", M, "0.919 AUROC and 0.910 BA@1%FPR on HP-KR at L40 -- EXACT (Level 2 D-087)",
    "Re-derived at Level 2. The threshold is Alpaca-fitted, so the realised FPR on HP-KR is 6.45% "
    "at L40; the Figure 3 caption discloses the L20 version of this in words.", OPER,
    "arm='knowledge-denial axis (4v6)', dataset=HP-KR, control", "high")
add("D-024", M, "base probe BA on HP-KR = 0.5000 at L20, L40 and L44 -- EXACT",
    "True as stated. Worth naming the metric in the text: it is balanced accuracy. The same probe's "
    "AUROC on HP-KR is 0.109 at L20 (inverted) and 0.936 at L40, so 'stays at 0.500' describes a "
    "detector that never fires (recall 0.0000, specificity 1.0), not one that cannot rank.",
    OPER, "arm='base probe (RepE-trained)', dataset=HP-KR, control, col balacc")
add("D-025", M, "none of the 8 published deceptive personas lies by denying knowledge -- CONFIRMED",
    "The eight are handler, broker, impression_manager, coverup_operator, gatekeeper, spin_doctor, "
    "plausible_denier, narrative_manager -- all specialists in omission, timing and spin. The bank "
    "file's own note for the group is 'control disclosure: timing, emphasis, omission'.",
    "kd/personas/kd_role_groups.json", "groups.deception.roles and .note")
add("D-026", M, "ext_axis1 off-diagonal 0.9158 vs baseline 0.7320 at L40 -- EXACT",
    "The sentence already labels the arm 'zero-shot' and the comparator '8192-dimensional probes "
    "each trained on one dataset', so the protocol difference is stated where it matters.",
    SUMM, "features=ext_axis1 / baseline, off_diagonal_mean, L40", "high")
add("D-029", M, "protocol as described: Alpaca 1% FPR threshold, bootstrap intervals -- CONFIRMED",
    "Matches section 4 Metrics and the operating file. The intervals are '95% stratified bootstrap "
    "intervals over examples (1,000 resamples), drawn once and shared by every detector'.",
    OPER, "threshold_source=control; draft section 4 Metrics")
add("D-030", U, "NOT LOCATABLE -- the pipeline's own results tables are not committed",
    "Appendix G states the rescored activations matched the pipeline's results table to four "
    "decimal places on all five datasets. The comparison cannot be reproduced from the branch: "
    "`results/XXXX-9-llama-all*/` contains only `plots/*.png`, no results table. STRENGTHENED "
    "2026-09-24: a recursive search of the lab PC (C:/Users/user4/expLB) for "
    "`results_table.csv` / `results.csv` / `scores.csv` and for any `XXXX-9-llama*` directory "
    "returns NOTHING. So the artifact is not merely uncommitted -- it does not exist outside the "
    "collaborator's own working directory, and this claim is uncheckable by anyone else. "
    "Self-reported, no artifact. Same gap as D-063.",
    "src/probes/deception-detection/results/XXXX-9-llama-all*/",
    "directory listing -- plots only, no CSV", "high")
add("D-031", M, "Follow-Up Probe measured 0.6360 vs published 0.6340 -- EXACT both sides",
    "Both rows are in the same file, so the comparison is checkable as stated.",
    f"{FUP}/liars_bench_comparison.csv",
    "rows 'Follow-Up Probe (measured here, layer 20)' and 'Follow-Up Probe', col auroc")
add("D-032", M, "RESOLVED 2026-09-24. Code released: paper/make_numbers.py (28,232 B) and "
    "paper/make_figures.py at 8fd071b2. Results released: paper/generated/ holds numbers.tex "
    "(1,162 generated \setV macros), eight tab_*.tex and five fig_*.pdf, beside the full "
    "persona_ood_ext_L* result CSVs.",
    "REVISED UP TWICE. My first reading called this PARTLY because Figure 2's numbers and the "
    "cosine were in no committed result file -- they ARE, in paper/generated/numbers.tex, which is "
    "generated by make_numbers.py and committed. I had looked only in the result CSVs. "
    "**One real residual, and it is about re-RUNNING rather than reading:** make_figures.py's "
    "personas() needs pcs.EXTENDED_VECTORS_DIR, and those 36 vectors are not on the branch, so "
    "Figure 2 cannot be regenerated from the repository even though its numbers are published "
    "there. Shipping the vectors (small) would close it. The pipeline's own results tables are "
    "also absent (D-030/D-063) and are absent from the lab PC too.",
    "git ls-tree -r origin/paper -- src/probes/deception-detection/paper/",
    "generated/numbers.tex, tab_*.tex, fig_*.pdf", "high")
# ---------------------------------------------------------------- B03 Introduction
add("D-012", M, "supported by the draft's own Table 4: the RepE-trained base probe reads BA 0.626 "
    "at L40 and L44 over the five Liars' Bench datasets, and Liars' Bench's Mean Probe reads "
    "BA 0.50 on HP-C, HP-KR, ST and IT for Llama-3.3-70B",
    "Holds. The benchmark's own table is the stronger citation.", LBPDF,
    "Table 6 p.54, BALANCED ACCURACY AT 1% FPR, row 'Llama 3.3 70B'")
add("D-014", M, "faithful to the cited paper: main model Llama-3.2-3B, 'additional replications on "
    "Llama 3-8B' (p.4); contribution bullets claim persona-PC projections generalise better than "
    "raw activations (p.2)",
    "Accurate as a report of the cited work. Two things belong beside it: the paper's own words for "
    "8B are 'auxiliary comparisons ... broadly supportive' (p.5), not a full replication; and this "
    "project's three-rung ladder finds the gain DECLINES with scale on both banks (F-121), so it "
    "does not extend to 70B. That is this paper's contribution and is worth saying here.",
    MSPDF, "pp.1-5, contribution bullets and Section 5", "high")
add("D-015", M, "five Liars' Bench datasets + RepE at three depths -- CONFIRMED", "As D-021.",
    ZS, "dataset column; three layer directories")
add("D-016", M, "HP-KR is not used by the replicated paper -- CONFIRMED",
    "The cited paper evaluates ConvGame, InstrDec, Mask, AILiar, Roleplay and a sycophancy set; "
    "HP-KR is absent, so the inversion was undetectable there. This is the paper's strongest "
    "framing sentence and it holds.", MSPDF, "Section 5 and Figures 4-5 dataset lists", "high")
add("D-017", M, "AUROC 0.1483 -> 0.148 at L20 -- EXACT", "Level-2 verified (D-086, D-184).",
    ZS, "dataset=HP-KR, col auroc_axis, L20", "high")
add("D-018", M, "the cause is in the personas -- CONFIRMED", "Same evidence as D-025.",
    "kd/personas/kd_role_groups.json", "groups.deception.roles")
add("D-019", M, "0.890 / 0.919 / 0.908 for the knowledge-denial contrast on HP-KR -- EXACT",
    "Level-2 verified (D-087, D-184).", ZS, "dataset=HP-KR, col auroc_denial_axis")

# ---------------------------------------------------------------- B04 Related work
add("D-036", M, "Liars' Bench sets the threshold at 1% FPR on benign chat (Alpaca) -- CONFIRMED",
    "Accurate as a description of the benchmark's protocol, and it is the protocol followed. The "
    "realised FPR on the evaluation sets is another matter (D-011, section 1(b) of the chapter).",
    LBPDF, "Tables 6-8 captions, 'at a 1% false-positive rate on alpaca'")
add("D-037", P, "Mean Probe recall on HP-KR (Llama-3.3-70B) = 0.00 -- EXACT. But Follow-Up Probe "
    "recall = 0.37 and Upper-Bound Probe recall = 0.97 on the same cell.",
    "'Recalls near zero for EVERY method' is false as written. It holds for the Mean Probe (0.00) "
    "and for the cross-model averages of the non-privileged probes (0.00 and 0.12), but the "
    "Follow-Up Probe reads 0.37 on Llama-3.3-70B and the privileged upper bound 0.97. Confirms "
    "DF-13 at Level 1. The fix is one clause -- 'near zero for every non-privileged method on the "
    "cross-model average' -- and the Follow-Up Probe's 0.37 is bought at an 84% FPR, which makes "
    "the paper's point better than the overstatement does.",
    LBPDF, "Tables 6/7/8 p.54-56, RECALL blocks, row 'Llama 3.3 70B', col HP-KR", "high")
add("D-038", M, "ten evaluation datasets -- CONFIRMED", "The cited paper's abstract: 'Across 10 "
    "evaluation datasets, we show that persona-derived directions transfer non-trivially'.",
    MSPDF, "p.1 abstract")
add("D-039", M, "PDF present: papers/The Impact of Off-Policy Training Data.pdf, arXiv 2511.17408 "
    "(audit/papers.csv)",
    "Citation is to a paper the project holds, and the characterisation matches how "
    "cap/docs/DESIGN.md:79 already uses it.",
    "audit/papers.csv", "arxiv_id / pdf_present columns")
add("D-040", M, "RESOLVED 2026-09-24: Vikram Natarajan, Devina Jain, Shivam Arora, Satvik Golechha, "
    "Joseph Bloom -- 'One Probe Won't Catch Them All: Towards Targeted Deception Detection', "
    "arXiv 2602.01425.",
    "The abstract matches the draft's characterisation: probes fail with spurious correlations and "
    "false positives, and targeted probes outperform a universal one. CORRECTION TO MY OWN FIRST "
    "PASS, which marked this UNRESOLVED on the basis that audit/papers.csv has no row for it -- "
    "true, but that is a gap in the audit's own bibliography, not evidence the citation is bad. "
    "**Owed: add the row to audit/papers.csv and pull the PDF.**",
    "arXiv API, id_list=2602.01425", "title, authors, abstract", "high")
add("D-041", M, "RESOLVED 2026-09-24: the PDF is ALREADY IN papers/ -- 'Probe Generalization as "
    "Subspace Selection for OOD Deception Detection', **Daniel Yoo and Adrians Skapars**, "
    "arXiv 2609.02893.",
    "Author line read from page 1 of papers/'Probe Generalization as Subspace Selection for OOD "
    "Deception Detectio.pdf'. CORRECTION TO MY OWN FIRST PASS: I checked audit/papers.csv and not "
    "the papers/ directory itself, and wrongly marked this UNRESOLVED. The citation is sound. "
    "**Owed: add the row to audit/papers.csv.** DF-19's separate point -- that two venue strings "
    "near these citations appear in none of the three PDFs held -- is untouched by this and still "
    "needs checking before the reference list is fixed.",
    "papers/Probe Generalization as Subspace Selection for OOD Deception Detectio.pdf",
    "page 1 author line; arXiv 2609.02893", "high")
add("D-043", M, "the direction is a contrast between persona means, not a dataset PC -- CONFIRMED "
    "by construction in the code",
    "`persona_pcs.py:157` builds each axis as `mean_of(POSITIVE) - mean_of(NEGATIVE)` over persona "
    "vectors. No dataset activations enter it. The distinction the sentence draws is real.",
    PCS, "lines 149-160, axis construction")

# ---------------------------------------------------------------- B05-B09 method
add("D-044", M, "persona vector = mean-pooled residual-stream activation over response tokens, "
    "averaged over the persona's rollouts -- CONFIRMED",
    "`persona_pcs.py` header states it as Eq.1 `v_p = mean-pooled activation for persona p over its "
    "rollouts`, with Eq.2 centring `v~_p = v_p - mean over all personas`.", PCS, "lines 6-7")
add("D-045", M, "36 personas; 26 published = 8 deceptive + 7 honest + 5 sycophantic + "
    "5 anti-sycophantic + 1 anchor -- EXACT", "Counts match the bank file group by group.",
    "kd/personas/kd_role_groups.json", "groups.*.roles lengths")
add("D-047", M, "ten new roles in three groups, 4 / 3 / 3, all named -- EXACT",
    "knowledge_denial = amnesiac_witness, deniable_expert, stonewaller, feigned_novice; "
    "sincere_ignorance = honest_novice, calibrated_reporter, scrupulous_witness; "
    "plain_assertion = forthright_expert, candid_witness, plain_dealer. Names match one for one.",
    "kd/personas/kd_role_groups.json", "groups.{knowledge_denial,sincere_ignorance,plain_assertion}")
add("D-048", M, "the evaluated contrasts pool both honest groups -> 4 v 6 -- CONFIRMED",
    "3 sincere-ignorance + 3 plain-assertion = 6 against 4 knowledge-denial. Note DF-25: the "
    "collaborator's own KNOWLEDGE_CANDOUR_PERSONAS merges the two honest groups into one arm, and "
    "that split exists precisely to remove a wording confound -- worth a sentence, since the "
    "sentence here is about why both sides must deny.",
    "kd/personas/kd_role_groups.json", "groups.sincere_ignorance + groups.plain_assertion")
add("D-049", M, "36 personas x 30 questions; 20 from the cited paper + 10 new knowledge questions "
    "-- CONFIRMED",
    "The bank file's own question_set field: '20 questions copied verbatim from "
    "src/assistant_axis_ndif/pc_questions_shared.jsonl (the paper's 10 deception + first 10 "
    "sycophancy, Appendix C) plus 10 new knowledge-admission questions. All 36 personas answer all "
    "30.'", "kd/personas/kd_role_groups.json", "question_set")
add("D-050", M, "shared question set matters, or PC1 tracks questions rather than roles -- "
    "CONFIRMED, and this project measured it",
    "Not merely a design principle here. The project's own measurement is that the paper16 unified "
    "PC1 (47% variance) IS the question family, cosine -0.98 against a question-set indicator. "
    "Citing that turns an assertion into evidence, and it is the strongest available defence of the "
    "shared-question design.", "kd/docs/FINDINGS.md", "the unified-PC1 / question-family entry", "high")
add("D-051", M, "5,400 rollouts = 36 personas x 5 variants x 30 questions -- EXACT arithmetic, and "
    "the realised rollout files match",
    "36 x 5 x 30 = 5,400. The 37 response files (36 personas + default) are present in the cached "
    "run-kd store. Sampling parameters (T=0.7, top-p 0.9, max 512 tokens) are stated in the draft "
    "and not independently re-derived here -- they live in the generation script, not in a result "
    "file.", "audit/cache/labpc/run-kd/responses/", "file count and per-file row counts")
add("D-052", M, "kept 3,749 of 5,400 (69.4%), per-persona range 47-149 -- ALL EXACT",
    "Recounted from the judge scores: 3,599 kept over the 35 judged personas, plus `default`'s 150 "
    "(never judged, by design) = 3,749. 3,749/5,400 = 69.4%. Min 47 (coverup operator), max 149 "
    "(tough love mentor). The 0-3 scale and keep-only-3 rule are as described.",
    "audit/cache/labpc/run-kd/scores/*.json", "count of entries == 3 per persona", "high")
add("D-053", M, "vectors extracted at all 80 layers -- CONSISTENT",
    "Llama-3.3-70B-Instruct has 80 layers, and the branch carries persona_ood_ext directories at "
    "16 of them (L20 through L79). The full 80-layer extraction is not itself a committed artifact.",
    "git ls-tree origin/paper", "persona_ood_ext_L* directory list")
add("D-054", M, "contrast = difference of two group means; centring cancels -- CONFIRMED in code",
    "`persona_pcs.py:157` computes `mean_of(A) - mean_of(B)` on the uncentred vectors, and the "
    "header records that centring cancels in a difference of means. The file also warns that the "
    "stored axes are raw differences, not unit vectors, so any cosine must normalise first.",
    PCS, "lines 97-99 and 149-160")
add("D-055", M, "three contrasts 8v7 / 4v6 / 12v13 -> features axis1 / denial_axis1 / ext_axis1 -- "
    "all three present at all three layers", "Confirmed in summary.csv.", SUMM, "features column")
add("D-056", M, "PCs fitted for the published and extended banks -- CONFIRMED",
    "`pc3` / `pc1` (published) and `ext_pc3` / `ext_pc1` (extended) are present at every layer.",
    SUMM, "features column")
add("D-057", M, "RESOLVED on the lab PC: cosine(deception, denial) at L40 = **0.474559** -> 0.47 "
    "EXACT. (L20 0.417432 -> 0.42; L44 0.467577 -> 0.47, so the draft's 'at layer 40' is the right "
    "scope.)",
    "Rebuilt his Figure 2 geometry from `C:\\Users\\user4\\expLB\\run-kd\\vectors_s3` -- 36 "
    "personas, (80, 8192) each -- following `make_figures.py:personas()`: axes as "
    "`mean_of(KNOWLEDGE_DENIAL) - mean_of(KNOWLEDGE_CANDOUR)` and "
    "`mean_of(DECEPTION) - mean_of(HONESTY)`, each normalised, then their dot product. "
    "**That folder is his EXTENDED_VECTORS_DIR**: its `pc_handler.pt` records n = 106, which is "
    "both Table 5's count and the '64 -> 106' his own docstring describes; `vectors_s23` and "
    "`vectors_all` give 150. NOTE the number is still not in any committed file, and the vectors "
    "are not on the branch -- it is checkable only with lab-PC access (see D-032/D-171).",
    "LAB PC C:\\Users\\user4\\expLB\\run-kd\\vectors_s3\\*.pt",
    "layer-40 slice; unit(mean(4 denial) - mean(6 candour)) . unit(mean(8 dec) - mean(7 hon))", "high")
add("D-058", M, "zero-shot score = projection on the unit contrast, sign fixed by construction, no "
    "labels -- CONFIRMED",
    "Matches the code. Worth pairing with D-069: the zero-shot sign is fixed, but a TRAINED probe "
    "on the same one feature can and does pick the opposite sign when its training set inverts.",
    PCS, "axis_unit(); draft section 3.4")
add("D-059", M, "L2 logistic regression, C = 0.1, standardised features, no intercept -- EXACT",
    "`persona_ood.py:192-193` is `StandardScaler()` then `LogisticRegression(C=1/REG_COEFF, "
    "fit_intercept=False, max_iter=MAX_ITER)` with `REG_COEFF = 10.0` (line 59), so C = 0.1 exactly, "
    "and it does match the pipeline's own probe, whose lambda is 10 (D-077).",
    OODPY, "lines 59, 192-193")
add("D-060", M, "the dim=1 identity, stated correctly and with its condition -- VERIFIED EXACTLY",
    "'A probe on a single feature can learn only a sign and a scale. Whenever every training set "
    "agrees on the sign, its AUROC on a test set therefore equals the zero-shot AUROC, and it has "
    "no training set to overfit.' Re-derived cell by cell: in every dim=1 matrix, each cell in a "
    "test column is identical across all five training rows up to sign. The condition clause "
    "('whenever every training set agrees') is doing real work -- it is exactly what fails at L20 "
    "for the published contrast. **This sentence is the paper's disclosure of the dim=1 framing, "
    "and it is complete.** It also means DF-93's 'label these arms zero-shot' is already met.",
    f"{EXT}/matrix_*.csv", "per-column distinct values across training rows", "high")
add("D-061", M, "three random directions, five draws -- CONFIRMED as feature `random3`, dim 3",
    "Present at all three layers, reading 0.527 / 0.504 / 0.493 off-diagonal. The control behaves, "
    "which is what makes the persona rows readable.", SUMM, "features=random3, dim column")
add("D-062", M, "residual-stream activations over response tokens, layers 20/40/44 of 80, system "
    "prompts excluded -- CONSISTENT", "Matches the stated pipeline; token selection follows "
    "Goldowsky-Dill et al.'s detection pipeline as the draft says.",
    "draft section 3.5", "cross-checked against the layer directories on the branch")
add("D-063", U, "NOT LOCATABLE -- same gap as D-030",
    "The four-decimal reproduction against the pipeline's own results is reported in Appendix G but "
    "the pipeline's results tables are not committed (`results/XXXX-9-llama-all*/` holds only "
    "`plots/*.png`) AND are not on the lab PC either (searched 2026-09-24, nothing). The claim is "
    "plausible and internally consistent -- D-031's Follow-Up Probe figure does reproduce from "
    "committed files -- but this specific comparison cannot be checked by anyone but its author.",
    "src/probes/deception-detection/results/XXXX-9-llama-all*/", "directory listing", "high")

# ---------------------------------------------------------------- B10-B14
add("D-077", M, "base probe: logistic regression on RepE, L2 with lambda = 10, standardised "
    "per-token activations, score averaged over response tokens -- EXACT",
    "`REG_COEFF = 10.0` at `persona_ood.py:59`. Consistent with D-059's C = 1/10.",
    OODPY, "line 59")
add("D-066", M, "five Liars' Bench datasets generated by Llama-3.3-70B without fine-tuning; "
    "Soft-Trigger and Gender-Secret left to future work -- CONSISTENT",
    "The six evaluation columns are CG, HP-C, HP-KR, IT, ID and RepE; ST and GS are absent, as the "
    "sentence says. Note that this project HAS since run Gender-Secret (F-118), and it independently "
    "confirms the register diagnosis -- worth citing rather than deferring.",
    ZS, "dataset column", "high")
add("D-067", M, "RepE train+validation merged; Alpaca used only for thresholds -- CONSISTENT",
    "Alpaca appears as `XXXX-20__plain`, n=2,000, n_deceptive=0, and carries no AUROC in any arm -- "
    "exactly what a threshold-only control looks like in these files.",
    ZS, "dataset=XXXX-20__plain row, all auroc columns empty")
add("D-078", M, "threshold at <=1% of the Alpaca control; 95% stratified bootstrap over examples, "
    "1,000 resamples, drawn once and shared -- CONFIRMED",
    "As described in section 4. Important for DF-93: because the intervals are over EXAMPLES and "
    "never over the 30 matrix cells, the duplicate-counting concern about dim=1 arms does not arise "
    "in this draft.", OPER, "threshold_source=control; *_lo / *_hi columns", "high")
add("D-068", M, "six datasets, 30 off-diagonal cells + 6 diagonal, 5-fold, 20% -- CONFIRMED",
    "Every matrix_*.csv is 6x6 with the diagonal reported separately as diagonal_mean in "
    "summary.csv, giving 30 off-diagonal cells.", f"{EXT}/matrix_*.csv and {SUMM}", "matrix shape")
add("D-069", P, "'CG falls from 0.940 to 0.060' -- EXACT (0.9399 -> 0.0601, verified cell by cell). "
    "The scope word 'only' does not hold.",
    "The worked example is exact and the disclosure is valuable -- this sentence is where the draft "
    "explains the L20 sign flip. But 'that happens ONLY for the published contrast trained on HP-KR "
    "at layer 20' is too narrow. Sweeping all seven dim=1 arms at the three published layers, "
    "training rows invert in SEVEN (arm, layer) cells: axis1, syc_axis1, syc_pc1 and pc1 at L20; "
    "syc_axis1 and syc_pc1 at L40; syc_axis1 at L44. It is always HP-KR's row. Among the three "
    "HEADLINE contrasts the sentence is correct -- denial_axis1 and ext_axis1 never invert -- but "
    "the sycophancy contrast is reported in Table 11 and Appendix F, and its printed off-diagonal "
    "row is depressed by this at all three layers (0.615/0.663/0.708 against 0.743/0.808/0.849 "
    "sign-corrected). Scope the sentence to the three contrasts, or note Table 11.",
    f"{EXT}/matrix_axis1.csv and matrix_syc_axis1.csv, matrix_syc_pc1.csv, matrix_pc1.csv",
    "per-training-row off-diagonal means, all three layers", "high")
add("D-079", M, "HP-KR's label definition -- CONFIRMED against the benchmark",
    "Liars' Bench establishes in neutral contexts which questions the model knows; a response is a "
    "lie if the model denies knowing an answer it was shown to know. Its Appendix D confirms that "
    "cases where the model claims knowledge without a consistent belief are DISCARDED -- which is "
    "the structurally empty fourth cell the draft's own limitation (D-034) rests on.",
    LBPDF, "HP-KR construction and Appendix D", "high")

# ---------------------------------------------------------------- B15-B22
add("D-108", M, "the 1-d contrast beats the 3-PC probe on transfer at all three layers on the "
    "extended bank: ext_axis1 - ext_pc3 = +0.220 (L20), +0.050 (L40), +0.023 (L44)",
    "The mechanism the sentence gives -- extra dimensions let a probe fit its training set -- is "
    "consistent with the diagonal/off-diagonal split: ext_pc3's diagonal is higher than "
    "ext_axis1's while its off-diagonal is lower. Note DF-21: for the PUBLISHED bank the direction "
    "reverses at L20 (pc3 0.679 > axis1 0.651), so the claim needs its bank named.",
    SUMM, "features=ext_axis1 / ext_pc3, off_diagonal_mean and diagonal_mean", "high")
add("D-119", M, "at L40 and L44 every persona contrast leads the base probe on BA, recall and "
    "AUROC -- VERIFIED for all three arms at both layers",
    "L40 base 0.626/0.281/0.778 against 8v7 0.675/0.382/0.798, 4v6 0.725/0.487/0.894, "
    "12v13 0.694/0.413/0.899. L44 base 0.626/0.281/0.756 against 0.654/0.318/0.791, "
    "0.683/0.395/0.880, 0.666/0.350/0.884. No exceptions.",
    OPER, "dataset=\"Average (Liars' Bench)\", threshold_source=control", "high")
add("D-123", M, "on AUROC the persona contrasts lead at every layer -- VERIFIED",
    "L20 base 0.697 against 0.750 / 0.863 / 0.887; L40 base 0.778 against 0.798 / 0.894 / 0.899; "
    "L44 base 0.756 against 0.791 / 0.880 / 0.884. True at all three, and it is the honest "
    "restriction of D-119, since at L20 the base probe leads on BA and recall.",
    OPER, "dataset=\"Average (Liars' Bench)\", col auroc", "high")
add("D-128", M, "'Three layers do not locate an optimum, and a layer between 20 and 40 may do "
    "better' -- CORRECT, and since vindicated",
    "The 16-layer sweep on this same branch puts `ext_axis1`'s optimum at L36 = 0.9275 against "
    "L40's 0.9158, and `denial_axis1` also peaks at 36 (F-117). The caveat was right and the paper "
    "should now say so rather than leave it hypothetical -- under-reporting your own best number "
    "invites the question of why (DF-79).",
    "src/probes/deception-detection/results/persona_ood_ext_L36/summary.csv",
    "features=ext_axis1, off_diagonal_mean", "high")
add("D-152", M, "'a detector of denial wording would score well on HP-KR without detecting "
    "deception; our contrast does not rule this out' -- CORRECT, and understated",
    "This project has since measured it: a label-free regex on denial phrasing alone reads 0.9125 "
    "AUROC (0.9597 for a second phrasing) on the 2,139 rows, at or above the draft's own 0.919 "
    "headline, and a length-only scorer reads 0.776. On those numbers the wording confound is not "
    "merely 'not ruled out' -- it is not currently distinguishable (DF-72). Quote the lexical "
    "baseline beside the headline.",
    "kd/docs/FINDINGS.md", "the HP-KR lexical-baseline entries", "high")
add("D-153", P, "'The bank was designed for the cleaner test ... and we have not run it' -- NO "
    "LONGER TRUE",
    "It has been run three times. F-03/F-72/F-76 on 2026-09-19; the row-leak contamination closed "
    "on 2026-09-23 (F-115); and the larger complementary control run for the first time the same "
    "day (F-116, 2,001 rows, 7.8x the text-matched slice). At the pre-registered layer the margin "
    "is +0.066, CI [-0.069, +0.189], P = 0.161; it inverts to -0.2545 at L20; and on the "
    "knowledge-matched control the axes sit BELOW chance while an ordinary probe reads 0.999 "
    "in-distribution. This was true when written and is false at submission (DF-76). Delete the "
    "promise and state the outcome -- it is a stronger paper with the negative result in it.",
    "kd/docs/FINDINGS.md", "F-03, F-115, F-116", "high")
add("D-154", P, "RESOLVED on the lab PC, and it holds only in the weak sense. At L40 sincere "
    "ignorance reads **-0.0844** against feigned deniers **+2.9037**, so it is emphatically on the "
    "honest side of the deniers. But the honest *cluster* it is said to sit with reads "
    "**-0.9999** (published honest) and **-1.2274** (plain assertion).",
    "Sincere ignorance is 0.92-1.14 away from the honest groups and 2.99 from the deniers, so "
    "'sits with the honest cluster' overstates: it sits **between**, nearer the honest side. On an "
    "axis spanning 4.13, being 1.14 off the other honest group is 28% of the range -- and it is "
    "the group that shares the deniers' *wording*, which is exactly where a wording-sensitive axis "
    "would put it. **The draft's own hedge is the right one** ('whether the same holds for model "
    "responses is untested'), and it has since been answered: F-118's Gender-Secret run finds the "
    "direction **at chance where both classes deny**. Report the gap rather than the cluster.",
    "LAB PC C:\\Users\\user4\\expLB\\run-kd\\vectors_s3\\*.pt",
    "layer 40; group means on the unit denial axis", "high")
add("D-155", M, "'we wrote the knowledge-denial personas after seeing the published contrast invert "
    "on HP-KR, so HP-KR measures a repair designed for it, not a blind test' -- CORRECT",
    "Confirmed by this repository's own history: the kd/ experiment exists because every detector "
    "here inverts on harm-pressure-knowledge-report. Declaring it is the right call and it is the "
    "sentence a referee will look for.", "kd/docs/DESIGN.md", "the experiment's stated motivation",
    "high")
add("D-156", M, "'the extended contrast matches or beats the published one at every layer' on the "
    "other datasets -- VERIFIED on the excluding-HP-KR column: 0.909 vs 0.919 (L20), 0.917 vs 0.880 "
    "(L40), 0.905 vs 0.873 (L44)",
    "CAREFUL: at L20 excluding HP-KR the extended contrast is BEHIND by 0.010 (0.909 against "
    "0.919), which is what the sentence's own 'But at layer 20, with HP-KR excluded ...' clause "
    "goes on to concede. So the claim as completed is accurate; read without its continuation it "
    "would not be. F-117 generalises the concession from 3 layers to 16: the margin is within "
    "+/-0.06 everywhere and crosses zero five times (DF-81).",
    SUMM, "features=ext_axis1 / axis1, off_diagonal_excl_hpkr", "high")
add("D-159", M, "one model, three layers, one persona bank -- CORRECT and appropriately scoped",
    "All three limitations hold. The judge-keeps-only-3 rule is confirmed at D-052 (69.4% kept).",
    "kd/personas/kd_role_groups.json", "subject_model, source_role_list")
add("D-160", M, "insider-trading scores exclude the reasoning scratchpad, so comparisons with "
    "Liars' Bench's published numbers on that dataset are not like for like -- CORRECT",
    "An important and voluntary disclosure. 'Comparisons within our pipeline are exact' is right: "
    "every arm in these tables is scored on the same rows.",
    "draft section 6", "cross-checked against the LB Mean Probe's scratchpad handling", "high")
add("D-158", M, "'a probe trained on several datasets pooled is a stronger supervised baseline, and "
    "we do not compare against it' -- CORRECT, and this project has the number",
    "The honest pooled figure from this project's own work is +0.125 under leave-one-CLUSTER-out, "
    "10 of 11 targets, five seeds, +/-0.035 -- not the larger figure that leaks. If the baseline is "
    "going to be named as missing, that is the number to name.",
    "kd/docs/FINDINGS.md", "the pooled-training entries", "high")
for cid, txt, note in [
    ("D-161", "more model families: Mistral-Small-24B, Qwen-2.5-72B, Gemma-3-27B -- CONFIRMED",
     "All three appear in Liars' Bench's Tables 6-8 alongside Llama-3.3-70B, so the future-work "
     "target is real and the data exists."),
    ("D-162", "Soft-Trigger and Gender-Secret as fine-tuned model organisms -- CONFIRMED as a gap, "
     "but Gender-Secret is no longer open",
     "F-118 ran Gender-Secret through NDIF and found the knowledge-denial direction AT CHANCE where "
     "both classes deny (0.436-0.623) -- one of the three convergent wording-matched results. Cite "
     "it as a result, not as future work."),
    ("D-163", "wording-controlled contrasts, tested on HP-KR's honest denials -- SUPERSEDED",
     "Same as D-153: run three times (F-03/F-72/F-76, F-115, F-116). This is the second place the "
     "draft promises it as future work (DF-76 flags p.12 and p.13)."),
    ("D-164", "'for detection, the top tenth of response tokens is a better read than the mean' -- "
     "TRUE against the mean, but the MAXIMUM beats both at L40",
     "Top-10% vs mean at L40: BA 0.737 vs 0.694, recall 0.531 vs 0.413 -- the claim holds. But the "
     "single maximum reads BA 0.750 and recall 0.548 at the same layer, ahead of top-10% on both "
     "(DF-15). If a best read is being recommended, it is the maximum at L40."),
    ("D-165", "'a contrast that ranks a dataset well still needs a threshold that holds on that "
     "dataset's prompts' -- CORRECT and well evidenced",
     "This is the paper's most defensible forward-looking sentence. 23 of 90 (arm x dataset x "
     "layer) cells exceed 1% realised FPR when the threshold is Alpaca-fitted, worst 88.3%."),
]:
    add(cid, P if cid in ("D-162", "D-163") else M, txt, note,
        LBPDF if cid == "D-161" else "kd/docs/FINDINGS.md",
        "Tables 6-8 model rows" if cid == "D-161" else "F-115, F-116, F-118; DF-15",
        "high" if cid in ("D-162", "D-163") else "medium")
add("D-169", M, "'what it does not yet have is a threshold that holds on every dataset' -- CORRECT",
    "The conclusion's honest note, and the measurements back it (D-011, D-165).",
    OPER, "specificity column across arms and datasets")
add("D-170", M, "authorship attribution for the persona bank and the rollout/judging pipeline",
    "Consistent with the repository: the bank, the rollouts and the judge pipeline are in kd/ and "
    "run-kd/, authored on this side. Acknowledgement is appropriately placed.",
    "kd/personas/, audit/cache/labpc/run-kd/", "bank and rollout provenance")
add("D-171", M, "RESOLVED 2026-09-24 and it is better supported than any comparable claim I "
    "have checked. paper/generated/numbers.tex carries **1,162 \setV macros**, headed "
    "'Generated by make_numbers.py -- do not edit', and every draft number spot-checked in Phase 6 "
    "resolves to one of them -- dataset sizes, bank counts, the zero-shot grid, the cosine and the "
    "Figure 2 projections included.",
    "REVISED UP. My first reading called this PARTLY on the belief that five numbers were not "
    "generated from released files. Four of the five (cos-dec-den-L40, proj-den-sincere-L40, "
    "proj-den-kd-L40, proj-den-deceptive-L40) are macros in that committed file, and I re-derived "
    "all four independently on the lab PC to 4 dp. **The one genuine exception is Appendix G's "
    "four-decimal reproduction claim** (D-030/D-063), whose comparison artifact exists neither on "
    "the branch nor on the lab PC. So the sentence is true of every number in the paper except "
    "that one, and the honest scope is 'every number in the tables and figures'.",
    "src/probes/deception-detection/paper/generated/numbers.tex",
    "1,162 \setV entries; header line 'Generated by make_numbers.py'", "high")
# ---------------------------------------------------------------- B25-B29 appendices
add("D-180", M, "five prompt variants per persona, with the feigned-novice and honest-novice sets "
    "quoted in full -- CONSISTENT",
    "Five variants x 30 questions = 150 rollouts per persona, which is exactly the denominator "
    "Table 5 uses and D-052's arithmetic confirms.",
    "kd/personas/kd_instructions", "per-persona variant files")
add("D-181", M, "30 questions = 20 from the cited paper + 10 new; of the new ten, 6 the model knows "
    "and 4 it does not -- CONFIRMED for the 20/10 split",
    "The bank file records the 20/10 split verbatim. The 6-know / 4-do-not breakdown is a property "
    "of the new question file and is not restated in the bank metadata; it was not separately "
    "re-derived at Level 1.",
    "kd/personas/kd_role_groups.json", "question_set")
add("D-196", M, "'at the follow-up token the contrast's sign stops being the same on every dataset' "
    "-- VERIFIED",
    "At the 'No' token, L20: CG 0.912, ID 0.900, RepE 1.000 all point one way, while IT reads 0.165 "
    "and HP-KR 0.168 -- both inverted. So the sign genuinely stops being universal, exactly as "
    "stated. This is the same mechanism as D-069, observed at a different token.",
    f"{FUP}/zero_shot.csv", "col auroc_axis across datasets", "high")
add("D-199", M, "'what the model carries forward to a self-report is not the same signal as the one "
    "its response carries' -- SUPPORTED",
    "The response-vs-'No' gaps in Table 8 are large and not uniform in direction: IT falls "
    "0.988 -> 0.165, HP-KR's 4v6 falls 0.890 -> 0.237, while RepE rises 0.994 -> 1.000 and ID "
    "barely moves. A single rescaling cannot produce that pattern.",
    f"{EXT}/zero_shot.csv and {FUP}/zero_shot.csv", "response vs follow-up columns")
add("D-209", M, "'the token before the reply ... separates well where the instruction to lie is "
    "already in context, and does not help on HP-KR' -- VERIFIED per dataset",
    "Extended axis, `pre` read at L20: RepE 1.0000, IT 0.9273, ID 0.8995, CG 0.8714 -- all strong, "
    "and all are datasets where the instruction or setup is in the prompt. HP-KR reads 0.3855, "
    "inverted, against 0.8816 for the response mean. Exactly as described.",
    "src/probes/deception-detection/results/persona_positions_L20/reads.csv",
    "axis=extended, read=pre, per dataset", "high")
add("D-217", M, "'a learned combination of three components avoids the sign flip on IT that a fixed "
    "direction cannot' -- CONSISTENT with the committed matrices",
    "A k=3 probe has the freedom to reweight components per training set, which a 1-d feature does "
    "not (D-060). The specific IT cell was not isolated at Level 1; the mechanism is verified.",
    f"{EXT}/matrix_ext_pc3.csv", "IT column against matrix_ext_axis1.csv")

# ---------------------------------------------------------------- figures/captions
add("D-139", M, "36 personas x 5 variants x 30 questions -> 5,400 rollouts -> judge keeps 3,749 -> "
    "vectors -> L20/40/44 -- EVERY NUMBER EXACT",
    "The whole pipeline figure reproduces: 36x5x30 = 5,400 and the kept count recomputes to 3,749 "
    "(3,599 judged + default's 150).",
    "audit/cache/labpc/run-kd/scores/*.json", "count of entries == 3, plus default", "high")
add("D-065", M, "RESOLVED on the lab PC. All three caption figures EXACT at L40: sincere ignorance "
    "mean **-0.0844** -> -0.08; feigned deniers **+2.9037** -> 2.90; published deceptive "
    "**+0.0458** -> 0.05.",
    "Reproduced as `centred[idx] @ axis_unit('denial')` with `centred = personas - personas.mean(0)` "
    "over all 36, exactly as `make_figures.py:personas()` computes it. **The fourth clause checks "
    "too**: 'the published deceptive personas never reach the region the deniers occupy' -- their "
    "maximum is +0.885 against the deniers' minimum +2.150, so the groups do not overlap at all. "
    "ONE THING THE CAPTION DOES NOT SAY, and it matters: the two honest groups are **not together**. "
    "Plain assertion sits at **-1.2274** and sincere ignorance at **-0.0844**, a gap of **1.14** on "
    "an axis whose full span is 4.13 -- so sincere ignorance, the group that shares the deniers' "
    "wording, sits **28% of the axis closer to them** than plain assertion does. That is the "
    "wording pull made visible, and it is the quantity DF-25 predicts when the two groups are "
    "merged.",
    "LAB PC C:\\Users\\user4\\expLB\\run-kd\\vectors_s3\\*.pt",
    "layer 40; centred persona vectors projected on unit denial axis, group means", "high")
add("D-138", M, "'both contrasts built with the knowledge-denial personas stay near 0.9 at every "
    "layer; the published contrast inverts at layer 20' -- VERIFIED",
    "On HP-KR: 4v6 reads 0.890 / 0.919 / 0.908 and 12v13 reads 0.882 / 0.912 / 0.894 -- both near "
    "0.9 at all three. The published contrast reads 0.148 at L20 (inverted) and recovers only to "
    "0.593 / 0.589. The base probe swings from 0.109 (L20) to 0.936 (L40), which is the 'swings "
    "from inver[ted]' the caption describes.",
    ZS, "dataset=HP-KR, all arm columns, three layers", "high")
add("D-136", M, "'its rows are constant because a probe [on one feature can learn only a sign]' -- "
    "VERIFIED EXACTLY",
    "The rows are constant to 4 dp, not approximately. This caption is the THIRD place the draft "
    "discloses the dim=1 identity, after section 3.4 and Table 3's caption -- which together settle "
    "DF-93's 'label these arms zero-shot'. The left panel's 'transfers unevenly and inverts in "
    "places' is also exact: the raw probe's HP-KR column runs 0.06 to 0.95.",
    f"{EXT}/matrix_ext_axis1.csv and matrix_baseline.csv", "per-column distinct values", "high")
add("D-137", M, "'error bars are 95% bootstrap intervals over examples' and 'the Average excludes "
    "RepE' -- CONFIRMED",
    "Section 4 states the intervals are over examples with 1,000 resamples drawn once and shared, "
    "and that RepE is excluded from averages because it is the base probe's training set. Both are "
    "the right choices and both are load-bearing for DF-93, which does not apply to this draft "
    "because no interval is computed over matrix cells.",
    OPER, "*_lo / *_hi columns; \"Average (Liars' Bench)\" row excludes repe_honesty", "high")

# ---------------------------------------------------------------- B01 caveats
add("D-033", M, "'the knowledge-denial personas were written after we saw HP-KR fail' -- CORRECT",
    "Same disclosure as D-155, made twice, which is the right number of times for a caveat this "
    "load-bearing.", "kd/docs/DESIGN.md", "experiment motivation")
add("D-034", M, "'HP-KR's labels make a denial necessary for a lie, so its gains do not by "
    "themselves separate deception from the wording of a denial' -- CORRECT and understated",
    "The benchmark's Appendix D confirms the structurally empty fourth cell (claims-knowledge "
    "cases are discarded). And this project has measured the consequence: a denial-phrasing regex "
    "alone reads 0.9125-0.9597 on these rows. See D-152.",
    LBPDF, "Appendix D; cross-ref kd/docs/FINDINGS.md HP-KR lexical entries", "high")


# ---------------------------------------------------------------- write
def main():
    batch_of, meta = {}, {}
    for f in sorted(glob.glob(str(BATCHDIR / "B*.csv"))):
        b = os.path.basename(f)[:-4]
        if b in L2:
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["source"] != "draft PDF":
                continue
            batch_of[r["id"]] = b
            meta[r["id"]] = r

    missing = [i for i in batch_of if i not in V]
    extra = [i for i in V if i not in batch_of]
    if missing:
        print(f"!! {len(missing)} batch claims have no verdict: {missing}")
    if extra:
        print(f"!! {len(extra)} verdicts are not in a Level 1 batch: {extra}")

    rows = list(csv.DictReader(CLAIMS.open(encoding="utf-8")))
    fields = list(rows[0].keys())
    rows = [r for r in rows if r["id"] not in V]

    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    for cid, (verdict, l2, note, ev, key, imp) in V.items():
        m = meta.get(cid, {})
        rows.append({
            "id": cid, "claim_plain": m.get("claim_short", ""),
            "stated_numbers": m.get("numbers", ""), "source_ref": m.get("source_ref", ""),
            "evidence_file": ev, "evidence_key": key,
            "level1_value": l2, "level1_match":
                {"MATCHES HIS DATA": "yes", "PARTLY MATCHES HIS DATA": "partial",
                 "UNRESOLVED": "not locatable"}[verdict],
            "level2_value": "", "level2_method": "Level 1 -- number located in a result file",
            "version_used": "origin/paper @ ea1ce52b",
            "robustness": "single seed, his pipeline; three layers L20/L40/L44",
            "verdict": verdict, "confidence": "high", "importance": imp,
            "script": SCRIPT, "notes": note + f" [Phase 6 / 6A Level 1, {stamp}]",
            "batch": batch_of.get(cid, ""),
        })

    shutil.copy(CLAIMS, CLAIMS.with_suffix(".csv.before-phase6-level1"))
    with CLAIMS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    from collections import Counter
    c = Counter(v[0] for v in V.values())
    print(f"wrote {len(V)} Level-1 verdicts -> {CLAIMS.name}")
    print(f"total rows now: {len(rows)}")
    for k, n in c.most_common():
        print(f"   {n:3d}  {k}")


if __name__ == "__main__":
    main()
