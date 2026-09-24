"""Write audit/phase2/claims_batch_P7-register-sarcasm.csv (batch P7: register/sarcasm line)."""
import csv

H = ["id", "claim_plain", "stated_numbers", "source_ref", "evidence_file", "evidence_key",
     "level1_value", "level1_match", "level2_value", "level2_method", "version_used",
     "robustness", "verdict", "confidence", "importance", "script", "notes"]
R = []

R.append(dict(
    id="F-73",
    claim_plain="On our own 11-dataset store at L40 (mean pooling, paper probe, 110 off-diagonal cells) a length-only rule reaches 0.5555 off-diagonal AUROC vs raw 0.6623, datapca 0.6651 and random 0.5849; the positive class is longer in 9 of 11 datasets, 6 of the 8 in the reviewer's repo, median gap 3.7 tokens; mask is 283.2 vs 176.4, not the previously quoted 245 vs 96.",
    stated_numbers="raw 0.6623; datapca 0.6651; random 0.5849; length 0.5555; 9 of 11 longer; 6 of 8; median gap 3.7; mask 283.2/176.4; sandbagging 529.4/340.4",
    source_ref="kd/docs/FINDINGS.md:2629",
    evidence_file="run-crossgen/kd11/matrices/mean_L40_s42.json (git=lab=Drive) + C:/Users/user4/expLB/run-coords/passB/<ds>/rows.jsonl (lab-only)",
    evidence_key="results[feature,probe=paper].auroc off-diagonal mean; rows.jsonl span_len grouped by label",
    level1_value="n/a - kd/scripts/check_length_claim.py is print-only, no artifact",
    level1_match="n/a",
    level2_value="raw 0.6623 / datapca 0.6651 / random 0.5849 / length 0.5555 (n=110); longer 9 of 11; 6 of 8; median |gap| 3.689; mask 283.1831/176.4235; sandbagging 529.396/340.386",
    level2_method="own script: off-diagonal means from the matrix; span_len means per label over 11 rows.jsonl on the lab PC",
    version_used="matrix = git working tree (Phase 1: identical to lab and Drive); rows.jsonl = lab originals, mtimes 2026-09-15..09-19, untouched by the 09-22 wave; no BH q quoted",
    robustness="length reads 0.5555 under all three probes; on block9 (10 ds) length 0.5467 / raw 0.6870, on 9 ds 0.5319 / 0.6841; random@2 only 0.4910; single layer + single seed as stated",
    verdict="PARTLY HOLDS",
    confidence="HIGH - both halves recomputed from primary data, inputs pre-date the entry",
    importance="MEDIUM-HIGH",
    script="audit/scripts/claims/F-73_F-74.py; audit/scripts/claims/P7_lab.py length",
    notes="All primary measurements reproduce exactly. Two statements fail: (1) 'our stated 0.553 vs 0.665 is confirmed to three decimals' - measured 0.5555/0.6623, agreement is to 2 dp, and F-73's own closing line uses the correct 0.556/0.662; (2) 'on harm-pressure-choice the gap is exactly zero' - measured +0.0035 (1.0075 vs 1.0040). SEPARATE DOC BUG: five docs (ONBOARDING:78, NOVELTY_CHECK_PROMPT:110, kd/docs/DESIGN.md:176+202, docs/thesis/persona-coordinates-70b-results.md:497+1105) and the memory note still say '9 of 10' - the store gives 8 of 10 on block9 and 9 of 11 with sycophancy_mt; 9 of 10 matches neither."))

R.append(dict(
    id="F-74",
    claim_plain="corr(raw, random@2 - raw) = -0.7684 at L40 sits inside its own coupling null (analytic -0.7462, permuted -0.7456 [-0.7986,-0.6892], P=0.2135), so the -0.76 is arithmetic; and the uncoupled test (pc:unified@2 - datapca@2 regressed on raw) gives slope -0.100, r -0.067, mean margin +0.0063.",
    stated_numbers="-0.7684; -0.7462; -0.7456 [-0.7986,-0.6892]; -0.7462 [-0.8106,-0.6683]; P=0.2135; sd(raw) 0.2128; sd(random@2) 0.1898; -0.100 / -0.067 / +0.0063; +0.630 / +0.571 / +0.0290",
    source_ref="kd/docs/FINDINGS.md:2706",
    evidence_file="run-crossgen/kd11/matrices/mean_L40_s42.json",
    evidence_key="110 off-diagonal cells of results[raw|random@2|pc:unified@2|datapca@2, paper].auroc",
    level1_value="n/a - kd/scripts/ceiling_null.py is print-only, no artifact",
    level1_match="n/a",
    level2_value="-0.7684; analytic -0.7462; permuted mean -0.7455 [-0.8001,-0.6889]; P=0.2117; sd 0.2128 / 0.1898; slope -0.0998, r -0.0671, margin +0.0063; coupled row +0.6297 / +0.5711 / +0.0290",
    level2_method="own implementation (longhand Pearson r and OLS), B=20,000 permutations + 2,000 noise draws, seed 42",
    version_used="git working tree copy = lab = Drive; no BH q quoted",
    robustness="my simulated-noise null gives P=0.3005 vs the permuted 0.2117 - the entry quotes only one P without saying which null; substituting datapca(k=5) for datapca@2 in the coupled row gives +0.398 / +0.460 / -0.0028",
    verdict="HOLDS",
    confidence="HIGH - single file, independent reimplementation, exact on every deterministic quantity",
    importance="HIGH",
    script="audit/scripts/claims/F-73_F-74.py",
    notes="P=0.2117 vs the stated 0.2135 is Monte-Carlo noise (binomial s.e. ~0.003 at B=20,000); conclusion unchanged. Caveat (1) of the entry is correct: this is the paper16 unified bank, and the cap bank's +0.154 margin over the same matched control (F-77) is untouched by it."))

R.append(dict(
    id="F-75",
    claim_plain="Across all 80 layers the matched-slice axes probe is below chance to L20 (floor L19 0.355), plateaus at L27-L48 (0.61-0.75) and rises late to L77 0.764; it beats raw at 28/80 layers and datapca at 31/80; L40 reads 0.739 (+0.034 vs raw, +0.065 vs datapca) and the longest contiguous positive-margin run is 16 layers.",
    stated_numbers="floor L19 0.355; L0-L20 0.355-0.483; plateau L27-L48 0.61-0.75; 16 contiguous; L20 0.374 (-0.216/-0.226); L22 0.624; L40 0.739 (+0.034/+0.065); L44 0.732; L77 0.764 (+0.055/+0.130); 28/80 raw; 31/80 datapca",
    source_ref="kd/docs/FINDINGS.md:2772",
    evidence_file="C:/Users/user4/expLB/run-crossgen/kd11/passA11/banks/kd_s23/matrices/mean_L{0..79}_s42.json (axes) + .../passA11/matrices/mean_L{0..79}_s42.json (raw, datapca)",
    evidence_key="row 'harm-pressure-kr-matched' of results[feature, probe=paper].auroc, mean over the other 10 columns",
    level1_value="n/a - matched_profile.py and the lab-only margin.py are print-only",
    level1_match="n/a",
    level2_value="L19 0.3555; L0-L20 0.3555-0.4834; L27-L48 0.6114-0.7469; 16 contiguous (L29-L44); L20 0.3743 (-0.2161/-0.2257); L22 0.6237 (+0.0823/+0.0843); L40 0.7393 (+0.0336/+0.0655); L44 0.7321 (+0.0273/+0.0304); L77 0.7642 (+0.0553/+0.1296); beats raw 28/80, datapca 31/80",
    level2_method="own loader + margin code over the 160 matrices, run read-only on the lab PC",
    version_used="lab originals - root matrices mtime 2026-09-21 12:20-12:52, bank 12:22-13:28; the 09-22 16:05 A-kd-all re-run did NOT rewrite them; passAC11.status WAS rewritten 09-22 16:05 so the quoted gate line is the re-run's; no BH q quoted",
    robustness="same bootstrap with raw as control: L40 +0.0336, 95% [-0.1350,+0.1850], P(<=0) 0.3296 - also fails, so the negative verdict is not datapca-specific; single seed, single pooling (only _s42 exists)",
    verdict="HOLDS",
    confidence="HIGH - 160 original files, recomputed with my own code, every quoted number exact",
    importance="MEDIUM",
    script="audit/scripts/claims/P7_lab.py f7576",
    notes="Two clarifications, not errors: the 16 CONTIGUOUS positive-margin layers are L29-L44, while 'L27-L48' is the band where the axes level sits in 0.61-0.75 - the entry uses 'plateau' for both. Against raw the longest run is 19 layers (L26-L44). F-75's own verdict was withdrawn by F-76; only the profile is checked here."))

R.append(dict(
    id="F-76",
    claim_plain="Resampling the ten transfer datasets (B=10,000, seed 42) over the 80-layer profile, the margin at the pre-specified L40 over datapca is +0.0655 with 95% CI [-0.0693,+0.1887] and P(<=0)=0.161; the argmax has median L31, 95% [L5,L79], never lands on L40 and is inside L27-L48 only 31.1% of the time; the median fraction of positive layers is 0.46 [0.16,0.75].",
    stated_numbers="+0.0655; [-0.0693,+0.1887]; P=0.161; median argmax L31; [L5,L79] width 74; 0.0% on L40; 31.1% in plateau; fraction positive median 0.46 [0.16,0.75]; B=10,000 seed 42",
    source_ref="kd/docs/FINDINGS.md:2846",
    evidence_file="same 160 matrices as F-75 (run-crossgen/kd11/passA11/{matrices,banks/kd_s23/matrices}/mean_L*_s42.json)",
    evidence_key="80 layers x 10 transfer cells: axes(bank) minus datapca(root) on row harm-pressure-kr-matched; bootstrap over the 10 columns",
    level1_value="n/a - kd/scripts/layer_significance.py is print-only",
    level1_match="n/a",
    level2_value="+0.0655; [-0.0693,+0.1887]; P(<=0)=0.1611; argmax median L31, 95% [L5,L79] width 74, 0.00% on L40, 31.1% inside L27-L48; fraction positive median 0.4625, 95% [0.16,0.75]",
    level2_method="own bootstrap (numpy default_rng(42), index resampling of the 10 units), B=10,000, no refitting",
    version_used="lab originals (09-21 12:20-13:28); bootstrap computed in-script, NOT read from significance.json, so RF-13's regenerated BH q-values do not apply",
    robustness="with raw as control: +0.0336, [-0.1350,+0.1850], P 0.3296 - also fails; selection-priced-in best-layer margin +0.1296 [+0.0596,+0.3138]; resampling unit is only 10, which is why the interval is wide",
    verdict="HOLDS",
    confidence="HIGH - independent reimplementation reproduces the interval to 4 dp",
    importance="HIGH",
    script="audit/scripts/claims/P7_lab.py f7576",
    notes="THIS SETTLES THE F-75 vs F-77/F-81 CONFLICT. F-77/F-81 quote the kd slice as '37/80 (46%)' positive layers. The observed counts are 28/80 (vs raw) and 31/80 (vs datapca) - F-75's figures, reproduced here. 0.46 is F-76's bootstrap MEDIAN FRACTION (measured 0.4625) and 0.4625 x 80 = 37.0, so 37/80 is a bootstrap median fraction written as a layer count. F-75 is right. The cluster-permutation test F-76 still owes has not run in this batch; run-premise/cluster_perm_kd_matched.json (09-22 16:07) may be it - flagged, not chased."))

R.append(dict(
    id="F-77",
    claim_plain="For pc:capitulation@2 against datapca@2 on sycophancy_mt as target, the margin is positive at all 21 stored layers; at the pre-specified L40 it is +0.1542 with 95% dataset-level CI [+0.0913,+0.2113] and P(<=0)=0.0000; argmax median L36, 95% [L28,L60], on L40 in 3.9%; peak L32 +0.175; +0.108 still at L76; vs raw +0.005..+0.139.",
    stated_numbers="21/21 positive; run 21; +0.1542; [+0.0913,+0.2113]; P=0.0000; argmax [L28,L60] median L36, 3.9% on L40; peak ~L32 +0.175; L12 +0.012; L76 +0.108; vs raw +0.005..+0.139; kd slice '37/80 (46%)'",
    source_ref="kd/docs/FINDINGS.md:2923",
    evidence_file="C:/Users/user4/expLB/run-crossgen/kd11/banks/cap_s23/matrices/mean_L{0,4,...,76}_s42.json (21 files)",
    evidence_key="column sycophancy_mt of results[pc:capitulation@2|datapca@2|raw, paper].auroc, mean over the 10 source rows",
    level1_value="n/a - cap/scripts/cap_layer_profile.py is print-only",
    level1_match="n/a",
    level2_value="21/21 positive vs datapca@2 and vs raw; L40 +0.1542, 95% [+0.0913,+0.2113], P(<=0) 0.0000; argmax median L36, 95% [L28,L60], on L40 3.85%; peak L32 +0.1747; L12 +0.0116; L76 +0.1081; vs raw +0.0045..+0.1388",
    level2_method="own loader + bootstrap (default_rng(42), 10 source datasets resampled, B=10,000)",
    version_used="lab originals (mtimes 2026-09-21 03:48-13:44); cap_layers.status overwritten 09-22 15:50-15:52 and cap_s23/significance.json regenerated 09-22 16:01 - neither is read by the entry or by me, so RF-13 does not bite; bank axes_s23.pt / bank_name=cap_s23",
    robustness="the 21/21 is specific to the sycophancy_mt column: on the full 11-dataset off-diagonal (audit/phase2/layer_sensitivity.csv) cap@2 is NEGATIVE vs raw at L12/L20/L22/L24 and the L40 margin over raw is +0.0983, which matches F-67's +0.098 that F-77 cites as its bound; F-87 confirms the L40 margin at four seeds",
    verdict="PARTLY HOLDS",
    confidence="HIGH for the cap numbers (exact); HIGH for the 37/80 error (arithmetic identity with F-76's 0.4625)",
    importance="HIGH",
    script="audit/scripts/claims/P7_lab.py f77",
    notes="Every cap-bank number reproduces exactly. The comparison column's 'kd matched slice 37/80 (46%)' does NOT: the observed counts are 28/80 vs raw and 31/80 vs datapca (F-75), and 37 is F-76's bootstrap median fraction 0.4625 x 80. The error runs against F-77's own argument (31/80 makes the contrast stronger), but it puts a bootstrap statistic and a layer count in the same column. F-81 carries the same figure."))

R.append(dict(
    id="F-87",
    claim_plain="At L40 on sycophancy_mt the cap bank's pc:capitulation@2 reads 0.7159/0.7132/0.7186/0.7136 at seeds 1/7/13/42, beating raw by +0.111 to +0.120 (spread 0.009) and the matched control by +0.149 to +0.177 on every seed.",
    stated_numbers="cap@2 0.7159/0.7132/0.7186/0.7136; raw 0.6050/0.5962/0.5988/0.6013; datapca@2 0.5545/0.5638/0.5418/0.5594; vs raw +0.111..+0.120 spread 0.009; vs control +0.149..+0.177",
    source_ref="kd/docs/FINDINGS.md:3532",
    evidence_file="C:/Users/user4/expLB/run-crossgen/kd11/banks/cap_s23/matrices/mean_L40_s{1,7,13,42}.json",
    evidence_key="results[feature, probe=paper].auroc column sycophancy_mt, mean over the 10 source rows",
    level1_value="n/a - the table came from a loose lab-only print-only script capseedread.py (not in git, not on Drive)",
    level1_match="n/a",
    level2_value="0.7159/0.7132/0.7186/0.7136; raw 0.6050/0.5962/0.5988/0.6013; datapca@2 0.5545/0.5638/0.5418/0.5594; vs raw +0.1110..+0.1198 (spread 0.0088); vs control +0.1494..+0.1767",
    level2_method="own loader over the four seed matrices, read-only on the lab PC",
    version_used="lab originals - s1/s7/s13 mtime 2026-09-21 17:28 (before commit 1155730 at 17:30), s42 03:49; cap_seeds.status overwritten 09-22 16:00-16:01 but the matrices were not; no BH q quoted",
    robustness="4 seeds at one layer only; depth covered separately by F-77 (21/21 at seed 42); the seed spread 0.0088 is an order of magnitude below F-66's CI width 0.135, as claimed",
    verdict="HOLDS",
    confidence="HIGH - all twelve cells exact",
    importance="MEDIUM-HIGH",
    script="audit/scripts/claims/P7_lab.py f87",
    notes="F-87's postscript about the --extra_* trap (extra-seed matrices written with no paper probe) is not present in the current files: all four contain the paper probe with the full cap feature set. Wording nit: the entry's opening contrasts 'five seeds' with 'one', then gives a table of four seeds."))

R.append(dict(
    id="F-88",
    claim_plain="Three papers were read in full and their bearing on our work recorded: Kumar (arXiv:2605.27958) rejects the single-direction hypothesis and contradicts the draft title, Kirch et al. (arXiv:2511.17408) give the text-evident/text-ambiguous taxonomy and the clustering argument, and Yoo & Skapars (arXiv:2609.02893) state our near-orthogonality result in their own vocabulary.",
    stated_numbers="k=1 captures 0.61-0.80 AUROC; >=0.998 clean; style-augmented 0.979-0.983; 8 stylistic shifts; peak-layer CIs span nearly the full network; TF-IDF <0.5 at 12B/27B; eight behaviours; Appendix N mixed-effects",
    source_ref="kd/docs/FINDINGS.md:3568",
    evidence_file="papers/Pressure-Testing Deception Probes in LLMs Scaling, Robustness, and the.pdf; papers/The Impact of Off-Policy Training Data.pdf; papers/Probe Generalization as Subspace Selection for OOD Deception Detectio.pdf",
    evidence_key="abstracts, contribution bullets, Table 3 caption (Kumar), Appendix N (Kirch), abstract (Yoo & Skapars)",
    level1_value="every quoted sentence located in the PDFs, most of it verbatim",
    level1_match="yes, except the two venue labels",
    level2_value="n/a - a reading, not a measurement",
    level2_method="pdftotext -enc UTF-8 over the three PDFs, then targeted search for each quoted phrase",
    version_used="git working-tree PDFs (first committed 4d0d046 2026-09-22 02:46); Kumar arXiv:2605.27958v1 27 May 2026, Kirch arXiv:2511.17408v4 19 Apr 2026, Yoo & Skapars arXiv:2609.02893v1 1 Jul 2026",
    robustness="n/a",
    verdict="HOLDS",
    confidence="HIGH for the quotations (read during this audit); LOW for the two venue labels (absent from the documents)",
    importance="HIGH",
    script="pdftotext + grep (commands recorded in audit/claims/F-88.md)",
    notes="PHASE-1 DATING FLAG RESOLVED: the PDFs' working-tree mtimes are 2026-09-21 17:20/17:23/17:24, 19-23 min BEFORE F-88's commit d55dd4d at 17:43. The 9-hour gap Phase 1 saw is a delayed git add, not a false claim. NOT VERIFIED: 'GEM @ ACL 2026' (Kumar) and 'Findings of ACL 2026' (Kirch) appear nowhere in the PDFs; Yoo & Skapars' 'ICML 2026 Workshop' IS in the PDF. Minor attribution shift: the abstract attaches 'nearly matches probes trained directly on the test distribution' to PC-subspace projection, while the judge-selected variant closes 78%/25% of the baseline-to-oracle gap. The 0.883 and cosine +0.051 re-quoted here belong to F-86 and are not checked in this card. None of the three PDFs has a notes file under docs/papers/."))

R.append(dict(
    id="F-92",
    claim_plain="Over 100 ironic-arm rollouts (5 instruction variants x 20 questions), the fraction of responses carrying at least one irony marker falls 0.85/0.15/0.10/0.00/0.00 from v1 to v5, with marker counts exclamation 27, rhetorical_q 13, stock_irony 11, allcaps 4 - so unmarked deadpan is producible at instruction strength v4 or above.",
    stated_numbers="0.85/0.15/0.10/0.00/0.00; exclamation 27; rhetorical_q 13; stock_irony 11; allcaps 4; n=100",
    source_ref="kd/docs/FINDINGS.md:3934",
    evidence_file="cap/run-sarc-pilot/marker_report_partial.json and responses/pc_ironic_critic.jsonl (lab-only; md5-verified copy in audit/cache/labpc/cap/run-sarc-pilot/)",
    evidence_key="markers.pc_ironic_critic.per_variant; per_marker.pc_ironic_critic",
    level1_value="per_variant [0.85,0.15,0.1,0.0,0.0]; exclamation 27, rhetorical_q 13, stock_irony 11, allcaps 4; n 100",
    level1_match="yes, exact",
    level2_value="recomputed from the 100 raw responses: 0.85/0.15/0.10/0.00/0.00; exclamation 27, rhetorical_q 13, stock_irony 11, allcaps 4",
    level2_method="own script re-using the scorer's MARKERS regexes and strip_question over the raw JSONL",
    version_used="marker_report_partial.json mtime 2026-09-21 21:31 (before commit 57cfa3a at 21:32) - the ORIGINAL; responses 21:26 originals; lab-only, no Drive backup (RF-10), now cached md5-verified by the audit",
    robustness="the eight-regex detector misfires on non-ironic text: recomputing the other two arms gives allcaps 23 and reveal_aside 13 on the sincere arm, so a marker rate is only interpretable within the ironic arm - which is F-92's own caveat (2); 20 questions, one temperature, one seed",
    verdict="HOLDS",
    confidence="HIGH - original artifact and recompute from raw both exact",
    importance="MEDIUM",
    script="audit/scripts/claims/F-92_F-93_F-105.py",
    notes="F-92's hedge is correct and should survive into any write-up: the marker check cannot distinguish deadpan sarcasm from a praise instruction simply complied with, and the entry says so. Its caveat (1) - the decisive separability check had no result yet - is what F-93 then answered negatively."))

R.append(dict(
    id="F-93",
    claim_plain="Bag-of-words logistic regression grouped by question id separates ironic from sycophantic responses at 0.953 raw / 0.947 stripped over all variants, 1.000 on v1, and still 0.919/0.921 on the marker-free v4+v5 subset, on ordinary content vocabulary - so a direction fit on these personas would be a register axis; the amendment's polarity check passes at 100%/96%/100% >=2.",
    stated_numbers="0.953/0.947 n=200; 1.000/1.000 n=40; 0.919/0.921 n=80; 0.997/0.996 n=80; marker rates ironic [.85,.15,.10,0,0], sincere [.60,.15,.15,.25,.05], sycophant [.85,1.0,.50,0,.05]; polarity 2.59/100%, 2.44/96%, 2.37/100%",
    source_ref="kd/docs/FINDINGS.md:3992 (+ its 23:05 amendment)",
    evidence_file="cap/run-sarc-pilot/marker_report.json (REGENERATED 2026-09-22 15:55) over original responses/*.jsonl (09-21 21:26-21:40) and scores/*.json (09-21 21:42-21:44)",
    evidence_key="lexical_subsets[pair,subset].{auroc_raw,auroc_stripped,n}; markers.<arm>.per_variant; polarity.<arm>",
    level1_value="0.9525/0.9472 n200; 1.0/1.0 n40; 0.91938/0.92125 n80; 0.99688/0.99625 n80; polarity 2.59/1.00, 2.44/0.96, 2.37/1.00",
    level1_match="yes, exact - but the file post-dates the entry, so Level 1 alone proves nothing",
    level2_value="A (repo functions) and B (independent sklearn pipeline) both give 0.9527/0.9472, 1.0000/1.0000, 0.9194/0.9213, 0.9969/0.9963; all marker rates and the full polarity table reproduce",
    level2_method="two passes over the original 300 responses: the scorer's own strip_question/bow/grouped_auc, and my own echo-stripping + CountVectorizer(min_df=3) + LeaveOneGroupOut logistic regression",
    version_used="marker_report.json is the 09-22 15:55 regeneration (RF-13/RF-14) but the inputs are 09-21 originals; lab-only data (RF-10) now md5-cached by the audit; no layer/bank/BH q involved",
    robustness="NEW: bootstrapping the 20 question ids (B=2,000, seed 42) on the out-of-fold scores gives the missing CI for the decisive 0.921 - 95% [0.8719, 0.9669], P(<=0.65 pre-registered ceiling) = 0.0000, so F-93's caveat (1) is discharged and the gate fails with room to spare",
    verdict="HOLDS",
    confidence="HIGH - two independent implementations agree to 4 dp on original inputs",
    importance="HIGH",
    script="audit/scripts/claims/F-92_F-93_F-105.py",
    notes="The gate-failure story checks out in the preserved status files: sarc_pilot.status.2026-09-21-buggy-verdict has 'VERDICT PRODUCIBLE' one line above 'stripped=0.947'. The 0.947 in the 09-22 02:29 status and the 0.921 in the 15:55 one are the same data under different gate subsets (all-variant vs unmarked v4+v5), both in my recompute. PROVENANCE NUANCE: F-93's ironic feature list is the UNSTRIPPED fit - it quotes 'definitely', which is a MARKER_TOKEN and cannot appear in a stripped model; F-105's corrected list matches the stripped fit. The register reading is the same either way."))

R.append(dict(
    id="F-105",
    claim_plain="F-93's four separability rows and both feature lists re-derive exactly from the raw responses; none of them was persisted by the saved tooling; and F-93's polarity amendment is withdrawn because the sincere arm declines to the same 2.10 over the same range.",
    stated_numbers="0.953/0.947, 1.000/1.000, 0.919/0.921, 0.997/0.996; marker_report.json holds only auroc_raw 0.9525 / auroc_stripped 0.9472 n=200; ironic 2.85->2.10, sincere 2.85->2.10, sycophant 2.45->2.80",
    source_ref="kd/docs/FINDINGS.md:4934 (commit 6c9a5cf 2026-09-22 03:25)",
    evidence_file="cap/run-sarc-pilot/responses/*.jsonl + scores/*.json (originals); cap/run-sarc-pilot/marker_report.json (09-22 15:55); git history of cap/scripts/score_sarcasm_markers.py",
    evidence_key="lexical_subsets; lexical_features.{ironic_side,sycophant_side}[0:12]; polarity.<arm>.per_variant",
    level1_value="the current marker_report.json DOES hold all six subset rows, both 15-word feature lists with coefficients and the per-variant polarity table",
    level1_match="values yes; the 'not persisted' statement no longer describes the file",
    level2_value="0.9194/0.9213 etc. all four rows exact; ironic top-12 and sycophant top-12 match F-105 word for word and in order; polarity table exact (ironic 2.85/2.85/2.60/2.55/2.10, sincere 2.85/2.25/2.40/2.25/2.10, sycophant 2.45/2.25/2.55/2.15/2.80)",
    level2_method="same two-pass recompute as F-93, plus git log/diff on the scorer",
    version_used="inputs are 09-21 originals; artifact is the 09-22 15:55 re-score; the persistence fix landed in commit 6c9a5cf - the SAME commit that added F-105",
    robustness="F-105's claim that F-93's verdict is unaffected is supported: 0.9213 with a question-level 95% CI [0.872,0.967] and P(<=0.65)=0.0000",
    verdict="PARTLY HOLDS",
    confidence="HIGH - recompute exact; the persistence question settled from git history and the current file",
    importance="MEDIUM",
    script="audit/scripts/claims/F-92_F-93_F-105.py",
    notes="Parts 1 and 3 hold exactly. Part 2 ('none of them is persisted') was true of the artifact it inspected but is SUPERSEDED by the fix committed alongside it (6c9a5cf adds lexical_subsets + lexical_features to the scorer) plus the 09-22 15:55 re-run, which wrote them into marker_report.json. Minor overstatement: 'all ten words F-93 quoted appear in the top-25' - 11 of 12 ironic words do; 'definitely' is a MARKER_TOKEN and is absent from the stripped model (14 of 14 sycophant words are present)."))

with open("audit/phase2/claims_batch_P7-register-sarcasm.csv", "w", encoding="utf-8",
          newline="") as f:
    w = csv.DictWriter(f, fieldnames=H)
    w.writeheader()
    for r in R:
        w.writerow(r)
print("rows written:", len(R))
