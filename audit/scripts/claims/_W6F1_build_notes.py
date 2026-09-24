"""Build audit/claims/M-0xx.md for batch W6F1-memory from the recorded script outputs."""
import os, io, csv

OUT = 'audit/claims'
SRC = 'audit/phase2/out/W6F1'
MEM = ('C:/Users/SURAIYA/.claude/projects/'
       'C--Users-SURAIYA-Documents-STUDY-thesis-expLB/memory')


def cap(name):
    p = os.path.join(SRC, name + '.txt')
    return open(p, encoding='utf-8').read().rstrip() if os.path.exists(p) else '(no captured output)'


N = {}


def note(cid, title, claim, method, outputs, verdict, reason, extra=''):
    body = [f'# {cid} — {title}', '', '## The claim (as stated)', '', claim.strip(), '',
            '## What I did', '', method.strip(), '', '## Script output (pasted)', '']
    for label, txt in outputs:
        body += [f'**{label}**', '', '```', txt, '```', '']
    body += ['## Verdict', '', f'**{verdict}**', '', reason.strip(), '']
    if extra:
        body += ['## Notes for the lead', '', extra.strip(), '']
    N[cid] = '\n'.join(body)


# ---------------------------------------------------------------- M-001
note('M-001', 'the collaborator mean-pools; `pool_activations.py:84`',
     f'`{MEM}/abstract-number-is-our-kd-bank.md:29-31` — "(1) §5b\'s claim they scored per-token '
     'from 50 GB downloads — they **mean-pool** (`pool_activations.py:84`, Δ=0.0000 '
     'reproductions)".',
     'Read `pool_activations.py` on the collaborator branch with `git show` (no checkout) at BOTH '
     'the commit the memory note names (`f386a72`) and at branch head `45b16d9`, printed line 84 '
     'verbatim, and located the mean-pool statement. Then grepped his five result `summary.md` '
     'files for the "Δ = 0.0000" pooling-reproduction statements.',
     [('audit/scripts/claims/M-001.py', cap('M-001'))],
     'HOLDS',
     'At `f386a72` — the commit the memory note explicitly names as head — line 84 IS the '
     'mean-pool line `pooled.append(acts[row_mask].float().mean(0).squeeze(0).numpy())`. The '
     'substantive claim (they mean-pool, they do not score per token) is verified from his code, '
     'and three of his five result summaries independently record the Δ = 0.0000 reproduction of '
     '`results_table.csv` from the pooled matrix.',
     '**Corrects an earlier audit row.** `audit/claims.csv` row F-97 (batch P3-kdbank-cap) says '
     '"the entry says the mean-pooling line is pool_activations.py:84. In his file at both commits '
     'the pooling line is 123 … line 84 is `if acts is None:`". That is true at branch head '
     '(45b16d9, 2026-09-22) and at `a8eb902`, but NOT at `f386a72` (2026-09-21 23:10 +0600), which '
     'is the commit the memory note cites and the one that predates the note (modified '
     '2026-09-21T18:16Z = 2026-09-22 00:16 +0600 — i.e. written just after f386a72). The file was '
     'refactored after the note was written: 157 lines -> 183 lines, mean-pool moved 84 -> 123. '
     'So the memory note\'s citation is correct as written and needs no fix; the F-97 audit note '
     'and any DOC_FIXES entry derived from it should be amended.\n\n'
     'Minor: F-97\'s prose "every summary records Δ = 0.0000" is loose — 3 of 5 do '
     '(`persona_ood`, `persona_ood_L40`, `persona_ood_ext_L44`); `persona_ood_ext_L20` and '
     '`_L40` do not. The memory note\'s own wording ("Δ=0.0000 reproductions") is accurate.')

# ---------------------------------------------------------------- M-002
note('M-002', 'at L20 the 8v7 axis reads 0.6511 trained vs 0.7909 zero-shot',
     f'`{MEM}/abstract-number-is-our-kd-bank.md:35-37` — "at L20 the same 8v7 axis reads '
     '**0.6511 trained** vs **0.7909 zero-shot**, because a probe trained on the inverted HP-KR '
     'row flips sign."',
     'From the collaborator branch with `git show` (no checkout): `summary.csv` '
     '`off_diagonal_mean` for feature `axis1` (the TRAINED off-diagonal), and the mean of '
     '`zero_shot.csv` column `auroc_axis` over the 6 scored datasets (the ZERO-SHOT figure). '
     'Level 2: recomputed the trained off-diagonal myself from `matrix_axis1.csv` (30 cells). '
     'Repeated at L40 and L44 as the robustness check the note\'s mechanism predicts.',
     [('audit/scripts/claims/M-002.py', cap('M-002'))],
     'HOLDS',
     'L20: trained `axis1` off-diagonal 0.6510800 (my own 30-cell recomputation: 0.6510733, the '
     'difference being his 4-dp per-cell CSV) -> **0.6511**; zero-shot mean(auroc_axis) over the '
     '6 datasets 0.7908804 -> **0.7909**. Both exact to the precision quoted. The mechanism check '
     'also passes: at L40 and L44 the trained and zero-shot values coincide exactly '
     '(0.8318937 / 0.8260481 in both columns), so L20 is the only depth where the trained probe '
     'carries a flipped sign — consistent with "a probe trained on the inverted HP-KR row".',
     'Metric/arm discipline this note exists to enforce is itself correct here: `axis1` (paper 8v7) '
     'vs `ext_axis1` (12v13), trained off-diagonal vs zero-shot. `zero_shot.csv` at both depths '
     'has a 7th row `XXXX-20__plain` with empty AUROC columns, which I excluded; including it '
     'would break the mean. Anyone re-deriving these numbers must drop that row.')

# ---------------------------------------------------------------- M-003
note('M-003', 'crossgen smoke: 9-block means reproduce the earlier sweep',
     f'`{MEM}/crossgen-80-20-retrain.md:15-16` — "Smoke (L44 last) … 9-block 100%-target means '
     'reproduce the earlier sweep (raw 0.724 vs 0.720, pc 0.696 vs 0.700, axes 0.748 vs 0.766)."',
     'Level 1: `run-crossgen/smoke/last_L44_s42.json` key `smoke_checks.consistency_with_earlier_'
     'sweep`. Level 2: recomputed the "new_9block" quantity myself from the stored 10x10 AUROC '
     'matrices — off-diagonal mean over the 9 `block9` datasets (72 cells) — for all three probes, '
     'to find which probe the smoke check used. Then located the "earlier sweep" values in '
     '`run-coords/layer_sweep_passB_last.json`.',
     [('audit/scripts/claims/M-003.py', cap('M-003')),
      ('probe identification + earlier-sweep source (inline)',
       "mms probe, 9-block off-diagonal (72 cells), L44 last:\n"
       "  raw          0.723757   (smoke_checks new_9block 0.723757)\n"
       "  pc:deception 0.695789   (smoke_checks new_9block 0.695789)\n"
       "  axes         0.747779   (smoke_checks new_9block 0.747779)\n"
       "paper / lr_reg10 probes do NOT match; the smoke check is the mms probe.\n\n"
       "run-coords/layer_sweep_passB_last.json (probe mms, k 5, bank axes_vectors_s3_fp32.pt,\n"
       "pooling last), sweep entry layer 44:\n"
       "  {'layer': 44, 'raw': 0.7200369127180023, 'pc:deception': 0.6997536614763713,\n"
       "   'axes': 0.7657696890932673}\n"
       "-> exactly the three 'earlier_sweep' values.")],
     'HOLDS',
     'All six numbers are exact: 0.723757/0.720037 -> 0.724/0.720; 0.695789/0.699754 -> '
     '0.696/0.700; 0.747779/0.765770 -> 0.748/0.766. Level 2 reproduces the new_9block side to '
     '1e-6 from the per-cell matrices (mms probe, 72 off-diagonal cells over `block9`), and the '
     '"earlier sweep" side is byte-identical to the layer-44 entry of '
     '`run-coords/layer_sweep_passB_last.json`.',
     'Two wording points, neither an error in the quoted numbers. (1) "reproduce" is generous for '
     '`axes`: 0.748 vs 0.766 is a 0.018 gap, six times the raw and pc gaps — the note prints both '
     'numbers, so a reader can see it. (2) The same sentence\'s "~50 s per layer" is not what the '
     'file records: `seconds = 36.96` for this smoke run (that clause is line 15, outside this '
     'claim row).')

# ---------------------------------------------------------------- M-005
note('M-005', 'ID-trained row at L40, and the "controls are high on HP-KR" pair',
     f'`{MEM}/crossgen-80-20-retrain.md:23-25` — "…last pooling raw 0.778 / pc 0.654 / axes '
     '0.724; lr_reg10 mean raw 0.727. Controls are high on HP-KR (random 0.767, length 0.776)."',
     'Computed the instructed-deception ROW mean over the nine other datasets (off-diagonal, '
     '100%-target) from `run-crossgen/matrices/{mean,last}_L40_s42.json` for all three probes and '
     'six features, plus the single ID->HP-KR cells and, separately, the HP-KR column means, to '
     'settle which quantity the "controls" sentence names.',
     [('audit/scripts/claims/M-005.py', cap('M-005'))],
     'HOLDS',
     'Every number reproduces exactly. mms/mean ID-row: raw 0.8167, pc:deception 0.6492, axes '
     '0.7271 (w/o sycophancy 0.8456 / 0.6932 / 0.7764); mms/last: 0.7776 / 0.6536 / 0.7240; '
     'lr_reg10/mean raw 0.7271. The controls: ID->HP-KR **cell** random 0.7675, length 0.7756.',
     'UNDERSPECIFIED, same point already logged for M-004: "Controls are high on HP-KR" names the '
     'ID->HP-KR **cells of that same ID-trained row**, not HP-KR column or row means. For contrast, '
     'the HP-KR column means (as transfer target, mean pooling) are random 0.6706 (mms) / 0.6932 '
     '(paper) / 0.6979 (lr_reg10) and length 0.6225. A future session that reads the sentence as a '
     'column mean will get a different number. Recommend the note say "the ID->HP-KR cell".')

# ---------------------------------------------------------------- M-006
note('M-006', 'Figure-6-style panel: no persona variant beats raw at L40',
     f'`{MEM}/crossgen-80-20-retrain.md:27-29` — "Figure-6-style panel, L40, 10 datasets, mean '
     'off-diagonal change vs raw (pairs won /90): paper probe mean pooling, raw off-diag 0.665: '
     'random k2 −0.174 (19), datapca k2 −0.031 (45), pc:unified k2 −0.034 (42); k5: −0.084 / '
     '−0.001 / −0.006."',
     'Recomputed all of it from the per-cell matrices in `run-crossgen/matrices/mean_L40_s42.json` '
     '(ROOT store, 10 datasets, bank `axes_vectors_s3_fp32.pt`, k=5, seed 42, paper probe): '
     'off-diagonal means over the 90 cells and, for each feature, the count of cells strictly '
     'above raw.',
     [('audit/scripts/claims/M-006.py', cap('M-006'))],
     'HOLDS',
     'Exact on all eight numbers. raw off-diagonal 0.6652 -> 0.665. k=2: random@2 −0.1744 (19/90), '
     'datapca@2 −0.0310 (45/90), pc:unified@2 −0.0338 (42/90). k=5: random −0.0844, datapca '
     '−0.0006, pc:unified −0.0059. The paired significance file agrees cell for cell '
     '(`run-crossgen/significance.json`, see M-017).',
     'This is one of the load-bearing negative results for the reframed thesis and it reproduces '
     'without qualification. Note the store: raw and the matched `datapca` must be joined from the '
     'ROOT store, which is where they live here.')

# ---------------------------------------------------------------- M-007
note('M-007', 'the paper\'s 3B numbers, and where the unified-axis gains sit',
     f'`{MEM}/crossgen-80-20-retrain.md:29-31` — "The paper\'s 3B had axis +0.10, random ~−0.14, '
     'dataset PCA ~−0.12. Unified-axis gains sit where raw is weak (ID->IT +0.68 from 0.26, '
     'SB->CG +0.68, SB->IT +0.66, HP-C->AIL +0.55)."',
     'Repo side: the four named cells recomputed from `run-crossgen/matrices/mean_L40_s42.json` '
     '(paper probe, pc:unified@2 minus raw). Paper side: extracted the numeric annotations of the '
     'paper\'s own Appendix Figure 9 panels from the PDF text layer '
     '(`papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf`, page 12, "Deception '
     '· Llama 3.2 3B (Layer 14, k=1) [HPC excluded]") and took the off-diagonal mean over the 20 '
     'cells of each 5x5 Δ panel.',
     [('audit/scripts/claims/M-006.py (repo cells)', cap('M-006')),
      ('audit/scripts/claims/M-027.py (paper panels, last block)', cap('M-027'))],
     'HOLDS',
     'Repo cells exact: ID->IT raw 0.2638 -> unified@2 0.9434, Δ +0.6796 ("+0.68 from 0.26"); '
     'SB->CG +0.6798; SB->IT +0.6566; HP-C->AIL +0.5510. Paper side recomputed directly from the '
     'PDF rather than read off a figure: Axis PC Projection +0.1015 (-> +0.10), Random Subspace '
     '−0.1430 (-> ~−0.14), Dataset PCA −0.1175 (-> ~−0.12). All three match.',
     'Two comparability caveats the note does not state, both relevant to the reframe:\n'
     '1. **k does not match.** The paper panel these three numbers come from is the 3B **deception** '
     'axis at **k=1**; the repo numbers it is set beside are the **unified** axis at k=2 and k=5. '
     '(The paper uses k=3 only for sycophancy; k=5 appears nowhere in it.)\n'
     '2. The paper\'s panel excludes HPC and uses 5 datasets / 20 off-diagonal pairs; ours is 10 '
     'datasets / 90 pairs.\n'
     'Upgrade for the paper notes: `docs/papers/persona-coordinates-notes.md:456` records these as '
     '"read from Figure 10, panel A" (IMAGE-ONLY). They are in fact recoverable as text from the '
     'PDF and now VERIFIED by computation: +0.1015 / −0.1430 / −0.1175.')

# ---------------------------------------------------------------- M-008
note('M-008', 'paper16 regenerated on the shared question set, L40',
     f'`{MEM}/crossgen-80-20-retrain.md:50-53` — "L40 mean, paper probe, delta vs raw over 90 '
     'pairs: unified k5 **+0.017 (46/90)** … unified k2 −0.022, deception PCs **−0.054** (worse…), '
     'axes **−0.021**; last pooling all negative."',
     'Persona features from `run-crossgen/banks/paper16_shared/matrices/{mean,last}_L40_s42.json` '
     '(bank_name `paper16_shared`), raw baseline merged from the ROOT store '
     '`run-crossgen/matrices/{mean,last}_L40_s42.json` — which is how the pipeline itself builds '
     'these panels (bank folders hold persona features only). Off-diagonal mean over 90 cells and '
     'cells-above-raw counts.',
     [('audit/scripts/claims/M-008.py', cap('M-008'))],
     'HOLDS',
     'Exact. mean pooling: pc:unified +0.0170 (46/90), pc:unified@2 −0.0218, pc:deception −0.0536, '
     'axes −0.0214. Last pooling: all five persona features negative (−0.0352 to −0.1005). '
     '"the only positive unified number anywhere" is consistent with M-006 (ROOT unified k5 '
     '−0.0059) and with the bank table (own100 / hf100 both negative).',
     'Bank-version note (RF-15): the `paper16_shared` matrices record `bank = axes_vectors_s3.pt` '
     'while the ROOT store records `axes_vectors_s3_fp32.pt`. That is expected — paper16_shared is '
     'a separately regenerated bank — but the raw baseline the delta is taken against comes from '
     'the fp32 ROOT run, so the two arms are not from one file. The datasets and their order are '
     'identical, which is what the subtraction needs.')

# ---------------------------------------------------------------- M-009
note('M-009', 'dataset identity is 99.3% decodable, the label lift +3.2 points',
     f'`{MEM}/dataset-identity-dominates-variance.md:3` (description) — "Eval activations are '
     'dominated by dataset identity (99.3% decodable vs +3.2% for the label) — and the deception '
     'axis is nearly orthogonal to it…".',
     'Level 1 from `run-coords/validate_L40.json` (written by '
     '`src/assistant_axis_ndif/17_validate.py`, layer 40, mean pooling, cap 400 rows/dataset, '
     'bank `run-paper16/axes_vectors_s3_fp32.pt`). The majority baselines and the lift are '
     '`logger.info` only — read from the lab log `C:/Users/user4/expLB/logs/validate.log` '
     'read-only via `audit/scripts/labpc_run.py` + `p1_peek.py`, and cross-checked by '
     'reconstructing them arithmetically from the log header row counts.',
     [('audit/scripts/claims/M-009.py', cap('M-009')),
      ('lab log excerpt (logs/validate.log, UTF-16, 5,748 B, 2026-09-16 18:45, read-only)',
       "2026-09-16 18:45:00,234 - INFO - layer 40, mean pooling: 3624 rows, 10 datasets, 1399D/2225H\n"
       "2026-09-16 18:45:10,476 - INFO -    PC1 25.7%  PC2 8.3%  PC3 5.5%  top-10 55.5%\n"
       "2026-09-16 18:45:12,959 - INFO -    label   : 0.646  (majority baseline 0.614)  lift +0.032\n"
       "2026-09-16 18:45:12,959 - INFO -    dataset : 0.993  (majority baseline 0.110)  lift +0.883  over 10 datasets\n"
       "2026-09-16 18:45:12,959 - INFO -    |cos(deception, PC1..PC5)| = 0.016, 0.032, 0.094, 0.238, 0.007\n"
       "2026-09-16 18:45:12,961 - INFO -    share of the deception axis inside the top-10 PC subspace: 16.6%\n"
       "2026-09-16 18:45:12,965 - INFO -    cos(deception, assistant) = -0.371   AUROC(deception proj) = 0.739\n"
       "2026-09-16 18:45:12,994 - INFO -    |deception axis| at layer  0: 0.025 ... at layer 76: 5.386\n"
       "2026-09-16 18:45:12,994 - INFO - VERDICT: dataset identity is MORE decodable than the label")],
     'HOLDS',
     'dataset_acc = 0.993377 -> **99.3%**; label lift = 0.646247 − 0.6140 = **+0.032**. The whole '
     'supporting table reproduces too: variance 25.7 / 8.3 / 5.5%, PC label AUROCs 0.704 / 0.503 / '
     '0.507 / 0.595 / 0.590 (the note prints the orientation-corrected max(a, 1−a) that the script '
     'logs, while the JSON stores the raw value — PC1 0.2959 -> 0.704), best-separated datasets '
     '1.000 / 1.000 / 1.000 / 0.951 / 0.968, |cos| 0.016–0.238, 16.6% in the top-10 subspace.',
     'Wording caveat in the **description line only**: it sets an accuracy (99.3%) beside a lift '
     '(+3.2 pp). The matched pair is either the two accuracies (0.993 vs 0.646) or the two lifts '
     '(+0.883 vs +0.032). The body of the note (lines 23-26) states both correctly, so a reader '
     'who opens the note is safe; a reader who sees only the memory index line is not.\n\n'
     'Scope: `label_acc` is 3-fold CV over all 3,624 capped rows; `dataset_acc` is over the '
     'datasets with >=100 rows (here all 10). One dataset contributes only 24 of the 3,624 rows '
     '(9x400 + 24).')

# ---------------------------------------------------------------- M-010
note('M-010', 'label decodability 0.646 vs 0.614 majority, lift +0.032',
     f'`{MEM}/dataset-identity-dominates-variance.md:23` — "predicting the **label**: 0.646 '
     'accuracy against a 0.614 majority baseline, **lift +0.032**".',
     'Same run as M-009. `label_acc` is in `run-coords/validate_L40.json`; the majority baseline '
     'and the lift are printed only (17_validate.py:128-129). Read the lab log read-only, and '
     'independently reconstructed the baseline from the log header ("3624 rows … 1399D/2225H") as '
     'max(1399, 2225)/3624.',
     [('audit/scripts/claims/M-009.py', cap('M-009'))],
     'HOLDS',
     'JSON `label_acc` = 0.6462472 -> **0.646**. Reconstructed baseline 2225/3624 = **0.61397** -> '
     '0.614; lift 0.6462472 − 0.61397 = **+0.0323** -> +0.032. The lab log prints exactly '
     '"label : 0.646 (majority baseline 0.614) lift +0.032". Independent arithmetic and the log '
     'agree.',
     'The reconstruction also confirms N = 3,624 (label_acc x 3624 = 2342.0 exactly), which is what '
     'makes the companion dataset baseline 400/3624 = 0.110 in M-009.')

# ---------------------------------------------------------------- M-011
note('M-011', 'the deception axis still scores AUROC 0.739',
     f'`{MEM}/dataset-identity-dominates-variance.md:31-33` — "only **16.6%** of the axis lies '
     'inside the top-10 PC subspace. It still scores AUROC 0.739. The axis is reading something in '
     'the low-variance residual, not dataset identity."',
     'The 16.6% is in `run-coords/validate_L40.json` (`deception_in_top10`). The 0.739 is '
     'log-only: `17_validate.py:143-144` prints `roc_auc_score(y, X @ dec)` and writes it to no '
     'file. Read the lab log read-only.',
     [('audit/scripts/claims/M-009.py', cap('M-009'))],
     'HOLDS',
     '`deception_in_top10` = 0.16582 -> **16.6%**. The lab log line reads '
     '"cos(deception, assistant) = -0.371   AUROC(deception proj) = **0.739**". Both exact.',
     'Two things a future session should not over-read:\n'
     '1. The 0.739 is a **pooled** AUROC over the 3,624-row, 400-per-dataset capped sample, i.e. '
     'one ranking across ten corpora at once. It is not a per-dataset AUROC and it is not '
     'comparable to the crossgen zero-shot column figures (e.g. 0.644 for the same axis on HP-KR '
     'at L40).\n'
     '2. It is the **un-normalised** dot product `X @ dec` — `dec` is unit-normalised but X is not '
     'centred or scaled, so between-dataset offsets contribute. Given the same note\'s point that '
     'dataset identity is 99.3% decodable, a pooled AUROC is the one shape of this measurement '
     'most exposed to that confound.\n'
     'Neither point makes the number wrong; both make "it still scores 0.739" weaker evidence for '
     '"the axis is reading the low-variance residual" than the sentence suggests.')

# ---------------------------------------------------------------- M-013
note('M-013', 'the "0.870 lexical null" that the note tells every session to print',
     f'`{MEM}/hpkr-cannot-attribute-itself.md:46-47` — "**How to apply:** report HP-KR three ways '
     '— full / text-matched / knowledge-matched — and print the 0.870 lexical null beside every '
     'one."',
     'Recomputed the denial-regex AUROC against HP-KR\'s `deceptive` label directly from '
     '`results/subsets/harm-pressure-knowledge-report.csv`, with the regex copied from '
     '`src/assistant_axis_ndif/eval_datasets.py:363-364` and a rank-AUROC written from scratch '
     '(no sklearn), on (a) all 7,068 rows and (b) the `llama-v3.3-70b-instruct` rows only. Then '
     'checked the ledger for a superseding entry.',
     [('audit/scripts/claims/M-013.py', cap('M-013')),
      ('kd/docs/FINDINGS.md F-01 and F-16 (quoted)',
       "### F-01 - 2026-09-19 - HP-KR is largely a lexical task\n"
       "A single binary feature ... scores **AUROC 0.913** ...\n"
       "- source: results/subsets/harm-pressure-knowledge-report.csv, Llama-3.3-70B rows only\n"
       "  (2,139 of 7,068); ...\n"
       "- supersedes: an earlier figure of **0.870** that I computed over all 7,068 rows. The CSV\n"
       "  holds three generating models (llama-v3.3-70b 2,139 / mistral-small-3.1-24b 2,358 /\n"
       "  qwen-2.5-72b 2,571) and the probe pipeline reads only the Llama rows. The 0.870\n"
       "  described a row set that is never probed.\n\n"
       "### F-16 - HP-KR row counts actually probed\n"
       "- caveats: any HP-KR figure quoted over 7,068 rows describes a row set that is never\n"
       "  probed. F-01 and F-02's counts were originally computed that way.")],
     'REVISED -> F-01',
     'The number itself reproduces exactly on the row set the note names: over all 7,068 rows the '
     'regex scores **0.8699** (TPR 0.952, FPR 0.212, 68 deceptive rows missed) — the note\'s 0.870. '
     'But that row set pools three generating models (qwen 2,571 / mistral 2,358 / llama 2,139) and '
     'the probe pipeline reads only the 2,139 Llama-3.3-70B rows, where the same regex scores '
     '**0.9125** (TPR 0.892, FPR 0.067, 15 missed). F-01 in the ledger explicitly supersedes 0.870 '
     'with 0.913 for exactly this reason. So the instruction "print the 0.870 lexical null beside '
     'every one" now puts the wrong null beside every HP-KR probe number.',
     '**This is the one note in my batch whose error would actively mislead a future session.** '
     'The correct lexical null for any HP-KR number produced by this pipeline is **0.913**. '
     'Two further lines of the same note carry the same stale row set (not in my batch, flagged '
     'here): the 2x2 table (1,199 / 4,405 / 1,339 / 0, total 6,943) and the slice sizes '
     '"text-matched 2,538 rows" and "knowledge-matched 5,812 rows" are all over the 7,068-row CSV; '
     'F-16 gives the probed equivalents as **258** (124 dec / 134 hon) and **2,001**, from 2,139 '
     'rows with 139 deceptive. The note\'s structural conclusion — the fourth cell is empty, so '
     'knowledge, wording and deception are inseparable inside HP-KR — is unaffected.\n\n'
     'Note the internal contradiction across the memory set: `own-mistakes-correctness-over-speed` '
     '(M-028) records 0.870 as a caught mistake, while this note still prescribes printing it.')

# ---------------------------------------------------------------- M-016
note('M-016', 'the ~7-9% own100-vs-hf275 gap confounded judge with generation run',
     f'`{MEM}/judge-choice-barely-matters.md:20-23` — "I expected plan outcome 7b, \'persona '
     'vectors are judge-dependent\', from the ~7-9% own100-vs-hf275 gap. That comparison '
     'confounded judge with generation run: own100 and hf275 are different rollouts."',
     'Traced the "~7-9%" to its only written source (the plan file) and then looked for a result '
     'file behind it. Verified the confound itself from code: `31_fetch_hf_vectors.py` downloads '
     'the hf275 vectors from `lu-christina/assistant-axis-vectors`, so they are Lu et al.\'s '
     'rollouts and Lu et al.\'s judge (gpt-4.1-mini), while own100 is our generation judged by '
     'Llama-3.1-70B. Computed the only own100-vs-hf275 quantity the repo holds.',
     [('audit/scripts/claims/M-016.py', cap('M-016'))],
     'HOLDS',
     'The substantive half is VERIFIED: own100 and hf275 differ in rollouts AND judge AND role '
     'count, so any own100-vs-hf275 gap cannot isolate the judge. The plan says so itself '
     '(`harmonic-cuddling-kay.md:1012`: "own100 vs hf275 is an upper bound only — it confounds '
     'judge with generation"). The "~7-9%" is consistent with the only measured own100-vs-hf275 '
     'quantity in the repo — the Q1 assistant-contamination gap, 5.9 / 6.8 / 7.6 / 9.9 % relative '
     'at layers 20 / 22 / 40 / 60 — but I could not tie it to a specific file.',
     'The "~7-9%" is **DOC-ONLY**: every occurrence (plan lines 87, 129, 301, 1031, 1512, 1560) '
     'says "already partly measured" and names no result file, and the own100 bank itself is '
     'lab-only and was not re-read (`persona-coordinates-70b-results.md:535`). Low importance — it '
     'is a retired prediction, not a result — but if it is ever quoted in the writeup it needs a '
     'source or a recomputation.')

# ---------------------------------------------------------------- M-017
note('M-017', 'the matched control: unified axis − dataset PCA = −0.003',
     f'`{MEM}/matched-control-kills-the-gain.md:3` (description) — "Against a matched control at '
     'the same k, the unified persona axis is indistinguishable from PCA of the training '
     'dataset\'s own activations (−0.003); its apparent gain over raw is the drift toward chance '
     'any low-dim feature shows".',
     'Read every L40 / mean / paper / block "10" test out of `run-crossgen/significance.json` '
     '(the git working-tree copy), printing diff, pairs won, the dataset bootstrap CI and the BH '
     'q, and recorded the BH family size of the copy I read.',
     [('audit/scripts/claims/M-017.py', cap('M-017'))],
     'HOLDS',
     '`pc:unified@2 − datapca@2` = **−0.0028** (40/90 pairs won, dataset 95% CI [−0.236, +0.234]) '
     '-> −0.003. The other two rows of the note\'s table are exact too: `pc:unified@2 − raw` '
     '−0.0338 (42/90, [−0.231, +0.179]) and `pc:unified@2 − random@2` +0.1406 (57/90, '
     '[−0.104, +0.408]).',
     'Version: the git copy I read holds **192 tests**, so any BH q here is corrected over a '
     '192-test family; the lab copy of the same file holds 4,788 (RF-13). The note quotes no q, so '
     'nothing moves — but the q printed above (0.973) is the 192-family value and must not be '
     'reused against the lab file.\n\n'
     'One wording risk in the description: "its apparent gain over raw" can be read as saying the '
     'unified axis beats raw. It does not — `pc:unified@2 − raw` is **−0.034** on this grid. The '
     'note means the per-cell gains where raw transfers badly (the ID->IT +0.68 family in M-007). '
     'Worth tightening, because this note is load-bearing for the reframe.')

# ---------------------------------------------------------------- M-019
note('M-019', 'persona features beat raw only inside the instructed family',
     f'`{MEM}/matched-control-kills-the-gain.md:26-28` — "persona features beat raw only inside '
     'the instructed family (intent->intent 0.71-0.76 vs raw 0.68; intent->judge 0.74-0.79 vs '
     '0.66)".',
     'Read the per-construct block means straight out of `run-crossgen/summary.json` '
     '(records with layer 40, pooling mean, probe paper, seed 42, key `blocks`), printing every '
     'block x feature cell with its cell count.',
     [('audit/scripts/claims/M-019.py', cap('M-019'))],
     'HOLDS',
     'intent->intent (42 cells): raw **0.679** -> 0.68; persona features pc:deception 0.761, '
     'pc:unified 0.727, pc:unified@2 0.724, axes 0.713 -> range **0.713-0.761** = "0.71-0.76". '
     'intent->judge (7 cells): raw **0.660** -> 0.66; persona features 0.744 / 0.755 / 0.774 / '
     '0.793 -> **0.744-0.793** = "0.74-0.79". Both ranges exact over the four persona features '
     '{pc:deception, axes, pc:unified, pc:unified@2}.',
     '`judge->intent` (7 cells) is a third block where persona features beat raw (0.768-0.825 vs '
     'raw 0.785 — three of four beat it), so "only inside the instructed family" is right if '
     '"instructed family" means the intent/judge constructs, and slightly generous if it is read '
     'as "only the two blocks named". Block sizes are very uneven (42 / 7 / 2 cells), which the '
     'note does not say.')

# ---------------------------------------------------------------- M-020
note('M-020', 'persona features collapse where belief or sycophancy is involved',
     f'`{MEM}/matched-control-kills-the-gain.md:27-28` — "…and collapse anywhere belief or '
     'sycophancy is involved (0.22-0.45 vs raw 0.53-0.67)."',
     'Same block table as M-019. Took the seven blocks in which belief or sycophancy appears on '
     'either side (belief->belief, belief->intent, belief->judge, intent->belief, judge->belief, '
     'sycophancy as source, sycophancy as target) and computed the range of raw and of each '
     'persona feature over them.',
     [('audit/scripts/claims/M-019.py', cap('M-019'))],
     'PARTLY HOLDS',
     'The raw half is exact: over those seven blocks raw runs **0.526 to 0.670** -> "0.53-0.67". '
     'The persona half is not reproducible as stated. Over the four persona features the range is '
     '**0.220 to 0.612** (pc:unified reaches 0.527 at belief->belief and 0.612 at belief->intent; '
     'axes reaches 0.510 and 0.541). "0.22-0.45" is exactly the range of **pc:unified@2 alone** '
     '(0.220-0.453); it is also close to the range of the per-block minima (0.220-0.442). Under '
     'either of those readings it holds; as a statement about "persona features" it understates '
     'the top of the range by about 0.16.',
     'Consequence: the sentence reads as "every persona feature is between 0.22 and 0.45 on these '
     'blocks", which is false for `pc:unified` and `axes`. The qualitative claim — persona '
     'features lose to raw on every belief/sycophancy block — is TRUE in all 28 feature x block '
     'cells, and is the stronger and safer form of the sentence.')

# ---------------------------------------------------------------- M-021
note('M-021', 'dataset PCA is the mirror image on sycophancy',
     f'`{MEM}/matched-control-kills-the-gain.md:29` — "Dataset PCA is the mirror image '
     '(sycophancy as source 0.70 vs the axes\' 0.22)."',
     'Same block table as M-019: the `sycophancy as source` row (9 cells), for every feature.',
     [('audit/scripts/claims/M-019.py', cap('M-019'))],
     'PARTLY HOLDS',
     'Both numbers exist and are exact **at matched k = 2**: `datapca@2` = **0.701** -> 0.70 and '
     '`pc:unified@2` = **0.220** -> 0.22. But the feature the note calls "the axes\'" is not the '
     'one that reads 0.22: the feature literally named `axes` (the 4-d persona-axis projection) '
     'reads **0.384** on that row, and `datapca` at k=5 reads **0.673**. So the pair is right, the '
     'label is wrong.',
     'Low risk inside this note — its whole table is pc:unified@2 vs datapca@2, so the intent is '
     'clear — but "the axes" is the name of a different feature in every matrix file in the repo, '
     'and a future session that looks up `axes` will get 0.384 and think the note is wrong. '
     'Recommend the note say "pc:unified@2\'s 0.22".')

# ---------------------------------------------------------------- M-022
note('M-022', 'NDIF attention table row: B=2 T=3000 -> 4.30 GiB FAIL',
     f'`{MEM}/ndif-attention-memory-limit.md:16` — "B=2  T=3000  product 6,000   attn 4.30 GiB  '
     'FAIL".',
     'Recomputed the note\'s own formula `rows x heads x T^2 x 4 bytes` with the head count taken '
     'from the model config in the local HF cache, and converted to both GiB (2^30) and GB (1e9), '
     'for all four rows of the table.',
     [('audit/scripts/claims/M-022.py', cap('M-022'))],
     'HOLDS',
     'Head count verified: `num_attention_heads = 64` in the cached Llama-3.3-70B-Instruct '
     '`config.json`. B=2, T=3000 gives 2 x 64 x 3000^2 x 4 = 4,608,000,000 bytes = **4.292 GiB**; '
     'the note says 4.30, i.e. rounded up by 0.008. Product 6,000 = 2 x 3000 exact. The two OK rows '
     'of the same table are exact in GiB (2.146 -> 2.15; 0.610 -> 0.61), which confirms the table '
     'is in GiB.',
     'The FAIL/OK outcomes are log-only (the note says "measured 2026-09-15 against the live '
     'Llama-3.3-70B deployment"); I verified the arithmetic and the head count, not the OOM events. '
     'The structural claim the row exists to make — equal padded-token products land on opposite '
     'sides, so a token budget cannot express the limit — follows from the table as printed.')

# ---------------------------------------------------------------- M-023
note('M-023', 'NDIF attention table row: B=6 T=1869 -> "5.36 GiB" FAIL',
     f'`{MEM}/ndif-attention-memory-limit.md:17` — "B=6  T=1869  product 11,214  attn 5.36 GiB  '
     'FAIL  \\"Tried to allocate 5.00 GiB\\"".',
     'Same recomputation as M-022, in both GiB and GB.',
     [('audit/scripts/claims/M-022.py', cap('M-022'))],
     'PARTLY HOLDS',
     'The byte count is right and the product is right: 6 x 64 x 1869^2 x 4 = **5,365,495,296 '
     'bytes**, product 11,214 = 6 x 1869. But **the unit on this row is wrong**: 5,365,495,296 B is '
     '5.365 **GB** and **4.997 GiB**. The note labels it "5.36 GiB". The CUDA message quoted on the '
     'same line — "Tried to allocate 5.00 GiB" — is the correct GiB value, so the note contradicts '
     'itself within one line.',
     'This is the only unit slip in the table: the other three rows (2.15, 0.61, 4.30) and the '
     'later ai_audit figures (4.2-13.6 GiB for T = 4,211-7,543) are all correct in GiB, which I '
     'reproduced. Consequence if uncorrected: anyone calibrating `--attn_budget_gib` from this row '
     'sets a budget ~7% too high. Suggested fix: "attn 5.00 GiB".')

# ---------------------------------------------------------------- M-024
note('M-024', 'ai_audit: 16 scenarios could not move any conclusion (AUROC CI ~+-0.25)',
     f'`{MEM}/ndif-attention-memory-limit.md:40-42` — "16 scenarios could not move any conclusion '
     '(AUROC CI ~+-0.25), and tail-truncating would delete the email history the deception depends '
     'on."',
     'Read `run-coords/ai_audit/items.pt` locally: row count, distinct scenario ids, samples per '
     'scenario and the tokenised context lengths. Then computed the Hanley-McNeil 95% CI '
     'half-width for an AUROC at n1 = n2 = 8 (scenario-clustered) and at n1 = n2 = 40 (rows '
     'treated as independent).',
     [('audit/scripts/claims/M-024.py', cap('M-024'))],
     'HOLDS',
     'The dataset facts are exact: **80 rows = 16 distinct scenarios x 5 samples**, token lengths '
     'min **4,211**, median **6,028**, max **7,543** — the same three numbers the note gives, and '
     'the eager-attention cost at batch 1 is 4.23 / 8.66 / 13.57 GiB, matching the note\'s '
     '"4.2-13.6 GiB". The CI figure is the right order of magnitude: with 8 deceptive vs 8 honest '
     'scenarios the 95% half-width is **+-0.29 at chance, +-0.26 at AUROC 0.70, +-0.23 at 0.80**, '
     'so "~+-0.25" is a fair single number.',
     'Worth stating explicitly if this is ever written up: +-0.25 is the **scenario-clustered** '
     'interval. If the 80 rows were treated as independent the half-width would be +-0.10 to '
     '+-0.13, which would make the exclusion look unjustified. The clustered choice is the correct '
     'one (5 samples share a scenario) and is what makes the decision defensible.\n\n'
     'The "8 deceptive, 8 honest" split is not recoverable from `items.pt` (it stores only '
     '(row_id, token_ids) pairs); 16 scenarios x 5 samples is.')

# ---------------------------------------------------------------- M-025
note('M-025', 'F-94 replaces F-89\'s "78% / three-to-one" with ~2:1',
     f'`{MEM}/no-margin-from-one-layer.md:17` — "| F-94 | F-89\'s \\"78% / three-to-one\\" | ~2:1 '
     'over 16 layers; only *presence* clears |".',
     'Parsed F-94\'s own 16-layer table out of `kd/docs/FINDINGS.md` (lines 4114-4130) and '
     'recomputed the two means, their ratio, the retention median and the clear-counts from the '
     'table\'s asterisk marks. Level 2 on the underlying matrices was done earlier in this audit '
     '(`audit/claims.csv` row F-94, batch P2-pooled, read from the lab '
     '`run-crossgen/kd11/banks/{cap_s23,cap_ctrl}` matrices); its numbers agree with mine.',
     [('audit/scripts/claims/M-025.py', cap('M-025'))],
     'PARTLY HOLDS',
     '"~2:1 over 16 layers" is exact: mean content **+0.0339**, mean presence **+0.0704**, ratio '
     '**2.08 : 1** (the audit\'s own P2 Level-2 run over the matrices gives the same +0.0339 / '
     '+0.0704 at 16 layers and +0.0356 / +0.0730 at 21). Retention median 53%, range -278% to 93%, '
     'as the entry says. "only *presence* clears" is a simplification: by the entry\'s own '
     'asterisks, presence clears at **10 of 16** layers and content clears at **3 of 16** '
     '(L0, L12, L36).',
     'Two things for the lead. (1) F-94\'s prose contradicts F-94\'s table — the prose says content '
     'clears "4 of 16" and presence "9 of 16"; the table marks 3 and 10. This is already recorded '
     'in the audit\'s F-94 row; the memory note inherits the ambiguity by compressing it to "only '
     'presence clears". (2) The two layers where content clears and presence does not (L0, L12) '
     'are near-chance layers where every arm sits within 0.03 of raw, so the substantive reading '
     'is safe; L36 is not, and content clears there.')

# ---------------------------------------------------------------- M-026
note('M-026', 'the pooled probe peaks at L36, not L40',
     f'`{MEM}/no-margin-from-one-layer.md:18` — "| F-95 | pooled probe peaks at L40 | peak is L36 '
     '(0.8916 vs 0.8872) |".',
     'Recomputed the seed-mean of the `real` AUROC over all 63 falsification files '
     '(21 layers x seeds 42/7/13) in `audit/cache/labpc/run-premise/falsify/`, and ranked the '
     'layers.',
     [('audit/scripts/claims/M-026.py', cap('M-026'))],
     'HOLDS',
     'Seed-mean by layer puts **L36 = 0.8916** first, **L40 = 0.8872** second and L32 = 0.8857 '
     'third — exactly the two numbers quoted, gap +0.0044. 21 layers x 3 seeds, all present.',
     'Provenance: these are the audit\'s cached copies of the lab files. One of the 63, '
     '`falsify_L40_s13.json`, was rewritten during the audit\'s own interrupted-call incident '
     '(PROGRESS.md, 2026-09-23 02:19). The rewrite touched the `null`/`budget10` keys, not `real` '
     '(L40 s13 real = 0.8830 both before and after per the F-95 row), so this claim is unaffected. '
     'Related: the audit\'s F-91 row found that "two independent methods put the optimum at 32-36" '
     'does NOT survive, because the quoted L36 0.8942 there was seed 7 while L32/L40 were seed 42 '
     '— averaged over seeds, as here, L40 beats L32. That does not touch M-026, whose L36 > L40 is '
     'the seed-averaged statement.')

# ---------------------------------------------------------------- M-027
note('M-027', 'F-63\'s 3B margin +0.202 is the best of 7 layers; layer-mean +0.120',
     f'`{MEM}/no-margin-from-one-layer.md:19` — "| **F-96** | F-63\'s 3B margin **+0.202** at L16 '
     '\\"matches the paper\'s +0.22\\" | **L16 is the max of 3B\'s 7 layers**; layer-mean '
     '**+0.120** |".',
     'Repo side: `audit/phase2/layer_sensitivity.csv` (W1, built from the lab scale stores), rows '
     'with pooling=mean, probe=paper, seed=42, dataset_set=paper5, k=3; margin = '
     'best(axes, pc:deception) − datapca per layer; then L16\'s value, its rank and the layer mean. '
     'Paper side: extracted the numeric annotations of the paper\'s Appendix Figure 9 (3B '
     'deception, layer 14, k=1, HPC excluded) from the PDF text layer and computed the 20-cell '
     'off-diagonal means of the Axis-PC and Dataset-PCA panels.',
     [('audit/scripts/claims/M-027.py', cap('M-027'))],
     'HOLDS',
     '3B: L16 = **+0.2023**, rank **1 of 7** layers, layer-mean **+0.1197** -> +0.120, range '
     '[+0.0136, +0.2023]. Every element of the claim reproduces. The paper comparator also '
     'reproduces by computation rather than by eye: Axis PC Projection +0.1015 minus Dataset PCA '
     '−0.1175 = **+0.2190** -> "+0.22". For context the script also prints 8B (L16 +0.1147, rank 4 '
     'of 8, mean +0.1027) and 70B (L40 +0.0794, rank 13 of 21, mean +0.0791).',
     '**Caveat the memory note does not carry, and it matters for the reframe:** the paper panel '
     'that supplies the "+0.22" is the 3B **deception** axis at **k = 1** on 5 datasets with HPC '
     'excluded; F-63/F-96\'s +0.202 is at **k = 3**. So "matches the paper\'s +0.22" is a '
     'cross-k comparison. The note\'s point — that +0.202 is a layer-selection artefact — is '
     'independent of that and stands.')

# ---------------------------------------------------------------- M-028
note('M-028', 'the 0.870 lexical AUROC was the wrong row set, three models pooled',
     f'`{MEM}/own-mistakes-correctness-over-speed.md:28-29` — "Several of this session\'s '
     'corrections came from checking a number against one computed independently: the 0.870 '
     'lexical AUROC (wrong row set, three models pooled)…".',
     'Same independent recomputation as M-013: the denial regex from `eval_datasets.py:363-364`, '
     'a rank-AUROC written from scratch, run on all 7,068 rows and on the Llama-3.3-70B rows only, '
     'plus the model breakdown of the CSV.',
     [('audit/scripts/claims/M-013.py', cap('M-013'))],
     'HOLDS',
     'Both halves of the parenthesis are verified. "Three models pooled": the CSV holds '
     'qwen-2.5-72b-instruct 2,571, mistral-small-3.1-24b-instruct 2,358 and llama-v3.3-70b-instruct '
     '2,139 rows. "Wrong row set": the all-rows AUROC is **0.8699** while the row set the probe '
     'pipeline actually reads (Llama only, 2,139 rows) gives **0.9125**. The ledger records the '
     'same correction at F-01 ("supersedes an earlier figure of 0.870 … The 0.870 described a row '
     'set that is never probed").',
     'Cross-note inconsistency for the lead: this note treats 0.870 as a corrected mistake, while '
     '`hpkr-cannot-attribute-itself` (M-013) still instructs every session to print 0.870 beside '
     'every HP-KR number. The two memory notes disagree and only one can be right.')

# ---------------------------------------------------------------- M-030
note('M-030', 'against a shared reference axis the contamination falls 65-79%',
     f'`{MEM}/paper16-assistant-direction-not-lu-axis.md:29-31` — "1. Q1 is *stronger* against a '
     'shared reference axis: contamination falls 65-79%, not ~50% (|cos| < 0.3 is plan outcome 1a '
     'territory)."',
     'Rebuilt the whole four-column table of the note from local files: the published Assistant '
     'Axis from the HF cache (`lu-christina/assistant-axis-vectors`, llama-3.3-70b/'
     '`assistant_axis.pt`, (80, 8192) bf16); the "HF 22v22" deception contrast as '
     'mean(22 deception roles) − mean(22 honesty roles) over the published role vectors, using the '
     'committed grouping `src/assistant_axis_ndif/role_groups.json` and the contrast definition in '
     '`6_axes.py:69-75`; the paper16 axes from `run-paper16/axes_vectors_s3.pt`. Then computed the '
     'per-layer reduction 1 − |p16 dec vs published AA| / |HF 22v22 vs published AA|.',
     [('audit/scripts/claims/M-030.py', cap('M-030'))],
     'PARTLY HOLDS',
     'The table reproduces **exactly** at all four layers and all four columns: cos(p16 assistant, '
     'published AA) 0.366 / 0.306 / 0.372 / 0.300; p16_s3 deception vs published AA −0.168 / '
     '−0.141 / −0.226 / −0.180; HF 22v22 vs published AA −0.686 / −0.662 / −0.610 / −0.516; p16_s3 '
     'deception vs its own assistant −0.364 / −0.338 / −0.370 / −0.374. But the stated reduction '
     'range is wrong at the bottom: the per-layer reductions are **75.5 / 78.7 / 62.9 / 65.1 %**, '
     'i.e. **62.9-78.7%**, not "65-79%". The 65% floor is layer 60\'s value; layer 40\'s 62.9% is '
     'lower and was dropped. Using `axes_vectors_all.pt` instead gives 78.1 / 82.3 / 62.5 / 65.8 % '
     '— still a 62-63% floor.',
     '**Refutes an earlier audit report.** The third numeric column ("HF 22v22 grouping vs '
     'published AA") was reported unreproducible from anything committed. It is fully '
     'reproducible: the 22-vs-22 grouping is `src/assistant_axis_ndif/role_groups.json` (committed, '
     'deception 22 / honesty 22 — the own100 bank\'s grouping applied to the published vectors), '
     'all 44 role vectors are present in the local HF cache, and the contrast is `6_axes.py`\'s '
     '`class_contrast`. Script `audit/scripts/claims/M-030.py` re-derives it in seconds, CPU only.\n\n'
     'The same "65-79%" appears in an owner document — `docs/thesis/persona-coordinates-70b-'
     'results.md:540` — so a DOC_FIXES entry should carry the corrected **63-79%** to both places. '
     'The qualitative point (a shared reference makes Q1 look stronger than each bank\'s own '
     'assistant direction does, and |cos| < 0.3 at every layer: 0.168 / 0.141 / 0.226 / 0.180) is '
     'unaffected.')

# ---------------------------------------------------------------- M-031
note('M-031', 'HP-KR: the persona axes read 0.427 target / 0.300 source',
     f'`{MEM}/persona-axes-encode-register.md:19` — "| harm-pressure-knowledge-report | confident '
     'assertion vs epistemic disclaimer | denying knowledge one has | 0.427 target / 0.300 source '
     '|".',
     'Computed the HP-KR column mean (as transfer TARGET: mean over the 10 other sources) and row '
     'mean (as SOURCE: mean over the 10 other targets) for the `axes` feature from '
     '`run-crossgen/kd11/matrices/mean_L40_s42.json` — the 11-dataset grid, L40, mean pooling, '
     'paper probe, seed 42, bank `axes_vectors_s3_fp32.pt` (bank_name paper16), k=5 — the exact '
     'configuration the note\'s header names. Repeated across seven layers as a robustness check.',
     [('audit/scripts/claims/M-031.py', cap('M-031')),
      ('layer robustness (axes vs random, target/source, paper probe, mean pooling)',
       "L20  harm-p: axes 0.358/0.319 random 0.572/0.500   sycoph: axes 0.462/0.487 random 0.519/0.546\n"
       "L22  harm-p: axes 0.305/0.303 random 0.460/0.427   sycoph: axes 0.450/0.464 random 0.507/0.544\n"
       "L32  harm-p: axes 0.359/0.261 random 0.513/0.505   sycoph: axes 0.432/0.504 random 0.508/0.507\n"
       "L36  harm-p: axes 0.336/0.221 random 0.648/0.589   sycoph: axes 0.400/0.465 random 0.534/0.567\n"
       "L40  harm-p: axes 0.427/0.300 random 0.698/0.659   sycoph: axes 0.355/0.379 random 0.541/0.633\n"
       "L44  harm-p: axes 0.535/0.489 random 0.662/0.606   sycoph: axes 0.334/0.349 random 0.534/0.609\n"
       "L48  harm-p: axes 0.503/0.408 random 0.627/0.589   sycoph: axes 0.347/0.389 random 0.529/0.608")],
     'HOLDS',
     'Exact: `axes` on HP-KR reads **0.427 as target / 0.300 as source**. The companion row of the '
     'same table is exact too (sycophancy_mt 0.355 / 0.379), as are the note\'s other measured '
     'lines that I recomputed in passing: dropping sycophancy_mt lifts the off-diagonal mean by '
     '+0.0247 (raw), +0.0609 (axes) and +0.0693 (pc:deception) — the note\'s "+0.025 / +0.061 / '
     '+0.069, about 2.5x more" — and `pc:unified` reads 0.6715 off-diagonal against raw\'s 0.6623 '
     '("0.672 vs 0.662").',
     'Both inversions are below chance at every layer I checked (L20-L48), so the direction of the '
     'finding is layer-robust even though the specific values are L40. Note the depth structure: '
     'HP-KR `axes` runs 0.305-0.535 as target across those layers, so quoting 0.427 without the '
     'layer would be the mistake `hpkr-cannot-attribute-itself` warns about.')

# ---------------------------------------------------------------- M-033
note('M-033', 'a random direction beats the persona axes on both failing datasets',
     f'`{MEM}/persona-axes-encode-register.md:22` — "On both, a RANDOM direction beats the persona '
     'axes (HP-KR 0.698/0.659; SYC 0.541/0.633)."',
     'Same grid and configuration as M-031: `random` target/source means for HP-KR and '
     'sycophancy_mt from `run-crossgen/kd11/matrices/mean_L40_s42.json`, paper probe. Also read '
     '`make_features` in `src/assistant_axis_ndif/9_persona_probes.py:116-180` to establish what '
     '`random` and `axes` actually are.',
     [('audit/scripts/claims/M-031.py', cap('M-031')),
      ('feature definitions, 9_persona_probes.py:116-180 (quoted)',
       "if kind == 'random':\n"
       "    rng = np.random.default_rng(seed)\n"
       "    Q, _ = np.linalg.qr(rng.standard_normal((X.shape[1], k)))   # k = args.k = 5 here\n"
       "    return X @ Q\n"
       "if kind in ('axes', 'axes_resid'):\n"
       "    ... dirs = [dec, syc, uni, asst]     # 4 directions\n"
       "    return X @ np.stack(dirs).T")],
     'HOLDS',
     'Exact: `random` reads **0.698 target / 0.659 source** on HP-KR and **0.541 / 0.633** on '
     'sycophancy_mt, against `axes` 0.427 / 0.300 and 0.355 / 0.379. The ordering random > axes '
     'holds at every layer I checked (L20, L22, L32, L36, L40, L44, L48) on both datasets and in '
     'both directions, so the claim is layer-robust.',
     'One precision point. `random` here is a **trained k=5 random orthonormal subspace probe**, '
     'not "a random direction"; `axes` is a trained 4-d projection. The comparison is therefore '
     'like-for-like in training and near-matched in rank (5 vs 4), which is what makes it '
     'damaging — but it is NOT the zero-shot random-direction null used elsewhere in this project '
     '(`run-coords/zeroshot_nulls.json`, `audit/phase2/controls.csv`). A future session must not '
     'quote 0.698 as a zero-shot null.')


os.makedirs(OUT, exist_ok=True)
for cid, body in N.items():
    open(os.path.join(OUT, cid + '.md'), 'w', encoding='utf-8').write(body)
print(f'wrote {len(N)} notes to {OUT}/: {", ".join(sorted(N))}')
