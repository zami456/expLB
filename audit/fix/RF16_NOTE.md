# RF-16 fixed: the 70B scale rung, measured like-for-like (owner-authorised item 1, 2026-09-23)

## The problem
`run-crossgen/scale_70b/` mixes two persona banks under one label:
- **layer 40** (written 09-20 23:42, the cell F-63 uses) came from the **paper16** bank;
- **every other layer** (the sweep of 09-22 00:52–01:33 behind F-100/F-101/F-102) came from the **36-persona knowledge-denial bank**.

Both matrices record `bank = axes_vectors_s3.pt`, `bank_name = paper16_70b`, because `run-paper16/` and `run-kd/` each contain a file of that name. F-100 states its sweep used "the same bank and datasets as the existing L40 cell" — it did not.

Identified by fingerprint: the same bank on the same activations must give identical zero-shot AUROCs. The 09-22 sweep's values are identical (10/10, |Δ|=0) to `kd11/banks/kd`; the L40 cell matches the paper16 matrices to ≤0.0003. Evidence: `audit/phase1/bank_fingerprint_scale70b*.txt`.

So the 3B and 8B rungs (paper16 banks) were being compared against a 70B rung built on a different, kd-augmented bank.

## The fix
I re-ran **your own** `27_crossgen.py` with everything identical to the 09-22 sweep — the passB activations, the paper's five datasets, k=3, both poolings, all 21 stored layers — changing only the bank to `run-paper16/axes_vectors_s3_fp32.pt`. CPU only, no NDIF calls. Output went to a **new** folder, `run-crossgen/scale_70b_paper16fp32/` (42 matrices); nothing existing was touched.
Driver: `audit/scripts/fix_rf16_run.ps1`. Recompute: `audit/scripts/fix_rf16_recompute.py` (same off-diagonal recipe as `kd/scripts/paper_comparison.py`). Raw output: `audit/fix/rf16_recompute_mean.txt`.

## Result: "best persona feature" vs raw and vs the matched control
Mean pooling, the paper's five datasets, k=3, `probe=paper`, best = per-layer max of {axes, pc:deception}, exactly as F-100 defines it. F-100's layer set (19 layers, excluding L72/L76):

| rung | bank | layers | raw | best − raw | best − dataset-PCA |
|---|---|---|---|---|---|
| 3B | paper16_3b | 7 | 0.6989 | +0.0328 | **+0.1197** |
| 8B | paper16_8b | 8 | 0.7149 | +0.0089 | **+0.1027** |
| 70B **as F-100 ran it** | kd (36 personas) | 19 | 0.7524 | +0.0079 | **+0.0751** |
| 70B **like-for-like** | paper16 fp32 | 19 | 0.7524 | **−0.0131** | **+0.0541** |

**The method is validated:** with the kd bank my recompute reproduces F-100's published figures exactly — +0.120 / +0.103 / +0.075, and +0.0079 against raw.

All 21 layers, both poolings:

| rung | mean: best−raw | mean: best−dPCA | last: best−raw | last: best−dPCA |
|---|---|---|---|---|
| 3B | +0.0328 | +0.1197 | −0.0178 | +0.0745 |
| 8B | +0.0089 | +0.1027 | −0.0742 | +0.0090 |
| 70B kd | +0.0070 | +0.0791 | +0.0333 | +0.0945 |
| 70B paper16 | −0.0171 | +0.0550 | +0.0069 | +0.0681 |

## What this does to your findings
1. **F-100's headline conclusion survives and gets stronger.** The monotone decline with model size holds like-for-like, and is steeper: **+0.120 → +0.103 → +0.054**, instead of the reported +0.120 → +0.103 → +0.075.
2. **One F-100 sentence must change.** "Against raw activations the 70B persona advantage is essentially nil" (+0.0079) becomes **negative**: like-for-like it is **−0.0131**. At 70B, on the paper's five datasets, the persona features are *worse* than raw activations on average across depth.
3. **F-100's source line is wrong** where it says the sweep used the same bank as the L40 cell. It used the kd bank.
4. **F-63 is unaffected**: its 70B cell is the L40 paper16 cell, so its three-rung comparison was already bank-consistent.
5. **F-101/F-102** (the pooling-dependence entries) need the same treatment: under `last` pooling the like-for-like 70B is +0.0069 vs raw and +0.0681 vs dPCA, so the pooling contrast remains, with smaller margins.

## Suggested ledger entry (NOT yet written — your ledger, your call)
> **F-110 · 2026-09-23 · CORRECTION to F-100–F-102: the 70B sweep used the kd bank, not paper16; like-for-like the 70B rung is +0.054 vs dataset-PCA and −0.013 vs raw.**
> The 09-22 70B k=3 sweep was run with `run-kd/axes_vectors_s3.pt` (36 personas), while the 3B/8B rungs use their paper16 banks and the L40 cell of F-63 uses `run-paper16/axes_vectors_s3.pt`; all three matrices carry `bank_name=paper16_70b`, so the mix was invisible. Re-running `27_crossgen.py` unchanged except for the bank (`run-crossgen/scale_70b_paper16fp32/`, 42 matrices) gives, over F-100's 19 layers: best−dPCA +0.1197 / +0.1027 / **+0.0541** for 3B / 8B / 70B, and best−raw +0.0328 / +0.0089 / **−0.0131**. The monotone decline stands and is steeper; "essentially nil against raw" at 70B becomes negative. Source: audit/fix/RF16_NOTE.md, audit/fix/rf16_recompute_mean.txt.
