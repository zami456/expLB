# F-39, checked and made robust (owner-authorised item 5, 2026-09-22)

**What F-39 says:** read against the paper itself, our pipeline is bigger and stricter but gets a *smaller* effect.

## 1. The numbers reproduce exactly (VERIFIED)
Running your own `kd/scripts/paper_comparison.py` on `run-crossgen/kd11/matrices/mean_L40_s42.json` (identical in git and on the lab PC) with the paper's five datasets gives every number F-39 quotes:

| quantity | value |
|---|---|
| raw off-diagonal mean, 20 pairs | 0.8090 |
| `pc:unified@2` − raw | +0.0262 |
| `datapca@2` − raw | −0.1120 |
| `pc:unified` (k=5) − raw | −0.0173 |
| `datapca` (k=5) − raw | −0.0214 |

## 2. The dataset set is the paper's — CORRECTED 2026-09-23 (this section was wrong)

> **CORRECTION (2026-09-23, wave-5 lead QA).** The paragraph below originally read: *"its deception
> datasets are ConvGame, InstrDec, Mask, AILiar, Roleplay (table on p.4 …). Three other entries
> (F-45, F-48, F-54) name harm-pressure-choice as the paper's fifth. **Those are wrong** and should
> be corrected."* **Both halves of that were wrong**, and the owner's own F-60 had already got it
> right. Batch W5C challenged it; the lead re-extracted the PDF and W5C is correct.

**What the PDF actually says** (independent extraction, `pypdfium2` over all 14 pages):

| term | total mentions | pages |
|---|---|---|
| `HarmPressureChoice` | **2** | **4 only** |
| `Mask` | 41 | 5, 6, 7, 12, 13, 14 — **never page 4** |
| ConvGame / InstrDec / AILiar / Roleplay | 43 / 43 / 42 / 42 | 4 and 5–7, 12–14 |

Table 1 on p.4 ("Evaluation datasets with train/test split sizes") lists the deception rows as
**AILiar, Roleplaying, ConvGame, InstrDec, HarmPressureChoice**. Mask has no row in it. Every
figure on pp.5–7 and every appendix table instead uses **ConvGame, InstrDec, Mask, AILiar,
Roleplay**, and HarmPressureChoice appears in none of them.

**So the paper is internally inconsistent**: the set it *declares* and the set it *evaluates* differ
by one dataset, HarmPressureChoice ↔ Mask.

**What follows:**
- F-39's numbers reproduce with **mask**, which matches the set the paper's *results* use. F-39 and
  F-60 are right — but for a reason this note originally stated wrongly (it credited p.4's table for
  a set that is not in p.4's table).
- **F-45, F-48 and F-54 are NOT wrong as written.** They say the five "ARE Mahadik & Skapars'
  **Table 1** deception benchmark — AILiar, Roleplaying, ConvGame, InstrDec, HarmPressureChoice"
  (`FINDINGS.md:1418`, `:1659`). That is *exactly true of Table 1*. Recommendation 3 below is
  withdrawn.
- **The real defect is different and smaller:** F-60 already recorded the whole inconsistency —
  including a mention-count table matching the one above — and said the Table-1 framing is "true of
  Table 1, and misleading" over-reach. **That correction was never back-annotated to F-48 or F-54**,
  which still assert the Table-1 framing with no supersede marker. Batch W5B found the same gap
  independently. → the owed fix is a pointer to F-60, not a change of dataset list.
- The owner had also flagged the inconsistency a third time, at `FINDINGS.md:417-419` and in
  `docs/papers/persona-coordinates-notes.md` §8.1. **She was ahead of this note on every point.**
- One small discrepancy, not material: F-60's table counts 33 `Mask` mentions; this extraction counts
  41. F-60's structural conclusion is unaffected.

## 3. But "smaller effect" rests on a comparison that moves (the real caveat)
`audit/fix/f39_sensitivity_L40.txt` (script `audit/scripts/fix_f39_sensitivity.py`), layer 40:

| dataset set | pairs | raw | axis@2 − raw | dPCA@2 − raw | axis@2 − dPCA@2 |
|---|---|---|---|---|---|
| the paper's five (mask) | 20 | 0.8090 | **+0.0262** | −0.1120 | **+0.1382** |
| swap mask → harm-pressure-choice | 20 | 0.7513 | +0.1275 | −0.0181 | +0.1456 |
| minus instructed-deception | 12 | 0.7460 | +0.0542 | −0.1375 | +0.1917 |
| minus mask | 12 | 0.8723 | +0.0522 | −0.0706 | +0.1228 |
| minus convincing-game | 12 | 0.7457 | +0.0452 | −0.1115 | +0.1567 |
| minus ai_liar | 12 | 0.8727 | −0.0487 | −0.1831 | +0.1344 |
| minus roleplaying | 12 | 0.8082 | +0.0280 | −0.0574 | +0.0854 |

Other layers (`audit/fix/f39_sensitivity_layers.txt`), the paper's five: axis@2 − raw = +0.057 (L20), +0.020 (L32), +0.011 (L44), +0.018 (L40, `last` pooling); axis@2 − dPCA@2 = +0.120, +0.096, +0.150, +0.110.

**What this shows.** The gain *over raw* swings from −0.049 to +0.128 depending on which five datasets are in the set, because the raw baseline itself swings from 0.746 to 0.873. The margin *over the matched control* (dataset PCA at the same k) stays between +0.085 and +0.19 in every variant and at every layer.

## 4. What I recommend you say (and tell your collaborator)
1. **Keep the finding, change its load-bearing comparison.** State the persona direction's margin over the **matched control** (dataset PCA at the same k), not over raw. On the paper's own five at layer 40 that is **+0.138**, against the paper's +0.13…+0.23 — i.e. our result sits at the low end of the paper's range, which is a fair "smaller but same sign" claim, and it is stable.
2. **Report the raw comparison with its caveat.** Ours is +0.026 against the paper's +0.10…+0.17, but say plainly that this difference is mostly about how strong the raw baseline is on the chosen datasets. Give the range (−0.05 to +0.13 across single-dataset swaps).
3. ~~**Fix F-45, F-48 and F-54** where they call harm-pressure-choice one of the paper's five. They are inconsistent with the paper and with F-39/F-60.~~
   **WITHDRAWN 2026-09-23 — see the correction in §2.** Those entries are true of the paper's
   Table 1. What is owed instead: back-annotate **F-48 (`FINDINGS.md:1418`) and F-54 (`:1659`)**
   with a pointer to **F-60**, so the reader learns that the set the paper *evaluates* substitutes
   Mask for HarmPressureChoice. And when writing the thesis, say which of the two sets is meant —
   the paper declares one and evaluates the other.
4. **Do not quote the k=5 numbers as the headline.** At k=5 the axis is below raw (−0.017), at k=2 it is above (+0.026). The paper's figure is k=1, so the k=2 comparison is the closer match; state k explicitly wherever the number appears.

Evidence: `audit/fix/f39_sensitivity_L40.txt`, `f39_sensitivity_layers.txt`, `audit/scripts/fix_f39_sensitivity.py`, `audit/phase1/fmap_all.csv` row F-39. The matrix is committed in git, so anyone can re-run this.
