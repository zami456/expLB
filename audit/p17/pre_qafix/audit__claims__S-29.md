# S-29 — "Across the full 80-layer sweep the persona axes LOSE more often than they win: axes beat raw at 28 of 80 layers and dataset PCA at 31 of 80. The ledger's '37/80 (46%)' appears in no printed output."

**Batch:** P13-session · **Checked:** 2026-09-23 · **Source:** session transcript `b035ccb3` line 10333, 2026-09-21T07:42:15Z (DOC-ONLY lead) · **Importance:** **HIGH** — it is an internal contradiction in the ledger about the project's own depth-robustness claim

## The claim, restated
On the 80-layer `passA11` sweep (kd matched slice `harm-pressure-kr-matched` as the training
row, tested on the other ten; mean pooling, paper probe, `axes` k=5), the persona axes beat
**raw** at 28 of 80 layers and **dataset PCA** at 31 of 80. The ledger states 37/80 (46%) in
two places, which the session flags as a bootstrap median re-labelled as an observed count.

## What I did
`audit/scripts/claims/S-29_lab.py`, read-only on the lab PC. It reads every matrix in
`run-crossgen/kd11/passA11/matrices/` (root store: `raw`, `datapca`, `random`, `length`) and
`run-crossgen/kd11/passA11/banks/kd_s23/matrices/` (bank store: `axes`), takes the
`harm-pressure-kr-matched` **row** mean over the other ten datasets exactly as
`kd/scripts/matched_profile.py` does, and counts. Output:
`audit/phase2/out/S-29_profile.txt` (full 80-layer table).

```
root layers: 80  bank layers: 80
bank field in bank matrices: {'axes_vectors_s23.pt'}  in root matrices: {'axes_vectors_s3_fp32.pt'}
datasets per matrix: {11} {11}

axes beats raw at      28 of 80 layers
axes beats datapca(root) at 31 of 80 layers
longest contiguous run with margin over datapca(root) > 0: 16

spot layers:
  L0   axes 0.4695  raw 0.5294 (-0.0599)  datapca 0.5284 (-0.0589)
  L20  axes 0.3743  raw 0.5904 (-0.2161)  datapca 0.6001 (-0.2257)
  L22  axes 0.6237  raw 0.5414 (+0.0823)  datapca 0.5394 (+0.0843)
  L40  axes 0.7393  raw 0.7057 (+0.0336)  datapca 0.6738 (+0.0655)
  L44  axes 0.7321  raw 0.7047 (+0.0273)  datapca 0.7016 (+0.0304)
  L77  axes 0.7642  raw 0.7089 (+0.0553)  datapca 0.6346 (+0.1296)
```

## Verdict — HOLDS, and it exposes a CONFLICT inside the ledger
**28/80 and 31/80 are the observed counts**, re-derived from the raw matrices during this
audit. The 16-layer contiguous run also reproduces, as do F-75's profile landmarks (floor
L19 0.3555, plateau L27–L48 0.61–0.75, peak L77 0.7642).

The ledger says both things:
- `kd/docs/FINDINGS.md:2807` (**F-75**, point (a)): *"`axes` beats raw at **28 of 80** layers
  and `datapca` at **31 of 80**"* — **correct**;
- `kd/docs/FINDINGS.md:2934` (**F-77** comparison table) and `:3233` (**F-81** summary table):
  *"layers with a positive margin | 37/80 (46%)"* for the kd matched slice — **not an observed
  count**. Nothing in the sweep gives 37.

The session's diagnosis is supported by the ledger's own text: **F-76** point (c)
(`FINDINGS.md`, ~line 2882) reports *"Fraction of layers with a positive margin, per resample:
median **0.46**, 95% [0.16, 0.75]"* — a **dataset-bootstrap median fraction**. 0.46 × 80 =
36.8 → "37/80". F-77 and F-81 then set that bootstrap median against the cap bank's genuine
observed **21/21**, i.e. the headline comparison of the project's strongest positive result
puts a resampled median on one side of the table and an observed count on the other.

The correct observed figure for the same comparator (`axes` vs `datapca`) is **31/80 (39%)**,
which makes the contrast with the cap bank *larger*, not smaller — so fixing it does not cost
the owner anything.

**Absent from the collaborator's draft**: no version of 28/80, 31/80 or 37/80 appears there.
Within the owner's own documents the printed counts *are* recorded (F-75), so this is a
**ledger-internal inconsistency**, not a lost measurement.

## One extra observation for the lead
In the bank store (`banks/kd_s23/matrices`) there is also a `datapca` feature, and `axes`
beats **it** at **0 of 80** layers. F-75's caveat (4) already warns that the bank matrices
lack `raw` by construction and that a reader joining the wrong pair gets a meaningless line
("beats raw at 0 of 80 layers" from a stale copy). The `datapca` column inside the bank store
looks like the same trap and should not be used as the matched control; the margins above use
the **root** store's `datapca`, which is what `matched_profile.py` joins.

## Version check
`passA11/matrices/` mtime 2026-09-21 13:13 — **not** rewritten in the 09-22 boot wave (RF-13);
`passA11/summary.json` and `significance.json` were (15:56 / 16:00), but this claim does not
use them and quotes no BH q-value. Banks recorded inside the files: `axes_vectors_s23.pt`
for the `axes` column, `axes_vectors_s3_fp32.pt` for the root `raw`/`datapca` columns.

- **evidence_file**: `C:/Users/user4/expLB/run-crossgen/kd11/passA11/matrices/*.json` and `.../banks/kd_s23/matrices/*.json`
- **evidence_key**: `results[feature,probe=paper].auroc`, row `harm-pressure-kr-matched`, diagonal skipped, mean pooling
- **script**: `audit/scripts/claims/S-29_lab.py` → `audit/phase2/out/S-29_profile.txt`
