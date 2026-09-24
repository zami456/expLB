# Extraction vs. the paper — line by line

**Workstream M2** of `audit/PHASE_M_PLAN.md`. Closes the second of the three gaps recorded at the
foot of `audit/METHODS_INVENTORY.md`: *"The extraction-vs-paper comparison is inferred, from the
ambiguity list and `6_axes.py`. A line-by-line reading of `2_activations_ndif.py` against Eq (1)–(4)
would settle it properly."*

Written 2026-09-23. Read-only: nothing outside `audit/` was written; no git operation beyond
reading the working tree; no model calls; the lab PC was not contacted.

---

## What this compares, and what counts as evidence

**The paper.** Mahadik & Skapars, *Do Linear Probes Generalize Better in Persona Coordinates?*
(arXiv:2605.09391v2). The audit rule is that prose is never proof, and the notes at
`docs/papers/persona-coordinates-notes.md` are this repo's own prose. So the "what the paper says"
column uses the notes' **quotations and transcriptions** (equations, table values, verbatim persona
descriptions) and marks everything else as the notes' **interpretation**. Where the notes contradict
themselves, both readings are shown and neither is picked (one such case: §D-4 below).

**The code.** `src/assistant_axis_ndif/`, read at the working tree of branch `asix` (git status
unchanged from `audit/START_STATE.md`).

**The numbers.** Every number below is either computed during this audit by one of the three
re-runnable scripts named next, or carried from a prior **already-audited** claim file with that
file cited inline (only `audit/claims/F-18.md`, which was checked on the lab copies in an earlier
batch). Numbers quoted from project prose are marked as such and attributed, never asserted.

| Script | What it prints |
|---|---|
| `audit/scripts/m2_extraction_check.py` | role/vector inventory, the Eq (2)→Eq (4) cancellation at L40, anchor-in-PCA effect, what the saved axes file holds |
| `audit/scripts/m2_centering_and_anchor.py` | the cancellation at **all 80 layers** for all three contrasts; anchor-in-PCA across all layers |
| `audit/scripts/m2_unified_qfamily.py` | whether the unified PC1 tracks the question bank rather than harmfulness |

Raw inputs read: `run-paper16/vectors_s3/*.pt` (26 files), `run-paper16/axes_vectors_s3.pt`,
`run-paper16/axes_vectors_all.pt`, `run-paper16/responses/*.jsonl`,
`run-crossgen/matrices/mean_L40_s42.json`, `src/assistant_axis_ndif/pc_role_groups.json`,
`src/assistant_axis_ndif/pc_questions_shared.jsonl`.

---

## Verdict in one paragraph

Of **36 steps** compared: **16 MATCH**, **11 RESOLVE AN AMBIGUITY** the paper leaves open,
**6 are DELIBERATE DEVIATIONS** with the reason recorded in the code or the run script, and
**3 rows are ACCIDENTAL DIVERGENCES** — those three rows carry **4 distinct divergences**, because
the k row (C-3) holds two. The sentence in `METHODS_INVENTORY.md` §1.4 that the extraction
"may differ from the paper" is now settled: **the extraction itself (Eq 1–4) is faithful**. Every
divergence that matters lives in the *bank construction* and the *choice of k*, not in the
activation maths. The load-bearing correction is that §1.4's claim "**we sweep k**" is not supported
by the artifacts — two values were run (5 and 2), the 3B/8B rungs used a third (3), and the paper's
own k=1 (deception) and k=3 (sycophancy) were never run at 70B.

### The four ACCIDENTAL DIVERGENCES

| # | What | Where |
|---|---|---|
| **AD-1** | **k = 5** is the default for `pc:deception` / `pc:unified` / `axes`; the paper's figures use **k = 1** for deception and **k = 3** for sycophancy. No recorded reason for 5. Measured: the result is strongly k-dependent (+0.138 vs matched control at k=2, +0.004 at k=5 — `kd/docs/FINDINGS.md:1030-1040`). | §C-3 |
| **AD-2** | **The unified axis was built without the paper's balanced shared question set.** Deception personas answered 10 deception questions, sycophancy personas answered 12 sycophancy questions, so *which question bank a persona saw* is a property of the persona. Measured in this audit: unified **PC1 correlates −0.945 with the question bank** and only **+0.230 with harmful/harmless**; PC2 is the harmfulness axis (+0.816). It hits the **PCA basis, not the contrast** — so it contaminates `pc:unified*` and the `PC1:unified` zero-shot scorer, and leaves `contrast:unified` almost untouched (`audit/claims/F-18.md`: cross-bank cos(unified axis) = +0.984). Found post hoc and repaired in a separate bank; the default crossgen bank still carries it. | §E-5 |
| **AD-3** | **k is inconsistent across the project's own arms**: 70B ran k=5, the 3B/8B scale rungs ran k=3 (`scale/scripts/run_scale_crossgen.ps1:28`). The scale comparison that carries F-63/F-110 therefore crosses a k boundary that the project's own ledger says the result depends on. | §C-3 |
| **AD-4** | **The judge filter is also inconsistent across those same arms**: the 70B headline bank keeps **score 3 only**, the 3B/8B rungs keep **scores 2 and 3** (`scale/scripts/run_scale_chain.ps1:186`), and the cap bank keeps 2–3 with no score-3 arm at all (`cap/scripts/run_cap_chain.ps1:100`). So the scale comparison differs from the 70B arm in *two* pipeline parameters at once, neither of them the model size it is meant to isolate. Compounded by a filename bug: the scale chain writes both vector variants to the same `axes_$tag.pt` (`run_scale_chain.ps1:197-203` — the loop variable is `$vt`, the output path uses `$Tag`), so only the last one survives. | §E-6 |

### The five questions the task asked to settle

1. **Activation sublocation** — **post-block output of `model.layers[L]`**, i.e. the residual stream
   *after* the MLP has been added: `h = llm.model.layers[L].output`
   (`2_activations_ndif.py:288`, and identically `8_coords_ndif.py:331`). For a Llama decoder layer
   the block output *is* the post-MLP residual stream, so this is the Assistant Axis's explicit
   choice, and it is the same hook Lu et al.'s released code uses
   (`assistant_axis/assistant_axis/internals/activations.py:86-97`, `register_forward_hook` on
   `model.layers[i]`). The persona-coordinates paper does not specify (notes §8.5) → **resolved, in
   favour of the method it inherits from.**
2. **Test-time centering of evaluation activations** — **none is applied.** Zero-shot scores are
   `sc = X32[t] @ d` on raw pooled activations (`27_crossgen.py:440`); PC features are
   `X @ comps.T` (`9_persona_probes.py:139`). This cannot change any AUROC in the project (proof in
   §F-2): AUROC is rank-based within a dataset and every candidate centering is a constant shift of
   that dataset's features.
3. **Is the default anchor in D, in H, or excluded?** — **Excluded from the contrast, included in
   the PCA basis.** `class_contrast` is called with `groups["deception"]["roles"]` vs
   `groups["honesty"]["roles"]` only (`6_axes.py:176-186`); `build_pca` appends `default_like`
   (`6_axes.py:234-246`). Verified in the saved file: `counts["deception"] == (8, 7)` while
   `pca["deception"]["roles"]` has 16 entries including `default`.
4. **Does "standardize" include per-dimension variance scaling?** — **No.** The only operation is
   `Xs - Xs.mean(0, keepdim=True)` (`6_axes.py:229`). Mean-centering only, matching the literal text
   of Eq (2) and of the Assistant Axis step it cites. (`StandardScaler` does appear, but on the
   *probe features*, and only for the `lr_reg10`/`mms` probes borrowed from the Apollo literature —
   the paper's own `paper` probe has no scaler, correctly.)
5. **The value of k** — **5 by default** (`27_crossgen.py:882`), with `pc:unified@2` at 2; five PCs
   are computed (`6_axes.py:134`, saved shape `(80, 5, 8192)`); the 3B/8B rungs used 3. See AD-1/AD-3.

### The centering argument in `6_axes.py:19-21` — checked, not quoted

The docstring claims: *"Eq (2) centers every persona vector by the mean across personas before
Eq (4) takes a difference of class means. That centering cancels exactly in a difference of means,
so it is applied only where it actually matters — PCA."*

**The argument is true of the code as written.** Algebraically, with `m = (1/|P|) Σ_{p∈P} v_p`,
`(1/|D|) Σ_{p∈D}(v_p − m) − (1/|H|) Σ_{p∈H}(v_p − m) = mean_D(v) − mean_H(v) + (−m + m)` — the
constant is subtracted once on each side and cancels, for **any** `P`, including or excluding the
anchor. Empirically, on the real `run-paper16/vectors_s3` vectors in float64, at **every one of the
80 layers** and for all three contrasts (`audit/scripts/m2_centering_and_anchor.py`):

```
deception    max |d_uncentered − d_centered| = 1.55e-15   (relative 1.2e-15)
sycophancy   max |d_uncentered − d_centered| = 2.22e-15   (relative 1.8e-15)
unified      max |d_uncentered − d_centered| = 2.89e-15   (relative 2.2e-15)
```

That is float64 round-off. The skipped step is genuinely a no-op, and — a point the docstring does
not make — it is a no-op *whether or not the anchor is in the Eq (2) mean*, which disposes of half
of the paper's ambiguity §8.8 for free.

---

## The table

Classification key: **M** = matches · **R** = resolves an ambiguity the paper leaves open ·
**D** = deliberate deviation (reason recorded — the record is cited) · **A** = accidental divergence
(the paper specifies, we do otherwise, no reason recorded at the time).

### A. Eq (1) — the persona vector

| # | Step | What the paper says | What our code does | Class | Changes a number? |
|---|---|---|---|---|---|
| A-1 | Who generates the rollouts | "the **target model itself** generates a response" for every (persona × instruction × question) (notes §2.1 Step 3) | Llama-3.3-70B-Instruct generates its own persona rollouts (`1_generate_ndif.py:269`, `run_paper16_chain.ps1`) | **M** | No |
| A-2 | Sampling parameters | Not stated anywhere in the paper (not even in the notes' §8 list) | `temperature=0.7`, `top_p=0.9`, `max_new_tokens=512` (`1_generate_ndif.py:281-283`) | **R** | Unknown. Would need a temperature-0 regeneration of the bank to bound. |
| A-3 | Activation **sublocation** | "hidden states at layer `l`" — "post-MLP residual stream? post-attention? post-block? **Unspecified**" (notes §8.5) | `h = llm.model.layers[L].output` — the decoder block's return value, i.e. the residual stream after the MLP has been added (`2_activations_ndif.py:288`; eval side `8_coords_ndif.py:331`) | **R** | No, if the intended reading is the AA default. Lu et al.'s own code hooks the same module (`assistant_axis/.../activations.py:86-97`). |
| A-4 | `pool(·)` — which tokens | "mean pooling **over output tokens only**"; Figure 1: `x_{p,q} = (1/\|T_out\|) Σ_{t∈T_out} h_t` (notes §2.1 Step 4, §3) | Masked mean over the **last assistant span** only: `num = (h*m).sum(1) / m.sum(1)` with `m` 1.0 exactly on `[a0, a1)` (`2_activations_ndif.py:290-292`, span built at `:185-206`) | **M** | No |
| A-5 | `pool(·)` — mean vs last | Mean. Limitations note last-token/attention pooling "can improve cross-dataset transfer" but was "not systematically compared" (notes §5.6) | Mean is the paper-faithful column; **`last` is stored alongside** (`8_coords_ndif.py:674`) and reported separately | **M** (+ extension) | The paper column is unchanged; the extra column is a new result, not a replacement. |
| A-6 | Averaging to one vector per persona | Eq (1) is a **flat mean over the rollout set `Q_p`** (all instruction × question combos); Figure 1 shows it as two stages (per-question, then over questions) | Flat mean over surviving rollouts: `(total / count)` where `total` sums per-response pooled vectors (`2_activations_ndif.py:389-422`) | **M** | Flat = two-stage only when the grid is complete. With judge filtering (D-3) the grid is **not** complete, so questions whose rollouts survived more often are weighted more. Unquantified; a re-extraction with a two-stage mean would settle it. |
| A-7 | Sequence-length handling | Silent | `--max_length 8192` for this bank, **tail**-truncation so the response span is never cut (`2_activations_ndif.py:194-201`, `run_paper16_chain.ps1:102`) | **R** | No for paper16 (responses ≤512 new tokens). The head-vs-tail choice matters only on the long-context *evaluation* datasets. |
| A-8 | Numeric precision of the reduction | Silent | mask kept fp32, every reduction names `dtype=torch.float32`; the bf16 path rounds token counts above 256 (`2_activations_ndif.py:285-292`) | **R** | Yes, historically — this is a bug fix, not a method choice. In bf16 ~27.6% of responses got the wrong denominator. |

### B. Eq (2) — centering / residualization

| # | Step | What the paper says | What our code does | Class | Changes a number? |
|---|---|---|---|---|---|
| B-1 | Where centering is applied | `ṽ_p = v_p − (1/\|P\|) Σ_{p'∈P} v_{p'}` (Eq 2), then used by **both** Eq (3) and Eq (4) | Applied **only** before PCA (`6_axes.py:229`); Eq (4) uses uncentered vectors (`6_axes.py:69-75`), with the cancellation argument at `6_axes.py:19-21` | **M** | **No — verified exactly.** Max discrepancy 2.9e-15 across all 80 layers, all three contrasts (`audit/scripts/m2_centering_and_anchor.py`). |
| B-2 | Per-dimension variance scaling | "Only mean-centering is described; the paper does not say whether per-dimension variance scaling is also applied" (notes §8.10, inherited from Lu et al.) | Mean-centering only: `Xs - Xs.mean(0, keepdim=True)` (`6_axes.py:229`). No scaler anywhere on the persona vectors. | **R** | Unknown. A variance-scaled PCA on 11–26 points in 8192-d would be a different object; nobody ran it. |
| B-3 | Which persona set `P` the mean is over | `P` is the behaviour's own persona set (16 deception / 11 sycophancy) | The mean is over exactly the roles in that basis: deception 16, sycophancy 11, unified 26 (verified from `axes_vectors_s3.pt`, `pca[*]["roles"]`) | **M** | No |

### C. Eq (3) — PCA and the PC features

| # | Step | What the paper says | What our code does | Class | Changes a number? |
|---|---|---|---|---|---|
| C-1 | PCA input and per-layer fitting | PCA on `{ṽ_p}` "**at each layer**"; `PC_k(z) = ⟨z, u_k⟩` (notes §2.1 Step 7, Eq 3) | `pca_per_layer` runs an SVD per layer over the centered subset; features are `X @ comps.T` with unit-normalised components (`6_axes.py:78-88`, `9_persona_probes.py:135-139`) | **M** | No |
| C-2 | PC **sign** convention | Not mentioned. (SVD signs are arbitrary, and AUROC on `⟨z,u_1⟩` is sign-dependent.) | `orient_pcs` pins each PC per layer against that behaviour's own contrast, falling back to "largest coordinate positive" when near-orthogonal (`6_axes.py:91-114`) | **R** | Yes, and necessarily so: without it a layer sweep of zero-shot AUROC flips to `1 − AUROC` at random layers. Docstring at `6_axes.py:95-99` records exactly this. |
| C-3 | **The value of k** | "§4 says probes are trained on 'the first 3 persona PCs', but the figures use **k = 1** for deception, **k = 3** for sycophancy, **k = 2** for the unified axis. There is no single stated k." (notes §8.2) | 5 components computed (`6_axes.py:134`); default feature k = **5** (`27_crossgen.py:882`); one extra feature at k = 2 (`pc:unified@2`). The 3B/8B rungs ran k = **3** (`scale/scripts/run_scale_crossgen.ps1:28`). **k = 1 was never run.** | **A** (AD-1, AD-3) | **Yes, materially.** The project's own ledger measures the unified axis at +0.138 over its matched control at k=2 and +0.004 at k=5 (`kd/docs/FINDINGS.md:1030-1040`), and notes "the rungs ran k=3; the 70B matrix I read was k=5" (`:1999`). `METHODS_INVENTORY.md` §1.4's "we sweep k" is **not supported**: three values exist across three separate arms; no within-arm sweep, and neither of the paper's figure values for deception/sycophancy. |
| C-4 | Reliability of PC2/PC3 | "PC2/PC3 are estimated from very few samples. The paper does not comment on this." (notes §8.9) | Same exposure (11–26 points in 8192-d), but the bank stores a per-layer `variance_ratio` and `6_axes.py` prints `cos(PC1, contrast)` per basis as a gate | **M** (+ diagnostic) | No |

### D. Eq (4) — the contrast direction

| # | Step | What the paper says | What our code does | Class | Changes a number? |
|---|---|---|---|---|---|
| D-1 | The contrast | `d = mean_D(ṽ) − mean_H(ṽ)`, sample score `s(z) = ⟨z, d⟩` (Eq 4) | `class_contrast` = `stack(pos).mean(0) - stack(neg).mean(0)` (`6_axes.py:69-75`); score `X32[t] @ d` (`27_crossgen.py:440`) | **M** | No (see B-1) |
| D-2 | Normalisation of `d` | Not stated | `P.unit(...)` before scoring (`27_crossgen.py:427`) | **R** | No — AUROC is invariant to positive rescaling. |
| D-3 | Is the anchor in `D` or `H`? | "§3 describes `d` as being between the deceptive and honest groups only; whether the anchor is in `D`, `H`, or neither **is not stated**" (notes §8.8) | **Neither.** The contrast takes `groups["deception"]` vs `groups["honesty"]` only; `counts["deception"] == (8, 7)` in the saved bank | **M** (with the notes' §3 reading) | No |
| D-4 | Is the anchor in the **PCA basis**? | **The notes contradict themselves.** §8.8: role "not stated"; §4.4 recap table: the unified anchor is "1 default (**excluded from PCA**)". Neither is a quotation of the paper's body text. | **Included** in every basis: deception 16, sycophancy 11, unified 26 — all contain `default` (`6_axes.py:234-246`; verified in `axes_vectors_s3.pt`) | **R**, on a **CONFLICT** in the source notes | Measured this audit: PC1 with vs without the anchor, \|cos\| across all 80 layers — deception min 0.9982 / median 0.9994; unified min 0.9995 / median 0.9997; **sycophancy min 0.8765 / median 0.9700**, with the L40 PC1 variance share moving 0.477 → 0.576. So: negligible for deception and unified, **not negligible for the sycophancy basis** (n = 11, so one point carries weight). Nothing in the project's headline set rests on the sycophancy basis, but a sycophancy-PC1 number would move. |

### E. Bank construction (the inputs to Eq 1)

| # | Step | What the paper says | What our code does | Class | Changes a number? |
|---|---|---|---|---|---|
| E-1 | Persona sets and their descriptions | 8 deceptive + 7 honest + 1 anchor = 16; 5 + 5 + 1 = 11; names and one-line descriptions printed verbatim in Tables 2–5 (notes §2.1 Step 1) | Exactly those names and descriptions, quoted verbatim with provenance recorded per persona (`pc_personas.py:66-251`, `_source.description_verbatim`); groups 8/7/5/5/1 (`pc_role_groups.json`) | **M** | No |
| E-2 | Persona **system prompts** | Never printed by the paper; "persona instructions in the same style as the Assistant Axis paper" | Reconstructed in AA style, calibrated against AA's 276 worked examples, and **declared as a reconstruction** in the file header (`pc_personas.py:13-19`) | **R** | Unquantifiable without the paper's prompts. Correctly flagged as the main fidelity limit in `docs/thesis/persona-coordinates-70b-results.md:1127-1131`. |
| E-3 | Instruction variants per persona | "**2 instruction variants**" (notes §2.1 Step 2, §9 self-check) | **5** (`pc_personas.py:65`; verified in the rollouts: `prompt_index` ∈ {0,1,2,3,4} in `run-paper16/responses/*.jsonl`) | **D** — recorded in `kd/docs/FINDINGS.md:1047` head-to-head table ("instruction variants 2 → 5") and in `docs/thesis/...:1129` | Rollouts/persona 20 → 50 (deception) and 24 → 60 (sycophancy). Direction unknown; a 2-variant re-extraction from the same rollouts would settle it cheaply (the rollouts exist). |
| E-4 | Shared question pool — per-behaviour | 10 omission-focused questions (deception), 12 (sycophancy), used for **every** persona in that behaviour | 10 and 12, texts written in the paper's register from its printed topics (`pc_personas.py:262-297`). Verified from the rollouts: deception personas + `default` saw 10 questions, sycophancy personas saw 12. | **M** | No |
| E-5 | Shared question pool — **unified axis** | Appendix C: a "**shared balanced question set of 10 + 10 questions**" — the deception 10 plus the sycophancy axis's "first 10 questions after shared-set balancing" (notes §2.4, §4.4) | The default bank builds the unified axis from the **per-behaviour** rollouts: deception personas on 10 deception questions, sycophancy personas on 12 sycophancy questions, anchor on the deception questions only. No shared set. | **A** (AD-2) | **Yes, decisively — measured in this audit** (`audit/scripts/m2_unified_qfamily.py`, `run-paper16/vectors_s3`): unified PC1 (47% of variance at L40) correlates **−0.945** with which question bank the persona answered and only **+0.230** with harmful-vs-harmless; PC2 is the harmfulness axis (+0.816). Same picture at L20 (−0.929 / +0.264) and L60 (−0.943 / +0.224). So the default bank's unified PC1 is the question family. **Independently corroborated by an already-audited claim**, `audit/claims/F-18.md` (verdict HOLDS, confidence HIGH): `cos(PC1_unified, unified contrast)` = **+0.389** on the default bank vs **+0.896** on the repaired one at L40 — my own script reproduces that +0.389 exactly — and the repaired bank is higher at **80 of 80 layers** (+0.78…+0.91 vs +0.30…+0.49 through the mid network). **Scope:** F-18 also shows the *contrast directions* barely move (cross-bank cos(unified axis) = +0.984), so the damage is confined to the **PCA basis** — it contaminates `pc:unified`, `pc:unified@2` and the `PC1:unified` zero-shot scorer, and leaves `contrast:unified` essentially intact. Found post hoc; two repairs were built and run — an approximate one (`29_bank_variants.py`, bank `paper16_famcentred`) and the faithful one (`run_paper16_shared_chain.ps1`, bank `paper16_shared`, 26 personas × 5 variants × the exact 10 + 10 set, verified identical to `DECEPTION_QUESTIONS + SYCOPHANCY_QUESTIONS[:10]`; 2,600 rollouts, confirmed by line count in F-18). Reported effect on transfer: −0.034 → −0.022 (k=2) and −0.006 → **+0.017** (k=5) (`docs/thesis/persona-coordinates-70b-results.md:1344-1348`). **The default crossgen bank is still the confounded one** (`27_crossgen.py:856` → `run-paper16/axes_vectors_s3_fp32.pt`). **Preservation note:** `run-paper16-shared/` is **not on the laptop and not in git** — lab PC plus Drive only (F-18 "Version"). |
| E-6 | Judge filtering of persona expression | "**No judge, no filtering** — every rollout is pooled in" (notes §4.4, §7) | A 0–3 judge (Llama-3.1-70B-Instruct) scores every rollout and **only score 3 is kept** (`2_activations_ndif.py:449-457`; `run_paper16_chain.ps1:82, 93-104`). Kept counts range 21–60 of 50/60 per persona. | **D** — the reason is written out at `run_paper16_chain.ps1:10-16`: both banks are built from one generation run, `vectors_all` ("unfiltered — **paper-faithful**") and `vectors_s3` ("matches how own100 was built"), so the paper16-vs-own100 comparison does not confound persona set with filtering. | Small on geometry, **measured this audit**: cos(filtered, unfiltered) ≥ 0.977 for every contrast at L20/40/60, and cos(PC1) ≥ 0.930 (worst case deception PC1 at L20). **But:** the paper-faithful `axes_vectors_all.pt` has **never been run through `27_crossgen.py`** — `run-crossgen/banks/` holds `hf100, own100, paper16_famcentred, paper16_shared` and no unfiltered arm. Cheap to close. |
| E-6b | **Which** scores are kept, across arms | — (the paper has no judge, so it specifies nothing) | **Not uniform.** paper16 and paper16-shared: `vectors_all` (0,1,2,3) + `vectors_s3` (**3**), and the bank used downstream is s3. kd: same pair, plus a separate `run_kd_s23.ps1` at (2,3). cap: `s23` (**2,3**) + `all` — **no score-3 arm exists**. scale (3B/8B rungs): `vectors_all` + `vectors_s23` (**2,3**) — **no score-3 arm exists** (`scale/scripts/run_scale_chain.ps1:186`, `cap/scripts/run_cap_chain.ps1:100`, `kd/scripts/run_kd_s23.ps1:71`). | **A** (AD-4) | **Yes, for the scale comparison.** F-63/F-110 read the 3B/8B rungs against the 70B as if model size were the only variable; in fact the rungs differ in the judge filter (2,3 vs 3) *and* in k (3 vs 5, AD-3). Neither is the model size. **Additional code-level finding:** `run_scale_chain.ps1:197-203` loops `$vt` over both variants but writes every iteration to `axes_$Tag.pt`, so the two banks cannot coexist — `vectors_s23` runs second and overwrites `vectors_all`, and `run_scale_crossgen.ps1:53` reads exactly that path. Certain from the code; the surviving artifact is lab-only (`run-scale/` is not on the laptop), so **which** file the lab now holds should be confirmed there with one `ls`/`torch.load` before F-63/F-110 are written up. |
| E-7 | Minimum rollouts to keep a persona | The paper keeps everything (no threshold). Lu et al. use ≥10. | `--min_count 20` (`run_paper16_chain.ps1:102`), against a script default of 50 — lowered *because* score-3 filtering would otherwise silently drop every persona (comment at `:88-91`) | **R** | No persona was dropped: the lowest survivor is `pc_fact_checker` / `pc_plausible_denier` at n = 21. |
| E-8 | Model | Llama 3.2-3B main, "Llama 3-8B" replication; analysis layer 14 / 16 | Llama-3.3-70B-Instruct main; 3B and 8B rungs in `scale/` | **D** — the whole point of the replication (`METHODS_INVENTORY.md` §1.1) | Yes — that is the finding (F-58 → F-63 → F-110). |
| E-9 | Analysis layer | Layer 14 of 28 and 16 of 32 — stated as bare numbers, both exact midpoints (notes §4.2, §8.4 — the "midpoint" reading is the notes' **inference**, not the paper's text) | Middle layer from `assistant_axis/models.py`, 40 of 80 for Llama-3.3-70B, with `n_layers // 2` as fallback (`2_activations_ndif.py:527-537`); all 80 layers extracted and a 21-layer sweep reported. The rungs hardcode the paper's own numbers: `$LAYERS = 28/32`, `$MID = 14/16` (`scale/scripts/run_scale_chain.ps1:52-53`). | **M** (+ extension) | No for the headline layer. The sweep is an addition the paper does not make, and `docs/LAYER_PREREGISTRATION.md` exists because of it. |

### F. The evaluation side and the readout

| # | Step | What the paper says | What our code does | Class | Changes a number? |
|---|---|---|---|---|---|
| F-1 | Evaluation activations | "run the same model on each evaluation dataset's instruction + user prompt, mean-pool layer-14 output-token activations → one vector `z` per example" (notes §6 step 7) | Teacher-forced forward pass over the stored rollout, masked mean over the assistant span, same sublocation (`8_coords_ndif.py:311-359`). Rollouts are filtered to Llama-3.3-70B generations on **every** source: `CSV_MODEL = "llama-v3.3-70b-instruct"`, `ROLLOUT_MODEL = "llama-70b-3.3"` (`eval_datasets.py:46-47`) — i.e. **on-policy**, as the paper requires. | **M** | No. (The off-policy sleeper datasets are deliberately excluded from `all_keys()` — `eval_datasets.py:262-266, 458`.) |
| F-2 | **Test-time centering of `z`** | "The axes are built from centered persona vectors, but Eq (3)/(4) score **raw** evaluation activations `z` (un-centered). Whether/how `z` is centered at test time … is not spelled out." (notes §8.7) | **No centering.** `sc = X32[t] @ d` (`27_crossgen.py:440`), `X @ comps.T` (`9_persona_probes.py:139`) | **R** | **No — and provably not, for any choice.** Any centering constant is either global (the persona mean) or per-dataset. Either way it shifts every score in a dataset by the same amount, `w·(z−m) = w·z − w·m`, so the within-dataset ranking is unchanged and **AUROC is identical**. The only quantity that could move is `TPR@1%FPR`, whose threshold is set on a *different* dataset (`alpaca`) — and even that is unchanged under a global constant. This closes ambiguity §8.7 for every number this project reports. |
| F-3 | Probe estimator and hyperparameters | scikit-learn `LogisticRegression`, **C = 1**, with intercept, **seed 42**, **max_iter 1000**, **no `StandardScaler`** (notes §4.5) | `make_probe("paper")` = `LogisticRegression(C=1.0, fit_intercept=True, max_iter=1000, random_state=seed)`, no scaler (`9_persona_probes.py:203-206`); `--main_seed 42` (`27_crossgen.py:883`) | **M** | No |
| F-4 | Extra probe families | The paper runs only the above | `lr_reg10` (StandardScaler + C=0.1, no intercept — the Apollo/Liars'-Bench config) and `mms` (difference of class means) are run alongside (`9_persona_probes.py:207-222`), all three reported | **D** — rationale in the docstrings at `:207-217`; they are the deception-probe literature's standard, not a substitute for the paper's | The paper column is intact. Reading `lr_reg10`/`mms` as "the replication" would be an error; they are separate columns in the same matrices. |
| F-5 | Train/test protocol | Train on the **source's full balanced train split** (Table 1 sizes), evaluate on the **target's full balanced test split**; 5 datasets per behaviour → 20 off-diagonal pairs (notes §4.3, §5.5) | Train on 80% of the source, split **by `item_id`** and stratified by label, balanced by downsampling; diagonal = the source's own 20% test split; **off-diagonal = 100% of the target** (unbalanced, because AUROC is ratio-invariant). 10 datasets → 90 off-diagonal pairs. (`27_crossgen.py:1-21`, `:104-112`, `:886`) | **D** — the design is stated in the module docstring and in `DESIGN` (`27_crossgen.py:113-115`) | Off-diagonal AUROC over 100% vs a 20–25% subset is a variance effect, not a bias one. The `item_id` grouping is a **leakage control the paper does not have** and can only lower a diagonal number. |
| F-6 | Evaluation corpus | 5 deception + 5 sycophancy datasets (Table 1), with the paper's own Table-1-vs-figures inconsistency over MASK/HPC (notes §8.1) | 10 labelled datasets + `alpaca` as an honest-only FPR control, spanning three different label constructs, annotated per dataset (`27_crossgen.py:47-51, 86-89`) | **D** — `METHODS_INVENTORY.md` §1.1 ("Breadth — 11 datasets against their 5–6") | Yes by construction — it is a different, larger benchmark. The 5-dataset paper set is reproduced separately in the scale rungs (`scale/scripts/run_scale_crossgen.ps1:45-47`). |
| F-7 | Controls | Random k-dim subspace and source-dataset PCA (notes §4.6, and `kd/docs/FINDINGS.md:1015-1026` corrects the earlier claim that these are ours — **they are the paper's**) | Both, **at matched k** (`27_crossgen.py:104-112`, `comparisons_for`), plus a length-only control and a dataset-level bootstrap the paper does not run | **M** (+ extension) | The matched-k requirement is what kills several headlines; see `METHODS_INVENTORY.md` §2.2. |

---

## What follows from this, for the thesis and for the ledger

1. **`METHODS_INVENTORY.md` §1.4 should be corrected on k.** "k is not fixed … → we sweep k" is not
   what the artifacts show. Replace with: *k = 5 at 70B, k = 3 at the 3B/8B rungs, k = 2 for one
   unified feature; the paper's k = 1 (deception) and k = 3 (sycophancy) were never run at 70B, and
   the ledger's own measurement (`kd/docs/FINDINGS.md:1030-1040`) shows the headline is
   k-dependent.* This is the single highest-value correction in this document.
2. **The 3B/8B-vs-70B scale comparison is not a clean one-variable comparison** (AD-3 + AD-4). It
   differs in k (3 vs 5) and in the judge filter (scores 2–3 vs score 3 only) as well as in model
   size, and the project's own ledger already establishes that the result is k-dependent. F-63 and
   F-110 should either be re-run with the 70B settings on the rungs, or stated with both parameter
   differences named. Before either, confirm on the lab PC which bank `run-scale/<rung>/axes_<rung>.pt`
   actually holds — the chain script overwrites one variant with the other.
3. **§1.4's bullet list is otherwise accurate**, and now checked rather than inferred: activation
   sublocation, test-time centering, the anchor, and variance scaling are all resolved as §1.4
   describes, and the centering proof it cites holds exactly.
4. **AD-2 is already repaired but not promoted.** `paper16_shared` is the faithful bank and it is
   the one that should carry any "the paper's method at 70B" claim; `27_crossgen.py`'s default bank
   is still the confounded `run-paper16/axes_vectors_s3_fp32.pt`. Any **unified-basis** figure
   sourced from `run-crossgen/matrices/*` without a `banks/paper16_shared` path is the confounded
   axis — but, per `audit/claims/F-18.md`, deception and sycophancy numbers and all *contrast*
   directions are unaffected, so this does **not** invalidate the bulk of the matrices. The
   repaired bank's vectors exist only on the lab PC and Drive, never in git.
5. **Two cheap runs would remove the remaining fidelity objections**, both from rollouts that already
   exist on disk (no NDIF, no generation):
   - re-extract `paper16` at **2 instruction variants** (E-3) and at the paper's **k = 1 / k = 3**
     (C-3) — closes the two fidelity gaps `docs/thesis/persona-coordinates-70b-results.md:1127-1131`
     itself lists as open;
   - run `run-paper16/axes_vectors_all.pt` (the unfiltered, paper-faithful bank) through
     `27_crossgen.py` (E-6) — it was built for exactly this and never used.
6. **One thing to fix in the paper notes, not the code.** `docs/papers/persona-coordinates-notes.md`
   §4.4 asserts the unified anchor is "excluded from PCA" while §8.8 says the anchor's role is not
   stated. Those cannot both be right about the same paper. Until the PDF is re-read on this point,
   D-4 should be cited as a resolution-under-conflict, not as a match.

## Audit trail

| Artifact | Path |
|---|---|
| This document | `audit/methods/EXTRACTION_VS_PAPER.md` |
| Inventory + L40 cancellation + anchor + saved-bank contents | `audit/scripts/m2_extraction_check.py` |
| Cancellation at all 80 layers; anchor effect at all layers | `audit/scripts/m2_centering_and_anchor.py` |
| Unified PC1 vs question bank | `audit/scripts/m2_unified_qfamily.py` |
| Prior audited claim corroborating AD-2 (independent reimplementation, lab copies, all 80 layers) | `audit/claims/F-18.md` → `audit/scripts/claims/P6_lab_b.py` |

**Cross-check worth recording:** `m2_unified_qfamily.py` computes `cos(PC1_unified, unified contrast)`
= **+0.389** at L40 from the laptop's `run-paper16/vectors_s3/*.pt` by rebuilding the PCA from
scratch; `audit/claims/F-18.md` reports **+0.389** from the lab PC's prebuilt
`run-paper16/axes_vectors_s3.pt`. Two independent paths, same value — so the saved bank and a fresh
rebuild of Eq (2)+Eq (3) agree, which is itself a check on `6_axes.py`.

Nothing outside `audit/` was created or modified. No lab-PC access, so `audit/LABPC_WRITES.md` is
unchanged.
