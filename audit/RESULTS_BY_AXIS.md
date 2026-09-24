# Every persona axis, what it was built from, and where it was tested

_2026-09-24. Every number read from a matrix file during this review; reproduce with
`audit/scripts/p13_all_results_by_axis.py` and `p12_kd_scale_ladder.py` (run on the lab PC)._

This answers four questions the project's own documents never state in one place: which personas
each axis is built from, whether the 36-persona bank merges the behaviours, what each axis scores at
each model size, and **which axes were never tested at all**.

---

## 1. The persona sets — what the paper does, and what we do

Verbatim from `src/assistant_axis_ndif/pc_role_groups.json`:

> *"8 deceptive + 7 honest = the paper's **16-persona deception set** once the shared `default`
> anchor is counted; 5 + 5 + default = its **11-persona sycophancy set**."*

**So the name `paper16` refers to the paper's deception set, not the bank size.** The bank holds
**26** personas because the two sets share the anchor: 15 + 10 + 1 = 26.

| axis | contrast | PCA fitted over | personas |
|---|---|---|---:|
| **deception** | 8 deceptive vs 7 honest | deception + honesty + anchor | **16** |
| **sycophancy** | 5 syc vs 5 anti-syc | sycophancy + anti-sycophancy + anchor | **11** |
| **unified** | (dec+syc) vs (hon+anti) | all four groups + anchor | **26** |
| **knowledge_denial** *(ours)* | 4 feigned vs 3 sincere-ignorance | those + anchor | **8** |
| **kd_assertion** *(ours)* | 4 feigned vs 3 plain-assertion | those + anchor | **8** |

**Source:** `6_axes.py:214-244`, whose own comment states the reasoning —

> *"Mahadik & Skapars run PCA over the behaviour-specific persona set ONLY, so their PC1 is a
> behaviour axis by construction. PCA over every role instead recovers the Assistant Axis — a
> different object (measured cos(PC1, assistant) = 0.77–0.82). Build both, labelled for what they
> are."*

### Does the 36-persona bank merge deception and sycophancy? **No.**

`kd/personas/kd_role_groups.json` declares **five separate contrasts**, each with its own basis:

```
deception          pos=[deception]        neg=[honesty]            basis = pos+neg+anchor  (16)
sycophancy         pos=[sycophancy]       neg=[anti_sycophancy]    basis = pos+neg+anchor  (11)
unified            pos=[deception,syc]    neg=[honesty,anti_syc]   basis = pos+neg+anchor  (26)
knowledge_denial   pos=[knowledge_denial] neg=[sincere_ignorance]  basis = pos+neg+anchor  (8)
kd_assertion       pos=[knowledge_denial] neg=[plain_assertion]    basis = pos+neg+anchor  (8)
```

**The kd bank is the paper's 26 personas plus 10 more, with the original four groups untouched.**
Its deception axis is built from exactly the same 16 personas as `paper16`'s. The 10 new roles form
**two additional small contrasts**, not one big merged one. Nothing is lumped together.

---

## 2. The scale ladder — 3B, 8B, 70B

Off-diagonal mean AUROC, layer-averaged, **5 datasets** (instructed-deception, mask, convincing-game,
ai_liar, roleplaying), k=3, mean pooling, `paper` probe, seed 42.
**These five contain no harm-pressure data.**

| feature | 3B paper16 | 3B kd | 8B paper16 | 8B kd | 70B paper16 | 70B kd |
|---|---:|---:|---:|---:|---:|---:|
| layers averaged | 7 | 7 | 8 | 8 | 21 | 21 |
| **raw** (8192-d probe) | 0.6989 | 0.6989 | 0.7149 | 0.7149 | 0.7546 | 0.7546 |
| `datapca` (matched control) | 0.6120 | 0.6120 | 0.6211 | 0.6211 | 0.6825 | 0.6825 |
| `random` | 0.4863 | 0.4863 | 0.5697 | 0.5697 | 0.4951 | 0.4951 |
| `length` | 0.4948 | 0.4948 | 0.4948 | 0.4948 | 0.4948 | 0.4948 |
| `axes` (contrast, no PCA) | 0.6926 | 0.6873 | 0.6959 | 0.6595 | 0.7291 | 0.7198 |
| **`pc:deception`** (16) | **0.7314** | 0.7136 | 0.7217 | 0.7241 | 0.7225 | 0.7363 |
| `pc:unified` (26) | 0.6855 | 0.6957 | 0.7129 | 0.7092 | 0.7328 | 0.7355 |
| `pc:unified@2` (26, k=2) | 0.7235 | 0.7256 | 0.7122 | 0.7116 | 0.7261 | 0.7271 |

### Margin over `raw` — the only comparison that matters

| feature | 3B paper16 | 3B kd | 8B paper16 | 8B kd | 70B paper16 | 70B kd |
|---|---:|---:|---:|---:|---:|---:|
| `axes` | −0.0063 | −0.0117 | −0.0190 | −0.0553 | −0.0254 | −0.0348 |
| **`pc:deception`** | **+0.0324** | +0.0146 | +0.0069 | +0.0093 | **−0.0321** | −0.0182 |
| `pc:unified` | −0.0135 | −0.0033 | −0.0020 | −0.0056 | −0.0218 | −0.0191 |
| `pc:unified@2` | +0.0246 | +0.0267 | −0.0027 | −0.0032 | −0.0285 | −0.0274 |

**Reading.** The paper's own deception axis is the only one clearly positive at 3B (+0.032). By 70B
it is **negative** (−0.032). The unified 26-persona axis never beats raw by much at any size. The
bare contrast without PCA (`axes`) is negative everywhere.

*(F-121 reports the **best** persona feature per cell, which is why its headline figures
— +0.0382 / +0.0148 / −0.0116 — differ from any single row here.)*

---

## 3. The knowledge-denial axes — **70B only**

Trained crossgen, `run-crossgen/kd11/banks/kd`, **11 datasets**, **k=5**, mean pooling, `paper`.
`raw` and the controls come from the ROOT store, because the bank store holds no `raw` block.

| layer | `pc:deception` (16) | `pc:unified` (26) | **`pc:knowledge_denial`** (8) | **`pc:kd_assertion`** (8) | raw | datapca |
|---|---:|---:|---:|---:|---:|---:|
| 20 | 0.5507 | 0.5840 | 0.6070 | **0.6389** | 0.6289 | 0.6102 |
| 40 | 0.6313 | 0.7380 | 0.7492 | **0.8119** | 0.6623 | 0.6651 |
| 44 | 0.6381 | 0.7437 | 0.7337 | **0.7926** | 0.6565 | 0.6569 |

**`pc:kd_assertion` is the best-transferring persona feature in the project** — +0.150 over raw at
L40 across 11 datasets. But note what its contrast is: **feigned ignorance vs plain assertion**. It
is an *assertion-register* axis, and it outperforms the knowledge-denial axis proper
(0.8119 vs 0.7492). **That is the register diagnosis (F-13) visible directly in the feature table.**

---

## 4. What was tested where — and a correction

**CORRECTED 2026-09-24.** An earlier version of this file said `pc:sycophancy` was *"never computed,
anywhere"*. **That was wrong.** The search covered only `run-crossgen/` on the lab PC — **our**
pipeline. **The collaborator's branch has four sycophancy arms**, at all 16 layers. The claim was
scoped to one pipeline and stated as if it covered the project.

> **UPDATE 2026-09-24 (DF-109, P17).** "Never trained on our side" is stale: 6D-1 trained
> `pc:sycophancy` in our pipeline (`run-coords/passB`, paper16 bank, 11 datasets). It **loses to raw
> at all six (layer, pooling) cells**, by −0.008 to −0.118, and turns positive only without
> `sycophancy_mt`, under mean pooling at L40/L44 (+0.0017 / +0.0342 over raw). Our-pipeline cell is
> now ✅ 70B only.
> **Source:** `kd/docs/FINDINGS.md` F-126 §1 (LAB-ONLY store); `audit/p17/C_REPORT.md` F-126 row (HOLDS 41/41).

| axis | our pipeline (`run-crossgen/`) | his pipeline (`origin/paper`) |
|---|---|---|
| deception (16) | ✅ 3B, 8B, 70B | ✅ `axis1`, `pc1`, `pc3`, `dec3_orth` |
| sycophancy (11) | ❌ **never** — absent from the default `--features` | ✅ **`syc_pc1`, `syc3`, `syc_axis1`, `syc3_orth`** |
| unified (26) | ✅ 3B, 8B, 70B | ✅ `uni_axis1`, `uni_pc3` |
| knowledge_denial (8) | ⚠️ 70B only | ✅ `denial_axis1`, `denial_pc3` |
| extended (12v13) | — | ✅ `ext_axis1`, `ext_pc1`, `ext_pc3` |

**So the sycophancy axis IS tested — on his side, not ours.** The gap in our pipeline is real
(`27_crossgen.py`'s default feature list omits it), but the project as a whole has the result.

### His `syc_pc1` tops the excl-HP-KR table at L40-56 — but it is one dataset, and it is zero-shot

Off-diagonal **excluding HP-KR**, best arm per layer:

> **UPDATE 2026-09-24 (P17).** The baseline column below was read @ ea1ce52b. At the pin 16def2de
> the collaborator's re-run baseline reads **0.7580** (L40) and **0.7296** (L44), −0.0001 each; the
> "vs baseline" column moves in the 4th decimal only.
> **Source:** `audit/p17/C_REPORT.md` F-122 row; `audit/p17/csv_drift.csv`.

| layer | his baseline | best arm | value | vs baseline | `ext_axis1` (draft headline) |
|---|---:|---|---:|---:|---:|
| 36 | 0.8124 | `ext_axis1` | 0.9403 | +0.128 | 0.9403 |
| **40** | 0.7581 | **`syc_pc1`** | **0.9370** | +0.179 | 0.9165 |
| **44** | 0.7297 | **`syc_pc1`** | **0.9274** | +0.198 | 0.9053 |
| **48** | 0.7375 | **`syc_pc1`** | **0.9228** | +0.185 | 0.8952 |
| **52** | 0.7515 | **`syc_pc1`** | **0.8999** | +0.148 | 0.8806 |
| **56** | 0.7419 | **`syc_pc1`** | **0.8685** | +0.127 | 0.8670 |

Win counts over the 16 layers: `syc_pc1`=5, `ext_pc3`=4, `ext_axis1`=4, `axis1`=1, `uni_axis1`=1,
`syc3`=1. **No arm dominates** — F-72's and F-120's warning again: do not report a best arm without
its layer.

**Two things stop this being "sycophancy beats the extension" (F-122).**

1. **The margin is one dataset.** Per test set at L40: `convince_me` **+0.086**,
   `harm_refusal_choice` +0.041, `insider_trading` +0.005, `instruct_dishonesty` **−0.028**,
   `repe_honesty` −0.001. `syc_pc1` wins **3 of 5** (2 of 5 at L56), and `convince_me` supplies most
   of the margin at every layer. The honest sentence is **"a sycophancy axis reads the convincing
   game better than the knowledge-denial extension does."** `convince_me` is persuasion under
   pressure — exactly where an agreement-versus-pushback direction should fire — so it is
   mechanistically coherent, supports the register diagnosis directionally, and is n=1.
2. **It is a zero-shot number.** For every `dim=1` arm the trained off-diagonal **equals** the
   zero-shot AUROC: a logistic regression on one scalar can only pick a sign. Measured spread of any
   off-diagonal cell across training rows is **0.000 at all 16 layers** for `ext_axis1` and
   `denial_axis1`, and 0.000 outside L56-L72 for the rest. So the 30-cell off-diagonal mean holds
   **6 distinct numbers**, replicated — an error bar over those cells counts duplicates as
   independent. Meanwhile the baseline is a **trained 8192-d probe**. Label the axis arms zero-shot
   and put the error bar over 5 test datasets.

**The standing caveat:** D-158 showed the fair comparator is a **pooled** probe, not a
single-dataset one. These +0.13 to +0.24 figures are against the weak baseline.

## 5. Direct answers to the four questions

1. **Did the paper test deception on only 16 personas?** Yes — 8 deceptive vs 7 honest, plus the
   shared anchor. That is where the name `paper16` comes from.
2. **And sycophancy on 10?** **Eleven**, not ten — 5 + 5 plus the same anchor.
3. **Did we lump all 26 together for both?** **No.** `pc:deception` uses the 16; `pc:sycophancy`
   would use the 11; the 26-persona basis exists separately as `pc:unified`, which is the paper's
   own Appendix-C axis. *(Caveat: `pc:unified`'s PC1 correlates **−0.98 with which question set** a
   persona answered rather than with deception — the 26-persona basis is confounded, the 16-persona
   one is not.)*
4. **Did we use all 36 together, no separation?** **No.** Five contrasts, five bases, the original
   four groups unchanged. **How kd was tested:** as two extra contrasts
   (`knowledge_denial`, `kd_assertion`) over 8 personas each, in the 70B 11-dataset store at k=5 —
   and, since 2026-09-24, its *deception* and *unified* axes also have 3B and 8B rungs (F-121).

---

## 6. Caveats that travel with every number above

- **One seed (42), mean pooling, `paper` probe.** No significance testing is applied in this file;
  these are point estimates.
- **The scale table's 5 datasets exclude all harm-pressure data** — the very thing the kd bank was
  built for — so it measures the kd bank's *general* effect, not its intended one.
- **The 70B kd table uses 11 datasets at k=5**, the scale table 5 datasets at k=3. **They are not
  directly comparable**; do not read across sections 2 and 3.
- **Layer counts differ by rung** (7 / 8 / 21), so "layer-averaged" means different things per rung.
- **`length` is layer-independent** by construction (0.4948 everywhere) — it is a text feature.
