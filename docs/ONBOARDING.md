# Onboarding — persona-coordinates replication at 70B

**Read this before touching anything.** It is written for a Claude session (or a person) that
has no history with this project. It says what the research is, what state it is in, how the
owner wants the work done, and which mistakes have already been made and paid for.

---

## 1. What this research is

Replicating **Mahadik & Skapars, *Do Linear Probes Generalize Better in Persona Coordinates?*
(arXiv:2605.09391)** on **Llama-3.3-70B-Instruct**, at far larger scale than the paper, with
**no local GPU** — every forward pass runs remotely on **NDIF** via **nnsight**.

The paper builds low-dimensional "persona axes" from contrastive persona prompts, then uses
them two ways: as zero-shot lie detectors, and as k-dimensional features for a probe claimed
to transfer across datasets better than a probe on raw activations (+0.10 mean off-diagonal
AUROC on Llama-3.2-3B at layer 14).

We test that at 70B on **15,278 rows across 10 datasets** — 5.5× the paper's corpus — drawn
from Liars' Bench (~85%), Apollo's deception-detection rollouts, and MASK.

**The owner is doing this as thesis work.** Correctness outranks speed, always.

---

## 2. Read these first, in this order

| # | File | Why |
|---|---|---|
| 1 | `~/.claude/plans/harmonic-cuddling-kay.md` | The full plan: pipeline, the 11 research questions (Q1–Q11) with every outcome pre-committed, infrastructure lessons, TODO. **The single most important file.** |
| 2 | `~/.claude/projects/C--Users-SURAIYA-.../memory/MEMORY.md` | Index of measured findings. Each entry is a fact that cost real time to learn. |
| 3 | `CLAUDE.md` (repo root) | Repo layout, submodules, and the full nnsight agent guide. |
| 4 | `docs/papers/persona-coordinates-notes.md` | The target paper's numbers, with a flagged ambiguities section. |

**Load these skills** before writing code:

- `nnsight-ndif-context` — this deployment's real failure modes. Not optional.
- `nnsight:probing`, `nnsight:interp-experiment-design` — probe methodology and controls.
- `ndif:ndif-troubleshoot` — when NDIF misbehaves.
- `persona-coordinates-context`, `liars-bench-context`, `deception-probes-context`, `mask-context`.

---

## 3. State as of 2026-09-16

**Answered:**

- **Q1** — the persona set is most of the contamination. paper16 (the paper's own personas)
  halves `cos(deception, assistant)`: −0.66 → −0.37. And it beats own100 on *detection*, not
  just geometry.
- **Q7** — judges barely matter. Llama-3.1-70B vs 405B on identical rollouts agree 91.2%; the
  score-3 rate that builds the vectors differs by 0.8 pp. This **overturned** the prediction.
- **Q4** — deception signal survives erasing the Assistant Axis from the activations (costs
  0.006–0.023 AUROC against 0.000 for a rank-matched random erasure).
- **Q2 — layer-dependent, and that is the finding.** Persona coordinates *lose* to the raw
  8192-d probe at layers 20 and 22 and *win* at layer 40, the paper's own midpoint layer.
  Zero-shot replicates and improves on the paper (PC1 0.789 vs their 0.734).
- **Q10** — the sycophancy sign REVERSES under paper16. An earlier claim that "sycophancy is
  clean under every construction" is false.

**Running / pending:**

- **Pass A** (80-layer sweep, ~2,080 stratified rows) — running as scheduled task `PassA-Sweep`.
  It turns "the answer flips between 22 and 40" into a curve.
- **Pass C** (per-token, layers 20/22/40) — not written. `8_coords_ndif.py` pools in-trace.
- **Sycophancy half** — 2,400 prompts prepped (`10_sycophancy_ndif.py --stage prep`),
  generation not yet run. The axis exists; nothing to test it against until this lands.
- **own100 rebuild** — deliberately abandoned; see §6.
- **`ai_audit`** — cannot be extracted. Its contexts are 4,211–7,543 tokens and eager
  attention needs 4.2–13.5 GiB for a single row. Needs NDIF to enable SDPA.

---

## 4. How the owner wants the work done

Learned from working with them, not assumed:

- **Measure before deciding.** "double check before major decision." Do not reason about an
  impact you could test in ten minutes. Two full re-extractions were decided this way, and one
  was avoided the same way.
- **Report honestly, including your own errors.** They said: *"we all make mistakes, but we
  gotta fix it."* Corrections are welcome; quiet corrections are not. If a number you gave
  earlier turns out wrong, say so plainly and give the new one.
- **Starting over is cheap; a wrong method is not.** Their words: *"if we have used wrong
  pipeline, we also start over that is not an issue, wrong method is."*
- **Save every insight the moment it is measured**, to the memory directory, and mark what it
  supersedes. See the `save-insights-when-measured` memory entry.
- **Keep the plan updated** as results land.
- **Monitor continuously and fix without being asked** when a run is live. They asked for
  "continuous monitoring and fixing, and keeping me updated."
- **Do not spawn subagents** unless asked.
- **Never paste credentials or config output into chat.** An OAuth token leaked that way once
  and had to be revoked. Read secrets from files; ship them machine-to-machine.

---

## 5. Infrastructure

| | |
|---|---|
| **Lab PC** | `user4@100.85.217.68` over Tailscale. Ryzen 9 9950X, 63 GB RAM, **RTX 5090 (32 GB, unused — the pipeline is a network client)**, ~130 GB free. |
| **NDIF** | `https://api.ndif.us`, nnsight 0.7.0. Subject `meta-llama/Llama-3.3-70B-Instruct` (unpinned — goes COLD), judge `meta-llama/Llama-3.1-70B-Instruct` (pinned). |
| **Storage** | Google Drive via rclone remote `tamara:` → `expLB/`. The owner wants results in **Drive, not on the laptop**. Token is Testing-mode and expires every 7 days: `rclone config reconnect tamara:`. |
| **Venvs** | Lab PC `.venv-aa` (NDIF work — **socket pins are load-bearing, do not upgrade**), `.venv-analysis` (pandas/sklearn/torch-cpu). Laptop `.venv-aa-win` is the only Windows-usable one. |

**Long-running work must be a Scheduled Task**, never `Start-Process` over SSH — Windows
OpenSSH takes the process tree down with the session. Use the S4U pattern in
`register_coords_task.ps1`. This was learned three times.

The laptop has 7.7 GB RAM and sleeps; the lab PC does the work and survives that.

---

## 6. Mistakes already made — do not repeat them

- **A fix applied to one script and not its siblings.** Twice: token-budget batching (fixed in
  extraction, missed in the judge → zero scores overnight) and stall handling (fixed in
  `8_coords_ndif`, missed everywhere else → a 40-minute silent hang). Shared helpers now live
  in `ndif_common.py`. **If you fix an NDIF failure mode, grep the other scripts.**
- **Reporting partial-corpus probe numbers.** On 5 of 10 datasets, persona coordinates read
  0.815 against raw's 0.768; on all 10 it was 0.583 against 0.660. The direction reversed, and
  so did the within/across-semantics effect. **Do not quote probe numbers before the corpus is
  complete, even with caveats.**
- **Assuming instead of measuring.** The bf16 count bug looked serious and turned out to move
  cosines by ≤0.001 — because a wrong denominator rescales without rotating, and everything
  downstream is scale-invariant. The bf16 *argmax* bug, which looked similar, was genuinely
  destructive. Ask whether a bug perturbs **magnitude or direction**.
- **`--limit` is not a sample.** It head-slices, and `instructed-deception` is label-sorted, so
  `--limit 200` returns 0 deceptive / 200 honest. Use `--sample`.
- **`--layers every4` skips layer 22**, which is Apollo's layer and one of the three the whole
  layer question turns on.

---

## 7. Pipeline map

```
1_generate_ndif.py     persona rollouts on NDIF
2_activations_ndif.py  rollouts -> persona vectors      (init_ndif lives here; hardens all callers)
3_judge_ndif.py        judge rollouts 0-3
6_axes.py              vectors -> axes + per-behaviour PCA + sign pinning
8_coords_ndif.py       eval corpus -> pooled activations (Passes A/B/C)
9_persona_probes.py    P1 transfer / P2 sample-efficiency / P3 LOSO, controls, zero-shot
10_sycophancy_ndif.py  sycophancy half: prep -> generate -> label
ndif_common.py         download floor + exception-tail logging, shared
eval_datasets.py       one EvalRow over three input shapes; 15,278 rows
pc_gate.py             the cosine gate: is paper16 cleaner than own100?
```

**A trace body must be defined in the script that runs it.** nnsight serialises the block from
its source module, and NDIF rejects one belonging to an importlib-loaded module with
`Module importlib is not whitelisted`. Import helpers like `init_ndif` freely; never import a
function that appears inside `trace`/`generate`/`session`.
