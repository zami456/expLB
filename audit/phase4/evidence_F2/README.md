# Evidence pulled from the lab PC by the Phase 4 batch-F2 agent

These 28 files were pulled read-only from `C:\Users\user4\expLB\` while re-deriving batch F2.
The agent left them in a scratch directory (`_f2work`); the lead **kept and renamed** them rather
than deleting, because several are the evidence chain for F2's verdicts and at least three are
**the only surviving source** for numbers the audit had previously recorded as unrecoverable.

## Why these are worth keeping

- **`crossgen_scale_{3b,8b,70b}.log`** — the crossgen logs are **append-mode**, so the
  2026-09-20 22:18–22:21 block survives *beneath* the 23:38–23:40 mask re-run that overwrote
  `run-crossgen/scale_{3b,8b}/matrices/*.json`. These logs are how batch F2 recovered **F-58's 3B/8B
  rows**, which `audit/claims.csv` describes as existing *"nowhere (lab, git, Drive)"*. See the F-58
  entry in `audit/CORRECTIONS.md` (adjudicated BOTH PARTLY RIGHT on exactly this point).
- **`cap_{all,s23}_mean_L{20,40,44}_s42.json`, `kdbank_mean_L40_s42.json`** — `run-crossgen/kd11/banks/`
  matrices that exist **only on the lab PC**; no git copy. They back the cap-bank claims in F2, F3,
  M1 and O1, including the `cap_s23` vs `cap_all` bank distinction that decides O-046.

## Provenance and status — read before citing

- **Source:** the lab PC, read-only, via the batch-F2 agent. Nothing was written on the lab PC.
- **NOT lead-verified.** The lead did **not** re-run a lab-side `Get-FileHash` comparison for these,
  so they are **not** entered in `audit/cache/MANIFEST.csv`, whose rows are all md5-MATCH verified.
  Treat them as agent-pulled working copies, not as a certified cache.
- `err_*.txt`, `e2.txt`, `cap_scores.txt` are the agent's own stdout/stderr, kept only so its
  derivations can be retraced.
- To promote any of these to the certified cache, re-pull with the verified
  `ssh … "tar -cf - …"` route and md5-check against the lab copy, as `audit/cache/MANIFEST.csv` does.
