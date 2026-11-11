"""Parity of a 150-question per-question store on its first 30 questions (a GATE-0 for the q150 runs).

`question_bank.gate0` recombines every question of a store, so it only fits the 30-question store. This check
restricts a q150 store (`store_<tag>_L*.npz`, written by `question_bank.py store` or
`question_bank_aletheia.py store`) to questions 0-29 and compares the n-weighted recombination per role and layer
with
  (a) the reference bank `thesis_dir` (Llama: the thesis bank `persona_pcs.EXTENDED_VECTORS_DIR`; Qwen3.5: the
      Aletheia bank `E:/aletheia/banks/Qwen3.5-27B/persona/vectors_s3`, read only): cosine >= 0.9995, equal n;
  (b) optionally a 30-question store `ref_tag` in the same output folder (Llama: the GATE-0 store `s3q30`),
      vectors and n cell by cell.
Development data only; no AUROC is computed. Writes <out_dir>/q150_parity.csv and .md.

    python experiments/external/questions_150/q150_store_parity.py                       # Llama
    python experiments/external/questions_150/q150_store_parity.py --out_dir results/question_bank/qwen35 \
        --thesis_dir E:/aletheia/banks/Qwen3.5-27B/persona/vectors_s3 --ref_tag "" --layers 16,19,22,26,29,32
"""

from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection import persona_pcs as pcs
from experiments.external import question_bank as QB


def main(tag: str = "s3", ref_tag: str = "s3q30", layers=QB.LAYERS, tol_cos: float = 0.9995,
         out_dir: str = "", thesis_dir: str = "") -> None:
    import torch

    QB.freeze_check()
    out = Path(out_dir) if out_dir else QB.OUT
    if not out.is_absolute():
        out = QB.REPO_ROOT / out
    tdir = Path(thesis_dir) if thesis_dir else pcs.EXTENDED_VECTORS_DIR
    if isinstance(layers, (int, str)):
        layers = [int(x) for x in str(layers).split(",")]
    rows, cells, fails = [], [], []
    for L in layers:
        V, n, roles, qids = QB.load_store(tag, L, out)
        pos = [qids.index(q) for q in range(30)]
        vec = QB.bank_vectors(V, n, pos, "plain")
        if ref_tag:
            Vr, nr, roles_r, qids_r = QB.load_store(ref_tag, L, out)
            assert roles == roles_r, "role order differs between the stores"
            pos_r = [qids_r.index(q) for q in range(30)]
            vec_r = QB.bank_vectors(Vr, nr, pos_r, "plain")
        for i, role in enumerate(roles):
            d = torch.load(tdir / f"{role}.pt", weights_only=True)
            sl = [int(x) for x in d.get("layers", range(d["vector"].shape[0]))]
            ref = d["vector"][sl.index(L)].double().numpy()
            cos_t = float(vec[i] @ ref / (np.linalg.norm(vec[i]) * np.linalg.norm(ref)))
            n_new, n_thesis = int(n[i, pos].sum()), int(d["n"])
            row = {"layer": L, "role": role, "cos_vs_reference_bank": cos_t, "n_q150": n_new, "n_reference": n_thesis}
            if cos_t < tol_cos:
                fails.append(f"L{L} {role}: cos vs reference bank {cos_t:.6f} < {tol_cos}")
            if n_new != n_thesis:
                fails.append(f"L{L} {role}: n {n_new} != reference {n_thesis}")
            if ref_tag:
                row["cos_vs_q30_store"] = float(vec[i] @ vec_r[i] / (np.linalg.norm(vec[i]) * np.linalg.norm(vec_r[i])))
                row["n_q30_store"] = int(nr[i, pos_r].sum())
                for j, (p, pr) in enumerate(zip(pos, pos_r)):
                    if n[i, p] == 0 and nr[i, pr] == 0:
                        continue
                    a, b = V[i, p].astype(np.float64), Vr[i, pr].astype(np.float64)
                    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) if n[i, p] and nr[i, pr] else float("nan")
                    cells.append({"layer": L, "role": role, "q": j, "cos": c, "n_q150": int(n[i, p]), "n_q30": int(nr[i, pr])})
            rows.append(row)
    t = pd.DataFrame(rows)
    t.round(8).to_csv(out / "q150_parity.csv", index=False)
    lines = [f"q150 store parity on questions 0-29 ({'PASSED' if not fails else 'FAILED'}); reference bank {tdir}", "",
             f"roles x layers: {len(t)}; min cos vs reference bank {t['cos_vs_reference_bank'].min():.7f}; "
             f"n mismatches vs reference {int((t['n_q150'] != t['n_reference']).sum())}"]
    cell_n_mis = 0
    if ref_tag:
        c = pd.DataFrame(cells)
        cell_n_mis = int((c["n_q150"] != c["n_q30"]).sum())
        lines.append(f"vs 30-question store {ref_tag}: min role cos {t['cos_vs_q30_store'].min():.7f}; cells compared "
                     f"{len(c)}; min cell cos {np.nanmin(c['cos']):.7f}; cell n mismatches {cell_n_mis}")
    lines += fails
    (out / "q150_parity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if fails or cell_n_mis:
        raise SystemExit(1)


if __name__ == "__main__":
    fire.Fire(main)
