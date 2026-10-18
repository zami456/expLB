"""Parity of the 150-question per-question store on its first 30 questions (a GATE-0 for the q150 run).

`question_bank.gate0` recombines every question of a store, so it only fits the 30-question store. This check
restricts the q150 store (`store_<tag>_L*.npz`, written by `question_bank.py store`) to questions 0-29 and
compares the n-weighted recombination per role and layer with
  (a) the thesis bank (`persona_pcs.EXTENDED_VECTORS_DIR`, the GATE-0 reference): cosine >= 0.9995, equal n;
  (b) the GATE-0 store of the 30 originals (`store_s3q30_L*.npz`): cosine and n, cell by cell as well.
Development data only; no AUROC is computed. Writes results/question_bank/llama/q150_parity.csv and .md.

    python deception_detection/scripts/questions_v2/q150_store_parity.py [--tag s3] [--ref_tag s3q30]
"""

from __future__ import annotations

import fire
import numpy as np
import pandas as pd

from deception_detection import persona_pcs as pcs
from deception_detection.scripts import question_bank as QB


def main(tag: str = "s3", ref_tag: str = "s3q30", layers=QB.LAYERS, tol_cos: float = 0.9995) -> None:
    import torch

    QB.freeze_check()
    rows, cells, fails = [], [], []
    for L in layers:
        V, n, roles, qids = QB.load_store(tag, L)
        Vr, nr, roles_r, qids_r = QB.load_store(ref_tag, L)
        assert roles == roles_r, "role order differs between the stores"
        pos = [qids.index(q) for q in range(30)]
        pos_r = [qids_r.index(q) for q in range(30)]
        vec = QB.bank_vectors(V, n, pos, "plain")
        vec_r = QB.bank_vectors(Vr, nr, pos_r, "plain")
        for i, role in enumerate(roles):
            d = torch.load(pcs.EXTENDED_VECTORS_DIR / f"{role}.pt", weights_only=True)
            sl = [int(x) for x in d.get("layers", range(d["vector"].shape[0]))]
            ref = d["vector"][sl.index(L)].double().numpy()
            cos_t = float(vec[i] @ ref / (np.linalg.norm(vec[i]) * np.linalg.norm(ref)))
            cos_s = float(vec[i] @ vec_r[i] / (np.linalg.norm(vec[i]) * np.linalg.norm(vec_r[i])))
            n_new, n_thesis, n_ref = int(n[i, pos].sum()), int(d["n"]), int(nr[i, pos_r].sum())
            rows.append({"layer": L, "role": role, "cos_vs_thesis": cos_t, "cos_vs_q30_store": cos_s,
                         "n_q150": n_new, "n_thesis": n_thesis, "n_q30_store": n_ref})
            if cos_t < tol_cos:
                fails.append(f"L{L} {role}: cos vs thesis {cos_t:.6f} < {tol_cos}")
            if n_new != n_thesis:
                fails.append(f"L{L} {role}: n {n_new} != thesis {n_thesis}")
            for j, (p, pr) in enumerate(zip(pos, pos_r)):
                a, b = V[i, p].astype(np.float64), Vr[i, pr].astype(np.float64)
                if n[i, p] == 0 and nr[i, pr] == 0:
                    continue
                c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) if n[i, p] and nr[i, pr] else float("nan")
                cells.append({"layer": L, "role": role, "q": j, "cos": c, "n_q150": int(n[i, p]), "n_q30": int(nr[i, pr])})
    t, c = pd.DataFrame(rows), pd.DataFrame(cells)
    out = QB.OUT
    t.round(8).to_csv(out / "q150_parity.csv", index=False)
    cell_n_mis = int((c["n_q150"] != c["n_q30"]).sum())
    lines = [f"q150 store parity on questions 0-29 ({'PASSED' if not fails else 'FAILED'})", "",
             f"roles x layers: {len(t)}; min cos vs thesis bank {t['cos_vs_thesis'].min():.7f}; "
             f"min cos vs q30 store {t['cos_vs_q30_store'].min():.7f}; n mismatches vs thesis "
             f"{int((t['n_q150'] != t['n_thesis']).sum())}",
             f"cells compared with the q30 store: {len(c)}; min cell cos {np.nanmin(c['cos']):.7f}; cell n mismatches {cell_n_mis}"]
    lines += fails
    (out / "q150_parity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if fails or cell_n_mis:
        raise SystemExit(1)


if __name__ == "__main__":
    fire.Fire(main)
