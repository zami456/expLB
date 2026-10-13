"""Per-row scores for T1-4 (review W6): D9, K and D9 + K on Llama-3.3-70B's five Liars' Bench datasets.

zami's per-row scores were never committed (`results/` is gitignored on origin/paper), so this rebuilds
them on the laptop CPU from our own stores, with zami's recipe:

    score_D9   = x @ D9_unit[layer]                       (persona_pcs extended contrast, 12 v 13)
    score_K    = x @ K[layer]                             (recall_denial none|general, unit)
    score_D9K  = score_D9 / sd_A(D9) + score_K / sd_A(K)  (recall_denial.standardised_sum, :428-431)

x = the reply's mean-pooled residual stream (8_coords_ndif.py Pass B store, span "response", pooling
"mean"); sd_A = numpy std (ddof 0) of the projection over the model's own 2,000 Liars' Bench Alpaca rows.

D9 is scored for all three persona banks (s3, s23, all) so the sanity step can pick the one that
reproduces zami's numbers BEFORE the OR-rule is computed. Alpaca projections come from the D9 + K
files (d9k.py stores them), so Alpaca is not re-read.

Writes, next to this script:
    per_row_scores.csv    one line per (row, layer): dataset, row_id, label, D9_<bank>, K
    alpaca_scores.csv     one line per (Alpaca row, layer): row_id, D9_<bank>, K
    inputs.json           every input path with its md5

    python review_response/T1-4_or_rule/score_rows.py --passb %USERPROFILE%/xmodel-runs/_scratch/passB_llama \
        --passq11 %USERPROFILE%/xmodel-runs/_scratch/passQ11_llama

Gender-Secret (added 2026-10-12) is scored from Pass Q11, the base model reading the LoRA's text, as "GS-base".
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = Path.home() / "xmodel-runs" / "Llama-3.3-70B-Instruct"
K_PATH = REPO / "random_persona_bank" / "results" / "Llama-3.3-70B-Instruct" / "K" / "llama-70b-3.3" / "first" / "K.npz"
BANKS = ("s3", "s23", "all")
LAYERS = (20, 36, 40, 44)  # zami's k_robustness LAYERS; 40 is the pre-registered one
DATASETS = {  # Pass B folder -> zami's short name (recall_denial.SHORT)
    "convincing-game": "CG",
    "harm-pressure-choice": "HP-C",
    "harm-pressure-knowledge-report": "HP-KR",
    "insider-trading": "IT",
    "instructed-deception": "ID",
}
# Added 2026-10-12 (owner: "run it on gender secret too"): Pass Q11's Gender-Secret store is the BASE model
# teacher-forced on the LoRA's text (20_q11_sleeper.py:274) -- zami's "base" read, not his "adapted" one.
EXTRA = {"gender-secret": "GS-base"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passb", required=True, type=Path)
    ap.add_argument("--passq11", required=True, type=Path, help="Pass Q11 store (holds gender-secret/)")
    a = ap.parse_args()

    out_rows, out_alp, meta = HERE / "per_row_scores.csv", HERE / "alpaca_scores.csv", HERE / "inputs.json"
    if out_rows.exists() or out_alp.exists():
        sys.exit("outputs exist; refusing to overwrite")

    # directions at the four layers
    kz = np.load(K_PATH)
    K = np.stack([kz[f"L{l}"] for l in LAYERS]).astype(np.float64)
    assert np.allclose(np.linalg.norm(K, axis=1), 1, atol=1e-6), "K is not unit"
    D9, d9k = {}, {}
    for b in BANKS:
        d = torch.load(RUN / "d9" / f"D9_{b}.pt", weights_only=False)
        assert d["name"] == "D9"
        lay = list(d["layers"])
        D9[b] = d["unit"].double().numpy()[[lay.index(l) for l in LAYERS]]
        assert np.allclose(np.linalg.norm(D9[b], axis=1), 1, atol=1e-5), f"D9_{b} is not unit"
        d9k[b] = torch.load(RUN / "d9k" / f"D9K_{b}.pt", weights_only=False)
        # the D9 + K file must have been built from this D9 and this K
        assert d9k[b]["d9_md5"] == md5(RUN / "d9" / f"D9_{b}.pt"), f"D9K_{b} was built from another D9"
        assert d9k[b]["k_md5"] == md5(K_PATH), f"D9K_{b} was built from another K"

    # Alpaca projections, from the D9 + K files (the same 2,000 rows for every bank)
    ids0 = d9k[BANKS[0]]["row_ids"]
    lay_a = list(d9k[BANKS[0]]["layers"])
    li = [lay_a.index(l) for l in LAYERS]
    alp = []
    for j, l in enumerate(LAYERS):
        rec = pd.DataFrame({"row_id": ids0, "layer": l})
        for b in BANKS:
            assert d9k[b]["row_ids"] == ids0, "Alpaca rows differ between banks"
            rec[f"D9_{b}"] = d9k[b]["alpaca_proj_d9"].double().numpy()[:, li[j]]
        rec["K"] = d9k[BANKS[0]]["alpaca_proj_k"].double().numpy()[:, li[j]]
        for b in BANKS:  # same K in every file
            assert np.allclose(d9k[b]["alpaca_proj_k"].double().numpy()[:, li[j]], rec["K"], atol=1e-4)
        alp.append(rec)
    alp = pd.concat(alp, ignore_index=True)

    # the five datasets, streamed chunk by chunk
    recs = []
    stores = [(a.passb / f, s) for f, s in DATASETS.items()] + [(a.passq11 / f, s) for f, s in EXTRA.items()]
    for root, short in stores:
        folder = root.name
        labels = {}
        with open(root / "rows.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                labels[r["row_id"]] = r["label"]
        want = [it[0] for it in torch.load(root / "items.pt", weights_only=False)]
        seen = []
        for c in sorted(root.glob("chunk_*.pt")):
            ch = torch.load(c, weights_only=False)
            lay = [int(x) for x in ch["layers"]]
            # Pass B / Q11 chunks predate the "span" field (always the reply there); later stores carry it
            if list(ch["pooling"])[0] != "mean" or ch.get("span", "response") != "response":
                sys.exit(f"{folder}/{c.name}: pooling {ch['pooling']} span {ch.get('span')}")
            x = ch["pooled"][:, [lay.index(l) for l in LAYERS], :, 0].double().numpy()  # (n, 4, H) mean pooling
            if not np.isfinite(x).all():
                sys.exit(f"{folder}/{c.name}: non-finite values")
            pk = np.einsum("nlh,lh->nl", x, K)
            pd9 = {b: np.einsum("nlh,lh->nl", x, D9[b]) for b in BANKS}
            for i, rid in enumerate(ch["row_ids"]):
                seen.append(rid)
                for j, l in enumerate(LAYERS):
                    rec = {"dataset": short, "row_id": rid, "label": labels[rid], "layer": l, "K": pk[i, j]}
                    rec.update({f"D9_{b}": pd9[b][i, j] for b in BANKS})
                    recs.append(rec)
            del x, ch
        if len(set(seen)) != len(seen):
            sys.exit(f"{folder}: a row appears twice")
        if set(seen) != set(want):
            sys.exit(f"{folder}: {len(set(want) - set(seen))} rows unread, {len(set(seen) - set(want))} unlisted")
        print(f"{short:6s} {folder:32s} rows {len(seen):5d}  labels {pd.Series([labels[r] for r in seen]).value_counts().to_dict()}")
    df = pd.DataFrame(recs)

    df.to_csv(out_rows, index=False, float_format="%.6f")
    alp.to_csv(out_alp, index=False, float_format="%.6f")
    inputs = {"K": [str(K_PATH), md5(K_PATH)]}
    for b in BANKS:
        inputs[f"D9_{b}"] = [str(RUN / "d9" / f"D9_{b}.pt"), md5(RUN / "d9" / f"D9_{b}.pt")]
        inputs[f"D9K_{b}"] = [str(RUN / "d9k" / f"D9K_{b}.pt"), md5(RUN / "d9k" / f"D9K_{b}.pt")]
    inputs["stores"] = {str(r): {p.name: md5(p) for p in sorted(r.iterdir()) if p.suffix in (".pt", ".jsonl")}
                        for r, _ in stores}
    inputs["sd_alpaca"] = {b: {f"L{l}": {"sd_d9": float(alp[alp.layer == l][f"D9_{b}"].std(ddof=0)),
                                         "sd_k": float(alp[alp.layer == l]["K"].std(ddof=0))} for l in LAYERS}
                           for b in BANKS}
    meta.write_text(json.dumps(inputs, indent=1), encoding="utf-8")
    print(f"wrote {out_rows.name} ({len(df)} lines), {out_alp.name} ({len(alp)} lines), {meta.name}")


if __name__ == "__main__":
    main()
