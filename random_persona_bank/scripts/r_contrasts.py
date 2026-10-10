"""R5-R9: the random-split nulls for D5-D9, from one random-persona bank, at every layer.

Same recipe as xmodel/d_contrasts.py (zami's persona_pcs._axes_from_groups): mean(positive roles) - mean(negative
roles) per layer in float64, unit-normalised for use; R7 and R9 are POOLED means over their union of groups, exactly as
D7 and D9 are. The groups come from bank/splits.json: the 35 random roles dealt into A-F with the kd groups' sizes.

    R5 = A v B (8 v 7)   R6 = C v D (5 v 5)   R7 = A+C v B+D (13 v 12)   R8 = E v F (4 v 6)   R9 = A+E v B+F (12 v 13)

Writes, for split 0 only (the primary), <out>/r5/R5_<tag>.pt ... <out>/r9/R9_<tag>.pt with the same keys as the D files
(`raw`, `unit` [n_layers, hidden] float32, `norm`, groups, rollouts and md5 per role) + a .json sidecar, so anything that
reads a D file reads an R file. And, for all 100 splits, <out>/r_null/R_null_<tag>.csv: per split, contrast and layer --
the norm, the norm relative to the bank's role spread (RMS distance of its 36 vectors from their mean), and the cosine
with the bank's assistant direction (default - mean of the 35 roles). Refuses to overwrite.

    python random_persona_bank/scripts/r_contrasts.py --vectors <run>/persona/vectors_s23 --out <run> --tag s23 \
        --model meta-llama/Llama-3.3-70B-Instruct

--check_groups FILE: build from a groups file instead of splits.json (used to show this script reproduces D5-D9 when
given the kd groups); writes nothing, compares with --compare_dir's D files.
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPLITS = HERE.parent / "bank" / "splits.json"
RECIPE = "zami persona_pcs._axes_from_groups (as xmodel/d_contrasts.py), groups from random_persona_bank/bank/splits.json"
REPORT_LAYERS = (20, 31, 36, 40, 44, 63)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bank(vectors: Path):
    out, meta, layers = {}, {}, None
    for p in sorted(vectors.glob("*.pt")):
        payload = torch.load(p, weights_only=False)
        v = payload["vector"].double().numpy()
        ls = [int(x) for x in payload.get("layers", range(v.shape[0]))]
        if layers is None:
            layers = ls
        if ls != layers:
            sys.exit(f"{p.name}: layers differ from the bank's")
        if not np.isfinite(v).all():
            sys.exit(f"{p.name}: non-finite values")
        out[p.stem] = v
        meta[p.stem] = {"n": int(payload["n"]), "type": payload.get("type"), "md5": md5(p)}
    return out, meta, layers


def contrasts_from(groups: dict, defs: dict):
    """-> {R: (positive roles, negative roles)} from groups A..F and the splits.json contrast definitions."""
    return {k: ([r for g in d["positive"] for r in groups[g]], [r for g in d["negative"] for r in groups[g]])
            for k, d in defs.items()}


def build(vec, pos, neg):
    raw = np.stack([vec[r] for r in pos]).mean(0) - np.stack([vec[r] for r in neg]).mean(0)
    norm = np.linalg.norm(raw, axis=1)
    if not (np.isfinite(raw).all() and (norm > 0).all()):
        sys.exit("non-finite or zero contrast")
    return raw, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tag")
    ap.add_argument("--model", default="")
    ap.add_argument("--splits", type=Path, default=SPLITS)
    ap.add_argument("--check_groups", type=Path, help="validation: a {A..F: roles} JSON instead of the splits")
    ap.add_argument("--compare_dir", type=Path, help="validation: run folder holding d5..d9/D*_<compare_tag>.pt")
    ap.add_argument("--compare_tag")
    a = ap.parse_args()

    S = json.loads(a.splits.read_text(encoding="utf-8"))
    defs = S["contrasts"]
    vec, meta, layers = load_bank(a.vectors)
    hidden = next(iter(vec.values())).shape[1]

    if a.check_groups:
        groups = json.loads(a.check_groups.read_text(encoding="utf-8"))
        worst = 0.0
        for k, (pos, neg) in contrasts_from(groups, defs).items():
            raw, norm = build(vec, pos, neg)
            d = torch.load(a.compare_dir / f"d{k[1]}" / f"D{k[1]}_{a.compare_tag}.pt", weights_only=False)
            ref = d["raw"].double().numpy()
            rel = np.abs(raw - ref).max() / np.abs(ref).max()
            same = sorted(d["positive"]) == sorted(pos) and sorted(d["negative"]) == sorted(neg)
            print(f"{k} vs D{k[1]}: groups identical {same}; max relative difference {rel:.2e}")
            worst = max(worst, rel)
        sys.exit(0 if worst < 1e-6 else 1)

    if not (a.out and a.tag):
        sys.exit("--out and --tag are required")
    roles = sorted(r for r in vec if r != "default")
    if "default" not in vec or len(roles) != 35:
        sys.exit(f"expected default + 35 roles in {a.vectors}, found {len(vec)}")
    if sorted(r for g in S["splits"][0]["groups"].values() for r in g) != roles:
        sys.exit("split 0's roles are not this bank's 35 roles")

    allv = np.stack([vec[r] for r in sorted(vec)])                       # [36, L, H]
    spread = np.sqrt(((allv - allv.mean(0)) ** 2).sum(-1).mean(0))       # [L] RMS distance from the bank mean
    asst = vec["default"] - np.stack([vec[r] for r in roles]).mean(0)    # [L, H]
    asst_u = asst / np.linalg.norm(asst, axis=1, keepdims=True)

    targets = [a.out / f"r{k[1]}" / f"{k}_{a.tag}.{e}" for k in defs for e in ("pt", "json")]
    null_csv = a.out / "r_null" / f"R_null_{a.tag}.csv"
    clash = [p for p in targets + [null_csv] if p.exists()]
    if clash:
        sys.exit(f"{clash[0]} exists; refusing to overwrite")

    stamp = datetime.now().isoformat(timespec="seconds")
    splits_md5 = md5(a.splits)
    rows = []
    for sp in S["splits"]:
        for k, (pos, neg) in contrasts_from(sp["groups"], defs).items():
            raw, norm = build(vec, pos, neg)
            unit = raw / norm[:, None]
            cos_a = (unit * asst_u).sum(1)
            for i, l in enumerate(layers):
                rows.append((sp["split"], k, l, norm[i], norm[i] / spread[i], cos_a[i]))
            if sp["split"] != 0:
                continue
            d = a.out / f"r{k[1]}"
            d.mkdir(parents=True, exist_ok=True)
            pt, js = d / f"{k}_{a.tag}.pt", d / f"{k}_{a.tag}.json"
            desc = f"random-split null for D{k[1]} ({defs[k]['sizes']}), split 0"
            torch.save({"name": k, "description": desc, "positive": pos, "negative": neg, "split": 0,
                        "raw": torch.from_numpy(raw).float(), "unit": torch.from_numpy(unit).float(),
                        "norm": torch.from_numpy(norm), "layers": layers, "hidden": hidden,
                        "bank": str(a.vectors), "bank_tag": a.tag, "model": a.model, "recipe": RECIPE,
                        "splits_md5": splits_md5, "roles": {n: meta[n] for n in pos + neg}, "created": stamp}, pt)
            js.write_text(json.dumps({
                "name": k, "description": desc, "model": a.model, "bank": str(a.vectors), "bank_tag": a.tag,
                "recipe": RECIPE, "split": 0, "splits_md5": splits_md5, "positive": pos, "negative": neg,
                "rollouts_per_role": {n: meta[n]["n"] for n in pos + neg},
                "input_md5": {n: meta[n]["md5"] for n in pos + neg}, "layers": len(layers), "hidden": hidden,
                "norm_at": {f"L{l}": round(float(norm[layers.index(l)]), 6) for l in REPORT_LAYERS if l in layers},
                "pt_md5": md5(pt), "created": stamp,
                "use": "score = activation @ unit[layer]; the sign is arbitrary (random groups)"}, indent=2),
                encoding="utf-8")
    null_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(null_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "contrast", "layer", "norm", "rel_norm", "cos_assistant"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.6f}", f"{r[4]:.6f}", f"{r[5]:+.6f}"])

    print(f"{len(vec)} vectors, {len(layers)} layers x {hidden}; wrote R5-R9 (split 0) and {len(rows)} null rows")
    arr = {(s, k, l): (rn, ca) for s, k, l, _, rn, ca in rows}
    for l in REPORT_LAYERS:
        if l not in layers:
            continue
        line = [f"L{l}"]
        for k in defs:
            rn = np.array([arr[(s, k, l)][0] for s in range(len(S["splits"]))])
            ca = np.abs([arr[(s, k, l)][1] for s in range(len(S["splits"]))])
            line.append(f"{k} rel {arr[(0, k, l)][0]:.3f} (null {np.percentile(rn, 2.5):.3f}-{np.percentile(rn, 97.5):.3f}), "
                        f"|cos asst| med {np.median(ca):.2f}")
        print("  ".join(line))


if __name__ == "__main__":
    main()
