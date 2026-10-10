"""D5, D8 and D9 -- the zero-shot persona contrasts of report section 4.2.7 -- for one persona bank, at every layer.

The recipe is zami's, copied rather than imported (`persona_pcs._axes_from_groups`, origin/paper @ e27ffaf9,
`deception_detection/persona_pcs.py:39-169`): each contrast is mean(positive roles) - mean(negative roles),
computed in float64 per layer; `axis_unit` divides by the norm. D9 is the POOLED mean over 12 v 13 roles,
not the average of D5 and D8.

    D5  published contrast       8 deception        v 7 honesty
    D8  knowledge-denial         4 knowledge denial v 6 knowledge candour (sincere ignorance + plain assertion)
    D9  extended                 D5's 8 + D8's 4    v D5's 7 + D8's 6

zami builds them from the score-3 bank (`run-kd/vectors_s3`). On Qwen2.5-72B that bank keeps only 25 of the
36 roles and loses 8 of the 25 these contrasts need, so the Qwen contrasts are built from the s23 bank (scores
2|3), as the audit re-created zami's arms on Llama's s23 bank (`thesis_guide/PITFALLS.md:47`).

Writes <out>/d5/D5_<tag>.pt, <out>/d8/D8_<tag>.pt, <out>/d9/D9_<tag>.pt (+ a .json sidecar each). Each .pt holds
`raw` and `unit` [n_layers, hidden] float32, `norm` per layer (float64), the groups, each role's rollout count
and input md5. Refuses to overwrite an existing output.

    python xmodel/d_contrasts.py --vectors <persona/vectors_s23_v2> --out <run folder> --tag s23_v2 \
        --axes <persona/axes_vectors_s23_v2.pt>

D6 and D7 (added 2026-10-08, report Table 4.1; built only when asked, `--only D6,D7`), from the same function:

    D6  sycophancy contrast      5 sycophancy       v 5 anti-sycophancy        (zami's "sycophancy" axis)
    D7  unified contrast         D5's 8 + D6's 5    v D5's 7 + D6's 5 (13 v 12) (zami's "unified" axis, pooled mean)

With `--axes`, D6 and D7 must equal the pipeline's own sycophancy and unified axes, as D5 must its deception axis.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

DECEPTION = ("pc_handler", "pc_broker", "pc_impression_manager", "pc_coverup_operator", "pc_gatekeeper",
             "pc_spin_doctor", "pc_plausible_denier", "pc_narrative_manager")
HONESTY = ("pc_archivist", "pc_disclosure_counsel", "pc_documentarian", "pc_fact_checker", "pc_ombudsman",
           "pc_transparency_officer", "pc_whistleblower")
KNOWLEDGE_DENIAL = ("pc_amnesiac_witness", "pc_deniable_expert", "pc_feigned_novice", "pc_stonewaller")
KNOWLEDGE_CANDOUR = ("pc_calibrated_reporter", "pc_candid_witness", "pc_forthright_expert", "pc_honest_novice",
                     "pc_plain_dealer", "pc_scrupulous_witness")
SYCOPHANCY = ("pc_cheerleader", "pc_enabler", "pc_flatterer", "pc_people_pleaser", "pc_yes_man")
ANTI_SYCOPHANCY = ("pc_blunt_advisor", "pc_contrarian", "pc_critic", "pc_skeptic", "pc_tough_love_mentor")

CONTRASTS = {
    "D5": ("published contrast (8 v 7)", DECEPTION, HONESTY),
    "D6": ("sycophancy contrast (5 v 5)", SYCOPHANCY, ANTI_SYCOPHANCY),
    "D7": ("unified contrast (13 v 12)", DECEPTION + SYCOPHANCY, HONESTY + ANTI_SYCOPHANCY),
    "D8": ("knowledge-denial contrast (4 v 6)", KNOWLEDGE_DENIAL, KNOWLEDGE_CANDOUR),
    "D9": ("extended contrast (12 v 13)", DECEPTION + KNOWLEDGE_DENIAL, HONESTY + KNOWLEDGE_CANDOUR),
}
# the pipeline's axes file names three of them; D8 and D9 have no stored counterpart
PIPELINE_AXIS = {"D5": "deception", "D6": "sycophancy", "D7": "unified"}
REPORT_LAYERS = (20, 36, 40, 44)
RECIPE = "zami persona_pcs._axes_from_groups, origin/paper @ e27ffaf9 (persona_pcs.py:39-169)"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(vectors: Path, names):
    out, meta, layers = {}, {}, None
    for n in names:
        p = vectors / f"{n}.pt"
        if not p.exists():
            raise FileNotFoundError(f"{n} is not in {vectors}")
        payload = torch.load(p, weights_only=False)
        v = payload["vector"].double().numpy()
        ls = [int(x) for x in payload.get("layers", range(v.shape[0]))]
        if layers is None:
            layers = ls
        if ls != layers or v.shape[0] != len(ls):
            raise ValueError(f"{n}: layers {ls[:3]}..{ls[-1]} ({v.shape[0]} rows) differ from the bank's")
        if not np.isfinite(v).all():
            raise ValueError(f"{n}: non-finite values")
        out[n] = v
        meta[n] = {"n": int(payload["n"]), "type": payload.get("type"), "md5": md5(p)}
    return out, meta, layers


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--axes", type=Path, help="the pipeline's axes file for this bank: D5 must equal its deception axis")
    ap.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    ap.add_argument("--only", default="D5,D8,D9", help="comma-separated contrasts to build (D5,D6,D7,D8,D9)")
    a = ap.parse_args()

    build = [k.strip().upper() for k in a.only.split(",") if k.strip()]
    if not build or set(build) - set(CONTRASTS):
        sys.exit(f"--only {a.only}: choose from {', '.join(CONTRASTS)}")
    contrasts = {k: CONTRASTS[k] for k in CONTRASTS if k in build}
    names = sorted({n for _, pos, neg in contrasts.values() for n in pos + neg})
    vec, meta, layers = load(a.vectors, names)
    hidden = next(iter(vec.values())).shape[1]

    if a.axes:  # the pipeline's own groups must be zami's, and its axes must be the contrasts it names
        ax = torch.load(a.axes, weights_only=False)
        g = ax["role_groups"]
        groups = {k: tuple(sorted(v["roles"] if isinstance(v, dict) else v)) for k, v in g.items()}
        checks = {"deception": DECEPTION, "honesty": HONESTY, "knowledge_denial": KNOWLEDGE_DENIAL,
                  "sycophancy": SYCOPHANCY, "anti_sycophancy": ANTI_SYCOPHANCY}
        for k, want in checks.items():
            if groups[k] != tuple(sorted(want)):
                sys.exit(f"group {k}: pipeline {groups[k]} != zami {sorted(want)}")
        if tuple(sorted(groups["sincere_ignorance"] + groups["plain_assertion"])) != tuple(sorted(KNOWLEDGE_CANDOUR)):
            sys.exit("sincere_ignorance + plain_assertion != zami's knowledge candour group")
        print("groups: the pipeline's role_groups match zami's (candour = sincere_ignorance + plain_assertion)")

    results = {}
    for key, (desc, pos, neg) in contrasts.items():
        raw = np.stack([vec[n] for n in pos]).mean(0) - np.stack([vec[n] for n in neg]).mean(0)
        norm = np.linalg.norm(raw, axis=1)
        if not (np.isfinite(raw).all() and (norm > 0).all()):
            raise ValueError(f"{key}: non-finite or zero contrast")
        results[key] = (desc, pos, neg, raw, norm)

    # D7 and D9 are pooled means, which are NOT the averages of their parts (unequal group sizes)
    def group_mean(g):
        return np.stack([vec[n] for n in g]).mean(0)
    if "D9" in results:
        pooled = ((8 * group_mean(DECEPTION) + 4 * group_mean(KNOWLEDGE_DENIAL)) / 12
                  - (7 * group_mean(HONESTY) + 6 * group_mean(KNOWLEDGE_CANDOUR)) / 13)
        assert np.allclose(pooled, results["D9"][3], atol=1e-10), "D9 is not the 12 v 13 pooled mean"
    if "D7" in results:
        pooled = ((8 * group_mean(DECEPTION) + 5 * group_mean(SYCOPHANCY)) / 13
                  - (7 * group_mean(HONESTY) + 5 * group_mean(ANTI_SYCOPHANCY)) / 12)
        assert np.allclose(pooled, results["D7"][3], atol=1e-10), "D7 is not the 13 v 12 pooled mean"

    if a.axes:
        for key, axis in PIPELINE_AXIS.items():
            if key not in results:
                continue
            d = results[key][3]
            stored = ax["axes"][axis]
            stored = stored.double().numpy() if hasattr(stored, "double") else np.asarray(stored, dtype=np.float64)
            if stored.shape != d.shape:
                sys.exit(f"axes {axis} shape {stored.shape} != {key} {d.shape}")
            diff = np.abs(stored - d).max()
            rel = diff / np.abs(d).max()
            print(f"check: {key} vs the pipeline's {axis} axis: max |diff| {diff:.3e} (relative {rel:.3e})")
            if rel > 1e-5:
                sys.exit(f"{key} does not reproduce the pipeline's {axis} axis")

    # refuse before writing anything, so a clash never leaves half a set behind
    for key in results:
        for ext in ("pt", "json"):
            if (a.out / key.lower() / f"{key}_{a.tag}.{ext}").exists():
                sys.exit(f"{a.out / key.lower() / f'{key}_{a.tag}.{ext}'} exists; refusing to overwrite")

    stamp = datetime.now().isoformat(timespec="seconds")
    for key, (desc, pos, neg, raw, norm) in results.items():
        d = a.out / key.lower()
        d.mkdir(parents=True, exist_ok=True)
        pt, js = d / f"{key}_{a.tag}.pt", d / f"{key}_{a.tag}.json"
        unit = raw / norm[:, None]
        torch.save({"name": key, "description": desc, "positive": list(pos), "negative": list(neg),
                    "raw": torch.from_numpy(raw).float(), "unit": torch.from_numpy(unit).float(),
                    "norm": torch.from_numpy(norm), "layers": layers, "hidden": hidden,
                    "bank": str(a.vectors), "bank_tag": a.tag, "model": a.model, "recipe": RECIPE,
                    "roles": {n: meta[n] for n in pos + neg}, "created": stamp}, pt)
        side = {"name": key, "description": desc, "model": a.model, "bank": str(a.vectors), "bank_tag": a.tag,
                "recipe": RECIPE, "positive": list(pos), "negative": list(neg),
                "rollouts_per_role": {n: meta[n]["n"] for n in pos + neg},
                "input_md5": {n: meta[n]["md5"] for n in pos + neg},
                "layers": len(layers), "hidden": hidden,
                "norm_at": {f"L{l}": round(float(norm[layers.index(l)]), 6) for l in REPORT_LAYERS if l in layers},
                "pt_md5": md5(pt), "created": stamp,
                "use": "score = activation @ unit[layer]; the sign is fixed (positive = the first group: "
                       "deceptive, or sycophantic for D6)"}
        js.write_text(json.dumps(side, indent=2), encoding="utf-8")
        print(f"wrote {pt} ({pt.stat().st_size / 1e6:.1f} MB)")

    keys = list(results)
    pairs = [(x, y) for i, x in enumerate(keys) for y in keys[i + 1:]]
    print("\nlayer  " + "  ".join(f"|{k}|".ljust(8) for k in keys) + "  " + "  ".join(f"cos({x},{y})" for x, y in pairs))
    for l in REPORT_LAYERS:
        if l in layers:
            i = layers.index(l)
            print(f"L{l:<4} " + "  ".join(f"{results[k][4][i]:8.4f}" for k in keys) + "  "
                  + "  ".join(f"{cos(results[x][3][i], results[y][3][i]):+11.4f}" for x, y in pairs))
    print("rollouts per role: min", min(m["n"] for m in meta.values()), "max", max(m["n"] for m in meta.values()))


if __name__ == "__main__":
    main()
