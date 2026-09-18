#!/usr/bin/env python
"""Bank variant: the unified axis with the QUESTION-FAMILY difference removed.

In run-paper16 the deception/honesty personas (and the default) answered the 10 deception
questions, while the sycophancy/anti-sycophancy personas answered the 12 sycophancy questions.
The unified-axis PCA therefore puts the question topic first: at layer 40, unified PC1 carries
47% of the variance and has cos -0.98 with the question-family direction, and only PC2 is the
harmful-vs-harmless direction (cos 0.88).

Approximate fix, CPU only: subtract each question family's mean persona vector before building
the unified contrast and PCA. Each family contains both harmful and harmless personas, so the
harm contrast survives; the family shift does not. This removes the average topic shift only,
not persona-specific topic effects -- the faithful fix is run-paper16-shared/, where every
persona answers the same questions.

Only axes["unified"] and pca["unified"] change. The deception and sycophancy objects were each
built within a single question family and are copied unchanged.

    python 29_bank_variants.py --bank run-paper16/axes_vectors_s3_fp32.pt \
        --vectors_dir run-paper16/vectors_s3_fp32 \
        --out run-paper16/axes_vectors_s3_fp32_famcentred.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

PIPELINE_DIR = Path(__file__).resolve().parent


def module(name, fname):
    spec = importlib.util.spec_from_file_location(name, PIPELINE_DIR / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--vectors_dir", required=True)
    ap.add_argument("--roles_file", default=str(PIPELINE_DIR / "pc_role_groups.json"))
    ap.add_argument("--family_a", nargs="+", default=["deception", "honesty", "assistant_anchor"],
                    help="groups that answered the first question set (default included)")
    ap.add_argument("--family_b", nargs="+", default=["sycophancy", "anti_sycophancy"])
    ap.add_argument("--pcs", type=int, default=5)
    ap.add_argument("--report_layer", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A6 = module("axes6", "6_axes.py")
    if Path(args.out).resolve() == Path(args.bank).resolve():
        raise SystemExit("refusing to overwrite the input bank")
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    vecs, _, layers = A6.load_vectors(Path(args.vectors_dir))
    groups = json.loads(Path(args.roles_file).read_text(encoding="utf-8"))["groups"]

    fam = {}
    for tag, gs in (("a", args.family_a), ("b", args.family_b)):
        for g in gs:
            for r in groups[g]["roles"]:
                if r in vecs:
                    fam[r] = tag
    means = {t: torch.stack([vecs[r] for r, f in fam.items() if f == t]).mean(0) for t in ("a", "b")}
    centred = {r: vecs[r] - means[f] for r, f in fam.items()}

    harmful = [r for r in groups["deception"]["roles"] + groups["sycophancy"]["roles"] if r in centred]
    harmless = [r for r in groups["honesty"]["roles"] + groups["anti_sycophancy"]["roles"] if r in centred]
    d = torch.stack([centred[r] for r in harmful]).mean(0) - torch.stack([centred[r] for r in harmless]).mean(0)
    subset = sorted(set(harmful + harmless + [r for r in groups["assistant_anchor"]["roles"] if r in centred]))
    Xs = torch.stack([centred[r] for r in subset])
    comps, ratios = A6.pca_per_layer(Xs - Xs.mean(0, keepdim=True), args.pcs)
    comps, signs = A6.orient_pcs(comps, d)

    li = layers.index(args.report_layer)
    harm_raw = (torch.stack([vecs[r] for r in harmful]).mean(0)
                - torch.stack([vecs[r] for r in harmless]).mean(0))[li]
    family_dir = means["a"][li] - means["b"][li]
    old = bank["pca"]["unified"]["components"][li]
    print(f"layer {args.report_layer}: {len(harmful)} harmful vs {len(harmless)} harmless, "
          f"{len(subset)} personas in the PCA")
    print(f"  {'':10}{'var':>7}{'cos harm':>10}{'cos family':>12}")
    for tag, C, R in (("old", old, bank["pca"]["unified"]["variance_ratio"][li]),
                      ("centred", comps[li], ratios[li])):
        for j in range(min(3, C.shape[0])):
            print(f"  {tag} PC{j + 1:<3}{float(R[j]):7.3f}{A6.cos(C[j], harm_raw).item():10.3f}"
                  f"{A6.cos(C[j], family_dir).item():12.3f}")

    out = dict(bank)
    out["axes"] = dict(bank["axes"])
    out["pca"] = dict(bank["pca"])
    out["axes"]["unified"] = d
    out["pca"]["unified"] = {"components": comps, "variance_ratio": ratios, "sign": signs,
                             "roles": subset, "reference": "unified contrast (question-family centred)"}
    out["variant"] = ("unified axis and PCA rebuilt after subtracting each question family's mean "
                      f"persona vector; families a={args.family_a}, b={args.family_b}; "
                      f"source bank {Path(args.bank).name}")
    torch.save(out, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
