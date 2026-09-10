#!/usr/bin/env python
"""Build the persona axes from per-role vectors, and validate them.

    src/assistant_axis_ndif/6_axes.py \
        --vectors_dir "$OUT_DIR/vectors" --output "$OUT_DIR/axes.pt"

Consumes <vectors_dir>/<role>.pt written by 2_activations_ndif.py:
    {"vector": (K, H) float32, "layers": [...], "type": "pos_3"|"mean", "role": str, "n": int}

Produces three per-layer directions, all shape (K, H):

  assistant   mean(default vectors) - mean(role vectors)            [Lu et al. 2026]
  deception   mean(deception group) - mean(honesty group)           [Eq 4, Mahadik & Skapars]
  sycophancy  mean(sycophancy group) - mean(anti_sycophancy group)  [Eq 4]

plus per-layer PCA over the centered role vectors, and the validation numbers that
say whether the geometry came out the way both papers report.

Note on centering: Eq (2) centers every persona vector by the mean across personas
before Eq (4) takes a difference of class means. That centering cancels exactly in a
difference of means, so it is applied only where it actually matters -- PCA.

Local CPU only; nothing here touches NDIF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))
from aa_roles import load_role_groups  # noqa: E402


def load_vectors(vectors_dir: Path):
    """role -> (K, H) float32, plus each file's metadata and the shared layer list."""
    vecs: dict[str, torch.Tensor] = {}
    meta: dict[str, dict] = {}
    layers: list[int] | None = None
    for f in sorted(vectors_dir.glob("*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        is_dict = isinstance(d, dict)
        v = d["vector"] if is_dict else d
        role = d.get("role", f.stem) if is_dict else f.stem
        this = list(d.get("layers", [])) if is_dict else []
        if layers is None and this:
            layers = this
        elif this and layers is not None and this != layers:
            raise SystemExit(
                f"{f.name}: layers {this[:4]}... differ from {layers[:4]}...; "
                "all roles must be extracted at the same layers"
            )
        vecs[role] = v.float()
        meta[role] = {k: d[k] for k in ("type", "n") if is_dict and k in d}
    if not vecs:
        raise SystemExit(f"no *.pt in {vectors_dir}")
    shapes = {tuple(v.shape) for v in vecs.values()}
    if len(shapes) != 1:
        raise SystemExit(f"inconsistent vector shapes: {shapes}")
    if layers is None:
        layers = list(range(next(iter(vecs.values())).shape[0]))
    return vecs, meta, layers


def class_contrast(vecs, pos: list[str], neg: list[str]):
    """mean(pos) - mean(neg), per layer. Returns (direction, n_pos, n_neg)."""
    p = [vecs[r] for r in pos if r in vecs]
    n = [vecs[r] for r in neg if r in vecs]
    if not p or not n:
        return None, len(p), len(n)
    return torch.stack(p).mean(0) - torch.stack(n).mean(0), len(p), len(n)


def pca_per_layer(X: torch.Tensor, k: int):
    """X: (n_roles, K, H), centered across roles. -> (K, k, H) components, (K, k) var ratio."""
    n_roles, K, _ = X.shape
    k = min(k, min(n_roles, X.shape[2]))
    comps, ratios = [], []
    for layer in range(K):
        _, S, Vh = torch.linalg.svd(X[:, layer, :], full_matrices=False)
        var = S ** 2 / max(n_roles - 1, 1)
        comps.append(Vh[:k])
        ratios.append((var / var.sum())[:k])
    return torch.stack(comps), torch.stack(ratios)


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def relative_position(x: float, values: list[float]) -> float:
    """Where x sits in [min, max]; the paper reports ~0.03 for the Assistant on PC1."""
    lo, hi = min(values), max(values)
    return 0.0 if hi == lo else (x - lo) / (hi - lo)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--vectors_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--roles_file", default=str(PIPELINE_DIR / "role_groups.json"))
    ap.add_argument("--pcs", type=int, default=5, help="principal components to keep")
    ap.add_argument("--report_layer", type=int, default=None,
                    help="layer to print the report at (default: middle of what was extracted)")
    args = ap.parse_args()

    vecs, meta, layers = load_vectors(Path(args.vectors_dir))
    groups = load_role_groups(args.roles_file)["groups"]
    K, H = next(iter(vecs.values())).shape
    print(f"{len(vecs)} role vectors, shape ({K}, {H}), layers {layers[0]}..{layers[-1]}")

    for g, spec in groups.items():
        absent = [r for r in spec["roles"] if r not in vecs]
        if absent:
            print(f"  ! {g}: {len(absent)} role(s) absent (min_count drop, or not yet run): "
                  f"{', '.join(absent[:8])}{' ...' if len(absent) > 8 else ''}")

    # --- the Assistant Axis: default anchor vs. every role-playing role ---------------
    anchors = [r for r in groups["assistant_anchor"]["roles"] if r in vecs]
    default_like = [r for r in anchors
                    if "default" in r or meta.get(r, {}).get("type") == "mean"]
    if not default_like:
        raise SystemExit("no 'default' role vector; the Assistant Axis needs the default anchor")
    role_side = [r for r in vecs if r not in default_like]
    assistant_axis = (torch.stack([vecs[r] for r in default_like]).mean(0)
                      - torch.stack([vecs[r] for r in role_side]).mean(0))

    axes = {"assistant": assistant_axis}
    counts = {"assistant": (len(default_like), len(role_side))}

    # --- behaviour contrasts (Eq 4) --------------------------------------------------
    for pos, neg, name in (("deception", "honesty", "deception"),
                           ("sycophancy", "anti_sycophancy", "sycophancy")):
        d, npos, nneg = class_contrast(vecs, groups[pos]["roles"], groups[neg]["roles"])
        if d is None:
            print(f"  ! {name} axis skipped: {npos} {pos} / {nneg} {neg} vectors")
            continue
        axes[name] = d
        counts[name] = (npos, nneg)

    # --- PCA over centered role vectors (Eq 2, then PCA) -----------------------------
    names = sorted(vecs)
    X = torch.stack([vecs[r] for r in names])
    comps, ratios = pca_per_layer(X - X.mean(0, keepdim=True), args.pcs)

    # --- report ----------------------------------------------------------------------
    target = args.report_layer if args.report_layer is not None else layers[len(layers) // 2]
    li = layers.index(target) if target in layers else len(layers) // 2
    print(f"\n=== report at layer {layers[li]} ===")
    for name, ax in axes.items():
        npos, nneg = counts[name]
        print(f"  {name:11} norm = {ax[li].norm():8.3f}   ({npos} vs {nneg} roles)")

    pc1 = comps[:, 0, :]
    here = cos(pc1[li], axes["assistant"][li]).abs().item()
    across = cos(pc1, axes["assistant"]).abs()
    print(f"\n  cos(PC1, assistant axis) = {here:.3f} at layer {layers[li]}"
          "   [paper: > 0.71 mid-layer]")
    print(f"  across layers: min {across.min():.3f}, median {across.median():.3f}"
          "   [paper: > 0.60 all layers]")
    print(f"  PC1 explains {ratios[li, 0]:.1%} of role variance; "
          f"top-{ratios.shape[1]} {ratios[li].sum():.1%}")

    proj = {r: (vecs[r][li] @ pc1[li]).item() for r in names}
    dvals = [proj[r] for r in default_like]
    rel = relative_position(sum(dvals) / len(dvals), list(proj.values()))
    print(f"  default anchor at relative position {rel:.3f} on PC1"
          "   [paper: ~0.03 or ~0.97 -- an extreme]")

    print("\n  mean projection on each axis, by group:")
    print("    " + f"{'group':18}" + "".join(f"{n:>13}" for n in axes))
    for g, spec in groups.items():
        rs = [r for r in spec["roles"] if r in vecs]
        if not rs:
            continue
        row = f"    {g:18}"
        centroid = torch.stack([vecs[r][li] for r in rs]).mean(0)
        for ax in axes.values():
            unit = ax[li] / (ax[li].norm() + 1e-8)
            row += f"{centroid @ unit:13.2f}"
        print(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"axes": axes, "counts": counts, "layers": layers,
                "pca_components": comps, "pca_variance_ratio": ratios,
                "roles": names, "role_groups": groups,
                "vectors_dir": str(Path(args.vectors_dir).resolve())}, out)
    print(f"\nwrote {out}  ({', '.join(axes)}; PCA {tuple(comps.shape)})")


if __name__ == "__main__":
    main()
