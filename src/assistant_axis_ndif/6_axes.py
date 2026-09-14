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


def orient_pcs(comps: torch.Tensor, reference: torch.Tensor):
    """Pin each PC's sign per layer. -> (comps, signs (K, k)).

    torch.linalg.svd picks an arbitrary sign, independently at every layer. Left alone,
    a layer sweep of <z, u_k> comes out as a sawtooth and AUROC flips to 1 - AUROC at
    random layers. Orient each component so it points the same way as `reference`
    (that behaviour's own contrast direction); for a component near-orthogonal to the
    reference, fall back to "largest-magnitude coordinate is positive" so the choice is
    still deterministic across rebuilds.
    """
    K, k, _ = comps.shape
    signs = torch.ones(K, k)
    for layer in range(K):
        for j in range(k):
            u = comps[layer, j]
            c = cos(u, reference[layer]).item()
            if abs(c) < 1e-3:
                idx = int(u.abs().argmax())
                s = 1.0 if u[idx] >= 0 else -1.0
            else:
                s = 1.0 if c > 0 else -1.0
            signs[layer, j] = s
            comps[layer, j] = u * s
    return comps, signs


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

    # the paper's unified harmful axis (Appendix C): both harmful groups vs both harmless
    harmful = groups["deception"]["roles"] + groups["sycophancy"]["roles"]
    harmless = groups["honesty"]["roles"] + groups["anti_sycophancy"]["roles"]
    d, npos, nneg = class_contrast(vecs, harmful, harmless)
    if d is not None:
        axes["unified"] = d
        counts["unified"] = (npos, nneg)

    # --- PCA: one basis per behaviour, plus the all-roles basis (Eq 2, then Eq 3) -----
    # Mahadik & Skapars run PCA over the behaviour-specific persona set ONLY, so their PC1
    # is a behaviour axis by construction. PCA over every role instead recovers the
    # Assistant Axis -- a different object (measured cos(PC1, assistant) = 0.77-0.82 on
    # both our vectors and the published HF ones). Build both, labelled for what they are.
    names = sorted(vecs)
    pcas: dict[str, dict] = {}

    def build_pca(tag: str, subset: list[str], reference: torch.Tensor, ref_name: str):
        present = sorted({r for r in subset if r in vecs})
        if len(present) < 3:
            print(f"  ! pca[{tag}] skipped: only {len(present)} vector(s)")
            return
        Xs = torch.stack([vecs[r] for r in present])
        c, rt = pca_per_layer(Xs - Xs.mean(0, keepdim=True), args.pcs)
        c, sg = orient_pcs(c, reference)
        pcas[tag] = {"components": c, "variance_ratio": rt, "sign": sg,
                     "roles": present, "reference": ref_name}

    if "deception" in axes:
        build_pca("deception",
                  groups["deception"]["roles"] + groups["honesty"]["roles"] + default_like,
                  axes["deception"], "deception contrast")
    if "sycophancy" in axes:
        build_pca("sycophancy",
                  groups["sycophancy"]["roles"] + groups["anti_sycophancy"]["roles"] + default_like,
                  axes["sycophancy"], "sycophancy contrast")
    if "unified" in axes:
        build_pca("unified",
                  groups["deception"]["roles"] + groups["honesty"]["roles"]
                  + groups["sycophancy"]["roles"] + groups["anti_sycophancy"]["roles"] + default_like,
                  axes["unified"], "unified contrast")
    build_pca("all", names, assistant_axis, "assistant axis")

    # kept at the top level under the original names for backward compatibility: this is
    # the ASSISTANT-AXIS basis, not a behaviour one.
    comps = pcas["all"]["components"]
    ratios = pcas["all"]["variance_ratio"]

    # --- report ----------------------------------------------------------------------
    target = args.report_layer if args.report_layer is not None else layers[len(layers) // 2]
    li = layers.index(target) if target in layers else len(layers) // 2
    print(f"\n=== report at layer {layers[li]} ===")
    for name, ax in axes.items():
        npos, nneg = counts[name]
        print(f"  {name:11} norm = {ax[li].norm():8.3f}   ({npos} vs {nneg} roles)")

    # --- all-roles PCA: this is the Assistant Axis replication (Lu et al. 2026) -------
    pc1 = comps[:, 0, :]
    here = cos(pc1[li], axes["assistant"][li]).item()
    across = cos(pc1, axes["assistant"])
    print()
    print(f"  [all-roles PCA = Assistant Axis basis, n={len(pcas['all']['roles'])}]")
    print(f"  cos(PC1, assistant axis) = {here:+.3f} at layer {layers[li]}"
          "   [Assistant Axis paper: > 0.71 mid-layer]")
    print(f"  across layers: min {across.min():+.3f}, median {across.median():+.3f}"
          "   [Assistant Axis paper: > 0.60 all layers]")
    print(f"  PC1 explains {ratios[li, 0]:.1%} of role variance; "
          f"top-{ratios.shape[1]} {ratios[li].sum():.1%}")

    # --- per-behaviour PCA: the basis Mahadik & Skapars actually use ------------------
    # Signs are pinned, so these cosines are signed and comparable across layers.
    print()
    print("  [per-behaviour PCA -- the persona-coordinates basis]")
    print("    " + f"{'basis':12}{'n':>4}{'cos(PC1,contrast)':>20}"
          f"{'cos(PC1,assistant)':>21}{'PC1 var':>10}")
    for tag in ("deception", "sycophancy", "unified"):
        if tag not in pcas:
            continue
        pk = pcas[tag]
        b1 = pk["components"][:, 0, :]
        print("    " + f"{tag:12}{len(pk['roles']):>4}"
              f"{cos(b1[li], axes[tag][li]).item():>20.3f}"
              f"{cos(b1[li], axes['assistant'][li]).item():>21.3f}"
              f"{pk['variance_ratio'][li, 0]:>10.1%}")
    print("    a behaviour PC1 near +1.0 on its own contrast is the expected geometry;")
    print("    a large |cos| with the assistant axis is inherited from the contrast itself.")

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
                # backward-compatible aliases: the ALL-ROLES (Assistant Axis) basis
                "pca_components": comps, "pca_variance_ratio": ratios,
                # per-behaviour bases, each sign-pinned against its own contrast
                "pca": pcas,
                "roles": names, "role_groups": groups,
                "vectors_dir": str(Path(args.vectors_dir).resolve())}, out)
    print()
    print(f"wrote {out}  (axes: {', '.join(axes)}; "
          f"pca bases: {', '.join(pcas)}; PCA {tuple(comps.shape)})")


if __name__ == "__main__":
    main()
