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
    doc = load_role_groups(args.roles_file)
    groups = doc["groups"]
    # A bank may declare its own contrasts, so a new group pair needs no code change here.
    # kd/personas/kd_role_groups.json uses this for knowledge_denial (feigned vs sincere
    # ignorance -- both sides deny, so the axis cannot be a wording artifact) and
    # kd_assertion (feigned vs plain assertion). Banks without the key keep the two
    # hardcoded pairs below, so paper16 / own100 / hf275 are unaffected.
    declared = doc.get("contrasts") or []
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
    # The four paper16 group names are OPTIONAL, not assumed. Every bank until 2026-09-21 was
    # paper16 or a superset of it (kd = paper16 + 10 roles), so indexing groups["deception"]
    # directly always worked. The cap bank is the first with entirely new group names --
    # capitulation / position_holding / assistant_anchor -- and it died here with
    # KeyError: 'deception' after the whole 1,300-rollout generation had succeeded.
    for pos, neg, name in (("deception", "honesty", "deception"),
                           ("sycophancy", "anti_sycophancy", "sycophancy")):
        if pos not in groups or neg not in groups:
            print(f"  - {name} axis not applicable: this bank has no {pos}/{neg} groups")
            continue
        d, npos, nneg = class_contrast(vecs, groups[pos]["roles"], groups[neg]["roles"])
        if d is None:
            print(f"  ! {name} axis skipped: {npos} {pos} / {nneg} {neg} vectors")
            continue
        axes[name] = d
        counts[name] = (npos, nneg)

    # the paper's unified harmful axis (Appendix C): both harmful groups vs both harmless
    if all(g in groups for g in ("deception", "sycophancy", "honesty", "anti_sycophancy")):
        harmful = groups["deception"]["roles"] + groups["sycophancy"]["roles"]
        harmless = groups["honesty"]["roles"] + groups["anti_sycophancy"]["roles"]
        d, npos, nneg = class_contrast(vecs, harmful, harmless)
        if d is not None:
            axes["unified"] = d
            counts["unified"] = (npos, nneg)
    else:
        print("  - unified axis not applicable: this bank has no paper16 behaviour groups")

    # --- contrasts the bank declares for itself --------------------------------------
    for spec in declared:
        name = spec["name"]
        if name in axes:
            continue                      # the hardcoded pairs above already built it
        pos = [r for g in spec["pos"] for r in groups.get(g, {}).get("roles", [])]
        neg = [r for g in spec["neg"] for r in groups.get(g, {}).get("roles", [])]
        d, npos, nneg = class_contrast(vecs, pos, neg)
        if d is None:
            print(f"  ! {name} skipped: {npos} pos / {nneg} neg vectors present")
            continue
        axes[name] = d
        counts[name] = (npos, nneg)
        print(f"  + declared contrast {name}: {'+'.join(spec['pos'])} - {'+'.join(spec['neg'])} "
              f"({npos} vs {nneg} roles)")

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
    # a declared contrast gets its own PCA basis too, over exactly the roles it contrasts
    # plus the anchor -- the same construction the behaviour bases use above.
    #
    # OPTIONAL "basis" KEY. A spec may name the groups the PCA runs over, independently of the
    # groups it contrasts. Without it the basis is pos+neg+anchor, which ties the two together
    # and makes one question unaskable: does a group contribute to the BASIS even when it is
    # not in the contrast? The cap bank needs exactly that -- F-55 found its position_holding
    # personas behave identically to no system prompt at all, so three variants are run:
    #   A  pos=capitulation  neg=position_holding                      basis = default (13 roles)
    #   B  pos=capitulation  neg=assistant_anchor  basis=all three     (13 roles)
    #   C  pos=capitulation  neg=assistant_anchor                      basis = default (7 roles)
    # A vs B isolates the contrast; B vs C isolates the basis. Banks without the key are
    # unaffected.
    for spec in declared:
        name = spec["name"]
        if name in pcas or name not in axes:
            continue
        basis_groups = spec.get("basis") or (list(spec["pos"]) + list(spec["neg"]))
        subset = [r for g in basis_groups
                  for r in groups.get(g, {}).get("roles", [])] + default_like
        build_pca(name, subset, axes[name], f"{name} contrast")

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
    # EVERY basis that has a matching axis, not a hardcoded paper16 triple. The literal
    # ("deception", "sycophancy", "unified") printed an EMPTY table for the cap bank on
    # 2026-09-21 -- and because the chain's 2e gate only checked that the .pt file existed,
    # it logged "written / written / PASS" while the gate it was supposed to enforce
    # (cos(PC1,contrast) >= 0.90) had never been computed at all. Sixth hardcoding of the
    # same kind found in this file family.
    for tag in [t for t in pcas if t != "all" and t in axes]:
        pk = pcas[tag]
        b1 = pk["components"][:, 0, :]
        print("    " + f"{tag:12}{len(pk['roles']):>4}"
              f"{cos(b1[li], axes[tag][li]).item():>20.3f}"
              f"{cos(b1[li], axes['assistant'][li]).item():>21.3f}"
              f"{pk['variance_ratio'][li, 0]:>10.1%}")
    print("    a behaviour PC1 near +1.0 on its own contrast is the expected geometry;")
    print("    a large |cos| with the assistant axis is inherited from the contrast itself.")

    # TWO BASES OVER THE SAME ROLES ARE THE SAME SUBSPACE, whatever they are named or
    # contrasted against -- build_pca only ever sees the role set. The per-PC sign differs
    # (orient_pcs pins against each basis's own contrast) but a linear probe is invariant to
    # a per-feature sign, so `pc:<a>` and `pc:<b>` score IDENTICALLY in 9_persona_probes.py.
    # Printed because the cap bank's A/B variants are exactly this case, and their identical
    # crossgen AUROCs read like a finding until you notice they cannot differ.
    _seen: dict = {}
    for tag in [t for t in pcas if t in axes or t == "all"]:
        _seen.setdefault(tuple(sorted(pcas[tag]["roles"])), []).append(tag)
    for _roles, _tags in _seen.items():
        if len(_tags) > 1:
            print(f"    NOTE bases {' == '.join(_tags)} share {len(_roles)} roles -- same "
                  "subspace, so any probe on them is identical by construction")

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
