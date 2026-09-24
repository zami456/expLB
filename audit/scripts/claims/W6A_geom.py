"""W6A-thesis1 -- geometry claims (O-057, O-064, O-067, O-068, O-069).

Read-only. Recomputes persona-bank cosines from the vector stores that are
available on the laptop:

  run-paper16/axes_vectors_s3.pt          the OLD paper16 bank (RF-15 "old")
  run-paper16/vectors_s3/*.pt             the OLD per-persona vectors
  audit/cache/labpc/run-paper16/vectors_s3_fp32/*.pt   the fp32 re-extraction
  run-hf275/axes_hf100.pt                 the hf275/hf100 bank (our grouping)
  run-hf275/published_assistant_axis.pt   Lu et al.'s published axis
  run-coords/regroup_hf.json              the disclosure-stance regrouping

own100 vectors are lab-only; that column cannot be recomputed here.

Usage: python audit/scripts/claims/W6A_geom.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
LAYERS = [20, 22, 40, 60]


def cos(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1).item()


def load_dir(d: pathlib.Path):
    """Replicates 6_axes.py load_vectors: role -> (80, 8192) float32."""
    vecs = {}
    for f in sorted(d.glob("*.pt")):
        obj = torch.load(f, map_location="cpu", weights_only=False)
        role = obj.get("role", f.stem) if isinstance(obj, dict) else f.stem
        v = obj["vector"] if isinstance(obj, dict) else obj
        vecs[role] = v.float()
    return vecs


def build_axes(vecs, groups):
    """Replicates 6_axes.py: assistant = default - mean(all other roles);
    contrast = mean(pos) - mean(neg)."""
    default_like = [r for r in vecs if "default" in r]
    role_side = [r for r in vecs if r not in default_like]
    asst = (torch.stack([vecs[r] for r in default_like]).mean(0)
            - torch.stack([vecs[r] for r in role_side]).mean(0))
    out = {"assistant": asst}
    for name, (pos, neg) in groups.items():
        p = [vecs[r] for r in pos if r in vecs]
        n = [vecs[r] for r in neg if r in vecs]
        out[name] = torch.stack(p).mean(0) - torch.stack(n).mean(0)
        out[name + "__n"] = (len(p), len(n))
    return out


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- load banks
p16_old = torch.load(ROOT / "run-paper16/axes_vectors_s3.pt",
                     map_location="cpu", weights_only=False)
hf = torch.load(ROOT / "run-hf275/axes_hf100.pt",
                map_location="cpu", weights_only=False)
pub = torch.load(ROOT / "run-hf275/published_assistant_axis.pt",
                 map_location="cpu", weights_only=False)

hdr("PROVENANCE")
print("paper16 bank  :", ROOT / "run-paper16/axes_vectors_s3.pt",
      "| vectors_dir =", p16_old["vectors_dir"], "| counts =", p16_old["counts"])
print("hf bank       :", ROOT / "run-hf275/axes_hf100.pt",
      "| vectors_dir =", hf["vectors_dir"], "| counts =", hf["counts"])
print("published axis:", ROOT / "run-hf275/published_assistant_axis.pt",
      "| shape", tuple(pub.shape))

# --------------------------------------------------- O-057 / Q1 contamination
hdr("O-057  cos(deception contrast, assistant direction) by layer")
v100 = load_dir(ROOT / "run-hf275/vectors_100")
v275 = load_dir(ROOT / "run-hf275/vectors_275")
dec_r = hf["role_groups"]["deception"]["roles"]
hon_r = hf["role_groups"]["honesty"]["roles"]
d_hf = (torch.stack([v275[r] for r in dec_r]).mean(0)
        - torch.stack([v275[r] for r in hon_r]).mean(0))
dflt = [r for r in v275 if "default" in r]
a275 = (torch.stack([v275[r] for r in dflt]).mean(0)
        - torch.stack([v275[r] for r in v275 if r not in dflt]).mean(0))
print("n roles: vectors_100 = %d, vectors_275 = %d; the 100 are byte-equal to their "
      "namesakes in the 275: %s"
      % (len(v100), len(v275),
         all(torch.allclose(v100[r], v275[r]) for r in v100 if r in v275)))
print("cos(assistant built over all 275 roles, PUBLISHED axis) = %.5f (L20), %.5f (L40)"
      % (cos(a275[20], pub[20]), cos(a275[40], pub[40])))
print("cos(assistant built over the 100-role subset, PUBLISHED axis) = %.4f (L20), "
      "%.4f (L40)" % (cos(hf["axes"]["assistant"][20], pub[20]),
                      cos(hf["axes"]["assistant"][40], pub[40])))
print()
print(f"{'layer':>6} {'paper16_s3':>12} {'hf: 22v22 vs PUBLISHED':>23} "
      f"{'hf: 22v22 vs asst(275)':>23} {'hf: 22v22 vs asst(100)':>23} "
      f"{'paper16_s3 vs PUBLISHED':>24}")
for L in LAYERS:
    a = cos(p16_old["axes"]["deception"][L], p16_old["axes"]["assistant"][L])
    b = cos(d_hf[L], pub[L])
    b2 = cos(d_hf[L], a275[L])
    b3 = cos(hf["axes"]["deception"][L], hf["axes"]["assistant"][L])
    c = cos(p16_old["axes"]["deception"][L], pub[L])
    print(f"{L:>6} {a:>12.4f} {b:>23.4f} {b2:>23.4f} {b3:>23.4f} {c:>24.4f}")
print("DOC's hf275 column is -0.686 / -0.662 / -0.610 / -0.516 -> matches the "
      "'vs PUBLISHED' variant to 4 dp.")
print("own100: NOT AVAILABLE on the laptop (vectors are lab-only) -> DEFERRED")
print("\nshared variance: cos^2 at L40  paper16 %.4f  published-ref %.4f"
      % (cos(p16_old["axes"]["deception"][40], p16_old["axes"]["assistant"][40]) ** 2,
         cos(p16_old["axes"]["deception"][40], pub[40]) ** 2))

# ------------------------------------------ O-064 old vs fp32 (the count bug)
hdr("O-064  old (bf16-count) vs fp32 re-extraction -- effect on bank cosines")
fp32_dir = ROOT / "audit/cache/labpc/run-paper16/vectors_s3_fp32"
old_dir = ROOT / "run-paper16/vectors_s3"
if fp32_dir.exists() and old_dir.exists():
    groups = {k: (p16_old["role_groups"]["deception"]["roles"],
                  p16_old["role_groups"]["honesty"]["roles"])
              for k in ["deception"]}
    groups["sycophancy"] = (p16_old["role_groups"]["sycophancy"]["roles"],
                            p16_old["role_groups"]["anti_sycophancy"]["roles"])
    v_old = load_dir(old_dir)
    v_new = load_dir(fp32_dir)
    print("roles old =", len(v_old), " roles fp32 =", len(v_new),
          " same names:", sorted(v_old) == sorted(v_new))
    ax_old = build_axes(v_old, groups)
    ax_new = build_axes(v_new, groups)
    print("rebuilt-from-old vs stored bank, cos(dec,asst) L40: %.6f vs %.6f (sanity)"
          % (cos(ax_old["deception"][40], ax_old["assistant"][40]),
             cos(p16_old["axes"]["deception"][40], p16_old["axes"]["assistant"][40])))
    print(f"\n{'layer':>6} {'cos(dec,asst) OLD':>18} {'cos(dec,asst) fp32':>19} "
          f"{'|delta|':>9} {'cos(ax_old,ax_new) dec':>23}")
    worst = 0.0
    for L in range(0, 80, 4):
        a = cos(ax_old["deception"][L], ax_old["assistant"][L])
        b = cos(ax_new["deception"][L], ax_new["assistant"][L])
        aa = cos(ax_old["deception"][L], ax_new["deception"][L])
        worst = max(worst, abs(a - b))
        if L in (0, 20, 22, 40, 44, 60, 76):
            print(f"{L:>6} {a:>18.5f} {b:>19.5f} {abs(a-b):>9.5f} {aa:>23.6f}")
    # full sweep
    deltas = []
    for L in range(80):
        a = cos(ax_old["deception"][L], ax_old["assistant"][L])
        b = cos(ax_new["deception"][L], ax_new["assistant"][L])
        deltas.append(abs(a - b))
    print("\nmax |delta cos(dec,asst)| over all 80 layers = %.6f" % max(deltas))
    for nm in ("deception", "sycophancy", "assistant"):
        cs = [cos(ax_old[nm][L], ax_new[nm][L]) for L in range(80)]
        nr = [abs(ax_new[nm][L].norm().item() / ax_old[nm][L].norm().item() - 1)
              for L in range(80)]
        print(f"axis {nm:>10}: min cos(old,new) over layers = {min(cs):.6f}; "
              f"max relative norm change = {max(nr)*100:.3f}%")
    # per-persona vector n fields (the count the bug corrupted)
    n_old = {r: torch.load(old_dir / f"{r}.pt", map_location="cpu",
                           weights_only=False).get("n") for r in ["pc_spin_doctor"]}
    print("per-file 'n' spot check (pc_spin_doctor): old", n_old)
else:
    print("MISSING vector dirs -> cannot run")

# ------------------------------------------------------ O-067 PC1 vs assistant
hdr("O-067  cos(PC1, assistant) and cos(PC1_deception, deception contrast)")
print("hf bank pca keys:", list(hf["pca"].keys()))
allc = hf["pca"]["all"]["components"]


def pc1_over(V, L):
    names = sorted(V)
    M = torch.stack([V[r][L] for r in names])
    M = M - M.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return Vh[0]


print("\nBROAD-LIBRARY PC1: |cos(PC1, assistant reference)| by layer")
print(f"{'layer':>6} {'PC1(100) vs asst(100)':>22} {'PC1(100) vs PUBLISHED':>22} "
      f"{'PC1(275) vs PUBLISHED':>22} {'PC1(100) vs dec contrast':>25}")
for L in LAYERS:
    p1 = allc[L, 0]
    p2 = pc1_over(v275, L)
    print(f"{L:>6} {abs(cos(p1, hf['axes']['assistant'][L])):>22.4f} "
          f"{abs(cos(p1, pub[L])):>22.4f} {abs(cos(p2, pub[L])):>22.4f} "
          f"{cos(p1, hf['axes']['deception'][L]):>25.4f}")
print("DOC says hf275 range 0.75-0.80 -> matches the 'PC1(100) vs PUBLISHED' column.")
band = list(range(20, 61))
for tag, V, ref in (("PC1(100) vs asst(100)", v100, hf["axes"]["assistant"]),
                    ("PC1(100) vs PUBLISHED", v100, pub),
                    ("PC1(275) vs PUBLISHED", v275, pub)):
    vals = [abs(cos(pc1_over(V, L), ref[L])) for L in band]
    print(f"   whole band L20..L60 ({len(band)} layers) {tag}: "
          f"min {min(vals):.4f} max {max(vals):.4f}")
print("\ntop-level legacy pca_components == pca['all']:",
      bool(torch.allclose(hf["pca_components"], allc)))
print("\nPER-BEHAVIOUR PCA, deception:")
dc = hf["pca"]["deception"]["components"]
for L in LAYERS:
    print(f"  L{L:<3} cos(PC1_deception, deception contrast) = "
          f"{cos(dc[L,0], hf['axes']['deception'][L]): .4f}"
          f"   var ratio {hf['pca']['deception']['variance_ratio'][L,0]:.3f}")
print("\nown100 column of this claim: lab-only vectors -> DEFERRED")

# ------------------------------------------------------------- O-068 regroup
hdr("O-068  regrouping the cached HF roles by disclosure stance")
rg = json.loads((ROOT / "run-coords/regroup_hf.json").read_text(encoding="utf-8"))
print("n conceal =", len(rg["conceal"]), " n disclose =", len(rg["disclose"]),
      " (n_disclosure field in results:", rg["results"][0]["n_disclosure"], ")")
ours = [abs(r["ours"]) for r in rg["results"]]
disc = [abs(r["disclosure"]) for r in rg["results"]]
for r in rg["results"]:
    print(f"  L{r['layer']:<3} ours {r['ours']: .4f}   disclosure {r['disclosure']: .4f}")
mo, md = sum(ours) / len(ours), sum(disc) / len(disc)
print(f"mean |cos| ours = {mo:.4f}   disclosure = {md:.4f}   "
      f"reduction = {(mo-md)/mo*100:.2f}%  (layers {[r['layer'] for r in rg['results']]})")

# -------------------------------------------------------------- O-069 traits
hdr("O-069  trait vectors vs role contrast at layer 40")
print("role contrast, hf275 bank, L40: cos(deception, assistant) = %.4f"
      % cos(hf["axes"]["deception"][40], hf["axes"]["assistant"][40]))
print("hf275 bank-local variant (asst over 275 roles):  %.4f"
      % cos(d_hf[40], a275[40]))
print("hf275 vs published axis:                         %.4f" % cos(d_hf[40], pub[40]))
print("assistant_axis/data/traits holds only prompt INSTRUCTIONS:",
      sorted(p.name for p in (ROOT / "assistant_axis/data/traits").iterdir())
      if (ROOT / "assistant_axis/data/traits").exists() else "absent")
print("-> the -0.725 trait-vector number has NO stored artefact locally or in the "
      "lab census (checked audit/phase1/labpc/joined_labpc.tsv.gz for *trait*.pt "
      "and hf_axis_check*: 0 hits).")

print("\nDONE")
