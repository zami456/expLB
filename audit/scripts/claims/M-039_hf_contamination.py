"""W6F2-memory claim M-039 (memory/persona-axis-diagnostics.md:43-47).

Claim: rebuilding the SAME 22-vs-22 deception contrast from the HuggingFace
`lu-christina/assistant-axis-vectors` llama-3.3-70b role vectors reproduces the
Assistant-Axis contamination: cos(deception, assistant) = -0.686 / -0.662 / -0.610 / -0.516
at layers 20 / 22 / 40 / 60 (ours: -0.729 / -0.710 / -0.660 / -0.573), and
cos(PC1, assistant) = 0.79-0.82 on HF too.

Raw file used: run-hf275/axes_hf100.pt  (laptop working tree, 62,938,229 B, mtime 2026-09-17 22:34)
Read-only.
"""
import torch, os, hashlib

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
P = os.path.join(ROOT, "run-hf275", "axes_hf100.pt")
print("file:", os.path.relpath(P, ROOT), os.path.getsize(P), "B")
h = hashlib.md5(open(P, 'rb').read()).hexdigest()
print("md5:", h)

b = torch.load(P, map_location="cpu", weights_only=False)
print("vectors_dir:", b.get("vectors_dir"))
print("counts:", {k: tuple(v) for k, v in b["counts"].items()})
print("n roles:", len(b["roles"]))
print("role_groups keys:", list(b["role_groups"].keys()))
for g, rs in b["role_groups"].items():
    print("   %-12s n=%d %s" % (g, len(rs), sorted(rs)[:6]))

layers = b["layers"]


def cos(a, c):
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), c.unsqueeze(0)).item())


print("\nlayer  cos(dec,asst)  cos(syc,asst)  cos(dec,syc)  cos(PC1_all,asst)  PC1_all var")
for L in (20, 22, 40, 60):
    i = layers.index(L)
    A = b["axes"]["assistant"][i]
    D = b["axes"]["deception"][i]
    S = b["axes"]["sycophancy"][i]
    pc1_all = b["pca"]["all"]["components"][i][0]
    var = float(b["pca"]["all"]["variance_ratio"][i][0])
    print("  %3d   %+.4f        %+.4f        %+.4f       %+.4f            %.3f"
          % (L, cos(D, A), cos(S, A), cos(D, S), cos(pc1_all, A), var))

print("\nsame with the flat top-level pca_components (the pre-patch 'PC1 over all roles'):")
for L in (20, 22, 40, 60):
    i = layers.index(L)
    A = b["axes"]["assistant"][i]
    pc1 = b["pca_components"][i][0]
    print("  L%-3d cos(PC1,asst)=%+.4f  |cos|=%.4f  var=%.3f"
          % (L, cos(pc1, A), abs(cos(pc1, A)), float(b["pca_variance_ratio"][i][0])))

print("\ncos(PC1_deception_subset, deception contrast) -- the 'per-behaviour PCA fix':")
for L in (20, 22, 40, 60):
    i = layers.index(L)
    print("  L%-3d %+.4f" % (L, cos(b["pca"]["deception"]["components"][i][0], b["axes"]["deception"][i])))
