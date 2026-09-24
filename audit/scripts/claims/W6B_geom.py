"""W6B-thesis2 : O-080.

Claim (docs/thesis/persona-coordinates-70b-results.md:677-679, the audit's own
caveat box under 7.4):

  "the erased direction is paper16's own assistant direction, which has only
   cos 0.30-0.37 with the published Assistant Axis."

The Q4 erasure (26_q3_q4_bootstrap.py, --bank default) uses
run-paper16/axes_vectors_s3_fp32.pt, which is LAB-ONLY; the laptop holds the
OLD run-paper16/axes_vectors_s3.pt plus the audit's cached copy of the fp32
per-persona vectors (audit/cache/labpc/run-paper16/vectors_s3_fp32).  So the
fp32 assistant direction is rebuilt here with 6_axes.py's construction
(assistant = default - mean(role personas)) and both versions are reported.

Reference: run-hf275/published_assistant_axis.pt  (80 x 8192).

Read-only.  Run from the repo root.
"""
from __future__ import annotations

import hashlib
import pathlib

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
LAYERS_DOC = [20, 22, 40, 44, 60]


def md5(p: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_dir(d: pathlib.Path):
    """6_axes.py load_vectors: role -> (80, 8192) float32."""
    vecs = {}
    for f in sorted(d.glob("*.pt")):
        obj = torch.load(f, map_location="cpu", weights_only=False)
        role = obj.get("role", f.stem) if isinstance(obj, dict) else f.stem
        v = obj["vector"] if isinstance(obj, dict) else obj
        vecs[role] = v.float()
    return vecs


def assistant_from(vecs):
    """6_axes.py: assistant = mean(default-like) - mean(all other roles)."""
    dflt = [r for r in vecs if "default" in r]
    other = [r for r in vecs if r not in dflt]
    return (torch.stack([vecs[r] for r in dflt]).mean(0)
            - torch.stack([vecs[r] for r in other]).mean(0)), len(dflt), len(other)


def main():
    pub_p = ROOT / "run-hf275/published_assistant_axis.pt"
    old_p = ROOT / "run-paper16/axes_vectors_s3.pt"
    fp32_dir = ROOT / "audit/cache/labpc/run-paper16/vectors_s3_fp32"

    pub = torch.load(pub_p, map_location="cpu", weights_only=False).float()
    old = torch.load(old_p, map_location="cpu", weights_only=False)

    print("published axis :", pub_p, pub.shape, "md5", md5(pub_p))
    print("paper16 OLD    :", old_p, "md5", md5(old_p),
          "| vectors_dir =", old.get("vectors_dir"), "| counts =", old.get("counts"))
    a_old = old["axes"]["assistant"].float()
    print("  old assistant shape", tuple(a_old.shape))

    vecs = load_dir(fp32_dir)
    a_new, nd, no = assistant_from(vecs)
    print("paper16 fp32   : rebuilt from", fp32_dir, "(%d files: %d default-like, "
          "%d role)" % (len(vecs), nd, no))

    # sanity: does the same construction over the OLD per-persona vectors
    # reproduce the stored OLD assistant axis?
    old_dir = ROOT / "run-paper16/vectors_s3"
    if old_dir.exists():
        a_old_rebuilt, _, _ = assistant_from(load_dir(old_dir))
        chk = min(F.cosine_similarity(a_old_rebuilt[L], a_old[L], dim=-1).item()
                  for L in range(a_old.shape[0]))
        print("  SANITY: rebuild of the OLD bank vs its stored assistant axis, "
              "min cos over 80 layers = %.6f" % chk)

    print()
    print("cos(paper16 assistant, published Assistant Axis), per layer")
    print("%6s %12s %12s %12s" % ("layer", "OLD bank", "fp32 rebuild", "old vs fp32"))
    rows = []
    for L in range(pub.shape[0]):
        c_old = F.cosine_similarity(a_old[L], pub[L], dim=-1).item()
        c_new = F.cosine_similarity(a_new[L], pub[L], dim=-1).item()
        c_oo = F.cosine_similarity(a_old[L], a_new[L], dim=-1).item()
        rows.append((L, c_old, c_new, c_oo))
    for L, c_old, c_new, c_oo in rows:
        mark = "  <-- doc layer" if L in LAYERS_DOC else ""
        if L % 4 == 0 or L in LAYERS_DOC:
            print("%6d %12.4f %12.4f %12.6f%s" % (L, c_old, c_new, c_oo, mark))

    doc_rows = [r for r in rows if r[0] in LAYERS_DOC]
    print()
    print("At the layers the thesis works at (20, 22, 40, 44, 60):")
    for L, c_old, c_new, _ in doc_rows:
        print("   L%-2d  old %.4f   fp32 %.4f" % (L, c_old, c_new))
    print("   OLD  range over those layers : %.4f .. %.4f"
          % (min(r[1] for r in doc_rows), max(r[1] for r in doc_rows)))
    print("   fp32 range over those layers : %.4f .. %.4f"
          % (min(r[2] for r in doc_rows), max(r[2] for r in doc_rows)))

    # the three layers the Q4 table actually erases at
    q4 = [r for r in rows if r[0] in (20, 40, 44)]
    print("   Q4 erasure layers only (20, 40, 44):")
    print("     OLD  %.4f .. %.4f    fp32 %.4f .. %.4f"
          % (min(r[1] for r in q4), max(r[1] for r in q4),
             min(r[2] for r in q4), max(r[2] for r in q4)))

    band = [r for r in rows if 20 <= r[0] <= 60]
    print("   whole L20-L60 band (41 layers):")
    print("     OLD  %.4f .. %.4f    fp32 %.4f .. %.4f"
          % (min(r[1] for r in band), max(r[1] for r in band),
             min(r[2] for r in band), max(r[2] for r in band)))
    all_ = rows
    print("   all 80 layers:")
    print("     OLD  %.4f .. %.4f    fp32 %.4f .. %.4f"
          % (min(r[1] for r in all_), max(r[1] for r in all_),
             min(r[2] for r in all_), max(r[2] for r in all_)))
    print("   max |old - fp32| cos-to-published over all layers: %.6f"
          % max(abs(r[1] - r[2]) for r in rows))
    print("   min cos(old assistant, fp32 assistant) over all layers: %.6f"
          % min(r[3] for r in rows))


if __name__ == "__main__":
    main()
