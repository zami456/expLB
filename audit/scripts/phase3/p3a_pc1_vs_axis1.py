"""P3-A: read the collaborator's own committed summary.csv files and print the
published-bank PC1 (the replicated paper's OWN k=1 deception setting) against the
published 8v7 contrast. Read-only; uses `git show`, no checkout."""
import csv, io, subprocess

REF = "origin/feat/persona-pc-ood"
BASE = "src/probes/deception-detection/results/persona_ood_ext_L{L}/summary.csv"

rows = {}
for L in (20, 40, 44):
    txt = subprocess.run(["git", "show", f"{REF}:{BASE.format(L=L)}"],
                         capture_output=True, text=True, check=True).stdout
    rows[L] = {r["features"]: r for r in csv.DictReader(io.StringIO(txt))}

def g(L, f, col="off_diagonal_mean"):
    return float(rows[L][f][col])

print("col = off_diagonal_mean (all six test sets)")
print(f"{'':28s} {'L20':>9s} {'L40':>9s} {'L44':>9s}")
for f in ("baseline", "axis1", "pc1", "pc3", "ext_axis1", "ext_pc1", "ext_pc3",
          "denial_axis1", "syc_axis1", "syc_pc1", "syc3", "syc3_orth", "dec3_orth", "random3"):
    print(f"{f:28s} " + " ".join(f"{g(L,f):9.4f}" for L in (20, 40, 44)))

print("\npublished PC1 minus published contrast  (paper's own deception setting is k=1)")
for col in ("off_diagonal_mean", "off_diagonal_excl_hpkr"):
    print(f"  {col:24s} " + " ".join(f"L{L}={g(L,'pc1',col)-g(L,'axis1',col):+.4f}" for L in (20, 40, 44)))

print("\nsycophancy PC1 minus published deception contrast")
print("  off_diagonal_mean        " + " ".join(f"L{L}={g(L,'syc_pc1')-g(L,'axis1'):+.4f}" for L in (20, 40, 44)))

print("\ncross-behaviour orthogonalisation cost (his control, absent from draft and from her docs)")
print("  dec3_orth - pc3          " + " ".join(f"L{L}={g(L,'dec3_orth')-g(L,'pc3'):+.4f}" for L in (20, 40, 44)))
print("  syc3_orth - syc3         " + " ".join(f"L{L}={g(L,'syc3_orth')-g(L,'syc3'):+.4f}" for L in (20, 40, 44)))
