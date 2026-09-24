"""F-39 sensitivity (owner-authorised, item 5): how much does the "our effect is smaller than the paper's"
conclusion depend on the dataset set? Runs the owner's own paper_comparison.py over dataset variants of the
paper's five (VERIFIED from the PDF: ConvGame, InstrDec, Mask, AILiar, Roleplay) and prints one table.
Read-only: it only reads run-crossgen matrices that are committed in git."""
import re, subprocess, sys
M = sys.argv[1] if len(sys.argv) > 1 else "run-crossgen/kd11/matrices/mean_L40_s42.json"
PAPER5 = ["instructed-deception", "mask", "convincing-game", "ai_liar", "roleplaying"]
def run(ds):
    out = subprocess.run([sys.executable, "kd/scripts/paper_comparison.py", M, "--datasets", *ds],
                         capture_output=True, text=True).stdout
    raw = float(re.search(r"raw off-diagonal mean = ([\d.]+)", out).group(1))
    feats = {m.group(1): (float(m.group(2)), float(m.group(3)))
             for m in re.finditer(r"^\s{2}(\S+)\s+([\d.]+)\s+([+-][\d.]+)\s+(\d+)", out, re.M)}
    pairs = int(re.search(r"over (\d+) pairs", out).group(1))
    return raw, feats, pairs
rows = []
rows.append(("paper's five (mask)", PAPER5))
rows.append(("swap mask -> harm-pressure-choice", [d if d != "mask" else "harm-pressure-choice" for d in PAPER5]))
for d in PAPER5:
    rows.append((f"paper's five minus {d}", [x for x in PAPER5 if x != d]))
print(f"matrix: {M}\n")
print(f"{'dataset set':38s} {'pairs':>5s} {'raw':>7s} {'axis@2 d':>9s} {'dPCA@2 d':>9s} {'axis@2 - dPCA@2':>16s} {'axes d':>8s} {'k=5 axis d':>11s}")
for label, ds in rows:
    raw, f, pairs = run(ds)
    a2 = f.get("pc:unified@2", (0, 0))[1]; d2 = f.get("datapca@2", (0, 0))[1]
    ax = f.get("axes", (0, 0))[1]; a5 = f.get("pc:unified", (0, 0))[1]
    print(f"{label:38s} {pairs:5d} {raw:7.4f} {a2:+9.4f} {d2:+9.4f} {a2 - d2:+16.4f} {ax:+8.4f} {a5:+11.4f}")
print("\nPaper (Mahadik & Skapars, Fig 9A, k=1): axis vs raw +0.10..+0.17; dataset-PCA vs raw -0.12..+0.01;")
print("so the paper's axis-minus-datasetPCA margin is +0.13..+0.23.")
