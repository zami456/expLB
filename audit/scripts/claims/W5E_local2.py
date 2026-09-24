"""W5E-ownerdoc part 2: O-015 (collaborator's 0.916 / 0.732 / +0.184) and
O-040 (the 'regex 0.91 vs probe ~0.87' pair).

O-015 is recomputed from the collaborator's committed CSVs, extracted to a temp dir with
`git show` (read-only) by the caller; here we read them from stdin-provided paths.
O-040 searches OUR kd11 matrices for any probe AUROC on harm-pressure-knowledge-report near 0.87.
"""
import csv, io, json, os, statistics, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BR = "origin/feat/persona-pc-ood"
BASE = "src/probes/deception-detection/results"


def gshow(path):
    return subprocess.run(["git", "show", f"{BR}:{path}"], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8", check=True).stdout


print("#" * 78)
print("O-015 -- briefing:332  '0.916 vs single-source 0.732 (the abstract) | +0.184'")
for L in (20, 40):
    d = f"{BASE}/persona_ood_ext_L{L}"
    summ = list(csv.DictReader(io.StringIO(gshow(f"{d}/summary.csv"))))
    by = {r["features"]: r for r in summ}
    ext = float(by["ext_axis1"]["off_diagonal_mean"])
    base = float(by["baseline"]["off_diagonal_mean"])
    ax1 = float(by["axis1"]["off_diagonal_mean"])
    print(f"L{L}  summary.csv (committed on {BR}):")
    print(f"   ext_axis1 off_diagonal_mean = {ext!r}")
    print(f"   baseline  off_diagonal_mean = {base!r}   (single-source: train on one, test on another)")
    print(f"   axis1     off_diagonal_mean = {ax1!r}")
    print(f"   ext_axis1 - baseline = {ext - base:+.6f}")
    print(f"   axis1     - baseline = {ax1 - base:+.6f}")
    # LEVEL 2a: baseline off-diagonal recomputed from the stored 6x6 matrix
    rows = list(csv.reader(io.StringIO(gshow(f"{d}/matrix_baseline.csv"))))
    hdr = rows[0][1:]
    m = {r[0]: {hdr[i]: float(v) for i, v in enumerate(r[1:])} for r in rows[1:]}
    off = [m[s][t] for s in m for t in hdr if s != t]
    print(f"   LEVEL2 matrix_baseline.csv: n_off={len(off)} off-diag mean = {statistics.fmean(off):.6f}"
          f"  (matrix is rounded to 4 dp)")
    # LEVEL 2b: ext_axis1 is zero-shot -> the per-dataset zero_shot.csv column
    zs = list(csv.DictReader(io.StringIO(gshow(f"{d}/zero_shot.csv"))))
    vals = [(r["dataset"], float(r["auroc_ext_axis"])) for r in zs if r["auroc_ext_axis"]]
    print(f"   LEVEL2 zero_shot.csv auroc_ext_axis over {len(vals)} datasets:")
    for k, v in vals:
        print(f"      {k:<42}{v!r}")
    print(f"      mean = {statistics.fmean(v for _, v in vals)!r}")
    print(f"      mean(ext) - baseline off-diag = {statistics.fmean(v for _, v in vals) - base:+.6f}")
    rows2 = list(csv.reader(io.StringIO(gshow(f"{d}/matrix_ext_axis1.csv"))))
    hdr2 = rows2[0][1:]
    m2 = {r[0]: {hdr2[i]: float(v) for i, v in enumerate(r[1:])} for r in rows2[1:]}
    off2 = [m2[s][t] for s in m2 for t in hdr2 if s != t]
    print(f"   LEVEL2 matrix_ext_axis1.csv off-diag mean = {statistics.fmean(off2):.6f} (n={len(off2)})")

print()
print("#" * 78)
print("O-040 -- NOVELTY_ASSESSMENT:30 / NOVELTY_CHECK_PROMPT:103-108")
print("  'A regex on denial phrasing alone reaches 0.91 AUROC on a dataset where probes report ~0.87'")
print("  Searching OUR kd11 matrices for probe AUROCs on harm-pressure-knowledge-report.")
mp = os.path.join(ROOT, "run-crossgen", "kd11", "matrices", "mean_L40_s42.json")
d = json.load(open(mp, encoding="utf-8"))
ds = d["datasets"]
print(f"  file: {mp}  bank={d.get('bank')} bank_name={d.get('bank_name')} k={d.get('k')}")
print(f"  datasets: {ds}")
tgt = "harm-pressure-knowledge-report"
j = ds.index(tgt)
print(f"  {'feature':<22}{'probe':<10}{'diagonal':>10}{'col-mean(as target)':>22}{'row-mean(as source)':>22}")
for r in d["results"]:
    col = [r["auroc"][i][j] for i in range(len(ds)) if i != j and r["auroc"][i][j] is not None]
    row = [r["auroc"][j][i] for i in range(len(ds)) if i != j and r["auroc"][j][i] is not None]
    diag = r["auroc"][j][j]
    print(f"  {r['feature']:<22}{r['probe']:<10}{diag if diag is None else round(diag,4):>10}"
          f"{round(statistics.fmean(col),4):>22}{round(statistics.fmean(row),4):>22}")
print("  zero-shot directions on HP-KR:")
for z in d.get("zeroshot", []):
    v = z["auroc"][j]
    print(f"    {z['direction']:<34}{v if v is None else round(v,4)}")
