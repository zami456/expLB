"""W6C-briefing: claims that live in the COLLABORATOR's committed CSVs.

Covers O-008 (extended-contrast margin +0.256/+0.184), O-019 (+0.143/+0.100),
O-023 (ext_pc1 trails ext_axis1 by 0.243 at L20; 1-d contrast beats 3-d at every
depth), O-035 (0.732 is the single-source trained baseline at L40) and O-036
(the "median over the same pairs is 0.722 / sign-oriented mean 0.735" sentence).

Everything is read out of git with `git show` -- no checkout, read-only.
Run from the repo root.
"""
import csv
import io
import subprocess
import sys

REPO = "C:/Users/SURAIYA/Documents/STUDY/thesis/expLB"
REFS = ["origin/feat/persona-pc-ood", "f386a72"]
BASE = "src/probes/deception-detection/results"


def gitshow(ref, path):
    out = subprocess.run(["git", "-C", REPO, "show", "%s:%s" % (ref, path)],
                         capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8", "replace"))
    return out.stdout.decode("utf-8", "replace")


def rows(ref, path):
    return list(csv.DictReader(io.StringIO(gitshow(ref, path))))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summary_val(ref, layer, feature, col="off_diagonal_mean"):
    for r in rows(ref, "%s/persona_ood_ext_L%d/summary.csv" % (BASE, layer)):
        if r.get("features") == feature or r.get(list(r.keys())[0]) == feature:
            return fnum(r[col])
    return None


def zs_mean(ref, layer, col):
    """Mean of a zero_shot.csv column over the scored datasets (blank rows dropped)."""
    vals = []
    names = []
    for r in rows(ref, "%s/persona_ood_ext_L%d/zero_shot.csv" % (BASE, layer)):
        v = fnum(r.get(col))
        if v is not None:
            vals.append(v)
            names.append(r.get(list(r.keys())[0]))
    return (sum(vals) / len(vals) if vals else None), names, vals


def offdiag(ref, layer, matrix="matrix_baseline.csv", drop=()):
    """Return (list of off-diagonal cells, row labels) of a transfer matrix."""
    txt = gitshow(ref, "%s/persona_ood_ext_L%d/%s" % (BASE, layer, matrix))
    rd = list(csv.reader(io.StringIO(txt)))
    hdr = rd[0]
    cols = hdr[1:]
    cells = {}
    for line in rd[1:]:
        src = line[0]
        for j, c in enumerate(line[1:]):
            v = fnum(c)
            if v is not None:
                cells[(src, cols[j])] = v
    srcs = sorted({k[0] for k in cells})
    tgts = sorted({k[1] for k in cells})
    keep_s = [s for s in srcs if s not in drop]
    keep_t = [t for t in tgts if t not in drop]
    od = [(s, t, cells[(s, t)]) for s in keep_s for t in keep_t
          if s != t and (s, t) in cells]
    return od, keep_s, keep_t


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


out = sys.stdout
p = out.write

p("=" * 78 + "\n")
p("SOURCE: origin/feat/persona-pc-ood, read with `git show` (no checkout)\n")
for ref in REFS:
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", ref],
                         capture_output=True).stdout.decode().strip()
    p("  %-28s -> %s\n" % (ref, sha))
p("=" * 78 + "\n\n")

REF = REFS[0]

p("### Level-1 / Level-2 : summary.csv off_diagonal_mean vs recomputed zero-shot means\n")
p("%-6s %-14s %12s %14s %12s\n" % ("layer", "feature", "summary.csv", "zs mean", "|diff|"))
table = {}
for layer in (20, 40, 44):
    for feat, zscol in (("baseline", None), ("axis1", "auroc_axis"),
                        ("ext_axis1", "auroc_ext_axis"), ("pc1", "auroc_pc1"),
                        ("ext_pc1", "auroc_ext_pc1"), ("pc3", None),
                        ("ext_pc3", None)):
        sv = summary_val(REF, layer, feat)
        if sv is None:
            continue
        zv = None
        if zscol:
            zv, _, _ = zs_mean(REF, layer, zscol)
        table[(layer, feat)] = (sv, zv)
        p("%-6d %-14s %12.10f %14s %12s\n" % (
            layer, feat, sv,
            ("%.10f" % zv) if zv is not None else "-",
            ("%.3e" % abs(sv - zv)) if zv is not None else "-"))
p("\n")

p("### O-008  margin, EXTENDED contrast vs baseline   (briefing: +0.256 L20, +0.184 L40)\n")
for layer in (20, 40):
    b = table[(layer, "baseline")][0]
    e = table[(layer, "ext_axis1")][0]
    ezs = table[(layer, "ext_axis1")][1]
    p("  L%-3d ext_axis1 %.4f - baseline %.4f = %+.4f   (zs-mean version %+.4f)\n"
      % (layer, e, b, e - b, ezs - b))
p("\n")

p("### O-019  margin, PAPER'S 8v7 contrast vs baseline  (briefing: +0.143 L20, +0.100 L40)\n")
for layer in (20, 40):
    b = table[(layer, "baseline")][0]
    a = table[(layer, "axis1")][0]
    azs = table[(layer, "axis1")][1]
    p("  L%-3d axis1 %.4f - baseline %.4f = %+.4f   (zs-mean version %+.4f)\n"
      % (layer, a, b, a - b, azs - b))
p("  [note] summary.csv axis1 is the TRAINED off-diagonal mean; the briefing's\n")
p("         0.7909/0.8319 are the ZERO-SHOT means of auroc_axis.\n\n")

p("### O-035  0.732 = the trained single-source baseline at L40 (his six datasets)\n")
for layer in (20, 40, 44):
    b = table[(layer, "baseline")][0]
    od, srcs, tgts = offdiag(REF, layer)
    p("  L%-3d summary.csv baseline off_diagonal_mean = %.6f ; recomputed from\n"
      "        matrix_baseline.csv off-diagonal = %.6f  (n=%d pairs, %d datasets)\n"
      % (layer, b, sum(v for _, _, v in od) / len(od), len(od), len(srcs)))
p("\n")

p("### O-036  median / sign-oriented mean OVER HIS OWN PAIRS (briefing quotes 0.722 / 0.735)\n")
for layer in (20, 40, 44):
    od, srcs, tgts = offdiag(REF, layer)
    vals = [v for _, _, v in od]
    som = [max(v, 1.0 - v) for v in vals]
    p("  L%-3d  n=%-3d  mean %.4f   median %.4f   sign-oriented mean %.4f   min %.4f\n"
      % (layer, len(vals), sum(vals) / len(vals), median(vals),
         sum(som) / len(som), min(vals)))
    p("        below 0.5: %d   below 0.4: %d\n"
      % (sum(1 for v in vals if v < 0.5), sum(1 for v in vals if v < 0.4)))
p("\n")

p("### O-023  ext_pc1 vs ext_axis1, and 1-d contrast vs 3-d subspace, at every layer he ran\n")
p("%-6s %10s %10s %10s | %10s %10s %10s\n" % (
    "layer", "ext_axis1", "ext_pc1", "pc1-axis", "axis1", "pc3", "ext_pc3"))
for layer in (20, 40, 44):
    ea = table[(layer, "ext_axis1")][0]
    ep = table.get((layer, "ext_pc1"), (None,))[0]
    a1 = table[(layer, "axis1")][0]
    p3 = table.get((layer, "pc3"), (None,))[0]
    ep3 = table.get((layer, "ext_pc3"), (None,))[0]
    p("%-6d %10.4f %10s %10s | %10.4f %10s %10s\n" % (
        layer, ea,
        "%.4f" % ep if ep is not None else "-",
        "%+.4f" % (ep - ea) if ep is not None else "-",
        a1,
        "%.4f" % p3 if p3 is not None else "-",
        "%.4f" % ep3 if ep3 is not None else "-"))
p("\n  1-d vs 3-d, per layer (positive = the 1-d contrast wins):\n")
for layer in (20, 40, 44):
    a1 = table[(layer, "axis1")][0]
    ea = table[(layer, "ext_axis1")][0]
    p3 = table.get((layer, "pc3"), (None,))[0]
    ep3 = table.get((layer, "ext_pc3"), (None,))[0]
    p("  L%-3d axis1-pc3 %+.4f    ext_axis1-ext_pc3 %+.4f\n"
      % (layer, a1 - p3, ea - ep3))
p("\n")

p("### full summary.csv rows at L20/L40/L44 (features, off_diagonal_mean)\n")
for layer in (20, 40, 44):
    p("  --- L%d ---\n" % layer)
    for r in rows(REF, "%s/persona_ood_ext_L%d/summary.csv" % (BASE, layer)):
        k = list(r.keys())
        p("    %-16s %s\n" % (r[k[0]], "  ".join("%s=%s" % (c, r[c]) for c in k[1:])))
p("\n")

p("### cross-check: identical at f386a72 ?\n")
for layer in (20, 40):
    for feat in ("baseline", "axis1", "ext_axis1", "ext_pc1"):
        a = summary_val(REFS[0], layer, feat)
        b = summary_val(REFS[1], layer, feat)
        p("  L%-3d %-11s head %.10f   f386a72 %.10f   %s\n"
          % (layer, feat, a, b, "SAME" if a == b else "DIFFERENT"))
