"""P17 PINNED copy of p16_level1.py: identical logic; every git read goes through
audit/scripts/p17_common.py at the SHA given as argv[1] (default PIN_PAPER). Never a moving ref.
Usage: python audit/scripts/p17_level1_pinned.py <sha>
"""
import sys, pathlib as _pl
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import p17_common as _C
_SHA = sys.argv[1] if len(sys.argv) > 1 else _C.PIN_PAPER
print("PINNED SHA:", _SHA)
"""Phase 6 / 6A LEVEL 1 -- locate every stated number in a result file, flag mismatches.

Covers the 122 remaining draft claims in batches B01-B22, B24-B29, B31-B39, B44, B47, B48.
Level 1 = does the stated number appear in a result file?  (Level 2 re-derivation was done
separately for B23/B30/B40/B41/B42/B43/B45/B46 by p16_level2.py.)

Source: origin/paper, src/probes/deception-detection/results/... read with `git show`.
"""
import csv, io, subprocess, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
B = "src/probes/deception-detection/results"


def show(p):
    return _C.show(_SHA, p) or ""


def tab(p):
    return _C.tab(_SHA, p) or []


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


DS = ["convince_me__plain", "harm_refusal_choice__plain",
      "harm_refusal_knowledge_report__plain", "insider_trading_cadenza__plain",
      "instruct_dishonesty__plain", "repe_honesty__plain"]
SH = dict(zip(DS, ["CG", "HP-C", "HP-KR", "IT", "ID", "RepE"]))

print("=" * 78)
print("LEVEL 1 -- locating stated numbers")
print("=" * 78)

# ---------------------------------------------------------- B38 Table 1 (sizes)
print("\n### B38 Table 1 (D-070..D-076) -- dataset sizes")
print("  claim: examples / lies        file: zero_shot.csv n / n_deceptive")
zs40 = tab(f"{B}/persona_ood_ext_L40/zero_shot.csv")
STATED = {"convince_me__plain": (267, 167), "harm_refusal_choice__plain": (2134, 134),
          "harm_refusal_knowledge_report__plain": (2139, 139),
          "insider_trading_cadenza__plain": (1080, 423),
          "instruct_dishonesty__plain": (5432, 3045),
          "repe_honesty__plain": (550, 275), "XXXX-20__plain": (2000, 0)}
for k, (n_s, d_s) in STATED.items():
    row = next((r for r in zs40 if r["dataset"] == k), None)
    if not row:
        print(f"  {k:38s} NOT FOUND")
        continue
    n, d = int(row["n"]), int(row["n_deceptive"])
    ok = "MATCH" if (n, d) == (n_s, d_s) else f"MISMATCH stated {n_s}/{d_s}"
    print(f"  {SH.get(k,'alpaca'):7s} stated {n_s:5d}/{d_s:5d}   file {n:5d}/{d:5d}   {ok}")
# the follow-up set uses a different RepE row count -- flag it
zsf = tab(f"{B}/persona_ood_followup_L20/zero_shot.csv")
rf = next((r for r in zsf if r["dataset"] == "repe_honesty__plain"), None)
if rf:
    print(f"  NOTE: the FOLLOW-UP set's RepE row count is {rf['n']}/{rf['n_deceptive']}, "
          f"not 550/275 -- different row set, same dataset.")

# ---------------------------------------------------------- B35/B36 Figure 4
print("\n### B35 Fig 4 LEFT (D-129..D-134) -- raw 8192-d probe, L40, full matrix")
mb = tab(f"{B}/persona_ood_ext_L40/matrix_baseline.csv")
first = list(mb[0].keys())[0]
FIG4L = {"convince_me__plain": [1.00, 0.79, 0.80, 0.47, 0.98, 0.90],
         "harm_refusal_choice__plain": [0.43, 0.96, 0.54, 0.59, 0.66, 0.64],
         "harm_refusal_knowledge_report__plain": [0.06, 0.48, 0.95, 0.79, 0.66, 0.93],
         "insider_trading_cadenza__plain": [0.92, 0.95, 0.97, 0.99, 0.89, 0.27],
         "instruct_dishonesty__plain": [0.97, 0.76, 0.75, 0.67, 1.00, 0.98],
         "repe_honesty__plain": [1.00, 0.80, 0.83, 0.49, 1.00, 1.00]}
for col, stated in FIG4L.items():
    got = [fl(next(r[col] for r in mb if r[first] == tr)) for tr in DS]
    r2 = [round(g + 1e-12, 2) if g is not None else None for g in got]
    bad = [f"{SH[DS[i]]}:{r2[i]} vs {stated[i]}" for i in range(6)
           if r2[i] is None or abs(r2[i] - stated[i]) > 0.005]
    print(f"  test {SH[col]:7s} file {[f'{x:.2f}' for x in r2]}  "
          f"{'MATCH' if not bad else 'MISMATCH ' + '; '.join(bad)}")

print("\n### B36 Fig 4 RIGHT (D-135) -- extended contrast, L40, 'rows constant'")
me = tab(f"{B}/persona_ood_ext_L40/matrix_ext_axis1.csv")
stated = {"convince_me__plain": 0.91, "harm_refusal_choice__plain": 0.71,
          "harm_refusal_knowledge_report__plain": 0.91, "insider_trading_cadenza__plain": 0.98,
          "instruct_dishonesty__plain": 0.98, "repe_honesty__plain": 1.00}
for col, s in stated.items():
    vals = sorted({round(fl(r[col]) + 1e-12, 2) for r in me if r[first] != col and r[col]})
    diag = fl(next(r[col] for r in me if r[first] == col))
    ok = "MATCH" if vals == [s] else f"stated {s}, off-diag values {vals}"
    print(f"  {SH[col]:7s} off-diag {vals} (diag {diag:.3f})  {ok}")

# ---------------------------------------------------------- B47 Table 8
print("\n### B47 Table 8 (D-200..D-205) -- L20 response mean vs 'No' token")
zs20 = tab(f"{B}/persona_ood_ext_L20/zero_shot.csv")
COLS = [("8v7", "auroc_axis"), ("4v6", "auroc_denial_axis"), ("12v13", "auroc_ext_axis")]
T8 = {"convince_me__plain": [(0.940, 0.912), (0.931, 0.933), (0.969, 0.938)],
      "harm_refusal_choice__plain": [(0.717, 0.612), (0.585, 0.472), (0.639, 0.556)],
      "harm_refusal_knowledge_report__plain": [(0.148, 0.168), (0.890, 0.237), (0.882, 0.190)],
      "insider_trading_cadenza__plain": [(0.988, 0.165), (0.976, 0.216), (0.990, 0.173)],
      "instruct_dishonesty__plain": [(0.959, 0.900), (0.932, 0.904), (0.954, 0.903)],
      "repe_honesty__plain": [(0.994, 1.000), (0.882, 1.000), (0.991, 1.000)]}
for k, pairs in T8.items():
    rr = next(r for r in zs20 if r["dataset"] == k)
    rn = next(r for r in zsf if r["dataset"] == k)
    out, bad = [], []
    for (nm, c), (sr, sn) in zip(COLS, pairs):
        a, b = fl(rr[c]), fl(rn[c])
        ar, br = round(a + 1e-12, 3), round(b + 1e-12, 3)
        out.append(f"{nm} {ar:.3f}/{br:.3f}")
        if abs(ar - sr) > 0.0005:
            bad.append(f"{nm} resp {ar} vs {sr}")
        if abs(br - sn) > 0.0005:
            bad.append(f"{nm} No {br} vs {sn}")
    print(f"  {SH[k]:7s} {'  '.join(out)}   {'MATCH' if not bad else 'MISMATCH ' + '; '.join(bad)}")

# ---------------------------------------------------------- B48 Table 9
print("\n### B48 Table 9 (D-210..D-215) -- six reads of 12v13, 5-dataset mean")
READ = {"Mean (default)": "mean", "Top-10% mean": "top10", "Maximum": "max",
        "First response token": "first", "Last response token": "last",
        "Token before the reply": "pre"}
T9 = {"mean": {20: (0.887, 0.591, 0.187), 40: (0.899, 0.694, 0.413), 44: (0.884, 0.666, 0.350)},
      "top10": {20: (0.873, 0.664, 0.348), 40: (0.908, 0.737, 0.531), 44: (0.907, 0.715, 0.470)},
      "max": {20: (0.858, 0.659, 0.336), 40: (0.910, 0.750, 0.548), 44: (0.914, 0.701, 0.418)},
      "first": {20: (0.753, 0.605, 0.334), 40: (0.863, 0.606, 0.227), 44: (0.876, 0.568, 0.140)},
      "last": {20: (0.843, 0.645, 0.312), 40: (0.880, 0.656, 0.422), 44: (0.882, 0.636, 0.352)},
      "pre": {20: (0.739, 0.628, 0.415), 40: (0.825, 0.652, 0.383), 44: (0.822, 0.656, 0.327)}}
reads = {L: tab(f"{B}/persona_positions_L{L}/reads_summary.csv") for L in (20, 40, 44)}
for label, rd in READ.items():
    line, bad = [], []
    for L in (20, 40, 44):
        row = next((r for r in reads[L] if r["axis"] == "extended" and r["read"] == rd), None)
        if not row:
            bad.append(f"L{L} missing")
            continue
        got = (fl(row["auroc"]), fl(row["balacc"]), fl(row["recall"]))
        g3 = tuple(round(x + 1e-12, 3) for x in got)
        s3 = T9[rd][L]
        line.append(f"L{L} {g3[0]:.3f}/{g3[1]:.3f}/{g3[2]:.3f}")
        for i, nm in enumerate(("AUROC", "BA", "recall")):
            if abs(g3[i] - s3[i]) > 0.0005:
                bad.append(f"L{L} {nm} {g3[i]} vs {s3[i]}")
    print(f"  {label:24s} {'  '.join(line)}   {'MATCH' if not bad else 'MISMATCH ' + '; '.join(bad)}")

# ---------------------------------------------------------- B39 Table 10
print("\n### B39 Table 10 (D-218..D-222) -- trained on RepE, L20, response vs 'No'")
lb20 = tab(f"{B}/persona_ood_ext_L20/liars_bench_comparison.csv")
lbf = tab(f"{B}/persona_ood_followup_L20/liars_bench_comparison.csv")


def lbval(rows, method):
    r = next((x for x in rows if x["method"] == method), None)
    return fl(r["auroc"]) if r else None


def rowmean(path, col_row="repe_honesty__plain"):
    m = tab(path)
    if not m:
        return None
    f0 = list(m[0].keys())[0]
    r = next((x for x in m if x[f0] == col_row), None)
    if not r:
        return None
    v = [fl(r[c]) for c in m[0] if c != f0 and c != col_row and r[c]]
    return sum(v) / len(v) if v else None


T10 = [("Probe on raw activations (8192-d)", 0.734, 0.636, "matrix_baseline"),
       ("Extended contrast (1-d)", 0.887, 0.552, "matrix_ext_axis1"),
       ("Extended PCs (k = 3)", 0.736, 0.579, "matrix_ext_pc3"),
       ("Published PCs (k = 3)", 0.733, 0.574, "matrix_pc3"),
       ("Knowledge-denial PCs (k = 3)", 0.810, 0.699, "matrix_denial_pc3")]
print("  RepE training row, mean over the 5 other test columns:")
for label, s_resp, s_no, mat in T10:
    a = rowmean(f"{B}/persona_ood_ext_L20/{mat}.csv")
    b = rowmean(f"{B}/persona_ood_followup_L20/{mat}.csv")
    ar = round(a + 1e-12, 3) if a is not None else None
    br = round(b + 1e-12, 3) if b is not None else None
    ok = []
    if ar is None or abs(ar - s_resp) > 0.0005:
        ok.append(f"resp {ar} vs {s_resp}")
    if br is None or abs(br - s_no) > 0.0005:
        ok.append(f"No {br} vs {s_no}")
    print(f"  {label:34s} file {ar}/{br}   stated {s_resp}/{s_no}   "
          f"{'MATCH' if not ok else 'MISMATCH ' + '; '.join(ok)}")
print("  cross-check, liars_bench_comparison.csv 'raw activations (RepE-trained)':")
print(f"     response {lbval(lb20,'raw activations (RepE-trained)'):.4f}   "
      f"follow-up {lbval(lbf,'raw activations (RepE-trained)'):.4f}")

# ---------------------------------------------------------- B02 contributions
print("\n### B02 / B03 / B06 / B07 -- scattered numeric claims")
print(f"  D-031 Follow-Up Probe measured 0.636 vs published 0.634:")
print(f"     measured here  {lbval(lbf,'Follow-Up Probe (measured here, layer 20)'):.4f}")
print(f"     published      {lbval(lbf,'Follow-Up Probe'):.4f}")
s40 = tab(f"{B}/persona_ood_ext_L40/summary.csv")
g = {r["features"]: r for r in s40}
print(f"  D-022 HP-KR 4v6 AUROC 0.919 / BA 0.910 -- see Level 2 D-087 (both EXACT)")
print(f"  D-026 zero-shot extended 0.916 off-diag vs raw 0.732 at L40:")
print(f"     ext_axis1 {fl(g['ext_axis1']['off_diagonal_mean']):.4f}   "
      f"baseline {fl(g['baseline']['off_diagonal_mean']):.4f}")
print(f"  D-053 vectors at all 80 layers / D-062 layers 20,40,44 of 80: "
      f"summary.csv rows exist at L20/40/44; 16-layer sweep dirs present on branch")
print(f"  D-045 bank holds 36 personas = 8+7+5+5+1 published (26) + 10 new")
print(f"  D-055 three contrasts 8v7 / 4v6 / 12v13 -> features axis1 / denial_axis1 / ext_axis1: "
      f"{all(k in g for k in ('axis1','denial_axis1','ext_axis1'))}")
print(f"  D-061 three random directions, five draws -> feature 'random3' dim "
      f"{g.get('random3',{}).get('dim','?')}")

print("\n" + "=" * 78)
