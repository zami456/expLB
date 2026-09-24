"""P5-draft-headlines — Level-1 check of the collaborator's draft headline claims.

Reads ONLY files committed on origin/feat/persona-pc-ood via `git show` (never checkout).
His activations are gone, so nothing here is a Level-2 recomputation of his pipeline;
what IS recomputed are the *aggregates* he derived from committed cell-level CSVs
(off-diagonal means, per-test-set means, differences) -- i.e. internal consistency of
his own committed numbers.

Run:  PYTHONIOENCODING=utf-8 python audit/scripts/claims/P5_draft_headlines.py
"""
import csv
import io
import subprocess
import sys

REPO = r"C:/Users/SURAIYA/Documents/STUDY/thesis/expLB"
BR = "origin/feat/persona-pc-ood"
RES = "src/probes/deception-detection/results"
LAYERS = [20, 40, 44]
DS = ["convince_me__plain", "harm_refusal_choice__plain",
      "harm_refusal_knowledge_report__plain", "insider_trading_cadenza__plain",
      "instruct_dishonesty__plain", "repe_honesty__plain"]
SHORT = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C",
         "harm_refusal_knowledge_report__plain": "HP-KR",
         "insider_trading_cadenza__plain": "IT",
         "instruct_dishonesty__plain": "ID", "repe_honesty__plain": "RepE"}


def show(path):
    out = subprocess.run(["git", "-C", REPO, "show", f"{BR}:{path}"],
                         capture_output=True)
    if out.returncode != 0:
        raise SystemExit(f"git show failed for {path}: {out.stderr.decode()[:200]}")
    return out.stdout.decode("utf-8")


def rows(path):
    return list(csv.DictReader(io.StringIO(show(path))))


def matrix(layer, name):
    """matrix_<name>.csv -> {train_ds: {test_ds: value}}  (rows = train, cols = test)."""
    txt = show(f"{RES}/persona_ood_ext_L{layer}/matrix_{name}.csv")
    rd = list(csv.reader(io.StringIO(txt)))
    hdr = rd[0][1:]
    m = {}
    for r in rd[1:]:
        m[r[0]] = {hdr[i]: float(v) for i, v in enumerate(r[1:])}
    return m


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def cmp(label, doc, got, tol=0.0005):
    ok = abs(doc - got) <= tol
    print(f"  {label:<56s} draft={doc:<8} file={got:.6f}  {'MATCH' if ok else '*** MISMATCH ***'}")
    return ok


# ---------------------------------------------------------------- zero_shot
hdr("A. zero_shot.csv (his committed per-dataset zero-shot AUROC)")
zs = {L: {r["dataset"]: r for r in rows(f"{RES}/persona_ood_ext_L{L}/zero_shot.csv")}
      for L in LAYERS}
for L in LAYERS:
    r = zs[L]["harm_refusal_knowledge_report__plain"]
    print(f"  L{L} HP-KR n={r['n']} n_dec={r['n_deceptive']} "
          f"axis(8v7)={float(r['auroc_axis']):.4f} "
          f"denial(4v6)={float(r['auroc_denial_axis']):.4f} "
          f"ext(12v13)={float(r['auroc_ext_axis']):.4f}")

hdr("D-003 / D-083 / D-006 : HP-KR zero-shot AUROC")
cmp("D-003 published contrast (8v7) HP-KR L20 = 0.148",
    0.148, float(zs[20]["harm_refusal_knowledge_report__plain"]["auroc_axis"]))
for L, v in zip(LAYERS, [0.890, 0.919, 0.908]):
    cmp(f"D-083 knowledge-denial (4v6) HP-KR L{L} = {v}",
        v, float(zs[L]["harm_refusal_knowledge_report__plain"]["auroc_denial_axis"]))
print("  [draft 5.1 also states 8v7 = 0.593 / 0.589 at L40 / L44]")
for L, v in zip([40, 44], [0.593, 0.589]):
    cmp(f"      published (8v7) HP-KR L{L} = {v}",
        v, float(zs[L]["harm_refusal_knowledge_report__plain"]["auroc_axis"]))
print("  [Table 2 extended contrast (12v13) = 0.882 / 0.912 / 0.894]")
for L, v in zip(LAYERS, [0.882, 0.912, 0.894]):
    cmp(f"      extended (12v13) HP-KR L{L} = {v}",
        v, float(zs[L]["harm_refusal_knowledge_report__plain"]["auroc_ext_axis"]))

hdr("Table 6 (all six datasets x three contrasts) -- underpins D-099")
t6 = {20: {"CG": (0.940, 0.931, 0.969), "HP-C": (0.717, 0.585, 0.639),
           "HP-KR": (0.148, 0.890, 0.882), "IT": (0.988, 0.976, 0.990),
           "ID": (0.959, 0.932, 0.954), "RepE": (0.994, 0.882, 0.991)},
      40: {"CG": (0.865, 0.902, 0.912), "HP-C": (0.581, 0.755, 0.712),
           "HP-KR": (0.593, 0.919, 0.912), "IT": (0.979, 0.913, 0.980),
           "ID": (0.974, 0.980, 0.978), "RepE": (1.000, 1.000, 1.000)},
      44: {"CG": (0.819, 0.858, 0.859), "HP-C": (0.607, 0.725, 0.712),
           "HP-KR": (0.589, 0.908, 0.894), "IT": (0.979, 0.933, 0.982),
           "ID": (0.962, 0.978, 0.974), "RepE": (1.000, 1.000, 1.000)}}
bad6 = 0
for L in LAYERS:
    for d in DS:
        s = SHORT[d]
        got = (float(zs[L][d]["auroc_axis"]), float(zs[L][d]["auroc_denial_axis"]),
               float(zs[L][d]["auroc_ext_axis"]))
        for lbl, dv, gv in zip(("8v7", "4v6", "12v13"), t6[L][s], got):
            if abs(dv - gv) > 0.0005:
                print(f"  *** L{L} {s} {lbl}: draft {dv} file {gv:.4f}")
                bad6 += 1
print(f"  Table 6 cells checked: 54, mismatches: {bad6}")
print("  Table 6 'Mean' row (draft): L20 .791/.866/.904  L40 .832/.912/.916  L44 .826/.900/.903")
print("  -> that Mean row is NOT the mean of the 6 zero-shot cells; it is summary.csv off_diagonal_mean:")
for L in LAYERS:
    m6 = {k: sum(float(zs[L][d][k]) for d in DS) / 6 for k in
          ("auroc_axis", "auroc_denial_axis", "auroc_ext_axis")}
    print(f"    L{L} mean-of-6-cells = {m6['auroc_axis']:.4f} / "
          f"{m6['auroc_denial_axis']:.4f} / {m6['auroc_ext_axis']:.4f}")

# ---------------------------------------------------------------- summary.csv
hdr("B. summary.csv off_diagonal_mean -- D-008 / D-098 / D-028 / D-010 / D-104 / D-168")
summ = {}
for L in LAYERS:
    summ[L] = {r["features"]: r for r in rows(f"{RES}/persona_ood_ext_L{L}/summary.csv")}
    for f in ("baseline", "axis1", "denial_axis1", "ext_axis1", "ext_pc3", "ext_pc1",
              "pc3", "random3"):
        r = summ[L][f]
        print(f"  L{L:<3} {f:<14} dim={r['dim']:<5} off_diag={float(r['off_diagonal_mean']):.6f} "
              f"excl_hpkr={float(r['off_diagonal_excl_hpkr']):.6f} "
              f"diag={float(r['diagonal_mean']):.6f}")

hdr("D-008 (abstract) / D-098 (5.2): off-diagonal means")
cmp("D-008 ext_axis1 L40 = 0.916", 0.916, float(summ[40]["ext_axis1"]["off_diagonal_mean"]))
cmp("D-008 baseline  L40 = 0.732", 0.732, float(summ[40]["baseline"]["off_diagonal_mean"]))
for L, v in zip(LAYERS, [0.904, 0.916, 0.903]):
    cmp(f"D-098 ext_axis1 L{L} = {v}", v, float(summ[L]["ext_axis1"]["off_diagonal_mean"]))
for L, v in zip(LAYERS, [0.648, 0.732, 0.716]):
    cmp(f"D-098 baseline  L{L} = {v}", v, float(summ[L]["baseline"]["off_diagonal_mean"]))
print("  [5.2 also: in-distribution 0.983 at L40, loses 0.251]")
d40 = float(summ[40]["baseline"]["diagonal_mean"])
o40 = float(summ[40]["baseline"]["off_diagonal_mean"])
cmp("      baseline diagonal_mean L40 = 0.983", 0.983, d40)
cmp("      diagonal - off-diagonal   = 0.251", 0.251, d40 - o40, tol=0.001)

hdr("D-028 / D-104 / D-010 : 1-d contrast vs persona PCs")
for L in LAYERS:
    a = float(summ[L]["ext_axis1"]["off_diagonal_mean"])
    p3 = float(summ[L]["ext_pc3"]["off_diagonal_mean"])
    p1 = float(summ[L]["ext_pc1"]["off_diagonal_mean"])
    print(f"  L{L}: ext_axis1 {a:.4f}  ext_pc3 {p3:.4f}  ext_pc1 {p1:.4f}  "
          f"axis1-pc3 = {a - p3:+.4f}   beats_pc3={a > p3}  beats_pc1={a > p1}")
cmp("D-028 gap L20 = 0.220", 0.220,
    float(summ[20]["ext_axis1"]["off_diagonal_mean"]) - float(summ[20]["ext_pc3"]["off_diagonal_mean"]))
cmp("D-028 gap L44 = 0.023", 0.023,
    float(summ[44]["ext_axis1"]["off_diagonal_mean"]) - float(summ[44]["ext_pc3"]["off_diagonal_mean"]))
cmp("5.3   gap L40 = 0.050", 0.050,
    float(summ[40]["ext_axis1"]["off_diagonal_mean"]) - float(summ[40]["ext_pc3"]["off_diagonal_mean"]))

hdr("Table 3 -- every cell (context for D-010/D-104/D-168)")
t3 = [("Probe on raw activations (8192-d)", "baseline", (0.648, 0.732, 0.716), (0.728, 0.758, 0.730)),
      ("Published contrast, 8 v 7 (1-d)", "axis1", (0.651, 0.832, 0.826), (0.919, 0.880, 0.873)),
      ("Knowledge-denial contrast, 4 v 6", "denial_axis1", (0.866, 0.912, 0.900), (0.861, 0.910, 0.899)),
      ("Extended contrast, 12 v 13 (1-d)", "ext_axis1", (0.904, 0.916, 0.903), (0.909, 0.917, 0.905)),
      ("Extended PCs (k=3)", "ext_pc3", (0.684, 0.866, 0.880), (0.900, 0.867, 0.870)),
      ("Extended PC1", "ext_pc1", (0.661, 0.768, 0.780), (0.682, 0.774, 0.793)),
      ("Published PCs (k=3)", "pc3", (0.679, 0.639, 0.617), (0.888, 0.751, 0.728)),
      ("Random directions (k=3)", "random3", (0.527, 0.504, 0.493), (0.560, 0.544, 0.547))]
bad3 = 0
for lbl, key, allv, exv in t3:
    for L, dv in zip(LAYERS, allv):
        gv = float(summ[L][key]["off_diagonal_mean"])
        if abs(dv - gv) > 0.0005:
            print(f"  *** {lbl} L{L} all-six: draft {dv} file {gv:.4f}"); bad3 += 1
    for L, dv in zip(LAYERS, exv):
        gv = float(summ[L][key]["off_diagonal_excl_hpkr"])
        if abs(dv - gv) > 0.0005:
            print(f"  *** {lbl} L{L} excl-HPKR: draft {dv} file {gv:.4f}"); bad3 += 1
print(f"  Table 3 cells checked: 48, mismatches: {bad3}")

# ------------------------------------------------- matrices -> Table 7
hdr("C. Table 7 recomputed from matrix_*.csv -- D-009 / D-027 / D-099 / D-195")
t7 = {20: {"CG": (0.663, 0.969), "HP-C": (0.535, 0.639), "HP-KR": (0.394, 0.882),
           "IT": (0.799, 0.990), "ID": (0.778, 0.954), "RepE": (0.718, 0.991)},
      40: {"CG": (0.788, 0.912), "HP-C": (0.571, 0.712), "HP-KR": (0.584, 0.912),
           "IT": (0.800, 0.980), "ID": (0.827, 0.978), "RepE": (0.824, 1.000)},
      44: {"CG": (0.757, 0.859), "HP-C": (0.570, 0.712), "HP-KR": (0.637, 0.894),
           "IT": (0.736, 0.982), "ID": (0.808, 0.974), "RepE": (0.785, 1.000)}}
bad7 = 0
percol = {}
for L in LAYERS:
    mb = matrix(L, "baseline")
    me = matrix(L, "ext_axis1")
    ma = matrix(L, "axis1")
    print(f"  --- L{L} (mean over the 5 OTHER training sets, per test set) ---")
    percol[L] = {}
    for d in DS:
        raw = sum(mb[t][d] for t in DS if t != d) / 5
        ext = sum(me[t][d] for t in DS if t != d) / 5
        pub = sum(ma[t][d] for t in DS if t != d) / 5
        percol[L][d] = (raw, ext, pub)
        dr, de = t7[L][SHORT[d]]
        okr = abs(dr - raw) <= 0.0006
        oke = abs(de - ext) <= 0.0006
        bad7 += (not okr) + (not oke)
        print(f"    {SHORT[d]:<6} raw draft={dr} calc={raw:.4f} {'ok' if okr else '***'} | "
              f"12v13 draft={de} calc={ext:.4f} {'ok' if oke else '***'} | "
              f"8v7 calc={pub:.4f} | ext>raw={ext > raw} ext>=pub={ext >= pub}")
print(f"  Table 7 cells checked: 36, mismatches: {bad7}")

hdr("D-009 / D-027 / D-195 verdict inputs: does 12v13 beat raw on EVERY test set, EVERY layer?")
allbeat = True
for L in LAYERS:
    for d in DS:
        raw, ext, pub = percol[L][d]
        if not ext > raw:
            print(f"  *** L{L} {SHORT[d]}: ext {ext:.4f} <= raw {raw:.4f}")
            allbeat = False
print(f"  12v13 > raw on all 18 (test set x layer) cells: {allbeat}")

hdr("D-099 second half: does 12v13 'match or beat' the published contrast (8v7)?")
for L in LAYERS:
    for d in DS:
        raw, ext, pub = percol[L][d]
        flag = "" if ext >= pub - 0.0005 else "   <-- BELOW published"
        print(f"  L{L} {SHORT[d]:<6} 12v13={ext:.4f}  8v7={pub:.4f}  diff={ext - pub:+.4f}{flag}")

hdr("Off-diagonal mean recomputed from the matrices (checks summary.csv internally)")
for L in LAYERS:
    for key in ("baseline", "ext_axis1", "axis1", "denial_axis1", "ext_pc3", "ext_pc1",
                "pc3", "random3"):
        mm = matrix(L, key)
        off = sum(mm[t][d] for t in DS for d in DS if t != d) / 30
        stated = float(summ[L][key]["off_diagonal_mean"])
        mark = "ok" if abs(off - stated) <= 0.0002 else "*** DIFFERS"
        print(f"  L{L} {key:<13} matrix-mean={off:.4f} summary={stated:.4f} {mark}")

# --------------------------------------------- operating points (Table 2 / 4)
hdr("D. zero_shot_operating.csv -- D-006 / D-093 / D-117 / D-167")
ops = {}
for L in LAYERS:
    for r in rows(f"{RES}/persona_ood_ext_L{L}/zero_shot_operating.csv"):
        ops[(L, r["arm"], r["dataset"], r["threshold_source"])] = r

KD = "knowledge-denial axis (4v6)"
EXT = "extended axis (12v13)"
PUB = "deception axis (8v7)"
BASE = "base probe (RepE-trained)"
HPKR = "harm_refusal_knowledge_report__plain"

print("  --- HP-KR, control threshold (1% FPR on Alpaca) ---")
for L in LAYERS:
    for arm in (PUB, KD, EXT, BASE):
        r = ops[(L, arm, HPKR, "control")]
        print(f"    L{L} {arm:<28} balacc={r['balacc']:<8} recall={r['recall']:<8} "
              f"auroc={r['auroc']:<8} spec={r['specificity']}")

hdr("D-093 / D-006: knowledge-denial axis, HP-KR, L40, control threshold")
r = ops[(40, KD, HPKR, "control")]
cmp("balanced accuracy = 0.910", 0.910, float(r["balacc"]), tol=0.0005)
cmp("recall            = 0.885", 0.885, float(r["recall"]), tol=0.0005)
spec = float(r["specificity"])
print(f"  specificity = {spec}  ->  honest flagged = {1 - spec:.4f}  (draft says '6%')")
cmp("AUROC             = 0.919", 0.919, float(r["auroc"]), tol=0.0005)
print(f"  95% CI: balacc [{r['balacc_lo']}, {r['balacc_hi']}]  recall [{r['recall_lo']}, {r['recall_hi']}]"
      f"  auroc [{r['auroc_lo']}, {r['auroc_hi']}]")
print("  [5.1 also: at L20 the same axis flags 88% of honest answers, balacc 0.533]")
r20 = ops[(20, KD, HPKR, "control")]
print(f"    L20 balacc={r20['balacc']} specificity={r20['specificity']} "
      f"-> honest flagged {1 - float(r20['specificity']):.4f}")

hdr("Table 2 balanced-accuracy column")
for L in LAYERS:
    for arm, want in ((PUB, 0.500), (KD, {20: 0.533, 40: 0.910, 44: 0.757}[L]),
                      (EXT, 0.500), (BASE, 0.500)):
        got = float(ops[(L, arm, HPKR, "control")]["balacc"])
        mark = "ok" if abs(got - want) <= 0.0005 else "*** DIFFERS"
        print(f"  L{L} {arm:<28} draft={want} file={got} {mark}")
print("  [Table 2 base-probe AUROC on HP-KR: 0.109 / 0.936 / 0.828]")
for L, v in zip(LAYERS, [0.109, 0.936, 0.828]):
    cmp(f"    base probe HP-KR AUROC L{L} = {v}", v,
        float(ops[(L, BASE, HPKR, "control")]["auroc"]))

hdr("D-117: Table 4, Average (Liars' Bench) at the control threshold")
want4 = {20: {BASE: (0.721, 0.489, 0.697), PUB: (0.534, 0.069, 0.750),
              KD: (0.559, 0.305, 0.863), EXT: (0.591, 0.187, 0.887)},
         40: {BASE: (0.626, 0.281, 0.778), PUB: (0.675, 0.382, 0.798),
              KD: (0.725, 0.487, 0.894), EXT: (0.694, 0.413, 0.899)},
         44: {BASE: (0.626, 0.281, 0.756), PUB: (0.654, 0.318, 0.791),
              KD: (0.683, 0.395, 0.880), EXT: (0.666, 0.350, 0.884)}}
bad4 = 0
for L in LAYERS:
    for arm, (ba, rc, au) in want4[L].items():
        r = ops[(L, arm, "Average (Liars' Bench)", "control")]
        g = (float(r["balacc"]), float(r["recall"]), float(r["auroc"]))
        m = ["ok" if abs(a - b) <= 0.0005 else "***" for a, b in zip((ba, rc, au), g)]
        bad4 += m.count("***")
        print(f"  L{L} {arm:<28} draft=({ba},{rc},{au}) file=({g[0]},{g[1]},{g[2]}) {m}")
print(f"  Table 4 cells checked: 36, mismatches: {bad4}")
print("  NOTE: base probe L40 and L44 balacc/recall are identical (0.626 / 0.281) in the file too.")

hdr("Base probe per-dataset at L40 vs L44 (is the identity a coincidence or a copied file?)")
for d in DS + ["Average (Liars' Bench)", "Average"]:
    a = ops[(40, BASE, d, "control")]
    b = ops[(44, BASE, d, "control")]
    print(f"  {SHORT.get(d, d):<22} L40 (ba={a['balacc']}, rec={a['recall']}, auroc={a['auroc']}, thr={a['threshold']})"
          f"  L44 (ba={b['balacc']}, rec={b['recall']}, auroc={b['auroc']}, thr={b['threshold']})")

hdr("5.4 supporting numbers")
for L, arm, d, lbl, want in [
        (20, BASE, "convince_me__plain", "base recall CG L20", 0.677),
        (20, EXT, "convince_me__plain", "ext recall CG L20", 0.174),
        (20, BASE, "insider_trading_cadenza__plain", "base spec IT L20 (draft: flags 13%)", None),
        (20, BASE, "instruct_dishonesty__plain", "base spec ID L20 (draft: flags 11%)", None),
        (20, EXT, "insider_trading_cadenza__plain", "ext spec IT L20 (draft: 0%)", None),
        (20, EXT, "instruct_dishonesty__plain", "ext spec ID L20 (draft: 2%)", None),
        (40, BASE, "insider_trading_cadenza__plain", "base AUROC IT L40 (draft 0.338)", 0.338),
        (44, BASE, "insider_trading_cadenza__plain", "base AUROC IT L44 (draft 0.289)", 0.289),
        (40, KD, "harm_refusal_choice__plain", "kd recall HP-C L40 (draft 0.015)", 0.015)]:
    r = ops[(L, arm, d, "control")]
    print(f"  {lbl:<42} balacc={r['balacc']} recall={r['recall']} auroc={r['auroc']} "
          f"spec={r['specificity']} (flags {1 - float(r['specificity']):.3f})")

hdr("E. Cross-file consistency: zero_shot.csv AUROC vs zero_shot_operating.csv AUROC")
nm = {PUB: "auroc_axis", KD: "auroc_denial_axis", EXT: "auroc_ext_axis"}
bad = 0
for L in LAYERS:
    for arm, col in nm.items():
        for d in DS:
            a = float(zs[L][d][col])
            b = float(ops[(L, arm, d, "control")]["auroc"])
            if abs(a - b) > 0.0005:
                print(f"  *** L{L} {arm} {SHORT[d]}: zero_shot {a:.4f} vs operating {b:.4f}")
                bad += 1
print(f"  54 comparisons, mismatches: {bad}")

hdr("F. Sample sizes behind the headline numbers")
for L in [40]:
    for d in DS:
        r = zs[L][d]
        print(f"  {SHORT[d]:<6} n={r['n']:<6} n_deceptive={r['n_deceptive']}")
print("  Alpaca control row:", zs[40]["XXXX-20__plain"]["n"], "rows, n_deceptive",
      zs[40]["XXXX-20__plain"]["n_deceptive"])
print("  Single run: no seed column anywhere in zero_shot.csv / summary.csv "
      "(zero-shot directions are deterministic; bootstrap CIs are over examples, 1000 resamples).")
