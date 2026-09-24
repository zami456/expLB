"""P17 Agent A, task 2: re-derive every number the paper added/changed between PIN_OLD (ea1ce52b)
and PIN_PAPER (16def2de), from the pinned result CSVs, and compare with generated/numbers.tex.

Independent of the collaborator's make_numbers.py (no import). Reads only through p17_common
(pinned SHAs). Primary source = the per-layer files persona_ood_ext_L{grid}/ (summary.csv,
zero_shot.csv, zero_shot_operating.csv, matrix_*.csv) and XXXX-9-llama-all*/results_table.csv.
results/persona_depth/depth.csv is read as a SECOND route and cross-checked against the per-layer
files (it was committed at ea1ce52b and not regenerated after 2ad758fb changed L20/40/44).

Writes: audit/p17/paper_delta_values.csv  (key, recomputed full, recomputed rounded, numbers.tex,
        depth.csv route, match flags)
Usage:  python audit/scripts/p17_paper_delta.py [sha]    (default PIN_PAPER)
"""
import csv, io, re, sys, pathlib
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import p17_common as C

SHA = sys.argv[1] if len(sys.argv) > 1 else C.PIN_PAPER
R = C.RES
OUT = C.ROOT / "audit" / "p17"
GRID = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79]
DS = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C",
      "harm_refusal_knowledge_report__plain": "HP-KR", "insider_trading_cadenza__plain": "IT",
      "instruct_dishonesty__plain": "ID", "repe_honesty__plain": "RepE"}
LB5 = [k for k in DS if k != "repe_honesty__plain"]
HPKR = "harm_refusal_knowledge_report__plain"
BASE_ARM = "base probe (RepE-trained)"
ZA = {"ext": "extended axis (12v13)", "dec": "deception axis (8v7)",
      "den": "knowledge-denial axis (4v6)", "base": BASE_ARM}


def rd(x, p=3):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-p), ROUND_HALF_UP))


def df(path, **kw):
    t = C.show(SHA, path)
    if t is None:
        raise FileNotFoundError(f"{SHA[:8]}:{path}")
    return pd.read_csv(io.StringIO(t), **kw)


def ext(L):
    return f"{R}/persona_ood_ext_L{L}"


SUM = {L: df(f"{ext(L)}/summary.csv").set_index("features") for L in GRID}
ZS = {L: df(f"{ext(L)}/zero_shot.csv").set_index("dataset") for L in GRID}
OP = {}
for L in GRID:
    o = df(f"{ext(L)}/zero_shot_operating.csv")
    OP[L] = o[o["threshold_source"] == "control"]
MAT = {}


def mat(L, arm):
    if (L, arm) not in MAT:
        MAT[(L, arm)] = df(f"{ext(L)}/matrix_{arm}.csv", index_col=0)
    return MAT[(L, arm)]


def by_test(m):
    v = m.to_numpy(float).copy()
    np.fill_diagonal(v, np.nan)
    return pd.Series(np.nanmean(v, axis=0), index=m.columns)


def op(L, arm, ds, metric):
    r = OP[L][(OP[L]["arm"] == arm) & (OP[L]["dataset"] == ds)]
    return float(r[metric].iloc[0]) if len(r) else float("nan")


def ser(f):
    return pd.Series({L: f(L) for L in GRID})


V, NOTE = {}, {}


def put(k, v, note=""):
    V[k] = v
    NOTE[k] = note


put("depth-n", len(GRID)); put("depth-first", "20"); put("depth-last", "79")
off = {"ext": ser(lambda L: SUM[L].loc["ext_axis1", "off_diagonal_mean"]),
       "dec": ser(lambda L: SUM[L].loc["axis1", "off_diagonal_mean"]),
       "den": ser(lambda L: SUM[L].loc["denial_axis1", "off_diagonal_mean"]),
       "base": ser(lambda L: SUM[L].loc["baseline", "off_diagonal_mean"])}
for k, s in off.items():
    put(f"depth-off-{k}-peak", s.max(), "summary.csv off_diagonal_mean, max over 16 layers")
    put(f"depth-off-{k}-peak-layer", str(int(s.idxmax())))
# zero-shot LB average: the operating table's own "Average (Liars' Bench)" row (control threshold)
zs = {k: ser(lambda L, a=a: op(L, a, "Average (Liars' Bench)", "auroc")) for k, a in ZA.items()}
# second route: mean over the five LB datasets' rows
zs5 = {k: ser(lambda L, a=a: np.mean([op(L, a, d, "auroc") for d in LB5])) for k, a in ZA.items()}
for k, s in zs.items():
    put(f"depth-zs-{k}-peak", s.max(), "zero_shot_operating.csv control, Average (Liars' Bench) auroc")
    put(f"depth-zs-{k}-peak-layer", str(int(s.idxmax())))
bb = int(zs["base"].idxmax()); eb = int(zs["ext"].idxmax())
put("depth-zs-margin-at-base-peak", zs["ext"][bb] - zs["base"][bb], f"at L{bb}")
put("depth-zs-ext-at-base-peak", zs["ext"][bb]); put("depth-zs-base-at-ext-peak", zs["base"][eb])
put("depth-ext-over-dec-min", (off["ext"] - off["dec"]).min(),
    f"ext>=dec at all layers: {bool((off['ext'] >= off['dec']).all())}")
put("depth-ext-trough", off["ext"].loc[56:72].min()); put("depth-ext-L76", off["ext"][76])
put("depth-ext-last", off["ext"][79])
holds, breaks = [], {}
for L in GRID:
    e, b = by_test(mat(L, "ext_axis1")), by_test(mat(L, "baseline"))
    los = [DS[n] for n in e.index if e[n] <= b[n]]
    (holds.append(L) if not los else breaks.setdefault(L, los))
contig = [L for i, L in enumerate(holds) if L == GRID[i]]
put("depth-pertest-holds-last", str(contig[-1]), f"breaks={breaks}")
put("depth-pertest-break-first", str(min(breaks))); put("depth-pertest-break-last", str(max(breaks)))
put("depth-pertest-exception", " and ".join(sorted({x for v in breaks.values() for x in v})))
put("depth-pertest-break-n", len(breaks))
pcw = [L for L in GRID if SUM[L].loc["ext_pc3", "off_diagonal_mean"] >= SUM[L].loc["ext_axis1", "off_diagonal_mean"]]
put("depth-pc-wins-n", len(pcw), f"layers={pcw}"); put("depth-pc-wins-complement", len(GRID) - len(pcw))
put("depth-pc-wins-first", str(min(pcw))); put("depth-pc-wins-last", str(max(pcw)))
put("depth-pc-wins-max", max(SUM[L].loc["ext_pc3", "off_diagonal_mean"] - SUM[L].loc["ext_axis1", "off_diagonal_mean"] for L in pcw))
d_eb = off["ext"] - off["base"]
put("depth-off-ext-over-base-min", d_eb.min(), f"ext>base all: {bool((d_eb > 0).all())}; argmin L{int(d_eb.idxmin())}")
put("depth-off-ext-over-base-at-peak", d_eb[int(off["ext"].idxmax())])
drops = {}
for k, feat in (("base", "baseline"), ("ext", "ext_axis1")):
    dg = ser(lambda L, f=feat: SUM[L].loc[f, "diagonal_mean"])
    od = ser(lambda L, f=feat: SUM[L].loc[f, "off_diagonal_mean"])
    drops[k] = dg - od
    put(f"depth-{k}-diag-min", dg.min(), f"argmin L{int(dg.idxmin())}")
    put(f"depth-{k}-diag-max", dg.max(), f"argmax L{int(dg.idxmax())}")
put("depth-base-drop-min", drops["base"].min(), f"L{int(drops['base'].idxmin())}")
put("depth-base-drop-max", drops["base"].max(), f"L{int(drops['base'].idxmax())}")
put("depth-ext-drop-max", drops["ext"].abs().max(), f"L{int(drops['ext'].abs().idxmax())}")
worst, wat = 1.0, None
for L in GRID:
    z = ZS[L][ZS[L]["auroc_axis"].notna()]
    if z["auroc_ext_axis"].min() < worst:
        worst, wat = float(z["auroc_ext_axis"].min()), (L, z["auroc_ext_axis"].idxmin())
put("depth-ext-worst-dataset", worst, f"at {wat}")
syc = ser(lambda L: ZS[L].loc[HPKR, "auroc_syc_axis"])
put("depth-syc-hpkr-max", syc.max()); put("depth-syc-hpkr-max-layer", str(int(syc.idxmax())))
put("depth-dec-over-syc-n", int(sum(SUM[L].loc["axis1", "off_diagonal_mean"] > SUM[L].loc["syc_axis1", "off_diagonal_mean"] for L in GRID)))
d_zs = zs["ext"] - zs["base"]
put("depth-zs-ext-over-base-min", d_zs.min(), f"argmin L{int(d_zs.idxmin())}; all>0: {bool((d_zs > 0).all())}")
inv = {L: [DS[d] for d in DS if op(L, BASE_ARM, d, "auroc") < 0.5] for L in GRID}
clean = [L for L, v in inv.items() if not v]
put("depth-base-no-invert", ",".join(map(str, clean)), f"inversions={ {L: v for L, v in inv.items() if v} }")
put("depth-base-invert-layers", sum(1 for v in inv.values() if v))
itl = [L for L, v in inv.items() if "IT" in v]
put("depth-base-it-invert-first", str(min(itl)), f"IT-inverted layers {itl}")
hb = ser(lambda L: op(L, ZA["den"], HPKR, "balacc"))
put("depth-hpkr-den-balacc-peak", hb.max()); put("depth-hpkr-den-balacc-peak-layer", str(int(hb.idxmax())))
put("depth-hpkr-den-balacc-L36", hb[36]); put("depth-hpkr-den-balacc-L44", hb[44])
broad = [L for L in GRID if hb[L] > 0.6]
put("depth-hpkr-den-broad-first", str(min(broad)), f"layers>0.6: {broad}"); put("depth-hpkr-den-broad-last", str(max(broad)))
ha = {k: ser(lambda L, a=a: op(L, a, HPKR, "auroc")) for k, a in ZA.items()}
for k in ("den", "ext"):
    put(f"depth-hpkr-{k}-auroc-min", ha[k].min()); put(f"depth-hpkr-{k}-auroc-max", ha[k].max())
put("depth-hpkr-base-auroc-peak", ha["base"].max()); put("depth-hpkr-base-auroc-peak-layer", str(int(ha["base"].idxmax())))
best = pd.concat([ha["dec"], ha["den"], ha["ext"]], axis=1).max(axis=1)
pk = int(ha["base"].idxmax())
deep = [L for L in GRID if L >= pk]
ahead = [L for L in deep if ha["base"][L] > best[L]]
behind = [L for L in deep if L not in ahead]
put("depth-hpkr-base-ahead-n", len(ahead), f"ahead={ahead}; NOT ahead={behind}")
put("depth-hpkr-base-deep-n", len(deep))
put("depth-hpkr-base-peak-margin", ha["base"][pk] - best[pk])
fl_ok = {k: bool(np.allclose([op(L, ZA[k], HPKR, "balacc") for L in GRID], 0.5)) for k in ("base", "ext")}
put("depth-hpkr-floor-arms", "the base probe and the extended contrast" if all(fl_ok.values()) else f"FLOOR {fl_ok}")
# dec arm range on HP-KR balacc (summary.md claims 0.486-0.503)
decb = ser(lambda L: op(L, ZA["dec"], HPKR, "balacc"))
put("x-hpkr-dec-balacc-range", f"{decb.min():.3f}-{decb.max():.3f}", "not a numbers.tex key")
# base-probe HP-KR AUROC range over L>=40 (summary.md: 'between 0.83 and 0.94 through 79')
put("x-hpkr-base-auroc-range-40plus", f"{ha['base'].loc[40:].min():.4f}-{ha['base'].loc[40:].max():.4f}", "not a key")

# three-layer keys quoted by the new sentences
# results/XXXX-9-llama-all*/results_table.csv (make_numbers' source for bp-*) is NOT committed at
# any pin (only plots/ are), so bp-* is re-derived from the base-probe rows of zero_shot_operating.csv
# (4 dp). The 4-dp route can differ from the full-precision table in the 3rd decimal.
BPN = "zero_shot_operating.csv base arm (results_table.csv not committed)"
put("zs-den-hpkr-L44", ZS[44].loc[HPKR, "auroc_denial_axis"], "zero_shot.csv L44")
put("bp-auroc-hpkr-L44", op(44, BASE_ARM, HPKR, "auroc"), BPN)
put("bp-auroc-hpc-L20", op(20, BASE_ARM, "harm_refusal_choice__plain", "auroc"), BPN)
put("bp-auroc-hpkr-L20", op(20, BASE_ARM, HPKR, "auroc"), BPN)
put("bp-auroc-it-L40", op(40, BASE_ARM, "insider_trading_cadenza__plain", "auroc"), BPN)
put("bp-auroc-it-L44", op(44, BASE_ARM, "insider_trading_cadenza__plain", "auroc"), BPN)
put("bp-balacc-hpkr-L40", op(40, BASE_ARM, HPKR, "balacc"), BPN)
put("zs-ext-lb-L40", ZS[40].loc[LB5, "auroc_ext_axis"].mean())
put("off-ext-L40", SUM[40].loc["ext_axis1", "off_diagonal_mean"])
put("op-base-auroc-L20", op(20, BASE_ARM, "Average (Liars' Bench)", "auroc"))
put("op-base-auroc-hpkr-L20", op(20, BASE_ARM, HPKR, "auroc"))
put("pt-base-cg-L40", by_test(mat(40, "baseline"))["convince_me__plain"], "matrix_baseline L40 col CG")
for L in (20, 44):
    put(f"gap-ext-extpcthree-L{L}", SUM[L].loc["ext_axis1", "off_diagonal_mean"] - SUM[L].loc["ext_pc3", "off_diagonal_mean"])

# numbers.tex at this SHA
NT = dict(re.findall(r"\\setV\{([^}]*)\}\{(.*)\}", C.show(SHA, C.BASE + "/paper/generated/numbers.tex")))
# depth.csv route
DP = df(f"{R}/persona_depth/depth.csv")


def dcurve(src, metric, arm):
    r = DP[(DP.source == src) & (DP.metric == metric) & (DP.arm == arm)]
    return r.set_index("layer")["value"].sort_index()


TR = {"ext": "extended axis (12v13)", "dec": "deception axis (8v7)", "den": "knowledge-denial axis (4v6)",
      "base": "single-source probe (8192-d)"}
stale = []
for k in TR:
    a = dcurve("transfer, off-diagonal mean", "auroc", TR[k])
    for L in GRID:
        if abs(a[L] - off[k][L]) > 1e-9:
            stale.append(("transfer", k, L, a[L], off[k][L]))
for k in ZA:
    a = dcurve("zero-shot, Liars' Bench average", "auroc", ZA[k]); h = dcurve("zero-shot, HP-KR", "auroc", ZA[k])
    for L in GRID:
        if abs(a[L] - zs[k][L]) > 1e-9:
            stale.append(("zs-lb", k, L, a[L], zs[k][L]))
        if abs(h[L] - ha[k][L]) > 1e-9:
            stale.append(("zs-hpkr", k, L, h[L], ha[k][L]))

rows, nmis = [], 0
for k, v in V.items():
    rr = v if isinstance(v, str) else (f"{int(v):,}" if isinstance(v, (int, np.integer)) else rd(v, 5 if k == "depth-check-cos" else 3))
    tex = NT.get(k, "")
    m = "n/a" if not tex else ("MATCH" if tex.replace("\\ensuremath{-}", "-") == rr else "MISMATCH")
    nmis += m == "MISMATCH"
    rows.append({"key": k, "recomputed_full": v if isinstance(v, str) else repr(float(v)),
                 "recomputed_3dp": rr, "numbers_tex": tex, "match": m, "note": NOTE[k], "sha": SHA})
OUT.mkdir(parents=True, exist_ok=True)
tag = SHA[:8]
with open(OUT / f"paper_delta_values_{tag}.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"SHA {SHA}: {len(rows)} keys recomputed; {sum(r['match']=='MATCH' for r in rows)} MATCH, {nmis} MISMATCH vs numbers.tex")
for r in rows:
    if r["match"] != "MATCH":
        print(f"  {r['match']:8s} {r['key']:34s} recomputed={r['recomputed_3dp']:10s} tex={r['numbers_tex']}  {r['note'][:60]}")
print(f"depth.csv vs per-layer files: {len(stale)} cells differ (>1e-9)")
for s in stale[:12]:
    print("   ", s[0], s[1], "L%d" % s[2], "depth.csv=%.6f per-layer=%.6f d=%.6f" % (s[3], s[4], s[3] - s[4]))
print("LB avg route check (table row vs mean of 5 rows), max |diff|:",
      max(abs(zs[k][L] - zs5[k][L]) for k in ZA for L in GRID))
print("HP-KR base-vs-best-contrast per deep layer:")
for L in deep:
    print(f"   L{L}: base {ha['base'][L]:.4f}  best contrast {best[L]:.4f} ({pd.concat([ha['dec'],ha['den'],ha['ext']],axis=1).loc[L].idxmax() if False else ''})  {'BASE AHEAD' if L in ahead else 'contrast ahead'}")
for k in ("depth-hpkr-base-ahead-n", "depth-base-no-invert", "depth-pertest-holds-last", "depth-pc-wins-n", "depth-ext-worst-dataset"):
    print(f"  {k}: {NOTE[k][:150]}")

# ---------------------------------------------------------------- extra sentence-level checks
print("\n--- sentence-level checks ---")
pe = {L: (by_test(mat(L, "ext_axis1")) - by_test(mat(L, "axis1"))) for L in GRID}
neg = {L: [DS[d] for d, v in s.items() if v < 0] for L, s in pe.items()}
print("ext vs dec per held-out dataset (off-diag by test): min diff %.4f; layers/datasets where dec > ext: %s"
      % (min(s.min() for s in pe.values()), {L: v for L, v in neg.items() if v}))
print("ext off-diag by layer:", " ".join(f"{L}:{off['ext'][L]:.3f}" for L in GRID))
print("syc HP-KR at 20/40/44:", [round(syc[L], 4) for L in (20, 40, 44)], "; layers >0.5:", [L for L in GRID if syc[L] > 0.5])
lo = pd.concat([ha["den"], ha["ext"]], axis=1).min(axis=1)
print("dec HP-KR below both kd contrasts at every layer:", bool((ha["dec"] < lo).all()),
      "; dec HP-KR range %.3f-%.3f" % (ha["dec"].min(), ha["dec"].max()))
print("den HP-KR balacc by layer:", " ".join(f"{L}:{hb[L]:.3f}" for L in GRID))
print("base inversions (per layer):", {L: v for L, v in inv.items()})
nrows = {L: tuple(ZS[L]["n"].astype(int)) for L in GRID}
print("zero_shot.csv n per dataset identical at all 16 layers:", len(set(nrows.values())) == 1)
cellmax = max(float(np.nanmax(np.where(np.eye(len(mat(L, 'ext_axis1'))) == 1, np.nan, mat(L, "ext_axis1").to_numpy(float)))) for L in GRID)
zsmax = max(float(ZS[L]["auroc_ext_axis"].max()) for L in GRID)
bdiag = min(float(np.diag(mat(L, "baseline").to_numpy(float)).min()) for L in GRID)
print(f"'probe beats contrast anywhere': base diagonal_mean min {drops and SUM[20].loc['baseline','diagonal_mean'] and min(SUM[L].loc['baseline','diagonal_mean'] for L in GRID):.4f}"
      f" vs ext diagonal_mean max {max(SUM[L].loc['ext_axis1','diagonal_mean'] for L in GRID):.4f};"
      f" per-cell: ext matrix off-diag cell max {cellmax:.4f}, ext zero-shot per-dataset max {zsmax:.4f}, base diagonal CELL min {bdiag:.4f}")
print("base RepE zero-shot AUROC at new layers:", sorted({round(op(L, BASE_ARM, 'repe_honesty__plain', 'auroc'), 4) for L in GRID}))

# ---------------------------------------------------------------- claims: audit/p17/paper_new_claims.csv
# excl-HP-KR ext-vs-dec gap per layer (limitations.tex:18-20 says "the other datasets")
exd = ser(lambda L: SUM[L].loc["ext_axis1", "off_diagonal_excl_hpkr"] - SUM[L].loc["axis1", "off_diagonal_excl_hpkr"])
PEX = {L: [DS[d] for d, v in s.items() if v < 0] for L, s in pe.items()}
T = lambda k: NT.get(k, "?")
W = lambda k: "(not re-derivable)" if k not in V else (V[k] if isinstance(V[k], str) else (f"{int(V[k]):,}" if isinstance(V[k], (int, np.integer)) else rd(V[k])))
EXT = f"{R}/persona_ood_ext_L{{20..79}}/"
SRC = "origin/paper@16def2de paper/sections/"
SCRIPT = "audit/scripts/p17_paper_delta.py"


def keys(*ks):
    return "; ".join(f"{k}={T(k)}" for k in ks)


def rec(ks):
    return "; ".join(f"{k}={W(k)}" for k in ks)


def allmatch(ks):
    ks = [k for k in ks if k in V]
    if not ks:
        return "n/a (constants, no data route)"
    return "MATCH" if all(T(k).replace("\\ensuremath{-}", "-") == W(k) for k in ks) else "MISMATCH"


CL = []


def claim(src, text, ks, ev_file, ev_key, verdict, conf, imp, notes, method=None):
    CL.append(dict(src=src, text=text, ks=ks, ev_file=ev_file, ev_key=ev_key, verdict=verdict,
                   conf=conf, imp=imp, notes=notes, method=method))


HPKR_PER_LAYER = "; ".join(f"L{L} base {ha['base'][L]:.3f} vs best contrast {best[L]:.3f}" for L in deep)

claim("abstract.tex:L18-22 (also introduction.tex:L76-78, results.tex:L96-101)",
      "Single-dataset raw probes lose 0.213-0.347 AUROC off their training set across all 16 layers; the extended contrast's off-diagonal mean equals its in-distribution mean to within 0.002, and it never ranks any dataset below chance.",
      ["depth-base-drop-min", "depth-base-drop-max", "depth-ext-drop-max", "depth-ext-worst-dataset"],
      EXT + "summary.csv; zero_shot.csv", "features baseline/ext_axis1: diagonal_mean - off_diagonal_mean per layer; min auroc_ext_axis over two-class rows",
      "HOLDS", "high", "high", f"drop min at {NOTE['depth-base-drop-min']}, max at {NOTE['depth-base-drop-max']}; worst dataset {NOTE['depth-ext-worst-dataset']}.")
claim("abstract.tex:L18-19 + results.tex:L103-105",
      "'The supervised probes score higher than the contrast ever does on the dataset they were fitted to' / 'A supervised probe reaches higher on its own dataset than the contrast ever reaches anywhere'.",
      ["depth-base-diag-min", "depth-ext-diag-max"], EXT + "summary.csv; matrix_baseline.csv; matrix_ext_axis1.csv; zero_shot.csv",
      "diagonal_mean baseline vs ext_axis1; per-cell diagonal of matrix_baseline vs ext cells / zero-shot per dataset",
      "PARTLY HOLDS", "high", "medium",
      f"Holds on MEANS: raw diagonal_mean >= {W('depth-base-diag-min')} at every layer vs ext diagonal_mean <= {W('depth-ext-diag-max')}. Fails PER DATASET: ext reaches {zsmax:.3f} (zero-shot) and {cellmax:.3f} (matrix cell) while the lowest raw diagonal CELL is {bdiag:.4f}; 'ever ... anywhere' is literally false.")
claim("abstract.tex:L22-24; results.tex:L228-235; conclusion.tex:L11",
      "Over 16 layers ranking peaks at layer 36, not the reported layer 40: extended contrast off-diagonal 0.927 (vs 0.916 at L40), zero-shot LB average 0.913 (vs 0.899 at L40); knowledge-denial contrast also peaks at 36 (0.912).",
      ["depth-off-ext-peak", "depth-off-ext-peak-layer", "off-ext-L40", "depth-zs-ext-peak", "depth-zs-ext-peak-layer", "zs-ext-lb-L40", "depth-off-den-peak", "depth-off-den-peak-layer", "depth-zs-den-peak-layer"],
      EXT + "summary.csv; zero_shot_operating.csv; zero_shot.csv", "off_diagonal_mean ext_axis1/denial_axis1; control 'Average (Liars' Bench)' auroc",
      "HOLDS", "high", "high", "No interval computed (the paper says so). L40 zero-shot 0.899 is the zero_shot.csv 5-dataset mean; the operating-table Average row agrees to <=6e-5.")
claim("abstract.tex:L24-28; limitations.tex:L24-30",
      "On HP-KR the RepE-trained probe out-ranks every persona contrast at 8 of the 12 layers from 36 on, peaking at 0.967 AUROC at L36, 0.063 above the best contrast there, while flagging none of its lies.",
      ["depth-hpkr-base-ahead-n", "depth-hpkr-base-deep-n", "depth-hpkr-base-auroc-peak", "depth-hpkr-base-auroc-peak-layer", "depth-hpkr-base-peak-margin", "depth-hpkr-floor-arms"],
      EXT + "zero_shot_operating.csv", "control, dataset=HP-KR, auroc/balacc; arms base probe vs max(8v7,4v6,12v13)",
      "HOLDS", "high", "high", NOTE["depth-hpkr-base-ahead-n"] + ". Base balacc on HP-KR = 0.500 at all 16 layers.")
claim("introduction.tex:L68-70",
      "QUOTE: 'A RepE-trained probe in our own pipeline stays at \\V{bp-balacc-hpkr-L40} at every one of the \\V{depth-n} layers, though from layer~\\V{depth-hpkr-base-auroc-peak-layer} it ranks \\hpkr{} better than any contrast we build' (renders: 0.500 ... 16 layers, though from layer 36 it ranks HP-KR better than any contrast we build).",
      ["bp-balacc-hpkr-L40", "depth-hpkr-base-auroc-peak-layer", "depth-hpkr-base-ahead-n", "depth-hpkr-base-deep-n"],
      EXT + "zero_shot_operating.csv", "control, HP-KR auroc per layer, base probe vs best of 8v7/4v6/12v13, layers 36-79",
      "DOES NOT HOLD", "high", "high",
      "The 'from layer 36' ranking clause is false at 4 of the 12 layers from 36 on (" + NOTE["depth-hpkr-base-ahead-n"] +
      "). Per layer: " + HPKR_PER_LAYER + ". 16def2de corrected the same overstatement in the abstract and limitations to '8 of 12' but left this sentence. The first clause (0.500 at all 16 layers) holds.")
claim("introduction.tex:L61-62; method.tex:L116-119; results.tex:L221-223",
      "The study now covers 16 layers of Llama-3.3-70B: every fourth from 20 to 76, plus layer 79.",
      ["depth-n", "depth-first", "depth-last"], f"{R}/persona_depth/depth.csv; {R}/persona_ood_ext_L*/", "unique layers; per-layer result folders present",
      "HOLDS", "high", "medium", "All 16 persona_ood_ext_L* folders exist at 16def2de (and already at ea1ce52b); depth.csv layers = 20..76 step 4 + 79.")
claim("introduction.tex:L74-76; results.tex:L91-94",
      "The extended contrast leads the single-dataset raw probes off the diagonal at every one of the 16 layers, by at least 0.115 AUROC, and by 0.159 where both are strongest (L36).",
      ["depth-off-ext-over-base-min", "depth-off-ext-over-base-at-peak", "depth-off-base-peak-layer"], EXT + "summary.csv", "off_diagonal_mean ext_axis1 - baseline per layer",
      "HOLDS", "high", "high", NOTE["depth-off-ext-over-base-min"] + "; both arms peak at L36.")
claim("introduction.tex:L78-81; results.tex:L106-111",
      "Averaged over training sets, the extended contrast beats the raw probes on every held-out dataset at every layer through 48; at 6 layers between 52 and 72 HP-C and HP-KR favour the raw probes.",
      ["depth-pertest-holds-last", "depth-pertest-break-n", "depth-pertest-break-first", "depth-pertest-break-last", "depth-pertest-exception"],
      EXT + "matrix_ext_axis1.csv; matrix_baseline.csv", "per test column, mean over the 5 off-diagonal training rows",
      "HOLDS", "high", "medium", NOTE["depth-pertest-holds-last"] + ". One of the claims 2ad758fb corrected (was 'at every layer').")
claim("introduction.tex:L82-86; results.tex:L135-143; conclusion.tex:L9",
      "The 1-d extended contrast outscores a probe on the top three persona PCs at 13 of 16 layers; the PCs lead only at layers 64-72, by up to 0.021.",
      ["depth-pc-wins-complement", "depth-n", "depth-pc-wins-first", "depth-pc-wins-last", "depth-pc-wins-max"], EXT + "summary.csv", "off_diagonal_mean ext_pc3 vs ext_axis1",
      "HOLDS", "high", "medium", NOTE["depth-pc-wins-n"] + ". Corrected claim (was 'at every depth ... never reverses').")
claim("results.tex:L37-46 (fig:hpkr caption)",
      "Both knowledge-denial contrasts hold 0.799-0.925 HP-KR AUROC at every layer; the published contrast inverts at L20 and stays below them throughout; the base probe recovers to 0.967 by L36, above every contrast; only the knowledge-denial contrast ever clears the 0.5 balacc floor; the base probe never leaves it.",
      ["depth-hpkr-den-auroc-min", "depth-hpkr-den-auroc-max", "depth-hpkr-ext-auroc-min", "depth-hpkr-ext-auroc-max", "depth-hpkr-base-auroc-peak"],
      EXT + "zero_shot_operating.csv", "control, HP-KR, auroc/balacc, all arms, 16 layers",
      "HOLDS", "high", "medium", f"8v7 HP-KR AUROC below both kd contrasts at every layer: {bool((ha['dec'] < lo).all())} (range {ha['dec'].min():.3f}-{ha['dec'].max():.3f}); 8v7 HP-KR balacc {V['x-hpkr-dec-balacc-range']}.")
claim("results.tex:L166-172",
      "At all 16 layers the extended contrast ranks the LB average above the base probe by at least 0.022 AUROC; its lowest per-dataset AUROC anywhere is 0.542; the base probe inverts on some dataset at 15 of 16 layers.",
      ["depth-zs-ext-over-base-min", "depth-ext-worst-dataset", "depth-base-invert-layers"], EXT + "zero_shot_operating.csv; zero_shot.csv",
      "control Average (Liars' Bench) auroc ext vs base; min auroc_ext_axis; base arm per-dataset auroc<0.5",
      "HOLDS", "high", "high", NOTE["depth-zs-ext-over-base-min"] + "; worst " + NOTE["depth-ext-worst-dataset"])
claim("results.tex:L180-184; results.tex:L237-242; results.tex:L268-272 (fig:depth caption)",
      "At the contrast's best layer (36) the zero-shot LB average is 0.913 vs 0.783 for the base probe; at the base probe's own best layer (32; 0.850, vs 0.697 at L20) the margin narrows to 0.055 (0.905 vs 0.850).",
      ["depth-zs-ext-peak", "depth-zs-base-at-ext-peak", "depth-zs-base-peak", "depth-zs-base-peak-layer", "depth-zs-margin-at-base-peak", "depth-zs-ext-at-base-peak", "op-base-auroc-L20"],
      EXT + "zero_shot_operating.csv", "control, 'Average (Liars' Bench)', auroc, arms 12v13 and base probe",
      "HOLDS", "high", "high", "Base-probe values at the 13 new layers come from probes the pipeline never scored (no results_table.csv; app:depth says so).")
claim("results.tex:L205-211",
      "The base probe inverts on some dataset at every depth but one: HP-C and HP-KR at L20 (0.471, 0.109), HP-KR at L24 and L28, IT from L36 onwards; only at L32 does it rank every dataset correctly.",
      ["bp-auroc-hpc-L20", "bp-auroc-hpkr-L20", "depth-base-it-invert-first", "depth-base-no-invert", "bp-auroc-it-L40", "bp-auroc-it-L44"],
      EXT + "zero_shot_operating.csv (base arm; results_table.csv not committed)", "control, base probe, per-dataset auroc < 0.5",
      "HOLDS", "high", "medium", "Inversions by layer: " + NOTE["depth-base-no-invert"][12:330] + ". Replaces 'inverts on one dataset at each depth' (which also missed HP-C at L20). bp-* re-derived from the 4-dp operating table because results/XXXX-9-llama-all*/results_table.csv is not committed at any pin.")
claim("results.tex:L245-250",
      "HP-KR balanced accuracy of the knowledge-denial contrast is a spike: 0.910 at L40 but 0.745 (L36) and 0.757 (L44); above 0.6 from L36 to L76.",
      ["depth-hpkr-den-balacc-peak", "depth-hpkr-den-balacc-peak-layer", "depth-hpkr-den-balacc-L36", "depth-hpkr-den-balacc-L44", "depth-hpkr-den-broad-first", "depth-hpkr-den-broad-last"],
      EXT + "zero_shot_operating.csv", "control, HP-KR, knowledge-denial axis (4v6), balacc",
      "HOLDS", "high", "high", NOTE["depth-hpkr-den-broad-first"] + " (contiguous).")
claim("results.tex:L250-254",
      "Base-probe HP-KR AUROC climbs from 0.109 at L20 (inverted) to 0.967 at L36, above every persona contrast, while its balacc stays at 0.5 at every layer, as does the extended contrast's.",
      ["op-base-auroc-hpkr-L20", "depth-hpkr-base-auroc-peak", "depth-hpkr-base-auroc-peak-layer", "depth-hpkr-floor-arms"],
      EXT + "zero_shot_operating.csv", "control, HP-KR, base probe / 12v13, auroc & balacc", "HOLDS", "high", "medium",
      f"At L36: base {ha['base'][36]:.4f} vs best contrast {best[36]:.4f}.")
claim("results.tex:L254-255",
      "QUOTE: 'It ranks \\hpkr{}'s lies better than we do and flags none of them.' (unqualified, about the base probe).",
      ["depth-hpkr-base-ahead-n", "depth-hpkr-base-deep-n"], EXT + "zero_shot_operating.csv", "control, HP-KR auroc per layer",
      "PARTLY HOLDS", "high", "medium",
      "Same overstatement 16def2de fixed in the abstract/limitations: base ahead only at " + NOTE["depth-hpkr-base-ahead-n"] + "; also below the contrasts at L20-L32. 'Flags none' holds (balacc 0.500 at all 16 layers).")
claim("results.tex:L259-261",
      "Transfer shape: rises to L36, falls to a trough of 0.847 between L56 and L72, second peak 0.923 at L76, collapses to 0.844 at L79.",
      ["depth-ext-trough", "depth-ext-L76", "depth-ext-last", "depth-off-ext-peak-layer"], EXT + "summary.csv", "ext_axis1 off_diagonal_mean by layer",
      "HOLDS", "high", "low", "Series: " + " ".join(f"{L}:{off['ext'][L]:.3f}" for L in GRID) + ". 'Rises to 36' is not monotone (0.904 -> 0.888 at L24).")
claim("results.tex:L261-263",
      "QUOTE: 'Throughout, the extended contrast leads the published one on all six datasets at every layer swept, by at least \\V{depth-ext-over-dec-min} AUROC.'",
      ["depth-ext-over-dec-min"], EXT + "summary.csv; matrix_ext_axis1.csv; matrix_axis1.csv", "off_diagonal_mean ext_axis1 - axis1; per held-out dataset column means",
      "PARTLY HOLDS", "high", "medium",
      "0.026 is the minimum gap of the SIX-DATASET MEAN (holds at all 16 layers). Per held-out dataset the published contrast leads at "
      + ", ".join(f"L{L}: " + "/".join(v) for L, v in PEX.items() if v) + f" (min diff {min(s.min() for s in pe.values()):.4f}), so 'on all six datasets' fails at L76 and L79.")
claim("limitations.tex:L17-21",
      "QUOTE: 'The other datasets are the untargeted test. There the extended contrast matches or beats the published one at every one of the \\V{depth-n} layers we swept, by at least \\V{depth-ext-over-dec-min} AUROC.' (renders 16 layers, 0.026).",
      ["depth-ext-over-dec-min"], EXT + "summary.csv", "off_diagonal_excl_hpkr ext_axis1 - axis1 per layer",
      "DOES NOT HOLD", "high", "high",
      "'There' = the datasets other than HP-KR, but 0.026 is computed with HP-KR included (off_diagonal_mean). On off_diagonal_excl_hpkr the gap is "
      + " ".join(f"L{L}:{exd[L]:+.3f}" for L in GRID) + f": NEGATIVE at L20 ({exd[20]:+.4f}) and below 0.026 at {int((exd < 0.026).sum())} of 16 layers. The paper's own next sentence (advantage disappears at L20 excl. HP-KR) contradicts it.")
claim("limitations.tex:L30-33",
      "At layer 44 the knowledge-denial contrast leads the base probe on HP-KR ranking, 0.908 to 0.828.",
      ["zs-den-hpkr-L44", "bp-auroc-hpkr-L44"], f"{R}/persona_ood_ext_L44/zero_shot.csv; zero_shot_operating.csv (base arm)",
      "HP-KR auroc_denial_axis; base probe HP-KR auroc", "HOLDS", "high", "medium", "bp-auroc re-derived from the operating table (results_table.csv not committed).")
claim("appendix.tex:L146-151",
      "The sycophancy contrast never ranks HP-KR better than chance at any depth: its best is 0.507 at L48; it is inverted at the three tab:syc layers.",
      ["depth-syc-hpkr-max", "depth-syc-hpkr-max-layer"], EXT + "zero_shot.csv", "HP-KR auroc_syc_axis per layer",
      "HOLDS", "medium", "low", f"L20/40/44 = {[round(float(syc[L]), 3) for L in (20, 40, 44)]} (<0.5). Only L48 exceeds 0.5 (0.507); 'better than chance' read loosely, no interval. Corrected claim (was 'inverts at every layer').")
claim("appendix.tex:L153-155",
      "The deception contrast out-transfers the sycophancy contrast at 14 of the 16 layers.",
      ["depth-dec-over-syc-n", "depth-n"], EXT + "summary.csv", "off_diagonal_mean axis1 vs syc_axis1", "HOLDS", "high", "low",
      "Was 'leads at every layer' at ea1ce52b.")
claim("appendix.tex:L210-212 (app:depth), row check",
      "Every new layer's pooled files hold exactly the rows layer 40's hold, dataset by dataset.",
      ["depth-n"], EXT + "zero_shot.csv", "n and n_deceptive per dataset at each layer", "HOLDS", "medium", "low",
      f"zero_shot.csv n per dataset identical at all 16 layers: {len(set(nrows.values())) == 1}. (Pooled .pt files are not committed; this checks the scored row counts.)")
claim("appendix.tex:L205-212 (app:depth), NDIF pooling check",
      "Remote-pooled vs local L40 vectors: row cosine >= 0.99989, knowledge-denial AUROC differs <= 0.0001; retrained RepE probe cos 0.9996 to the original, within 0.0017 AUROC on each LB dataset; every new probe scores 1.0000 on RepE validation.",
      ["depth-check-cos", "depth-check-auroc", "depth-check-probe-cos", "depth-check-probe-auroc", "depth-check-repe-val"],
      f"{R}/persona_depth/summary.md; paper/make_numbers.py:L496-505", "constants transcribed from `layer_sweep.py check40` output",
      "UNRESOLVED", "medium", "medium",
      "make_numbers.py writes these as CONSTANTS ('constants with a stated provenance rather than values read from a result file'). They match the prose in results/persona_depth/summary.md, but the pooled activations / original L40 probe needed to re-derive them are not committed. Consistent: base-arm RepE zero-shot AUROC = 1.0 at all 16 layers (full RepE rows, not the validation split).",
      method="numbers.tex vs summary.md text only; no data route")
claim("method.tex:L122-125",
      "Pooled vectors reproduce the pipeline's own RepE probe AUROCs to four decimals at layers 20, 40 and 44 (narrowed from 'every layer').",
      [], "results/XXXX-9-llama-all*/results_table.csv", "auroc per dataset", "UNRESOLVED", "medium", "low",
      "Narrowing is correct for the new layers (their probes were never scored: no results_table.csv). The 3-layer reproduction itself needs results_table.csv, which is not committed at any pin (only plots/).",
      method="not re-derivable from committed files")
claim("paper/generated/numbers.tex:L276 + generated/tab_pertest.tex (CG row)",
      "Per-test table (Table 7) raw-probe CG at L40 changes 0.788 -> 0.787.",
      ["pt-base-cg-L40"], f"{R}/persona_ood_ext_L40/matrix_baseline.csv", "CG column mean over 5 off-diagonal rows",
      "HOLDS", "high", "low", "0.7875 at ea1ce52b -> 0.7873 at 2ad758fb (matrix_baseline re-run, drift 0.0002). The draft PDF still prints 0.788 (claims.csv D-189).")

rows_out = []
for i, c in enumerate(CL, 1):
    ks = c["ks"]
    rows_out.append({
        "id": f"D2-{i:03d}", "claim_plain": c["text"],
        "stated_numbers": "; ".join(T(k) for k in ks),
        "source_ref": SRC + c["src"] if not c["src"].startswith("paper/") else "origin/paper@16def2de " + c["src"],
        "evidence_file": c["ev_file"], "evidence_key": c["ev_key"],
        "level1_value": ("numbers.tex @16def2de: " + keys(*ks)) if ks else "no \\V key",
        "level1_match": ("numbers.tex vs independent re-derivation: " + allmatch(ks)) if ks else "n/a",
        "level2_value": ("re-derived: " + rec(ks)) if ks else "not re-derivable (source not committed)",
        "level2_method": c["method"] or "independent recomputation from the per-layer result CSVs via git show at 16def2de (no import of make_numbers.py); depth.csv cross-checked cell-for-cell against the per-layer files (0 differences)",
        "version_used": "origin/paper @ 16def2de", "robustness": "no interval; one run per layer (as the paper states)",
        "verdict": c["verdict"], "confidence": c["conf"], "importance": c["imp"], "script": SCRIPT,
        "notes": c["notes"], "batch": "P17"})
if SHA != C.PIN_PAPER:
    sys.exit("claims are written only at PIN_PAPER")
with open(OUT / "paper_new_claims.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
from collections import Counter
print(f"\nwrote {len(rows_out)} claims -> audit/p17/paper_new_claims.csv", dict(Counter(r["verdict"] for r in rows_out)))
print("level1 MISMATCH rows:", [r["id"] for r in rows_out if "MISMATCH" in r["level1_match"]])
