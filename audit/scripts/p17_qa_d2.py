"""P17 QA: independent re-derivation of sampled D2 rows from per-layer CSVs @16def2de (own code)."""
import csv, io, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
SHA = "16def2deec34aeee532511df116adb7de5522ca7"
R = "src/probes/deception-detection/results"
Ls = list(range(20, 80, 4)) + [79]; Ls = [l for l in Ls if l <= 76] + [79]
def show(p): return subprocess.run(["git","show",f"{SHA}:{p}"],cwd=ROOT,capture_output=True,text=True,encoding="utf-8").stdout
def tab(p): return list(csv.DictReader(io.StringIO(show(p))))
def mat(L, arm):
    rows = list(csv.reader(io.StringIO(show(f"{R}/persona_ood_ext_L{L}/matrix_{arm}.csv"))))
    cols = rows[0][1:]; return {(r[0], c): float(v) for r in rows[1:] for c, v in zip(cols, r[1:])}, cols
print("layers", Ls, len(Ls))
HK = "harm_refusal_knowledge_report__plain"
op = {L: tab(f"{R}/persona_ood_ext_L{L}/zero_shot_operating.csv") for L in Ls}
summ = {L: {r["features"]: r for r in tab(f"{R}/persona_ood_ext_L{L}/summary.csv")} for L in Ls}
def opv(L, arm, ds, k="auroc"):
    for r in op[L]:
        if r["arm"].startswith(arm) and r["dataset"] == ds: return float(r[k])
# D2-005 / D2-016 / D2-004: base probe vs best contrast on HP-KR, layers >= 36
arms = ["deception axis", "extended axis", "knowledge-denial axis", "sycophancy axis", "unified axis"]
lead = []
for L in Ls:
    b = opv(L, "base probe", HK); best = max((opv(L, a, HK), a) for a in arms)
    lead.append((L, b, best)); 
    print(f"L{L}: base {b:.4f} best {best[0]:.4f} ({best[1]}) {'BASE' if b > best[0] else 'contrast'}")
deep = [x for x in lead if x[0] >= 36]
print("D2-005/016: base leads at", sum(x[1] > x[2][0] for x in deep), "of", len(deep), "deep layers; trails at", [(x[0], round(x[1],3), round(x[2][0],3)) for x in deep if x[1] <= x[2][0]])
print("   shallow (<36) base leads at", [x[0] for x in lead if x[0] < 36 and x[1] > x[2][0]])
# D2-019 / D2-018: ext_axis1 - axis1 (published 8v7) off-diagonal, incl vs excl HP-KR
for L in Ls:
    s = summ[L]
    d_inc = float(s["ext_axis1"]["off_diagonal_mean"]) - float(s["axis1"]["off_diagonal_mean"])
    d_exc = float(s["ext_axis1"]["off_diagonal_excl_hpkr"]) - float(s["axis1"]["off_diagonal_excl_hpkr"])
    # per held-out dataset (column mean off-diagonal)
    me, cols = mat(L, "ext_axis1"); ma, _ = mat(L, "axis1")
    per = {c: sum(me[(r, c)] - ma[(r, c)] for r in cols if r != c) / (len(cols) - 1) for c in cols}
    worst = min(per.items(), key=lambda kv: kv[1])
    print(f"L{L}: ext-dec incl {d_inc:+.4f} excl {d_exc:+.4f}; worst per-dataset {worst[0][:14]} {worst[1]:+.4f}")
# D2-002: base diag min vs ext diag max; per-dataset max ext cell
bd = [float(summ[L]["baseline"]["diagonal_mean"]) for L in Ls]; ed = [float(summ[L]["ext_axis1"]["diagonal_mean"]) for L in Ls]
eo = [float(summ[L]["ext_axis1"]["off_diagonal_mean"]) for L in Ls]
print("D2-002: base diag min %.4f; ext diag max %.4f; ext off max %.4f" % (min(bd), max(ed), max(eo)))
mx = 0; mn = 1
for L in Ls:
    me, cols = mat(L, "ext_axis1"); mb, _ = mat(L, "baseline")
    mx = max(mx, max(me.values())); mn = min(mn, min(mb[(c, c)] for c in cols))
print("   max ext cell anywhere %.4f ; min raw diagonal cell %.4f" % (mx, mn))
# D2-012: zero-shot LB average, ext vs base at L36 and L32
for L in (20, 32, 36):
    for a in ("extended axis", "base probe"):
        print(f"D2-012 L{L} {a} avg auroc", opv(L, a, "Average"))
# D2-014: denial balacc on HP-KR per layer
print("D2-014 denial balacc HP-KR", {L: opv(L, "knowledge-denial axis", HK, "balacc") for L in Ls})
# D2-021: sycophancy axis HP-KR auroc max
sy = {L: opv(L, "sycophancy axis", HK) for L in Ls}; print("D2-021 syc HP-KR max", max(sy.items(), key=lambda kv: kv[1]))
# D2-022: deception axis vs syc axis off-diagonal (axis1 vs syc_axis1)
w = [L for L in Ls if float(summ[L]["axis1"]["off_diagonal_mean"]) > float(summ[L]["syc_axis1"]["off_diagonal_mean"])]
print("D2-022 dec>syc off-diag at", len(w), "of", len(Ls), "missing:", [L for L in Ls if L not in w])
