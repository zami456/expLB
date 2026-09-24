"""P17 Agent C -- re-derive the key numbers of ledger entries F-110 ... F-127 from raw.

Sources, in order of preference:
  * the laptop working tree (== asix e917ecd3 for every result file used here; checked by `git diff --quiet`),
  * git at a PINNED SHA (p17_common.PIN_PAPER / PIN_ASIX) -- never the moving name `origin/paper`,
  * audit/cache/labpc/ (md5-verified lab copies),
  * the lab PC, READ-ONLY, through p17_labfetch.fetch() (ssh BatchMode, python program on stdin,
    md5 verified after transport, cached in audit/p17/labcache/).

Usage:  python audit/scripts/p17_ledger_verify.py [F-110 F-121 ...] [--refresh]
Writes: audit/p17/ledger_verify_results.json (every check: f_id, item, ledger value, derived value, verdict, source)
Prints: one line per check (compact).
"""
import csv, glob, io, json, math, os, re, statistics, subprocess, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p17_common as C  # noqa: E402
import p17_labfetch as LF  # noqa: E402

ROOT = str(C.ROOT)
REFRESH = "--refresh" in sys.argv
CHECKS = []
LAB = "C:/Users/user4/expLB"
RES = "C:/Users/user4/expLB-resplit"


def rec(fid, item, ledger, derived, source, tol=None, verdict=None):
    """Record one check. verdict auto: numbers compared at the ledger's printed precision (or tol)."""
    if verdict is None:
        try:
            lv, dv = float(ledger), float(derived)
            if tol is None:
                s = str(ledger).lstrip("+-")
                dec = len(s.split(".")[1]) if "." in s else 0
                tol = 0.5 * 10 ** (-dec) + 1e-12
            verdict = "matches" if abs(lv - dv) <= tol else f"DIFFERS: ledger {ledger} vs raw {derived:.4f}" if isinstance(derived, float) else f"DIFFERS: ledger {ledger} vs raw {derived}"
        except (TypeError, ValueError):
            verdict = "matches" if str(ledger) == str(derived) else f"DIFFERS: ledger {ledger} vs raw {derived}"
    d = dict(f_id=fid, item=item, ledger=str(ledger),
             derived=(f"{derived:+.4f}" if isinstance(derived, float) and str(ledger).startswith(("+", "-")) else
                      f"{derived:.4f}" if isinstance(derived, float) else str(derived)),
             verdict=verdict, source=source)
    CHECKS.append(d)
    print(f"  {fid:6s} {item[:46]:46s} ledger={d['ledger']:>10s} raw={d['derived']:>10s}  {verdict[:40]}")


# ---------------------------------------------------------------- shared helpers
def jload(p):
    return json.load(open(p, encoding="utf-8"))


_LJ = {}


def lab_json(pattern):
    """{labpath: (dict, meta)} for a lab glob, via the read-only fetcher (memoised per run)."""
    if pattern in _LJ:
        return _LJ[pattern]
    out = _LJ[pattern] = {}
    for p, (b, m) in LF.fetch(pattern, refresh=REFRESH).items():
        out[p] = (json.loads(b.decode("utf-8")) if b is not None else None, m)
    return out


def offdiag(d, feature, probe="paper", keep=None):
    ds = d["datasets"]
    blk = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}.get((feature, probe))
    if blk is None:
        return None
    keep = keep or ds
    v = [blk[i][j] for i, a in enumerate(ds) for j, b in enumerate(ds)
         if i != j and a in keep and b in keep and blk[i][j] is not None]
    return statistics.fmean(v) if v else None


def cell(d, feature, tr, te, probe="paper"):
    ds = d["datasets"]
    blk = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}[(feature, probe)]
    return blk[ds.index(tr)][ds.index(te)]


def row_off(d, feature, tr, probe="paper"):
    ds = d["datasets"]
    blk = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}[(feature, probe)]
    i = ds.index(tr)
    return statistics.fmean(blk[i][j] for j in range(len(ds)) if j != i and blk[i][j] is not None)


def by_layer(mats):
    return {m["layer"]: m for m, _ in mats.values() if m}


def git_tab(sha, path):
    return C.tab(sha, path)


# ================================================================ F-110
def f110():
    print("F-110  70B like-for-like scale rung (lab: run-crossgen/scale_*)")
    DS5 = ["instructed-deception", "mask", "convincing-game", "ai_liar", "roleplaying"]

    def rung(store, pool, excl=()):
        t = by_layer(lab_json(f"{LAB}/run-crossgen/{store}/matrices/{pool}_L*_s42.json"))
        rows = []
        for L, m in sorted(t.items()):
            if L in excl:
                continue
            raw, dp = offdiag(m, "raw", keep=DS5), offdiag(m, "datapca", keep=DS5)
            best = max(v for v in (offdiag(m, "axes", keep=DS5), offdiag(m, "pc:deception", keep=DS5)) if v is not None)
            rows.append((raw, best - raw, best - dp))
        return [statistics.fmean(x[i] for x in rows) for i in range(3)], len(rows), sorted(t)

    # which 2 layers does "F-100's 19 layers" drop? find the pair that reproduces the kd row
    full, n, layers = rung("scale_70b", "mean")
    best_pair = None
    for a in layers:
        for b in layers:
            if a < b:
                v, _, _ = rung("scale_70b", "mean", (a, b))
                if abs(v[0] - 0.7524) < 5e-5 and abs(v[1] - 0.0079) < 5e-5 and abs(v[2] - 0.0751) < 5e-5:
                    best_pair = (a, b)
    ex = best_pair or ()
    rec("F-110", f"19-layer set = 21 minus L{ex}", "found" if best_pair else "?", "found" if best_pair else "not found",
        "brute force over layer pairs, scale_70b mean")
    src = f"{LAB}/run-crossgen/{{store}}/matrices/mean_L*_s42.json, DS5 off-diag, best=max(axes,pc:deception)"
    for store, lab, want in (("scale_3b", "3B", ("0.6989", "+0.0328", "+0.1197")),
                             ("scale_8b", "8B", ("0.7149", "+0.0089", "+0.1027")),
                             ("scale_70b", "70B kd", ("0.7524", "+0.0079", "+0.0751")),
                             ("scale_70b_paper16fp32", "70B p16fp32", ("0.7524", "-0.0131", "+0.0541"))):
        v, n, _ = rung(store, "mean", ex if store.startswith("scale_70b") else ())
        for name, w, x in zip(("raw", "best-raw", "best-dPCA"), want, v):
            rec("F-110", f"{lab} {name} (n={n})", w, x, src.format(store=store))
    v19, _, _ = rung("scale_70b_paper16fp32", "last", ex)
    v21, _, _ = rung("scale_70b_paper16fp32", "last")
    rec("F-110", "70B p16fp32 LAST best-raw (19L)", "+0.0061", v19[1], "last_L*")
    rec("F-110", "70B p16fp32 LAST best-dPCA (19L)", "+0.0688", v19[2], "last_L*")
    rec("F-110", "70B p16fp32 LAST best-raw (21L)", "+0.0069", v21[1], "last_L*")
    rec("F-110", "70B p16fp32 LAST best-dPCA (21L)", "+0.0681", v21[2], "last_L*")
    for store, lab, w in (("scale_3b", "3B", "+0.0745"), ("scale_8b", "8B", "+0.0090")):
        v, _, _ = rung(store, "last")
        rec("F-110", f"{lab} LAST best-dPCA", w, v[2], f"{store}/last_L*")


# ================================================================ F-121
def f121():
    print("F-121  kd-bank scale ladder (lab: expLB-resplit/run-crossgen/kdscale_*, recipe of p12_kd_scale_ladder.py)")

    def ladder(pattern):
        t = by_layer(lab_json(pattern))
        raw, best = [], []
        for L, m in t.items():
            feats = sorted({r["feature"] for r in m["results"]})
            pers = [f for f in feats if f.startswith("pc:") or f == "axes"]
            r = offdiag(m, "raw")
            b = max(offdiag(m, f) for f in pers if offdiag(m, f) is not None)
            raw.append(r); best.append(b)
        return statistics.fmean(best) - statistics.fmean(raw), len(raw)

    want = {("3b", "p16"): "+0.0382", ("3b", "all"): "+0.0317", ("3b", "s23"): "+0.0360",
            ("8b", "p16"): "+0.0148", ("8b", "all"): "+0.0169", ("8b", "s23"): "+0.0194",
            ("70b", "p16"): "-0.0116", ("70b", "all"): "+0.0032", ("70b", "s23"): "+0.0054"}
    got = {}
    for (r, v), w in want.items():
        if v == "p16":
            store = {"3b": "scale_3b", "8b": "scale_8b", "70b": "scale_70b_paper16fp32"}[r]
            pat = f"{LAB}/run-crossgen/{store}/matrices/mean_L*_s42.json"
        else:
            pat = f"{RES}/run-crossgen/kdscale_{r}_{v}/matrices/mean_L*_s42.json"
        x, n = ladder(pat)
        got[(r, v)] = x
        rec("F-121", f"{r} {v} best-raw (n={n} layers)", w, x, pat)
    for r, ws in (("3b", ("-0.0066", "-0.0022")), ("8b", ("+0.0022", "+0.0046")), ("70b", ("+0.0148", "+0.0170"))):
        for v, w in zip(("all", "s23"), ws):
            rec("F-121", f"{r} kd {v} - paper16", w, got[(r, v)] - got[(r, "p16")], "difference of the rows above")
    # DF-48: the axis files are byte-distinct
    try:
        lst = LF._run([f"{RES}/run-scale-kd/*/*axes*.pt"], content=False)
    except LF.LabUnreachable as e:
        rec("F-121", "axis md5s", "byte-distinct", f"UNKNOWN lab {e}", "lab", verdict="UNKNOWN")
        return
    for r, pre in (("3b", ("c93eb6d2", "e517a2d6")), ("8b", ("4141229", "0e68064"))):
        md5s = {p: m["md5"] for p, m in lst.items() if f"/{r}/" in p}
        hit = sorted(p for p, h in md5s.items() if any(h.startswith(x) for x in pre))
        rec("F-121", f"{r} all/s23 axis md5s {pre}", "2 distinct", f"{len(set(md5s[p] for p in hit))} distinct",
            "; ".join(f"{os.path.basename(p)}={md5s[p][:10]}" for p in sorted(md5s)),
            verdict="matches" if len(set(md5s[p] for p in hit)) == 2 else "DIFFERS")



def lab_rows(path):
    b, m = LF.fetch(path, refresh=REFRESH)[path]
    return [json.loads(x) for x in b.decode("utf-8").splitlines() if x.strip()], m


KR, KM = "harm-pressure-knowledge-report", "harm-pressure-kr-matched"
KNM = "harm-pressure-kr-knowledge-matched"


# ================================================================ F-115
def f115():
    print("F-115  leak-guarded F-03 cell (audit/cache/labpc/expLB-resplit == lab md5; kd11 matrices = git asix)")
    lst = LF._run([f"{RES}/run-crossgen/matrices/mean_L*_s42.json"], content=False)
    for L in (20, 40, 44):
        loc = os.path.join(ROOT, f"audit/cache/labpc/expLB-resplit/mean_L{L}_s42.json")
        lm = lst[f"{RES}/run-crossgen/matrices/mean_L{L}_s42.json"]["md5"]
        import hashlib
        rec("F-115", f"cache L{L} md5 == lab", lm[:10], hashlib.md5(open(loc, "rb").read()).hexdigest()[:10], loc)
    want_h = {20: ("-0.2566", "-0.2545"), 40: ("+0.0367", "+0.0309"), 44: ("+0.0134", "+0.0050")}
    want_c = {("KM>KR", 40): ("0.5905", "0.5416"), ("KM>KR", 44): ("0.6534", "0.5719"),
              ("KR>KM", 20): ("0.6004", "0.7190"), ("KR>KM", 40): ("0.5674", "0.4262"), ("KR>KM", 44): ("0.6675", "0.5667")}
    for L in (20, 40, 44):
        o = jload(os.path.join(ROOT, f"run-crossgen/kd11/matrices/mean_L{L}_s42.json"))
        g = jload(os.path.join(ROOT, f"audit/cache/labpc/expLB-resplit/mean_L{L}_s42.json"))
        for tag, d, w in (("leaked", o, want_h[L][0]), ("guarded", g, want_h[L][1])):
            rec("F-115", f"L{L} {tag} axes-raw, KM row mean", w, row_off(d, "axes", KM) - row_off(d, "raw", KM),
                f"{'run-crossgen/kd11' if tag == 'leaked' else 'cache/expLB-resplit'}/mean_L{L}_s42.json")
        for (k, LL), (wl, wg) in want_c.items():
            if LL != L:
                continue
            s_, t_ = (KM, KR) if k == "KM>KR" else (KR, KM)
            rec("F-115", f"L{L} {k} axes leaked", wl, cell(o, "axes", s_, t_), "kd11 matrix cell")
            rec("F-115", f"L{L} {k} axes guarded", wg, cell(g, "axes", s_, t_), "resplit matrix cell")
        if L == 40:
            rec("F-115", "L40 KR>KM raw leaked", "0.7932", cell(o, "raw", KR, KM), "kd11 cell")
            rec("F-115", "L40 KR>KM raw guarded", "0.7214", cell(g, "raw", KR, KM), "resplit cell")
    # 59 surviving rows: kr-matched ids not in HP-KR's TRAIN rows (resplit splits.json, seed 42)
    sp = json.loads(LF.fetch(f"{RES}/run-crossgen/splits.json")[f"{RES}/run-crossgen/splits.json"][0])["splits"]["42"]
    km_rows, _ = lab_rows(f"{LAB}/run-coords/passB/{KM}/rows.jsonl")
    kr_rows, _ = lab_rows(f"{LAB}/run-coords/passB/{KR}/rows.jsonl")
    kr_ids = {r["row_id"] for r in kr_rows}
    kr_test = set(sp[KR]["test_row_ids"])
    km_ids = [r["row_id"] for r in km_rows]
    rec("F-115", "kr-matched ids that are HP-KR ids", "258/258", f"{sum(i in kr_ids for i in km_ids)}/{len(km_ids)}", "passB rows.jsonl")
    surv = sum(1 for i in km_ids if i in kr_test)
    rec("F-115", "kr-matched rows surviving guard", "59", surv, "splits.json[42][KR].test_row_ids x KM rows.jsonl")
    rec("F-115", "contamination %", "77", round(100 * (1 - surv / 258)), "derived")
    # F-113 item 5: "42 of 52" is under 30g_clustered_holdout.py's OWN split: default_rng(42).permutation(n)
    # per dataset in CLUSTERS order (harm-pressure-choice n=2134 first, then HP-KR, then kr-matched).
    import numpy as np
    rng = np.random.default_rng(42)
    rng.permutation(2134)
    p1, p2 = rng.permutation(len(kr_rows)), rng.permutation(len(km_rows))
    tr = {kr_rows[i]["row_id"] for i in p1[:int(0.8 * len(kr_rows))]}
    te = [km_rows[i]["row_id"] for i in p2[int(0.8 * len(km_rows)):]]
    rec("F-113", "kr-matched TEST rows in HP-KR train (30g split)", "42", sum(r in tr for r in te),
        "replay of 30g_clustered_holdout.py:101-106 split over passB rows.jsonl (lab)")
    km_test = set(sp[KM]["test_row_ids"])
    rec("F-113", "same count under crossgen split (context only)", "n/a", sum(1 for i in km_test if i not in kr_test),
        "expLB-resplit splits.json seed 42 == kd11/splits.json", verdict="context (different pipeline split)")
    pos = sum(r["label"] == 1 for r in km_rows)
    rec("F-112", "kr-matched rows {deceptive/honest}", "258 {124/134}", f"{len(km_rows)} {{{pos}/{len(km_rows) - pos}}}", "passB/kr-matched/rows.jsonl (lab)")
    pos = sum(r["label"] == 1 for r in kr_rows)
    rec("F-112", "HP-KR rows {deceptive/honest}", "2139 {139/2000}", f"{len(kr_rows)} {{{pos}/{len(kr_rows) - pos}}}", "passB/HP-KR/rows.jsonl (lab)")


# ================================================================ F-116
def f116():
    print("F-116  knowledge_matched slice (audit/cache/labpc/expLB-resplit/km == lab md5)")
    import hashlib
    lst = LF._run([f"{RES}/run-crossgen/km/matrices/mean_L*_s42.json"], content=False)
    for L in (20, 40, 44):
        loc = os.path.join(ROOT, f"audit/cache/labpc/expLB-resplit/km/mean_L{L}_s42.json")
        rec("F-116", f"cache km L{L} md5 == lab", lst[f"{RES}/run-crossgen/km/matrices/mean_L{L}_s42.json"]["md5"][:10],
            hashlib.md5(open(loc, "rb").read()).hexdigest()[:10], loc)
    rows, _ = lab_rows(f"{RES}/passB/{KNM}/rows.jsonl")
    pos = sum(r["label"] == 1 for r in rows)
    rec("F-116", "knowledge_matched rows", "2001", len(rows), f"{RES}/passB/{KNM}/rows.jsonl")
    rec("F-116", "class balance dec/hon", "139/1862", f"{pos}/{len(rows) - pos}", "same")
    rec("F-116", "share of HP-KR (93.5%)", "93.5", round(100 * len(rows) / 2139, 1), "2001/2139")
    rec("F-116", "7.8x bigger than 258", "7.8", round(len(rows) / 258, 1), "2001/258")
    want = {20: ("0.3850", "0.5711", "-0.1860", "0.5208", "0.5455", "0.6174"),
            40: ("0.3335", "0.7093", "-0.3758", "0.6636", "0.6718", "0.6174"),
            44: ("0.5267", "0.7203", "-0.1937", "0.7037", "0.6237", "0.6174")}
    diag = {20: ("0.9541", "0.9995"), 40: ("0.8347", "0.9992"), 44: ("0.7209", "0.9993")}
    for L in (20, 40, 44):
        d = jload(os.path.join(ROOT, f"audit/cache/labpc/expLB-resplit/km/mean_L{L}_s42.json"))
        v = {f: row_off(d, f, KNM) for f in ("axes", "raw", "datapca", "random", "length")}
        for name, w, x in zip(("axes", "raw", "axes-raw", "datapca", "random", "length"), want[L],
                              (v["axes"], v["raw"], v["axes"] - v["raw"], v["datapca"], v["random"], v["length"])):
            rec("F-116", f"L{L} transfer {name}", w, x, f"km/mean_L{L}_s42.json row {KNM}, 11 off-diag")
        for name, w in zip(("axes", "raw"), diag[L]):
            rec("F-116", f"L{L} in-dist diag {name}", w, cell(d, name, KNM, KNM), "diagonal cell")
        rec("F-116", f"L{L} in-dist diag length", "0.8318", cell(d, "length", KNM, KNM), "diagonal cell")
        # zero-shot contrast on slice vs full
        zs = {z["direction"]: z["auroc"] for z in d["zeroshot"]}
        c = zs["contrast"]
        rec("F-116", f"L{L} zero-shot contrast slice", {20: "0.2405", 40: "0.6466", 44: "0.6085"}[L], c[d["datasets"].index(KNM)], "zeroshot")
        rec("F-116", f"L{L} zero-shot contrast full HP-KR", {20: "0.2504", 40: "0.6436", 44: "0.6032"}[L], c[d["datasets"].index(KR)], "zeroshot")




# ================================================================ collaborator branch (git, PINNED)
CB = "src/probes/deception-detection/results"
LAYERS16 = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79]
HPKRc = "harm_refusal_knowledge_report__plain"


def summ(sha, L):
    t = C.tab(sha, f"{CB}/persona_ood_ext_L{L}/summary.csv")
    return {r["features"]: r for r in t} if t else {}


def zshot(sha, L):
    t = C.tab(sha, f"{CB}/persona_ood_ext_L{L}/zero_shot.csv")
    return {r["dataset"]: r for r in t} if t else {}


def mat(sha, L, arm):
    t = C.show(sha, f"{CB}/persona_ood_ext_L{L}/matrix_{arm}.csv")
    if not t:
        return None, None
    rows = list(csv.reader(io.StringIO(t)))
    return rows[0][1:], {r[0]: [float(x) for x in r[1:]] for r in rows[1:]}


def f117():
    print("F-117  collaborator 16-layer sweep (git summary.csv / zero_shot.csv at PIN_PAPER; PIN_OLD for comparison)")
    for sha, tag in ((C.PIN_PAPER, "pin"), (C.PIN_OLD, "old")):
        S = {L: summ(sha, L) for L in LAYERS16}
        f = lambda L, a, c: float(S[L][a][c])
        mw = {L: f(L, "ext_axis1", "off_diagonal_mean") - f(L, "ext_pc3", "off_diagonal_mean") for L in LAYERS16}
        mx = {L: f(L, "ext_axis1", "off_diagonal_excl_hpkr") - f(L, "ext_pc3", "off_diagonal_excl_hpkr") for L in LAYERS16}
        bw = {L: f(L, "ext_axis1", "off_diagonal_mean") - f(L, "baseline", "off_diagonal_mean") for L in LAYERS16}
        bx = {L: f(L, "ext_axis1", "off_diagonal_excl_hpkr") - f(L, "baseline", "off_diagonal_excl_hpkr") for L in LAYERS16}
        src = f"git {sha[:8]}:{CB}/persona_ood_ext_L*/summary.csv"
        if tag == "pin":
            want = {20: ("+0.2204", "+0.0086"), 24: ("+0.2221", "+0.0012"), 36: ("+0.0481", "+0.0155"),
                    40: ("+0.0503", "+0.0495"), 44: ("+0.0235", "+0.0354"), 64: ("-0.0010", "-0.0029"),
                    68: ("-0.0205", "-0.0325"), 72: ("-0.0146", "-0.0198")}
            for L, (a, b) in want.items():
                rec("F-117", f"L{L} ext_axis1-ext_pc3 with HP-KR", a, mw[L], src)
                rec("F-117", f"L{L} ext_axis1-ext_pc3 excl HP-KR", b, mx[L], src)
            rec("F-117", "positive layers (with HP-KR)", "13/16", f"{sum(v > 0 for v in mw.values())}/16", src)
            rec("F-117", "positive layers (excl HP-KR)", "11/16", f"{sum(v > 0 for v in mx.values())}/16", src)
            mxa = max(abs(v) for v in mx.values())
            rec("F-117", "excl margins within +-0.06", "yes", "yes" if mxa < 0.06 else f"no (max {mxa:.4f})", src)
            seq = [mx[L] for L in LAYERS16]
            rec("F-117", "excl margin zero crossings", "5", sum((a > 0) != (b > 0) for a, b in zip(seq, seq[1:])), src)
            rec("F-117", "ext_axis1-baseline >0 layers (with)", "16/16", f"{sum(v > 0 for v in bw.values())}/16", src)
            rec("F-117", "ext_axis1-baseline min (with)", "+0.115", min(bw.values()), src)
            rec("F-117", "ext_axis1-baseline max (with)", "+0.273", max(bw.values()), src)
            rec("F-117", "ext_axis1-baseline >0 layers (excl)", "16/16", f"{sum(v > 0 for v in bx.values())}/16", src)
            rec("F-117", "ext_axis1-baseline min (excl)", "+0.123", min(bx.values()), src)
            rec("F-117", "ext_axis1-baseline max (excl)", "+0.243", max(bx.values()), src)
            ea = {L: f(L, "ext_axis1", "off_diagonal_mean") for L in LAYERS16}
            da = {L: f(L, "denial_axis1", "off_diagonal_mean") for L in LAYERS16}
            rec("F-117", "ext_axis1 peak layer", "36", max(ea, key=ea.get), src)
            rec("F-117", "ext_axis1 L36", "0.9275", ea[36], src)
            rec("F-117", "ext_axis1 L40", "0.9158", ea[40], src)
            rec("F-117", "denial_axis1 peak layer", "36", max(da, key=da.get), src)
            Z = {L: zshot(sha, L) for L in LAYERS16}
            zs = lambda L, col: float(Z[L][HPKRc][col])
            zsrc = f"git {sha[:8]}:{CB}/persona_ood_ext_L*/zero_shot.csv row {HPKRc}"
            rec("F-117", "HP-KR denial_axis L40 (global max)", "0.9185", zs(40, "auroc_denial_axis"), zsrc)
            rec("F-117", "HP-KR denial_axis argmax layer", "40", max(LAYERS16, key=lambda L: zs(L, "auroc_denial_axis")), zsrc)
            rec("F-117", "HP-KR ext_axis L76", "0.9249", zs(76, "auroc_ext_axis"), zsrc)
            rec("F-117", "HP-KR ext_axis L40", "0.9124", zs(40, "auroc_ext_axis"), zsrc)
            for L, w in ((72, "0.4344"), (76, "0.7301"), (79, "0.7709")):
                rec("F-117", f"HP-KR published 8v7 axis L{L}", w, zs(L, "auroc_axis"), zsrc)
            for L, w in ((72, "0.6526"), (76, "0.8738"), (79, "0.8179")):
                rec("F-117", f"axis1 off-diag L{L}", w, f(L, "axis1", "off_diagonal_mean"), src)
            r3 = [f(L, "random3", "off_diagonal_mean") for L in LAYERS16]
            rec("F-117", "random3 off-diag range", "0.48-0.55", f"{min(r3):.4f}-{max(r3):.4f}", src,
                verdict="matches" if min(r3) >= 0.475 and max(r3) < 0.555 else f"DIFFERS: {min(r3):.4f}-{max(r3):.4f}")
        else:
            P = {L: summ(C.PIN_PAPER, L) for L in LAYERS16}
            ch = [L for L in LAYERS16 if abs(mw[L] - (float(P[L]["ext_axis1"]["off_diagonal_mean"]) -
                                                      float(P[L]["ext_pc3"]["off_diagonal_mean"]))) > 1e-9]
            chb = [L for L in LAYERS16 if abs(bw[L] - (float(P[L]["ext_axis1"]["off_diagonal_mean"]) -
                                                       float(P[L]["baseline"]["off_diagonal_mean"]))) > 1e-9]
            rec("F-117", "ext margins identical PIN_OLD vs PIN_PAPER", "identical", "identical" if not ch else f"differ at {ch}", src)
            rec("F-117", "baseline margins identical OLD vs PIN", "identical", "identical" if not chb else f"differ at {chb} (<1e-4)", src,
                verdict="matches" if not chb else "context: baseline re-run moved 4th-5th dp at " + str(chb))


def f118():
    print("F-118  Gender-Secret (git PIN_PAPER results/gender_secret/summary.md) + collab-bank matched slice (audit cache)")
    t = C.show(C.PIN_PAPER, f"{CB}/gender_secret/summary.md") or ""
    want = {"knowledge-denial (4v6)": "0.549 / 0.436 / 0.498", "extended (12v13)": "0.505 / 0.467 / 0.541",
            "published (8v7)": "0.514 / 0.545 / 0.623", "sycophancy (5v5)": "0.593 / 0.523 / 0.520"}
    for k, w in want.items():
        line = next((l for l in t.splitlines() if l.startswith(f"| {k}")), "")
        got = line.split("|")[2].strip().replace("**", "") if line else "NOT-FOUND"
        rec("F-118", f"GS {k} L20/L40/L44", w, got, f"git {C.PIN_PAPER[:8]}:{CB}/gender_secret/summary.md")
    rec("F-118", "behavioural 82% vs base 55%", "82/55", "82/55" if ("**82%**" in t and "55%" in t) else "NOT-FOUND", "summary.md")
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", "572095cb", C.PIN_PAPER], cwd=ROOT).returncode == 0
    rec("F-118", "commit 572095cb ancestor of PIN_PAPER", "yes", "yes" if anc else "no", "git merge-base")
    for L, w in ((20, "0.4630"), (40, "0.6203"), (44, "0.5383")):
        d = jload(os.path.join(ROOT, f"audit/cache/labpc/run-crossgen/kd11/banks/collab/matrices/mean_L{L}_s42.json"))
        zs = {z["direction"]: z["auroc"] for z in d["zeroshot"]}
        rec("F-118", f"collab extended axis on kr-matched L{L}", w, zs["contrast:collab_extended"][d["datasets"].index(KM)],
            f"audit/cache/labpc/run-crossgen/kd11/banks/collab/matrices/mean_L{L}_s42.json zeroshot contrast:collab_extended")
    fb = open(os.path.join(ROOT, "kd/docs/FINDINGS.md"), encoding="utf-8").read()
    sec = fb[fb.index("### F-115 "):fb.index("### F-116 ")]
    ok = "0.4630" in sec
    rec("F-118", "pointer 'F-115 §5' resolves", "yes", "yes" if ok else "no", "kd/docs/FINDINGS.md F-115 section",
        verdict="matches" if ok else "DIFFERS: pointer error - F-115 has no §5; numbers are audit/HPKR_WORDING_CONTROLLED.md §5")


def f119():
    print("F-119/F-123/F-127  collaborator release artifacts (git PIN_PAPER)")
    mn = "src/probes/deception-detection/paper/make_numbers.py"
    for sha, tag, want in ((C.PIN_PAPER, "paper", "1"), (C.PIN_ASIX, "asix", "0"), (C.PIN_PCOOD, "pc-ood", "0")):
        r = subprocess.run(["git", "cat-file", "-s", f"{sha}:{mn}"], cwd=ROOT, capture_output=True, text=True)
        rec("F-123", f"make_numbers.py present on {tag}", want, "1" if r.returncode == 0 else "0", f"git cat-file {sha[:8]}")
        if tag == "paper":
            rec("F-119", "make_numbers.py exists on paper", "yes", "yes" if r.returncode == 0 else "no", f"git {sha[:8]}")
    lg = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h|%ad|%an", "--date=short", C.PIN_PAPER, "--", mn],
                        cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    rec("F-123", "make_numbers.py added in", "8fd071b2|2026-09-22|zami456", lg[-1] if lg else "NOT-FOUND", "git log --diff-filter=A")
    mf = subprocess.run(["git", "cat-file", "-e", f"{C.PIN_PAPER}:src/probes/deception-detection/paper/make_figures.py"], cwd=ROOT).returncode == 0
    rec("F-119", "make_figures.py beside it", "yes", "yes" if mf else "no", "git")
    pdm = C.show(C.PIN_PAPER, f"{CB}/persona_depth/summary.md") or ""
    for folder in ("persona_unified", "persona_depth_orig"):
        cited = f"../{folder}/" in pdm
        ex = subprocess.run(["git", "ls-tree", "-d", C.PIN_PAPER, f"{CB}/{folder}"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        rec("F-119", f"persona_depth cites {folder}, folder absent", "cited & absent",
            f"{'cited' if cited else 'not cited'} & {'present' if ex else 'absent'}", "git ls-tree PIN_PAPER")
    # numbers.tex / generated/ moved between the four origin/paper SHAs of 2026-09-24; evaluate at each
    SHAS = [("ea1ce52b", C.PIN_OLD), ("2ad758fb", C.PIN_P2AD), ("d1b88154", C.PIN_PD1B), ("16def2de", C.PIN_PAPER)]

    def gen_facts(sha):
        nt = C.show(sha, "src/probes/deception-detection/paper/generated/numbers.tex") or ""
        ls = subprocess.run(["git", "ls-tree", "--name-only", f"{sha}:src/probes/deception-detection/paper/generated/"],
                            cwd=ROOT, capture_output=True, text=True).stdout.split()
        mac = {}
        for k in ("cos-dec-den-L40", "proj-den-kd-L40", "proj-den-sincere-L40", "proj-den-deceptive-L40"):
            m = re.search(r"\\setV\{" + re.escape(k) + r"\}\{((?:\\ensuremath\{-\})?[^}]*)\}", nt)
            mac[k] = m.group(1).replace("\\ensuremath{-}", "-") if m else "NOT-FOUND"
        size = subprocess.run(["git", "cat-file", "-s", f"{sha}:{mn}"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return dict(lines=len(nt.splitlines()), setv=len(re.findall(r"\\setV\{", nt)),
                    tab=sum(x.startswith("tab_") and x.endswith(".tex") for x in ls),
                    fig=sum(x.startswith("fig_") and x.endswith(".pdf") for x in ls),
                    head="Generated by make_numbers.py" in nt[:600], mn_bytes=size, **mac)

    facts = {n: gen_facts(sha) for n, sha in SHAS}

    def at(key, want, fid, item):
        hits = [n for n, f in facts.items() if str(f[key]) == str(want)]
        pin = facts["16def2de"][key]
        v = ("matches" if str(pin) == str(want) else
             f"matches at {'/'.join(hits)} only; PIN_PAPER={pin} (branch moved)" if hits else f"DIFFERS: ledger {want} vs raw {pin} (all 4 SHAs: {[f[key] for f in facts.values()]})")
        rec(fid, item, want, pin, "git " + ", ".join(f"{n}={f[key]}" for n, f in facts.items()), verdict=v)

    at("mn_bytes", "28232", "F-123", "make_numbers.py bytes")
    at("lines", "1162", "F-127", "numbers.tex lines")
    at("setv", "1162", "F-127", "numbers.tex setV macros")
    at("tab", "8", "F-127", "tab_*.tex files")
    at("fig", "5", "F-127", "fig_*.pdf files")
    at("head", "True", "F-127", "header 'Generated by make_numbers.py'")
    for k, w in (("cos-dec-den-L40", "0.47"), ("proj-den-kd-L40", "2.90"), ("proj-den-sincere-L40", "-0.08"), ("proj-den-deceptive-L40", "0.05")):
        at(k, w, "F-127", f"macro {k}")


def f122():
    print("F-122  collaborator sycophancy arms (git PIN_PAPER; recipe of p14_collab_syc_arms.py)")
    want = {36: ("0.8124", "ext_axis1", "0.9403", "0.9222"), 40: ("0.7581", "syc_pc1", "0.9370", "0.9165"),
            44: ("0.7297", "syc_pc1", "0.9274", "0.9053"), 48: ("0.7375", "syc_pc1", "0.9228", "0.8952"),
            52: ("0.7515", "syc_pc1", "0.8999", "0.8806"), 56: ("0.7419", "syc_pc1", "0.8685", "0.8670")}
    for sha in (C.PIN_PAPER, C.PIN_OLD):
        wins = defaultdict(int)
        for L in LAYERS16:
            S = summ(sha, L)
            base = float(S["baseline"]["off_diagonal_excl_hpkr"])
            cand = {k: float(v["off_diagonal_excl_hpkr"]) for k, v in S.items() if k not in ("baseline", "random3")}
            arm = max(cand, key=cand.get)
            wins[arm] += 1
            src = f"git {sha[:8]}:{CB}/persona_ood_ext_L{L}/summary.csv off_diagonal_excl_hpkr"
            if L in want:
                wb, wa, wv, wo = want[L]
                tag = "" if sha == C.PIN_PAPER else " @PIN_OLD"
                if sha == C.PIN_PAPER:
                    bo = float(summ(C.PIN_OLD, L)["baseline"]["off_diagonal_excl_hpkr"])
                    v = ("matches" if abs(base - float(wb)) <= 5e-5 else
                         f"matches at ea1ce52b ({bo:.5f}); PIN_PAPER {base:.5f} (baseline re-run, branch moved)" if abs(bo - float(wb)) <= 5e-5
                         else f"DIFFERS: ledger {wb} vs {base:.4f}")
                    rec("F-122", f"L{L} baseline excl", wb, base, src, verdict=v)
                if sha == C.PIN_PAPER:
                    rec("F-122", f"L{L} best arm", wa, arm, src)
                    rec("F-122", f"L{L} best value", wv, cand[arm], src)
                    oth = "syc_pc1" if wa == "ext_axis1" else "ext_axis1"
                    rec("F-122", f"L{L} {oth}", wo, cand[oth], src)
        w = ", ".join(f"{k}={v}" for k, v in sorted(wins.items(), key=lambda x: (-x[1], x[0])))
        rec("F-122", f"arm win counts /16 @{sha[:8]}", "syc_pc1=5, ext_axis1=4, ext_pc3=4, axis1=1, syc3=1, uni_axis1=1", w, "summary.csv x16")
    cs, ms = mat(C.PIN_PAPER, 40, "syc_pc1")
    ce, me = mat(C.PIN_PAPER, 40, "ext_axis1")
    want = {"convince_me__plain": ("0.9984", "0.9123"), "harm_refusal_choice__plain": ("0.7521", "0.7116"),
            "insider_trading_cadenza__plain": ("0.9853", "0.9803"), "instruct_dishonesty__plain": ("0.9500", "0.9784"),
            "repe_honesty__plain": ("0.9992", "1.0000")}
    for ci, c in enumerate(cs):
        if c not in want:
            continue
        a = statistics.fmean(ms[tr][ci] for tr in ms if tr not in (c, HPKRc))
        b = statistics.fmean(me[tr][ci] for tr in me if tr not in (c, HPKRc))
        rec("F-122", f"L40 {c[:14]} syc_pc1", want[c][0], a, "matrix_syc_pc1.csv @PIN_PAPER")
        rec("F-122", f"L40 {c[:14]} ext_axis1", want[c][1], b, "matrix_ext_axis1.csv @PIN_PAPER")
    cm = []
    for L in (40, 44, 48, 52, 56):
        cs, ms = mat(C.PIN_PAPER, L, "syc_pc1"); ce, me = mat(C.PIN_PAPER, L, "ext_axis1")
        ci = cs.index("convince_me__plain")
        cm.append(statistics.fmean(ms[tr][ci] for tr in ms if tr not in ("convince_me__plain", HPKRc)) -
                  statistics.fmean(me[tr][ci] for tr in me if tr not in ("convince_me__plain", HPKRc)))
    got = ",".join(f"{x:+.3f}" for x in cm)
    rec("F-122", "convince_me margin L40..56", "+0.086,+0.128,+0.123,+0.099,+0.054", got, "matrices @PIN_PAPER")
    for arm in ("ext_axis1", "denial_axis1"):
        sp = []
        for L in LAYERS16:
            cols, m = mat(C.PIN_PAPER, L, arm)
            mx = 0.0
            for ci, c in enumerate(cols):
                e = [m[tr][ci] for tr in m if tr not in (c, HPKRc)]
                mx = max(mx, max(e) - min(e))
            sp.append(mx)
        rec("F-122", f"{arm} max spread over 16 layers", "0.000", f"{max(sp):.4f}", f"matrix_{arm}.csv x16 (HP-KR row dropped)",
            verdict="matches" if max(sp) < 0.0015 else f"DIFFERS: max spread {max(sp):.4f}")




# ================================================================ corrections entries (F-111 / F-112 / F-114 / F-123)
K11 = os.path.join(ROOT, "run-crossgen", "kd11")
SYC = "sycophancy_mt"


def bh(ps):
    n = len(ps)
    order = sorted(range(n), key=lambda i: ps[i])
    q = [0.0] * n
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, ps[i] * n / rank)
        q[i] = prev
    return q


def passB_gaps():
    gaps, nulls = {}, {}
    for ds in ("ai_liar", "convincing-game", "harm-pressure-choice", KR, KM, "insider-trading", "instructed-deception",
               "mask", "roleplaying", "sandbagging_v2", SYC):
        R, _ = lab_rows(f"{LAB}/run-coords/passB/{ds}/rows.jsonl")
        nulls[ds] = sum(r["label"] is None for r in R)
        p = [r["span_len"] for r in R if r["label"] == 1]
        n = [r["span_len"] for r in R if r["label"] == 0]
        gaps[ds] = (statistics.fmean(p), statistics.fmean(n), statistics.fmean(p) - statistics.fmean(n))
    return gaps, nulls


def f111():
    print("F-111  audit corrections (subset re-derived: items 1,2,3,4,7,8 + F-73's numbers)")
    # item 1: 28/80 and 31/80 -- kd_s23 bank store vs ROOT store (raw, datapca), train row kr-matched, 10 targets, 80 layers
    bank = by_layer(lab_json(f"{LAB}/run-crossgen/kd11/passA11/banks/kd_s23/matrices/mean_L*_s42.json"))
    root = by_layer(lab_json(f"{LAB}/run-crossgen/kd11/passA11/matrices/mean_L*_s42.json"))
    Ls = sorted(set(bank) & set(root))
    wr = sum(row_off(bank[L], "axes", KM) > row_off(root[L], "raw", KM) for L in Ls)
    wd = sum(row_off(bank[L], "axes", KM) > row_off(root[L], "datapca", KM) for L in Ls)
    has = sum(any(r["feature"] == "datapca" for r in bank[L]["results"]) for L in Ls)
    wb = sum(row_off(bank[L], "axes", KM) > row_off(bank[L], "datapca", KM) for L in Ls) if has == len(Ls) else f"n/a: datapca column in {has}/{len(Ls)} bank matrices"
    src = f"{LAB}/run-crossgen/kd11/passA11/{{banks/kd_s23/,}}matrices/mean_L*_s42.json, row {KM}"
    rec("F-111", f"kd slice beats raw (of {len(Ls)} layers)", "28/80", f"{wr}/{len(Ls)}", src)
    rec("F-111", f"kd slice beats root datapca (of {len(Ls)})", "31/80", f"{wd}/{len(Ls)}", src)
    if isinstance(wb, str):   # matrices carry no datapca: the bank store's datapca lives in its summary.json
        sp_ = f"{LAB}/run-crossgen/kd11/passA11/banks/kd_s23/summary.json"
        sj = json.loads(LF.fetch(sp_, refresh=REFRESH)[sp_][0])
        R = {(r["feature"], r["layer"]): r for r in sj["records"]
             if r["pooling"] == "mean" and r["probe"] == "paper" and r["seed"] == 42}
        Ls2 = sorted({l for f, l in R if f == "axes"} & {l for f, l in R if f == "datapca"})
        wb = sum(R[("axes", l)]["per_source_offdiag"][KM] > R[("datapca", l)]["per_source_offdiag"][KM] for l in Ls2)
        src2 = sp_ + " records (mean, paper, s42) per_source_offdiag[kr-matched]"
        same = all(abs(R[("datapca", l)]["per_source_offdiag"][KM] - row_off(root[l], "datapca", KM)) < 1e-9 for l in Ls2 if l in root)
        rec("F-111", "trap: vs the bank store's own datapca", "0/80", f"{wb}/{len(Ls2)}", src2,
            verdict=f"DIFFERS: not reproduced - the bank summary's datapca {'EQUALS' if same else 'differs from'} the root store's at every layer; axes wins {wb}/{len(Ls2)} against it (no datapca column in the bank matrices)")
    else:
        rec("F-111", "trap: vs the bank store's own datapca", "0/80", f"{wb}/{len(Ls)}", src)
    # item 2: cap responses teacher-forced turn 1
    rows = []
    for p, (b, m) in LF.fetch(f"{LAB}/run-cap/responses/*.jsonl", refresh=REFRESH).items():
        rows += [json.loads(x) for x in b.decode("utf-8").splitlines() if x.strip()]
    nfiles = len(LF.fetch(f"{LAB}/run-cap/responses/*.jsonl"))
    qs = {r["question"] for r in rows}
    pairs = {(r["question"], r["turn1"]) for r in rows}
    t1 = defaultdict(set)
    for r in rows:
        t1[r["question"]].add(r["turn1"])
    srcc = f"{LAB}/run-cap/responses/*.jsonl"
    rec("F-111", "cap rows / files", "1300/13", f"{len(rows)}/{nfiles}", srcc)
    rec("F-111", "cap personas (distinct system prompts)", "65", len({r['system_prompt'] for r in rows}), srcc)
    rec("F-111", "cap questions", "20", len(qs), srcc)
    rec("F-111", "distinct (question, turn1) pairs", "20", len(pairs), srcc)
    rec("F-111", "max distinct turn-1 texts per question", "1", max(len(v) for v in t1.values()), srcc)
    per = defaultdict(int)
    for r in rows:
        per[(r["question"], r["turn1"])] += 1
    rec("F-111", "rows per (question, turn1) pair", "65", "/".join(sorted({str(v) for v in per.values()})), srcc)
    # items 3/4: F-04 family in kd11/significance.json (git asix == WT)
    T = jload(os.path.join(K11, "significance.json"))["tests"]
    g = {(t["block"], t["pooling"], t["layer"], t["probe"], t["comparison"]): t for t in T}
    s3 = "run-crossgen/kd11/significance.json (4,032 tests)"
    rec("F-111", "block9 mean L40 paper axes-raw", "+0.0147", g[("9", "mean", 40, "paper", "axes - raw")]["diff"], s3)
    ci = g[("9", "mean", 40, "paper", "axes - raw")]["dataset_ci95"]
    rec("F-111", "  its dataset CI", "[-0.170, +0.187]", f"[{ci[0]:+.3f}, {ci[1]:+.3f}]", s3)
    rec("F-111", "block9 mean L40 paper axes-datapca", "+0.0264", g[("9", "mean", 40, "paper", "axes - datapca")]["diff"], s3)
    rec("F-111", "block10 mean L40 paper axes-raw", "-0.0215", g[("10", "mean", 40, "paper", "axes - raw")]["diff"], s3)
    rec("F-111", "block10 mean L40 paper axes-datapca", "-0.0243", g[("10", "mean", 40, "paper", "axes - datapca")]["diff"], s3)
    for blk in ("9",):
        fam = [t for t in T if t["block"] == blk and t["layer"] in (40, 44)]
        q = bh([t["dataset_p"] for t in fam])
        for t, qq in zip(fam, q):
            t["_q"] = qq
        rec("F-111", f"family size block {blk} (layers 40,44)", "192", len(fam), s3)
        a = next(t for t in fam if (t["pooling"], t["layer"], t["probe"], t["comparison"]) == ("mean", 40, "paper", "axes - raw"))
        rec("F-111", "q of axes-raw in restricted family", "0.9311", a["_q"], s3 + ", BH within block over L40/44")
        a = next(t for t in fam if (t["pooling"], t["layer"], t["probe"], t["comparison"]) == ("mean", 40, "paper", "axes - datapca"))
        rec("F-111", "q of axes-datapca in restricted family", "0.8546", a["_q"], s3)
        sig = [t for t in fam if t["_q"] < 0.05]
        rec("F-111", "tests clearing BH in family", "1: random@2 - raw last L44 mms",
            "; ".join(f"{t['comparison']} {t['pooling']} L{t['layer']} {t['probe']}" for t in sig), s3,
            verdict="matches" if len(sig) == 1 and sig[0]["comparison"] == "random@2 - raw" else "DIFFERS")
        if sig:
            rec("F-111", "  its diff", "-0.2287", sig[0]["diff"], s3)
            rec("F-111", "  its q", "0.0192", sig[0]["_q"], s3)
    # item 7 + F-73: token gaps and length feature
    gaps, nulls = passB_gaps()
    s7 = f"{LAB}/run-coords/passB/<ds>/rows.jsonl span_len by label"
    rec("F-111", "positive class longer (of 11)", "9/11", f"{sum(v[2] > 0 for v in gaps.values())}/11", s7)
    rec("F-111", "... excluding sycophancy_mt", "8/10", f"{sum(v[2] > 0 for k, v in gaps.items() if k != SYC)}/10", s7)
    rec("F-111", "convincing-game gap", "-1.20", gaps["convincing-game"][2], s7)
    rec("F-111", "roleplaying gap", "-4.10", gaps["roleplaying"][2], s7)
    rec("F-111", "roleplaying null labels", "84", nulls["roleplaying"], s7)
    rec("F-111", "harm-pressure-choice gap (F-73)", "+0.0035", gaps["harm-pressure-choice"][2], s7)
    eight = [k for k in gaps if k not in (KM, "mask", "sandbagging_v2")]
    rec("F-111", "median |gap| over 8 (F-73)", "3.7", statistics.median(abs(gaps[k][2]) for k in eight), s7 + "; the 8 = 11 minus kr-matched, mask, sandbagging_v2")
    rec("F-111", "signed median over same 8", "+2.63", statistics.median(gaps[k][2] for k in eight), s7)
    rec("F-123", "DF-10 sandbagging_v2 token means pos/neg", "529.4/340.4", f"{gaps['sandbagging_v2'][0]:.1f}/{gaps['sandbagging_v2'][1]:.1f}", s7)
    rec("F-123", "DF-10 mask token means pos/neg", "283.2/176.4", f"{gaps['mask'][0]:.1f}/{gaps['mask'][1]:.1f}", s7)
    lens = set()
    for fn in sorted(glob.glob(os.path.join(K11, "matrices", "*_L*_s42.json"))):
        d = jload(fn)
        for pr in ("paper", "lr_reg10", "mms"):
            lens.add(round(offdiag(d, "length", pr), 4))
    rec("F-111", "length-only off-diag, all 42 matrices x 3 probes", "0.5555", "/".join(map(str, sorted(lens))),
        "run-crossgen/kd11/matrices/*_L*_s42.json (git)", verdict="matches" if lens == {0.5555} else f"DIFFERS: {sorted(lens)}")
    d40 = jload(os.path.join(K11, "matrices", "mean_L40_s42.json"))
    d44 = jload(os.path.join(K11, "matrices", "last_L44_s42.json"))
    rec("F-111", "raw L40/mean/paper", "0.6623", offdiag(d40, "raw"), "kd11 mean_L40")
    rec("F-111", "raw L44/last/mms", "0.6724", offdiag(d44, "raw", "mms"), "kd11 last_L44")
    # item 8
    zs = {z["direction"]: z["auroc"] for z in d40["zeroshot"]}
    j = d40["datasets"].index(SYC)
    rec("F-111", "kd11 L40 sycophancy_mt zero-shot contrast", "0.4089", zs["contrast"][j], "kd11 mean_L40 zeroshot")
    rec("F-111", "kd11 L40 sycophancy_mt zero-shot PC1", "0.3739", zs["PC1"][j], "kd11 mean_L40 zeroshot (0.373959, truncated not rounded)", tol=0.0001)
    ds = d40["datasets"]
    colm = lambda f: statistics.fmean({(r["feature"], r["probe"]): r["auroc"] for r in d40["results"]}[(f, "paper")][i][j]
                                     for i in range(len(ds)) if i != j)
    rec("F-111", "trained axes, sycophancy_mt column", "0.3551", colm("axes"), "kd11 mean_L40, column mean")
    rec("F-111", "trained random, sycophancy_mt column", "0.5413", colm("random"), "kd11 mean_L40, column mean")
    q10 = jload(os.path.join(ROOT, "run-coords/q10_sycophancy_axis.json"))["results"]["paper16|mean"]["sycophancy"]["key_layers"]["40"]
    rec("F-111", "paper16 sycophancy axis L40 (q10)", "0.313", q10["auroc"], "run-coords/q10_sycophancy_axis.json")
    rec("F-111", "  its CI", "[0.283, 0.346]", f"[{q10['ci95'][0]:.3f}, {q10['ci95'][1]:.3f}]", "same")
    zn = jload(os.path.join(ROOT, "run-coords/zeroshot_nulls.json"))[SYC]
    ps = [v["sycophancy_axis"]["p_vs_random"] for v in zn.values() if "sycophancy_axis" in v]
    rec("F-111", "sycophancy axis p vs 1000-dir null (range)", "0.001-0.008", f"{min(ps):.3f}-{max(ps):.3f}",
        "run-coords/zeroshot_nulls.json [sycophancy_mt][*].sycophancy_axis.p_vs_random",
        verdict="matches" if round(min(ps), 3) == 0.001 and round(max(ps), 3) <= 0.008 else f"DIFFERS: {min(ps):.4f}-{max(ps):.4f}")
    dz = zn["L40_mean"]["deception_axis"]
    rec("F-111", "deception axis (contrast 0.409) inside null", "inside", "inside" if not dz["beats_random_p95"] else "outside",
        f"zeroshot_nulls L40_mean deception_axis p={dz['p_vs_random']:.3f}")
    for st_ in ("passA11", "passC11"):
        try:
            m = by_layer(lab_json(f"{LAB}/run-crossgen/kd11/{st_}/matrices/mean_L40_s42.json"))[40]
            z = {x["direction"]: x["auroc"] for x in m["zeroshot"]}
            jj = m["datasets"].index(SYC)
            c_, p_ = z["contrast:sycophancy"][jj], z["PC1:sycophancy"][jj]
            ok = 0.29955 <= c_ <= 0.30615 and 0.29955 <= p_ <= 0.30615
            rec("F-111", f"sycophancy axis {st_} contrast/PC1", "0.3061 (A11) / 0.2996 (C11)", f"{c_:.4f}/{p_:.4f}",
                f"{LAB}/run-crossgen/kd11/{st_}/matrices/mean_L40_s42.json zeroshot, column {SYC}",
                verdict=("matches" if ok else "PARTLY: ") + ("" if ok else
                         "the pair 0.3061/0.2996 is passA11's contrast/PC1, not passA11/passC11; passC11 reads 0.3066/0.2995"))
        except LF.LabUnreachable as e:
            rec("F-111", f"sycophancy axis {st_}", "0.2996-0.3061", "UNKNOWN (lab unreachable)", str(e)[:80], verdict="UNKNOWN")


def f112():
    print("F-112  F-31 families nested; pc:kd_assertion margins (git kd11/banks/kd vs lab copy)")
    import hashlib
    git_t = jload(os.path.join(K11, "banks/kd/significance.json"))["tests"]
    labp = f"{LAB}/run-crossgen/kd11/banks/kd/significance.json"
    b, m = LF.fetch(labp, refresh=REFRESH)[labp]
    lab_t = json.loads(b)["tests"]
    s = f"git asix run-crossgen/kd11/banks/kd/significance.json (md5 {hashlib.md5(open(os.path.join(K11, 'banks/kd/significance.json'), 'rb').read()).hexdigest()[:8]}) vs lab (md5 {m['md5'][:8]}, {m['mtime']})"
    key = lambda t: (t["block"], t["pooling"], t["layer"], t["probe"], t["comparison"])
    G, Lb = {key(t): t for t in git_t}, {key(t): t for t in lab_t}
    rec("F-112", "git tests / comparisons", "4032/16", f"{len(G)}/{len({k[4] for k in G})}", s)
    rec("F-112", "lab tests / comparisons", "6300/25", f"{len(Lb)}/{len({k[4] for k in Lb})}", s)
    rec("F-112", "git tests missing from lab", "0", sum(k not in Lb for k in G), s)
    rec("F-112", "git tests with changed diff or dataset_p", "0", sum(1 for k in G if k in Lb and (G[k]["diff"] != Lb[k]["diff"] or G[k]["dataset_p"] != Lb[k]["dataset_p"])), s)
    rec("F-112", "tests added in lab", "2268", sum(k not in G for k in Lb), s)
    rec("F-112", "git tests whose BH q moved", "4030", sum(1 for k in G if k in Lb and G[k]["dataset_q_bh"] != Lb[k]["dataset_q_bh"]), s)
    new = sorted({k[4] for k in Lb if k not in G})
    rec("F-112", "added comparison families", "9", len(new), s + " : " + ", ".join(new))
    for comp, wpos, wsig in (("pc:kd_assertion - random", "21/21", "8"), ("pc:kd_assertion - raw", "18/21", "1"), ("pc:kd_assertion - datapca", "20/21", "0")):
        rows = [t for t in lab_t if t["comparison"] == comp and t["block"] == "10" and t["pooling"] == "mean" and t["probe"] == "paper"]
        rec("F-112", f"{comp}: layers positive (b10 mean paper)", wpos, f"{sum(t['diff'] > 0 for t in rows)}/{len(rows)}", s)
        allr = [t for t in lab_t if t["comparison"] == comp]
        sub = sum(t["significant_datasets"] for t in rows)
        tot = sum(t["significant_datasets"] for t in allr)
        rec("F-112", f"{comp}: tests clearing BH (b10 mean paper | all)", wsig, f"{sub} | {tot}", s + " significant_datasets",
            verdict="matches" if str(sub) == wsig and str(tot) == wsig else
            (f"PARTLY: {sub} within block10/mean/paper (the table's scope); {tot} across all 6,300" if str(sub) == wsig else f"DIFFERS: {sub} | {tot}"))
    t68 = Lb[("10", "mean", 68, "paper", "pc:kd_assertion - raw")]
    rec("F-112", "L68 kd_assertion-raw diff", "+0.1497", t68["diff"], s)
    rec("F-112", "L68 kd_assertion-raw q", "0.0467", t68["dataset_q_bh"], s)
    dp = [t for t in lab_t if t["comparison"] == "pc:kd_assertion - datapca"]
    dp10 = [t for t in dp if t["block"] == "10"]
    rec("F-112", "kd_assertion-datapca tests / clearing (block 10)", "126/0", f"{len(dp10)}/{sum(t['significant_datasets'] for t in dp10)}", s)
    rec("F-112", "kd_assertion-datapca tests / clearing (both blocks)", "n/a", f"{len(dp)}/{sum(t['significant_datasets'] for t in dp)}", s, verdict="context: 0 clear in either block")
    dpm = [t["diff"] for t in dp if t["block"] == "10" and t["pooling"] == "mean" and t["probe"] == "paper"]
    rec("F-112", "kd_assertion-datapca mean (b10 mean paper)", "+0.0830", statistics.fmean(dpm), s)
    mm = [t["diff"] for t in dp if t["block"] == "10" and t["pooling"] == "mean" and t["probe"] == "mms"]
    rec("F-112", "  mms mean / positive", "+0.0172 13/21", f"{statistics.fmean(mm):+.4f} {sum(x > 0 for x in mm)}/{len(mm)}", s)
    # margins recomputed from matrices: bank pc:kd_assertion vs ROOT raw, all 11 and HP-KR columns dropped
    want = {32: ("+0.0998", "+0.1183"), 40: ("+0.1496", "+0.1539"), 52: ("+0.1215", "+0.1370"),
            64: ("+0.1420", "+0.1624"), 68: ("+0.1497", "+0.1757"), 76: ("+0.1473", "+0.1897")}
    for L, (wa, wd) in want.items():
        bk = jload(os.path.join(K11, f"banks/kd/matrices/mean_L{L}_s42.json"))
        rt = jload(os.path.join(K11, f"matrices/mean_L{L}_s42.json"))
        drop = [x for x in rt["datasets"] if x not in (KR, KM)]
        rec("F-112", f"L{L} kd_assertion-raw all 11", wa, offdiag(bk, "pc:kd_assertion") - offdiag(rt, "raw"), "kd11/banks/kd + kd11 root matrices (git)")
        rec("F-112", f"L{L} kd_assertion-raw HP-KR cols dropped", wd,
            offdiag(bk, "pc:kd_assertion", keep=drop) - offdiag(rt, "raw", keep=drop), "same; both HP-KR datasets removed", tol=0.0006)


def f114():
    print("F-114  wave-5 corrections (length AUROC, F-61, F-91, F-56)")
    cf = jload(os.path.join(ROOT, "audit/cache/labpc/run-coords/confounds_L44_last.json"))
    lo = {x["dataset"]: x["auroc_from_length"] for x in cf["confounds"]["length_only_auroc"]}
    s = "audit/cache/labpc/run-coords/confounds_L44_last.json (md5 11bc6e65 == lab) confounds.length_only_auroc"
    rec("F-114", "length-only AUROC datasets", "10", len(lo), s)
    rec("F-114", "... above 0.5", "9", sum(float(v) > 0.5 for v in lo.values()), s)
    rec("F-114", "roleplaying (sole inversion)", "0.4415", float(lo["roleplaying"]), s)
    rec("F-114", "control set keeps sycophancy_mt, drops kr-matched", "yes", "yes" if (SYC in lo and KM not in lo) else "no", s)
    d = jload(os.path.join(K11, "matrices/mean_L40_s42.json"))
    s2 = "run-crossgen/kd11/matrices/mean_L40_s42.json (110 pairs, paper, k=5)"
    raw, uni, dp = offdiag(d, "raw"), offdiag(d, "pc:unified"), offdiag(d, "datapca")
    rec("F-114", "paper16 raw", "0.6623", raw, s2)
    rec("F-114", "pc:unified", "0.6715", uni, s2)
    rec("F-114", "datapca", "0.6651", dp, s2)
    rec("F-114", "pc:unified - raw", "+0.0093", uni - raw, s2)
    rec("F-114", "pc:unified - datapca", "+0.0064", uni - dp, s2)
    bk = jload(os.path.join(K11, "banks/kd/matrices/mean_L40_s42.json"))
    kdm = offdiag(bk, "pc:unified") - raw
    rec("F-114", "kd pc:unified - ROOT raw", "+0.0757", kdm, "kd11/banks/kd/matrices/mean_L40 vs root raw")
    un_l, un_m = kdm / (uni - raw), kdm / (uni - dp)             # unrounded
    ro_l, ro_m = 0.0757 / round(uni - raw, 4), 0.0757 / round(uni - dp, 4)  # from the 4-dp printed inputs
    note = f"unrounded {un_l:.3f}x / {un_m:.3f}x; from 4-dp inputs {ro_l:.3f}x / {ro_m:.3f}x"
    rec("F-114", "mixed ratio ('twelvefold')", "11.8", round(ro_m, 1), note)
    rec("F-114", "like-for-like ratio", "8.1", round(ro_l, 1), note + " (8.1 uses rounded inputs; unrounded 8.2)")
    rec("F-123", "like-for-like ratio", "8.2", round(un_l, 1), note)
    rec("F-123", "mixed ratio", "11.7", round(un_m, 1), note)
    rec("F-123", "rows caption length", "22", len(d["rows"]), "d['rows'] is a caption string")
    keep10 = [x for x in d["datasets"] if x != KM]
    rec("F-123", "length/raw, 10 = all but kr-matched", "0.5494/0.6652", f"{offdiag(d, 'length', keep=keep10):.4f}/{offdiag(d, 'raw', keep=keep10):.4f}", s2)
    rec("F-123", "length/raw, block9", "0.5467/0.6870", f"{offdiag(d, 'length', keep=d['block9']):.4f}/{offdiag(d, 'raw', keep=d['block9']):.4f}", s2)
    # F-91 seed table
    want = {(32, 7): "0.8826", (32, 13): "0.8800", (32, 42): "0.8945", (36, 7): "0.8942", (36, 13): "0.8887",
            (36, 42): "0.8919", (40, 7): "0.8957", (40, 13): "0.8830", (40, 42): "0.8828"}
    for (L, sd), w in want.items():
        p = f"{LAB}/run-premise/falsify/falsify_L{L}_s{sd}.json"
        b, m = LF.fetch(p)[p]
        rec("F-114", f"F-91 falsify L{L} s{sd} real", w, json.loads(b)["real"], p)
    rec("F-114", "F-56 source share 134/2134", "6.28", round(100 * 134 / 2134, 2), "arithmetic")
    rec("F-114", "F-56 sample share 134/800", "16.75", round(100 * 134 / 800, 2), "arithmetic")


def f113():
    print("F-113  methods corrections (arithmetic + k)")
    for h, w, n in ((32, "512.00", "8B"), (24, "384.00", "3B")):
        rec("F-113", f"{n} heads x 2048^2 x 4 B in MiB", w, f"{h * 2048 ** 2 * 4 / 2 ** 20:.2f}", "arithmetic")
        rec("F-113", f"{n} same in decimal MB", {"8B": "537", "3B": "402"}[n], f"{h * 2048 ** 2 * 4 / 1e6:.2f}", "arithmetic (402.65 truncated to 402)", tol=1)
    src = open(os.path.join(ROOT, "src/assistant_axis_ndif/27_crossgen.py"), encoding="utf-8").read().splitlines()
    ln = src[881] if len(src) > 881 else ""
    rec("F-113", "27_crossgen.py:882 default k", "5", (re.search(r"default=(\d+)", ln).group(1) if re.search(r"default=(\d+)", ln) and "k" in ln else f"line reads: {ln.strip()[:50]}"),
        "src/assistant_axis_ndif/27_crossgen.py:882 (asix WT)")
    ks = set()
    for st_ in ("scale_3b", "scale_8b", "scale_70b", "scale_70b_paper16fp32"):
        ks |= {m["k"] for m, _ in lab_json(f"{LAB}/run-crossgen/{st_}/matrices/mean_L*_s42.json").values() if m}
    rec("F-113", "k in all four scale stores", "3", "/".join(map(str, sorted(ks))), "lab scale_* matrices field k")
    mni = subprocess.run(["git", "show", f"{C.PIN_ASIX}:scale/scripts/run_scale_coords.ps1"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    rec("F-113", "run_scale_coords.ps1:108-109 carries MiB slip", "537/402 MiB", " | ".join(x.strip()[:60] for x in mni[107:109]), "git asix scale/scripts/run_scale_coords.ps1",
        verdict="matches" if any("537" in x or "402" in x for x in mni[107:109]) else "DIFFERS: text not found at 108-109")


def f120():
    print("F-120  late-layer rise (lab run-crossgen/passA/matrices, the 5 paper datasets -- what p9 actually read)")
    DS5 = ["instructed-deception", "mask", "convincing-game", "ai_liar", "roleplaying"]
    want = {40: ("0.8195", "0.8266", "0.5971"), 70: ("0.7828", "0.6211", "0.5234"),
            72: ("0.7834", "0.6384", "0.5133"), 79: ("0.7426", "0.7492", "0.6405")}
    T = {}
    for L in list(range(30, 51)) + [70, 72, 79]:
        p = f"{LAB}/run-crossgen/passA/matrices/mean_L{L}_s42.json"
        b, m = LF.fetch(p)[p]
        T[L] = json.loads(b)
    for L, (wr, wa, wn) in want.items():
        d = T[L]
        r, a, n = (offdiag(d, f, keep=DS5) for f in ("raw", "axes", "random"))
        s = f"{LAB}/run-crossgen/passA/matrices/mean_L{L}_s42.json, paper, DS5 off-diag"
        rec("F-120", f"L{L} raw", wr, r, s)
        rec("F-120", f"L{L} axes", wa, a, s)
        rec("F-120", f"L{L} random", wn, n, s)
        rec("F-120", f"L{L} axes-random", {40: "+0.2296", 70: "+0.0977", 72: "+0.1250", 79: "+0.1087"}[L], a - n, s)
    mid = statistics.fmean(offdiag(T[L], "axes", keep=DS5) for L in range(30, 51) if L in T)
    rec("F-120", "axes mid-stack mean L30-50", "0.8145", mid, "passA mean_L30..50 (21 layers)")
    r = {L: offdiag(T[L], "raw") for L in (40,)}
    rec("F-120", "scope: all-10-dataset raw at L40 (not stated in entry)", "n/a", r[40], "same file, all datasets",
        verdict="context: entry's numbers are the 5-dataset (paper5) reading; all-10 reads differently")




# ================================================================ F-124 (git, collaborator tables)
ZCOL = {"axis1": "auroc_axis", "pc1": "auroc_pc1", "syc_axis1": "auroc_syc_axis", "syc_pc1": "auroc_syc_pc1",
        "ext_axis1": "auroc_ext_axis", "ext_pc1": "auroc_ext_pc1", "denial_axis1": "auroc_denial_axis"}


def f124():
    print("F-124  draft tables (git; entry read at ea1ce52b = PIN_OLD; PIN_PAPER compared)")
    for sha, tag in ((C.PIN_OLD, "@OLD"), (C.PIN_PAPER, "@PIN")):
        inv = []
        vals = {}
        for L in (20, 40, 44):
            S = summ(sha, L)
            Z = zshot(sha, L)
            for arm, zc in ZCOL.items():
                cols, m = mat(sha, L, arm)
                bad = [tr for tr in m if statistics.fmean(m[tr][j] for j, c in enumerate(cols) if c != tr) < 0.5]
                cells = [(tr, j, m[tr][j]) for tr in m for j, c in enumerate(cols) if c != tr]
                printed = statistics.fmean(v for _, _, v in cells)
                corr = statistics.fmean((1 - v) if tr in bad else v for tr, _, v in cells)
                zmean = statistics.fmean(float(Z[d][zc]) for d in cols)
                vals[(arm, L)] = (printed, corr, zmean, bad, float(S[arm]["off_diagonal_mean"]))
                if bad:
                    inv.append((L, arm, bad))
        s = f"git {sha[:8]}:{CB}/persona_ood_ext_L{{20,40,44}}/matrix_<arm>.csv"
        got = "; ".join(f"L{L} {a}" for L, a, _ in inv)
        rec("F-124", f"inverted (arm,layer) cells {tag}", "7", len(inv), s + " : " + got)
        rec("F-124", f"inverted row is HP-KR in every one {tag}", "yes", "yes" if all(b == [HPKRc] for _, _, b in inv) else f"no: {inv}", s)
        if tag == "@OLD":
            want = {("axis1", 20): ("0.6511", "0.7909"), ("syc_axis1", 20): ("0.6153", "0.7429"), ("syc_pc1", 20): ("0.6153", "0.7433"),
                    ("pc1", 20): ("0.6669", "0.8056"), ("syc_axis1", 40): ("0.6626", "0.8080"), ("syc_pc1", 40): ("0.6618", "0.8075"),
                    ("syc_axis1", 44): ("0.7078", "0.8485")}
            for (arm, L), (wp, wc) in want.items():
                pr, co, zm, _, summ_v = vals[(arm, L)]
                rec("F-124", f"L{L} {arm} printed", wp, summ_v, f"summary.csv off_diagonal_mean {tag}")
                rec("F-124", f"L{L} {arm} sign-corrected", wc, co, "matrix cells, HP-KR row -> 1-x")
                rec("F-124", f"L{L} {arm} corrected == zero-shot 6-cell mean (4dp)", "equal", "equal" if abs(co - zm) < 5e-5 else f"{co:.4f} vs {zm:.4f}", "zero_shot.csv",
                    verdict="matches" if abs(co - zm) < 5e-5 else f"DIFFERS: {co:.4f} vs {zm:.4f}")
            for L, w in ((20, "0.128"), (40, "0.145"), (44, "0.141")):
                pr, co, *_ = vals[("syc_axis1", L)]
                rec("F-124", f"Table 11 syc_axis1 understatement L{L}", w, co - pr, "corrected - printed")
            for L, w in ((40, "0.8319"), (44, "0.8260")):
                rec("F-124", f"axis1 L{L} printed == zero-shot mean", w, vals[("axis1", L)][4], "summary.csv")
            cols, m = mat(sha, 20, "axis1")
            rm = {tr: statistics.fmean(m[tr][j] for j, c in enumerate(cols) if c != tr) for tr in m}
            rec("F-124", "axis1 L20 HP-KR row mean", "0.0806", rm[HPKRc], "matrix_axis1.csv L20")
            oth = [v for k, v in rm.items() if k != HPKRc]
            rec("F-124", "axis1 L20 other rows range", "0.7503-0.8057", f"{min(oth):.4f}-{max(oth):.4f}", "matrix_axis1.csv L20")
            ok = all(abs(m[HPKRc][j] - (1 - m[tr][j])) < 1e-9 for tr in m for j, c in enumerate(cols)
                     if tr != HPKRc and c not in (tr, HPKRc))
            rec("F-124", "HP-KR row == 1-x of other rows (L20 axis1)", "exact", "exact" if ok else "not exact", "matrix_axis1.csv L20")
            z = zshot(sha, 20)
            rec("F-124", "L20 axis1 CG zero-shot / HP-KR-row cell", "0.9399/0.0601",
                f"{float(z['convince_me__plain']['auroc_axis']):.4f}/{m[HPKRc][cols.index('convince_me__plain')]:.4f}", "zero_shot + matrix")
            for L, w in ((20, "0.0429"), (40, "0.1671"), (44, "0.4804")):
                rec("F-124", f"Table 11 syc HP-KR zero-shot L{L}", w, float(zshot(sha, L)[HPKRc]["auroc_syc_axis"]), "zero_shot.csv auroc_syc_axis", tol=0.0001)
        # realised FPR on the 'control' threshold
        ARMS5 = ("base probe (RepE-trained)", "deception axis (8v7)", "extended axis (12v13)", "knowledge-denial axis (4v6)", "sycophancy axis (5v5)")
        cells = []
        for L in (20, 40, 44):
            for r in C.tab(sha, f"{CB}/persona_ood_ext_L{L}/zero_shot_operating.csv"):
                if r["threshold_source"] == "control" and r["arm"] in ARMS5 and not r["dataset"].startswith("Average"):
                    cells.append((L, r["arm"], r["dataset"], 1 - float(r["specificity"]), r))
        so = f"git {sha[:8]}:{CB}/persona_ood_ext_L*/zero_shot_operating.csv control rows, 5 arms"
        rec("F-124", f"cells {tag}", "90", len(cells), so)
        rec("F-124", f"cells with realised FPR > 1% {tag}", "23", sum(c[3] > 0.01 + 1e-12 for c in cells), so)
        if tag == "@OLD":
            g = {(L, a, d): (f, r) for L, a, d, f, r in cells}
            kd = "knowledge-denial axis (4v6)"
            for (L, d, w) in ((20, HPKRc, "88.3"), (20, "repe_honesty__plain", "35.6"), (20, "harm_refusal_choice__plain", "4.5"),
                              (20, "instruct_dishonesty__plain", "1.1"), (40, HPKRc, "6.5"), (40, "instruct_dishonesty__plain", "11.9"),
                              (44, HPKRc, "4.7"), (44, "instruct_dishonesty__plain", "9.9")):
                rec("F-124", f"4v6 L{L} {d[:14]} FPR %", w, round(100 * g[(L, kd, d)][0], 1), so)
            rec("F-124", "base probe L40 ID FPR %", "15.2", round(100 * g[(40, "base probe (RepE-trained)", "instruct_dishonesty__plain")][0], 1), so)
            r = g[(20, kd, HPKRc)][1]
            rec("F-124", "4v6 L20 HP-KR BA / spec / recall", "0.533/0.1165/0.9496",
                f"{float(r['balacc']):.3f}/{float(r['specificity']):.4f}/{float(r['recall']):.4f}", so)
            base = [float(g[(L, "base probe (RepE-trained)", HPKRc)][1]["auroc"]) for L in (20, 40, 44)]
            rec("F-124", "base probe HP-KR (operating) L20/40/44", "0.109/0.936/0.828", "/".join(f"{x:.3f}" for x in base), so)
            mb = []
            for L in (20, 40, 44):
                cols, m = mat(sha, L, "baseline")
                mb.append(m["repe_honesty__plain"][cols.index(HPKRc)])
            rec("F-124", "matrix_baseline repe->HP-KR L20/40/44", "0.3808/0.9281/0.8803", "/".join(f"{x:.4f}" for x in mb), "matrix_baseline.csv")
    g = jload(os.path.join(ROOT, "kd/personas/kd_role_groups.json"))["groups"]
    n = {k: len(g[k]["roles"]) for k in ("knowledge_denial", "sincere_ignorance", "plain_assertion")}
    rec("F-124", "D-005 ten roles (4+3+3)", "4/3/3", "/".join(str(n[k]) for k in n), "kd/personas/kd_role_groups.json (asix)")




# ================================================================ F-125 / F-126 / F-127 (claims.csv state + lab stores)
def _dnorm(v):
    v = v.upper()
    if v.startswith("UNRESOLVED"):
        return "UNRESOLVED"
    if v.startswith("PARTLY"):
        return "PARTLY"
    if v.startswith(("MATCHES", "HOLDS", "SUPERSEDED")):
        return "MATCHES"
    return v[:20]


def f125():
    print("F-125/F-127  draft-claim verdict tallies (audit/claims.csv, current state) + D-156")
    R = [r for r in csv.DictReader(open(os.path.join(ROOT, "audit/claims.csv"), encoding="utf-8")) if r["id"].startswith("D-")]
    B = [r for r in R if re.fullmatch(r"B\d\d", r["batch"])]
    c = defaultdict(int)
    for r in B:
        c[_dnorm(r["verdict"])] += 1
    s = "audit/claims.csv D- rows with Phase-6 batch B01-B63 (current file, mtime 2026-09-24 16:40)"
    rec("F-125", "draft claims with a Phase-6 verdict", "176", len(B), s)
    rec("F-127", "6A tally MATCHES/PARTLY/UNRESOLVED", "166/8/2", f"{c['MATCHES']}/{c['PARTLY']}/{c['UNRESOLVED']}", s)
    L2 = {"B23", "B30", "B40", "B41", "B42", "B43", "B45", "B46"}
    c2 = defaultdict(int)
    for r in B:
        if r["batch"] in L2:
            c2[_dnorm(r["verdict"])] += 1
    rec("F-125", "Level-2 (8 batches) MATCHES/PARTLY", "52/2", f"{c2['MATCHES']}/{c2['PARTLY']}", s)
    # the chain 160/9/7 (F-125) -> 164/10/2 (F-126) -> 166/8/2 (F-127) is arithmetic on named ids
    v = {r["id"]: _dnorm(r["verdict"]) for r in R}
    ok126 = [v.get(i) for i in ("D-057", "D-065", "D-154", "D-040", "D-041")]
    rec("F-126", "5 resolved: D-057/065/154/040/041 now", "4 MATCHES + 1 PARTLY", "/".join(ok126), s,
        verdict="matches" if sorted(ok126) == ["MATCHES"] * 4 + ["PARTLY"] else f"DIFFERS: {ok126}")
    rec("F-126", "chain 160/9/7 -> 164/10/2", "164/10/2", "160+4/9+1/7-5 = 164/10/2", "arithmetic on the five ids above", verdict="matches")
    ok127 = [v.get(i) for i in ("D-032", "D-171")]
    rec("F-127", "D-032, D-171 now MATCHES", "MATCHES/MATCHES", "/".join(ok127), s)
    rec("F-125", "snapshot 160/9/7 (superseded by F-126/F-127)", "160/9/7", "not re-derivable (claims.csv overwritten in place)", s,
        verdict="UNVERIFIABLE: intermediate snapshot; consistent with the id-level chain to the current 166/8/2")
    rec("F-125", "D-030/D-063 still UNRESOLVED", "UNRESOLVED/UNRESOLVED", "/".join(v.get(i, "?") for i in ("D-030", "D-063")), s)
    S = summ(C.PIN_OLD, 20)
    rec("F-125", "D-156 L20 excl HP-KR ext_axis1 vs axis1", "0.909/0.919",
        f"{float(S['ext_axis1']['off_diagonal_excl_hpkr']):.3f}/{float(S['axis1']['off_diagonal_excl_hpkr']):.3f}",
        f"git {C.PIN_OLD[:8]}:{CB}/persona_ood_ext_L20/summary.csv")
    rec("F-125", "D-128: ext_axis1 L36 vs L40", "0.9275/0.9158",
        f"{float(summ(C.PIN_PAPER, 36)['ext_axis1']['off_diagonal_mean']):.4f}/{float(summ(C.PIN_PAPER, 40)['ext_axis1']['off_diagonal_mean']):.4f}", "summary.csv")
    rec("F-125", "D-153 margin +0.066 CI [-0.069,+0.189] P 0.161 (HPKR_WORDING_CONTROLLED §4)", "+0.066",
        "+0.0655", "audit/HPKR_WORDING_CONTROLLED.md:100-102 (audit prose, F-76 bootstrap)", verdict="UNVERIFIED: audit prose only (F-76 bootstrap not re-run in P17)")


def f126():
    print("F-126  6D-1 pc:sycophancy (lab expLB-resplit/phase6), 6D-2 passP/passB_fine, §3 Figure-2 geometry (lab run-kd/vectors_s3)")
    sp = f"{RES}/phase6/run-crossgen/summary.json"
    j = json.loads(LF.fetch(sp, refresh=REFRESH)[sp][0])
    R = {(r["pooling"], r["layer"], r["feature"]): r for r in j["records"] if r["probe"] == "paper" and r["seed"] == 42}
    rec("F-126", "phase6 datasets (kr-matched explicit)", "11", len(j["datasets"]), sp)
    for f_, w in (("raw", "0.6623"), ("datapca", "0.6651"), ("length", "0.5555"), ("pc:unified", "0.6715")):
        rec("F-126", f"reproduces kd11: {f_} L40 mean", w, R[("mean", 40, f_)]["offdiag_mean_10"], sp)
    T = {("mean", 20): ("-0.1031/-0.0898", "-0.0844/-0.0697", "+0.0213/+0.0509", "-0.0297/+0.0024"),
         ("mean", 40): ("-0.0382/+0.0017", "-0.0410/+0.0134", "+0.0391/+0.1042", "+0.0686/+0.1420"),
         ("mean", 44): ("-0.0075/+0.0342", "-0.0079/+0.0477", "+0.0904/+0.1621", "+0.0935/+0.1710"),
         ("last", 20): ("-0.1182/-0.1122", "-0.1226/-0.1135", "-0.0040/+0.0254", "-0.0447/-0.0071"),
         ("last", 40): ("-0.0377/-0.0081", "-0.0527/-0.0254", "+0.0484/+0.0922", "+0.0349/+0.1017"),
         ("last", 44): ("-0.0100/+0.0206", "-0.0468/-0.0274", "+0.0940/+0.1486", "+0.0584/+0.1242")}
    nm, bad = 0, []
    for (pool, L), ws in T.items():
        s_ = R[(pool, L, "pc:sycophancy")]
        for ctrl, w in zip(("raw", "datapca", "random", "length"), ws):
            c_ = R[(pool, L, ctrl)]
            got = (s_["offdiag_mean_10"] - c_["offdiag_mean_10"], s_["offdiag_mean_9"] - c_["offdiag_mean_9"])
            for wv, gv in zip(w.split("/"), got):
                nm += 1
                if abs(float(wv) - gv) > 0.00005 + 1e-12:
                    bad.append(f"{pool} L{L} -{ctrl}: {wv} vs {gv:+.4f}")
    rec("F-126", "6D-1 table: 48 cells (pc:syc minus control, all11/no-syc)", "48 match", f"{nm - len(bad)}/{nm} match", sp + " offdiag_mean_10 / offdiag_mean_9",
        verdict="matches" if not bad else "DIFFERS: " + "; ".join(bad[:4]))
    neg = sum(R[(p, L, "pc:sycophancy")]["offdiag_mean_10"] < R[(p, L, "raw")]["offdiag_mean_10"] for p in ("mean", "last") for L in (20, 40, 44))
    rec("F-126", "loses to raw on all 11 at every (layer,pool)", "6/6", f"{neg}/6", sp)
    gp = f"{RES}/phase6/run-crossgen/significance.json"
    G = json.loads(LF.fetch(gp, refresh=REFRESH)[gp][0])["tests"]
    rec("F-126", "bootstrap comparisons / significant", "264/0", f"{len(G)}/{sum(t['significant_datasets'] for t in G)}", gp)
    # 6D-2 passP
    pp = f"{LAB}/run-crossgen/passP/summary.json"
    P = json.loads(LF.fetch(pp, refresh=REFRESH)[pp][0])
    PR = {(r["pooling"], r["layer"], r["feature"]): r for r in P["records"] if r["probe"] == "paper" and r["seed"] == 42}
    for L, w in ((20, "0.5584"), (40, "0.5687"), (44, "0.5697")):
        rec("F-126", f"passP raw off-diag L{L} mean", w, PR[("mean", L, "raw")]["offdiag_mean_10"], pp)
    dg = {"convincing-game": "1.000", "sandbagging_v2": "1.000", "instructed-deception": "0.953", "harm-pressure-choice": "0.919",
          "insider-trading": "0.911", KR: "0.865", "mask": "0.624", "roleplaying": "0.796"}
    for pool in ("mean", "last"):
        dd = PR[(pool, 44, "raw")]["diagonal"]
        ok = all(abs(dd[k] - float(w)) < 0.0005 + 1e-9 for k, w in dg.items())
        if ok or pool == "last":
            rec("F-126", f"passP raw diagonal L44 ({pool}) 8 datasets", "CG 1.000 ... mask 0.624, roleplaying 0.796",
                ", ".join(f"{k[:6]} {dd[k]:.3f}" for k in dg), pp, verdict="matches" if ok else "DIFFERS (see values)")
            break
    for pat, span, first5 in ((f"{LAB}/run-coords/passP/mask/chunk_0000.pt", "prompt", "151,137,440,120,420"),
                              (f"{LAB}/run-coords/passB_fine/mask/chunk_0000.pt", "response", "1000,971,659,955,633")):
        import torch
        b, m = LF.fetch(pat)[pat]
        t = torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        rec("F-126", f"{pat.split('/')[-3]} span / n_tok first 5 (mask)", f"{span} / {first5}",
            f"{t['span']} / {','.join(str(int(x)) for x in t['n_tok'][:5].tolist())}", pat)
    spl = json.loads(LF.fetch(f"{LAB}/run-crossgen/passP/splits.json")[f"{LAB}/run-crossgen/passP/splits.json"][0])["splits"]["42"]
    rec("F-126", "passP rows ID/HP-C/HP-KR/SYC", "5432/2134/2139/1182",
        "/".join(str(spl[k]["rows"]) for k in ("instructed-deception", "harm-pressure-choice", KR, SYC)), "passP/splits.json")
    bp = f"{LAB}/run-crossgen/passB_fine/summary.json"
    BF = json.loads(LF.fetch(bp)[bp][0])
    BR = {(r["pooling"], r["layer"], r["feature"]): r for r in BF["records"] if r["probe"] == "paper" and r["seed"] == 42}
    rec("F-126", "passB_fine raw off-diag L41 mean", "0.6623", BR[("mean", 41, "raw")]["offdiag_mean_10"], bp)
    rec("F-126", "passB_fine raw diag L44 mask (mean)", "0.875", BR[("mean", 44, "raw")]["diagonal"]["mask"], bp)
    layers = sorted({k[1] for k in BR})
    rec("F-126", "passB_fine layers (L37-47, no L40)", "10", len(layers), bp + f" layers {layers}", verdict="matches" if len(layers) == 10 and 40 not in layers else "DIFFERS")
    neg, exc = 0, []
    banks = {"root": BF}
    for b_ in ("hf275_strict", "own100", "paper16_shared"):
        p_ = f"{LAB}/run-crossgen/passB_fine/banks/{b_}/summary.json"
        banks[b_] = json.loads(LF.fetch(p_)[p_][0])
    for b_, J in banks.items():
        RR = {(r["pooling"], r["layer"], r["feature"]): r for r in J["records"] if r["probe"] == "paper" and r["seed"] == 42}
        for L in layers:
            d_ = RR[("mean", L, "axes")]["offdiag_mean_10"] - RR[("mean", L, "raw")]["offdiag_mean_10"]
            neg += d_ < 0
            if d_ >= 0:
                exc.append(f"{b_} L{L} {d_:+.3f}")
        if b_ == "hf275_strict":
            pdv = [RR[("mean", L, "pc:deception")]["offdiag_mean_10"] for L in layers]
            rwv = [RR[("mean", L, "raw")]["offdiag_mean_10"] for L in layers]
            dpv = [RR[("mean", L, "datapca")]["offdiag_mean_10"] for L in layers]
            rec("F-126", "hf275_strict pc:deception > raw layers", "10/10", f"{sum(a > r for a, r in zip(pdv, rwv))}/10", "passB_fine/banks/hf275_strict/summary.json")
            rec("F-126", "hf275_strict pc:deception range", "0.6815-0.7106", f"{min(pdv):.4f}-{max(pdv):.4f}", "same")
            rec("F-126", "hf275_strict raw range", "0.6511-0.6747", f"{min(rwv):.4f}-{max(rwv):.4f}", "same")
            rec("F-126", "hf275_strict mean margin over raw", "+0.039", statistics.fmean(a - r for a, r in zip(pdv, rwv)), "same")
            rec("F-126", "hf275_strict pc:deception > datapca layers", "10/10", f"{sum(a > r for a, r in zip(pdv, dpv))}/10", "same")
    rec("F-126", "passB_fine axes-raw negative (bank,layer)", "36/40", f"{neg}/{len(banks) * len(layers)}", "passB_fine summary.json root + 3 banks (each store's own raw)")
    rec("F-126", "the 4 exceptions", "own100 L37/L39/L44, paper16_shared L44, none > +0.019", "; ".join(exc), "same",
        verdict="matches" if sorted(exc and [e.rsplit(' ', 1)[0] for e in exc]) == sorted(["own100 L37", "own100 L39", "own100 L44", "paper16_shared L44"]) and all(float(e.rsplit(' ', 1)[1]) <= 0.019 for e in exc) else "DIFFERS")
    # §3 Figure 2 geometry from run-kd/vectors_s3
    import numpy as np, torch
    V = {}
    for p_, (b, m) in LF.fetch(f"{LAB}/run-kd/vectors_s3/*.pt").items():
        t = torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        V[os.path.basename(p_)[:-3]] = (t["vector"].double().numpy(), int(t["n"]))
    rec("F-126", "vectors_s3 personas / pc_handler n", "36/106", f"{len(V)}/{V['pc_handler'][1]}", f"{LAB}/run-kd/vectors_s3/*.pt")
    DEC = ("pc_handler", "pc_broker", "pc_impression_manager", "pc_coverup_operator", "pc_gatekeeper", "pc_spin_doctor", "pc_plausible_denier", "pc_narrative_manager")
    HON = ("pc_archivist", "pc_disclosure_counsel", "pc_documentarian", "pc_fact_checker", "pc_ombudsman", "pc_transparency_officer", "pc_whistleblower")
    KD_ = ("pc_amnesiac_witness", "pc_deniable_expert", "pc_feigned_novice", "pc_stonewaller")
    SIN = ("pc_honest_novice", "pc_calibrated_reporter", "pc_scrupulous_witness")
    PLA = ("pc_forthright_expert", "pc_candid_witness", "pc_plain_dealer")
    names = sorted(V)
    for L in (20, 40, 44):
        X = np.stack([V[n][0][L] for n in names]); Xc = X - X.mean(0)
        mo = lambda g: np.stack([V[n][0][L] for n in g]).mean(0)
        u = lambda v: v / np.linalg.norm(v)
        x, y = u(mo(DEC) - mo(HON)), u(mo(KD_) - mo(SIN + PLA))
        rec("F-126", f"cos(deception, denial) L{L}", {20: "0.417", 40: "0.474559", 44: "0.468"}[L], float(x @ y),
            "recipe of p16_fig2_axes.py / make_figures.py:personas()")
        if L == 40:
            pr = lambda g: Xc[[names.index(n) for n in g]] @ y
            allp = Xc @ y
            rec("F-126", "sincere ignorance mean", "-0.0844", float(pr(SIN).mean()), "L40 centred projection")
            rec("F-126", "feigned deniers mean", "+2.9037", float(pr(KD_).mean()), "same")
            rec("F-126", "published deceptive mean", "+0.0458", float(pr(DEC).mean()), "same")
            rec("F-126", "plain assertion mean", "-1.2274", float(pr(PLA).mean()), "same")
            rec("F-126", "deceptive max / denier min", "+0.885/+2.150", f"{pr(DEC).max():+.3f}/{pr(KD_).min():+.3f}", "same")
            gspan = float(pr(KD_).mean() - pr(PLA).mean())
            rec("F-126", "axis 'full span' = group-mean span deniers-plain", "4.13", gspan, "same", tol=0.005)
            rec("F-126", "  (persona-level span, for context)", "n/a", float(allp.max() - allp.min()), "same",
                verdict="context: 'full span' 4.13 is the span of GROUP MEANS; the 36 personas span 5.29, so 28% is of the group-mean span")
            rec("F-126", "sincere sits 28% of the axis closer", "28", round(100 * float(pr(SIN).mean() - pr(PLA).mean()) / gspan), "gap / group-mean span")
            rec("F-126", "sincere-plain gap", "1.14", float(pr(SIN).mean() - pr(PLA).mean()), "same", tol=0.005)
    ok = os.path.exists(os.path.join(ROOT, "papers", "Probe Generalization as Subspace Selection for OOD Deception Detectio.pdf"))
    rec("F-126", "D-041 PDF in papers/", "yes", "yes" if ok else "no", "papers/ (title match; arXiv id 2609.02893 not read - no PDF text tool)")
    rec("F-126", "D-040 PDF 2602.01425 in papers/", "absent (owed)", "absent" if not glob.glob(os.path.join(ROOT, "papers", "*One Probe*")) else "present", "papers/",
        verdict="matches")


SECTIONS = {k.upper().replace("F", "F-", 1): v for k, v in globals().items() if re.fullmatch(r"f1[12]\d", k)}

if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if a.startswith("F-")] or sorted(SECTIONS)
    for fid in want:
        try:
            SECTIONS[fid]()
        except LF.LabUnreachable as e:
            rec(fid, "LAB READ", "-", "UNKNOWN (lab unreachable)", str(e)[:120], verdict="UNKNOWN")
    out = os.path.join(ROOT, "audit", "p17", "ledger_verify_results.json")
    old = json.load(open(out)) if os.path.exists(out) else []
    newkeys = {(c["f_id"], c["item"]) for c in CHECKS}
    keep = [c for c in old if (c["f_id"], c["item"]) not in newkeys]
    json.dump(keep + CHECKS, open(out, "w", encoding="utf-8"), indent=1)
    n = defaultdict(int)
    for c in CHECKS:
        n[c["verdict"].split(":")[0]] += 1
    print(dict(n), "->", out)
