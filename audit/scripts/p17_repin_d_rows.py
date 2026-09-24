"""P17 Agent A, task 1: which commit's data do the 176 D- rows stamped `origin/paper @ ea1ce52b`
actually reproduce, and do their verdicts still hold at PIN_PAPER?

Method (per row):
 1. Resolve every branch file the row's evidence_file names (expanding L{20,40,44} and matrix_*),
    and hash it at PIN_OLD, PIN_P2AD and PIN_PAPER. Evidence outside the branch (kd/, papers/,
    audit/cache, lab PC) cannot depend on the SHA.
 2. If every resolved branch file is byte-identical at the three SHAs, the stored value is
    reproduced by all three: AMBIGUOUS-identical.
 3. Otherwise, collect every cell that differs between PIN_OLD and PIN_P2AD in those files (and the
    per-test column means of matrix_baseline / matrix_balacc_baseline, which Table 7 quotes), format
    old and new at 3 and 4 dp (half-up and Python rounding), keep only strings that differ, and look
    for them in the stored level1_value/level2_value. Also look for rows that exist only at
    2ad758fb+ (unified axis 13v12, uni_*, the 'measured here' Follow-Up row).
    Old-only hit -> ea1ce52b; new-only hit -> 2ad758fb (== 16def2de: no result file changed after
    2ad758fb); neither -> AMBIGUOUS-identical (the changed cells are not the ones the row quotes).
 4. The pinned re-runs audit/p17/level{1,2}_<sha>.txt are compared as a whole-output cross-check.
Writes audit/p17/d_rows_repinned.csv.
"""
import csv, hashlib, io, re, subprocess, sys, pathlib
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import p17_common as C

SHAS = {"ea1ce52b": C.PIN_OLD, "2ad758fb": C.PIN_P2AD, "16def2de": C.PIN_PAPER}
P17 = C.ROOT / "audit" / "p17"


def ls(sha, prefix):
    r = subprocess.run(["git", "ls-tree", "-r", "--name-only", sha, "--", prefix], cwd=C.ROOT,
                       capture_output=True, text=True, encoding="utf-8")
    return r.stdout.splitlines()


def md5(sha, path):
    t = C.show(sha, path)
    return None if t is None else hashlib.md5(t.encode("utf-8")).hexdigest()


def resolve(ev):
    """branch paths named in an evidence_file string"""
    ev = ev.replace("origin/paper:", "")
    out, lastdir = [], None
    for tok in re.findall(r"[\w./{},*<>-]+", ev):
        tok = tok.strip(".,;")
        if tok.startswith("src/probes/"):
            d, _, f = tok.rpartition("/")
            lastdir = d
            paths = [tok]
        elif re.match(r"(matrix_[\w*]+|summary(_balacc)?|zero_shot\w*|liars_bench_comparison|reads(_summary)?)\.csv$", tok) and lastdir:
            paths = [f"{lastdir}/{tok}"]
        else:
            continue
        for p in paths:
            m = re.search(r"\{([\d,]+)\}", p)
            exp = [p.replace(m.group(0), x) for x in m.group(1).split(",")] if m else [p]
            for q in exp:
                if "*" in q:
                    d, _, pat = q.rpartition("/")
                    rx = re.compile("^" + re.escape(pat).replace(r"\*", ".*") + "$")
                    files = {f for s in SHAS.values() for f in ls(s, d + "/") if rx.match(f.rpartition("/")[2]) and f.rpartition("/")[0] == d}
                    out += sorted(files) if files else [q]
                elif q.endswith("/"):
                    out.append(q)
                else:
                    out.append(q)
    return sorted(set(out))


def fmts(x):
    s = set()
    for p in (3, 4):
        s.add(str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-p), ROUND_HALF_UP)))
        s.add(f"{float(x):.{p}f}")
    return s


def keyed(rows):
    return {tuple(v for v in r.values() if C.fl(v) is None): r for r in rows}


def value_set(rows, keys, cols, is_raw):
    """numeric cells in the COMMON rows/columns (so rows/columns added at 2ad758fb cannot collide),
    plus for the raw-probe matrices the per-test column means Table 7 quotes; 3 and 4 dp, both
    rounding conventions"""
    s = set()
    k = keyed(rows)
    for key in keys:
        for h in cols:
            v = C.fl(k[key].get(h))
            if v is not None:
                s |= fmts(v)
    if is_raw:
        m = np.array([[C.fl(v) for kk, v in r.items() if kk != list(r.keys())[0]] for r in rows], float)
        np.fill_diagonal(m, np.nan)
        for x in np.nanmean(m, axis=0):
            s |= fmts(x)
    return s


def changed_strings(path):
    """strings present only at PIN_OLD / only at PIN_P2AD, over the cells both versions share"""
    a, b = C.tab(C.PIN_OLD, path) or [], C.tab(C.PIN_P2AD, path) or []
    if not a or not b:
        return set(), set()
    keys = set(keyed(a)) & set(keyed(b))
    cols = [h for h in a[0] if h in b[0]]
    raw = bool(re.search(r"matrix_(balacc_)?baseline\.csv$", path))
    so, sn = value_set(a, keys, cols, raw), value_set(b, keys, cols, raw)
    return so - sn, sn - so


NEWROW = re.compile(r"13v12|uni_axis|uni_pc|unified axis|measured here|auroc_uni", re.I)

rows = [r for r in csv.DictReader(open(C.ROOT / "audit" / "claims.csv", encoding="utf-8"))
        if r["id"].startswith("D-") and "ea1ce52b" in r["version_used"]]
cache, out = {}, []
for r in rows:
    ev = r["evidence_file"]
    stored = (r["level1_value"] + " || " + r["level2_value"]).strip(" |")
    paths = resolve(ev)
    special = []
    if "ls-tree" in ev:
        pre = C.BASE + "/paper/" if "paper/" in ev else ""
        special = [("ls-tree " + (pre or "<root>"), {n: hashlib.md5("\n".join(ls(s, pre or ".")).encode()).hexdigest() for n, s in SHAS.items()})]
    h = {}
    for p in paths:
        if p not in cache:
            cache[p] = {n: md5(s, p) for n, s in SHAS.items()}
        h[p] = cache[p]
    for name, hh in special:
        h[name] = hh
    missing = [p for p, v in h.items() if any(x is None for x in v.values())]
    diff = [p for p, v in h.items() if len(set(v.values())) > 1]
    diff_old_2ad = [p for p in diff if h[p]["ea1ce52b"] != h[p]["2ad758fb"]]
    diff_2ad_pin = [p for p in diff if h[p]["2ad758fb"] != h[p]["16def2de"]]
    old_hits, new_hits = set(), set()
    for p in diff_old_2ad:
        if p.endswith(".csv"):
            o, n = changed_strings(p)
            nums = set(re.findall(r"\d\.\d{3,4}", stored))
            old_hits |= {x for x in nums if x in o}
            new_hits |= {x for x in nums if x in n}
    newrow = bool(NEWROW.search(stored))
    if not paths and not special:
        src, v_old = "AMBIGUOUS-identical", "no branch file in evidence (SHA-independent source)"
    elif not diff:
        src, v_old = "AMBIGUOUS-identical", f"identical: {len(h)} evidence file(s) byte-identical at all 3 SHAs"
    elif old_hits and not new_hits:
        src, v_old = "ea1ce52b", "old-only values quoted: " + ",".join(sorted(old_hits))
    elif new_hits and not old_hits or newrow:
        src, v_old = "2ad758fb", "new-only values quoted: " + ",".join(sorted(new_hits)) + (" +2ad-only rows" if newrow else "")
    elif old_hits and new_hits:
        src, v_old = "CONFLICT", f"old {sorted(old_hits)} new {sorted(new_hits)}"
    else:
        src, v_old = "AMBIGUOUS-identical", (f"{len(diff)} evidence file(s) differ but none of the quoted numbers is a changed cell"
                                             f" (changed: {', '.join(p.split('/results/')[-1] for p in diff)[:200]})")
    out.append(dict(id=r["id"], stored_value=stored[:300], src=src, v=v_old, verdict=r["verdict"],
                    missing=missing, diff=diff, diff_2ad_pin=diff_2ad_pin, old_hits=old_hits, new_hits=new_hits))

# ---- targeted re-derivation: every row that quotes a raw-probe ('baseline') value or a paper/
# generated/ count, i.e. the only quantities that changed. The string heuristic above is collision-
# prone (a quoted 0.799 can equal some other changed cell), so these rows are settled by computing
# the exact quantity at each SHA, formatted as the row quotes it.
import pandas as pd
ORD = ["convince_me__plain", "harm_refusal_choice__plain", "harm_refusal_knowledge_report__plain",
       "insider_trading_cadenza__plain", "instruct_dishonesty__plain", "repe_honesty__plain"]
_m = {}


def M(sha, L, arm):
    k = (sha, L, arm)
    if k not in _m:
        _m[k] = pd.read_csv(io.StringIO(C.show(sha, f"{C.RES}/persona_ood_ext_L{L}/{arm}")), index_col=0)
    return _m[k]


def hu(x, p):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-p), ROUND_HALF_UP))


def pt(sha, L, arm, ds):
    v = M(sha, L, f"matrix_{arm}.csv").to_numpy(float).copy(); np.fill_diagonal(v, np.nan)
    return np.nanmean(v, axis=0)[ORD.index(ds)]


def sm(sha, L, feat, col):
    return M(sha, L, "summary.csv").loc[feat, col]


def gen(sha):
    t = C.show(sha, C.BASE + "/paper/generated/numbers.tex") or ""
    names = [x.rsplit("/", 1)[1] for x in ls(sha, C.BASE + "/paper/generated/")]
    return (t.count("\setV{"), len(t.splitlines()), sum(x.startswith("tab_") for x in names),
            sum(x.startswith("fig_") and x.endswith(".pdf") for x in names),
            int(subprocess.run(["git", "cat-file", "-s", f"{sha}:{C.BASE}/paper/make_numbers.py"], cwd=C.ROOT,
                               capture_output=True, text=True).stdout.strip() or 0))


T7 = {"D-189": "convince_me__plain", "D-190": "harm_refusal_choice__plain", "D-191": "harm_refusal_knowledge_report__plain",
      "D-192": "insider_trading_cadenza__plain", "D-193": "instruct_dishonesty__plain", "D-194": "repe_honesty__plain"}
TGT = {}
for cid, ds in T7.items():
    # at the precision the row quotes (3 dp; D-189 also quotes CG L40 at 4 dp)
    TGT[cid] = lambda s, ds=ds, cid=cid: " | ".join(f"L{L} raw {hu(pt(s, L, 'baseline', ds), 3)}" + (f" ({hu(pt(s, L, 'baseline', ds), 4)})" if cid == "D-189" and L == 40 else "") + f" ext {hu(pt(s, L, 'ext_axis1', ds), 3)}" for L in (20, 40, 44))
TGT["D-001"] = lambda s: " | ".join(f"L{L} diag {hu(sm(s, L, 'baseline', 'diagonal_mean'), 4)} off {hu(sm(s, L, 'baseline', 'off_diagonal_mean'), 4)} drop {hu(sm(s, L, 'baseline', 'diagonal_mean') - sm(s, L, 'baseline', 'off_diagonal_mean'), 3)}" for L in (20, 40, 44))
TGT["D-109"] = lambda s: " | ".join(f"L{L} {hu(sm(s, L, 'baseline', 'off_diagonal_mean'), 3)}/{hu(sm(s, L, 'baseline', 'off_diagonal_excl_hpkr'), 3)}" for L in (20, 40, 44))
TGT["D-026"] = lambda s: f"L40 ext {hu(sm(s, 40, 'ext_axis1', 'off_diagonal_mean'), 4)} base {hu(sm(s, 40, 'baseline', 'off_diagonal_mean'), 4)}"
TGT["D-218"] = lambda s: "L20 RepE-row raw mean(5 others) " + hu(np.delete(M(s, 20, 'matrix_baseline.csv').to_numpy(float)[5], 5).mean(), 3)
TGT["D-171"] = lambda s: "setV macros %d; numbers.tex lines %d" % gen(s)[:2]
TGT["D-032"] = lambda s: "tab_* %d; fig_*.pdf %d; make_numbers.py %s B" % (gen(s)[2], gen(s)[3], f"{gen(s)[4]:,}")

MAN = {  # id -> (source_sha_of_stored, verdict_at_pin, changed, note)
    "D-001": ("NONE", None, "n",
              "Stored L40 '0.9833/0.7320' reproduces at ea1ce52b only (diag 0.98327 -> 0.9833; 2ad758fb 0.98325 -> 0.9832). "
              "Stored 'L20 0.9787/0.6478; L44 0.9826/0.7159' and 'gap largest at L20 (0.331)' reproduce at NO SHA: summary.csv "
              "baseline is L20 0.9564/0.6478 (gap 0.309), L44 0.9822/0.7156 at ea1ce52b and 45b16d90 alike. Evidence text "
              "wrong; the claim ('probes often fail to transfer': drop 0.309/0.251/0.267) still holds, so the verdict stands."),
    "D-189": ("ea1ce52b", "PARTLY MATCHES HIS DATA", "y",
              "Stored '0.7875->0.788' is the ea1ce52b value. At 2ad758fb/16def2de CG raw L40 = 0.7873 -> 0.787 "
              "(matrix_baseline.csv L40 re-run, drift 0.0002). The draft's 0.788 is now 0.001 off at 3 dp; the paper at "
              "16def2de prints 0.787 (numbers.tex pt-base-cg-L40; tab_pertest.tex). Other five numbers unchanged."),
    "D-171": ("2ad758fb", None, "n",
              "Stored '1,162 \setV macros' matches NO SHA as a macro count: macros = 1,100 (ea1ce52b) / 1,161 (2ad758fb) / "
              "1,176 (16def2de); 1,162 = numbers.tex LINE count at 2ad758fb (1,161 macros + header). So the row was written "
              "from 2ad758fb, with an off-by-one (header counted). Verdict (numbers are generated) unaffected."),
    "D-032": ("CONFLICT", None, "n",
              "Mixed provenance: 'make_numbers.py (28,232 B)' is the ea1ce52b size (2ad758fb 36,337 B; 16def2de 39,532 B); "
              "'five fig_*.pdf' exists only from 2ad758fb (ea1ce52b has 4; fig_depth.pdf added); 'eight tab_*.tex' matches "
              "no SHA (11 at all three). Verdict (code + results released) unaffected."),
}
# ---- locator: how many of the row's quoted decimals (>=2 dp) are present, at each SHA, in the row's
# evidence CSVs (every cell, 2/3/4 dp, both roundings; per-test column means for matrix files) or in
# the pinned level1/level2 re-run output at that SHA. A quoted number found at NO SHA is a value the
# stored prose does not get from the data (a derived number, a PDF value, or an error).
def fmts2(x):
    s = fmts(x)
    s.add(str(Decimal(repr(float(x))).quantize(Decimal("0.01"), ROUND_HALF_UP)))
    return s


_vs = {}


def allvals(sha, path):
    k = (sha, path)
    if k not in _vs:
        s = set()
        rows = C.tab(sha, path) if path.endswith(".csv") else None
        for r in rows or []:
            for v in r.values():
                if C.fl(v) is not None:
                    s |= fmts2(C.fl(v))
        if rows and "/matrix_" in path:
            try:
                m = np.array([[C.fl(v) for kk, v in r.items() if kk != list(r.keys())[0]] for r in rows], float)
                np.fill_diagonal(m, np.nan)
                for x in list(np.nanmean(m, axis=0)) + [np.nanmean(m)]:
                    s |= fmts2(x)
            except Exception:
                pass
        _vs[k] = s
    return _vs[k]


LOUT = {n: set(re.findall(r"\d\.\d{2,4}", (P17 / f"level1_{n}.txt").read_text(encoding="utf-8") +
                          (P17 / f"level2_{n}.txt").read_text(encoding="utf-8"))) for n in SHAS}
LOC = {}
for o in out:
    r = next(x for x in rows if x["id"] == o["id"])
    nums = sorted(set(re.findall(r"(?<![\d.])\d\.\d{2,4}(?!\d)", r["level1_value"] + " " + r["level2_value"])))
    paths = [p for p in resolve(r["evidence_file"]) if p.endswith(".csv")]
    found = {}
    for n_, s in SHAS.items():
        pool = set(LOUT[n_])
        for p in paths:
            pool |= allvals(s, p)
        found[n_] = [x for x in nums if x in pool]
    LOC[o["id"]] = (nums, found)
print("locator: rows whose found-count differs between SHAs:",
      [(k, {n: len(v) for n, v in f.items()}) for k, (nums, f) in LOC.items() if len({len(v) for v in f.values()}) > 1])
nf = {k: [x for x in nums if all(x not in f[n] for n in f)] for k, (nums, f) in LOC.items()}
print("rows with >=1 quoted decimal found at NO SHA:", sum(1 for v in nf.values() if v), "of", len(nf))
for k, v in nf.items():
    if v:
        print("   ", k, v[:8])

TV = {cid: {n: f(s) for n, s in SHAS.items()} for cid, f in TGT.items()}
fields = ["id", "stored_value", "value_at_ea1ce52b", "value_at_2ad758fb", "value_at_16def2de",
          "source_sha_of_stored", "verdict_stored", "verdict_at_pin", "changed", "evidence"]
recs = []
for o in out:
    cid = o["id"]
    vpin, ch, src = o["verdict"], "n", o["src"]
    ev = "; ".join(p.split("deception-detection/")[-1] for p in o["diff"]) or "all evidence identical"
    if cid in TV:
        va, v2, vp = TV[cid]["ea1ce52b"], TV[cid]["2ad758fb"], TV[cid]["16def2de"]
        if cid in MAN:
            src, vv, ch, note = MAN[cid]
            vpin = vv or vpin
        else:
            src = "AMBIGUOUS-identical" if va == v2 == vp else "REVIEW"
            note = ("targeted re-derivation: quoted quantities identical at the quoted precision at all 3 SHAs"
                    if src == "AMBIGUOUS-identical" else "targeted re-derivation differs")
        evid = note + " | files: " + ev[:250]
    elif o["src"] == "AMBIGUOUS-identical" and not o["diff"]:
        va = v2 = vp = "same as stored (evidence byte-identical)"
        evid = o["v"]
    else:
        va = v2 = vp = "same as stored (quoted cells unchanged)" if src == "AMBIGUOUS-identical" else o["v"]
        if src != "AMBIGUOUS-identical":
            vpin, ch = "REVIEW (" + src + ")", "y"
        evid = o["v"][:300] + " | differing files: " + ev[:250]
    nums, f = LOC[cid]
    loc = {n: f"{len(f[n])}/{len(nums)} quoted decimals located" for n in SHAS}
    if cid not in TV:
        va, v2, vp = (x + "; " + loc[n] for x, n in zip((va, v2, vp), SHAS))
    if nf[cid]:
        evid = evid + " | not located at any SHA: " + ",".join(nf[cid][:8])
    recs.append({"id": cid, "stored_value": o["stored_value"], "value_at_ea1ce52b": va,
                 "value_at_2ad758fb": v2, "value_at_16def2de": vp, "source_sha_of_stored": src,
                 "verdict_stored": o["verdict"][:120], "verdict_at_pin": vpin[:160], "changed": ch,
                 "evidence": evid[:600]})
with open(P17 / "d_rows_repinned.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(recs)
for cid in TV:
    print(cid, "|", "SAME" if len(set(TV[cid].values())) == 1 else "DIFF", "|", TV[cid]["ea1ce52b"][:110], "=>", TV[cid]["16def2de"][:110] if len(set(TV[cid].values())) > 1 else "")

from collections import Counter
print(f"{len(recs)} D- rows stamped ea1ce52b")
print("source_sha_of_stored:", dict(Counter(r["source_sha_of_stored"] for r in recs)))
print("rows with >=1 evidence file differing across SHAs:", sum(1 for o in out if o["diff"]))
print("rows with a file differing 2ad758fb->16def2de:", [o["id"] for o in out if o["diff_2ad_pin"]])
print("rows with unresolvable (missing) evidence paths:", [(o["id"], o["missing"][:2]) for o in out if o["missing"]][:8])
print("verdict changed:", [r["id"] for r in recs if r["changed"] == "y"])
print("heuristic string hits (before targeted override):", [(o["id"], o["src"]) for o in out if o["old_hits"] or o["new_hits"]])
# whole-output cross-check of the pinned re-runs
for lv in (1, 2):
    t = {n: (P17 / f"level{lv}_{n}.txt").read_text(encoding="utf-8").splitlines()[1:] for n in SHAS}
    body = {n: [ln for ln in v if not ln.startswith("SOURCE:")] for n, v in t.items()}
    add = [ln for ln in body["2ad758fb"] if ln not in body["ea1ce52b"]]
    rem = [ln for ln in body["ea1ce52b"] if ln not in body["2ad758fb"]]
    print(f"level{lv} pinned output: ea1ce52b vs 2ad758fb +{len(add)} -{len(rem)} lines; 2ad758fb==16def2de: {body['2ad758fb'] == body['16def2de']}")
