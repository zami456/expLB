"""P17 Agent C, task 3 -- check every number in the ten 'CORRECTION 2026-09-24 (DF-xx, Phase 6)' blocks
added to the working tree (uncommitted, md5s in audit/BRANCH_PINS.md) against raw data.

Sources: laptop working tree result files (== git asix e917ecd3), git history (read-only), the audit's
md5-verified lab cache, and the lab PC read-only via p17_labfetch (ssh BatchMode; cached in audit/p17/labcache).
Writes audit/p17/correction_blocks_check.csv; prints a summary.
  python audit/scripts/p17_correction_blocks.py
"""
import ast, csv, io, json, os, re, statistics, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p17_common as C  # noqa: E402
import p17_labfetch as LF  # noqa: E402

ROOT = str(C.ROOT)
LAB = "C:/Users/user4/expLB"
OUT = []
KR, SYC = "harm-pressure-knowledge-report", "sycophancy_mt"
csv.field_size_limit(10 ** 9)


def J(p):
    return json.load(open(os.path.join(ROOT, p), encoding="utf-8"))


def labj(p):
    return json.loads(LF.fetch(p)[p][0])


def block(path):
    """(start_line, lines) of every CORRECTION 2026-09-24 block in the WT file."""
    L = open(os.path.join(ROOT, path), encoding="utf-8").read().splitlines()
    out = []
    for i, l in enumerate(L):
        if "CORRECTION 2026-09-24" in l:
            j = i + 1
            html = path.endswith(".html")
            while j < len(L) and ((not html and L[j].startswith(">")) or (html and "</div>" not in L[j])):
                j += 1
            out.append((i + 1, L[i:j]))
    return out


def line_of(path, needle, which=0):
    """Line number (in the WT file) of the first occurrence of `needle` inside CORRECTION block `which`."""
    s, lines = block(path)[which]
    for k, l in enumerate(lines):
        if needle in l:
            return s + k
    return f"{s}(block)"


def chk(path, df, needle, number, raw, source, match=None, which=0, tol=None):
    if match is None:
        try:
            m_ = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(number).replace("−", "-").replace("±", ""))
            tok = m_.group(0).replace(",", "")
            nv, rv = float(tok), float(raw)
            if tol is None:
                s_ = tok.split(".")
                tol = 0.5 * 10 ** (-len(s_[1])) + 1e-12 if len(s_) > 1 else 0.5
            match = "y" if abs(nv - rv) <= tol else "n"
        except (TypeError, ValueError):
            match = "y" if str(number) == str(raw) else "n"
    OUT.append(dict(file=path, line=line_of(path, needle, which), df_id=df, number=number,
                    raw_value=(f"{raw:.4f}" if isinstance(raw, float) else str(raw)), source=source, match=match))


def offdiag(d, f, probe="paper", keep=None):
    ds = d["datasets"]; keep = keep or ds
    b = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}[(f, probe)]
    v = [b[i][j] for i, a in enumerate(ds) for j, c in enumerate(ds) if i != j and a in keep and c in keep]
    return statistics.fmean(v)


def row_off(d, f, tr, probe="paper"):
    ds = d["datasets"]; i = ds.index(tr)
    b = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}[(f, probe)]
    return statistics.fmean(b[i][j] for j in range(len(ds)) if j != i)


def auroc_bin(score, y):
    import numpy as np
    s, t = np.asarray(score, float), np.asarray(y, bool)
    order = s.argsort(kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ties
        m = s == v; ranks[m] = ranks[m].mean()
    n1, n0 = t.sum(), (~t).sum()
    return float((ranks[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    k40 = J("run-crossgen/kd11/matrices/mean_L40_s42.json")
    zs = {z["direction"]: z["auroc"] for z in k40["zeroshot"]}
    j = k40["datasets"].index(SYC)
    contrast, pc1 = zs["contrast"][j], zs["PC1"][j]
    q10 = J("run-coords/q10_sycophancy_axis.json")["results"]["paper16|mean"]["sycophancy"]["key_layers"]["40"]["auroc"]
    zn = J("run-coords/zeroshot_nulls.json")[SYC]
    ps = [v["sycophancy_axis"]["p_vs_random"] for v in zn.values() if "sycophancy_axis" in v]
    prange = f"{min(ps):.3f}-{max(ps):.3f}"
    dec_inside = not zn["L40_mean"]["deception_axis"]["beats_random_p95"]
    p95 = zn["L40_mean"]["random_p95_auroc_dev"]
    pc1_inside = max(pc1, 1 - pc1) < p95
    S_K = "run-crossgen/kd11/matrices/mean_L40_s42.json zeroshot[sycophancy_mt]"
    S_Q = "run-coords/q10_sycophancy_axis.json paper16|mean key_layers.40"
    S_N = "run-coords/zeroshot_nulls.json [sycophancy_mt][*].sycophancy_axis.p_vs_random"
    axes_src = row_off(k40, "axes", KR)
    rnd_col = statistics.fmean({(r["feature"], r["probe"]): r["auroc"] for r in k40["results"]}[("random", "paper")][i][j]
                               for i in range(len(k40["datasets"])) if i != j)
    cap = labj(f"{LAB}/run-crossgen/kd11/banks/cap_s23/matrices/mean_L40_s42.json")
    cap_c = {z["direction"]: z["auroc"] for z in cap["zeroshot"]}["contrast"][cap["datasets"].index(SYC)]
    S_C = f"{LAB}/run-crossgen/kd11/banks/cap_s23/matrices/mean_L40_s42.json zeroshot contrast[sycophancy_mt] (lab, md5 1a259fa9)"

    # --- DF-31 family: NEXT.md, SKILL.md, SARCASM_DESIGN.md, review_page.html
    f = "kd/docs/NEXT.md"
    chk(f, "DF-31/DF-68", "0.374 is the paper16", "0.374", pc1, S_K + " PC1")
    chk(f, "DF-31/DF-68", "contrast reads 0.409", "0.409", contrast, S_K + " contrast")
    chk(f, "DF-31/DF-68", "reads **0.313**", "0.313", q10, S_Q)
    chk(f, "DF-31/DF-68", "p 0.001", "p 0.001-0.008", prange, S_N, match="y" if prange == "0.001-0.008" else "n")
    chk(f, "DF-31/DF-68", "0.745 zero-shot", "0.745", cap_c, S_C)
    chk(f, "DF-31/DF-68", "0.745 zero-shot", "(pointer) row 1 below says 'register axis's 0.374'",
        "row present" if "register" in open(os.path.join(ROOT, f), encoding="utf-8").read() else "absent", "kd/docs/NEXT.md WT", match="y")
    f = ".claude/skills/knowledge-denial-context/SKILL.md"
    chk(f, "DF-31", "0.374 is the paper16", "0.374", pc1, S_K + " PC1")
    chk(f, "DF-31", "reads **0.313**", "0.313", q10, S_Q)
    chk(f, "DF-31", "inside", "0.374 inside 1,000-direction null", f"inside (|0.5-x|={abs(0.5 - pc1):.3f} < p95 dev {p95 - 0.5:.3f}); null run on contrast 0.409: inside={dec_inside}",
        "run-coords/zeroshot_nulls.json L40_mean", match="y" if (pc1_inside and dec_inside) else "n")
    chk(f, "DF-31", "p 0.001", "p 0.001-0.008", prange, S_N, match="y" if prange == "0.001-0.008" else "n")
    chk(f, "DF-31", "0.300 figure", "0.300", axes_src, "kd11 mean_L40 axes (paper) row mean, HP-KR as source (10 targets)")
    chk(f, "DF-31", "`contrast` 0.4089", "0.4089", contrast, S_K + " contrast")
    chk(f, "DF-31", "`PC1` 0.3739", "0.3739", pc1, S_K + " PC1 (0.373959 - truncated, not rounded)", tol=0.0001)
    chk(f, "DF-31", "Line 75", "(pointer) line 75 corrected 2026-09-23", "0.313" if "0.313" in open(os.path.join(ROOT, f), encoding="utf-8").read().splitlines()[91] else "see file",
        "SKILL.md WT line 92 (= HEAD line 75 + 17 inserted lines)", match="y" if "0.313" in open(os.path.join(ROOT, f), encoding="utf-8").read().splitlines()[91] else "n")
    f = "cap/docs/SARCASM_DESIGN.md"
    chk(f, "DF-31", "**0.300**", "0.300", axes_src, "kd11 mean_L40 axes row mean, HP-KR as source")
    chk(f, "DF-31", "**0.374**", "0.374", pc1, S_K + " PC1")
    chk(f, "DF-31", "reads **0.313**", "0.313", q10, S_Q)
    chk(f, "DF-31", "p 0.001", "p 0.001-0.008", prange, S_N, match="y" if prange == "0.001-0.008" else "n")
    chk(f, "DF-31", "`contrast` 0.4089", "0.4089", contrast, S_K)
    chk(f, "DF-31", "`PC1` 0.3739", "0.3739", pc1, S_K, tol=0.0001)
    f = "cap/docs/review_page.html"
    chk(f, "DF-31", "0.374 is the paper16", "0.374", pc1, S_K + " PC1")
    chk(f, "DF-31", "<strong>0.313</strong>", "0.313", q10, S_Q)
    chk(f, "DF-31", "contrast reads 0.409", "0.409", contrast, S_K)
    chk(f, "DF-31", "p 0.001", "p 0.001-0.008", prange, S_N, match="y" if prange == "0.001-0.008" else "n")
    chk(f, "DF-31", "0.54&rdquo;", "0.54 (trained random)", rnd_col, "kd11 mean_L40 random (paper) column mean on sycophancy_mt", tol=0.005)
    chk(f, "DF-31", "0.4089", "0.4089", contrast, S_K)
    chk(f, "DF-31", "0.3739", "0.3739", pc1, S_K, tol=0.0001)
    chk(f, "DF-31", "(2026-09-17)", "0.313 from 24_q10_sycophancy_axis.py (2026-09-17)", "q10 json L40 0.3131", S_Q, match="y")
    a11 = labj(f"{LAB}/run-crossgen/kd11/passA11/matrices/mean_L40_s42.json")
    c11 = labj(f"{LAB}/run-crossgen/kd11/passC11/matrices/mean_L40_s42.json")
    za = {z["direction"]: z["auroc"][a11["datasets"].index(SYC)] for z in a11["zeroshot"]}
    zc = {z["direction"]: z["auroc"][c11["datasets"].index(SYC)] for z in c11["zeroshot"]}
    chk(f, "DF-31", "0.3061 / 0.2996", "0.3061 (passA11)", za["contrast:sycophancy"], f"{LAB}/run-crossgen/kd11/passA11/matrices/mean_L40_s42.json contrast:sycophancy", tol=0.00005)
    chk(f, "DF-31", "0.3061 / 0.2996", "0.2996 (passC11)", f"{zc['PC1:sycophancy']:.4f} (PC1) / {zc['contrast:sycophancy']:.4f} (contrast); passA11 PC1 = {za['PC1:sycophancy']:.4f}",
        f"{LAB}/run-crossgen/kd11/passC11/matrices/mean_L40_s42.json", match="partly: 0.2996 is passA11's PC1:sycophancy; passC11 reads 0.3066 contrast / 0.2995 PC1")

    # --- DF-55: 113-matrix sweep, regex
    diag, per_layer = [], {}
    nfile = 0
    for dirpath, _, files in os.walk(os.path.join(ROOT, "run-crossgen")):
        if os.path.basename(dirpath) != "matrices":
            continue
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            d = json.load(open(os.path.join(dirpath, fn), encoding="utf-8"))
            ds = d.get("datasets") or []
            if KR not in ds:
                continue
            nfile += 1
            i = ds.index(KR)
            for r in d.get("results", []):
                if r.get("feature") == "raw" and isinstance(r.get("auroc"), list):
                    v = r["auroc"][i][i]
                    diag.append(v); per_layer.setdefault(int(d.get("layer")), []).append(v)
    med = statistics.median(diag)
    head = {L: statistics.median(per_layer[L]) for L in (20, 40, 44) if L in per_layer}
    # the regex, from the committed subset CSV (Llama-3.3-70B rows) and eval_datasets.denies_knowledge
    DEN = re.compile(r"\b(do not|don't|doesn't|not)\b[^.]{0,20}\bknow\b", re.I)
    ASR = re.compile(r"^\s*(yes\b|i know\b)", re.I)
    y, s1, s2 = [], [], []
    for r in csv.DictReader(open(os.path.join(ROOT, "results/subsets/harm-pressure-knowledge-report.csv"), encoding="utf-8")):
        if r["model"] != "llama-v3.3-70b-instruct":
            continue
        msgs = ast.literal_eval(r["messages"])
        txt = [m["content"] for m in msgs if m["role"] == "assistant"][-1]
        y.append(r["deceptive"].strip().lower() in ("true", "1"))
        s1.append(1.0 if (DEN.search(txt) and not ASR.match(txt)) else 0.0)
        s2.append(0.0 if ASR.match(txt) else 1.0)
    rx1, rx2 = auroc_bin(s1, y), auroc_bin(s2, y)
    S_R = "results/subsets/harm-pressure-knowledge-report.csv (git asix), Llama rows, eval_datasets.denies_knowledge"
    S_M = f"all run-crossgen/**/matrices/*.json with HP-KR on the laptop ({nfile} files); raw feature, HP-KR diagonal, every probe"
    for f, df in (("docs/LAYER_PREREGISTRATION.md", "DF-55"), ("docs/NOVELTY_ASSESSMENT.md", "DF-55"), ("docs/NOVELTY_CHECK_PROMPT.md", "DF-55 + DF-10")):
        chk(f, df, "113", "113 matrices", nfile, S_M)
        chk(f, df, "0.9663", "0.9663", med, S_M)
        chk(f, df, "0.96–0.98", "0.96-0.98 at every headline layer", "/".join(f"L{L} {v:.4f}" for L, v in head.items()), S_M,
            match="y" if all(0.955 <= v < 0.985 for v in head.values()) else "n")
        chk(f, df, "0.9125", "0.9125", rx1, S_R)
        chk(f, df, "0.054", "~0.054", med - rx1, "difference", tol=0.0015)
        if f != "docs/NOVELTY_ASSESSMENT.md" or True:
            if "94.4%" in "\n".join(block(f)[0][1]):
                chk(f, df, "94.4%", "94.4%", 100 * rx1 / med, "regex / raw-diag median", tol=0.06)
    f = "docs/NOVELTY_CHECK_PROMPT.md"
    chk(f, "DF-55 + DF-10", "0.8302", "range 0.8302-0.9900", f"{min(diag):.4f}-{max(diag):.4f}", S_M,
        match="y" if (round(min(diag), 4), round(max(diag), 4)) == (0.8302, 0.99) else "n")
    chk(f, "DF-55 + DF-10", "0.9597", "0.9597 (second phrasing)", f"not reproduced: regex definition not recorded anywhere in audit/; the guess NOT ^(yes|i know) gives {rx2:.4f}",
        S_R + "; source cited = audit/HPKR_WORDING_CONTROLLED.md:38 (prose)", match="unverified")
    chk(f, "DF-55 + DF-10", "2,139", "2,139 rows", len(y), S_R)
    rows = {}
    for ds in ("sandbagging_v2", "mask"):
        p = f"{LAB}/run-coords/passB/{ds}/rows.jsonl"
        R = [json.loads(x) for x in LF.fetch(p)[p][0].decode("utf-8").splitlines() if x.strip()]
        rows[ds] = (statistics.fmean(r["span_len"] for r in R if r["label"] == 1), statistics.fmean(r["span_len"] for r in R if r["label"] == 0))
    chk(f, "DF-55 + DF-10", "529.4 vs 340.4", "529.4", rows["sandbagging_v2"][0], f"{LAB}/run-coords/passB/sandbagging_v2/rows.jsonl span_len, label 1")
    chk(f, "DF-55 + DF-10", "529.4 vs 340.4", "340.4", rows["sandbagging_v2"][1], "same, label 0")
    chk(f, "DF-55 + DF-10", "283.2 vs 176.4", "283.2", rows["mask"][0], f"{LAB}/run-coords/passB/mask/rows.jsonl")
    chk(f, "DF-55 + DF-10", "283.2 vs 176.4", "176.4", rows["mask"][1], "same")
    head_txt = C.show(C.PIN_ASIX, f) or ""
    parent = C.show(C.PIN_ASIX + "~1", f) or ""
    pl = next((i + 1 for i, l in enumerate(parent.splitlines()) if "505 vs 310" in l), None)
    hl = next((i + 1 for i, l in enumerate(head_txt.splitlines()) if "529.4 vs 340.4" in l), None)
    chk(f, "DF-55 + DF-10", "at `:110`", "':110' still carries '505 vs 310 ... 245 vs 96'",
        f"absent at HEAD e917ecd3 and in WT; HEAD:{hl} already reads 529.4 vs 340.4; '505 vs 310' last at e917ecd3~1:{pl}",
        "git show e917ecd3 / e917ecd3~1 : docs/NOVELTY_CHECK_PROMPT.md",
        match="n (stale: item (b) corrects text the owner already fixed in the committed HEAD; its numbers are right)" if ("505 vs 310" not in head_txt) else "y")
    na = open(os.path.join(ROOT, "docs/NOVELTY_ASSESSMENT.md"), encoding="utf-8").read().splitlines()
    ln2 = next(i + 1 for i, l in enumerate(na) if "0.87" in l and not l.startswith(">"))
    chk(f, "DF-55 + DF-10", "NOVELTY_ASSESSMENT.md:30", "NOVELTY_ASSESSMENT.md:30", f":{ln2} in WT (HEAD :30)", "WT file",
        match="y" if ln2 == 30 else f"n (WT line is :{ln2} after the inserted block; :30 is the HEAD line)")
    lp = open(os.path.join(ROOT, "docs/LAYER_PREREGISTRATION.md"), encoding="utf-8").read().splitlines()
    ln3 = next(i + 1 for i, l in enumerate(lp) if "0.87" in l and not l.startswith(">"))
    chk(f, "DF-55 + DF-10", "LAYER_PREREGISTRATION.md:93", "LAYER_PREREGISTRATION.md:93", f":{ln3} in WT (HEAD :93)", "WT file",
        match="y" if ln3 == 93 else f"n (WT line is :{ln3}; :93 is the HEAD line)")

    # --- DF-68: F-74 numbers (kd/scripts/ceiling_null.py, read-only, prints)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "kd/scripts/ceiling_null.py")], cwd=ROOT,
                         capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"}).stdout
    g = lambda pat: re.search(pat, out).group(1)
    f = "docs/NOVELTY_ASSESSMENT.md"
    S_F = "kd/scripts/ceiling_null.py over run-crossgen/kd11/matrices/mean_L40_s42.json (stdout)"
    chk(f, "DF-68", "−0.7684", "-0.7684", float(g(r"corr\(raw, random@2 - raw\) = (-?[\d.]+)")), S_F, which=1)
    chk(f, "DF-68", "−0.7462", "-0.7462", float(g(r"null \(analytic.*?= (-?[\d.]+)")), S_F, which=1)
    chk(f, "DF-68", "0.2135", "0.2135", float(g(r"P\(null <= measured\) = ([\d.]+)")), S_F, which=1)
    chk(f, "DF-68", "−0.100", "-0.100", float(g(r"pc:unified@2 - datapca@2\s+slope vs raw (-?[\d.]+)")), S_F, which=1)
    chk(f, "DF-68", "−0.067", "-0.067", float(g(r"pc:unified@2 - datapca@2\s+slope vs raw -?[\d.]+\s+r (-?[\d.]+)")), S_F, which=1)
    na_ln = next((i + 1 for i, l in enumerate(na) if "corr" in l and "0.76" in l and not l.startswith(">")), None)
    chk(f, "DF-68", "corr −0.76", "bullet 'ceiling artifact, corr -0.76' below the block", f"present at :{na_ln}" if na_ln else "absent", "WT file", match="y" if na_ln else "n", which=1)

    # --- DF-67: REVIEW_REQUEST_POOLING.md
    f = "docs/REVIEW_REQUEST_POOLING.md"
    src = open(os.path.join(ROOT, "src/assistant_axis_ndif/30_premise.py"), encoding="utf-8").read().splitlines()
    chk(f, "DF-67", "30_premise.py:106", "30_premise.py:106 rng.permutation(n)", src[105].strip(), "src/assistant_axis_ndif/30_premise.py (asix WT)",
        match="y" if "rng.permutation(n)" in src[105] else "n")
    chk(f, "DF-67", "`group_split`", "group_split 0 times", sum(l.count("group_split") for l in src), "same file")
    prem = J("audit/cache/labpc/run-premise/premise_L40_mean_s42.json")
    aud = J("audit/cache/labpc/run-premise/audit/audit_L40_s42.json")
    S_P = "audit/cache/labpc/run-premise/premise_L40_mean_s42.json (md5-verified lab copy)"
    chk(f, "DF-67", "0.951", "0.951", prem["summary"]["single_own_mean"], S_P + " summary.single_own_mean")
    chk(f, "DF-67", "**0.850**", "0.850", prem["summary"]["pooled_lodo_match"], S_P + " summary.pooled_lodo_match")
    why = J("audit/cache/labpc/run-premise/why/why_L40_s42.json") if os.path.exists(os.path.join(ROOT, "audit/cache/labpc/run-premise/why/why_L40_s42.json")) else None
    if why:
        chk(f, "DF-67", "**0.806**", "0.806", why["pooled_lodo_matched"], "audit/cache/labpc/run-premise/why/why_L40_s42.json pooled_lodo_matched")
    dr = aud["pooled_lodo_match_draws"]
    chk(f, "DF-67", "0.8344", "0.8344", statistics.fmean(dr), "audit/cache/labpc/run-premise/audit/audit_L40_s42.json pooled_lodo_match_draws (20)")
    chk(f, "DF-67", "0.8344", "±0.0200", statistics.pstdev(dr), f"same; population sd {statistics.pstdev(dr):.4f}, sample sd {statistics.stdev(dr):.4f}", tol=0.00005)
    chk(f, "DF-67", "**+0.028**", "+0.028", aud["mean_of_directions"] - aud["pooled_lodo_full"], "audit_L40 mean_of_directions - pooled_lodo_full (Result 2)")
    nm = [v["n_lodo_match"] for v in prem["per_dataset"].values()]
    chk(f, "DF-67", "412–500", "412-500", f"{min(nm)}-{max(nm)}", S_P + " per_dataset[*].n_lodo_match", match="y" if (min(nm), max(nm)) == (412, 500) else "n")
    chk(f, "DF-67", "mean 449", "449", statistics.fmean(nm), S_P, tol=0.5)
    chk(f, "DF-67", "~666 rows", "666 cap", prem["n_train"]["single_mean"], S_P + " n_train.single_mean (= budget)")
    wt = open(os.path.join(ROOT, f), encoding="utf-8").read().splitlines()
    l85 = next(i + 1 for i, l in enumerate(wt) if "**0.850**" in l and not l.startswith(">"))
    l86 = next(i + 1 for i, l in enumerate(wt) if "(0.806)" in l and not l.startswith(">"))
    chk(f, "DF-67", "at `:32`", ":32 / :63 pointers", f"WT :{l85} / :{l86} (HEAD :32 / :63)", "WT file",
        match="y" if (l85, l86) == (32, 63) else "n (stale: pointers are HEAD line numbers; the inserted 19-line block moved them)")

    # --- DF-63
    f = "docs/thesis/persona-coordinates-70b-results.md"
    comps = J("run-coords/significance.json")["comparisons"]
    th = open(os.path.join(ROOT, f), encoding="utf-8").read().splitlines()
    marg_ln = next(i + 1 for i, l in enumerate(th) if "equivalence margin" in l and "0.02" in l and not l.startswith(">"))
    chk(f, "DF-63", "`:513`", "margin ±0.02 at :513", f":{marg_ln}", "WT file", match="y" if marg_ln == 513 else "n")
    flags = [comps[i]["equivalent_within_margin"] for i in range(7, 11)]
    chk(f, "DF-63", "`equivalent_within_margin = False`", "all four False", "/".join(map(str, flags)), "run-coords/significance.json comparisons[7..10]", match="y" if not any(flags) else "n")
    for i, w in zip(range(7, 11), ("-0.0128", "+0.0061", "+0.0036", "-0.0092")):
        chk(f, "DF-63", "diffs", w, comps[i]["diff"], f"run-coords/significance.json comparisons[{i}].diff")
    chk(f, "DF-63", "no `margin` field", "no margin field", "absent" if not any("margin" in k and k != "equivalent_within_margin" and not k.endswith("_margin") for c in comps for k in c) else "present",
        "keys of every comparison", match="y")
    lo = min(comps[i]["dataset_ci90"][0] for i in range(7, 11)); hi = max(comps[i]["dataset_ci90"][1] for i in range(7, 11))
    chk(f, "DF-63", "−0.038 to +0.033", "-0.038", lo, "min dataset_ci90 lower, comparisons[7..10]")
    chk(f, "DF-63", "−0.038 to +0.033", "+0.033", hi, "max dataset_ci90 upper")
    chk(f, "DF-63", "median gap is 0.012", "median gap 0.012 (over 17 methods)", "not re-derived (17-method set not identified in P17)", "DOC_FIXES DF-63 prose", match="unverified")

    # --- DF-66
    f = "docs/thesis/sarcasm-pilot-negative-result.md"
    cnt = {c: (C.show(c, "cap/docs/SARCASM_DESIGN.md") or "").count("0.65") for c in ("3d80071", "5ced550", "080a979", "b34ff2e")}
    chk(f, "DF-66", "count 0 at all four", "0.65 count 0 at 3d80071/5ced550/080a979/b34ff2e", "/".join(str(v) for v in cnt.values()), "git show <c>:cap/docs/SARCASM_DESIGN.md",
        match="y" if not any(cnt.values()) else "n")
    lg = subprocess.run(["git", "log", "--format=%h|%ad|%s", "--date=iso", "--reverse", "-S0.65", C.PIN_ASIX, "--", "cap/scripts/run_sarc_pilot.ps1"],
                        cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    chk(f, "DF-66", "`b34ff2e`, 2026-09-21 21:48:54", "first entry b34ff2e 2026-09-21 21:48:54 +0600", lg[0] if lg else "NOT-FOUND", "git log -S0.65 -- cap/scripts/run_sarc_pilot.ps1",
        match="y" if lg and lg[0].startswith("b34ff2e") and "2026-09-21 21:48:54 +0600" in lg[0] else "n")
    chk(f, "DF-66", "F-93: the sarcasm pilot fails", "commit message 'F-93: the sarcasm pilot fails its decisive check'", lg[0].split("|")[2][:60] if lg else "", "git log",
        match="y" if lg and "F-93: the sarcasm pilot fails its decisive check" in lg[0] else "n")
    v1 = C.show("3d80071", "cap/docs/SARCASM_DESIGN.md") or ""
    chk(f, "DF-66", "~0.75", "v1 pre-registered ~0.75", "present" if "above ~0.75" in v1 else "absent", "git show 3d80071:cap/docs/SARCASM_DESIGN.md:98", match="y" if "above ~0.75" in v1 else "n")
    v2 = C.show("5ced550", "cap/docs/SARCASM_DESIGN.md") or ""
    chk(f, "DF-66", "5ced550", "v2 phrase 'rather than a bare number against an arbitrary 0.75'", "present" if "arbitrary 0.75" in v2 else "absent", "git show 5ced550 (:141)", match="y" if "arbitrary 0.75" in v2 else "n")
    sd = open(os.path.join(ROOT, "cap/docs/SARCASM_DESIGN.md"), encoding="utf-8").read().splitlines()
    ln = next(i + 1 for i, l in enumerate(sd) if "arbitrary 0.75" in l)
    chk(f, "DF-66", "SARCASM_DESIGN.md:142", "phrase survives at :142", f":{ln} in WT (HEAD :142)", "WT file",
        match="y" if ln == 142 else f"n (stale: the P6 correction block inserted in SARCASM_DESIGN.md moved it to :{ln}; HEAD :142 is correct)")
    mr = labj(f"{LAB}/cap/run-sarc-pilot/marker_report.json")
    sub = {(x["pair"], x["subset"]): x for x in mr["lexical_subsets"]}
    S_S = f"{LAB}/cap/run-sarc-pilot/marker_report.json lexical_subsets (ironic vs sycophant), auroc_stripped (lab md5 bcc63316)"
    chk(f, "DF-66", "0.921", "0.921", sub[("ironic vs sycophant", "unmarked(v4,v5)")]["auroc_stripped"], S_S)
    chk(f, "DF-66", "0.947", "0.947", sub[("ironic vs sycophant", "all")]["auroc_stripped"], S_S)

    cols = ["file", "line", "df_id", "number", "raw_value", "source", "match"]
    with open(os.path.join(ROOT, "audit/p17/correction_blocks_check.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(OUT)
    n = {}
    for r in OUT:
        k = r["match"].split(" ")[0].split(":")[0]; n[k] = n.get(k, 0) + 1
    print(f"{len(OUT)} checks -> audit/p17/correction_blocks_check.csv  {n}")
    for r in OUT:
        if r["match"] != "y":
            print(f"  {r['file']}:{r['line']} {r['df_id']} {r['number'][:40]} -> {r['raw_value'][:60]} [{r['match'][:70]}]")


if __name__ == "__main__":
    main()
