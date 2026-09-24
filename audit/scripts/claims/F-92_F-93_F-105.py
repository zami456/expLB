"""Phase 2 batch P7 — F-92 / F-93 / F-105: the sarcasm pilot, recomputed from the raw rollouts.

Two independent passes over the same 300 responses:
  A. the repo's own functions (cap/scripts/score_sarcasm_markers.py: strip_question, bow,
     grouped_auc, MARKERS) — reproduces the pipeline exactly;
  B. an independent implementation (sklearn CountVectorizer + LeaveOneGroupOut logistic
     regression, my own question-echo stripping) — checks the METHOD, not just the arithmetic.

Plus a bootstrap over question ids for the decisive 0.921, which F-93 lists as missing.

Data: audit/cache/labpc/cap/run-sarc-pilot (md5-verified copy of the lab-only originals).
Read-only.
"""
import importlib.util, json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE = Path("audit/cache/labpc/cap/run-sarc-pilot")
SCORER = Path("cap/scripts/score_sarcasm_markers.py")

spec = importlib.util.spec_from_file_location("sarcscore", SCORER)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

rows = []
for f in sorted((CACHE / "responses").glob("*.jsonl")):
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        resp = r["conversation"][-1]["content"]
        rows.append({"role": f.stem, "variant": r["prompt_index"], "qid": r["question_index"],
                     "question": r["question"], "resp": resp,
                     "clean": S.strip_question(resp, r["question"])})
print(f"{len(rows)} responses, roles={sorted({r['role'] for r in rows})}, "
      f"variants={sorted({r['variant'] for r in rows})}, "
      f"questions={len({r['qid'] for r in rows})}")

# ------------------------------------------------------------------ F-92 marker rates
print("\n=== F-92 / F-93 : marker rate per role x variant (>=1 marker, echo stripped) ===")
print(f"  {'role':<20}{'v1':>7}{'v2':>7}{'v3':>7}{'v4':>7}{'v5':>7}{'all':>8}")
per_marker = {}
for role in sorted({r["role"] for r in rows}):
    cells = []
    for v in range(5):
        sub = [r for r in rows if r["role"] == role and r["variant"] == v]
        cells.append(float(np.mean([sum(S.marker_hits(r["clean"]).values()) > 0 for r in sub]))
                     if sub else float("nan"))
    sub = [r for r in rows if r["role"] == role]
    hits = [S.marker_hits(r["clean"]) for r in sub]
    agg = defaultdict(int)
    for h in hits:
        for k, c in h.items():
            agg[k] += c
    per_marker[role] = dict(agg)
    print(f"  {role:<20}" + "".join(f"{c:>7.2f}" for c in cells)
          + f"{np.mean([sum(h.values()) > 0 for h in hits]):>8.2f}")
print("\n  marker counts by type:")
for role, agg in per_marker.items():
    print(f"  {role:<20}" + "  ".join(f"{k}={v}" for k, v in sorted(agg.items(), key=lambda kv: -kv[1]) if v))

# the ironic-arm-only counts F-92 quotes (n=100, the partial report)
print("\n  F-92 quotes the IRONIC ARM ONLY (n=100): "
      + "  ".join(f"{k}={v}" for k, v in sorted(per_marker['pc_ironic_critic'].items(),
                                                key=lambda kv: -kv[1]) if v))

# ------------------------------------------------------------------ F-93 / F-105 separability
SUBSETS = [("all v1-v5", [0, 1, 2, 3, 4]), ("v1 only", [0]), ("v4+v5", [3, 4])]
PAIRS = [("pc_sycophant", "ironic vs sycophant"), ("pc_sincere_critic", "ironic vs sincere")]


def pass_A(sub):
    y = [1 if r["role"] == "pc_ironic_critic" else 0 for r in sub]
    g = [r["qid"] for r in sub]
    t = [r["clean"] for r in sub]
    return S.grouped_auc(S.bow(t, False), y, g), S.grouped_auc(S.bow(t, True), y, g)


def pass_B(sub, strip_markers):
    """Independent: my own echo strip, sklearn CountVectorizer, LeaveOneGroupOut."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut
    y = np.array([1 if r["role"] == "pc_ironic_critic" else 0 for r in sub])
    g = np.array([r["qid"] for r in sub])
    txt = []
    for r in sub:
        t = r["resp"].lower()
        qw = re.findall(r"[a-z']+", r["question"].lower())
        for n in range(len(qw), 3, -1):                    # drop echoed >=4-grams
            for i in range(len(qw) - n + 1):
                frag = " ".join(qw[i:i + n])
                if frag in t:
                    t = t.replace(frag, " ")
        if strip_markers:
            t = " ".join(w for w in re.findall(r"[a-z']+", t) if w not in S.MARKER_TOKENS)
        txt.append(t)
    vec = CountVectorizer(min_df=3, token_pattern=r"[a-z']+", lowercase=True)
    X = vec.fit_transform(txt).toarray().astype(float)
    sc = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        sc[te] = LogisticRegression(max_iter=2000, C=1.0,
                                    random_state=42).fit(X[tr], y[tr]).decision_function(X[te])
    return float(roc_auc_score(y, sc))


print("\n=== F-93 / F-105 : lexical separability (grouped by question id) ===")
print(f"  {'subset':<12}{'pair':<22}{'A raw':>8}{'A strip':>9}{'B raw':>8}{'B strip':>9}{'n':>5}"
      "   F-93/F-105 state")
STATED = {("all v1-v5", "ironic vs sycophant"): "0.953 / 0.947",
          ("v1 only", "ironic vs sycophant"): "1.000 / 1.000",
          ("v4+v5", "ironic vs sycophant"): "0.919 / 0.921",
          ("v4+v5", "ironic vs sincere"): "0.997 / 0.996"}
for other, plabel in PAIRS:
    for slabel, vs in SUBSETS:
        sub = [r for r in rows
               if r["role"] in ("pc_ironic_critic", other) and r["variant"] in vs]
        if len({r["role"] for r in sub}) != 2:
            continue
        a_raw, a_str = pass_A(sub)
        b_raw, b_str = pass_B(sub, False), pass_B(sub, True)
        print(f"  {slabel:<12}{plabel:<22}{a_raw:>8.4f}{a_str:>9.4f}{b_raw:>8.4f}{b_str:>9.4f}"
              f"{len(sub):>5}   {STATED.get((slabel, plabel), '-')}")

# bootstrap over questions for the decisive cell (F-93 caveat 1: "has no confidence interval")
print("\n  bootstrap over the 20 question ids, v4+v5 ironic vs sycophant, stripped (B=2000):")
sub = [r for r in rows
       if r["role"] in ("pc_ironic_critic", "pc_sycophant") and r["variant"] in (3, 4)]
y = np.array([1 if r["role"] == "pc_ironic_critic" else 0 for r in sub])
g = np.array([r["qid"] for r in sub])
t = [r["clean"] for r in sub]
X = S.bow(t, True)
# out-of-fold scores once, then resample questions (the unit) on the fixed scores
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sc = np.zeros(len(y))
for q in np.unique(g):
    te = g == q
    tr = ~te
    sc[te] = LogisticRegression(max_iter=2000, C=1.0,
                                random_state=42).fit(X[tr], y[tr]).decision_function(X[te])
print(f"    point (out-of-fold) {roc_auc_score(y, sc):.4f}")
rng = np.random.default_rng(42)
qs = np.unique(g)
bs = []
for _ in range(2000):
    pick = rng.integers(0, len(qs), len(qs))
    idx = np.concatenate([np.where(g == qs[p])[0] for p in pick])
    if len(np.unique(y[idx])) < 2:
        continue
    bs.append(roc_auc_score(y[idx], sc[idx]))
bs = np.sort(bs)
print(f"    95% CI over questions [{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]"
      f"   P(<=0.65 pre-registered ceiling) = {float((bs <= 0.65).mean()):.4f}")

# ------------------------------------------------------------------ feature lists (F-93/F-105)
print("\n=== F-105 : discriminating words, v4+v5, markers stripped, single fit ===")
fsub = sub
vocab = defaultdict(int)
for r in fsub:
    for w in {x for x in S.WORD.findall(r["clean"].lower()) if x not in S.MARKER_TOKENS}:
        vocab[w] += 1
keep = sorted([w for w, c in vocab.items() if c >= 3])
Xf = S.bow([r["clean"] for r in fsub], True)
yf = np.array([1 if r["role"] == "pc_ironic_critic" else 0 for r in fsub])
clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42).fit(Xf, yf)
order = np.argsort(clf.coef_[0])
ironic = [keep[i] for i in order[::-1][:25]]
syco = [keep[i] for i in order[:25]]
print(f"  vocab={len(keep)}  n={len(fsub)}")
print("  ironic top-12   : " + ", ".join(ironic[:12]))
print("  sycophant top-12: " + ", ".join(syco[:12]))
F93_IR = "of that's ability undoubtedly i i'm fact like definitely decision think very".split()
F93_SY = ("plan can off good get you're straightforward team us you you've more pay run").split()
print(f"  F-93's 12 ironic words in my top-25:    "
      f"{sum(w in ironic for w in F93_IR)}/{len(F93_IR)}  "
      f"(missing: {[w for w in F93_IR if w not in ironic]})")
print(f"  F-93's 14 sycophant words in my top-25: "
      f"{sum(w in syco for w in F93_SY)}/{len(F93_SY)}  "
      f"(missing: {[w for w in F93_SY if w not in syco]})")

# ------------------------------------------------------------------ polarity (F-93 amendment / F-105)
print("\n=== F-93 amendment / F-105 : polarity (judge role score 0-3) ===")
keyre = re.compile(r"^pos_p(\d+)_q(\d+)$")
print(f"  {'arm':<20}{'mean':>7}{'n':>5}{'>=2':>7}   v1    v2    v3    v4    v5")
for f in sorted((CACHE / "scores").glob("*.json")):
    j = json.loads(f.read_text(encoding="utf-8"))
    byv = defaultdict(list)
    vals = []
    for k, v in j.items():
        m = keyre.match(str(k))
        if m and isinstance(v, (int, float)):
            byv[int(m.group(1))].append(float(v))
            vals.append(float(v))
    cells = "  ".join(f"{np.mean(byv[v]):.2f}" if byv.get(v) else " -- " for v in range(5))
    print(f"  {f.stem:<20}{np.mean(vals):>7.2f}{len(vals):>5}{np.mean([x >= 2 for x in vals]):>7.2f}"
          f"   {cells}")
