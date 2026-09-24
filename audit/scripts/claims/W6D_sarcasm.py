"""W6D-ownerdocs: sarcasm-pilot claims O-097..O-103.

Recomputes, from the md5-verified cached rollouts, the four bag-of-words rows of
docs/thesis/sarcasm-pilot-negative-result.md 3.2, the marker rates of 3.1, and the
per-variant judged-role table of 3.3 (which no F-entry has checked per-variant).

Two independent implementations of the separability number:
  A: the repo's own bow / strip_question / grouped_auc (cap/scripts/score_sarcasm_markers.py)
  B: an own sklearn CountVectorizer + LeaveOneGroupOut pipeline written here.
Read-only.
"""
import json, hashlib, os, sys, re
from collections import defaultdict
import numpy as np

ROOT = r"C:/Users/SURAIYA/Documents/STUDY/thesis/expLB"
SARC = os.path.join(ROOT, "audit/cache/labpc/cap/run-sarc-pilot")
sys.path.insert(0, os.path.join(ROOT, "cap/scripts"))
import score_sarcasm_markers as S  # noqa: E402


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


print("=" * 92)
print("PROVENANCE")
print("=" * 92)
for rel in ["marker_report.json", "marker_report_partial.json",
            "responses/pc_ironic_critic.jsonl", "responses/pc_sycophant.jsonl",
            "responses/pc_sincere_critic.jsonl",
            "scores/pc_ironic_critic.json", "scores/pc_sycophant.json",
            "scores/pc_sincere_critic.json"]:
    p = os.path.join(SARC, rel)
    print(f"  {rel:<40} {os.path.getsize(p):>9} B  md5={md5(p)}")

rows = []
for role in ["pc_ironic_critic", "pc_sycophant", "pc_sincere_critic"]:
    f = os.path.join(SARC, "responses", role + ".jsonl")
    for line in open(f, encoding="utf-8").read().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        resp = r["conversation"][-1]["content"]
        rows.append({"role": role, "variant": r["prompt_index"], "qid": r["question_index"],
                     "question": r["question"], "resp": resp,
                     "clean": S.strip_question(resp, r["question"])})
print(f"\n  {len(rows)} rollouts; per role:",
      {r: sum(1 for x in rows if x['role'] == r) for r in sorted({x['role'] for x in rows})})
print("  variants per role:", sorted({x['variant'] for x in rows}),
      " questions:", len({x['qid'] for x in rows}))

# ---------------------------------------------------------------- B: own implementation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer

TOKEN = r"(?u)[a-z']+"


def own_grouped_auc(texts, y, groups, strip):
    txt = list(texts)
    if strip:
        pat = re.compile(r"\b(" + "|".join(sorted(S.MARKER_TOKENS)) + r")\b", re.I)
        txt = [pat.sub(" ", t) for t in txt]
    y = np.asarray(y); groups = np.asarray(groups)
    vec = CountVectorizer(lowercase=True, token_pattern=TOKEN, min_df=3)
    X = vec.fit_transform(txt).toarray().astype(np.float32)
    sc = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42).fit(X[tr], y[tr])
        sc[te] = clf.decision_function(X[te])
    return float(roc_auc_score(y, sc)), X.shape


def pair_rows(pos, neg, variants=None):
    sub = [r for r in rows if r["role"] in (pos, neg)]
    if variants is not None:
        sub = [r for r in sub if r["variant"] in variants]
    y = [1 if r["role"] == pos else 0 for r in sub]
    return sub, y


print("\n" + "=" * 92)
print("O-098/099/100/101  section 3.2 table -- bag-of-words AUROC, grouped by question id")
print("=" * 92)
CASES = [
    ("O-098", "all v1-v5   ironic vs sycophantic", "pc_ironic_critic", "pc_sycophant", None,
     0.953, 0.947, 200),
    ("O-099", "v1 only     ironic vs sycophantic", "pc_ironic_critic", "pc_sycophant", {0},
     1.000, 1.000, 40),
    ("O-100", "v4+v5       ironic vs sycophantic", "pc_ironic_critic", "pc_sycophant", {3, 4},
     0.919, 0.921, 80),
    ("O-101", "v4+v5       ironic vs sincere",     "pc_ironic_critic", "pc_sincere_critic", {3, 4},
     0.997, 0.996, 80),
]
print(f"  {'id':<7}{'subset':<38}{'n':>5}{'rawA':>8}{'strA':>8}{'rawB':>8}{'strB':>8}"
      f"{'docRaw':>8}{'docStr':>8}{'docN':>6}")
for cid, lab, pos, neg, vs, draw, dstr, dn in CASES:
    sub, y = pair_rows(pos, neg, vs)
    g = [r["qid"] for r in sub]
    texts = [r["clean"] for r in sub]
    aA_raw = S.grouped_auc(S.bow(texts, False), y, g)
    aA_str = S.grouped_auc(S.bow(texts, True), y, g)
    aB_raw, shp1 = own_grouped_auc(texts, y, g, False)
    aB_str, shp2 = own_grouped_auc(texts, y, g, True)
    print(f"  {cid:<7}{lab:<38}{len(sub):>5}{aA_raw:>8.4f}{aA_str:>8.4f}{aB_raw:>8.4f}"
          f"{aB_str:>8.4f}{draw:>8.3f}{dstr:>8.3f}{dn:>6}")

print("\n  Level 1 against the only stored separability figure:")
mr = json.load(open(os.path.join(SARC, "marker_report.json")))
print("   marker_report.json['lexical'] =", json.dumps(mr.get("lexical")))
mrp = json.load(open(os.path.join(SARC, "marker_report_partial.json")))
print("   marker_report_partial.json has 'lexical'? ", "lexical" in mrp)
print("   -> the v1-only, v4+v5 and ironic-vs-sincere rows exist in NO file (doc limitation 6).")

print("\n" + "=" * 92)
print("O-097 / 3.1  marker rates, and the pre-registered thresholds")
print("=" * 92)
for role in ["pc_ironic_critic", "pc_sincere_critic", "pc_sycophant"]:
    cells = []
    for v in range(5):
        sub = [r for r in rows if r["role"] == role and r["variant"] == v]
        cells.append(float(np.mean([sum(S.marker_hits(r["clean"]).values()) > 0 for r in sub])))
    print(f"  {role:<20}" + "".join(f"{c:>7.2f}" for c in cells),
          "   stored:", mr["markers"][role]["per_variant"])
agg = mr["per_marker"]["pc_sincere_critic"]
print("  sincere-arm marker counts (doc limitation 4):", {k: v for k, v in agg.items() if v})

print("\n  Gate thresholds as written in the design document:")
dsg = os.path.join(ROOT, "cap/docs/SARCASM_DESIGN.md")
txt = open(dsg, encoding="utf-8").read().splitlines()
for i, l in enumerate(txt):
    if "0.65" in l or "0.20" in l or "abandon" in l.lower():
        print(f"   SARCASM_DESIGN.md:{i+1}: {l.strip()[:160]}")

print("\n" + "=" * 92)
print("O-102 / 3.3  judged role score 0-3 per arm per variant")
print("=" * 92)
key = re.compile(r"^pos_p(\d+)_q(\d+)$")
print(f"  {'arm':<20}{'v1':>7}{'v2':>7}{'v3':>7}{'v4':>7}{'v5':>7}{'mean':>8}{'n':>5}")
docrow = {"pc_ironic_critic": [2.85, 2.85, 2.60, 2.55, 2.10, 2.59],
          "pc_sincere_critic": [2.85, 2.25, 2.40, 2.25, 2.10, 2.37],
          "pc_sycophant": [2.45, 2.25, 2.55, 2.15, 2.80, 2.44]}
for role in ["pc_ironic_critic", "pc_sincere_critic", "pc_sycophant"]:
    sc = json.load(open(os.path.join(SARC, "scores", role + ".json")))
    byv = defaultdict(list)
    for k, v in sc.items():
        m = key.match(k)
        assert m, k
        byv[int(m.group(1))].append(v)
    cells = [float(np.mean(byv[v])) for v in range(5)]
    allv = [v for k, v in sc.items()]
    print(f"  {role:<20}" + "".join(f"{c:>7.2f}" for c in cells)
          + f"{np.mean(allv):>8.2f}{len(allv):>5}")
    print(f"  {'(doc says)':<20}" + "".join(f"{c:>7.2f}" for c in docrow[role][:5])
          + f"{docrow[role][5]:>8.2f}")
    print(f"  {'(exact)':<20}" + "".join(f"{c:>7.4f}" for c in cells)
          + f"{np.mean(allv):>10.4f}")

print("\n  v1 -> v5 change: ironic %.2f -> %.2f ; sincere %.2f -> %.2f" % (
    np.mean([v for k, v in json.load(open(os.path.join(SARC, "scores/pc_ironic_critic.json"))).items()
             if key.match(k).group(1) == "0"]),
    np.mean([v for k, v in json.load(open(os.path.join(SARC, "scores/pc_ironic_critic.json"))).items()
             if key.match(k).group(1) == "4"]),
    np.mean([v for k, v in json.load(open(os.path.join(SARC, "scores/pc_sincere_critic.json"))).items()
             if key.match(k).group(1) == "0"]),
    np.mean([v for k, v in json.load(open(os.path.join(SARC, "scores/pc_sincere_critic.json"))).items()
             if key.match(k).group(1) == "4"])))

print("\n" + "=" * 92)
print("O-103  the chain's own verdict block")
print("=" * 92)
p = os.path.join(ROOT, "audit/phase2/out/P7_status_files.txt")
print(f"  source: {p} (batch P7's read-only dump of the lab logs/ status files)  md5={md5(p)}")
txt = open(p, encoding="utf-8", errors="replace").read().splitlines()
emit = False
for l in txt:
    if l.startswith("########## sarc_pilot"):
        emit = True
    elif l.startswith("##########"):
        emit = False
    if emit:
        print("   " + l)

doc = open(os.path.join(ROOT, "docs/thesis/sarcasm-pilot-negative-result.md"),
           encoding="utf-8").read().splitlines()
print("\n  What the document quotes (lines 199-204):")
for i in range(198, 204):
    print(f"   :{i+1}: {doc[i]}")
print("\n  and line 121:", doc[120].strip())
