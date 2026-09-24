"""P11-abstract: check the collaborator's 2026-09-21 abstract (A-02..A-11) plus D-080, D-100, D-126.

Everything is read out of the collaborator's branch with `git show` (NO checkout) or out of the
PDFs in papers/ with pypdf. Nothing is read by eye: every mean is recomputed here.

Run:  python audit/scripts/claims/P11_abstract.py
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
BRANCH = "origin/feat/persona-pc-ood"
RES = "src/probes/deception-detection/results"

DS6 = [
    "convince_me__plain",
    "harm_refusal_choice__plain",
    "harm_refusal_knowledge_report__plain",
    "insider_trading_cadenza__plain",
    "instruct_dishonesty__plain",
    "repe_honesty__plain",
]


def show(path: str) -> str:
    out = subprocess.run(
        ["git", "show", f"{BRANCH}:{path}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if out.returncode != 0:
        raise RuntimeError(f"git show failed for {path}: {out.stderr[:300]}")
    return out.stdout


def csv(path: str, **kw) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(show(path)), **kw)


def hr(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def chk(label: str, stated, got, tol=5e-4) -> None:
    ok = "MATCH" if (got is not None and abs(float(stated) - float(got)) <= tol) else "MISMATCH"
    print(f"  {label:<62s} stated={stated:<8} file={got:.6f}  -> {ok}")


# --------------------------------------------------------------------------------------
hr("commit / provenance")
head = subprocess.run(
    ["git", "rev-parse", BRANCH], cwd=REPO, capture_output=True, text=True
).stdout.strip()
print("  branch head:", head)
log = subprocess.run(
    [
        "git", "log", "--format=%h %ad %an %s", "--date=iso", BRANCH, "--",
        f"{RES}/persona_ood_ext_L40/summary.csv",
    ],
    cwd=REPO, capture_output=True, text=True,
).stdout.strip()
print("  history of persona_ood_ext_L40/summary.csv:")
for line in log.splitlines():
    print("   ", line)
print("  (the abstract was pasted 2026-09-21 10:38Z; the results predate it)")

summ = {L: csv(f"{RES}/persona_ood_ext_L{L}/summary.csv").set_index("features") for L in (20, 40, 44)}
summ_pub = {20: csv(f"{RES}/persona_ood/summary.csv").set_index("features"),
            40: csv(f"{RES}/persona_ood_L40/summary.csv").set_index("features")}
bal = {L: csv(f"{RES}/persona_ood_ext_L{L}/summary_balacc.csv").set_index("features") for L in (20, 40, 44)}
zs = {L: csv(f"{RES}/persona_ood_ext_L{L}/zero_shot.csv").set_index("dataset") for L in (20, 40, 44)}
ops = {L: csv(f"{RES}/persona_ood_ext_L{L}/zero_shot_operating.csv") for L in (20, 40, 44)}


def matrix(L: int, feat: str) -> pd.DataFrame:
    return csv(f"{RES}/persona_ood_ext_L{L}/matrix_{feat}.csv", index_col=0)


# ======================================================================================
hr("A-02  the published paper: persona coordinates = PCs of contrastive persona vectors;"
   " models Llama 3.2-3B and Llama 3-8B")
try:
    from pypdf import PdfReader

    r = PdfReader(str(REPO / "papers" / "Do Linear Probes Generalize Better in Persona Coordinates.pdf"))
    pages = [p.extract_text() or "" for p in r.pages]
    full = "\n".join(pages)
    for needle in [
        "Llama 3.2-3B,",
        "Llama 3-8B. For each model",
        "first principal components",
        "unsupervised PCA",
        "generalize better than",
        "Llama 3-8B evaluations are reported in Ap",
        "broadly sup",
    ]:
        pg = next((i + 1 for i, t in enumerate(pages) if needle in t), None)
        j = full.find(needle)
        snip = full[max(0, j - 110): j + 130].replace("\n", " ") if j >= 0 else "NOT FOUND"
        print(f"  [p{pg}] ...{snip}...")
except Exception as e:  # pragma: no cover
    print("  PDF read failed:", e)

# ======================================================================================
hr("A-03  'six Liars' Bench datasets at layers 20 and 40 ... fifteen feature sets'")
for L in (20, 40, 44):
    print(f"  L{L}: feature rows in persona_ood_ext_L{L}/summary.csv = {len(summ[L])}")
    print(f"       features = {list(summ[L].index)}")
for L in (20, 40):
    print(f"  L{L}: feature rows in the PUBLISHED-bank dir = {len(summ_pub[L])}")
print("  datasets scored in zero_shot.csv (L40):")
for d, row in zs[40].iterrows():
    print(f"       {d:<40s} n={row['n']:>6}  n_deceptive={row['n_deceptive']:>6}")
print("  -> 'XXXX-20__plain' is the alpaca control (n_deceptive 0), not a transfer dataset")
print("  which of the six are LIARS' BENCH datasets?")
print("     convince_me / harm_refusal_choice / harm_refusal_knowledge_report /")
print("     insider_trading_cadenza / instruct_dishonesty  = 5 (Liars' Bench)")
print("     repe_honesty                                    = RepE honesty (Zou et al.), NOT Liars' Bench")
print("  his own operating table names the 5-dataset mean row \"Average (Liars' Bench)\":")
print("    ", sorted(set(ops[40]["dataset"]))[:2])
print("  layer dirs on his branch:")
dirs = subprocess.run(
    ["git", "ls-tree", "--name-only", f"{BRANCH}:{RES}"], cwd=REPO, capture_output=True, text=True
).stdout.split()
print("   ", [d for d in dirs if d.startswith("persona_ood")])

# ======================================================================================
hr("A-04  0.904 / 0.916 for the 1-d contrast vs 0.648 / 0.732 for the 8192-d probes;"
   " 'wins on every held-out dataset individually'")
for L, v in ((20, 0.904), (40, 0.916)):
    chk(f"ext_axis1 off_diagonal_mean L{L}", v, summ[L].loc["ext_axis1", "off_diagonal_mean"])
for L, v in ((20, 0.648), (40, 0.732)):
    chk(f"baseline  off_diagonal_mean L{L}", v, summ[L].loc["baseline", "off_diagonal_mean"])
print("  -- recomputed from the 30 off-diagonal cells of his own matrices --")
for L in (20, 40, 44):
    for feat in ("ext_axis1", "baseline", "axis1"):
        m = matrix(L, feat).loc[DS6, DS6].to_numpy(dtype=float)
        off = m[~np.eye(6, dtype=bool)]
        print(f"  L{L} {feat:<10s} off-diag mean over {off.size} cells = {off.mean():.6f}")
print("  -- 'wins on every held-out dataset': column means over the 5 training sets --")
for L in (20, 40, 44):
    me = matrix(L, "ext_axis1").loc[DS6, DS6].to_numpy(dtype=float)
    mb = matrix(L, "baseline").loc[DS6, DS6].to_numpy(dtype=float)
    mask = ~np.eye(6, dtype=bool)
    wins = 0
    for j, d in enumerate(DS6):
        ce = me[:, j][mask[:, j]].mean()
        cb = mb[:, j][mask[:, j]].mean()
        wins += ce > cb
        print(f"  L{L} held-out {d:<40s} ext={ce:.4f}  raw={cb:.4f}  delta={ce - cb:+.4f}")
    print(f"  L{L} -> ext wins on {wins}/6 held-out datasets")
print("  -- ATTRIBUTION: which contrast is 0.904/0.916? --")
for L in (20, 40):
    a = zs[L]["auroc_axis"].loc[DS6].mean()
    e = zs[L]["auroc_ext_axis"].loc[DS6].mean()
    d = zs[L]["auroc_denial_axis"].loc[DS6].mean()
    print(f"  L{L} zero-shot mean over the 6: paper 8v7 axis1={a:.4f}   extended 12v13={e:.4f}   4v6 denial={d:.4f}")
print("  extended_arms() in his scripts/persona_ood.py builds 8+4 vs 7+6 (see A-11 section)")

# ======================================================================================
hr("A-05  'its generalization gap is zero; the trained probes lose 0.31 and 0.25'")
for L, v in ((20, 0.31), (40, 0.25)):
    diag = summ[L].loc["baseline", "diagonal_mean"]
    off = summ[L].loc["baseline", "off_diagonal_mean"]
    chk(f"baseline diagonal-off_diagonal L{L}", v, diag - off, tol=5e-3)
    print(f"        (diagonal {diag:.6f} - off-diagonal {off:.6f})")
for L in (20, 40, 44):
    diag = summ[L].loc["ext_axis1", "diagonal_mean"]
    off = summ[L].loc["ext_axis1", "off_diagonal_mean"]
    print(f"  L{L} ext_axis1 diagonal {diag:.6f} - off-diagonal {off:.6f} = {diag - off:+.6f}")
print("  L44 baseline gap:", f"{summ[44].loc['baseline','diagonal_mean'] - summ[44].loc['baseline','off_diagonal_mean']:.6f}")

# ======================================================================================
hr("A-06  'the PCs do not contribute: top-3 reaches 0.684 and 0.866, and at L40 its first"
   " component scores 0.09 below the contrast it is derived from'")
for L, v in ((20, 0.684), (40, 0.866)):
    chk(f"ext_pc3 off_diagonal_mean L{L}", v, summ[L].loc["ext_pc3", "off_diagonal_mean"])
print(f"  ext_pc3 L44 = {summ[44].loc['ext_pc3','off_diagonal_mean']:.6f} (not quoted)")
print("  -- candidate readings of 'first component 0.09 below the contrast', L40 --")
cands = {
    "ext_pc1 vs ext_axis1 (extended bank)": ("ext_pc1", "ext_axis1"),
    "pc1 vs axis1 (published bank, ext dir)": ("pc1", "axis1"),
    "ext_pc1 vs pc1": ("ext_pc1", "pc1"),
    "ext_pc1 vs axis1": ("ext_pc1", "axis1"),
    "denial_pc3 vs denial_axis1": ("denial_pc3", "denial_axis1"),
    "syc_pc1 vs syc_axis1": ("syc_pc1", "syc_axis1"),
    "ext_pc3 vs ext_axis1": ("ext_pc3", "ext_axis1"),
}
for lab, (a, b) in cands.items():
    va = summ[40].loc[a, "off_diagonal_mean"]
    vb = summ[40].loc[b, "off_diagonal_mean"]
    print(f"    {lab:<42s} {a}={va:.4f}  {b}={vb:.4f}  {a}-{b} = {va - vb:+.4f}")
print("  published-bank dir (persona_ood_L40): pc1 =",
      f"{summ_pub[40].loc['pc1','off_diagonal_mean']:.4f}  axis1 =",
      f"{summ_pub[40].loc['axis1','off_diagonal_mean']:.4f}",
      f"  pc1-axis1 = {summ_pub[40].loc['pc1','off_diagonal_mean'] - summ_pub[40].loc['axis1','off_diagonal_mean']:+.4f}")
print("  -- exhaustive scan: every ordered pair of features in L40 summary with |gap - 0.09| <= 0.005 --")
feats = list(summ[40].index)
for a in feats:
    for b in feats:
        if a == b:
            continue
        g = summ[40].loc[b, "off_diagonal_mean"] - summ[40].loc[a, "off_diagonal_mean"]
        if abs(g - 0.09) <= 0.005:
            print(f"    {a} is {g:.4f} below {b}")
print("  -- does PC1 beat the contrast it derives from? (all three layers) --")
for L in (20, 40, 44):
    zl = zs[L]
    print(f"  L{L} zero-shot mean over the 6:"
          f"  pc1={zl['auroc_pc1'].loc[DS6].mean():.4f} vs axis1={zl['auroc_axis'].loc[DS6].mean():.4f}"
          f"  ({zl['auroc_pc1'].loc[DS6].mean() - zl['auroc_axis'].loc[DS6].mean():+.4f})"
          f"   |  ext_pc1={zl['auroc_ext_pc1'].loc[DS6].mean():.4f} vs ext_axis1={zl['auroc_ext_axis'].loc[DS6].mean():.4f}"
          f"  ({zl['auroc_ext_pc1'].loc[DS6].mean() - zl['auroc_ext_axis'].loc[DS6].mean():+.4f})")
    print(f"       trained off-diag:  pc1={summ[L].loc['pc1','off_diagonal_mean']:.4f} vs axis1={summ[L].loc['axis1','off_diagonal_mean']:.4f}"
          f"  ({summ[L].loc['pc1','off_diagonal_mean'] - summ[L].loc['axis1','off_diagonal_mean']:+.4f})")

# ======================================================================================
hr("A-08  Liars' Bench Mean Probe 0.12 / Upper-Bound 0.97 on HP-KR; published 8v7 inverts"
   " at L20 (0.148) and barely clears chance at L40 (0.593)")
for L, v in ((20, 0.148), (40, 0.593)):
    chk(f"axis1 zero-shot HP-KR L{L}", v, zs[L].loc["harm_refusal_knowledge_report__plain", "auroc_axis"])
print(f"  axis1 zero-shot HP-KR L44 = {zs[44].loc['harm_refusal_knowledge_report__plain','auroc_axis']:.6f} (not quoted in the abstract)")
try:
    from pypdf import PdfReader

    lb = PdfReader(str(REPO / "papers" / "2511.16035v2 (1).pdf"))
    t6 = lb.pages[53].extract_text()
    t8 = lb.pages[55].extract_text()
    for tag, txt in (("Table 6 Mean Probe", t6), ("Table 8 Upper Bound Probe", t8)):
        i = txt.find("AUROC")
        j = txt.find("Llama 3.3 70B", i)
        print(f"  {tag}: {txt[j:j + 90]!r}")
    # hardest-column check for the Mean Probe
    print("  Mean Probe AUROC (Table 6), Llama 3.3 70B row: HP-C .41 HP-KR .12 CG 1.00 ST .57 ID .95 GS .45 IT .93")
    print("     -> HP-KR is the minimum for the Mean Probe on that model, and for the across-model average (.36)")
    print("  BUT the Unrelated-Question Classifier (Table 11) scores HP-KR 0.90 on the same model,")
    print("     and LLM-as-a-Judge (Table 10) scores GS 0.45 below HP-KR 0.48 -> 'hardest column' is method-specific")
except Exception as e:  # pragma: no cover
    print("  PDF read failed:", e)

# ======================================================================================
hr("A-10  'the independently constructed sycophancy axis inverts further still (0.043)'")
for L in (20, 40, 44):
    print(f"  L{L} syc_axis1 zero-shot HP-KR = {zs[L].loc['harm_refusal_knowledge_report__plain','auroc_syc_axis']:.6f}"
          f"   syc_pc1 = {zs[L].loc['harm_refusal_knowledge_report__plain','auroc_syc_pc1']:.6f}"
          f"   (axis1 = {zs[L].loc['harm_refusal_knowledge_report__plain','auroc_axis']:.6f})")
chk("syc_axis1 zero-shot HP-KR L20", 0.043,
    zs[20].loc["harm_refusal_knowledge_report__plain", "auroc_syc_axis"])
print("  -- is the sycophancy axis 'independently constructed'? --")
print("     his persona_pcs.py builds it from SYCOPHANCY_PERSONAS - ANTI_SYCOPHANCY_PERSONAS")
print("     of the SAME paper16_L20 bank that gives the deception axis (see the file listing below)")

# ======================================================================================
hr("A-11  'adding four feigned-ignorance personas against six candour personas raises zero-shot"
   " AUROC to 0.882 and 0.912, and lifts transfer on the four unrelated datasets by 0.18-0.21'")
for L, v in ((20, 0.882), (40, 0.912)):
    chk(f"ext_axis1 zero-shot HP-KR L{L}", v,
        zs[L].loc["harm_refusal_knowledge_report__plain", "auroc_ext_axis"])
print(f"  ext_axis1 zero-shot HP-KR L44 = {zs[44].loc['harm_refusal_knowledge_report__plain','auroc_ext_axis']:.6f}")
print(f"  denial_axis1 (4v6 alone) HP-KR: L20={zs[20].loc['harm_refusal_knowledge_report__plain','auroc_denial_axis']:.4f}"
      f"  L40={zs[40].loc['harm_refusal_knowledge_report__plain','auroc_denial_axis']:.4f}"
      f"  L44={zs[44].loc['harm_refusal_knowledge_report__plain','auroc_denial_axis']:.4f}")
print("  -- persona counts, from his source and the owner's bank --")
src = show("src/probes/deception-detection/deception_detection/persona_pcs.py")
import re

for name in ("KNOWLEDGE_DENIAL_PERSONAS", "KNOWLEDGE_CANDOUR_PERSONAS", "DECEPTION_PERSONAS", "HONESTY_PERSONAS"):
    m = re.search(rf"^{name} = \((.*?)\)\n", src, re.S | re.M)
    roles = re.findall(r'"([^"]+)"', m.group(1))
    print(f"    his {name:<28s} n={len(roles)}  {roles}")
import json

groups = json.loads((REPO / "kd" / "personas" / "kd_role_groups.json").read_text(encoding="utf-8"))["groups"]
for g in ("knowledge_denial", "sincere_ignorance", "plain_assertion"):
    print(f"    owner kd_role_groups.json[{g:<18s}] n={len(groups[g]['roles'])}  {groups[g]['roles']}")
kd = set(groups["knowledge_denial"]["roles"])
cand = set(groups["sincere_ignorance"]["roles"]) | set(groups["plain_assertion"]["roles"])
his_kd = set(re.findall(r'"([^"]+)"', re.search(r"^KNOWLEDGE_DENIAL_PERSONAS = \((.*?)\)\n", src, re.S | re.M).group(1)))
his_cd = set(re.findall(r'"([^"]+)"', re.search(r"^KNOWLEDGE_CANDOUR_PERSONAS = \((.*?)\)\n", src, re.S | re.M).group(1)))
print(f"    his 4 denial  == owner knowledge_denial       : {his_kd == kd}")
print(f"    his 6 candour == owner sincere_ignorance+plain_assertion : {his_cd == cand}")
files = subprocess.run(
    ["git", "ls-tree", "--name-only", f"{BRANCH}:src/probes/deception-detection/persona including knowlede denial"],
    cwd=REPO, capture_output=True, text=True, encoding="utf-8",
).stdout.split("\n")
files = [f for f in files if f.strip()]
print(f"    his committed extended vector folder holds {len(files)} .pt files")
print("  -- 'lifts transfer on the four unrelated datasets by 0.18-0.21' --")
print("     reading A: TRAINED column means (mean over the 5 training sets), ext_axis1 - axis1")
for L in (20, 40, 44):
    me = matrix(L, "ext_axis1").loc[DS6, DS6].to_numpy(dtype=float)
    ma = matrix(L, "axis1").loc[DS6, DS6].to_numpy(dtype=float)
    mask = ~np.eye(6, dtype=bool)
    print(f"     L{L}:")
    for j, d in enumerate(DS6):
        ce = me[:, j][mask[:, j]].mean()
        ca = ma[:, j][mask[:, j]].mean()
        flag = "   <-- in 0.18-0.21" if 0.175 <= ce - ca <= 0.215 else ""
        print(f"        {d:<40s} axis1={ca:.4f} ext={ce:.4f}  delta={ce - ca:+.4f}{flag}")
print("     reading B: ZERO-SHOT per dataset, ext_axis - axis (the protocol the 0.882/0.912 use)")
for L in (20, 40, 44):
    print(f"     L{L}:")
    for d in DS6:
        a = zs[L].loc[d, "auroc_axis"]
        e = zs[L].loc[d, "auroc_ext_axis"]
        print(f"        {d:<40s} axis1={a:.4f} ext={e:.4f}  delta={e - a:+.4f}")
print("     why reading A is large at L20: the axis1 transfer matrix has ONE sign-flipped row")
print("     (the probe trained on HP-KR), which drags every other column down by 1/5 of the flip:")
print(matrix(20, "axis1").to_string())

# ======================================================================================
hr("D-080  HP-KR: 0.148 at L20; 0.593 and 0.589 at L40 and L44")
for L, v in ((20, 0.148), (40, 0.593), (44, 0.589)):
    chk(f"axis1 zero-shot HP-KR L{L}", v, zs[L].loc["harm_refusal_knowledge_report__plain", "auroc_axis"])

# ======================================================================================
hr("D-100  at L40 the probe trained on ID scores higher on CG than the contrast does (0.98 vs 0.91)")
b = matrix(40, "baseline")
e = matrix(40, "ext_axis1")
chk("baseline  train=instruct_dishonesty test=convince_me", 0.98,
    float(b.loc["instruct_dishonesty__plain", "convince_me__plain"]), tol=5e-3)
chk("ext_axis1 train=instruct_dishonesty test=convince_me", 0.91,
    float(e.loc["instruct_dishonesty__plain", "convince_me__plain"]), tol=5e-3)
print("  how common is that? raw-beats-ext off-diagonal cells at L40:")
mb = b.loc[DS6, DS6].to_numpy(dtype=float)
me = e.loc[DS6, DS6].to_numpy(dtype=float)
mask = ~np.eye(6, dtype=bool)
print(f"    raw > ext in {int((mb[mask] > me[mask]).sum())} of {int(mask.sum())} off-diagonal cells")
for L in (20, 44):
    mb2 = matrix(L, "baseline").loc[DS6, DS6].to_numpy(dtype=float)
    me2 = matrix(L, "ext_axis1").loc[DS6, DS6].to_numpy(dtype=float)
    print(f"    L{L}: raw > ext in {int((mb2[mask] > me2[mask]).sum())} of {int(mask.sum())} off-diagonal cells")

# ======================================================================================
hr("D-126  layer 40 is best for the persona contrasts: transfer 0.904/0.916/0.903 and"
   " balanced accuracy 0.591/0.694/0.666")
for L, v in ((20, 0.904), (40, 0.916), (44, 0.903)):
    chk(f"ext_axis1 off_diagonal_mean L{L}", v, summ[L].loc["ext_axis1", "off_diagonal_mean"])
for L, v in ((20, 0.591), (40, 0.694), (44, 0.666)):
    row = ops[L][(ops[L]["arm"] == "extended axis (12v13)")
                 & (ops[L]["dataset"] == "Average (Liars' Bench)")
                 & (ops[L]["threshold_source"] == "control")]
    chk(f"extended axis balacc, Average(LB), control threshold L{L}", v, float(row["balacc"].iloc[0]), tol=5e-4)
print("  is L40 the best of the three layers for every persona contrast?")
for feat in ("ext_axis1", "denial_axis1", "axis1", "pc1", "ext_pc3", "syc_axis1"):
    vals = {L: summ[L].loc[feat, "off_diagonal_mean"] for L in (20, 40, 44)}
    best = max(vals, key=vals.get)
    print(f"    {feat:<14s} L20={vals[20]:.4f} L40={vals[40]:.4f} L44={vals[44]:.4f}  best=L{best}")
for arm in ("extended axis (12v13)", "knowledge-denial axis (4v6)", "deception axis (8v7)",
            "sycophancy axis (5v5)", "base probe (RepE-trained)"):
    vals = {}
    for L in (20, 40, 44):
        row = ops[L][(ops[L]["arm"] == arm) & (ops[L]["dataset"] == "Average (Liars' Bench)")
                     & (ops[L]["threshold_source"] == "control")]
        vals[L] = float(row["balacc"].iloc[0])
    best = max(vals, key=vals.get)
    print(f"    balacc {arm:<30s} L20={vals[20]:.4f} L40={vals[40]:.4f} L44={vals[44]:.4f}  best=L{best}")
print("  NB only three layers exist on his branch (20/40/44); no other depth was run.")

print("\nDONE")
