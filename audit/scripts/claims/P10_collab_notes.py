"""Phase 2 batch P10-collab-notes: re-derive the collaborator's result-note numbers
(H-001, H-003, H-005, H-007, H-008, H-009, H-010, H-012, H-013, H-014).

Read-only. Every input is read with `git show origin/feat/persona-pc-ood:<path>`
so nothing is checked out and the script is re-runnable from the repo root.

Sources checked:
  results/XXXX-9-llama-all-followup/{results_table.csv,cfg.yaml}   (probe-replication note)
  results/loo_L20/*, results/loo_L40/*                             (leave-one-out note)
  results/persona_ood/*, results/persona_ood_ext_L{20,40}/*        (persona_ood note)
  papers/2511.16035v2 (1).pdf  Table 7 p.55                        (published baseline)

Usage:  python audit/scripts/claims/P10_collab_notes.py
"""
import csv
import io
import subprocess
import statistics

BR = "origin/feat/persona-pc-ood"
RES = "src/probes/deception-detection/results"
EVAL5 = [
    "convince_me__plain",
    "harm_refusal_choice__plain",
    "harm_refusal_knowledge_report__plain",
    "insider_trading_cadenza__plain",
    "instruct_dishonesty__plain",
]


def gitshow(path):
    out = subprocess.run(["git", "show", f"{BR}:{path}"],
                         capture_output=True, check=True)
    return out.stdout.decode("utf-8")


def rows(path):
    return list(csv.DictReader(io.StringIO(gitshow(path))))


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def f(x, n=4):
    return f"{x:.{n}f}"


# ---------------------------------------------------------------- H-001 / H-003 / H-005
hdr("H-001 / H-003 / H-005  follow-up probe, XXXX-9-llama-all-followup/results_table.csv")
ft = {r[""]: r for r in rows(f"{RES}/XXXX-9-llama-all-followup/results_table.csv")}
# transcription of Liars' Bench Table 7, Llama 3.3 70B row (verified against the PDF below)
TABLE7 = {  # dataset: (auroc, balacc@1%FPR)
    "convince_me__plain": (0.75, 0.60),
    "harm_refusal_choice__plain": (0.50, 0.50),
    "harm_refusal_knowledge_report__plain": (0.14, 0.26),
    "insider_trading_cadenza__plain": (0.85, 0.66),
    "instruct_dishonesty__plain": (0.93, 0.87),
}
meas_auroc, meas_bal = [], []
print(f"{'dataset':38s} {'AUROC':>8s} {'T7':>6s} {'|d|':>7s} {'balacc':>8s} {'T7':>6s} {'|d|':>7s}")
for d in EVAL5:
    a = float(ft[d]["auroc"])
    b = float(ft[d]["balanced_accuracy_at_fpr_1%"])
    meas_auroc.append(a)
    meas_bal.append(b)
    ta, tb = TABLE7[d]
    print(f"{d:38s} {a:8.4f} {ta:6.2f} {abs(a-ta):7.4f} {b:8.4f} {tb:6.2f} {abs(b-tb):7.4f}")
ma, mb = sum(meas_auroc) / 5, sum(meas_bal) / 5
t7a = sum(v[0] for v in TABLE7.values()) / 5
t7b = sum(v[1] for v in TABLE7.values()) / 5
print(f"{'5-shared MEAN':38s} {ma:8.4f} {t7a:6.3f} {abs(ma-t7a):7.4f} {mb:8.4f} {t7b:6.3f} {abs(mb-t7b):7.4f}")
print(f"\nH-001 'mean AUROC matches the paper to within 0.002': |d| = {abs(ma-t7a):.5f}")
print("H-001 'per dataset, within ~0.05 everywhere': max per-dataset |dAUROC| = "
      f"{max(abs(float(ft[d]['auroc'])-TABLE7[d][0]) for d in EVAL5):.4f}")
over = [(d, abs(float(ft[d]["auroc"]) - TABLE7[d][0])) for d in EVAL5
        if abs(float(ft[d]["auroc"]) - TABLE7[d][0]) > 0.05]
print("   datasets whose AUROC gap EXCEEDS 0.05:", [(d, round(v, 4)) for d, v in over])
print("H-002 'the one real gap is CG balanced accuracy, 0.80 vs 0.60': per-dataset |dbalacc| = "
      + ", ".join(f"{d.split('__')[0][:14]}={abs(float(ft[d]['balanced_accuracy_at_fpr_1%'])-TABLE7[d][1]):.3f}"
                  for d in EVAL5))
print("H-003 HP-KR follow-up AUROC =", f(float(ft["harm_refusal_knowledge_report__plain"]["auroc"])))
print("H-004 HP-KR realised FPR at the 1%-on-alpaca threshold =",
      f(float(ft["harm_refusal_knowledge_report__plain"]["fpr 1.0%"])),
      " balacc =", f(float(ft["harm_refusal_knowledge_report__plain"]["balanced_accuracy_at_fpr_1%"])))
print("RepE validation row:", {k: v for k, v in ft["repe_honesty__plain_val"].items()
                               if k in ("auroc", "balanced_accuracy_at_fpr_1%", "recall 1.0%")})

hdr("H-003 / H-005  the 'Mean Probe here' and 'published persona grouping' comparators")
op20 = rows(f"{RES}/persona_ood_ext_L20/zero_shot_operating.csv")
base20 = {r["dataset"]: r for r in op20
          if r["arm"] == "base probe (RepE-trained)" and r["threshold_source"] == "control"}
print("base probe (RepE-trained), control threshold, layer 20:")
for d in EVAL5:
    print(f"   {d:38s} auroc={base20[d]['auroc']:>7s}  balacc={base20[d]['balacc']:>7s}")
mp_a = base20["Average (Liars' Bench)"]["auroc"]
mp_b = base20["Average (Liars' Bench)"]["balacc"]
print(f"   {'Average (Liars Bench)':38s} auroc={mp_a:>7s}  balacc={mp_b:>7s}")
print(f"\nH-005 stated 0.636 vs 0.697 AUROC / 0.622 vs 0.721 balacc ->"
      f" measured {ma:.4f} vs {mp_a} / {mb:.4f} vs {mp_b}")
zs20 = {r["dataset"]: r for r in rows(f"{RES}/persona_ood_ext_L20/zero_shot.csv")}
print("H-003 'published persona grouping' on HP-KR (auroc_axis, 8v7 deception axis, L20) =",
      zs20["harm_refusal_knowledge_report__plain"]["auroc_axis"])
print("      which OTHER arms invert (AUROC<0.5) on HP-KR at L20:")
for col in ("auroc_axis", "auroc_pc1", "auroc_syc_axis", "auroc_syc_pc1",
            "auroc_ext_axis", "auroc_ext_pc1", "auroc_denial_axis"):
    v = float(zs20["harm_refusal_knowledge_report__plain"][col])
    if v < 0.5:
        print(f"         {col:20s} {v:.4f}")
print(f"         base probe (Mean Probe)   {base20['harm_refusal_knowledge_report__plain']['auroc']}")
print(f"         follow-up probe           {float(ft['harm_refusal_knowledge_report__plain']['auroc']):.4f}")

# ---------------------------------------------------------------- H-006 / H-007
hdr("H-006 / H-007  zero-shot persona arms, Liars'-Bench average (5 datasets)")
for L in (20, 40):
    zs = {r["dataset"]: r for r in rows(f"{RES}/persona_ood_ext_L{L}/zero_shot.csv")}
    op = rows(f"{RES}/persona_ood_ext_L{L}/zero_shot_operating.csv")
    avg = {r["arm"]: r for r in op
           if r["dataset"] == "Average (Liars' Bench)" and r["threshold_source"] == "control"}
    print(f"-- layer {L} --")
    for col, arm in (("auroc_ext_axis", "extended axis (12v13)"),
                     ("auroc_denial_axis", "knowledge-denial axis (4v6)"),
                     ("auroc_axis", "deception axis (8v7)")):
        m = sum(float(zs[d][col]) for d in EVAL5) / 5
        print(f"   {arm:30s} recomputed mean AUROC over the 5 = {m:.4f}"
              f"   | his file: auroc={avg[arm]['auroc']:>7s} balacc={avg[arm]['balacc']:>7s}")

# ---------------------------------------------------------------- H-008 .. H-013
hdr("H-008  single-source and pooled baselines (loo_L20 / loo_L40)")
for L in (20, 40):
    ss = rows(f"{RES}/loo_L{L}/single_source_reference.csv")
    su = rows(f"{RES}/loo_L{L}/loo_summary.csv")
    m_ss = sum(float(r["single_source_mean"]) for r in ss) / len(ss)
    m_nv = sum(float(r["pooled_loo_naive"]) for r in ss) / len(ss)
    m_bl = sum(float(r["pooled_loo_balanced"]) for r in ss) / len(ss)
    base = {r["weighting"]: r for r in su if r["arm"] == "baseline"}
    ext = {r["weighting"]: r for r in su if r["arm"] == "ext_axis1"}
    print(f"-- layer {L} ({len(ss)} held-out datasets) --")
    print(f"   single-source mean  recomputed {m_ss:.4f}")
    print(f"   pooled naive        recomputed {m_nv:.4f}   loo_summary: {base['naive']['auroc']}")
    print(f"   pooled balanced     recomputed {m_bl:.4f}   loo_summary: {base['balanced']['auroc']}")
    print(f"   ext_axis1           loo_summary: {ext['balanced']['auroc']}"
          f"   excl_hpkr={ext['balanced']['auroc_excl_hpkr']}"
          f"   worst={ext['balanced']['auroc_worst_dataset']}")
    print(f"   baseline naive      excl_hpkr={base['naive']['auroc_excl_hpkr']}"
          f"   worst={base['naive']['auroc_worst_dataset']}")
    print(f"   baseline balanced   excl_hpkr={base['balanced']['auroc_excl_hpkr']}"
          f"   worst={base['balanced']['auroc_worst_dataset']}")

hdr("H-009 / H-010 / H-011 / H-013  derived quantities")
for L in (20, 40):
    su = rows(f"{RES}/loo_L{L}/loo_summary.csv")
    ss = {r["dataset"]: r for r in rows(f"{RES}/loo_L{L}/single_source_reference.csv")}
    base = {r["weighting"]: r for r in su if r["arm"] == "baseline"}
    ext = {r["weighting"]: r for r in su if r["arm"] == "ext_axis1"}
    e = float(ext["balanced"]["auroc"]); ee = float(ext["balanced"]["auroc_excl_hpkr"])
    bn, bb = float(base["naive"]["auroc"]), float(base["balanced"]["auroc"])
    ben, beb = float(base["naive"]["auroc_excl_hpkr"]), float(base["balanced"]["auroc_excl_hpkr"])
    print(f"-- layer {L} --")
    print(f"   ext_axis1 mean {e:.4f} excl-HPKR {ee:.4f}   (advantage {e-max(bn,bb):+.4f} -> {ee-max(ben,beb):+.4f} excl-HPKR)")
    print(f"   baseline excl-HPKR: naive {ben:.4f}, balanced {beb:.4f}"
          f"  -> beats ext_axis1 excl-HPKR? naive={ben>ee}, balanced={beb>ee}")
    print(f"   H-011 balanced-minus-naive mean AUROC = {bb-bn:+.4f}")
    hp = ss["harm_refusal_knowledge_report__plain"]
    print(f"   HP-KR pooled baseline: naive {hp['pooled_loo_naive']}, balanced {hp['pooled_loo_balanced']}"
          f"   (below 0.5 -> inverted? naive={float(hp['pooled_loo_naive'])<0.5},"
          f" balanced={float(hp['pooled_loo_balanced'])<0.5})")
    # H-013 floors
    worst_ext = float(ext["balanced"]["auroc_worst_dataset"])
    others = sorted(((float(r["auroc_worst_dataset"]), r["arm"], r["weighting"])
                     for r in su if r["arm"] != "ext_axis1"), reverse=True)
    print(f"   H-013 floors: ext_axis1 worst={worst_ext:.4f}; pooled worst naive="
          f"{base['naive']['auroc_worst_dataset']}, balanced={base['balanced']['auroc_worst_dataset']}")
    print(f"   runner-up floor over all 10 arm/weighting rows: {others[0][0]:.4f}"
          f" ({others[0][1]}/{others[0][2]}); gap = {worst_ext-others[0][0]:+.4f}")

hdr("H-010 bootstrap: which datasets the pooled (balanced) baseline wins")
for L in (20, 40):
    bs = [r for r in rows(f"{RES}/loo_L{L}/loo_bootstrap.csv") if r["weighting"] == "balanced"]
    print(f"-- layer {L}, balanced --")
    for r in bs:
        print(f"   {r['dataset']:38s} delta={r['delta']:>8s} [{r['ci_low']}, {r['ci_high']}]"
              f"  excludes_zero={r['excludes_zero']}")

hdr("H-012  distance to the privileged upper bound")
for L in (20, 40):
    lb = {r["method"]: r for r in rows(f"{RES}/loo_L{L}/liars_bench_comparison_loo.csv")}
    ub = float(lb["Upper-Bound Probe"]["auroc"])
    ext = float(lb["persona axis, extended (zero-shot)"]["auroc"])
    print(f"-- layer {L} -- upper bound {ub:.4f}, extended axis {ext:.4f}, gap = {ub-ext:.4f}"
          f"  (<=0.03? {ub-ext <= 0.03})")
    print(f"   balacc: upper {lb['Upper-Bound Probe']['balacc_1pct_fpr']},"
          f" ext {lb['persona axis, extended (zero-shot)']['balacc_1pct_fpr']},"
          f" MeanProbe {lb['Mean Probe']['balacc_1pct_fpr']},"
          f" FollowUp {lb['Follow-Up Probe']['balacc_1pct_fpr']}")

# ---------------------------------------------------------------- H-014
hdr("H-014  persona_ood: does projecting onto persona PCs improve cross-dataset transfer?")


def matmean(path):
    rs = list(csv.reader(io.StringIO(gitshow(path))))
    cols = rs[0][1:]
    M, names = {}, []
    for r in rs[1:]:
        names.append(r[0])
        for c, v in zip(cols, r[1:]):
            M[(r[0], c)] = float(v)
    return M, names, cols


Mb, rn, cn = matmean(f"{RES}/persona_ood/matrix_baseline.csv")
Mp, _, _ = matmean(f"{RES}/persona_ood/matrix_pc3.csv")
HP = "harm_refusal_knowledge_report__plain"
off = [(r, c) for r in rn for c in cn if r != c]
off_nohp = [(r, c) for r, c in off if r != HP and c != HP]
sm = {r["features"]: r for r in rows(f"{RES}/persona_ood/summary.csv")}
for name, M in (("baseline", Mb), ("pc3", Mp)):
    o = sum(M[k] for k in off) / len(off)
    oh = sum(M[k] for k in off_nohp) / len(off_nohp)
    dg = sum(M[(d, d)] for d in rn) / len(rn)
    s = sm[name]
    print(f"   {name:9s} off-diag {o:.4f} (his {float(s['off_diagonal_mean']):.4f})"
          f"   excl-HPKR {oh:.4f} (his {float(s['off_diagonal_excl_hpkr']):.4f})"
          f"   diag {dg:.4f} (his {float(s['diagonal_mean']):.4f})   n_off={len(off)}")
d = [Mp[k] - Mb[k] for k in off]
print(f"   pc3 - baseline over the {len(off)} off-diagonal cells:"
      f" mean {sum(d)/len(d):+.4f}, median {statistics.median(d):+.4f},"
      f" improved {sum(1 for x in d if x > 0)}/{len(d)}")
dn = [Mp[k] - Mb[k] for k in off_nohp]
print(f"   excl-HPKR: mean {sum(dn)/len(dn):+.4f}")
r3 = sm["random3"]
print(f"   random-3d control: off-diag {r3['off_diagonal_mean'][:6]}"
      f"  excl-HPKR {r3['off_diagonal_excl_hpkr'][:6]}  diag {r3['diagonal_mean'][:6]}")
print("   layer recorded in his summary.csv:", sm["baseline"]["layer"],
      "; dim of baseline features:", sm["baseline"]["dim"])
print("   n rows in his zero_shot.csv (7 datasets):",
      sum(int(r["n"]) for r in rows(f"{RES}/persona_ood/zero_shot.csv")))

hdr("Published Table 7 (Liars' Bench, arXiv 2511.16035v2), Llama 3.3 70B row")
try:
    import pypdf
    rd = pypdf.PdfReader("papers/2511.16035v2 (1).pdf")
    t = rd.pages[54].extract_text()
    for line in t.splitlines():
        if line.startswith("Llama 3.3 70B") or line.startswith("Table 7") or "ACCURACY AT1%" in line or line.startswith("AUROC"):
            print("   " + line.encode("ascii", "replace").decode())
except Exception as e:  # pragma: no cover
    print("   pypdf unavailable:", e)


# ---------------------------------------------------------------- LEVEL 2 on the loo note
def auroc(scores, labels):
    """Mann-Whitney AUROC with average ranks for ties (no sklearn dependency)."""
    import numpy as np
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(int)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def level2_loo():
    import numpy as np
    import os
    import tempfile
    hdr("LEVEL 2 - loo AUROCs recomputed from the committed raw held-out scores (loo_scores.npz)")
    HP = "harm_refusal_knowledge_report__plain"
    for L in (20, 40):
        blob = subprocess.run(
            ["git", "show", f"{BR}:{RES}/loo_L{L}/loo_scores.npz"],
            capture_output=True, check=True).stdout
        tmp = os.path.join(tempfile.gettempdir(), f"p10_loo_L{L}.npz")
        with open(tmp, "wb") as fh:
            fh.write(blob)
        z = np.load(tmp, allow_pickle=True)
        lab = {k.split("|", 1)[1]: z[k] for k in z.files if k.startswith("labels|")}
        arms = sorted({k.split("|")[0] for k in z.files if not k.startswith("labels|")})
        wts = sorted({k.split("|")[1] for k in z.files if not k.startswith("labels|")})
        book = {r["arm"] + "|" + r["weighting"]: r
                for r in rows(f"{RES}/loo_L{L}/loo_summary.csv")}
        print(f"-- layer {L} --  (n per dataset: "
              + ", ".join(f"{d.split('__')[0][:6]}={len(v)}" for d, v in lab.items()) + ")")
        print(f"   {'arm|weighting':22s} {'mean':>7s} {'his':>7s} {'exclHP':>7s} {'his':>7s} {'worst':>7s} {'his':>7s}")
        for a in arms:
            for w in wts:
                per = {}
                for d in lab:
                    k = f"{a}|{w}|{d}"
                    if k in z.files:
                        per[d] = auroc(z[k], lab[d])
                if len(per) != 6:
                    continue
                m = sum(per.values()) / 6
                mh = sum(v for d, v in per.items() if d != HP) / 5
                wr = min(per.values())
                b = book.get(f"{a}|{w}")
                print(f"   {a + '|' + w:22s} {m:7.4f} {float(b['auroc']):7.4f}"
                      f" {mh:7.4f} {float(b['auroc_excl_hpkr']):7.4f}"
                      f" {wr:7.4f} {float(b['auroc_worst_dataset']):7.4f}")
                if a == "baseline":
                    print(f"      per-dataset: " + ", ".join(
                        f"{d.split('__')[0][:6]}={v:.4f}" for d, v in per.items()))
        z.close()
        try:
            os.remove(tmp)
        except OSError:
            pass


def crosscheck_single_source():
    hdr("CROSS-CHECK  loo single_source_reference == column means of the persona_ood baseline matrix")
    Mb, rn, cn = matmean(f"{RES}/persona_ood/matrix_baseline.csv")
    ss = {r["dataset"]: r for r in rows(f"{RES}/loo_L20/single_source_reference.csv")}
    for c in cn:
        col = [Mb[(r, c)] for r in rn if r != c]
        print(f"   {c:38s} off-diag col mean {sum(col)/len(col):.4f}"
              f"   loo single_source_mean {ss[c]['single_source_mean']}")


crosscheck_single_source()
level2_loo()


def published_tables():
    """Re-derive the three published comparators he transcribes, from the Liars' Bench PDF."""
    hdr("PUBLISHED COMPARATORS  Liars' Bench arXiv 2511.16035v2, Llama-3.3-70B row, 5 shared datasets")
    import pypdf
    rd = pypdf.PdfReader("papers/2511.16035v2 (1).pdf")
    # page order in the file: HP-C HP-KR CG ST ID GS IT AVERAGE; the 5 shared drop ST and GS
    want = {53: ("Table 6 Mean Probe", 0.682, 0.636),
            54: ("Table 7 Follow-Up Probe", 0.634, 0.578),
            55: ("Table 8 Upper-Bound Probe", 0.930, 0.890)}
    for pg, (label, his_a, his_b) in want.items():
        t = rd.pages[pg].extract_text() or ""
        rowsy = [l for l in t.splitlines() if l.startswith("Llama 3.3 70B")]
        # section order differs: T6/T7 = AUROC then BALACC@1%; T8 = BALACC then AUROC
        def nums(line):
            return [float(x) for x in line.replace("Llama 3.3 70B", "").split()
                    if x.replace(".", "", 1).replace("-", "", 1).isdigit()]
        a_line, b_line = (rowsy[0], rowsy[1]) if pg != 55 else (rowsy[1], rowsy[0])
        A, B = nums(a_line), nums(b_line)
        # columns: HP-C HP-KR CG ST ID GS IT AVERAGE -> shared indices 0,1,2,4,6.
        # Table 8 prints '?' for ST and GS, so those columns are simply absent.
        idx = [0, 1, 2, 4, 6] if len(A) == 8 else [0, 1, 2, 3, 4]
        sa = sum(A[i] for i in idx) / 5
        sb = sum(B[i] for i in idx) / 5
        print(f"   {label:28s} 5-shared AUROC {sa:.3f} (his CSV {his_a})"
              f"   balacc {sb:.3f} (his CSV {his_b})")
        print(f"      raw row AUROC  {A}")
        print(f"      raw row balacc {B}")


published_tables()
