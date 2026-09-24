"""F-97 - the abstract's 0.904 / 0.916 is `ext_axis1` (12v13, built on OUR kd personas), not the
paper-faithful 8v7 `axis1` (0.791 / 0.832); ~44-46% of the headline margin is the kd bank.

Everything is read from the collaborator's branch with `git show` (no checkout).
Usage: python F-97.py [<commit-ish>]   default: both f386a72 (the commit F-97 cites) and the head.
"""
import csv, io, subprocess, sys, statistics

REPO = r"C:\Users\SURAIYA\Documents\STUDY\thesis\expLB"
BASE = "src/probes/deception-detection/results"


def show(commit, path):
    p = subprocess.run(["git", "-C", REPO, "show", f"{commit}:{path}"], capture_output=True)
    if p.returncode:
        return None
    return p.stdout.decode("utf-8", "replace")


def rows(commit, path):
    txt = show(commit, path)
    return None if txt is None else list(csv.DictReader(io.StringIO(txt)))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


commits = sys.argv[1:] or ["f386a72", "origin/feat/persona-pc-ood"]
for commit in commits:
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", commit], capture_output=True).stdout.decode().strip()
    print("=" * 78)
    print("commit:", commit, "->", sha)
    for L in (20, 40):
        zs = rows(commit, f"{BASE}/persona_ood_ext_L{L}/zero_shot.csv")
        sm = rows(commit, f"{BASE}/persona_ood_ext_L{L}/summary.csv")
        if zs is None:
            print(f"  L{L}: zero_shot.csv NOT PRESENT at this commit"); continue
        keep = [r for r in zs if num(r.get("auroc_axis")) is not None]
        dsets = [r["dataset"] for r in keep]
        ax = [num(r["auroc_axis"]) for r in keep]
        ex = [num(r["auroc_ext_axis"]) for r in keep]
        pc1 = [num(r.get("auroc_pc1")) for r in keep]
        epc1 = [num(r.get("auroc_ext_pc1")) for r in keep]
        print(f"  L{L}: summary.csv (trained off-diagonal means):")
        for r in (sm or []):
            if r["features"] in ("baseline", "axis1", "ext_axis1", "pc1", "ext_pc1", "denial_axis1"):
                print(f"       {r['features']:<14} dim={r['dim']:<5} off_diag={float(r['off_diagonal_mean']):.4f}"
                      f"  off_diag_excl_hpkr={float(r['off_diagonal_excl_hpkr']):.4f}"
                      f"  diag={float(r['diagonal_mean']):.4f}")
        base = next((r for r in (sm or []) if r["features"] == "baseline"), None)
        print(f"  L{L}: zero_shot.csv, n datasets = {len(keep)} -> {dsets}")
        print(f"       axis1      (8v7)  mean zero-shot AUROC = {statistics.fmean(ax):.4f}")
        print(f"       ext_axis1  (12v13) mean zero-shot AUROC = {statistics.fmean(ex):.4f}")
        if all(v is not None for v in pc1):
            print(f"       pc1               mean = {statistics.fmean(pc1):.4f}   ext_pc1 mean = {statistics.fmean(epc1):.4f}")
        if base:
            b = float(base["off_diagonal_mean"])
            ma, me = statistics.fmean(ax) - b, statistics.fmean(ex) - b
            print(f"       margin vs trained baseline {b:.4f}:  axis1 {ma:+.4f}   ext_axis1 {me:+.4f}"
                  f"   kd personas worth {me-ma:+.4f}  share {100*(me-ma)/me:.1f}%")
        # minus repe_honesty
        k2 = [i for i, dn in enumerate(dsets) if not dn.startswith("repe_honesty")]
        print(f"       minus repe_honesty (n={len(k2)}): axis1={statistics.fmean([ax[i] for i in k2]):.4f}"
              f"  ext_axis1={statistics.fmean([ex[i] for i in k2]):.4f}")
        print(f"       per-dataset: " + "  ".join(f"{dn.split('__')[0]}: {a:.3f}->{e:.3f}" for dn, a, e in zip(dsets, ax, ex)))
        if base:
            b2 = float(base["off_diagonal_excl_hpkr"])
            k3 = [i for i, dn in enumerate(dsets) if not dn.startswith("repe_honesty")]
            a2, e2 = statistics.fmean([ax[i] for i in k3]), statistics.fmean([ex[i] for i in k3])
            # F-97's 'drop repe_honesty' uses the 5-dataset zero-shot means vs the same baseline
            print(f"       [drop repe] baseline(off_diag,all6)={float(base['off_diagonal_mean']):.4f}"
                  f" -> axis1 margin {a2-float(base['off_diagonal_mean']):+.4f}"
                  f"  kd worth {e2-a2:+.4f}")

# ---- extra checks: the 'drop repe_honesty' baseline, the 1-d trained==zero-shot identity,
#      and the three COLLABORATOR_BRIEFING errors F-97 names ----
print("\n" + "=" * 78 + "\nEXTRA CHECKS (commit f386a72)\n" + "=" * 78)
C = "f386a72"
for L in (20, 40):
    zs = rows(C, f"{BASE}/persona_ood_ext_L{L}/zero_shot.csv")
    keep = [r for r in zs if num(r.get("auroc_axis")) is not None]
    dsets = [r["dataset"] for r in keep]
    mb = rows(C, f"{BASE}/persona_ood_ext_L{L}/matrix_baseline.csv")
    hdr = [c for c in mb[0] if c != list(mb[0])[0]]
    first = list(mb[0])[0]
    M = {r[first]: r for r in mb}
    def offdiag(names):
        v = [num(M[a][b]) for a in names for b in names if a != b and num(M[a].get(b)) is not None]
        return statistics.fmean(v), len(v)
    names6 = [d for d in dsets if d in M]
    names5 = [d for d in names6 if not d.startswith("repe_honesty")]
    b6, n6 = offdiag(names6); b5, n5 = offdiag(names5)
    ax5 = statistics.fmean([num(r["auroc_axis"]) for r in keep if not r["dataset"].startswith("repe_honesty")])
    ex5 = statistics.fmean([num(r["auroc_ext_axis"]) for r in keep if not r["dataset"].startswith("repe_honesty")])
    print(f"  L{L}: matrix_baseline off-diag over 6 = {b6:.4f} (n={n6}); over 5 (no repe) = {b5:.4f} (n={n5})")
    print(f"       axis1 5-ds zero-shot {ax5:.4f} -> margin vs 5-ds baseline {ax5-b5:+.4f};"
          f"  kd worth {ex5-ax5:+.4f}")
    # 1-d identity
    for feat, zcol in (("axis1", "auroc_axis"), ("ext_axis1", "auroc_ext_axis"), ("denial_axis1", "auroc_denial_axis")):
        sm = rows(C, f"{BASE}/persona_ood_ext_L{L}/summary.csv")
        tr = next((float(r["off_diagonal_mean"]) for r in sm if r["features"] == feat), None)
        zz = statistics.fmean([num(r[zcol]) for r in keep if num(r.get(zcol)) is not None])
        print(f"       {feat:<13} trained_offdiag={tr:.6f}  zero_shot_mean={zz:.6f}  |diff|={abs(tr-zz):.2e}")
