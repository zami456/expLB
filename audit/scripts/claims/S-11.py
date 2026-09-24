"""S-11: drop the in-distribution RepE column from the collaborator's L20 transfer matrices and
recompute the off-diagonal means for `baseline` and `axis1` (the paper's 8-vs-7 deception contrast).
Everything is read from origin/feat/persona-pc-ood with `git show` -- no checkout."""
import csv, io, subprocess, sys, statistics
REPO = r"C:\Users\SURAIYA\Documents\STUDY\thesis\expLB"
BASE = "src/probes/deception-detection/results"
COMMITS = sys.argv[1:] or ["f386a72", "origin/feat/persona-pc-ood"]

def show(c, p):
    r = subprocess.run(["git", "-C", REPO, "show", f"{c}:{p}"], capture_output=True)
    return None if r.returncode else r.stdout.decode("utf-8", "replace")

def matrix(c, L, feat):
    t = show(c, f"{BASE}/persona_ood_ext_L{L}/matrix_{feat}.csv")
    if t is None: return None, None
    rows = list(csv.reader(io.StringIO(t)))
    hdr = rows[0][1:]
    M = {r[0]: {h: (float(v) if v not in ("", "nan") else None) for h, v in zip(hdr, r[1:])} for r in rows[1:]}
    return hdr, M

def offdiag(hdr, M, drop=()):
    keep = [h for h in hdr if not any(h.startswith(d) for d in drop)]
    rk = [r for r in M if not any(r.startswith(d) for d in drop)]
    vals = [M[r][c] for r in rk for c in keep if r != c and M[r].get(c) is not None]
    return statistics.fmean(vals), len(vals)

for c in COMMITS:
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", c], capture_output=True).stdout.decode().strip()[:8]
    print("=" * 76); print("commit", c, "->", sha)
    for L in (20, 40):
        print(f"  --- L{L} ---")
        for feat in ("baseline", "axis1", "ext_axis1", "pc1", "denial_axis1"):
            hdr, M = matrix(c, L, feat)
            if M is None: print(f"    {feat}: MISSING"); continue
            a, na = offdiag(hdr, M)
            b, nb = offdiag(hdr, M, drop=("repe_honesty",))
            print(f"    {feat:14s} off-diag all   {a:.4f} (n={na})   "
                  f"off-diag minus repe_honesty {b:.4f} (n={nb})")
        # the two numbers S-11 quotes
        hb, Mb = matrix(c, L, "baseline"); ha, Ma = matrix(c, L, "axis1")
        if Mb and Ma:
            b, _ = offdiag(hb, Mb, drop=("repe_honesty",)); a, _ = offdiag(ha, Ma, drop=("repe_honesty",))
            ball, _ = offdiag(hb, Mb); aall, _ = offdiag(ha, Ma)
            print(f"    >>> S-11: axis1 {a:.4f} vs baseline {b:.4f}  delta {a-b:+.4f}"
                  f"   | with repe: axis1 {aall:.4f} vs baseline {ball:.4f} delta {aall-ball:+.4f}")
        print("    datasets (matrix columns):", hb)
