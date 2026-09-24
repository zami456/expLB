"""The collaborator's sycophancy arms: are they transfer results, or one fixed direction?

Reads his matrices straight out of `origin/paper` via `git show` (nothing is checked out).

Two questions:
  1. Which arm has the best off-diagonal EXCLUDING harm-pressure-knowledge-report, per layer?
  2. For a dim=1 arm, does the training dataset change the off-diagonal at all?
     AUROC of a one-dimensional score is invariant to any monotone transform, so a
     logistic regression on one feature can only choose a SIGN. If every training row
     gives the same column values, the "transfer matrix" is a zero-shot reading wearing
     a trained probe's clothes. HP-KR is excluded as a TRAINING row in the second
     spread column, because on these features it flips the sign and would mask the test.
"""
import csv
import io
import subprocess

BR = "origin/paper"
BASE = "src/probes/deception-detection/results"
LAYERS = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79]
HPKR = "harm_refusal_knowledge_report__plain"
ARMS = ("baseline", "pc1", "axis1", "syc_pc1", "syc_axis1", "ext_axis1",
        "denial_axis1", "syc3", "ext_pc3", "denial_pc3")


def show(path):
    r = subprocess.run(["git", "show", f"{BR}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def summary(layer):
    t = show(f"{BASE}/persona_ood_ext_L{layer}/summary.csv")
    return {r["features"]: r for r in csv.DictReader(io.StringIO(t))} if t else None


def matrix(layer, arm):
    t = show(f"{BASE}/persona_ood_ext_L{layer}/matrix_{arm}.csv")
    if not t:
        return None, None
    rows = list(csv.reader(io.StringIO(t)))
    return rows[0][1:], {r[0]: [float(x) for x in r[1:]] for r in rows[1:]}


def spreads(layer, arm):
    """Largest range of any single off-diagonal cell across training rows."""
    cols, m = matrix(layer, arm)
    if not m:
        return None
    sp_all = sp_ex = 0.0
    for ci, c in enumerate(cols):
        a = [m[tr][ci] for tr in m if tr != c]
        e = [m[tr][ci] for tr in m if tr != c and tr != HPKR]
        if a:
            sp_all = max(sp_all, max(a) - min(a))
        if e:
            sp_ex = max(sp_ex, max(e) - min(e))
    return sp_all, sp_ex


def main():
    print("=" * 100)
    print("1. BEST ARM EXCLUDING HP-KR, PER LAYER  (collaborator's branch, origin/paper)")
    print("=" * 100)
    print(f"{'layer':>5s} {'baseline':>9s} {'best arm':>14s} {'value':>8s} {'vs base':>8s} "
          f"{'ext_axis1':>10s} {'syc_pc1':>9s}")
    wins = {}
    for L in LAYERS:
        s = summary(L)
        if not s:
            continue
        base = float(s["baseline"]["off_diagonal_excl_hpkr"])
        cand = {k: float(v["off_diagonal_excl_hpkr"]) for k, v in s.items()
                if k not in ("baseline", "random3")}
        arm = max(cand, key=cand.get)
        wins[arm] = wins.get(arm, 0) + 1
        print(f"{L:5d} {base:9.4f} {arm:>14s} {cand[arm]:8.4f} {cand[arm]-base:+8.4f} "
              f"{cand.get('ext_axis1', float('nan')):10.4f} "
              f"{cand.get('syc_pc1', float('nan')):9.4f}")
    print("\n  arm win counts: " + ", ".join(f"{k}={v}" for k, v in
                                             sorted(wins.items(), key=lambda x: -x[1])))

    print("\n" + "=" * 100)
    print("2. DOES THE TRAINING SET CHANGE ANYTHING?")
    print("   spread = largest range of any one off-diagonal cell across training rows.")
    print("   0.0000 with HP-KR dropped => the probe only ever picked a sign => ZERO-SHOT.")
    print("=" * 100)
    print(f"{'arm':>14s} {'dim':>5s} " + "".join(f"{L:>7d}" for L in LAYERS))
    for arm in ARMS:
        row, dim = "", "?"
        for L in LAYERS:
            s = summary(L)
            sp = spreads(L, arm) if s and arm in s else None
            if s and arm in s:
                dim = s[arm]["dim"]
            row += f"{sp[1]:7.3f}" if sp else f"{'-':>7s}"
        print(f"{arm:>14s} {dim:>5s} {row}")

    print("\n  READING: every dim=1 arm whose row is all 0.000 is a ZERO-SHOT reading of one")
    print("  fixed direction. Training on 8192-d activations and training on one scalar are")
    print("  not the same experiment, so 'this axis beats the baseline' compares a zero-shot")
    print("  direction against a trained probe. Report it as zero-shot, or compare like with like.")


if __name__ == "__main__":
    main()


def per_dataset():
    """Is syc_pc1's win over ext_axis1 spread across datasets, or one column?

    For a dim=1 zero-shot arm the off-diagonal mean over 30 cells has only ~6 distinct
    values (one per TEST column, replicated across training rows), so the honest unit
    of evidence is the test column, n=5 excluding HP-KR.
    """
    print("\n" + "=" * 100)
    print("3. PER-TEST-DATASET, syc_pc1 vs ext_axis1  (zero-shot column readings)")
    print("=" * 100)
    for L in (40, 44, 48, 52, 56):
        cs, ms = matrix(L, "syc_pc1")
        ce, me = matrix(L, "ext_axis1")
        if not ms or not me:
            continue
        print(f"\n--- layer {L} ---")
        print(f"{'test dataset':>38s} {'syc_pc1':>9s} {'ext_axis1':>10s} {'diff':>8s}")
        wins = 0
        n = 0
        for ci, c in enumerate(cs):
            if c == HPKR:
                continue
            # off-diagonal readings, HP-KR dropped as a training row too
            a = [ms[tr][ci] for tr in ms if tr != c and tr != HPKR]
            b = [me[tr][ci] for tr in me if tr != c and tr != HPKR]
            va, vb = sum(a) / len(a), sum(b) / len(b)
            wins += va > vb
            n += 1
            print(f"{c:>38s} {va:9.4f} {vb:10.4f} {va-vb:+8.4f}")
        print(f"{'syc_pc1 wins':>38s} {wins:>4d} of {n}")


if __name__ != "__main__":
    pass
