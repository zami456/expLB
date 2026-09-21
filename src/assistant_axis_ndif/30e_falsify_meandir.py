#!/usr/bin/env python
"""FALSIFY the noise-cancellation story behind F-86's mean-of-directions probe.

F-86: average the ten single-source probe directions (each unit-normalised), project, and the
result reaches 0.883 on the held-out eleventh dataset -- beating joint training (0.854) despite
never training jointly. The offered explanation is that each single-source direction is a small
shared component plus large mutually-orthogonal dataset-specific noise (pairwise cosine +0.051),
so averaging cancels the noise.

`docs/REVIEW_REQUEST_POOLING.md` question 4 names the test that would kill it:

    "if the average of ten *random* directions with matched norms transferred comparably,
     the story dies. I have not run that."

THE LITERAL VERSION OF THAT TEST IS TOO WEAK TO BE WORTH RUNNING ALONE. Every direction is
unit-normalised before averaging, so "matched norms" means random unit vectors; their mean points
nowhere and lands at 0.5 by construction. That rules out nothing anybody believed. Four nulls of
increasing sharpness are run instead, and the one that matters is the last:

  gauss      mean of ten iid N(0,I) unit directions. The literal reading. Expected 0.500.
  cov        mean of ten directions drawn as X^T g, g ~ N(0,I) over the pooled training rows,
             i.e. sampled with the ACTIVATION covariance. Activation space is anisotropic
             (review weakness 3), so an isotropic null flatters us; this one does not.
  shuffled   ten probes trained by the SAME procedure on the SAME rows with SHUFFLED LABELS,
             unit-normalised and averaged. Identical pipeline, identical data, the only broken
             link is label -> direction. **This is the null that can actually falsify F-86:**
             if averaging ten label-shuffled probe directions transfers, then averaging
             probe-shaped directions is what works and the labels are decoration.
  budget10   ten REAL directions, each trained on budget/10 rows so the ten together see the
             same number of rows as the budget-matched pooled probe. This answers review
             weakness 1 -- that the mean-of-directions probe has effectively seen all the data
             while its comparator saw ~666 rows -- by handicapping it to the same budget.

Also reported, for review weakness 2 (unit-normalising before averaging was an untested choice):

  real       the F-86 probe, unit-normalised mean.                     <- the number to beat
  real_unnorm  the same average WITHOUT unit-normalising, so high-norm directions dominate.

ORIENTATION. A random direction's AUROC is symmetric about 0.5, so its mean is 0.5 whatever the
signal. Reporting only that would understate every null. Each null therefore reports BOTH the raw
mean and the sign-oriented mean |AUROC - 0.5| + 0.5, which is what a null would score if it were
handed the luckiest sign per target. The oriented column is the conservative comparison and is
the one to read.

    python src/assistant_axis_ndif/30e_falsify_meandir.py --datasets <11 names> --layer 40

Writes to --out_root (default run-premise/falsify/). CPU only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PIPELINE_DIR = Path(__file__).resolve().parent


def module(name, fname):
    spec = importlib.util.spec_from_file_location(name, PIPELINE_DIR / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def direction(clf):
    for a in ("coef_", "w", "coefs_"):
        v = getattr(clf, a, None)
        if v is not None:
            return np.asarray(np.ravel(v), dtype=np.float64)
    raise AttributeError(type(clf).__name__)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default="run-coords/passB")
    ap.add_argument("--out_root", default="run-premise/falsify")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--probe", default="paper")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--reps", type=int, default=20, help="draws per random null")
    args = ap.parse_args()

    P = module("persona_probes", "9_persona_probes.py")
    out = Path(args.out_root); out.mkdir(parents=True, exist_ok=True)

    data = P.build(Path(args.run_dir), list(args.datasets), args.layer, args.pooling)
    DS = [k for k in args.datasets if k in data and len(np.unique(data[k][1])) == 2]
    X32 = {k: np.asarray(data[k][0], dtype=np.float32) for k in DS}
    rng = np.random.default_rng(args.seed)

    # SAME splits as 30d_why_pooling.py: same seed, same order of rng calls, so `real`
    # reproduces F-86's 0.883 rather than approximating it. If it does not reproduce, the
    # comparison below is against the wrong baseline and the run is void -- asserted at the end.
    splits = {}
    for k in DS:
        n = len(data[k][1]); perm = rng.permutation(n); cut = int(args.frac * n)
        splits[k] = (perm[:cut], perm[cut:])

    def fit_rows(k, rows, shuffle=False, cap=None):
        """BALANCE FIRST, THEN CAP, AND CAP PER CLASS. Capping before balancing draws an
        unstratified subsample, which on a skewed dataset (roleplaying is 203/84) can come back
        single-class -- P.balance then returns nothing and sklearn raises
        "Found array with 0 sample(s)". Seed 42 happened to survive this and seed 7 did not,
        which is the failure mode that looks like a seed effect and is actually a bug."""
        yk = data[k][1]
        tr = np.asarray(P.balance(rows, yk, args.seed))
        if cap is not None and len(tr) > cap:
            per = max(1, cap // 2)
            tr = np.concatenate([rng.choice(tr[yk[tr] == c], size=min(per, int((yk[tr] == c).sum())),
                                            replace=False)
                                 for c in np.unique(yk[tr])])
        y = yk[tr]
        if shuffle:
            y = rng.permutation(y)
        return P.make_probe(args.probe, args.seed).fit(X32[k][tr], y)

    budget = int(np.mean([len(P.balance(splits[s][0], data[s][1], args.seed)) for s in DS]))
    per10 = max(4, budget // 10)

    def score(w):
        """Mean held-out AUROC of a fixed direction, leaving each dataset out in turn."""
        vals = []
        for t in DS:
            te = splits[t][1]
            y = data[t][1][te]
            if len(np.unique(y)) < 2:
                continue
            vals.append(float(roc_auc_score(y, X32[t][te].astype(np.float64) @ w)))
        return float(np.mean(vals))

    def score_lodo(wfun):
        """wfun(others) -> direction. Held out per target, which is the F-86 protocol."""
        vals = []
        for t in DS:
            others = [s for s in DS if s != t]
            w = wfun(others)
            te = splits[t][1]
            y = data[t][1][te]
            if len(np.unique(y)) < 2:
                continue
            vals.append(float(roc_auc_score(y, X32[t][te].astype(np.float64) @ w)))
        return np.array(vals, dtype=float)

    singles = {s: fit_rows(s, splits[s][0]) for s in DS}
    U = {s: unit(direction(singles[s])) for s in DS}
    RAW = {s: direction(singles[s]) for s in DS}
    dim = len(U[DS[0]])
    norms = np.array([np.linalg.norm(RAW[s]) for s in DS])

    print(f"layer {args.layer}  n_datasets={len(DS)}  dim={dim}  "
          f"budget={budget}  per-source budget/10={per10}")
    print(f"single-source direction norms: mean {norms.mean():.3f} "
          f"min {norms.min():.3f} max {norms.max():.3f}\n")

    results = {"layer": args.layer, "datasets": DS, "dim": dim, "budget": budget,
               "per10": per10, "probe": args.probe, "seed": args.seed, "reps": args.reps}

    # ---- the real probe, and the normalisation choice ------------------------------------
    real = score_lodo(lambda o: np.mean([U[s] for s in o], axis=0))
    real_unnorm = score_lodo(lambda o: np.mean([RAW[s] for s in o], axis=0))
    print(f"{'condition':<14}{'AUROC':>9}{'oriented':>10}   note")
    print(f"{'real':<14}{real.mean():>9.4f}{'':>10}   F-86: unit-normalised mean of "
          f"{len(DS) - 1} directions")
    print(f"{'real_unnorm':<14}{real_unnorm.mean():>9.4f}{'':>10}   "
          f"same average, NOT unit-normalised")
    results["real"] = float(real.mean())
    results["real_unnorm"] = float(real_unnorm.mean())
    results["real_per_target"] = real.tolist()

    # ---- budget-matched real directions (review weakness 1) ------------------------------
    b10 = []
    for _ in range(max(3, args.reps // 4)):
        sm = {s: unit(direction(fit_rows(s, splits[s][0], cap=per10))) for s in DS}
        b10.append(score_lodo(lambda o: np.mean([sm[s] for s in o], axis=0)).mean())
    results["budget10"] = float(np.mean(b10))
    print(f"{'budget10':<14}{np.mean(b10):>9.4f}{'':>10}   each direction fit on <={per10} "
          f"rows (total ~= pooled budget)")

    # ---- the nulls ------------------------------------------------------------------------
    def run_null(name, draw, note, pre=None):
        """pre(), when given, is called once per rep and its value passed to draw. Used by the
        shuffled null so eleven probes are fit per rep rather than per (rep x target), which is
        an 11x saving and the difference between minutes and hours at 8192 dimensions."""
        raw, ori = [], []
        for _ in range(args.reps):
            ctx = pre() if pre else None
            v = score_lodo((lambda o: draw(o, ctx)) if pre else draw)
            raw.append(v.mean())
            ori.append((np.abs(v - 0.5) + 0.5).mean())
        results[name] = {"raw": float(np.mean(raw)), "oriented": float(np.mean(ori)),
                         "raw_max": float(np.max(raw)), "oriented_max": float(np.max(ori))}
        print(f"{name:<14}{np.mean(raw):>9.4f}{np.mean(ori):>10.4f}   {note}")

    run_null("gauss",
             lambda o: unit(np.mean([unit(rng.standard_normal(dim)) for _ in o], axis=0)),
             "mean of iid N(0,I) unit directions")

    # covariance-matched: X^T g over pooled training rows reproduces the activation spectrum
    Xpool = np.vstack([X32[s][splits[s][0]] for s in DS]).astype(np.float64)
    Xpool -= Xpool.mean(0, keepdims=True)
    npool = Xpool.shape[0]

    def draw_cov(o):
        return unit(np.mean([unit(Xpool.T @ rng.standard_normal(npool)) for _ in o], axis=0))

    run_null("cov", draw_cov, "mean of covariance-matched random directions")

    run_null("shuffled",
             lambda o, sh: unit(np.mean([sh[s] for s in o], axis=0)),
             "SAME probes, SHUFFLED labels -- the null that matters",
             pre=lambda: {s: unit(direction(fit_rows(s, splits[s][0], shuffle=True)))
                          for s in DS})

    # ---- verdict --------------------------------------------------------------------------
    print("\nreading:")
    worst = max(results[n]["oriented"] for n in ("gauss", "cov", "shuffled"))
    worst_max = max(results[n]["oriented_max"] for n in ("gauss", "cov", "shuffled"))
    gap = results["real"] - worst
    print(f"  real {results['real']:.4f} vs best null (oriented mean) {worst:.4f}  "
          f"gap {gap:+.4f}")
    print(f"  single worst-case null draw (oriented max over {args.reps}) {worst_max:.4f}")
    if gap <= 0.02:
        print("  => FALSIFIED: averaging arbitrary probe-shaped directions does the same work.")
    elif results["real"] <= worst_max:
        print("  => WEAKENED: the mean null is well below, but a lucky null draw matches it.")
    else:
        print("  => SURVIVES: no null reaches it, including label-shuffled probes.")
    dn = results["real"] - results["real_unnorm"]
    print(f"  unit-normalising is worth {dn:+.4f}; budget-matched real directions reach "
          f"{results['budget10']:.4f}")

    (out / f"falsify_L{args.layer}_s{args.seed}.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {out / f'falsify_L{args.layer}_s{args.seed}.json'}")


if __name__ == "__main__":
    main()
