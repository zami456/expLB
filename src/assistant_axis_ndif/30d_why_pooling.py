#!/usr/bin/env python
"""WHY does pooled training transfer better? Three hypotheses, separated.

F-80/F-81 established that a probe trained on ten datasets and tested on a held-out eleventh
beats a single-source probe by +0.178, that the gain survives at every layer and every seed, and
that it comes from DIVERSITY rather than data volume (row-matched pooling is indistinguishable
from full pooling). What none of that says is *why*.

Three live explanations, which make different predictions:

  H1 DOMAIN-GENERAL DIRECTION. There is a direction most datasets share, and pooling finds it
     while a single source finds its own idiosyncratic one. Predicts: the pooled direction has
     HIGHER average cosine with the single-source directions than they have with each other,
     i.e. it sits near their centroid.

  H2 SHORTCUT CANCELLATION. Each dataset carries its own spurious features; averaging over
     sources cancels them. Predicts: performance rises with the NUMBER of sources and keeps
     rising, roughly as noise averaging would -- and a pooled probe over k sources should
     approach the mean of the single-source DIRECTIONS, not beat it.

  H3 REGULARISATION. Heterogeneous data simply constrains the fit. Predicts: the benefit is
     mostly reproduced by a SIMPLE AVERAGE of single-source directions, with no joint training
     needed at all.

H3 is the cheap sceptical option and the one to rule out first: if averaging the single-source
probe directions works as well as training jointly, "pooling" is doing nothing a mean could not.

WHAT THIS MEASURES
  1. transfer as a function of the number of pooled sources, 1..10, averaged over random subsets
  2. the mean-of-directions probe, which never trains jointly
  3. the geometry: cosine of the pooled direction to each single-source direction, and the
     single-source directions to each other

Writes to --out_root (default run-premise/why/). CPU only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default="run-coords/passB")
    ap.add_argument("--out_root", default="run-premise/why")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--probe", default="paper")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--reps", type=int, default=20, help="random source subsets per size")
    args = ap.parse_args()

    P = module("persona_probes", "9_persona_probes.py")
    out = Path(args.out_root); out.mkdir(parents=True, exist_ok=True)

    data = P.build(Path(args.run_dir), list(args.datasets), args.layer, args.pooling)
    DS = [k for k in args.datasets if k in data and len(np.unique(data[k][1])) == 2]
    X32 = {k: np.asarray(data[k][0], dtype=np.float32) for k in DS}
    rng = np.random.default_rng(args.seed)

    splits = {}
    for k in DS:
        n = len(data[k][1]); perm = rng.permutation(n); cut = int(args.frac * n)
        splits[k] = (perm[:cut], perm[cut:])

    def fit(rows_by_ds, budget=None):
        Xs, ys = [], []
        for k, rows in rows_by_ds.items():
            if budget:
                per = max(2, budget // len(rows_by_ds))
                rows = rng.choice(rows, size=min(per, len(rows)), replace=False)
            yk = data[k][1]
            tr = P.balance(rows, yk, args.seed)
            Xs.append(X32[k][tr]); ys.append(yk[tr])
        return P.make_probe(args.probe, args.seed).fit(np.vstack(Xs), np.concatenate(ys))

    singles = {s: fit({s: splits[s][0]}) for s in DS}
    budget = int(np.mean([len(P.balance(splits[s][0], data[s][1], args.seed)) for s in DS]))
    res = {"layer": args.layer, "datasets": DS, "budget": budget}

    # ---- 1. transfer vs number of sources, at a FIXED row budget ------------------------
    print(f"sources -> held-out AUROC (row budget fixed at {budget})")
    curve = {}
    for n_src in range(1, len(DS)):
        vals = []
        for _ in range(args.reps):
            t = DS[rng.integers(0, len(DS))]
            pool = [s for s in DS if s != t]
            chosen = list(rng.choice(pool, size=min(n_src, len(pool)), replace=False))
            clf = fit({s: splits[s][0] for s in chosen}, budget=budget)
            te = splits[t][1]
            vals.append(P.auroc(clf, X32[t][te], data[t][1][te]))
        curve[n_src] = float(np.nanmean(vals))
        print(f"  {n_src:>2} source(s)   {curve[n_src]:.4f}")
    res["curve_vs_n_sources"] = curve

    # ---- 2. the H3 test: mean of single-source DIRECTIONS, no joint training ------------
    print("\nH3 -- mean of single-source directions, never trained jointly:")
    mean_only = []
    for t in DS:
        others = [s for s in DS if s != t]
        w = np.mean([direction(singles[s]) / np.linalg.norm(direction(singles[s]))
                     for s in others], axis=0)
        te = splits[t][1]
        sc = X32[t][te].astype(np.float64) @ w
        y = data[t][1][te]
        from sklearn.metrics import roc_auc_score
        mean_only.append(float(roc_auc_score(y, sc)) if len(np.unique(y)) > 1 else np.nan)
    res["mean_of_directions"] = float(np.nanmean(mean_only))
    print(f"  {res['mean_of_directions']:.4f}")

    # pooled LODO for comparison, same budget
    pooled = []
    for t in DS:
        clf = fit({s: splits[s][0] for s in DS if s != t}, budget=budget)
        te = splits[t][1]
        pooled.append(P.auroc(clf, X32[t][te], data[t][1][te]))
    res["pooled_lodo_matched"] = float(np.nanmean(pooled))
    single_off = []
    for t in DS:
        te = splits[t][1]
        single_off.append(np.nanmean([P.auroc(singles[s], X32[t][te], data[t][1][te])
                                      for s in DS if s != t]))
    res["single_offdiag"] = float(np.nanmean(single_off))
    print(f"  pooled LODO (same budget)  {res['pooled_lodo_matched']:.4f}")
    print(f"  single-source off-diagonal {res['single_offdiag']:.4f}")

    # ---- 3. geometry: is the pooled direction near the centroid? ------------------------
    U = {s: direction(singles[s]) / np.linalg.norm(direction(singles[s])) for s in DS}
    pair = [float(U[a] @ U[b]) for i, a in enumerate(DS) for b in DS[i + 1:]]
    full = fit({s: splits[s][0] for s in DS}, budget=budget)
    wp = direction(full); wp /= np.linalg.norm(wp)
    to_pooled = [float(wp @ U[s]) for s in DS]
    res["cos_single_to_single_mean"] = float(np.mean(pair))
    res["cos_pooled_to_single_mean"] = float(np.mean(to_pooled))
    print(f"\ngeometry:  single<->single cos {np.mean(pair):+.3f}   "
          f"pooled<->single cos {np.mean(to_pooled):+.3f}")

    (out / f"why_L{args.layer}_s{args.seed}.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8")

    print("\nreading:")
    d = res["mean_of_directions"] - res["single_offdiag"]
    print(f"  mean-of-directions recovers {d:+.4f} of pooling's "
          f"{res['pooled_lodo_matched'] - res['single_offdiag']:+.4f}")
    if res["mean_of_directions"] >= res["pooled_lodo_matched"] - 0.02:
        print("  => H3 NOT ruled out: averaging directions does what joint training does")
    else:
        print("  => joint training beats averaging directions; H3 is not the whole story")


if __name__ == "__main__":
    main()
