#!/usr/bin/env python
"""Does the persona-coordinates PREMISE hold? Two tests, from Ying et al. (arXiv:2602.20273).

THE PREMISE. Mahadik & Skapars motivate persona axes by the claim that probes trained on one
dataset transfer badly, so you need persona structure to generalise. Ying et al. report that
"training on all domains jointly recovers strong performance, confirming that domain-general
directions exist despite poor pairwise transfer" -- i.e. the fix may simply be to train on more
than one dataset, with no personas involved. Every matrix in this project is
train-on-one/test-on-others, so we have never run that cell.

TEST 1 -- POOLED TRAINING, with the control the comparison actually needs.
  single        per-source probes, off-diagonal mean          (what we have always reported)
  pooled_full   one probe on the training rows of ALL datasets
  pooled_match  one probe on the SAME TOTAL ROWS as an average single-source probe, drawn
                across all datasets

  pooled_match is the point. pooled_full sees ~11x more data than a single-source probe, so if
  it wins, "more data" and "more diverse data" are confounded and nothing has been learned about
  domain-generality. pooled_match holds the row budget fixed and varies only diversity. Ying et
  al. do not report this control; without it their result and ours are not comparable.

TEST 2 -- MAHALANOBIS COSINE SIMILARITY as a predictor of transfer.
  They report R^2 = 0.98 for Mahalanobis cosine between probe directions against pairwise OOD
  AUROC, versus R^2 = 0.56 for plain cosine. Recomputed here on our grid. Mahalanobis cosine
  uses the pooled within-class covariance as the metric, i.e. cos_M(u,v) = u'Sv / sqrt(u'Su
  v'Sv) with S = Sigma^-1 -- directions are compared in the space the data actually varies in,
  not in raw coordinates.

A CAVEAT THAT HAS TO TRAVEL WITH TEST 1. Ying et al. pool across five truth types that are all
recognisably "is this statement true" -- a coherent construct. Our eleven datasets span
intent-based deception, belief-based lying and judge-labelled roleplay. Pooled training may work
less well here for reasons that have nothing to do with personas, so a negative result is NOT
evidence that their finding fails to replicate; it is evidence that our corpus is more
heterogeneous than theirs.

Writes everything under --out_root (default run-premise/), touching no existing results.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]


def module(name, fname):
    spec = importlib.util.spec_from_file_location(name, PIPELINE_DIR / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def mahalanobis_cos(u, v, Sinv):
    a = float(u @ Sinv @ v)
    b = float(np.sqrt((u @ Sinv @ u) * (v @ Sinv @ v)))
    return a / b if b > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default="run-coords/passB")
    ap.add_argument("--out_root", default="run-premise")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--probe", default="paper")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--covariance", choices=["total", "within"], default="within",
                    help="metric for Mahalanobis cosine. 'within' is the pooled WITHIN-CLASS "
                         "scatter, which is what Mahalanobis normally means and what Ying et al. "
                         "most likely use; 'total' is the covariance of all rows ignoring "
                         "labels, which is what the first run of this script used and is why "
                         "its R^2 of 0.017 was not a fair test of their 0.98.")
    args = ap.parse_args()

    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    P = module("persona_probes", "9_persona_probes.py")
    CG = module("crossgen", "27_crossgen.py")

    DATASETS = list(args.datasets)
    logger.info("loading %d datasets at layer %d (%s)", len(DATASETS), args.layer, args.pooling)
    # build() already slices to ONE layer and ONE pooling on the way in -- the store is 21
    # layers x 8192 x 2 poolings and loading it whole would be ~10 GB.
    data = P.build(Path(args.run_dir), DATASETS, args.layer, args.pooling)
    DATASETS = [k for k in DATASETS if k in data and len(np.unique(data[k][1])) == 2]
    X32 = {k: np.asarray(data[k][0], dtype=np.float32) for k in DATASETS}

    rng = np.random.default_rng(args.seed)
    splits = {}
    for k in DATASETS:
        n = len(data[k][1])
        perm = rng.permutation(n)
        cut = int(args.frac * n)
        splits[k] = (perm[:cut], perm[cut:])

    def fit_on(rows_by_ds):
        Xs, ys = [], []
        for k, rows in rows_by_ds.items():
            yk = data[k][1]
            tr = P.balance(rows, yk, args.seed)
            Xs.append(X32[k][tr]); ys.append(yk[tr])
        Xc, yc = np.vstack(Xs), np.concatenate(ys)
        return P.make_probe(args.probe, args.seed).fit(Xc, yc), len(yc)

    # ---- single-source probes -----------------------------------------------------------
    singles, n_train = {}, []
    for s in DATASETS:
        clf, n = fit_on({s: splits[s][0]})
        singles[s] = clf; n_train.append(n)
    budget = int(np.mean(n_train))
    logger.info("single-source probes: mean training rows = %d", budget)

    # ---- pooled, full and row-budget-matched --------------------------------------------
    pooled_full, n_full = fit_on({k: splits[k][0] for k in DATASETS})
    per = max(2, budget // len(DATASETS))
    matched_rows = {}
    for k in DATASETS:
        tr = splits[k][0]
        matched_rows[k] = rng.choice(tr, size=min(per, len(tr)), replace=False)
    pooled_match, n_match = fit_on(matched_rows)
    logger.info("pooled_full n=%d   pooled_match n=%d (budget %d)", n_full, n_match, budget)

    # ---- evaluate ------------------------------------------------------------------------
    res = {"layer": args.layer, "pooling": args.pooling, "probe": args.probe,
           "seed": args.seed, "datasets": DATASETS,
           "n_train": {"single_mean": budget, "pooled_full": n_full, "pooled_match": n_match},
           "per_dataset": {}}
    off = {"single": [], "pooled_full": [], "pooled_match": [],
           "pooled_lodo": [], "pooled_lodo_match": []}
    for t in DATASETS:
        te = splits[t][1]
        yt = data[t][1][te]
        if len(set(yt.tolist())) < 2:
            continue
        row = {}
        vals = [P.auroc(singles[s], X32[t][te], yt) for s in DATASETS if s != t]
        row["single_offdiag"] = float(np.mean(vals))
        row["single_own"] = float(P.auroc(singles[t], X32[t][te], yt))
        for tag, clf in (("pooled_full", pooled_full), ("pooled_match", pooled_match)):
            row[tag] = float(P.auroc(clf, X32[t][te], yt))
        # THE ONLY FAIR COMPARISON TO OFF-DIAGONAL TRANSFER. pooled_full and pooled_match both
        # include the TARGET dataset's own training rows, so they are partly in-domain and beat
        # transfer trivially. Leave-one-dataset-out trains on the pooled rows of every OTHER
        # dataset and tests on this one -- exactly what a single-source probe does, but with
        # 10 sources instead of 1. This is the number that decides whether pooling, rather
        # than persona structure, is the answer to poor transfer.
        others = {k: splits[k][0] for k in DATASETS if k != t}
        clf_lodo, n_lodo = fit_on(others)
        row["pooled_lodo"] = float(P.auroc(clf_lodo, X32[t][te], yt))
        row["n_lodo"] = n_lodo
        # ...and the same, held to the single-source row budget, so diversity is the only
        # thing that differs from a single-source probe.
        per_o = max(2, budget // max(1, len(others)))
        om = {k: rng.choice(splits[k][0], size=min(per_o, len(splits[k][0])), replace=False)
              for k in others}
        clf_lodo_m, n_lodo_m = fit_on(om)
        row["pooled_lodo_match"] = float(P.auroc(clf_lodo_m, X32[t][te], yt))
        row["n_lodo_match"] = n_lodo_m
        res["per_dataset"][t] = row
        off["single"].append(row["single_offdiag"])
        off["pooled_full"].append(row["pooled_full"])
        off["pooled_match"].append(row["pooled_match"])
        off["pooled_lodo"].append(row["pooled_lodo"])
        off["pooled_lodo_match"].append(row["pooled_lodo_match"])

    res["summary"] = {k: float(np.mean(v)) for k, v in off.items() if v}
    res["summary"]["single_own_mean"] = float(
        np.mean([r["single_own"] for r in res["per_dataset"].values()]))

    # ---- Mahalanobis similarity vs pairwise transfer -------------------------------------
    # Sigma is 8192x8192. pinv on it costs minutes and ~1.6 GB, and we only ever need
    # Sigma^-1 applied to the 11 probe directions -- so factor once and solve with 11
    # right-hand sides instead. Shrinkage toward a scaled identity keeps it invertible:
    # with ~15k rows in 8192 dimensions the raw sample covariance is rank-deficient.
    if args.covariance == "within":
        # Pooled WITHIN-CLASS scatter: centre each dataset's rows on their own class mean, so
        # between-dataset and between-class offsets do not enter the metric. This is the
        # standard Mahalanobis construction and the one a probe-direction comparison wants.
        parts = []
        for k in DATASETS:
            Xk, yk = X32[k].astype(np.float64), data[k][1]
            for c in (0, 1):
                m = yk == c
                if m.sum() > 1:
                    parts.append(Xk[m] - Xk[m].mean(axis=0, keepdims=True))
        Xall = np.vstack(parts)
        Sigma = (Xall.T @ Xall) / max(1, len(Xall) - len(DATASETS) * 2)
    else:
        Xall = np.vstack([X32[k] for k in DATASETS]).astype(np.float64)
        Xall -= Xall.mean(axis=0, keepdims=True)
        Sigma = (Xall.T @ Xall) / max(1, len(Xall) - 1)
    dim = Sigma.shape[0]
    Sigma = 0.95 * Sigma + 0.05 * (np.trace(Sigma) / dim) * np.eye(dim)
    del Xall
    # No probe_direction helper in 9_persona_probes; every probe it builds exposes either
    # coef_ (logistic / LDA) or .w (difference-of-means wrapper).
    def direction(clf):
        for attr in ("coef_", "w", "coefs_"):
            v = getattr(clf, attr, None)
            if v is not None:
                return np.asarray(np.ravel(v), dtype=np.float32)
        raise AttributeError(f"no direction on {type(clf).__name__}")
    dirs = {s: direction(singles[s]) for s in DATASETS}
    D = np.vstack([dirs[s].astype(np.float64) for s in DATASETS])
    SD = np.linalg.solve(Sigma, D.T).T
    SDm = {s: SD[i] for i, s in enumerate(DATASETS)}
    del Sigma

    def mcos(a, b):
        ua, ub = dirs[a].astype(np.float64), dirs[b].astype(np.float64)
        den = float(np.sqrt((ua @ SDm[a]) * (ub @ SDm[b])))
        return float(ua @ SDm[b]) / den if den > 0 else float("nan")

    pairs = []
    for s in DATASETS:
        for t in DATASETS:
            if s == t:
                continue
            te = splits[t][1]
            yt = data[t][1][te]
            if len(set(yt.tolist())) < 2:
                continue
            pairs.append({
                "source": s, "target": t,
                "auroc": float(P.auroc(singles[s], X32[t][te], yt)),
                "cos": float(dirs[s] @ dirs[t] /
                             (np.linalg.norm(dirs[s]) * np.linalg.norm(dirs[t]))),
                "mahal_cos": mcos(s, t)})
    res["pairs"] = pairs
    for key in ("cos", "mahal_cos"):
        x = np.array([p[key] for p in pairs]); y = np.array([p["auroc"] for p in pairs])
        ok = np.isfinite(x) & np.isfinite(y)
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        res.setdefault("predictors", {})[key] = {"r": float(r), "r2": float(r ** 2),
                                                 "n": int(ok.sum())}

    res["covariance"] = args.covariance
    tag = "" if args.covariance == "within" else f"_{args.covariance}"
    path = out / f"premise_L{args.layer}_{args.pooling}_s{args.seed}{tag}.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")

    print(f"\n=== POOLED TRAINING (layer {args.layer}, {args.pooling}) ===")
    print(f"  single-source, trained on own dataset      {res['summary']['single_own_mean']:.4f}")
    print(f"  single-source, OFF-DIAGONAL (transfer)     {res['summary']['single']:.4f}")
    print(f"  pooled, ALL training rows (n={n_full})      {res['summary']['pooled_full']:.4f}")
    print(f"  pooled, ROW-BUDGET MATCHED (n={n_match})     {res['summary']['pooled_match']:.4f}")
    print(f"    ^ both include the target's own training rows -- partly IN-DOMAIN, not transfer")
    print(f"  pooled LEAVE-ONE-DATASET-OUT                {res['summary']['pooled_lodo']:.4f}"
          f"   <- the fair comparison to 0.676")
    print(f"  pooled LODO, row-budget matched             "
          f"{res['summary']['pooled_lodo_match']:.4f}")
    print(f"\n=== TRANSFER PREDICTORS (Ying et al. report R2=0.98 mahal, 0.56 cos) ===")
    for k, v in res["predictors"].items():
        print(f"  {k:>10}  R2 = {v['r2']:.3f}   (r = {v['r']:+.3f}, n = {v['n']} pairs)")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
