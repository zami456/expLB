#!/usr/bin/env python
"""AUDIT of Results 1 and 2 against an external review. Everything in ONE run, ONE set of splits.

WHY THIS EXISTS. A reviewer asked two questions that the existing outputs cannot answer because
the numbers were produced by two different scripts:

  Q1  "Which budget-matched number is right: 0.850 (30_premise) or 0.806 (30d_why_pooling)?
       That 0.044 spread is larger than the 0.029 gap Result 2 rests on. Do 0.883 and 0.854
       come from the same run, targets and splits?"

  Q4  "Is 0.676 the mean over all ten sources for each target? Do you have each target's best
       single source, and how many of the 110 pairs fall below 0.5? Sign-flipped pairs pull the
       average down, which makes the mean over sources the weakest baseline you could pick."

ON Q1, THE DIAGNOSIS IS ALREADY IN: both scripts subsample identically (`per = budget //
n_sources`, then balance), so 0.850 and 0.806 are the SAME ESTIMATOR ON DIFFERENT RANDOM ROW
DRAWS. They differ because 30d consumes its rng on the source-count curve before reaching the
pooled fit, so the two runs draw different rows. A single draw was reported as if it were the
quantity. This script draws it --draws times and reports the distribution, so the comparison
Result 2 rests on is made against a spread rather than against one lucky sample.

ON Q4, THE POINT IS CONCEDED IN ADVANCE and measured properly. Four baselines, not one:

  mean_over_sources    what we reported. Averages in every anti-transferring source.
  median_over_sources  robust to the inverted pairs, still no target labels needed.
  oriented_mean        max(a, 1-a) per pair: what you would get if you knew each source's SIGN
                       but nothing else. Isolates "pooling fixes sign errors" from
                       "pooling finds a better direction".
  best_source          ORACLE -- needs target labels to pick, so it is not deployable. Reported
                       because it is the honest ceiling on source selection, and if it beats
                       pooled training then "pool everything" is not the best use of the data.

Result 2's margin is recomputed against pooled LODO from THESE splits, full and budget-matched,
with the mean-of-directions probe measured in the same process on the same rows.

    python src/assistant_axis_ndif/30f_result2_audit.py --datasets <11> --layer 40 --draws 20

Writes to --out_root (default run-premise/audit/). CPU only.
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
    ap.add_argument("--out_root", default="run-premise/audit")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--probe", default="paper")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--draws", type=int, default=20,
                    help="independent row subsamples for every budget-matched quantity")
    # Q5: "have you run the Mahadik & Skapars persona method in the same leave-one-out setup?"
    # The answer was no -- every persona number in this project is train-one-test-all, because
    # that is the design the paper uses. With a bank the persona FEATURES go through the exact
    # same LODO harness as the raw probe, which is the only way to compare them fairly.
    ap.add_argument("--bank", default=None, help="axes .pt; adds the persona-coordinate arm")
    ap.add_argument("--bank_features", nargs="+",
                    default=["pc:unified@2", "pc:deception", "axes"])
    args = ap.parse_args()

    P = module("persona_probes", "9_persona_probes.py")
    out = Path(args.out_root); out.mkdir(parents=True, exist_ok=True)

    data = P.build(Path(args.run_dir), list(args.datasets), args.layer, args.pooling)
    DS = [k for k in args.datasets if k in data and len(np.unique(data[k][1])) == 2]
    X32 = {k: np.asarray(data[k][0], dtype=np.float32) for k in DS}
    rng = np.random.default_rng(args.seed)

    # Splits drawn exactly as 30_premise.py and 30d_why_pooling.py draw them, in the same rng
    # order, so `single_offdiag` here must reproduce theirs bit-for-bit. Asserted below: if it
    # does not, this audit is measuring different data than the numbers under review.
    splits = {}
    for k in DS:
        n = len(data[k][1]); perm = rng.permutation(n); cut = int(args.frac * n)
        splits[k] = (perm[:cut], perm[cut:])

    def fit_on(rows_by_ds):
        Xs, ys = [], []
        for k, rows in rows_by_ds.items():
            yk = data[k][1]
            tr = P.balance(rows, yk, args.seed)
            Xs.append(X32[k][tr]); ys.append(yk[tr])
        return P.make_probe(args.probe, args.seed).fit(np.vstack(Xs), np.concatenate(ys)), \
            len(np.concatenate(ys))

    singles, ntr = {}, []
    for s in DS:
        clf, n = fit_on({s: splits[s][0]})
        singles[s] = clf; ntr.append(n)
    budget = int(np.mean(ntr))

    print(f"layer {args.layer}  n={len(DS)}  probe={args.probe}  budget={budget}  "
           f"draws={args.draws}")
    print(f"probe config: {P.make_probe(args.probe, args.seed)}\n")

    # ---- Q4: the 110 pairs and four baselines --------------------------------------------
    pairs, per_target = [], {}
    for t in DS:
        te = splits[t][1]; yt = data[t][1][te]
        if len(np.unique(yt)) < 2:
            continue
        vals = []
        for s in DS:
            if s == t:
                continue
            a = float(P.auroc(singles[s], X32[t][te], yt))
            vals.append(a); pairs.append({"source": s, "target": t, "auroc": a})
        per_target[t] = np.array(vals, dtype=float)

    A = np.array([p["auroc"] for p in pairs], dtype=float)
    base = {
        "mean_over_sources": float(np.mean([v.mean() for v in per_target.values()])),
        "median_over_sources": float(np.mean([np.median(v) for v in per_target.values()])),
        "oriented_mean": float(np.mean([np.maximum(v, 1 - v).mean() for v in per_target.values()])),
        "best_source_oracle": float(np.mean([v.max() for v in per_target.values()])),
    }
    print("Q4  SINGLE-SOURCE BASELINES")
    for k, v in base.items():
        tag = "  <- what we reported" if k == "mean_over_sources" else (
            "  <- ORACLE, needs target labels" if "oracle" in k else "")
        print(f"    {k:<22}{v:.4f}{tag}")
    print(f"    pairs below 0.5: {int((A < 0.5).sum())}/{len(A)}   "
          f"below 0.4: {int((A < 0.4).sum())}   min {A.min():.4f}")

    # ---- Q1: pooled LODO, full and budget-matched over many draws ------------------------
    lodo_full, lodo_match = [], []
    for t in DS:
        te = splits[t][1]; yt = data[t][1][te]
        if len(np.unique(yt)) < 2:
            continue
        others = [s for s in DS if s != t]
        clf, _ = fit_on({s: splits[s][0] for s in others})
        lodo_full.append(float(P.auroc(clf, X32[t][te], yt)))

    per_o = max(2, budget // (len(DS) - 1))
    draws = []
    for _ in range(args.draws):
        vals = []
        for t in DS:
            te = splits[t][1]; yt = data[t][1][te]
            if len(np.unique(yt)) < 2:
                continue
            others = [s for s in DS if s != t]
            rows = {s: rng.choice(splits[s][0], size=min(per_o, len(splits[s][0])),
                                  replace=False) for s in others}
            clf, _ = fit_on(rows)
            vals.append(float(P.auroc(clf, X32[t][te], yt)))
        draws.append(float(np.mean(vals)))
    draws = np.array(draws)

    # ---- Result 2: mean of directions, same splits, same process -------------------------
    U = {s: unit(direction(singles[s])) for s in DS}
    mdir = []
    for t in DS:
        te = splits[t][1]; yt = data[t][1][te]
        if len(np.unique(yt)) < 2:
            continue
        w = np.mean([U[s] for s in DS if s != t], axis=0)
        mdir.append(float(roc_auc_score(yt, X32[t][te].astype(np.float64) @ w)))

    # Per-target vectors, kept so the margin can be bootstrapped over DATASETS -- the project's
    # significance bar (F-04). Without these the audit reports point estimates only, which is
    # the exact failure F-75 was overturned for.
    tg = [t for t in DS if len(np.unique(data[t][1][splits[t][1]])) > 1]
    pt = {"pooled_lodo_full": np.array(lodo_full), "mean_of_directions": np.array(mdir),
          "mean_over_sources": np.array([per_target[t].mean() for t in tg]),
          "median_over_sources": np.array([np.median(per_target[t]) for t in tg]),
          "oriented_mean": np.array([np.maximum(per_target[t], 1 - per_target[t]).mean()
                                     for t in tg]),
          "best_source_oracle": np.array([per_target[t].max() for t in tg])}
    nb = len(tg); bidx = rng.integers(0, nb, size=(10000, nb))
    print("\n    DATASET-LEVEL BOOTSTRAP (paired, B=10,000, resampling targets)")
    boots = {}
    for a, b in [("mean_of_directions", "pooled_lodo_full"),
                 ("pooled_lodo_full", "mean_over_sources"),
                 ("pooled_lodo_full", "median_over_sources"),
                 ("pooled_lodo_full", "oriented_mean"),
                 ("pooled_lodo_full", "best_source_oracle")]:
        d = pt[a] - pt[b]; bs = d[bidx].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        boots[f"{a}-{b}"] = {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi)}
        mark = "  CLEARS" if lo > 0 else ("  (negative)" if hi < 0 else "  DOES NOT CLEAR")
        print(f"      {a} - {b:<22}{d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]{mark}")

    # ---- Q5: the persona method through the SAME leave-one-dataset-out harness ------------
    persona = {}
    if args.bank:
        import torch
        bank = torch.load(args.bank, map_location="cpu", weights_only=False)
        print(f"\nQ5  PERSONA COORDINATES IN THE SAME LODO SETUP  (bank={args.bank})")
        print(f"    {'feature':<18}{'single-src':>12}{'pooled LODO':>13}{'vs raw LODO':>13}")
        for fk in args.bank_features:
            b, kf = fk.partition("@")[0], int(fk.partition("@")[2] or 2)
            try:
                F = {k: P.make_features(X32[k], b, bank, args.layer, kf, args.seed,
                                        meta=data[k][2]) for k in DS}
            except Exception as e:  # noqa: BLE001
                print(f"    {fk:<18} unavailable: {str(e)[:60]}")
                continue

            def ffit(rows_by_ds):
                Xs, ys = [], []
                for k, rows in rows_by_ds.items():
                    yk = data[k][1]; tr = P.balance(rows, yk, args.seed)
                    Xs.append(F[k][tr]); ys.append(yk[tr])
                return P.make_probe(args.probe, args.seed).fit(np.vstack(Xs),
                                                               np.concatenate(ys))
            fsing = {s: ffit({s: splits[s][0]}) for s in DS}
            so, lo_ = [], []
            for t in tg:
                te = splits[t][1]; yt = data[t][1][te]
                so.append(np.mean([float(P.auroc(fsing[s], F[t][te], yt))
                                   for s in DS if s != t]))
                clf = ffit({s: splits[s][0] for s in DS if s != t})
                lo_.append(float(P.auroc(clf, F[t][te], yt)))
            persona[fk] = {"single": float(np.mean(so)), "lodo": float(np.mean(lo_)),
                           "lodo_per_target": lo_}
            d = np.array(lo_) - pt["pooled_lodo_full"]
            print(f"    {fk:<18}{np.mean(so):>12.4f}{np.mean(lo_):>13.4f}{d.mean():>+13.4f}")

    res = {"layer": args.layer, "probe": args.probe, "seed": args.seed, "budget": budget,
           "persona_lodo": persona,
           "draws": args.draws, "datasets": DS, "baselines": base,
           "pairs_below_half": int((A < 0.5).sum()), "n_pairs": len(A),
           "pooled_lodo_full": float(np.mean(lodo_full)),
           "pooled_lodo_match_draws": draws.tolist(),
           "mean_of_directions": float(np.mean(mdir)),
           "per_target": {k: v.tolist() for k, v in pt.items()},
           "targets": tg, "bootstrap": boots, "pairs": pairs}

    print("\nQ1  POOLED LODO, AND RESULT 2 AGAINST IT (same splits, same process)")
    print(f"    pooled_lodo_full        {np.mean(lodo_full):.4f}")
    print(f"    pooled_lodo_match       {draws.mean():.4f}   "
          f"range [{draws.min():.4f}, {draws.max():.4f}]  sd {draws.std():.4f}  "
          f"over {args.draws} row draws")
    print(f"    mean_of_directions      {np.mean(mdir):.4f}")
    print(f"\n    Result 2 margin over pooled_lodo_full   "
          f"{np.mean(mdir) - np.mean(lodo_full):+.4f}")
    print(f"    spread of the matched draw alone        "
          f"{draws.max() - draws.min():.4f}")
    if (np.mean(mdir) - np.mean(lodo_full)) <= (draws.max() - draws.min()):
        print("    => the margin is INSIDE the subsampling spread of its own comparator.")
    print(f"\n    vs mean_over_sources   {np.mean(lodo_full) - base['mean_over_sources']:+.4f}")
    print(f"    vs median_over_sources {np.mean(lodo_full) - base['median_over_sources']:+.4f}")
    print(f"    vs oriented_mean       {np.mean(lodo_full) - base['oriented_mean']:+.4f}")
    print(f"    vs best_source (oracle){np.mean(lodo_full) - base['best_source_oracle']:+.4f}")

    ref = 0.6763163062289969
    got = base["mean_over_sources"]
    print(f"\n    REPRODUCTION CHECK  single-source mean {got:.10f} vs stored {ref:.10f}  "
          f"{'MATCH' if abs(got - ref) < 1e-9 else 'DIFFERS -- audit is on other splits'}")

    (out / f"audit_L{args.layer}_s{args.seed}.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {out / f'audit_L{args.layer}_s{args.seed}.json'}")


if __name__ == "__main__":
    main()
