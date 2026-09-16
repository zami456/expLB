#!/usr/bin/env python
"""Are the headline differences real? Bootstrap CIs, paired tests and BH correction.

Every Q2 / Q6 / Q8 claim so far is a difference of mean AUROCs with no uncertainty attached.
This script re-derives the transfer matrices exactly as `11_layer_sweep.py` built them
(same splits, seeds, probe, features, datasets) and checks that the point estimates
reproduce the sweep files before testing anything -- a significance test on numbers that do
not match the reported ones would be testing something else.

**Two kinds of uncertainty, reported separately, because they answer different questions.**

  row bootstrap      resample TEST ROWS within each target dataset (stratified by label,
                     the same resample for every method -- paired). "Would a different draw
                     of test rows from these datasets change the answer?" Probes stay fixed,
                     so training noise is not included.
  dataset bootstrap  resample the 9 labelled DATASETS with replacement and re-average the
                     off-diagonal pairs among the drawn ones (paired across methods). "Would
                     a different collection of deception datasets change the answer?" This
                     is usually the wider interval and the more honest one for a claim about
                     generalisation.

A difference is called significant only if its **dataset-bootstrap** p-value survives
Benjamini-Hochberg at q = 0.05 across the whole family. Layer-agreement claims (20 vs 22) are
tested as EQUIVALENCE, not as "not significant": the 90% CI of the difference must lie inside
+/- `--margin` (two one-sided tests). Absence of a significant difference is not evidence of
no difference, and the Q6 claim is the stronger one.

**Selection caveat, stated here once.** Comparisons of each method "at its own best layer"
pick the maximum of 21 layers after looking, which inflates both sides. The same-layer
comparisons in the family carry no such selection and should be preferred when they agree.

Q8's trained comparison is tested from the stored per-fold AUROCs (9 held-out datasets,
paired fold bootstrap); per-row scores were not kept for that run.

    python 23_significance.py --run_dir run-coords/passB
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]

# (name, question, A, B, kind). A and B are (pooling, feature, layer). kind "diff" asks
# whether A - B differs from 0; kind "equiv" asks whether |A - B| < margin.
COMPARISONS = [
    ("axes@44 vs raw@32, each at its best (mean)", "Q2",
     ("mean", "axes", 44), ("mean", "raw", 32), "diff"),
    ("axes vs raw, same layer 40 (mean)", "Q2", ("mean", "axes", 40), ("mean", "raw", 40), "diff"),
    ("axes vs raw, same layer 44 (mean)", "Q2", ("mean", "axes", 44), ("mean", "raw", 44), "diff"),
    ("axes@44 vs raw@28, each at its best (last)", "Q2",
     ("last", "axes", 44), ("last", "raw", 28), "diff"),
    ("axes vs raw, same layer 44 (last)", "Q2", ("last", "axes", 44), ("last", "raw", 44), "diff"),
    ("pc:deception@40 vs raw@32 (mean)", "Q2",
     ("mean", "pc:deception", 40), ("mean", "raw", 32), "diff"),
    ("axes vs raw, same layer 20 (mean)", "Q2", ("mean", "axes", 20), ("mean", "raw", 20), "diff"),
    ("raw: layer 20 vs 22 (last)", "Q6", ("last", "raw", 20), ("last", "raw", 22), "equiv"),
    ("raw: layer 20 vs 22 (mean)", "Q6", ("mean", "raw", 20), ("mean", "raw", 22), "equiv"),
    ("axes: layer 20 vs 22 (mean)", "Q6", ("mean", "axes", 20), ("mean", "axes", 22), "equiv"),
    ("axes: layer 20 vs 22 (last)", "Q6", ("last", "axes", 20), ("last", "axes", 22), "equiv"),
    ("axes: layer 40 vs 22 (mean)", "Q6", ("mean", "axes", 40), ("mean", "axes", 22), "diff"),
    ("axes: layer 40 vs 22 (last)", "Q6", ("last", "axes", 40), ("last", "axes", 22), "diff"),
    ("raw: layer 40 vs 22 (mean)", "Q6", ("mean", "raw", 40), ("mean", "raw", 22), "diff"),
    ("raw: layer 40 vs 22 (last)", "Q6", ("last", "raw", 40), ("last", "raw", 22), "diff"),
]
SWEEP_FILES = {"mean": "layer_sweep_passB_mms.json", "last": "layer_sweep_passB_last.json"}


def module(name, fname):
    spec = importlib.util.spec_from_file_location(name, PIPELINE_DIR / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def auroc_rows(xb: np.ndarray, n_pos: int) -> np.ndarray:
    """xb (B, n): positives in the first n_pos columns. -> (B,) AUROC, ties averaged."""
    r = rankdata(xb, axis=1)
    n_neg = xb.shape[1] - n_pos
    return (r[:, :n_pos].sum(axis=1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def two_sided_p(d: np.ndarray) -> float:
    """Bootstrap p for H0: difference = 0 (percentile method, floored at 1/B)."""
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(max(min(p, 1.0), 1.0 / len(d)))


def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=str(REPO_ROOT / "run-coords" / "passB"))
    ap.add_argument("--bank", default=str(REPO_ROOT / "run-paper16" / "axes_vectors_s3_fp32.pt"))
    ap.add_argument("--exclude", nargs="+", default=["sycophancy_mt"],
                    help="datasets absent when the Q2 sweep ran; excluded so the point "
                         "estimates reproduce the sweep files")
    ap.add_argument("--boot_rows", type=int, default=1000)
    ap.add_argument("--boot_datasets", type=int, default=10000)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    P = module("probes", "9_persona_probes.py")
    S = module("sweep", "11_layer_sweep.py")
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    run_dir = Path(args.run_dir)
    keys = sorted(d.name for d in run_dir.iterdir()
                  if d.is_dir() and (d / "rows.jsonl").exists() and d.name not in args.exclude)
    need = sorted({c[2] for c in COMPARISONS} | {c[3] for c in COMPARISONS})
    rng = np.random.default_rng(args.seed)

    matrices = {}      # (pool, feat, layer) -> {"point": {(s,t): auc}, "boot": (B,) mean}
    datasets = None
    resample = {}      # target -> (Rp, Rn) index matrices, shared by every method: paired
    for pool in ("mean", "last"):
        todo = [m for m in need if m[0] == pool]
        if not todo:
            continue
        logger.info("loading pooling=%s for %d matrices", pool, len(todo))
        data = {}
        for k in keys:
            X, meta, layers = S.load_all_layers(run_dir, k, pool)
            if X is None:
                continue
            data[k] = (X, np.array([m["label"] for m in meta], dtype=int), meta, layers)
        src = [k for k, (_, y, _, _) in data.items()
               if k != P.CONTROL and len(np.unique(y)) == 2]
        datasets = src
        tests = {}
        for t, (_, yt, _, _) in data.items():
            te = P.balance(np.arange(len(yt)), yt, args.seed + 1)   # [pos..., neg...]
            if len(te) < 8:
                continue
            n = len(te) // 2
            tests[t] = te
            if t not in resample:
                resample[t] = (rng.integers(0, n, size=(args.boot_rows, n)),
                               rng.integers(0, n, size=(args.boot_rows, n)) + n)
        for (_, feat, L) in todo:
            point, boot_sum, n_pairs = {}, np.zeros(args.boot_rows), 0
            # features depend on the dataset and layer, never on the source a probe came
            # from, so build each once instead of once per source
            feats = {}
            for k, (X16, _, _, layers) in data.items():
                feats[k] = P.make_features(X16[:, layers.index(L), :].to(torch.float32).numpy(),
                                           feat, bank, L, 5, args.seed)
            for s in src:
                _, ys, ms, _ = data[s]
                tr, _ = P.group_split(ms, 0.75, args.seed)
                tr = P.balance(tr, ys, args.seed)
                if len(tr) < 8:
                    continue
                F = feats[s]
                clf = P.make_probe("mms", args.seed).fit(F[tr], ys[tr])
                for t, te in tests.items():
                    if t == s:
                        continue
                    x = P._scores(clf, feats[t][te])
                    n = len(te) // 2
                    point[(s, t)] = float(auroc_rows(x[None, :], n)[0])
                    Rp, Rn = resample[t]
                    boot_sum += auroc_rows(np.concatenate([x[Rp], x[Rn]], axis=1), n)
                    n_pairs += 1
            matrices[(pool, feat, L)] = {"point": point, "boot": boot_sum / n_pairs,
                                         "mean": float(np.mean(list(point.values())))}
            logger.info("  %-5s %-13s L%-2d mean off-diag %.3f over %d pairs   %.0fs",
                        pool, feat, L, matrices[(pool, feat, L)]["mean"], n_pairs,
                        time.time() - t0)
        del data

    # ---- reproduce the sweep files before testing anything -------------------------
    logger.info("")
    logger.info("reproduction check against the sweep files:")
    worst = 0.0
    for (pool, feat, L), m in matrices.items():
        sw = json.loads((run_dir.parent / SWEEP_FILES[pool]).read_text(encoding="utf-8"))
        ref = next(r[feat] for r in sw["sweep"] if r["layer"] == L)
        dev = abs(ref - m["mean"])
        worst = max(worst, dev)
        logger.info("  %-5s %-13s L%-2d recomputed %.4f  sweep %.4f  |diff| %.4f",
                    pool, feat, L, m["mean"], ref, dev)
    if worst > 0.002:
        raise SystemExit(f"recomputed means differ from the sweep by up to {worst:.4f} -- "
                         "not testing numbers that do not match the reported ones")

    # ---- dataset bootstrap: resample the labelled datasets, paired ------------------
    idx = {k: i for i, k in enumerate(datasets)}
    D = len(datasets)
    counts = rng.multinomial(D, np.full(D, 1.0 / D), size=args.boot_datasets)   # (B, D)
    W = counts[:, :, None] * counts[:, None, :]
    W[:, np.arange(D), np.arange(D)] = 0

    def dataset_boot(m):
        A = np.full((D, D), np.nan)
        for (s, t), v in m["point"].items():
            if s in idx and t in idx:
                A[idx[s], idx[t]] = v
        mask = ~np.isnan(A)
        Aw = np.where(mask, A, 0.0)
        num = (W * Aw[None]).sum(axis=(1, 2))
        den = (W * mask[None]).sum(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den

    dboot = {k: dataset_boot(m) for k, m in matrices.items()}

    # ---- per-method intervals ------------------------------------------------------
    logger.info("")
    logger.info("mean off-diagonal AUROC with 95%% intervals")
    per_method = []
    for k, m in sorted(matrices.items()):
        rb, db = m["boot"], dboot[k][~np.isnan(dboot[k])]
        per_method.append({"pooling": k[0], "feature": k[1], "layer": k[2], "mean": m["mean"],
                           "row_ci95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
                           "dataset_ci95": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]})
        logger.info("  %-5s %-13s L%-2d %.3f   rows [%.3f, %.3f]   datasets [%.3f, %.3f]",
                    k[0], k[1], k[2], m["mean"], *per_method[-1]["row_ci95"],
                    *per_method[-1]["dataset_ci95"])

    # ---- comparisons ---------------------------------------------------------------
    results = []
    for name, q, A, B, kind in COMPARISONS:
        d_point = matrices[A]["mean"] - matrices[B]["mean"]
        d_rows = matrices[A]["boot"] - matrices[B]["boot"]
        d_ds = dboot[A] - dboot[B]
        d_ds = d_ds[~np.isnan(d_ds)]
        rec = {"name": name, "question": q, "kind": kind, "A": list(A), "B": list(B),
               "diff": d_point,
               "row_ci95": [float(np.percentile(d_rows, 2.5)), float(np.percentile(d_rows, 97.5))],
               "dataset_ci95": [float(np.percentile(d_ds, 2.5)), float(np.percentile(d_ds, 97.5))],
               "row_p": two_sided_p(d_rows), "dataset_p": two_sided_p(d_ds)}
        if kind == "equiv":
            lo, hi = np.percentile(d_ds, 5), np.percentile(d_ds, 95)
            rec["dataset_ci90"] = [float(lo), float(hi)]
            rec["equivalent_within_margin"] = bool(lo > -args.margin and hi < args.margin)
            lo_r, hi_r = np.percentile(d_rows, 5), np.percentile(d_rows, 95)
            rec["row_ci90"] = [float(lo_r), float(hi_r)]
            rec["row_equivalent_within_margin"] = bool(lo_r > -args.margin and hi_r < args.margin)
        results.append(rec)

    # ---- Q8, fold-level -------------------------------------------------------------
    q8p = run_dir.parent / "q8_trained_pertoken.json"
    if q8p.exists():
        q8 = json.loads(q8p.read_text(encoding="utf-8"))
        by = {(r["probe"], r["aggregation"], r["layer"]): {f["held_out"]: f["auroc"]
                                                          for f in r["per_fold"]}
              for r in q8["results"]}
        folds = q8["held_out"]
        fb = rng.integers(0, len(folds), size=(args.boot_datasets, len(folds)))
        for L in (20, 28, 40, 44):
            for tok in (("tok_lr", "mean"), ("tok_lr", "top10pct")):
                a = np.array([by[("pool_mean_mms", "pooled", L)][f] for f in folds])
                b = np.array([by[(tok[0], tok[1], L)][f] for f in folds])
                d = (a - b)[fb].mean(axis=1)
                results.append({"name": f"pooled-mean mms vs {tok[0]}/{tok[1]}, layer {L}",
                                "question": "Q8", "kind": "diff",
                                "diff": float((a - b).mean()),
                                "dataset_ci95": [float(np.percentile(d, 2.5)),
                                                 float(np.percentile(d, 97.5))],
                                "dataset_p": two_sided_p(d), "row_p": None,
                                "folds_pooled_wins": int((a > b).sum()), "folds": len(folds)})

    # ---- BH over the family, on the dataset-level p ----------------------------------
    diffs = [r for r in results if r["kind"] == "diff"]
    q = bh([r["dataset_p"] for r in diffs])
    for r, qq in zip(diffs, q):
        r["dataset_q_bh"] = float(qq)
        r["significant"] = bool(qq < 0.05)

    logger.info("")
    logger.info("=" * 110)
    logger.info("%-4s %-46s %7s  %-19s %-19s %7s  %s", "Q", "comparison", "diff",
                "rows 95% CI", "datasets 95% CI", "q(BH)", "verdict")
    for r in results:
        rci = (f"[{r['row_ci95'][0]:+.3f},{r['row_ci95'][1]:+.3f}]" if "row_ci95" in r
               else f"{'(fold-level)':>19}")
        dci = f"[{r['dataset_ci95'][0]:+.3f},{r['dataset_ci95'][1]:+.3f}]"
        if r["kind"] == "equiv":
            verdict = ("EQUIVALENT within ±%.2f" % args.margin if r["equivalent_within_margin"]
                       else "not shown equivalent (datasets); rows: %s"
                       % ("equivalent" if r["row_equivalent_within_margin"] else "not"))
            qs = "   --"
        else:
            verdict = "significant" if r["significant"] else "NOT significant"
            if "folds_pooled_wins" in r:
                verdict += f"  (pooled wins {r['folds_pooled_wins']}/{r['folds']} folds)"
            qs = f"{r['dataset_q_bh']:.3f}"
        logger.info("%-4s %-46s %+7.3f  %-19s %-19s %7s  %s", r["question"], r["name"][:46],
                    r["diff"], rci, dci, qs, verdict)

    path = Path(args.out) if args.out else run_dir.parent / "significance.json"
    path.write_text(json.dumps({"boot_rows": args.boot_rows,
                                "boot_datasets": args.boot_datasets, "margin": args.margin,
                                "datasets": datasets, "reproduction_worst_diff": worst,
                                "per_method": per_method, "comparisons": results},
                               indent=2), encoding="utf-8")
    logger.info("wrote %s   %.0fs", path, time.time() - t0)


if __name__ == "__main__":
    main()
