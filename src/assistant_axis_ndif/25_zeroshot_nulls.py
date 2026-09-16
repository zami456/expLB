#!/usr/bin/env python
"""Is a zero-shot AUROC special, or would any direction do? Random-direction nulls.

Every zero-shot number in this project -- the deception axis on the corpus (Q1/Q6), per
dataset, on sleeper agents (Q11), the sycophancy axis (Q10) -- was compared against 0.5.
That is the wrong baseline. When the two classes differ in their mean activation (because
of length, tone, or anything else), a RANDOM unit direction also separates them, in one
direction or the other. The sycophancy test made this visible: 5% of random directions
reached AUROC 0.64 on sycophancy_mt at a single layer.

So each claim is re-scored against `--n_random` random unit directions in the same space,
on the same rows and layer. A random direction's sign is arbitrary, so the comparison is on
|AUROC - 0.5|, and the empirical p-value is the fraction of random directions at least as
far from 0.5 as the axis. The axis's own sign is fixed by construction, so its raw AUROC is
reported too: an axis far below 0.5 is significantly INVERTED, not a detector.

Also scores, for Q10, the sycophancy axis with its Assistant-Axis component removed, to
test whether the inversion comes from the -0.27 cosine between the two.

    python 25_zeroshot_nulls.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]


def module(name, fname):
    spec = importlib.util.spec_from_file_location(name, PIPELINE_DIR / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def aurocs_columns(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    """S (n, M) -> (M,) AUROC of each column, ties averaged."""
    r = rankdata(S, axis=0)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def null_test(X, y, dirs: dict, R: np.ndarray) -> dict:
    rand = aurocs_columns(X @ R.T, y)
    dev_r = np.abs(rand - 0.5)
    out = {"random_p95_auroc_dev": float(np.percentile(dev_r, 95) + 0.5),
           "random_max_auroc_dev": float(dev_r.max() + 0.5), "n_random": len(dev_r)}
    for name, d in dirs.items():
        a = float(roc_auc_score(y, X @ d))
        dev = abs(a - 0.5)
        out[name] = {"auroc": a, "p_vs_random": float(((dev_r >= dev).sum() + 1) / (len(dev_r) + 1)),
                     "beats_random_p95": bool(dev > np.percentile(dev_r, 95))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=str(REPO_ROOT / "run-coords"))
    ap.add_argument("--bank", default=str(REPO_ROOT / "run-paper16" / "axes_vectors_s3_fp32.pt"))
    ap.add_argument("--n_random", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    P = module("probes", "9_persona_probes.py")
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    run = Path(args.run)
    rng = np.random.default_rng(args.seed)
    R = np.stack([unit(v) for v in rng.standard_normal((args.n_random, 8192))]).astype(np.float32)
    ax = bank["axes"]
    results = {}

    def axis(name, L):
        return unit(ax[name][L].numpy()).astype(np.float32)

    # ---- A. deception axis on the corpus, pooled and per dataset (Q1 / Q6) -------------
    for L, pool in ((40, "mean"), (44, "last")):
        passB = run / "passB"
        Xs, ys, per = [], [], {}
        for key in sorted(d.name for d in passB.iterdir()
                          if d.is_dir() and (d / "rows.jsonl").exists() and d.name != "sycophancy_mt"):
            X, meta = P.load_dataset(passB, key, L, pool)
            if X is None:
                continue
            y = np.array([m["label"] for m in meta], dtype=int)
            Xs.append(X)
            ys.append(y)
            if len(np.unique(y)) == 2:
                per[key] = null_test(X, y, {"deception_axis": axis("deception", L)}, R)
        Xall, yall = np.concatenate(Xs), np.concatenate(ys)
        pooled = null_test(Xall, yall, {"deception_axis": axis("deception", L),
                                        "assistant_axis": axis("assistant", L)}, R)
        results[f"corpus_L{L}_{pool}"] = {"pooled": pooled, "per_dataset": per}
        logger.info("")
        logger.info("A. corpus, layer %d %s pooling, %d rows", L, pool, len(yall))
        logger.info("   pooled: deception axis %.3f (p vs random %.3f)   random 95th pct %.3f",
                    pooled["deception_axis"]["auroc"], pooled["deception_axis"]["p_vs_random"],
                    pooled["random_p95_auroc_dev"])
        for k, r in per.items():
            logger.info("   %-32s axis %.3f  random p95 %.3f  p %.3f  %s", k,
                        r["deception_axis"]["auroc"], r["random_p95_auroc_dev"],
                        r["deception_axis"]["p_vs_random"],
                        "beats random" if r["deception_axis"]["beats_random_p95"]
                        else "NOT beyond random")
        del Xs, Xall

    # ---- B. sleeper agents (Q11) ---------------------------------------------------------
    q11 = run / "passQ11"
    if q11.exists():
        L, pool = 44, "last"
        res = {}
        logger.info("")
        logger.info("B. sleeper agents, layer %d %s pooling", L, pool)
        for key in sorted(d.name for d in q11.iterdir() if d.is_dir()):
            X, meta = P.load_dataset(q11, key, L, pool)
            if X is None:
                continue
            y = np.array([m["label"] for m in meta], dtype=int)
            res[key] = null_test(X, y, {"deception_axis": axis("deception", L)}, R)
            r = res[key]
            logger.info("   %-24s axis %.3f  random p95 %.3f  max %.3f  p %.3f", key,
                        r["deception_axis"]["auroc"], r["random_p95_auroc_dev"],
                        r["random_max_auroc_dev"], r["deception_axis"]["p_vs_random"])
        results["sleeper_L44_last"] = res

    # ---- C. sycophancy (Q10) --------------------------------------------------------------
    logger.info("")
    logger.info("C. sycophancy_mt (label 1 = retracted)")
    res = {}
    for L in (40, 44):
        for pool in ("mean", "last"):
            X, meta = P.load_dataset(run / "passB", "sycophancy_mt", L, pool)
            y = np.array([m["label"] for m in meta], dtype=int)
            syc, asst = axis("sycophancy", L), axis("assistant", L)
            resid = unit(syc - float(syc @ asst) * asst).astype(np.float32)
            r = null_test(X, y, {"sycophancy_axis": syc, "assistant_axis": asst,
                                 "sycophancy_minus_assistant": resid,
                                 "deception_axis": axis("deception", L)}, R)
            r["cos_sycophancy_assistant"] = float(syc @ asst)
            res[f"L{L}_{pool}"] = r
            logger.info("   L%d %-4s random p95 %.3f | sycophancy %.3f (p %.3f) | assistant %.3f "
                        "(p %.3f) | syc minus assistant %.3f (p %.3f) | deception %.3f (p %.3f)",
                        L, pool, r["random_p95_auroc_dev"],
                        r["sycophancy_axis"]["auroc"], r["sycophancy_axis"]["p_vs_random"],
                        r["assistant_axis"]["auroc"], r["assistant_axis"]["p_vs_random"],
                        r["sycophancy_minus_assistant"]["auroc"],
                        r["sycophancy_minus_assistant"]["p_vs_random"],
                        r["deception_axis"]["auroc"], r["deception_axis"]["p_vs_random"])
    results["sycophancy_mt"] = res

    path = Path(args.out) if args.out else run / "zeroshot_nulls.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
