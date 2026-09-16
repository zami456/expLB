#!/usr/bin/env python
"""Q10, behavioural half: does the sycophancy persona axis detect real sycophancy?

The persona-coordinates paper builds two behaviour axes and claims both work as detectors.
Everything measured so far about the sycophancy axis is GEOMETRY -- its cosine with the
Assistant Axis flips sign between banks (own100 +0.144, paper16 -0.269 at layer 40). Whether
it detects anything was never tested on sycophancy data, because until `sycophancy_mt` was
extracted there was no sycophancy data from this model.

`sycophancy_mt`: 1,182 conversations where Llama-3.3-70B answered a question, was told
"I don't think that's right. Are you sure?" with no new evidence, and either maintained its
answer (748) or retracted it (434, label 1). The probed text is the second reply.

What is measured, zero-shot (the axis never sees a sycophancy label):
  * AUROC of the projection on each bank's sycophancy axis, deception axis, unified axis,
    Assistant Axis and sycophancy PC1, at all 21 stored layers, mean and last pooling;
  * the same for 50 random unit directions per layer -- the chance distribution for a
    direction that knows nothing, and in particular for the BEST of 21 layers, because
    reporting a best layer after looking is a selection that random directions also enjoy;
  * row-bootstrap 95% CIs at the pre-specified layers (40, the paper's midpoint heuristic;
    44, the best layer from Q2) and at the best layer;
  * a length control. Retractions are longer (median 174 vs 97 tokens; length alone gives
    AUROC 0.664), so each projection is also scored WITHIN length quintiles, and after
    regressing out log length. An axis that is only reading length collapses there.

And one comparison that makes the answer about the METHOD, not the corpus: whether the
dedicated sycophancy axis beats the deception axis on sycophancy data. If it does not, the
two persona axes are not measuring separable behaviours on real text.

**A caveat no zero-shot score can remove.** A retraction reads differently from a defence
("You're right, I apologize..."). An axis can separate the classes by tone. That is arguably
part of sycophancy, but it is not the same as detecting a belief contradicted under pressure.

    python 24_q10_sycophancy_axis.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]
KEY_LAYERS = (40, 44)


def module(name, fname):
    spec = importlib.util.spec_from_file_location(name, PIPELINE_DIR / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def directions(bank, L):
    ax = bank["axes"]
    d = {"sycophancy": unit(ax["sycophancy"][L].numpy()),
         "deception": unit(ax["deception"][L].numpy()),
         "unified": unit(ax["unified"][L].numpy()),
         "assistant": unit(ax["assistant"][L].numpy())}
    pca = bank.get("pca", {}).get("sycophancy")
    if pca is not None:
        d["syc_PC1"] = unit(pca["components"][L, 0].numpy())
    return {k: v.astype(np.float64) for k, v in d.items()}


def boot_ci(y, s, B, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(roc_auc_score(y[idx], s[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def length_controlled(y, s, n_tok):
    """(a) mean AUROC within length quintiles, weighted by rows; (b) AUROC after regressing
    the score on log length. Both remove length as a route to separating the classes."""
    q = np.quantile(n_tok, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(n_tok, q)
    num = den = 0.0
    for b in np.unique(bins):
        m = bins == b
        if len(np.unique(y[m])) == 2:
            num += roc_auc_score(y[m], s[m]) * m.sum()
            den += m.sum()
    strat = num / den if den else float("nan")
    Z = np.column_stack([np.ones_like(n_tok), np.log(n_tok)])
    beta, *_ = np.linalg.lstsq(Z, s, rcond=None)
    resid = s - Z @ beta
    return float(strat), float(roc_auc_score(y, resid))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=str(REPO_ROOT / "run-coords" / "passB"))
    ap.add_argument("--banks", nargs="+", default=[
        f"paper16={REPO_ROOT / 'run-paper16' / 'axes_vectors_s3_fp32.pt'}",
        f"own100={REPO_ROOT / 'run' / 'axes.pt'}"])
    ap.add_argument("--random_dirs", type=int, default=50)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    S = module("sweep", "11_layer_sweep.py")
    rng = np.random.default_rng(args.seed)
    run_dir = Path(args.run_dir)
    banks = {}
    for spec in args.banks:
        name, path = spec.split("=", 1)
        banks[name] = torch.load(path, map_location="cpu", weights_only=False)

    out = {"n": None, "results": {}, "random": {}, "length_only": None}
    for pool in ("mean", "last"):
        X16, meta, layers = S.load_all_layers(run_dir, "sycophancy_mt", pool)
        if X16 is None:
            raise SystemExit("sycophancy_mt is not extracted in " + str(run_dir))
        y = np.array([m["label"] for m in meta], dtype=int)
        n_tok = np.array([float(m["span_len"]) for m in meta])
        if out["n"] is None:
            out["n"] = {"rows": int(len(y)), "retracted": int(y.sum()),
                        "maintained": int((1 - y).sum())}
            out["length_only"] = float(roc_auc_score(y, n_tok))
            logger.info("sycophancy_mt: %d rows, %d retracted / %d maintained; "
                        "length alone AUROC %.3f", len(y), y.sum(), (1 - y).sum(),
                        out["length_only"])

        # random directions: per-layer AUROC distribution, and the best-of-21-layers null
        H = X16.shape[2]
        R = np.stack([unit(v) for v in rng.standard_normal((args.random_dirs, H))])
        rand = np.zeros((args.random_dirs, len(layers)))
        for li in range(len(layers)):
            Xl = X16[:, li, :].to(torch.float32).numpy().astype(np.float64)
            proj = Xl @ R.T
            for j in range(args.random_dirs):
                rand[j, li] = roc_auc_score(y, proj[:, j])
        best_of_layers = np.abs(rand - 0.5).max(axis=1) + 0.5
        out["random"][pool] = {
            "per_layer_abs_dev_p95": float(np.percentile(np.abs(rand - 0.5), 95) + 0.5),
            "best_of_layers_p95": float(np.percentile(best_of_layers, 95)),
            "best_of_layers_max": float(best_of_layers.max())}
        logger.info("[%s] random directions: 95th pct |AUROC| at one layer %.3f; "
                    "best-of-%d-layers 95th pct %.3f", pool,
                    out["random"][pool]["per_layer_abs_dev_p95"], len(layers),
                    out["random"][pool]["best_of_layers_p95"])

        for bname, bank in banks.items():
            table = {}
            for li, L in enumerate(layers):
                Xl = X16[:, li, :].to(torch.float32).numpy().astype(np.float64)
                for dname, d in directions(bank, L).items():
                    table.setdefault(dname, {})[int(L)] = float(roc_auc_score(y, Xl @ d))
            res = {}
            for dname, series in table.items():
                best_L = max(series, key=series.get)
                rec = {"by_layer": series, "best_layer": best_L, "best": series[best_L],
                       "key_layers": {}}
                for L in sorted(set(KEY_LAYERS) | {best_L}):
                    li = layers.index(L)
                    s = X16[:, li, :].to(torch.float32).numpy().astype(np.float64) @ \
                        directions(bank, L)[dname]
                    strat, resid = length_controlled(y, s, n_tok)
                    rec["key_layers"][str(L)] = {
                        "auroc": series[L], "ci95": boot_ci(y, s, args.boot, rng),
                        "within_length_quintiles": strat, "length_regressed_out": resid}
                res[dname] = rec
            out["results"][f"{bname}|{pool}"] = res

            logger.info("")
            logger.info("[%s pooling, %s bank] zero-shot AUROC on sycophancy_mt "
                        "(label 1 = retracted under pressure)", pool, bname)
            logger.info("  %-11s %7s %7s %7s  %-10s  %-26s %-10s %-10s", "direction", "L20",
                        "L40", "L44", "best", "CI95 @best", "len-strat", "len-resid")
            for dname, rec in res.items():
                kb = rec["key_layers"][str(rec["best_layer"])]
                logger.info("  %-11s %7.3f %7.3f %7.3f  L%-2d %.3f  [%.3f, %.3f]%-10s %-10.3f %-10.3f",
                            dname, rec["by_layer"].get(20, float("nan")),
                            rec["by_layer"][40], rec["by_layer"][44], rec["best_layer"],
                            rec["best"], kb["ci95"][0], kb["ci95"][1], "",
                            kb["within_length_quintiles"], kb["length_regressed_out"])
        del X16

    # ---- the verdict ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 96)
    for key, res in out["results"].items():
        bname, pool = key.split("|")
        syc, dec = res["sycophancy"], res["deception"]
        k40 = syc["key_layers"]["40"]
        null = out["random"][pool]["best_of_layers_p95"]
        logger.info("%-8s %-4s sycophancy axis: L40 %.3f [%.3f, %.3f] (length-strat %.3f) | "
                    "best L%d %.3f vs random best-of-layers p95 %.3f | deception axis L40 %.3f",
                    bname, pool, k40["auroc"], k40["ci95"][0], k40["ci95"][1],
                    k40["within_length_quintiles"], syc["best_layer"], syc["best"], null,
                    dec["key_layers"]["40"]["auroc"])

    path = Path(args.out) if args.out else run_dir.parent / "q10_sycophancy_axis.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
