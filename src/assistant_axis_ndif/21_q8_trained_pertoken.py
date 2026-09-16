#!/usr/bin/env python
"""Q8, finished: a TRAINED per-token probe -- Apollo's design -- against pooled probes.

`14_pertoken.py` answered half of Q8 zero-shot: projecting every token on the persona
deception axis and taking the top decile beats projecting the mean (0.807 vs 0.754). That
left the question an examiner will actually ask: Apollo's detector is not a fixed axis, it
is a probe TRAINED on per-token activations and aggregated afterwards. Does training on
tokens beat training on pooled vectors?

**One protocol for everything.** Leave one dataset out over the Pass C rows (2,000,
stratified, 200 per dataset): train on the other labelled datasets, test on the held-out
one, never train on `alpaca`, which is honest-only and serves as the control that sets the
TPR@1%FPR threshold. Same rows, same folds, same layer for every contender:

  tok_mms / tok_lr      trained on individual response TOKENS (each carries its row's
                        label), token scores aggregated per row: mean | max | top10pct | last
  pool_mean_* / pool_last_*
                        trained on the pooled vector of the same rows
  zeroshot_axis         the paper16 deception axis applied per token (no training)

`mms` and `lr` are exactly `9_persona_probes.make_probe("mms" / "lr_reg10")`: StandardScaler
then mean-difference, or StandardScaler then C=0.1 logistic regression without intercept.

**What "per-token" can and cannot add, stated up front.** Every probe here is linear, so a
token probe aggregated by the MEAN is arithmetically a pooled-mean probe with different
training weights. The per-token design can win for two reasons only: the token-level
training objective finds a better direction, or a nonlinear aggregation (max, top-decile)
recovers a signal that averaging dilutes. The table separates the two.

**Memory.** 28 GB of fp16 tokens do not fit as float32. Two passes:
  1. read every chunk once, keeping (a) the pooled vectors, (b) up to `--tokens_per_row`
     randomly sampled tokens per row for logistic regression, and (c) exact streaming
     first and second moments of ALL tokens per (dataset, label, layer), from which the
     token-weighted mms probe, scaler included, is computed exactly;
  2. read every chunk again and score every token with the trained probes.

**Self-checks that abort the run if they fail:**
  * the zero-shot axis, mean-aggregated over tokens, must reproduce the axis applied to the
    separately-returned mean-pooled vector -- this exercises the offsets, the row order and
    the layer index in one comparison;
  * the streaming mms (moments over all tokens) must match sklearn's mms pipeline fitted on
    the raw tokens, on a dataset whose rows are short enough that every token was sampled;
  * each probe's extracted linear form (w, b) must reproduce sklearn's own decision scores.
And a label-shuffle null for the token probe at one layer, which should land near 0.50.

    python 21_q8_trained_pertoken.py --run_dir run-coords/passC
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
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]
CONTROL = "alpaca"
AGGS = ("mean", "max", "top10pct", "last")
TOKEN_PROBES = ("tok_mms", "tok_lr", "zeroshot_axis")


def probes_module():
    """9_persona_probes' probe and balancing code. importlib is safe here -- nothing in this
    file is a trace body, so NDIF's whitelist never sees it."""
    spec = importlib.util.spec_from_file_location("probes",
                                                  PIPELINE_DIR / "9_persona_probes.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def aggregate(s: np.ndarray) -> dict:
    """s: (n_tok, n_probes) -> {agg: (n_probes,)}"""
    k = max(1, int(round(0.1 * s.shape[0])))
    srt = np.sort(s, axis=0)
    return {"mean": s.mean(axis=0), "max": s.max(axis=0),
            "top10pct": srt[-k:].mean(axis=0), "last": s[-1]}


def linear_form(clf):
    """sklearn pipeline (StandardScaler -> linear estimator) -> raw-space (w, b)."""
    sc = clf.steps[0][1]
    est = clf.steps[-1][1]
    w_scaled = est.w_ if hasattr(est, "w_") else est.coef_[0]
    b0 = float(est.intercept_[0]) if getattr(est, "fit_intercept", False) else 0.0
    w = np.asarray(w_scaled, dtype=np.float64) / sc.scale_
    return w, b0 - float(sc.mean_ @ w)


def mms_from_moments(stats: dict, keys: list, li: int):
    """Token-weighted StandardScaler + mean-difference, exactly, from streamed moments.

    StandardScaler uses the population variance over ALL training samples (both classes),
    so sigma^2 = E[x^2] - E[x]^2 over the pooled tokens; the probe in scaled space is
    (mu1 - mu0) / sigma, which in raw space is (mu1 - mu0) / sigma^2.
    """
    S = {0: 0.0, 1: 0.0}
    N = {0: 0, 1: 0}
    SS = 0.0
    for k in keys:
        for lab in (0, 1):
            st = stats.get((k, lab))
            if st is None:
                continue
            S[lab] = S[lab] + st["sum"][li]
            SS = SS + st["sumsq"][li]
            N[lab] += st["n"]
    if N[0] == 0 or N[1] == 0:
        return None
    n = N[0] + N[1]
    mu = (S[0] + S[1]) / n
    var = SS / n - mu * mu
    sd = np.sqrt(np.clip(var, 0.0, None))
    sd[sd == 0] = 1.0                      # sklearn's convention for constant features
    w = (S[1] / N[1] - S[0] / N[0]) / (sd * sd)
    return w, -float(mu @ w)


def chunks_of(run_dir: Path, key: str, limit: int = 0):
    paths = sorted((run_dir / key).glob("chunk_*.pt"))
    for c in (paths[:limit] if limit else paths):
        yield torch.load(c, weights_only=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=str(REPO_ROOT / "run-coords" / "passC"))
    ap.add_argument("--bank", default=str(REPO_ROOT / "run-paper16" / "axes_vectors_s3_fp32.pt"))
    ap.add_argument("--tokens_per_row", type=int, default=16,
                    help="tokens sampled per row for logistic regression")
    ap.add_argument("--max_train_tokens", type=int, default=40000)
    ap.add_argument("--null_layer", type=int, default=44)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_chunks", type=int, default=0,
                    help="smoke test: read only the first N chunks of each dataset. Results "
                         "from a limited run are NOT comparable and must not be quoted")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    P = probes_module()
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    run_dir = Path(args.run_dir)
    keys = sorted(d.name for d in run_dir.iterdir()
                  if d.is_dir() and (d / "rows.jsonl").exists())
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------ pass 1
    logger.info("pass 1: pooled vectors, sampled tokens, streaming moments over %d datasets",
                len(keys))
    rows, row_of = [], {}
    pooled = {"mean": [], "last": []}
    tok_samples, samp_row = [], []
    stats: dict = {}
    layers = None
    for key in keys:
        meta = {}
        with (run_dir / key / "rows.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                meta[r["row_id"]] = r
        for blob in chunks_of(run_dir, key, args.limit_chunks):
            if "tokens" not in blob:
                raise SystemExit(f"{key}: chunk has no per-token block -- not a Pass C store")
            if layers is None:
                layers = list(blob["layers"])
            elif list(blob["layers"]) != layers:
                raise SystemExit(f"{key}: layer list changed between chunks")
            pools = list(blob["pooling"])
            tk, offs, pb = blob["tokens"], blob["tok_offsets"], blob["pooled"]
            for i, rid in enumerate(blob["row_ids"]):
                m = meta.get(rid)
                if m is None or m.get("label") is None:
                    continue
                seg = tk[int(offs[i]):int(offs[i + 1])]              # (n, K, H) fp16
                n = seg.shape[0]
                if n == 0:
                    continue
                ri = len(rows)
                row_of[rid] = ri
                rows.append({"key": key, "row_id": rid, "label": int(m["label"]), "n_tok": n})
                for pn in ("mean", "last"):
                    pooled[pn].append(pb[i, :, :, pools.index(pn)].clone())
                pick = np.sort(rng.choice(n, size=min(n, args.tokens_per_row), replace=False))
                tok_samples.append(seg[torch.as_tensor(pick)].clone())
                samp_row.extend([ri] * len(pick))
                if key != CONTROL:
                    f = seg.to(torch.float32)
                    st = stats.setdefault((key, int(m["label"])),
                                          {"sum": 0.0, "sumsq": 0.0, "n": 0})
                    st["sum"] = st["sum"] + f.sum(dim=0).double().numpy()
                    st["sumsq"] = st["sumsq"] + (f * f).sum(dim=0).double().numpy()
                    st["n"] += n
        logger.info("  %-32s rows so far %5d   %.0fs", key, len(rows), time.time() - t0)

    y = np.array([r["label"] for r in rows])
    kr = np.array([r["key"] for r in rows])
    pooled = {pn: torch.stack(v) for pn, v in pooled.items()}          # (R, K, H) fp16
    tok_samples = torch.cat(tok_samples)                                # (S, K, H) fp16
    samp_row = np.array(samp_row)
    ys_tok = y[samp_row]
    heldouts = [k for k in keys if k != CONTROL and len(np.unique(y[kr == k])) == 2]
    logger.info("rows %d, sampled tokens %d, layers %s, folds %d",
                len(rows), len(samp_row), layers, len(heldouts))
    if CONTROL not in keys:
        raise SystemExit("no alpaca control in the store; TPR@1%FPR undefined")

    # ------------------------------------------------------------------ self-check: moments
    # a dataset whose every row fits inside tokens_per_row has ALL its tokens sampled, so
    # sklearn's mms on those tokens must equal the streamed moments.
    full = [k for k in heldouts
            if max(r["n_tok"] for r in rows if r["key"] == k) <= args.tokens_per_row]
    if full:
        k = full[0]
        li = len(layers) // 2
        sel = np.where(kr[samp_row] == k)[0]
        clf = P.make_probe("mms", args.seed).fit(
            tok_samples[torch.as_tensor(sel), li].to(torch.float32).numpy(), ys_tok[sel])
        w_sk, b_sk = linear_form(clf)
        w_st, b_st = mms_from_moments(stats, [k], li)
        cos = float(unit(w_sk) @ unit(w_st))
        logger.info("self-check moments: %s layer %d, cos(sklearn mms, streamed mms) = %.6f",
                    k, layers[li], cos)
        if cos < 0.999:
            raise SystemExit("streamed moments disagree with sklearn -- do not trust tok_mms")
    else:
        logger.warning("self-check moments skipped: no dataset fits in tokens_per_row")

    # ------------------------------------------------------------------ train
    logger.info("training: %d folds x %d layers", len(heldouts), len(layers))
    lin: dict = {}          # (fold, L, probe) -> (w float64 (H,), b)
    checked_linear = False
    for li, L in enumerate(layers):
        axis = unit(bank["axes"]["deception"][L].numpy().astype(np.float64))
        for fold in heldouts:
            train_keys = [k for k in heldouts if k != fold]
            lin[(fold, L, "tok_mms")] = mms_from_moments(stats, train_keys, li)
            lin[(fold, L, "zeroshot_axis")] = (axis, 0.0)

            idx = np.where(np.isin(kr[samp_row], train_keys))[0]
            idx = P.balance(idx, ys_tok, args.seed, cap=args.max_train_tokens)
            Xt = tok_samples[torch.as_tensor(idx), li].to(torch.float32).numpy()
            clf = P.make_probe("lr_reg10", args.seed).fit(Xt, ys_tok[idx])
            lin[(fold, L, "tok_lr")] = linear_form(clf)
            if not checked_linear:
                w, b = linear_form(clf)
                d = np.abs(Xt[:200] @ w + b - P._scores(clf, Xt[:200])).max()
                logger.info("self-check linear form: max |w.x+b - sklearn| = %.2e", d)
                if d > 1e-2:
                    raise SystemExit("linear form extraction is wrong")
                checked_linear = True

            if L == args.null_layer:
                # shuffle ROW labels, then let tokens inherit them, so the null keeps the
                # within-row token correlation the real probe sees. Shuffle WITHIN each
                # dataset: a shuffle across datasets leaves some with more positives than
                # others, and the "null" then learns dataset identity -- which is 99.3%
                # decodable here -- and lands systematically off 0.50 (a smoke run gave
                # 0.39 that way).
                y_perm = y.copy()
                prng = np.random.default_rng(args.seed + 7)
                for tk_ in train_keys:
                    rr = np.where(kr == tk_)[0]
                    y_perm[rr] = prng.permutation(y[rr])
                yt_null = y_perm[samp_row]
                idn = P.balance(np.where(np.isin(kr[samp_row], train_keys))[0], yt_null,
                                args.seed, cap=args.max_train_tokens)
                cn = P.make_probe("lr_reg10", args.seed).fit(
                    tok_samples[torch.as_tensor(idn), li].to(torch.float32).numpy(),
                    yt_null[idn])
                lin[(fold, L, "tok_lr_null")] = linear_form(cn)

            tr_rows = P.balance(np.where(np.isin(kr, train_keys))[0], y, args.seed)
            for pn in ("mean", "last"):
                Xp = pooled[pn][torch.as_tensor(tr_rows), li].to(torch.float32).numpy()
                for name in ("mms", "lr_reg10"):
                    c = P.make_probe(name, args.seed).fit(Xp, y[tr_rows])
                    short = "mms" if name == "mms" else "lr"
                    lin[(fold, L, f"pool_{pn}_{short}")] = linear_form(c)
        logger.info("  layer %2d trained   %.0fs", L, time.time() - t0)
    del tok_samples

    # ------------------------------------------------------------------ pass 2: token scores
    logger.info("pass 2: scoring every token")
    tok_probe_names = list(TOKEN_PROBES) + ["tok_lr_null"]
    score: dict = {}        # (fold, L, probe, agg) -> np.array (R,) nan-filled
    R = len(rows)
    worst_dev, n_dev_bad, n_dev = 0.0, 0, 0
    for key in keys:
        folds_here = heldouts if key == CONTROL else ([key] if key in heldouts else [])
        if not folds_here:
            continue
        for blob in chunks_of(run_dir, key, args.limit_chunks):
            tk, offs = blob["tokens"], blob["tok_offsets"]
            rids = blob["row_ids"]
            for li, L in enumerate(layers):
                probes_L = [p for p in tok_probe_names if (folds_here[0], L, p) in lin]
                cols = [(f, p) for f in folds_here for p in probes_L]
                W = np.stack([lin[(f, L, p)][0] for f, p in cols]).astype(np.float32)
                b = np.array([lin[(f, L, p)][1] for f, p in cols], dtype=np.float32)
                X = tk[:, li].to(torch.float32).numpy()                 # (S, H)
                S = X @ W.T + b                                          # (S, n_cols)
                for i, rid in enumerate(rids):
                    ri = row_of.get(rid)
                    if ri is None:
                        continue
                    agg = aggregate(S[int(offs[i]):int(offs[i + 1])])
                    for a, vals in agg.items():
                        for c, (f, p) in enumerate(cols):
                            arr = score.setdefault((f, L, p, a), np.full(R, np.nan))
                            arr[ri] = vals[c]
                    # self-check: mean of per-token axis scores == axis . mean-pooled vector
                    c0 = cols.index((folds_here[0], "zeroshot_axis"))
                    ref = float(pooled["mean"][ri, li].to(torch.float32).numpy()
                                @ W[c0].astype(np.float32))
                    dev = abs(float(agg["mean"][c0]) - ref)
                    tol = 1e-2 * abs(ref) + 1e-3
                    worst_dev = max(worst_dev, dev / (abs(ref) + 1e-6))
                    n_dev += 1
                    n_dev_bad += int(dev > tol)
        logger.info("  %-32s scored   %.0fs", key, time.time() - t0)
    logger.info("self-check offsets/layers: %d of %d row-layer means outside tolerance, "
                "worst relative deviation %.2e", n_dev_bad, n_dev, worst_dev)
    if n_dev_bad > 0.01 * max(n_dev, 1):
        raise SystemExit("per-token mean does not reproduce mean pooling -- offsets or "
                         "layer indexing are wrong; results withheld")

    # pooled probes score directly from the pooled vectors
    for li, L in enumerate(layers):
        for pn in ("mean", "last"):
            Xp = pooled[pn][:, li].to(torch.float32).numpy()
            for short in ("mms", "lr"):
                for fold in heldouts:
                    w, b = lin[(fold, L, f"pool_{pn}_{short}")]
                    score[(fold, L, f"pool_{pn}_{short}", "pooled")] = Xp @ w + b

    # response length as a no-activation reference on exactly these rows and folds. Length
    # predicts the label with the same sign across most of the corpus (longer = deceptive),
    # so any score that correlates with length -- including a shuffled-label probe read at
    # the last token -- inherits it. Layer-independent; recorded at every layer so it sits
    # in each table.
    n_tok = np.array([r["n_tok"] for r in rows], dtype=float)
    for L in layers:
        for fold in heldouts:
            score[(fold, L, "length_only", "n_tok")] = n_tok

    # ------------------------------------------------------------------ metrics
    results = []
    combos = sorted({(L, p, a) for (_, L, p, a) in score})
    for L, p, a in combos:
        per = []
        for fold in heldouts:
            s = score.get((fold, L, p, a))
            if s is None:
                continue
            te = kr == fold
            ctl = kr == CONTROL
            if np.isnan(s[te]).any() or np.isnan(s[ctl]).any():
                raise SystemExit(f"missing scores for {fold} {L} {p} {a}")
            au = float(roc_auc_score(y[te], s[te]))
            thr = float(np.quantile(s[ctl], 0.99))
            tpr = float((s[te & (y == 1)] > thr).mean())
            per.append({"held_out": fold, "auroc": au, "tpr_at_1pct_fpr": tpr})
        if per:
            results.append({"layer": L, "probe": p, "aggregation": a,
                            "auroc_mean": float(np.mean([r["auroc"] for r in per])),
                            "tpr_mean": float(np.mean([r["tpr_at_1pct_fpr"] for r in per])),
                            "per_fold": per})

    def table_row(r):
        return (f"  {r['probe']:<14} {r['aggregation']:<9} "
                f"AUROC {r['auroc_mean']:.3f}   TPR@1%FPR {r['tpr_mean']:.3f}")

    for L in layers:
        logger.info("")
        logger.info("layer %d -- leave-one-dataset-out, mean over %d held-out datasets",
                    L, len(heldouts))
        for r in sorted([r for r in results if r["layer"] == L],
                        key=lambda r: -r["auroc_mean"]):
            logger.info(table_row(r))

    logger.info("")
    logger.info("=" * 78)
    logger.info("best TRAINED token probe vs best TRAINED pooled probe, per layer")
    summary = []
    for L in layers:
        rl = [r for r in results if r["layer"] == L]
        tok = max([r for r in rl if r["probe"] in ("tok_mms", "tok_lr")],
                  key=lambda r: r["auroc_mean"])
        tok_mean = max([r for r in rl if r["probe"] in ("tok_mms", "tok_lr")
                        and r["aggregation"] == "mean"], key=lambda r: r["auroc_mean"])
        pool = max([r for r in rl if r["probe"].startswith("pool_")],
                   key=lambda r: r["auroc_mean"])
        zs = max([r for r in rl if r["probe"] == "zeroshot_axis"],
                 key=lambda r: r["auroc_mean"])
        null = {r["aggregation"]: r["auroc_mean"] for r in rl if r["probe"] == "tok_lr_null"}
        length = [r["auroc_mean"] for r in rl if r["probe"] == "length_only"]
        summary.append({"layer": L, "best_token": tok, "best_token_mean_agg": tok_mean,
                        "best_pooled": pool, "best_zeroshot": zs,
                        "null_by_aggregation": null or None,
                        "length_only": length[0] if length else None})
        logger.info("  L%-3d token %s/%s %.3f  (mean-agg %.3f)  |  pooled %s %.3f  |  "
                    "zero-shot axis %s %.3f  |  token-minus-pooled %+.3f  |  length %.3f",
                    L, tok["probe"], tok["aggregation"], tok["auroc_mean"],
                    tok_mean["auroc_mean"], pool["probe"], pool["auroc_mean"],
                    zs["aggregation"], zs["auroc_mean"],
                    tok["auroc_mean"] - pool["auroc_mean"],
                    length[0] if length else float("nan"))
        if null:
            logger.info("        shuffled-label token probe: %s",
                        "  ".join(f"{a} {v:.3f}" for a, v in sorted(null.items())))

    out = Path(args.out) if args.out else run_dir.parent / "q8_trained_pertoken.json"
    out.write_text(json.dumps({"protocol": "leave-one-dataset-out", "control": CONTROL,
                               "layers": layers, "held_out": heldouts,
                               "tokens_per_row": args.tokens_per_row,
                               "max_train_tokens": args.max_train_tokens,
                               "n_rows": len(rows), "summary": summary,
                               "results": results,
                               "self_checks": {"offset_worst_rel_dev": worst_dev,
                                               "offset_rows_bad": n_dev_bad,
                                               "offset_rows": n_dev}},
                              indent=2), encoding="utf-8")
    logger.info("wrote %s   total %.0fs", out, time.time() - t0)


if __name__ == "__main__":
    main()
