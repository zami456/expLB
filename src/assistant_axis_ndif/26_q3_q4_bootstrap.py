#!/usr/bin/env python
"""Q3 and Q4 under the same statistical standard as Q2/Q6/Q8 (`23_significance.py`).

**Q3 — sample efficiency.** The claim: with more training rows the raw probe gets WORSE
(0.691 at n=16 -> 0.659 at n=1024) while persona axes get BETTER (0.690 -> 0.717). The P2 curve
is recomputed exactly as `9_persona_probes.p2_curves` built it (layer 44, last token,
lr_reg10, 3 seeds, the 10-dataset corpus), checked against `p2_L44_last.json`, and then tested:

  * each feature's change from n=16 to n=1024, and its slope over log2(n) across all 7 sizes;
  * axes minus raw at n=16 and at n=1024, and the difference-in-differences.

**Q4 — is there deception signal beyond the Assistant Axis?** The number quoted so far
("erasing it costs 0.006-0.023") came from an EARLY run on 4 datasets at layer 20 only
(`erase_test.json`), which breaks this project's rule about partial-corpus numbers. So Q4 is
not just bootstrapped, it is RE-RUN on the full corpus at layers 20 (mean), 40 (mean) and
44 (last), with both probes:

  raw               the 8192-d pooled activation
  erase:assistant   the same activations with the Assistant Axis projected out
  erase:random_r    5 different random directions projected out -- the rank-matched control

and tested: the cost of erasing the Assistant Axis (raw - erased), whether that cost exceeds
the cost of erasing a random direction, and whether the erased probe still transfers (its
AUROC interval stays above 0.5).

Both parts report the row bootstrap (resample test rows, paired) and the dataset bootstrap
(resample the 9 labelled datasets, paired), BH across the family on the dataset p-values,
exactly as `23_significance.py` does and with its tested helpers.

    python 26_q3_q4_bootstrap.py
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]
NS = [16, 32, 64, 128, 256, 512, 1024]
Q4_SETTINGS = [(20, "mean"), (40, "mean"), (44, "last")]


def module(name, fname):
    spec = importlib.util.spec_from_file_location(name, PIPELINE_DIR / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ci(v, lo=2.5, hi=97.5):
    v = np.asarray(v)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


class Bootstrapper:
    """Shared, paired resampling: one row resample per (target, test-set seed) and one
    dataset resample, reused by every method so differences are paired."""

    def __init__(self, G, datasets, boot_rows, boot_datasets, rng):
        self.G, self.datasets = G, datasets
        self.idx = {k: i for i, k in enumerate(datasets)}
        self.boot_rows, self.rng = boot_rows, rng
        self.rows = {}
        D = len(datasets)
        counts = rng.multinomial(D, np.full(D, 1.0 / D), size=boot_datasets)
        self.W = counts[:, :, None] * counts[:, None, :]
        self.W[:, np.arange(D), np.arange(D)] = 0

    def row_index(self, t, seed, n):
        key = (t, seed)
        if key not in self.rows:
            self.rows[key] = (self.rng.integers(0, n, size=(self.boot_rows, n)),
                              self.rng.integers(0, n, size=(self.boot_rows, n)) + n)
        return self.rows[key]

    def score_pair(self, x, n, t, seed):
        """x: scores on the balanced test set [pos..., neg...] -> (point, (B,) boot)."""
        Rp, Rn = self.row_index(t, seed, n)
        point = float(self.G.auroc_rows(x[None, :], n)[0])
        boot = self.G.auroc_rows(np.concatenate([x[Rp], x[Rn]], axis=1), n)
        return point, boot

    def dataset_boot(self, point: dict):
        """point {(s, t): auc} -> (B,) mean over off-diagonal pairs among drawn datasets."""
        D = len(self.datasets)
        A = np.full((D, D), np.nan)
        for (s, t), v in point.items():
            if s in self.idx and t in self.idx:
                A[self.idx[s], self.idx[t]] = v
        mask = ~np.isnan(A)
        num = (self.W * np.where(mask, A, 0.0)[None]).sum(axis=(1, 2))
        den = (self.W * mask[None]).sum(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den


def load(S, P, run_dir, keys, pooling):
    data = {}
    for k in keys:
        X, meta, layers = S.load_all_layers(run_dir, k, pooling)
        if X is None:
            continue
        data[k] = (X, np.array([m["label"] for m in meta], dtype=int), meta, layers)
    src = [k for k, (_, y, _, _) in data.items() if k != P.CONTROL and len(np.unique(y)) == 2]
    return data, src


def feats_at(P, data, L, fkind, bank, seed):
    return {k: P.make_features(X16[:, layers.index(L), :].to(torch.float32).numpy(),
                               fkind, bank, L, 5, seed)
            for k, (X16, _, _, layers) in data.items()}


def compare(name, question, a_rows, b_rows, a_ds, b_ds, a_pt, b_pt, G, kind="diff"):
    d_rows = a_rows - b_rows
    d_ds = a_ds - b_ds
    d_ds = d_ds[~np.isnan(d_ds)]
    return {"name": name, "question": question, "kind": kind, "diff": float(a_pt - b_pt),
            "row_ci95": ci(d_rows), "dataset_ci95": ci(d_ds),
            "row_p": G.two_sided_p(d_rows), "dataset_p": G.two_sided_p(d_ds)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=str(REPO_ROOT / "run-coords" / "passB"))
    ap.add_argument("--bank", default=str(REPO_ROOT / "run-paper16" / "axes_vectors_s3_fp32.pt"))
    ap.add_argument("--exclude", nargs="+", default=["sycophancy_mt"])
    ap.add_argument("--p2_seeds", type=int, default=3,
                    help="the stored P2 run used 3 seeds (216 obs = 3 x 72 pairs)")
    ap.add_argument("--random_erasures", type=int, default=5)
    ap.add_argument("--boot_rows", type=int, default=1000)
    ap.add_argument("--boot_datasets", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    P = module("probes", "9_persona_probes.py")
    S = module("sweep", "11_layer_sweep.py")
    G = module("sig", "23_significance.py")
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    run_dir = Path(args.run_dir)
    keys = sorted(d.name for d in run_dir.iterdir()
                  if d.is_dir() and (d / "rows.jsonl").exists() and d.name not in args.exclude)
    rng = np.random.default_rng(args.seed)
    results, curves, q4_table = [], {}, []

    # =============================== Q3 =================================================
    logger.info("Q3: recomputing the P2 curve, layer 44 last-token")
    data, src = load(S, P, run_dir, keys, "last")
    B = Bootstrapper(G, src, args.boot_rows, args.boot_datasets, rng)
    L = 44
    q3 = {}   # (feat, probe, n) -> {"point", "rows" (B,), "ds" (B,)}
    for feat in ("raw", "axes"):
        # raw and axes features do not depend on the seed (only `random` would), so build
        # them once per feature instead of once per seed and size
        feats = feats_at(P, data, L, feat, bank, 0)
        for probe in ("lr_reg10", "mms"):
            for n in NS:
                pts, boots, ds_list = [], [], []
                for seed in range(args.p2_seeds):
                    point = {}
                    for s in src:
                        _, ys, ms, _ = data[s]
                        tr, _ = P.group_split(ms, 0.75, seed)
                        tr = P.balance(tr, ys, seed, cap=n)
                        if len(tr) < min(8, n):
                            continue
                        clf = P.make_probe(probe, seed).fit(feats[s][tr], ys[tr])
                        for t, (_, yt, _, _) in data.items():
                            if t == s:
                                continue
                            te = P.balance(np.arange(len(yt)), yt, seed + 1)
                            if len(te) < 8:
                                continue
                            x = P._scores(clf, feats[t][te])
                            pt, bt = B.score_pair(x, len(te) // 2, t, seed)
                            point[(s, t)] = pt
                            pts.append(pt)
                            boots.append(bt)
                    ds_list.append(B.dataset_boot(point))
                q3[(feat, probe, n)] = {"point": float(np.mean(pts)),
                                        "rows": np.mean(boots, axis=0),
                                        "ds": np.nanmean(np.stack(ds_list), axis=0),
                                        "n_obs": len(pts)}
                logger.info("  %-5s %-8s n=%-5d %.4f (%d obs)   %.0fs", feat, probe, n,
                            q3[(feat, probe, n)]["point"], len(pts), time.time() - t0)

    # reproduce the stored curve before testing it
    ref = json.loads((run_dir.parent / "p2_L44_last.json").read_text(encoding="utf-8"))
    worst = 0.0
    for r in ref["p2"]:
        k = (r["features"], r["probe"], r["n"])
        if k in q3:
            worst = max(worst, abs(q3[k]["point"] - r["auroc_mean"]))
            if q3[k]["n_obs"] != r["n_obs"]:
                raise SystemExit(f"{k}: {q3[k]['n_obs']} observations, stored run had {r['n_obs']}")
    logger.info("Q3 reproduction against p2_L44_last.json: worst |diff| %.5f", worst)
    if worst > 0.002:
        raise SystemExit("recomputed P2 curve does not match the stored one; not testing it")

    x = np.log2(NS)
    xc = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    for probe in ("lr_reg10", "mms"):
        for feat in ("raw", "axes"):
            curves[f"{feat}|{probe}"] = [{"n": n, "auroc": q3[(feat, probe, n)]["point"],
                                          "row_ci95": ci(q3[(feat, probe, n)]["rows"]),
                                          "dataset_ci95": ci(q3[(feat, probe, n)]["ds"])}
                                         for n in NS]
            lo, hi = q3[(feat, probe, 16)], q3[(feat, probe, 1024)]
            results.append(compare(f"{feat} ({probe}): n=1024 minus n=16", "Q3",
                                   hi["rows"], lo["rows"], hi["ds"], lo["ds"],
                                   hi["point"], lo["point"], G))
            slope_rows = sum(c * q3[(feat, probe, n)]["rows"] for c, n in zip(xc, NS))
            slope_ds = sum(c * q3[(feat, probe, n)]["ds"] for c, n in zip(xc, NS))
            slope_pt = sum(c * q3[(feat, probe, n)]["point"] for c, n in zip(xc, NS))
            zeros = np.zeros_like(slope_rows)
            r = compare(f"{feat} ({probe}): slope per doubling of n", "Q3", slope_rows, zeros,
                        slope_ds, np.zeros_like(slope_ds), slope_pt, 0.0, G)
            results.append(r)
        for n in (16, 1024):
            a, b = q3[("axes", probe, n)], q3[("raw", probe, n)]
            results.append(compare(f"axes minus raw at n={n} ({probe})", "Q3", a["rows"], b["rows"],
                                   a["ds"], b["ds"], a["point"], b["point"], G))
        a16, b16 = q3[("axes", probe, 16)], q3[("raw", probe, 16)]
        a1k, b1k = q3[("axes", probe, 1024)], q3[("raw", probe, 1024)]
        results.append(compare(f"difference-in-differences, axes vs raw ({probe})", "Q3",
                               (a1k["rows"] - b1k["rows"]), (a16["rows"] - b16["rows"]),
                               (a1k["ds"] - b1k["ds"]), (a16["ds"] - b16["ds"]),
                               a1k["point"] - b1k["point"], a16["point"] - b16["point"], G))
    del data

    # =============================== Q4 =================================================
    by_pool = {}
    for L, pool in Q4_SETTINGS:
        by_pool.setdefault(pool, []).append(L)
    for pool, layers_needed in by_pool.items():
        logger.info("Q4: loading pooling=%s", pool)
        data, src = load(S, P, run_dir, keys, pool)
        B = Bootstrapper(G, src, args.boot_rows, args.boot_datasets, rng)
        sweep = json.loads((run_dir.parent / G.SWEEP_FILES[pool]).read_text(encoding="utf-8"))
        for L in layers_needed:
            fkinds = [("raw", "raw", args.seed), ("erase:assistant", "erase:assistant", args.seed)]
            fkinds += [(f"erase:random_{r}", "erase:random", 1000 + r)
                       for r in range(args.random_erasures)]
            for probe in ("mms", "lr_reg10"):
                res = {}
                for label, fkind, fseed in fkinds:
                    feats = feats_at(P, data, L, fkind, bank, fseed)
                    point, boots = {}, []
                    for s in src:
                        _, ys, ms, _ = data[s]
                        tr, _ = P.group_split(ms, 0.75, args.seed)
                        tr = P.balance(tr, ys, args.seed)
                        clf = P.make_probe(probe, args.seed).fit(feats[s][tr], ys[tr])
                        for t, (_, yt, _, _) in data.items():
                            if t == s:
                                continue
                            te = P.balance(np.arange(len(yt)), yt, args.seed + 1)
                            if len(te) < 8:
                                continue
                            pt, bt = B.score_pair(P._scores(clf, feats[t][te]), len(te) // 2,
                                                  t, args.seed)
                            point[(s, t)] = pt
                            boots.append(bt)
                    res[label] = {"point": float(np.mean(list(point.values()))),
                                  "rows": np.mean(boots, axis=0), "ds": B.dataset_boot(point)}
                    logger.info("  L%d %-4s %-8s %-17s off-diag %.4f   %.0fs", L, pool, probe,
                                label, res[label]["point"], time.time() - t0)
                if probe == "mms":
                    ref = next(r["raw"] for r in sweep["sweep"] if r["layer"] == L)
                    dev = abs(ref - res["raw"]["point"])
                    logger.info("  reproduction: raw mms L%d %s %.4f vs sweep %.4f (|diff| %.5f)",
                                L, pool, res["raw"]["point"], ref, dev)
                    if dev > 0.002:
                        raise SystemExit("Q4 raw baseline does not reproduce the sweep")
                rand = [res[f"erase:random_{r}"] for r in range(args.random_erasures)]
                rand_rows = np.mean([r["rows"] for r in rand], axis=0)
                rand_ds = np.mean([r["ds"] for r in rand], axis=0)
                rand_pt = float(np.mean([r["point"] for r in rand]))
                tag = f"L{L} {pool} {probe}"
                raw, era = res["raw"], res["erase:assistant"]
                results.append(compare(f"cost of erasing Assistant Axis, {tag}", "Q4",
                                       raw["rows"], era["rows"], raw["ds"], era["ds"],
                                       raw["point"], era["point"], G))
                results.append(compare(f"assistant erasure vs random erasures, {tag}", "Q4",
                                       rand_rows, era["rows"], rand_ds, era["ds"],
                                       rand_pt, era["point"], G))
                costs_random = [raw["point"] - r["point"] for r in rand]
                q4_table.append({"layer": L, "pooling": pool, "probe": probe,
                                 "raw": raw["point"], "erase_assistant": era["point"],
                                 "erase_random_mean": rand_pt,
                                 "cost_assistant": raw["point"] - era["point"],
                                 "cost_random": costs_random,
                                 "erased_row_ci95": ci(era["rows"]),
                                 "erased_dataset_ci95": ci(era["ds"]),
                                 "erased_above_chance_datasets":
                                     bool(ci(era["ds"])[0] > 0.5)})
        del data

    # =============================== report =============================================
    diffs = [r for r in results]
    q = G.bh([r["dataset_p"] for r in diffs])
    for r, qq in zip(diffs, q):
        r["dataset_q_bh"] = float(qq)
        r["significant"] = bool(qq < 0.05)

    logger.info("")
    logger.info("=" * 118)
    logger.info("%-3s %-56s %8s  %-19s %-19s %7s  %s", "Q", "comparison", "diff",
                "rows 95% CI", "datasets 95% CI", "q(BH)", "verdict")
    for r in results:
        logger.info("%-3s %-56s %+8.4f  [%+.3f,%+.3f]     [%+.3f,%+.3f]     %7.3f  %s",
                    r["question"], r["name"][:56], r["diff"], *r["row_ci95"],
                    *r["dataset_ci95"], r["dataset_q_bh"],
                    "significant" if r["significant"] else "NOT significant")
    logger.info("")
    logger.info("Q4 summary (full corpus):")
    for t in q4_table:
        logger.info("  L%-2d %-4s %-8s raw %.3f | erase assistant %.3f (cost %+.3f) | erase random "
                    "%.3f (costs %s) | erased CI datasets [%.3f, %.3f]",
                    t["layer"], t["pooling"], t["probe"], t["raw"], t["erase_assistant"],
                    t["cost_assistant"], t["erase_random_mean"],
                    ", ".join(f"{c:+.3f}" for c in t["cost_random"]), *t["erased_dataset_ci95"])

    path = Path(args.out) if args.out else run_dir.parent / "q3_q4_bootstrap.json"
    path.write_text(json.dumps({"q3_reproduction_worst": worst, "q3_curves": curves,
                                "q4_table": q4_table, "comparisons": results,
                                "boot_rows": args.boot_rows,
                                "boot_datasets": args.boot_datasets},
                               indent=2), encoding="utf-8")
    logger.info("wrote %s   %.0fs", path, time.time() - t0)


if __name__ == "__main__":
    main()
