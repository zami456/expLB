#!/usr/bin/env python
"""Q9: how many personas does the axis actually need?

The paper asserts, without a curve, that adding personas does not monotonically improve
performance. Our paper16 bank holds 8 concealment personas and 7 disclosure personas, so
this rebuilds the contrast from random subsets of those and measures detection at each
size -- the version of the question answerable with a CLEAN persona set, at the cost of a
short x-axis.

    python 15_persona_count.py

**Why not the 275-role HF bank.** It is cached and would give a much longer x-axis, but its
deception/honesty split is OUR grouping, and Q1 established that grouping is what carries the
Assistant-Axis contamination: cos(deception, assistant) is -0.66 there against -0.37 for
paper16. A curve built on it would measure how performance scales with *badly grouped*
personas, which is a different and less useful question. The tradeoff is a shorter range --
1 to 7 rather than 8 to 275 -- in exchange for measuring the right thing.

Each size is sampled `--draws` times with different personas, because which personas you pick
matters as much as how many, and a single draw per size would report that variation as trend.
"""

from __future__ import annotations

import argparse
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


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vectors_dir", default=str(REPO_ROOT / "run-paper16" / "vectors_s3_fp32"))
    ap.add_argument("--roles_file", default=str(PIPELINE_DIR / "pc_role_groups.json"))
    ap.add_argument("--run_dir", default=str(REPO_ROOT / "run-coords" / "passB"))
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    groups = json.loads(Path(args.roles_file).read_text(encoding="utf-8"))["groups"]
    # each group is {"contrast_with":..., "note":..., "roles":[...]}, not a bare list
    dec = sorted(groups["deception"]["roles"])
    hon = sorted(groups["honesty"]["roles"])
    logger.info("paper16 personas: %d deceptive, %d honest", len(dec), len(hon))

    vdir = Path(args.vectors_dir)
    vecs = {}
    for name in dec + hon:
        p = vdir / f"{name}.pt"
        if p.exists():
            vecs[name] = torch.load(p, map_location="cpu",
                                    weights_only=False)["vector"][args.layer].float().numpy()
    dec = [d for d in dec if d in vecs]
    hon = [h for h in hon if h in vecs]
    logger.info("vectors found: %d deceptive, %d honest at layer %d",
                len(dec), len(hon), args.layer)

    # eval activations, whole corpus, one layer
    import importlib.util
    spec = importlib.util.spec_from_file_location("sweep", PIPELINE_DIR / "11_layer_sweep.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    run_dir = Path(args.run_dir)
    Xs, ys = [], []
    for d in sorted(run_dir.iterdir()):
        if not (d.is_dir() and (d / "rows.jsonl").exists()):
            continue
        X, meta, layers = S.load_all_layers(run_dir, d.name, args.pooling)
        if X is None:
            continue
        y = np.array([m["label"] for m in meta], dtype=int)
        if len(np.unique(y)) < 2:
            continue
        li = layers.index(args.layer)
        Xs.append(X[:, li, :].to(torch.float32).numpy())
        ys.append(y)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    logger.info("eval rows: %d (%dD/%dH)", len(y), int(y.sum()), int((1 - y).sum()))

    rng = np.random.default_rng(args.seed)
    sizes = [n for n in (1, 2, 3, 4, 6, 8, 12, 16) if n <= min(len(dec), len(hon))]
    if min(len(dec), len(hon)) not in sizes:
        sizes.append(min(len(dec), len(hon)))

    logger.info("")
    logger.info("  %8s %10s %10s %8s", "personas", "AUROC", "sd", "draws")
    out = []
    for n in sizes:
        vals = []
        for _ in range(args.draws):
            ds = rng.choice(dec, size=n, replace=False)
            hs = rng.choice(hon, size=n, replace=False)
            d = unit(np.mean([vecs[a] for a in ds], axis=0)
                     - np.mean([vecs[b] for b in hs], axis=0))
            vals.append(float(roc_auc_score(y, X @ d)))
        out.append({"n": int(n), "auroc_mean": float(np.mean(vals)),
                    "auroc_sd": float(np.std(vals)), "draws": len(vals)})
        logger.info("  %8d %10.3f %10.3f %8d", n, out[-1]["auroc_mean"],
                    out[-1]["auroc_sd"], len(vals))

    best = max(out, key=lambda r: r["auroc_mean"])
    logger.info("")
    logger.info("  peak at n=%d (%.3f); full set n=%d (%.3f)",
                best["n"], best["auroc_mean"], out[-1]["n"], out[-1]["auroc_mean"])
    gains = [out[i]["auroc_mean"] - out[i - 1]["auroc_mean"] for i in range(1, len(out))]
    logger.info("  monotone increasing: %s", all(g > -0.002 for g in gains))
    logger.info("  gain from n=1 to full: %+.3f", out[-1]["auroc_mean"] - out[0]["auroc_mean"])

    path = Path(args.out) if args.out else run_dir.parent / f"persona_count_L{args.layer}.json"
    path.write_text(json.dumps({"layer": args.layer, "pooling": args.pooling, "curve": out},
                               indent=2), encoding="utf-8")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
