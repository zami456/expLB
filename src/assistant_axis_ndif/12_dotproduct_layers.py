#!/usr/bin/env python
"""Which layer gives the best raw dot product, and does a CROSS-layer one make sense?

Three questions, all measured rather than argued:

1. Zero-shot AUROC of <activation_L, axis_L> at every stored layer -- "the best dot product",
   with no probe and no training anywhere.
2. How much the deception axis rotates with depth: cos(axis_L, axis_M) for every pair. If the
   direction is stable across a band, projecting layer L onto axis M inside that band should
   be nearly free; if it rotates fast, cross-layer is meaningless.
3. The cross-layer projection itself: AUROC of <activation_L, axis_M> for all L, M. This is
   the thing the literature does NOT do, so it is worth having the number rather than an
   opinion about it.

    python 12_dotproduct_layers.py --run_dir run-coords/passB
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


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=str(REPO_ROOT / "run-coords" / "passB"))
    ap.add_argument("--bank", default=str(REPO_ROOT / "run-paper16" / "axes_vectors_s3_fp32.pt"))
    ap.add_argument("--pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--axis", default="deception")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("sweep", PIPELINE_DIR / "11_layer_sweep.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)

    run_dir = Path(args.run_dir)
    keys = sorted(d.name for d in run_dir.iterdir()
                  if d.is_dir() and (d / "rows.jsonl").exists())
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)

    Xs, ys, layers = [], [], None
    for k in keys:
        X, meta, L = S.load_all_layers(run_dir, k, args.pooling)
        if X is None:
            continue
        y = np.array([m["label"] for m in meta], dtype=int)
        if len(np.unique(y)) < 2:          # alpaca: honest only, no AUROC to compute
            continue
        layers = L
        Xs.append(X)
        ys.append(y)
        logger.info("  %-32s n=%d", k, len(y))
    X = torch.cat(Xs)
    y = np.concatenate(ys)
    logger.info("pooled: %d rows, %d layers, %dD/%dH",
                len(y), len(layers), int(y.sum()), int((1 - y).sum()))

    A = np.stack([unit(bank["axes"][args.axis][L].numpy()) for L in layers])   # (K, H)
    PC1 = np.stack([unit(bank["pca"]["deception"]["components"][L, 0].numpy()) for L in layers])

    # --- 1. matched-layer dot product -------------------------------------------------
    logger.info("")
    logger.info("matched-layer zero-shot AUROC  <act_L, axis_L>")
    logger.info("  %6s %10s %10s" % ("layer", "contrast", "PC1"))
    matched = []
    for li, L in enumerate(layers):
        x = X[:, li, :].to(torch.float32).numpy()
        a = float(roc_auc_score(y, x @ A[li]))
        p = float(roc_auc_score(y, x @ PC1[li]))
        matched.append({"layer": int(L), "contrast": a, "pc1": p})
        logger.info("  %6d %10.3f %10.3f" % (L, a, p))
    best = max(matched, key=lambda r: r["contrast"])
    logger.info("  best matched-layer dot product: layer %d, AUROC %.3f",
                best["layer"], best["contrast"])

    # --- 2. how fast does the axis rotate with depth? ---------------------------------
    logger.info("")
    logger.info("axis stability, cos(axis_L, axis_M) -- near 1 means the direction persists")
    Sim = A @ A.T
    hdr = "        " + "".join(f"{L:>6}" for L in layers)
    logger.info(hdr)
    for li, L in enumerate(layers):
        logger.info("  %5d " % L + "".join(f"{Sim[li, mj]:>6.2f}" for mj in range(len(layers))))

    # --- 3. cross-layer projection ----------------------------------------------------
    logger.info("")
    logger.info("cross-layer AUROC  <act_L, axis_M>   (rows = activation layer L)")
    logger.info(hdr)
    cross = np.zeros((len(layers), len(layers)))
    for li in range(len(layers)):
        x = X[:, li, :].to(torch.float32).numpy()
        for mj in range(len(layers)):
            cross[li, mj] = roc_auc_score(y, x @ A[mj])
        logger.info("  %5d " % layers[li] + "".join(f"{cross[li, mj]:>6.2f}"
                                                    for mj in range(len(layers))))
    di = np.diag(cross)
    best_flat = np.unravel_index(np.argmax(cross), cross.shape)
    logger.info("")
    logger.info("  best matched (diagonal): layer %d, %.3f",
                layers[int(np.argmax(di))], di.max())
    logger.info("  best cross overall:      act layer %d onto axis layer %d, %.3f",
                layers[best_flat[0]], layers[best_flat[1]], cross[best_flat])
    logger.info("  cross beats its own diagonal at %d of %d activation layers",
                int(sum(cross[i].max() > di[i] + 1e-9 for i in range(len(layers)))), len(layers))

    out = Path(args.out) if args.out else run_dir.parent / f"dotproducts_{args.pooling}.json"
    out.write_text(json.dumps({"layers": [int(L) for L in layers], "matched": matched,
                               "axis_cos": Sim.tolist(), "cross_auroc": cross.tolist()},
                              indent=2), encoding="utf-8")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
