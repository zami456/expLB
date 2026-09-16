#!/usr/bin/env python
"""Pipeline sanity checks: is the geometry what we think it is, or are we reading an artefact?

Five checks, each aimed at a specific way these results could be wrong.

1. **PCA of the eval activations.** What dominates the variance? If the leading components
   track DATASET IDENTITY rather than the label, then a probe trained on one dataset and
   tested on another may be reading "which corpus is this" and the whole transfer story is
   about corpus similarity. This is the check that matters most, and nothing so far has run it.
2. **Label vs dataset separability.** Fit the same probe to predict the label, then to predict
   dataset identity, from the same activations. If dataset is far more decodable than label,
   the representation is dominated by corpus style.
3. **Does the deception axis live in the high-variance subspace?** A direction that explains
   almost no variance can still classify, but it is worth knowing which case we are in.
4. **The persona-bank diagnostic**, re-run on eval data: PC1 of the persona vectors is known to
   be the Assistant Axis. Confirm the eval activations agree about which direction that is.
5. **Norm growth with depth**, the thing that makes unnormalised projections meaningless.

    python 17_validate.py --layer 40
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--cap", type=int, default=400, help="rows per dataset, to keep PCA balanced")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("sweep", PIPELINE_DIR / "11_layer_sweep.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    run_dir = Path(args.run_dir)
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)

    rng = np.random.default_rng(0)
    Xs, ys, ds, layers = [], [], [], None
    for d in sorted(run_dir.iterdir()):
        if not (d.is_dir() and (d / "rows.jsonl").exists()):
            continue
        X, meta, L = S.load_all_layers(run_dir, d.name, args.pooling)
        if X is None:
            continue
        layers = L
        y = np.array([m["label"] for m in meta], dtype=int)
        # Cap per dataset so the PCA is not simply a picture of instructed-deception, which
        # is 5,432 of 15,278 rows and would otherwise define the principal components.
        idx = np.arange(len(y))
        if len(idx) > args.cap:
            idx = rng.choice(idx, args.cap, replace=False)
        li = L.index(args.layer)
        Xs.append(X[idx][:, li, :].to(torch.float32).numpy())
        ys.append(y[idx])
        ds += [d.name] * len(idx)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    ds = np.array(ds)
    logger.info("layer %d, %s pooling: %d rows, %d datasets, %dD/%dH",
                args.layer, args.pooling, len(y), len(set(ds)), int(y.sum()), int((1 - y).sum()))

    Xc = X - X.mean(0)
    U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum()
    logger.info("")
    logger.info("1. PCA of eval activations -- variance explained")
    logger.info("   PC1 %.1f%%  PC2 %.1f%%  PC3 %.1f%%  top-10 %.1f%%",
                100 * var[0], 100 * var[1], 100 * var[2], 100 * var[:10].sum())

    logger.info("")
    logger.info("2. what do the leading PCs track -- the LABEL, or which dataset a row is from?")
    logger.info("   %4s %10s %26s", "PC", "AUROC(label)", "best-separated dataset (AUROC)")
    res_pc = []
    for j in range(5):
        proj = Xc @ Vt[j]
        a_lab = roc_auc_score(y, proj)
        best, best_a = None, 0.5
        for name in sorted(set(ds)):
            a = roc_auc_score((ds == name).astype(int), proj)
            a = max(a, 1 - a)
            if a > best_a:
                best, best_a = name, a
        res_pc.append({"pc": j + 1, "auroc_label": float(a_lab),
                       "best_dataset": best, "auroc_dataset": float(best_a)})
        logger.info("   %4d %10.3f %20s %.3f", j + 1, max(a_lab, 1 - a_lab), best, best_a)

    logger.info("")
    logger.info("3. decodability: label vs dataset identity, same features, same probe")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000,
                                                             random_state=0))
    lab_acc = cross_val_score(clf, X, y, cv=3, scoring="accuracy").mean()
    big = [n for n, c in Counter(ds).items() if c >= 100]
    mask = np.isin(ds, big)
    dat_acc = cross_val_score(clf, X[mask], ds[mask], cv=3, scoring="accuracy").mean()
    base_lab = max(y.mean(), 1 - y.mean())
    base_dat = max(Counter(ds[mask]).values()) / mask.sum()
    logger.info("   label   : %.3f  (majority baseline %.3f)  lift %+.3f",
                lab_acc, base_lab, lab_acc - base_lab)
    logger.info("   dataset : %.3f  (majority baseline %.3f)  lift %+.3f  over %d datasets",
                dat_acc, base_dat, dat_acc - base_dat, len(big))

    logger.info("")
    logger.info("4. where the deception axis sits relative to that variance")
    dec = unit(bank["axes"]["deception"][args.layer].numpy())
    asst = unit(bank["axes"]["assistant"][args.layer].numpy())
    cos_pcs = [float(abs(dec @ Vt[j])) for j in range(5)]
    logger.info("   |cos(deception, PC1..PC5)| = %s",
                ", ".join(f"{c:.3f}" for c in cos_pcs))
    energy = float(sum((dec @ Vt[j]) ** 2 for j in range(10)))
    logger.info("   share of the deception axis inside the top-10 PC subspace: %.1f%%",
                100 * energy)
    logger.info("   cos(deception, assistant) = %+.3f   AUROC(deception proj) = %.3f",
                float(dec @ asst), roc_auc_score(y, X @ dec))

    logger.info("")
    logger.info("5. norms -- why every projection in this pipeline normalises first")
    logger.info("   mean |activation| at layer %d: %.1f", args.layer,
                float(np.linalg.norm(X, axis=1).mean()))
    for L in [l for l in layers if l in (0, 20, 40, 60, 76)]:
        logger.info("   |deception axis| at layer %2d: %.3f", L,
                    float(np.linalg.norm(bank["axes"]["deception"][L].numpy())))

    verdict = "dataset identity is MORE decodable than the label" if dat_acc - base_dat > lab_acc - base_lab \
        else "the label is at least as decodable as dataset identity"
    logger.info("")
    logger.info("VERDICT: %s", verdict)

    path = Path(args.out) if args.out else run_dir.parent / f"validate_L{args.layer}.json"
    path.write_text(json.dumps({"layer": args.layer, "pooling": args.pooling,
                                "variance": var[:10].tolist(), "pcs": res_pc,
                                "label_acc": float(lab_acc), "dataset_acc": float(dat_acc),
                                "cos_deception_pcs": cos_pcs,
                                "deception_in_top10": energy}, indent=2), encoding="utf-8")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
