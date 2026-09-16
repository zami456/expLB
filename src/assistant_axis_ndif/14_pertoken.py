#!/usr/bin/env python
"""Q8: does scoring tokens individually beat pooling them first? And where does the signal sit?

Pass C keeps the response tokens unpooled, so a direction can be applied to every token and
the per-token scores aggregated afterwards — Apollo's design, and the strongest competing
detector, because a lie localised to a few tokens is diluted by a mean over the whole response.
Pooled vectors cannot answer this; that is the entire reason Pass C exists.

Already known from Passes A and B: last-token pooling beats mean pooling everywhere
(0.766 against 0.731 at layer 44), which is the paper's own untested speculation confirmed.
If one final token beats the average of the response, the signal is not spread evenly — this
script asks where it actually is.

    python 14_pertoken.py --run_dir run-coords/passC

Streams one chunk at a time and keeps only scalars. The store is 28 GB; the scores are one
float per token, so nothing needs to be held.
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


def aggregate(scores: np.ndarray) -> dict:
    """Per-token scores -> one number per row, several ways.

    `max` and `topk` are the aggregations that can beat pooling: a mean over the response is
    exactly what dilutes a short burst of deceptive text, so if the signal is localised they
    should pull ahead of `mean`, and if it is diffuse they should not.
    """
    n = len(scores)
    k = max(1, int(round(0.1 * n)))
    srt = np.sort(scores)
    return {
        "mean": float(scores.mean()),
        "max": float(scores.max()),
        "last": float(scores[-1]),
        "first": float(scores[0]),
        "top10pct": float(srt[-k:].mean()),
        "last8": float(scores[-8:].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=str(REPO_ROOT / "run-coords" / "passC"))
    ap.add_argument("--bank", default=str(REPO_ROOT / "run-paper16" / "axes_vectors_s3_fp32.pt"))
    ap.add_argument("--axis", default="deception")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    keys = sorted(d.name for d in run_dir.iterdir()
                  if d.is_dir() and (d / "rows.jsonl").exists())

    per_layer_rows: dict = {}      # layer -> list of (agg dict, label, dataset)
    decile: dict = {}              # layer -> (sum, count) over relative position, by label
    layers = None

    for key in keys:
        meta = {}
        with (run_dir / key / "rows.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                meta[r["row_id"]] = r
        for c in sorted((run_dir / key).glob("chunk_*.pt")):
            blob = torch.load(c, weights_only=False)
            if "tokens" not in blob:
                continue
            if layers is None:
                layers = list(blob["layers"])
                for L in layers:
                    per_layer_rows[L] = []
                    decile[L] = {0: np.zeros((10, 2)), 1: np.zeros((10, 2))}
            tk, offs = blob["tokens"], blob["tok_offsets"]
            for i, rid in enumerate(blob["row_ids"]):
                m = meta.get(rid)
                if m is None or m.get("label") is None:
                    continue
                seg = tk[offs[i]:offs[i + 1]].to(torch.float32).numpy()   # (n_tok, K, H)
                if seg.shape[0] == 0:
                    continue
                for li, L in enumerate(layers):
                    d = unit(bank["axes"][args.axis][L].numpy())
                    s = seg[:, li, :] @ d                                  # (n_tok,)
                    per_layer_rows[L].append((aggregate(s), m["label"], key))
                    # where in the response does the score sit?
                    pos = np.minimum((np.arange(len(s)) / max(len(s) - 1, 1) * 10).astype(int), 9)
                    for b in range(10):
                        sel = s[pos == b]
                        if len(sel):
                            decile[L][m["label"]][b, 0] += sel.sum()
                            decile[L][m["label"]][b, 1] += len(sel)
        logger.info("  %-32s done", key)

    if not layers:
        raise SystemExit("no per-token data found — was --per_token used?")

    aggs = ["mean", "last", "last8", "max", "top10pct", "first"]
    logger.info("")
    logger.info("Q8 — pooled AUROC over the whole corpus, by aggregation (zero-shot, no training)")
    logger.info("  %6s" % "layer" + "".join(f"{a:>11}" for a in aggs))
    out = []
    for L in layers:
        rows = per_layer_rows[L]
        y = np.array([lab for _, lab, _ in rows])
        cells, rec = [], {"layer": int(L)}
        for a in aggs:
            x = np.array([agg[a] for agg, _, _ in rows])
            v = float(roc_auc_score(y, x)) if len(np.unique(y)) > 1 else float("nan")
            rec[a] = v
            cells.append(f"{v:>11.3f}")
        out.append(rec)
        logger.info("  %6d" % L + "".join(cells))

    best = max(out, key=lambda r: max(r[a] for a in aggs))
    ba = max(aggs, key=lambda a: best[a])
    logger.info("")
    logger.info("  best: %s at layer %d = %.3f", ba, best["layer"], best[ba])
    logger.info("  (mean pooling at the same layer = %.3f)", best["mean"])

    logger.info("")
    logger.info("localisation — mean projection by position in the response, layer %d", best["layer"])
    L = best["layer"]
    logger.info("  %8s" + "".join(f"{i*10:>6}%" for i in range(10)), "decile")
    for lab, name in ((1, "deceptive"), (0, "honest")):
        d = decile[L][lab]
        vals = [d[b, 0] / d[b, 1] if d[b, 1] else float("nan") for b in range(10)]
        logger.info("  %8s" % name + "".join(f"{v:>7.2f}" for v in vals))

    path = Path(args.out) if args.out else run_dir.parent / "pertoken_q8.json"
    path.write_text(json.dumps({"axis": args.axis, "bank": Path(args.bank).name,
                                "aggregations": out,
                                "localisation": {str(L): {str(k): decile[L][k].tolist()
                                                          for k in (0, 1)} for L in layers}},
                               indent=2), encoding="utf-8")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
