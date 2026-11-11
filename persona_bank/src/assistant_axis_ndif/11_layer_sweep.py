#!/usr/bin/env python
"""The layer curve: where do persona coordinates beat a raw-activation probe, and where not?

Pass B answered Q2 at three layers and got two different answers -- persona coordinates lose
to the raw 8192-d probe at layers 20 and 22 and win at 40, the paper's own midpoint. Three
points cannot say whether that is a crossover, a bump, or noise. Pass A stores all 80 layers
for ~2,080 label-stratified rows, so this draws the whole curve.

    python 11_layer_sweep.py --run_dir run-coords/passA --bank run-paper16/axes_vectors_s3_fp32.pt

Loads the store ONCE and holds it as fp16 (2,080 x 80 x 8192 x 2 bytes is ~2.7 GB), casting
one layer at a time to fp32. Running 9_persona_probes.py per layer would re-read all 83 chunk
files eighty times over.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]


def _probes_module():
    """Reuse 9_persona_probes' feature/probe/split code rather than restating it.

    Safe to importlib here: nothing in this file goes near a trace body, so the
    "Module importlib is not whitelisted" rejection does not apply -- that only bites code
    nnsight has to serialise and send to NDIF.
    """
    spec = importlib.util.spec_from_file_location("probes", PIPELINE_DIR / "9_persona_probes.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_all_layers(run_dir: Path, key: str, pooling: str):
    """-> X (n, K, H) fp16, meta list, layers. Every layer at once, one pass over the chunks."""
    d = run_dir / key
    rows_path, chunks = d / "rows.jsonl", sorted(d.glob("chunk_*.pt"))
    if not rows_path.exists() or not chunks:
        return None, None, None
    meta = {}
    with rows_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            meta[r["row_id"]] = r
    xs, keep, layers, pool_idx = [], [], None, None
    for c in chunks:
        blob = torch.load(c, weights_only=False)
        if layers is None:
            layers = list(blob["layers"])
            pools = list(blob.get("pooling", ["mean", "last"]))
            pool_idx = pools.index(pooling)
        pooled = blob["pooled"][:, :, :, pool_idx]          # (n, K, H) fp16
        for i, rid in enumerate(blob["row_ids"]):
            m = meta.get(rid)
            if m is None or m.get("label") is None:
                continue
            xs.append(pooled[i])
            keep.append(m)
    if not xs:
        return None, None, None
    return torch.stack(xs), keep, layers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=str(REPO_ROOT / "run-coords" / "passA"))
    ap.add_argument("--bank", default=str(REPO_ROOT / "run-paper16" / "axes_vectors_s3_fp32.pt"))
    ap.add_argument("--pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--features", nargs="+", default=["raw", "pc:deception", "axes"])
    ap.add_argument("--probe", default="lr_reg10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--every", type=int, default=1, help="stride over layers")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    P = _probes_module()
    run_dir = Path(args.run_dir)
    keys = sorted(d.name for d in run_dir.iterdir()
                  if d.is_dir() and (d / "rows.jsonl").exists())
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)

    logger.info("loading all layers once, pooling=%s", args.pooling)
    data = {}
    layers = None
    for k in keys:
        X, meta, L = load_all_layers(run_dir, k, args.pooling)
        if X is None:
            continue
        layers = L
        y = np.array([m["label"] for m in meta], dtype=int)
        data[k] = (X, y, meta)
        logger.info("  %-32s n=%-6d %dD/%dH", k, len(y), int(y.sum()), int((1 - y).sum()))
    if not data:
        raise SystemExit("no Pass A data found")
    src = [k for k, (_, y, _) in data.items()
           if k != P.CONTROL and len(np.unique(y)) == 2]
    logger.info("training sources: %s", ", ".join(src))
    logger.info("layers stored: %d", len(layers))

    sweep = []
    logger.info("%6s" % "layer" + "".join(f"{f:>16}" for f in args.features))
    for li, layer in enumerate(layers):
        if li % args.every:
            continue
        row = {"layer": int(layer)}
        cells = []
        for fkind in args.features:
            mat = {}
            for s in src:
                Xs16, ys, ms = data[s]
                Xs = Xs16[:, li, :].to(torch.float32).numpy()
                tr, _ = P.group_split(ms, 0.75, args.seed)
                tr = P.balance(tr, ys, args.seed)
                if len(tr) < 8:
                    continue
                F = P.make_features(Xs, fkind, bank, layer, args.k, args.seed)
                clf = P.make_probe(args.probe, args.seed)
                clf.fit(F[tr], ys[tr])
                for t in data:
                    if t == s:
                        continue
                    Xt16, yt, _ = data[t]
                    te = P.balance(np.arange(len(yt)), yt, args.seed + 1)
                    if len(te) < 8:
                        continue
                    Ft = P.make_features(Xt16[:, li, :].to(torch.float32).numpy(),
                                         fkind, bank, layer, args.k, args.seed)
                    mat[(s, t)] = P.auroc(clf, Ft[te], yt[te])
            vals = [v for v in mat.values() if not np.isnan(v)]
            row[fkind] = float(np.mean(vals)) if vals else float("nan")
            cells.append(f"{row[fkind]:>16.3f}")
        sweep.append(row)
        logger.info("%6d" % layer + "".join(cells))

    # where does the persona feature overtake raw?
    if "raw" in args.features:
        best_persona = [f for f in args.features if f != "raw"]
        for f in best_persona:
            wins = [r["layer"] for r in sweep if r.get(f, 0) > r.get("raw", 1)]
            if wins:
                logger.info("%s beats raw at %d of %d layers: %s ... %s",
                            f, len(wins), len(sweep), wins[:6], wins[-3:])
            else:
                logger.info("%s never beats raw at any layer", f)
        peak = max(sweep, key=lambda r: max(r.get(f, 0) for f in best_persona))
        logger.info("best persona layer: %d", peak["layer"])
        peak_raw = max(sweep, key=lambda r: r.get("raw", 0))
        logger.info("best raw layer:     %d (%.3f)", peak_raw["layer"], peak_raw["raw"])

    out = Path(args.out) if args.out else run_dir.parent / f"layer_sweep_{args.pooling}.json"
    out.write_text(json.dumps({"pooling": args.pooling, "k": args.k, "probe": args.probe,
                               "bank": Path(args.bank).name, "sweep": sweep}, indent=2),
                   encoding="utf-8")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
