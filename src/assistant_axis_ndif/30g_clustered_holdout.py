#!/usr/bin/env python
"""Leave-one-CLUSTER-out, the control F-79/F-80 has been missing.

THE OBJECTION. Claim 3 rests on leave-one-DATASET-out: train on the pooled rows of ten
datasets, test on the eleventh, and beat single-source transfer by +0.178. That treats the
eleven as eleven independent draws. They are not. They come from four provenances, and inside
one of them the overlap is total:

    harm-pressure-kr-matched is a VIEW over harm-pressure-knowledge-report -- 258 rows, and
    100% of its row_ids are HP-KR row_ids (measured, not assumed).

So two of the eleven LODO cells are contaminated by construction. When kr-matched is the held-out
target, every one of its 258 rows is in the training pool; when HP-KR is the target, 258 of its
2,139 rows (12%) are. That is train-on-test, not shared provenance, and it is the strongest
version of the reviewer's objection recorded in the briefing's section 5d.

WHAT THIS SCRIPT DOES. Re-runs the pooled-vs-single comparison three ways on identical splits:

    lodo          leave one DATASET out              (what F-79/F-80 report)
    lodo_clean    same, but the two leaked targets dropped from the reported mean
    loco          leave one CLUSTER out -- every dataset sharing the target's provenance is
                  removed from training, so no source material is on both sides

CLUSTERS are provenance, not topic. Datasets built from the same upstream material by the same
pipeline go together, because that is the thing that could leak: shared prompts, shared
generator, shared judge.

A NOTE ON WHAT A NEGATIVE RESULT WOULD MEAN. loco is strictly harder than lodo -- it trains on
fewer datasets AND removes the nearest neighbours. Some drop is expected and is not evidence of
contamination; the question is whether the +0.178 SURVIVES as a positive margin over
single-source transfer, which is the claim actually being made.

    python src/assistant_axis_ndif/30g_clustered_holdout.py --layer 40

Writes <out_root>/clustered_L<layer>_<pooling>_s<seed>.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]

# Provenance clusters. The harm-pressure trio is one cluster on the strongest grounds: all three
# are WMDP-derived, and kr-matched is literally a subset of HP-KR's rows. The Liars' Bench
# scenarios are a second cluster -- same repo, same generation pipeline, same judge -- though
# their source material differs, so this is the conservative call rather than the obvious one.
CLUSTERS = {
    "lb_harm": ["harm-pressure-choice", "harm-pressure-knowledge-report",
                "harm-pressure-kr-matched"],
    "lb_scenario": ["instructed-deception", "convincing-game", "insider-trading"],
    "apollo": ["ai_liar", "roleplaying", "sandbagging_v2"],
    "mask": ["mask"],
    "ours": ["sycophancy_mt"],
}
# targets whose LODO cell is contaminated by the row overlap above
LEAKED = {"harm-pressure-kr-matched", "harm-pressure-knowledge-report"}


def module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PIPELINE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default="run-coords/passB")
    ap.add_argument("--out_root", default="run-premise/clustered")
    ap.add_argument("--datasets", nargs="+", default=[d for c in CLUSTERS.values() for d in c])
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--probe", default="paper")
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    P = module("persona_probes", "9_persona_probes.py")

    DATASETS = list(args.datasets)
    data = P.build(Path(args.run_dir), DATASETS, args.layer, args.pooling)
    DATASETS = [k for k in DATASETS if k in data and len(np.unique(data[k][1])) == 2]
    X32 = {k: np.asarray(data[k][0], dtype=np.float32) for k in DATASETS}
    cluster_of = {d: c for c, ds in CLUSTERS.items() for d in ds}
    logger.info("loaded %d datasets, %d clusters", len(DATASETS),
                len({cluster_of[d] for d in DATASETS}))

    rng = np.random.default_rng(args.seed)
    splits = {}
    for k in DATASETS:
        n = len(data[k][1])
        perm = rng.permutation(n)
        splits[k] = (perm[:int(args.frac * n)], perm[int(args.frac * n):])

    def fit_on(rows_by_ds):
        Xs, ys = [], []
        for k, rows in rows_by_ds.items():
            yk = data[k][1]
            tr = P.balance(rows, yk, args.seed)
            Xs.append(X32[k][tr]); ys.append(yk[tr])
        return P.make_probe(args.probe, args.seed).fit(np.vstack(Xs), np.concatenate(ys))

    singles = {s: fit_on({s: splits[s][0]}) for s in DATASETS}

    res = {"layer": args.layer, "pooling": args.pooling, "probe": args.probe,
           "seed": args.seed, "clusters": CLUSTERS, "leaked": sorted(LEAKED),
           "per_dataset": {}}
    for t in DATASETS:
        te = splits[t][1]
        yt = data[t][1][te]
        if len(set(yt.tolist())) < 2:
            continue
        row = {"cluster": cluster_of[t]}
        row["single_offdiag"] = float(np.mean(
            [P.auroc(singles[s], X32[t][te], yt) for s in DATASETS if s != t]))
        # leave-one-DATASET-out: every other dataset, including same-provenance siblings
        lodo = {k: splits[k][0] for k in DATASETS if k != t}
        row["pooled_lodo"] = float(P.auroc(fit_on(lodo), X32[t][te], yt))
        row["n_lodo_datasets"] = len(lodo)
        # leave-one-CLUSTER-out: drop every dataset sharing the target's provenance
        loco = {k: splits[k][0] for k in DATASETS if cluster_of[k] != cluster_of[t]}
        row["pooled_loco"] = float(P.auroc(fit_on(loco), X32[t][te], yt))
        row["n_loco_datasets"] = len(loco)
        res["per_dataset"][t] = row
        logger.info("%-34s single %.4f  lodo %.4f  loco %.4f  (%d->%d datasets)",
                    t, row["single_offdiag"], row["pooled_lodo"], row["pooled_loco"],
                    len(lodo), len(loco))

    def agg(keys):
        g = [res["per_dataset"][k]["pooled_lodo"] - res["per_dataset"][k]["single_offdiag"]
             for k in keys]
        gc = [res["per_dataset"][k]["pooled_loco"] - res["per_dataset"][k]["single_offdiag"]
              for k in keys]
        return {"n": len(keys), "gain_lodo": float(np.mean(g)), "gain_loco": float(np.mean(gc)),
                "loco_positive": int(sum(1 for x in gc if x > 0))}

    allk = list(res["per_dataset"])
    clean = [k for k in allk if k not in LEAKED]
    res["summary"] = {"all": agg(allk), "clean": agg(clean)}
    logger.info("ALL %d  gain lodo %+.4f  gain loco %+.4f  (loco positive %d/%d)",
                res["summary"]["all"]["n"], res["summary"]["all"]["gain_lodo"],
                res["summary"]["all"]["gain_loco"], res["summary"]["all"]["loco_positive"],
                res["summary"]["all"]["n"])
    logger.info("CLEAN %d gain lodo %+.4f  gain loco %+.4f  (loco positive %d/%d)",
                res["summary"]["clean"]["n"], res["summary"]["clean"]["gain_lodo"],
                res["summary"]["clean"]["gain_loco"], res["summary"]["clean"]["loco_positive"],
                res["summary"]["clean"]["n"])

    f = out / f"clustered_L{args.layer}_{args.pooling}_s{args.seed}.json"
    f.write_text(json.dumps(res, indent=1), encoding="utf-8")
    logger.info("wrote %s", f)


if __name__ == "__main__":
    main()
