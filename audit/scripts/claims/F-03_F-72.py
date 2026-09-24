#!/usr/bin/env python3
"""Recompute F-03's and F-72's HP-KR cells from the crossgen matrices.

Reader logic transcribed from kd/scripts/matched_profile.py: for a given row dataset,
`row_mean` = mean AUROC over that row, diagonal skipped ("trained on X, tested on the
other N-1"). Probe = `paper` unless overridden.

Usage:  python F-03_F-72.py <store-dir> [<store-dir> ...]
A store dir is one that holds matrices/*.json.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

FULL = "harm-pressure-knowledge-report"
MATCHED = "harm-pressure-kr-matched"
FEATS = ["axes", "raw", "random", "length", "datapca", "contrast", "pc1"]


def row_mean(entry, n, i):
    A = entry["auroc"]
    v = [A[i][j] for j in range(n) if j != i and A[i][j] is not None]
    return statistics.fmean(v) if v else float("nan")


def read(store, probe="paper"):
    mdir = Path(store) / "matrices"
    out = []
    for f in sorted(mdir.glob("*.json")):
        d = json.load(open(f, encoding="utf-8"))
        ds = d["datasets"]
        n = len(ds)
        rec = {"file": f.name, "pooling": d["pooling"], "layer": d["layer"],
               "seed": d["seed"], "bank": d.get("bank"), "bank_name": d.get("bank_name"),
               "k": d.get("k"), "n_ds": n, "datasets": ds}
        for tag, name in (("full", FULL), ("matched", MATCHED)):
            if name not in ds:
                continue
            i = ds.index(name)
            for r in d["results"]:
                if r.get("probe") == probe and r.get("feature") in FEATS:
                    rec[f"{tag}:{r['feature']}"] = row_mean(r, n, i)
                    rec[f"{tag}:{r['feature']}:diag"] = r["auroc"][i][i]
        # zero-shot entries, if present
        zs = d.get("zeroshot")
        if isinstance(zs, dict):
            rec["zeroshot_keys"] = list(zs.keys())[:6]
        out.append(rec)
    return out


def main():
    for store in sys.argv[1:]:
        recs = read(store)
        if not recs:
            print(f"=== {store}: no matrices ===\n")
            continue
        print(f"=== {store} ({len(recs)} matrices) ===")
        print(f"  bank={recs[0]['bank']} bank_name={recs[0]['bank_name']} k={recs[0]['k']} "
              f"n_datasets={recs[0]['n_ds']}")
        print(f"  datasets={recs[0]['datasets']}")
        for pooling in sorted({r["pooling"] for r in recs}):
            sel = sorted([r for r in recs if r["pooling"] == pooling], key=lambda r: r["layer"])
            print(f"\n  pooling={pooling}")
            hdr = f"  {'layer':>5} | " + " ".join(f"{('m:'+f):>10}" for f in ["axes", "raw", "random", "length", "datapca"]) \
                  + " || " + " ".join(f"{('f:'+f):>10}" for f in ["axes", "raw", "random", "length", "datapca"])
            print(hdr)
            for r in sel:
                def g(t, f):
                    v = r.get(f"{t}:{f}")
                    return f"{v:>10.3f}" if v is not None else f"{'--':>10}"
                print(f"  {r['layer']:>5} | " + " ".join(g("matched", f) for f in ["axes", "raw", "random", "length", "datapca"])
                      + " || " + " ".join(g("full", f) for f in ["axes", "raw", "random", "length", "datapca"]))
        print()


if __name__ == "__main__":
    main()
