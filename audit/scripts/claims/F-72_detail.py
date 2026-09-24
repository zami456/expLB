#!/usr/bin/env python3
"""AUDIT, READ-ONLY. F-72 detail: the matched-slice row across every layer that passA11
and passC11 store, for both the kd_s23 bank (where F-72's `axes` column comes from) and
the paper16 root store (where its raw/random/length/datapca columns come from).

Prints: the six-layer table F-72 quotes, the passA11-vs-passC11 agreement check, and the
span over the FULL passA11 sweep (F-72 measured a span over six layers only).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path("C:/Users/user4/expLB")
DS = "harm-pressure-kr-matched"
BASE = ["raw", "random", "length", "datapca"]


def row_mean(entry, n, i):
    A = entry["auroc"]
    v = [A[i][j] for j in range(n) if j != i and A[i][j] is not None]
    return statistics.fmean(v) if v else float("nan")


def read(store, feats, probe="paper"):
    """-> {(pooling, layer): {feature: value}}"""
    out = {}
    for f in sorted((Path(store) / "matrices").glob("*.json")):
        d = json.load(open(f, encoding="utf-8"))
        ds = d["datasets"]
        if DS not in ds:
            continue
        i, n = ds.index(DS), len(ds)
        vals = {r["feature"]: row_mean(r, n, i)
                for r in d["results"] if r.get("probe") == probe and r["feature"] in feats}
        if vals:
            out[(d["pooling"], d["layer"])] = vals
    return out


def main():
    A_root = read(REPO / "run-crossgen/kd11/passA11", BASE)
    C_root = read(REPO / "run-crossgen/kd11/passC11", BASE)
    A_kd = read(REPO / "run-crossgen/kd11/passA11/banks/kd_s23", ["axes"])
    C_kd = read(REPO / "run-crossgen/kd11/passC11/banks/kd_s23", ["axes"])
    A_p16 = read(REPO / "run-crossgen/kd11/passA11", ["axes"])
    C_p16 = read(REPO / "run-crossgen/kd11/passC11", ["axes"])
    B_root = read(REPO / "run-crossgen/kd11", BASE + ["axes"])   # F-03's store (passB)

    print("### F-72's six-layer table, mean pooling, probe=paper, from passC11")
    print(f"{'layer':>5} {'axes(kd_s23)':>13} {'axes(paper16)':>14} {'raw':>7} {'random':>7} "
          f"{'length':>7} {'datapca':>8} {'axes-raw':>9} {'axes-datapca':>13}")
    for L in (20, 22, 28, 40, 44, 48):
        k = ("mean", L)
        a = C_kd.get(k, {}).get("axes")
        p = C_p16.get(k, {}).get("axes")
        b = C_root.get(k, {})
        print(f"{L:>5} {a:>13.3f} {p:>14.3f} {b['raw']:>7.3f} {b['random']:>7.3f} "
              f"{b['length']:>7.3f} {b['datapca']:>8.3f} {a-b['raw']:>9.3f} {a-b['datapca']:>13.3f}")
    six = [C_kd[("mean", L)]["axes"] for L in (20, 22, 28, 40, 44, 48)]
    print(f"  six-layer span (kd_s23, mean) = {max(six)-min(six):.3f}   peak {max(six):.3f}  floor {min(six):.3f}")
    six_p = [C_p16[("mean", L)]["axes"] for L in (20, 22, 28, 40, 44, 48)]
    print(f"  six-layer span (paper16, mean) = {max(six_p)-min(six_p):.3f}  peak {max(six_p):.3f}  floor {min(six_p):.3f}")
    beats_raw = [L for L in (20, 22, 28, 40, 44, 48) if C_kd[("mean", L)]["axes"] > C_root[("mean", L)]["raw"]]
    beats_pca = [L for L in (20, 22, 28, 40, 44, 48) if C_kd[("mean", L)]["axes"] > C_root[("mean", L)]["datapca"]]
    print(f"  axes(kd_s23) beats raw at {len(beats_raw)}/6 layers {beats_raw}; "
          f"beats datapca at {len(beats_pca)}/6 {beats_pca}")
    print()

    print("### passA11 vs passC11, same cell (axes, kd_s23 bank)")
    print(f"{'pooling':>7} {'layer':>5} {'passA11':>8} {'passC11':>8} {'diff':>8}")
    for pool in ("mean", "last"):
        for L in (20, 40, 44):
            a = A_kd.get((pool, L), {}).get("axes")
            c = C_kd.get((pool, L), {}).get("axes")
            if a is None or c is None:
                print(f"{pool:>7} {L:>5} {'--':>8} {'--':>8}")
            else:
                print(f"{pool:>7} {L:>5} {a:>8.3f} {c:>8.3f} {a-c:>8.4f}")
    print()

    print("### FULL passA11 sweep, mean pooling, axes(kd_s23): the span F-72 could not measure")
    for pool in ("mean", "last"):
        sel = sorted([(L, v["axes"]) for (p, L), v in A_kd.items() if p == pool])
        if not sel:
            continue
        vals = [v for _, v in sel]
        peak = max(sel, key=lambda t: t[1]); floor = min(sel, key=lambda t: t[1])
        print(f"  pooling={pool}: {len(sel)} layers  peak L{peak[0]} {peak[1]:.3f}  "
              f"floor L{floor[0]} {floor[1]:.3f}  SPAN {peak[1]-floor[1]:.3f}  "
              f"median {statistics.median(vals):.3f}")
        raws = {L: A_root[(pool, L)]["raw"] for L, _ in sel if (pool, L) in A_root}
        pcas = {L: A_root[(pool, L)]["datapca"] for L, _ in sel if (pool, L) in A_root}
        br = [L for L, v in sel if L in raws and v > raws[L]]
        bp = [L for L, v in sel if L in pcas and v > pcas[L]]
        print(f"    beats raw at {len(br)}/{len(raws)} layers; beats datapca at {len(bp)}/{len(pcas)}")
        print(f"    length flat? distinct length values = "
              f"{sorted({round(A_root[(pool,L)]['length'],4) for L,_ in sel if (pool,L) in A_root})}")
    print()

    print("### F-03's own store (run-crossgen/kd11 root = passB), mean L40, paper16 bank")
    b = B_root.get(("mean", 40), {})
    print("   ", {k: round(v, 4) for k, v in b.items()})
    print("    axes - raw =", round(b["axes"] - b["raw"], 4),
          " axes - datapca =", round(b["axes"] - b["datapca"], 4))


if __name__ == "__main__":
    main()
