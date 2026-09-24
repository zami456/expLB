"""Phase-2 batch P2-pooled, F-94 — runs READ-ONLY on the lab PC.

Extracts, for every stored layer and both poolings, the `sycophancy_mt` TARGET column mean of
the cap_s23 and cap_ctrl AUROC matrices, i.e. the quantity F-89/F-94 tabulate:
  holding    = cap_s23  pc:capitulation@2
  paraphrase = cap_ctrl pc:cap_vs_para@2
  neutral    = cap_ctrl pc:cap_vs_neutral@2
  none       = cap_ctrl pc:cap_only_ctrl@2
  raw        = raw
and recomputes the content effect (holding - neutral), the presence effect (holding-side six
minus none), the retention ratio, and a paired dataset-level bootstrap over the source datasets
(the unit F-94 names) so the `*` marks can be re-derived rather than read.

Usage on the lab:  python p2p_F94_lab.py [structure]
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys

import numpy as np

ROOT = r"C:\Users\user4\expLB\run-crossgen\kd11\banks"
S23 = os.path.join(ROOT, "cap_s23", "matrices")
CTRL = os.path.join(ROOT, "cap_ctrl", "matrices")
TARGET = "sycophancy_mt"
B = 10000
SEED = 42


def layers(d, pooling):
    out = []
    for fn in os.listdir(d):
        m = re.match(rf"{pooling}_L(\d+)_s42\.json$", fn)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def load(d, pooling, L):
    p = os.path.join(d, f"{pooling}_L{L}_s42.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def col(j, feature, probe="paper"):
    """Return {source: auroc} for the TARGET column of `feature`, plus the dataset list.

    A matrix file's `results` is a list of {feature, probe, auroc: N x N}; rows = trained on,
    columns = tested on. The diagonal is the row dataset's own 20% split, so it is excluded."""
    ds = j["datasets"]
    blk = None
    for r in j["results"]:
        if r.get("feature") == feature and r.get("probe") == probe:
            blk = r["auroc"]
            break
    if blk is None:
        return None, ds
    A = np.array([[np.nan if v is None else float(v) for v in row] for row in blk], dtype=float)
    ti = ds.index(TARGET)
    return {s: A[i, ti] for i, s in enumerate(ds) if i != ti}, ds


if len(sys.argv) > 1 and sys.argv[1] == "structure":
    j = load(S23, "mean", 40)
    print("top keys:", sorted(j.keys())[:40])
    for k, v in j.items():
        if isinstance(v, (str, int, float)):
            print("  ", k, "=", v)
    print("  datasets:", j["datasets"])
    print("  results entries:", [(r["feature"], r["probe"]) for r in j["results"]])
    sys.exit(0)

FEATS = {"holding": (S23, "pc:capitulation@2"),
         "paraphrase": (CTRL, "pc:cap_vs_para@2"),
         "neutral": (CTRL, "pc:cap_vs_neutral@2"),
         "none": (CTRL, "pc:cap_only_ctrl@2"),
         "raw": (S23, "raw")}

for pooling in ("mean", "last"):
    Ls = sorted(set(layers(S23, pooling)) & set(layers(CTRL, pooling)))
    print(f"\n=== pooling={pooling}  layers with BOTH banks: {len(Ls)}  {Ls}")
    print(f"  {'L':>4}{'holding':>9}{'paraphr':>9}{'neutral':>9}{'none':>9}{'raw':>9}"
          f"{'hold-neut':>11}{'six-none':>10}{'retain':>9}  sig(content/presence)")
    content, presence, retain, ORD, CLR = [], [], [], [], []
    nc = npz = 0
    for L in Ls:
        cols, sigs = {}, {}
        ok = True
        for name, (d, feat) in FEATS.items():
            j = load(d, pooling, L)
            c, ds = col(j, feat)
            if c is None:
                ok = False
                break
            cols[name] = c
        if not ok:
            print(f"  {L:>4}  feature missing")
            continue
        srcs = sorted(set.intersection(*[set(c) for c in cols.values()]))
        m = {k: statistics.fmean(cols[k][s] for s in srcs) for k in cols}
        ce = m["holding"] - m["neutral"]
        pe = m["holding"] - m["none"]
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(srcs), size=(B, len(srcs)))

        def bootp(a, b):
            d_ = np.array([cols[a][s] - cols[b][s] for s in srcs])
            bs = d_[idx].mean(axis=1)
            return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

        lo1, hi1 = bootp("holding", "neutral")
        lo2, hi2 = bootp("holding", "none")
        s1 = "*" if lo1 > 0 else " "
        s2 = "*" if lo2 > 0 else " "
        nc += lo1 > 0
        npz += lo2 > 0
        CLR.append((L, lo1 > 0, lo2 > 0))
        # F-94's retention is measured against RAW, not against `none`: at L0 it prints 24%
        # ((0.5262-0.5096)/(0.5786-0.5096)) and at L12 -278% (denominator 0.0045).
        den = m["holding"] - m["raw"]
        r = (m["neutral"] - m["raw"]) / den if den != 0 else float("nan")
        content.append(ce)
        presence.append(pe)
        retain.append(r)
        ordA = (m["paraphrase"] >= m["holding"] and m["holding"] > m["neutral"]
                and m["neutral"] > m["none"])                       # strict chain
        ordB = min(m["holding"], m["paraphrase"]) > max(m["neutral"], m["none"])  # group ladder
        ordC = m["holding"] > m["neutral"] > m["none"]              # ignoring paraphrase
        ORD.append((L, ordA, ordB, ordC))
        print(f"  {L:>4}{m['holding']:>9.4f}{m['paraphrase']:>9.4f}{m['neutral']:>9.4f}"
              f"{m['none']:>9.4f}{m['raw']:>9.4f}{ce:>+11.4f}{s1}{pe:>+9.4f}{s2}"
              f"{r * 100:>8.0f}%  [{lo1:+.4f},{hi1:+.4f}] [{lo2:+.4f},{hi2:+.4f}]")
    if content:
        print(f"  n_layers={len(content)}  content mean {statistics.fmean(content):+.4f}  "
              f"presence mean {statistics.fmean(presence):+.4f}  "
              f"ratio {statistics.fmean(presence) / statistics.fmean(content):.2f}:1")
        print(f"  retention median {statistics.median(retain) * 100:.0f}%  "
              f"range [{min(retain) * 100:.0f}%, {max(retain) * 100:.0f}%]")
        print(f"  content clears at {nc}/{len(content)} layers "
              f"{[L for L, a, b in CLR if a]}; presence at {npz}/{len(presence)} "
              f"{[L for L, a, b in CLR if b]}")
        print(f"  ordering, strict 'paraphrase>=holding>neutral>none': "
              f"{sum(1 for _, a, _, _ in ORD if a)}/{len(ORD)}  fails at "
              f"{[L for L, a, _, _ in ORD if not a]}")
        print(f"  ordering, group 'min(hold,para) > max(neut,none)':  "
              f"{sum(1 for _, _, b, _ in ORD if b)}/{len(ORD)}  fails at "
              f"{[L for L, _, b, _ in ORD if not b]}")
        print(f"  ordering, 'holding>neutral>none':                   "
              f"{sum(1 for _, _, _, c in ORD if c)}/{len(ORD)}  fails at "
              f"{[L for L, _, _, c in ORD if not c]}")
        # the 16-layer subtotals F-94's main table reports (L0..L56)
        n16 = 16
        c16, p16, r16 = content[:n16], presence[:n16], retain[:n16]
        print(f"  --- first {n16} layers (F-94's main table): content mean "
              f"{statistics.fmean(c16):+.4f} (ledger +0.0339), presence mean "
              f"{statistics.fmean(p16):+.4f} (ledger +0.0704), retention median "
              f"{statistics.median(r16) * 100:.0f}% (ledger 53%)")
        print(f"      content clears {sum(1 for L, a, b in CLR[:n16] if a)}/16 "
              f"(ledger prose says 4, its own table marks 3); presence "
              f"{sum(1 for L, a, b in CLR[:n16] if b)}/16 (ledger prose says 9, "
              f"its own table marks 10)")
