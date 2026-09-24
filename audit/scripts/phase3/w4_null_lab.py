#!/usr/bin/env python
"""W4 (READ-ONLY, lab PC): the audit's central ZERO-SHOT random-direction null.

WHAT THIS IS
    A "zero-shot" claim in this project takes a FIXED direction in the model's residual stream
    (a persona contrast axis, or PC1 of a persona PCA basis), projects a dataset's pooled
    activations onto it, and reports the AUROC of that single number against the dataset's
    label.  Nothing is trained.  The right null is therefore NOT 0.5: when the two classes
    differ in mean activation at all, a RANDOM unit direction also separates them.  This
    script draws random unit directions in the same 8192-d space, on the same rows, and
    reports the whole distribution of their AUROCs.

    It is deliberately the SAME OBJECT as src/assistant_axis_ndif/25_zeroshot_nulls.py, whose
    output (run-coords/zeroshot_nulls.json) stored only p95 and max for 1,000 draws at seed 42
    on two (layer, pooling) cells.  Arm A below reproduces that draw exactly, as a check that
    this pipeline is the same pipeline.  Arm B is the audit's own, larger null.

    It is NOT the same object as the `random` / `random@2` columns of the crossgen matrices
    (audit/phase2/layer_sensitivity.csv): those are random directions used as FEATURES for a
    TRAINED probe, which can learn a sign (k=1) or a rotation (k>1).  Comparing a zero-shot
    axis to a trained random probe is the error DF-31 records.  The two never meet in this file.

SIGN CONVENTION (stated explicitly because the audit has been bitten three times)
    * signed   = roc_auc_score(y, X @ d) as-is.  0.31 means the direction points the WRONG
                 way: it is an inverted detector, not a weak one.
    * unsigned = max(a, 1-a) = 0.5 + |a - 0.5|.  A random direction's sign is arbitrary, so
                 the project's null is built on this scale.  0.5 is the floor.
    Both distributions are reported.  A claim about an INVERTED axis must be judged against the
    SIGNED LOWER tail (p01/p05), not against the unsigned upper tail.

READ-ONLY: opens only files under C:/Users/user4/expLB; writes one JSON into audit_scratch.
"""
import json, os, sys, time
import numpy as np
import torch
from scipy.stats import rankdata

R = "C:/Users/user4/expLB"
PASSB = os.path.join(R, "run-coords", "passB")
OUT = "C:/Users/user4/audit_scratch/w4_nulls.json"

LAYERS = [20, 40, 44]
POOLS = ["mean", "last"]
N_A, SEED_A = 1000, 42                # arm A: reproduce the stored draw
N_B, SEED_B = 2000, 20260923          # arm B: the audit's own null
HID = 8192

BANKS = {
    "paper16":  "run-paper16/axes_vectors_s3_fp32.pt",
    "kd":       "run-kd/axes_vectors_s3.pt",
    "kd_s23":   "run-kd/axes_vectors_s23.pt",
    "collab":   "run-kd/axes_collab_s23.pt",
    "cap_s23":  "run-cap/axes_s23.pt",
    "cap_all":  "run-cap/axes_all.pt",
    "cap_ctrl": "run-cap-ctrl/axes_s23.pt",
    "own100":   "run/axes.pt",
}

# the pooled sets, named so a claim can say which one it means
CORPUS9 = ["ai_liar", "convincing-game", "harm-pressure-choice",
           "harm-pressure-knowledge-report", "insider-trading", "instructed-deception",
           "mask", "roleplaying", "sandbagging_v2"]
SETS = {
    "pooled_corpus9_plus_alpaca": CORPUS9 + ["alpaca"],   # what zeroshot_nulls.json pooled
    "pooled_corpus9":             CORPUS9,
    "pooled_root10":              CORPUS9 + ["sycophancy_mt"],
    "pooled_kd11":                CORPUS9 + ["harm-pressure-kr-matched", "sycophancy_mt"],
}


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def aurocs_columns(S, y):
    """S (n, M) -> (M,) AUROC of each column, ties averaged (same as 25_zeroshot_nulls.py)."""
    r = rankdata(S, axis=0)
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---------------------------------------------------------------- directions
dir_names, dir_vecs = [], []          # (bank, axis, kind, layer) -> row of M
for bname, rel in BANKS.items():
    p = os.path.join(R, rel)
    if not os.path.exists(p):
        print("  BANK MISSING %s %s" % (bname, p), flush=True); continue
    b = torch.load(p, map_location="cpu", weights_only=False)
    ax = b.get("axes", {}); pca = b.get("pca", {})
    for name in ax:
        for L in LAYERS:
            try:
                v = unit(ax[name][L].float().numpy()).astype(np.float32)
            except Exception:
                continue
            if v.shape[0] != HID:
                continue
            dir_names.append((bname, name, "contrast", L)); dir_vecs.append(v)
        if name in pca:
            for L in LAYERS:
                try:
                    v = unit(pca[name]["components"][L][0].float().numpy()).astype(np.float32)
                except Exception:
                    continue
                if v.shape[0] != HID:
                    continue
                dir_names.append((bname, name, "PC1", L)); dir_vecs.append(v)
    print("  bank %-9s roles=%s axes=%s" % (bname, len(b.get("roles", [])), sorted(ax)), flush=True)
    del b

rngA = np.random.default_rng(SEED_A)
RA = np.stack([unit(v) for v in rngA.standard_normal((N_A, HID))]).astype(np.float32)
rngB = np.random.default_rng(SEED_B)
RB = np.stack([unit(v) for v in rngB.standard_normal((N_B, HID))]).astype(np.float32)

# one projection matrix: [ axis directions | arm A | arm B ]
M = np.concatenate([np.stack(dir_vecs).astype(np.float32), RA, RB], axis=0)
n_ax = len(dir_vecs)
print("directions: %d axis + %d randA + %d randB = %d" % (n_ax, N_A, N_B, M.shape[0]), flush=True)

# ---------------------------------------------------------------- project
scores, labels = {}, {}                # ds -> {(L,pool): S}, ds -> y
for ds in sorted(os.listdir(PASSB)):
    d = os.path.join(PASSB, ds)
    if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "rows.jsonl")):
        continue
    t0 = time.time()
    meta = {}
    for line in open(os.path.join(d, "rows.jsonl"), encoding="utf-8"):
        r = json.loads(line); meta[r["row_id"]] = r
    acc = {(L, p): [] for L in LAYERS for p in POOLS}
    ys = []
    for c in sorted(f for f in os.listdir(d) if f.startswith("chunk_")):
        blob = torch.load(os.path.join(d, c), weights_only=False, map_location="cpu")
        lay = list(blob["layers"]); pool = list(blob.get("pooling", ["mean", "last"]))
        keep = [i for i, rid in enumerate(blob["row_ids"])
                if meta.get(rid) is not None and meta[rid].get("label") is not None]
        if keep:
            ys.extend(int(meta[blob["row_ids"][i]]["label"]) for i in keep)
            for L in LAYERS:
                for p in POOLS:
                    X = blob["pooled"][keep][:, lay.index(L), :, pool.index(p)].float().numpy()
                    acc[(L, p)].append((X @ M.T).astype(np.float32))
        del blob
    if not ys:
        continue
    labels[ds] = np.asarray(ys, dtype=int)
    scores[ds] = {k: np.concatenate(v) for k, v in acc.items()}
    print("  %-34s n=%d pos=%d %.0fs" % (ds, len(ys), int(labels[ds].sum()), time.time() - t0),
          flush=True)


# ---------------------------------------------------------------- score
def cell(S, y):
    a = aurocs_columns(S, y)
    ax = a[:n_ax]
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
           "axes": {"%s|%s|%s|L%d" % (b, nm, kd, L): float(v)
                    for (b, nm, kd, L), v in zip(dir_names, ax)}}
    for tag, lo, hi in (("randA", n_ax, n_ax + N_A), ("randB", n_ax + N_A, n_ax + N_A + N_B)):
        r = a[lo:hi]
        u = np.maximum(r, 1 - r)
        out[tag] = {
            "n_draws": int(len(r)),
            "seed": SEED_A if tag == "randA" else SEED_B,
            "unsigned_mean": float(u.mean()),
            "unsigned_p50": float(np.percentile(u, 50)),
            "unsigned_p95": float(np.percentile(u, 95)),
            "unsigned_p99": float(np.percentile(u, 99)),
            "unsigned_max": float(u.max()),
            "signed_min": float(r.min()),
            "signed_p01": float(np.percentile(r, 1)),
            "signed_p05": float(np.percentile(r, 5)),
            "signed_p50": float(np.percentile(r, 50)),
            "signed_p95": float(np.percentile(r, 95)),
            "signed_p99": float(np.percentile(r, 99)),
            "signed_max": float(r.max()),
        }
    return out


res = {}
for L in LAYERS:
    for p in POOLS:
        key = "L%d_%s" % (L, p)
        res[key] = {"per_dataset": {}, "pooled": {}}
        for ds in scores:
            if len(np.unique(labels[ds])) == 2:
                res[key]["per_dataset"][ds] = cell(scores[ds][(L, p)], labels[ds])
        for sname, members in SETS.items():
            have = [m for m in members if m in scores]
            if len(have) != len(members):
                res[key]["pooled"][sname] = {"MISSING": [m for m in members if m not in scores]}
                continue
            S = np.concatenate([scores[m][(L, p)] for m in have])
            y = np.concatenate([labels[m] for m in have])
            c = cell(S, y); c["members"] = have
            res[key]["pooled"][sname] = c
            del S
        print("scored " + key, flush=True)

meta_out = {"layers": LAYERS, "poolings": POOLS, "hidden": HID,
            "armA": {"n_draws": N_A, "seed": SEED_A,
                     "note": "reproduces src/assistant_axis_ndif/25_zeroshot_nulls.py"},
            "armB": {"n_draws": N_B, "seed": SEED_B, "note": "the audit's own central null"},
            "draw": "np.random.default_rng(seed).standard_normal((n, 8192)), each row L2-normalised",
            "store": PASSB, "banks": BANKS, "sets": SETS,
            "sign_convention": "signed = roc_auc_score as-is; unsigned = max(a, 1-a) = 0.5+|a-0.5|"}
json.dump({"meta": meta_out, "cells": res}, open(OUT, "w", encoding="utf-8"), indent=1)
print("WROTE " + OUT, flush=True)
