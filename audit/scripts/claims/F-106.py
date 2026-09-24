#!/usr/bin/env python
"""M1 claim check F-106 -- the sarcasm Gate-3 trivial-feature floors.

Level 1: read cap/run-sarcasm-natural/baselines.json and compare with kd/docs/FINDINGS.md F-106.
Level 2: independent re-implementation of both baselines from the raw corpus files
         (figlang2020_reddit.jsonl, 4,400 rows) and from the prepared 600-row matched set
         (rows.jsonl), plus the corpus facts the entry asserts (balance, n, mean words,
         which class is shorter) and a shuffled-CV robustness variant.

Read-only.  Run:  python audit/scripts/claims/F-106.py [--twitter <path to twitter jsonl>]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "cap" / "run-sarcasm-natural"
WORD = re.compile(r"[a-z']+")


def bow(texts, min_df=3):
    vocab = defaultdict(int)
    toks = []
    for t in texts:
        w = WORD.findall(t.lower())
        toks.append(w)
        for x in set(w):
            vocab[x] += 1
    keep = sorted([w for w, c in vocab.items() if c >= min_df])
    idx = {w: i for i, w in enumerate(keep)}
    X = np.zeros((len(toks), len(keep)), dtype=np.float32)
    for i, w in enumerate(toks):
        for x in w:
            if x in idx:
                X[i, idx[x]] += 1
    return X, keep


def auroc_pair(texts, y, cv):
    lens = np.array([len(t.split()) for t in texts], dtype=float)
    X, keep = bow(texts)
    p = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), X, y, cv=cv,
                          method="decision_function")
    return {"n": int(len(y)), "vocab": len(keep),
            "auroc_length_neg": float(roc_auc_score(y, -lens)),
            "auroc_length_pos": float(roc_auc_score(y, lens)),
            "auroc_bow": float(roc_auc_score(y, p)),
            "mean_words_pos": float(lens[y == 1].mean()),
            "mean_words_neg": float(lens[y == 0].mean()),
            "median_words_pos": float(np.median(lens[y == 1])),
            "median_words_neg": float(np.median(lens[y == 0]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twitter", default=None)
    args = ap.parse_args()

    print("=" * 78)
    print("LEVEL 1 -- cap/run-sarcasm-natural/baselines.json")
    print("=" * 78)
    b = json.loads((D / "baselines.json").read_text(encoding="utf-8"))
    print(json.dumps(b, indent=1))
    print()
    stated = {"unmatched.auroc_length": 0.5781, "unmatched.auroc_bow": 0.6778,
              "matched.auroc_length": 0.5000, "matched.auroc_bow": 0.6320,
              "unmatched.n": 4400, "matched.n": 600}
    for k, v in stated.items():
        sec, fld = k.split(".")
        got = b[sec][fld]
        print(f"  FINDINGS says {k:28} = {v:<8}   file = {got!r}   "
              f"{'MATCH' if abs(got - v) < 5e-5 else 'DIFF'}")

    print()
    print("=" * 78)
    print("LEVEL 2 -- recomputed from the raw corpus, own implementation")
    print("=" * 78)
    raw = [json.loads(l) for l in (D / "figlang2020_reddit.jsonl").open(encoding="utf-8")]
    print(f"figlang2020_reddit.jsonl rows      : {len(raw)}")
    print(f"label counts                       : {Counter(r['label'] for r in raw)}")
    print(f"keys on a row                      : {sorted(raw[0])}")
    ctx = [len(r.get('context') or []) for r in raw]
    print(f"context turns  min/median/max      : {min(ctx)}/{np.median(ctx)}/{max(ctx)}")

    y = np.array([1 if r["label"] == "SARCASM" else 0 for r in raw])
    texts = [r["response"] for r in raw]
    cv5 = StratifiedKFold(n_splits=5)           # same as cross_val_predict's default for ints
    un = auroc_pair(texts, y, cv5)
    print("\nUNMATCHED (all 4,400 reddit rows):")
    for k, v in un.items():
        print(f"   {k:20} {v}")

    rows = [json.loads(l) for l in (D / "rows.jsonl").open(encoding="utf-8")]
    ym = np.array([r["label"] for r in rows])
    tm = [r["response_str"] for r in rows]
    mt = auroc_pair(tm, ym, cv5)
    print(f"\nMATCHED set rows.jsonl: {len(rows)}  label counts {Counter(ym.tolist())}")
    for k, v in mt.items():
        print(f"   {k:20} {v}")
    nw = np.array([r["n_words"] for r in rows])
    print(f"   stored n_words mean pos/neg    {nw[ym == 1].mean():.4f} / {nw[ym == 0].mean():.4f}")
    gaps = np.abs(np.sort(nw[ym == 1]) - np.sort(nw[ym == 0]))
    print(f"   |word gap| within sorted pairs  max {gaps.max()}  mean {gaps.mean():.3f}")
    fl = {r["floor_auroc_bow"] for r in rows}
    print(f"   floor_auroc_bow stored on rows  {fl}")

    print("\nROBUSTNESS -- shuffled 5-fold (seed 0) instead of the sequential default:")
    cvs = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    print(f"   unmatched bow  {auroc_pair(texts, y, cvs)['auroc_bow']:.4f}")
    print(f"   matched   bow  {auroc_pair(tm, ym, cvs)['auroc_bow']:.4f}")

    print("\nROBUSTNESS -- min_df sensitivity on the matched set (sequential 5-fold):")
    for mdf in (1, 2, 3, 5, 10):
        X, keep = bow(tm, min_df=mdf)
        p = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), X, ym, cv=cv5,
                              method="decision_function")
        print(f"   min_df={mdf:<3} vocab {len(keep):<6} bow AUROC {roc_auc_score(ym, p):.4f}")

    if args.twitter:
        tw = [json.loads(l) for l in Path(args.twitter).open(encoding="utf-8")]
        ytw = np.array([1 if r["label"] == "SARCASM" else 0 for r in tw])
        res = auroc_pair([r["response"] for r in tw], ytw, cv5)
        print("\nTWITTER arm (entry: n 5,000, length 0.6252, bow 0.7826):")
        for k, v in res.items():
            print(f"   {k:20} {v}")


if __name__ == "__main__":
    main()
