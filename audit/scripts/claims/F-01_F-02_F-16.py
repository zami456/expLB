#!/usr/bin/env python3
"""Recompute F-01 (lexical AUROC), F-02 (cell counts) and F-16 (row counts probed)
straight from results/subsets/harm-pressure-knowledge-report.csv.

Nothing is imported from the project; the regex and the parsing rules are transcribed
from src/assistant_axis_ndif/eval_datasets.py (lines 363-364, 385-420) so the audit
reproduces them independently. Run from the repo root.
"""
from __future__ import annotations

import ast
import csv
import re
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(1 << 30)

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "results" / "subsets" / "harm-pressure-knowledge-report.csv"
CSV_MODEL = "llama-v3.3-70b-instruct"

DENIAL_RE = re.compile(r"\b(do not|don't|doesn't|not)\b[^.]{0,20}\bknow\b", re.I)
ASSERT_RE = re.compile(r"^\s*(yes\b|i know\b)", re.I)

META_ENV = {"nan": float("nan"), "inf": float("inf"), "NaN": float("nan")}
try:
    import numpy as _np
    META_ENV["np"] = _np
except ImportError:
    pass


def parse_meta(raw):
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return eval(raw, {"__builtins__": {}}, META_ENV)  # noqa: S307


def denies(text):
    return bool(DENIAL_RE.search(text or "")) and not ASSERT_RE.match(text or "")


def auroc(scores, labels):
    """Rank-based AUROC with ties at 0.5 (Mann-Whitney U)."""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    npos = sum(1 for _, l in pairs if l == 1)
    nneg = len(pairs) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    s = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (s - npos * (npos + 1) / 2.0) / (npos * nneg)


def main():
    rows = []
    with CSV.open(encoding="utf-8", newline="") as fh:
        for i, rec in enumerate(csv.DictReader(fh)):
            try:
                msgs = ast.literal_eval(rec["messages"])
            except (ValueError, SyntaxError):
                msgs = None
            try:
                meta = parse_meta(rec["meta"])
            except Exception:
                meta = None
            asst = [m for m in msgs if m.get("role") == "assistant"] if msgs else []
            raw = str(rec.get("deceptive", "")).strip().lower()
            rows.append({
                "i": i,
                "model": rec.get("model"),
                "label": 1 if raw == "true" else 0 if raw == "false" else None,
                "resp": asst[-1].get("content", "") if asst else None,
                "meta": meta,
                "sub": rec.get("dataset") or "",
            })

    print(f"CSV: {CSV}")
    print(f"total CSV rows: {len(rows)}")
    print("models:", dict(Counter(r['model'] for r in rows).most_common()))
    print()

    for scope, sel in (("ALL rows", rows),
                       ("Llama-3.3-70B rows only", [r for r in rows if r["model"] == CSV_MODEL])):
        usable = [r for r in sel if r["resp"] is not None and r["label"] is not None]
        print(f"=== {scope}: n={len(sel)}, parseable-with-label={len(usable)} ===")
        lab = [r["label"] for r in usable]
        den = [1 if denies(r["resp"]) else 0 for r in usable]
        a = auroc(den, lab)
        npos, nneg = sum(lab), len(lab) - sum(lab)
        tp = sum(1 for d, l in zip(den, lab) if d == 1 and l == 1)
        fn = npos - tp
        fp = sum(1 for d, l in zip(den, lab) if d == 1 and l == 0)
        tn = nneg - fp
        print(f"  deceptive={npos}  honest={nneg}  positive-rate={npos/len(lab):.4f}")
        print(f"  denial-regex AUROC = {a:.4f}   (TPR={tp/npos:.4f}, FPR={fp/nneg:.4f}, "
              f"balanced-acc={(tp/npos + tn/nneg)/2:.4f})")
        print(f"  deceptive rows the regex MISSES: {fn}")
        # F-02 cell counts
        cells = Counter()
        nometa = 0
        for r in usable:
            if r["meta"] is None:
                nometa += 1
                continue
            knows = r["meta"].get("model_belief", -1) != -1
            d = denies(r["resp"])
            cells[(knows, d, r["label"])] += 1
        print(f"  meta unparseable: {nometa}")
        print("  cell counts (knows, denies, deceptive) -> n:")
        for k in sorted(cells, key=lambda t: (t[0], t[1])):
            print(f"    knows={k[0]!s:<5} denies={k[1]!s:<5} deceptive={'deceptive' if k[2] else 'honest':<9} n={cells[k]}")
        missing = [(kn, dn) for kn in (False, True) for dn in (False, True)
                   if not any(k[0] == kn and k[1] == dn for k in cells)]
        print(f"  empty (knows,denies) cells: {missing}")
        print()

    # F-16 slice counts
    llama = [r for r in rows if r["model"] == CSV_MODEL]
    usable = [r for r in llama if r["resp"] is not None and r["label"] is not None and r["meta"] is not None]
    tm = [r for r in usable if denies(r["resp"])]
    km = [r for r in usable if r["meta"].get("model_belief", -1) != -1]
    print("=== F-16 slice counts (Llama rows) ===")
    print(f"  text_matched  n={len(tm)}  deceptive={sum(r['label'] for r in tm)}  honest={len(tm)-sum(r['label'] for r in tm)}")
    print(f"  knowledge_matched n={len(km)}")
    print(f"  items with consistent belief (model_belief != -1): {len(km)}")
    knows = sum(1 for r in usable if r["meta"].get("model_belief", -1) != -1)
    notknows = sum(1 for r in usable if r["meta"].get("model_belief", -1) == -1)
    print(f"  rows: knows={knows}  not-knows={notknows}")
    # 'items' = unique underlying question ids, as load_csv_subset builds them
    items = {}
    for r in usable:
        idx = r["sub"] + ":" + str(r["i"])  # placeholder, replaced below
    # unique item ids using the dataset_index/index column
    with CSV.open(encoding="utf-8", newline="") as fh:
        recs = list(csv.DictReader(fh))
    item_of = {}
    for i, rec in enumerate(recs):
        if rec.get("model") != CSV_MODEL:
            continue
        sub = rec.get("dataset") or ""
        idx = rec.get("dataset_index") or rec.get("index") or str(i)
        item_of[i] = f"{sub}:{idx}" if sub else str(idx)
    uniq_items = set(item_of.values())
    print(f"  unique item ids among Llama rows: {len(uniq_items)}  (rows {len(item_of)})")
    kn_items, nk_items = set(), set()
    for r in usable:
        it = item_of.get(r["i"])
        if r["meta"].get("model_belief", -1) != -1:
            kn_items.add(it)
        else:
            nk_items.add(it)
    print(f"  unique items knows={len(kn_items)}  not-knows={len(nk_items)}  "
          f"(items appearing in both: {len(kn_items & nk_items)})")


if __name__ == "__main__":
    main()
