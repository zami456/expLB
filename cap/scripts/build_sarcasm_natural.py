#!/usr/bin/env python
"""Build the natural-sarcasm eval set for SARCASM_DESIGN.md gate 3.

WHAT GATE 3 ASKS. Apply our EXISTING deception axes to marked natural sarcasm. If they do not
fire even there, "the deception axis is really a mismatch detector" is already weak and the
sarcasm bank stays unbuilt. This script only prepares the rows; the axes are applied later.

WHY FIGLANG 2020 AND NOT iSarcasmEval, WHICH THE DESIGN NAMES.
  - `arbml/iSarcasmEval_task_A` on the Hub is the ARABIC subtask (dialect in nile/msa/gulf/
    levant/magreb, non-Latin script). Our axes and corpora are English.
  - `CreativeLang/SARC_Sarcasm` is 12.7M rows with the sarcasm LABEL COLUMN ABSENT.
  - FigLang 2020 (Ghosh, Vajpayee & Muresan, 2nd Workshop on Figurative Language Processing,
    ACL 2020) is English, exactly balanced, context + response, and AUTHOR-labelled -- Reddit
    self-annotation and Twitter hashtags. Author labelling is the property that matters: it
    gives INTENDED sarcasm, where a third-party annotation would give PERCEIVED sarcasm and
    reduce to marker detection, which is the confound the gate exists to avoid.

THE LENGTH TRAP. In this corpus the SARCASTIC class is SHORTER (reddit median 11 vs 13 words),
the opposite direction to the rest of the project, where the positive class is longer in 9 of 10
datasets. Unmatched, length alone scores 0.578 on reddit and 0.625 on twitter. So rows are
paired within 1 word and the matched set scores exactly 0.5000 on length.

THE FLOOR THE GATE DID NOT HAVE. Bag-of-words separability is 0.678 (reddit) and 0.783
(twitter) unmatched, and 0.632 on the matched reddit set. An axis scoring under that floor has
measured vocabulary, not sarcasm. Gate 3 as written specifies no floor at all; this script
prints one and stores it in meta so the comparison cannot be skipped later. See F-106.

    python cap/scripts/build_sarcasm_natural.py --write

Writes cap/run-sarcasm-natural/rows.jsonl in the EvalRow shape eval_datasets.load_generated
expects, keyed `sarcasm_natural`, which is EVAL_ONLY and absent from all_keys().
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "cap" / "run-sarcasm-natural"
BASE = ("https://huggingface.co/datasets/tasksource/figlang2020-sarcasm/resolve/main/"
        "sarcasm_detection_shared_task_{arm}_training.jsonl")
WORD = re.compile(r"[a-z']+")


def fetch(arm: str, cache: Path) -> list[dict]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        urllib.request.urlretrieve(BASE.format(arm=arm), cache)
    return [json.loads(line) for line in cache.open(encoding="utf-8")]


def bow(texts):
    vocab: dict[str, int] = defaultdict(int)
    toks = []
    for t in texts:
        w = WORD.findall(t.lower())
        toks.append(w)
        for x in set(w):
            vocab[x] += 1
    keep = sorted([w for w, c in vocab.items() if c >= 3])
    idx = {w: i for i, w in enumerate(keep)}
    X = np.zeros((len(toks), len(keep)), dtype=np.float32)
    for i, w in enumerate(toks):
        for x in w:
            if x in idx:
                X[i, idx[x]] += 1
    return X, keep


def baselines(rows: list[dict]) -> dict:
    """Length-only and bag-of-words AUROC. Both are free and both bound any axis result."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    y = np.array([1 if r["label"] == "SARCASM" else 0 for r in rows])
    texts = [r["response"] for r in rows]
    lens = np.array([len(t.split()) for t in texts], dtype=float)
    X, keep = bow(texts)
    p = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), X, y, cv=5,
                          method="decision_function")
    # sarcastic is the SHORTER class here, so the length predictor is -length
    return {"n": int(len(y)), "vocab": len(keep),
            "auroc_length": float(roc_auc_score(y, -lens)),
            "auroc_bow": float(roc_auc_score(y, p))}


def length_match(rows: list[dict], n_pairs: int, tol: int, seed: int) -> list[tuple[dict, int]]:
    """Pair each sarcastic row with an unused non-sarcastic row of near-identical length.

    Matching on the DV's strongest nuisance correlate before extraction is cheaper than
    regressing it out afterwards, and it makes the length control exact rather than modelled.
    """
    rng = random.Random(seed)
    sarc = [r for r in rows if r["label"] == "SARCASM"]
    nots = sorted((r for r in rows if r["label"] == "NOT_SARCASM"),
                  key=lambda r: len(r["response"].split()))
    lens = [len(r["response"].split()) for r in nots]
    used: set[int] = set()
    pairs = []
    for s in rng.sample(sarc, len(sarc)):
        w = len(s["response"].split())
        i = int(np.searchsorted(lens, w))
        best = None
        for j in range(max(0, i - 40), min(len(nots), i + 40)):
            if j in used:
                continue
            d = abs(lens[j] - w)
            if best is None or d < best[0]:
                best = (d, j)
            if d == 0:
                break
        if best and best[0] <= tol:
            used.add(best[1])
            pairs.append((s, nots[best[1]]))
        if len(pairs) >= n_pairs:
            break
    return [(s, 1) for s, _ in pairs] + [(n, 0) for _, n in pairs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="reddit", choices=["reddit", "twitter"],
                    help="reddit has the lower trivial-feature floor and is the primary arm")
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--tol", type=int, default=1, help="max word-count gap within a pair")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write", action="store_true", help="without this, report and exit")
    args = ap.parse_args()

    raw = fetch(args.arm, OUT_DIR / f"figlang2020_{args.arm}.jsonl")
    full = baselines(raw)
    print(f"{args.arm}: {full['n']} rows, vocab {full['vocab']}")
    print(f"  UNMATCHED   length-only {full['auroc_length']:.4f}   "
          f"bag-of-words {full['auroc_bow']:.4f}")

    sel = length_match(raw, args.pairs, args.tol, args.seed)
    matched_rows = [{"response": r["response"],
                     "label": "SARCASM" if lab else "NOT_SARCASM"} for r, lab in sel]
    m = baselines(matched_rows)
    print(f"matched: {m['n']} rows, vocab {m['vocab']}")
    print(f"  MATCHED     length-only {m['auroc_length']:.4f}   "
          f"bag-of-words {m['auroc_bow']:.4f}")
    print(f"\n  => GATE 3 FLOOR: an axis AUROC at or below {m['auroc_bow']:.3f} has measured "
          f"vocabulary, not sarcasm.")

    if not args.write:
        print("\n(dry run; pass --write to emit rows.jsonl)")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "rows.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for i, (r, lab) in enumerate(sel):
            ctx = r["context"][-1] if r.get("context") else ""
            msgs = [{"role": "user", "content": ctx},
                    {"role": "assistant", "content": r["response"]}]
            item = f"{args.arm}-{i:04d}"
            fh.write(json.dumps({
                "row_id": f"sarcasm_natural:{item}:0",
                "dataset": "sarcasm_natural",
                "item_id": item,
                "sample_idx": 0,
                "response_str": r["response"],
                "label": int(lab),
                "label_raw": r["label"],
                # NOT a deception semantics -- see eval_datasets.AUTHOR_MARKED_SARCASM
                "label_semantics": "author_marked_sarcasm",
                # the text is human-written and placed in an assistant turn, so the prompt is
                # a reconstruction, not what any model was actually given
                "prompt_fidelity": "reconstructed",
                "prompt_str": ctx,
                "messages": msgs,
                "src": f"figlang2020_{args.arm}",
                "n_words": len(r["response"].split()),
                "floor_auroc_bow": m["auroc_bow"],
                "floor_auroc_length": m["auroc_length"],
            }) + "\n")
    (OUT_DIR / "baselines.json").write_text(
        json.dumps({"arm": args.arm, "seed": args.seed, "tol": args.tol,
                    "unmatched": full, "matched": m}, indent=1), encoding="utf-8")
    print(f"\nwrote {out} ({len(sel)} rows) and baselines.json")


if __name__ == "__main__":
    main()
