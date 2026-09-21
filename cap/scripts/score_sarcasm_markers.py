#!/usr/bin/env python
"""The deadpan-elicitation pilot's measurement instrument. No NDIF, no model -- pure text.

WHAT THE PILOT IS FOR. `cap/docs/SARCASM_DESIGN.md` step 2: before building the sarcasm
factorial, find out whether Llama-3.3-70B can produce sarcasm that is NOT marked. The
literature says probably not -- Sarc7 reports models "slip into blunt reproach", SarcasmBench
reports LLMs underperform supervised baselines -- and if it cannot, the on-policy design is dead
and the factorial must move to teacher-forced identical responses.

THREE MEASUREMENTS, IN DESCENDING ORDER OF IMPORTANCE.

  1. MARKER RATE per ironic variant. The five ironic variants escalate how explicitly they
     forbid markers. This reports which instruction strength, if any, suppresses them. A
     marker is anything that tells the reader "this is a joke" without them having to reason
     about the plan: scare quotes, exclamation marks, emoji, /s, an aside that steps outside
     the praise, a closing "but seriously", ALL-CAPS emphasis, or the stock ironic intensifiers
     ("genius", "flawless", "what could go wrong").

  2. LEXICAL SEPARABILITY between the ironic and sycophantic arms -- the design's corrected
     abandon-check. Both arms produce approving text about a bad plan, so once markers are
     stripped they should be textually near-identical and a bag-of-words classifier should sit
     near chance. **Grouped by item, so no question spans train and test** -- the v1 check was
     mis-specified precisely because question echo alone separated the arms. Reported at two
     settings: raw, and with marker tokens removed. The gap between them IS the marker effect.

  3. POLARITY. Does the sincere critic actually criticise, and the sycophant actually praise?
     A failure here is informative rather than fatal: if the SINCERE persona praises a bad
     plan, that is the model's own sycophancy base rate, which bears on the sycophantic arm.

WHAT THIS DOES NOT MEASURE. Whether a human or a judge perceives the response as sarcastic.
That is the blind five-rating judge check in the design, and it is a separate step. A response
can be marker-free and simply not sarcastic; this script cannot tell those apart, which is why
it reports marker rate ALONGSIDE the judge's role score rather than instead of it.

    python cap/scripts/score_sarcasm_markers.py --responses_dir cap/run-sarc-pilot/responses

Writes <responses_dir>/../marker_report.json and prints the tables.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# Each marker is a name and a regex applied to the RESPONSE ONLY. Question echo is stripped
# first, because a response quoting the user's own words back is not marking anything.
MARKERS = {
    "scare_quotes": r'["“‘’”][^"“”]{1,28}["“‘’”]',
    "exclamation": r"!",
    "emoji": r"[\U0001F300-\U0001FAFF☀-➿]",
    "slash_s": r"/s\b",
    "allcaps": r"\b[A-Z]{3,}\b",
    "reveal_aside": r"\b(just kidding|kidding|i'?m joking|obviously not|to be clear|"
                    r"in all seriousness|but seriously|jokes? aside|sarcas(m|tic))\b",
    "stock_irony": r"\b(what could (possibly )?go wrong|genius|flawless|foolproof|"
                   r"bulletproof|nothing could|truly inspired|bold strategy|masterpiece)\b",
    "rhetorical_q": r"\b(who needs|why would anyone|what's the worst)\b",
}

# Words that are markers rather than content. Removing them is what separates "the arms differ
# in wording" from "the arms differ only in how they are flagged".
MARKER_TOKENS = set("""genius flawless foolproof bulletproof masterpiece inspired bold
kidding joking sarcasm sarcastic seriously obviously surely certainly definitely absolutely
brilliant perfect impeccable stellar magnificent superb""".split())

WORD = re.compile(r"[a-z']+")


def strip_question(text: str, question: str) -> str:
    """Remove any run of >=4 consecutive question words echoed in the response."""
    qw = WORD.findall(question.lower())
    out = text
    for n in range(len(qw), 3, -1):
        for i in range(len(qw) - n + 1):
            frag = " ".join(qw[i:i + n])
            if frag and frag in out.lower():
                idx = out.lower().find(frag)
                out = out[:idx] + " " + out[idx + len(frag):]
    return out


def marker_hits(resp: str) -> dict:
    return {k: len(re.findall(v, resp, flags=re.I if k != "allcaps" else 0))
            for k, v in MARKERS.items()}


def bow(texts, drop_markers: bool):
    vocab = defaultdict(int)
    toks = []
    for t in texts:
        w = [x for x in WORD.findall(t.lower()) if not (drop_markers and x in MARKER_TOKENS)]
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
    return X


def grouped_auc(X, y, groups, seed=42):
    """Leave-one-group-out logistic regression. Groups are question ids, so no item spans
    train and test -- the correction the v1 lexical check needed."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); groups = np.asarray(groups)
    scores = np.zeros(len(y), dtype=float)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(X[tr], y[tr])
        scores[te] = clf.decision_function(X[te])
    return float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses_dir", required=True)
    ap.add_argument("--scores_dir", default=None, help="judge output, to join role scores")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rd = Path(args.responses_dir)
    rows = []
    for f in sorted(rd.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            resp = r["conversation"][-1]["content"]
            # Role comes from the FILENAME, not from r["label"]: 1_generate_ndif.py writes the
            # literal string "pos" there for every row, while gen_multiturn_ndif.py writes the
            # role. Keying on the filename works for both generators.
            rows.append({"role": f.stem, "variant": r["prompt_index"],
                         "qid": r["question_index"], "question": r["question"],
                         "resp": resp, "clean": strip_question(resp, r["question"])})
    if not rows:
        print(f"no responses under {rd}"); return
    print(f"{len(rows)} responses from {len({r['role'] for r in rows})} roles\n")

    # ---- 1. marker rate per role x variant ------------------------------------------------
    print("1. MARKER RATE -- fraction of responses carrying at least one marker")
    print(f"   {'role':<20}{'v1':>7}{'v2':>7}{'v3':>7}{'v4':>7}{'v5':>7}{'all':>8}   mean markers/resp")
    report = {"n": len(rows), "markers": {}, "per_marker": {}}
    for role in sorted({r["role"] for r in rows}):
        cells, allhit, allcnt = [], [], []
        for v in range(5):
            sub = [r for r in rows if r["role"] == role and r["variant"] == v]
            if not sub:
                cells.append(float("nan")); continue
            h = [sum(marker_hits(r["clean"]).values()) > 0 for r in sub]
            cells.append(float(np.mean(h)))
        sub = [r for r in rows if r["role"] == role]
        hits = [marker_hits(r["clean"]) for r in sub]
        allhit = [sum(h.values()) > 0 for h in hits]
        allcnt = [sum(h.values()) for h in hits]
        print(f"   {role:<20}" + "".join(f"{c:>7.2f}" if c == c else f"{'--':>7}" for c in cells)
              + f"{np.mean(allhit):>8.2f}   {np.mean(allcnt):>5.2f}")
        report["markers"][role] = {"per_variant": cells, "overall": float(np.mean(allhit)),
                                   "mean_count": float(np.mean(allcnt))}
        agg = defaultdict(int)
        for h in hits:
            for k, c in h.items():
                agg[k] += c
        report["per_marker"][role] = dict(agg)

    print("\n   which markers, by role (total counts):")
    for role, agg in report["per_marker"].items():
        top = sorted(agg.items(), key=lambda kv: -kv[1])[:5]
        print(f"   {role:<20}" + "  ".join(f"{k}={v}" for k, v in top if v))

    # ---- 2. lexical separability, ironic vs sycophantic -----------------------------------
    print("\n2. LEXICAL SEPARABILITY  ironic_critic vs sycophant, grouped by question id")
    pair = [r for r in rows if r["role"] in ("pc_ironic_critic", "pc_sycophant")]
    if len({r["role"] for r in pair}) == 2:
        y = [1 if r["role"] == "pc_ironic_critic" else 0 for r in pair]
        g = [r["qid"] for r in pair]
        texts = [r["clean"] for r in pair]
        a_raw = grouped_auc(bow(texts, False), y, g)
        a_str = grouped_auc(bow(texts, True), y, g)
        print(f"   bag-of-words AUROC, markers kept    {a_raw:.3f}")
        print(f"   bag-of-words AUROC, markers removed {a_str:.3f}")
        print(f"   marker contribution                 {a_raw - a_str:+.3f}")
        report["lexical"] = {"auroc_raw": a_raw, "auroc_stripped": a_str,
                             "marker_effect": a_raw - a_str, "n": len(pair)}
        print("   reading: near 0.5 with markers removed = the arms are textually alike, which")
        print("            is the premise the factorial needs. Near 1.0 = a lexical detector.")
    else:
        print("   both arms not present; skipped")

    # ---- 3. polarity ----------------------------------------------------------------------
    if args.scores_dir:
        sd = Path(args.scores_dir)
        # 3_judge_ndif.py writes a FLAT dict keyed "pos_p<variant>_q<question>" -> int, not a
        # "scores" list. The first version of this block looked for j["scores"], found nothing,
        # and printed an empty section while the chain reported the judge step as PASS -- so the
        # polarity check silently went unmeasured (recorded in F-93 as owed). Parse the real
        # shape, and break it out per variant, which the flat key makes free.
        print("\n3. POLARITY -- judge role score by role (0-3), and by variant")
        keyre = re.compile(r"^pos_p(\d+)_q(\d+)$")
        for f in sorted(sd.glob("*.json")):
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            byv = defaultdict(list)
            vals = []
            for k, v in (j or {}).items():
                m = keyre.match(str(k))
                if m and isinstance(v, (int, float)):
                    byv[int(m.group(1))].append(float(v))
                    vals.append(float(v))
            if not vals:
                print(f"   {f.stem:<20} no parseable scores in {f.name}")
                continue
            cells = " ".join(f"{np.mean(byv[v]):.2f}" if byv.get(v) else " -- " for v in range(5))
            print(f"   {f.stem:<20} mean {np.mean(vals):.2f}  n={len(vals):<4} "
                  f">=2: {np.mean([x >= 2 for x in vals]):.2f}   v1-v5 [{cells}]")
            report.setdefault("polarity", {})[f.stem] = {
                "mean": float(np.mean(vals)), "n": len(vals),
                "frac_ge2": float(np.mean([x >= 2 for x in vals])),
                "per_variant": [float(np.mean(byv[v])) if byv.get(v) else None for v in range(5)]}
    else:
        print("\n3. POLARITY -- skipped (--scores_dir not given; run the judge first)")

    out = Path(args.out) if args.out else rd.parent / "marker_report.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")

    # ---- the decision the pilot exists to make ---------------------------------------------
    ir = report["markers"].get("pc_ironic_critic")
    if ir:
        best = min([c for c in ir["per_variant"] if c == c] or [1.0])
        print("\nPILOT VERDICT")
        print(f"  lowest marker rate across the five ironic variants: {best:.2f}")
        if best <= 0.20:
            print("  => on-policy deadpan is PRODUCIBLE; build the factorial on the variant that")
            print("     achieved it, and keep the marker check as a per-item filter.")
        elif best <= 0.50:
            print("  => PARTIAL. Deadpan is producible but not reliably. Filtering to marker-free")
            print("     responses is possible but selects on the dependent variable -- prefer")
            print("     teacher-forced identical responses, per the design's fallback.")
        else:
            print("  => on-policy deadpan is NOT producible at any instruction strength tested.")
            print("     The factorial must use teacher-forced identical responses, accepting")
            print("     Kirch et al.'s point that text-ambiguous behaviours are exactly where")
            print("     off-policy data is least reliable. Record that as a limitation.")


if __name__ == "__main__":
    main()
