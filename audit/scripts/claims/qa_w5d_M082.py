"""LEAD QA for M-082 — is the audit's own length correction metric-blind?

Runs READ-ONLY on the lab PC over C:/Users/user4/expLB/run-coords/passB/<dataset>/rows.jsonl.

The memory note `response-length-transfers-on-its-own.md` carries an audit correction
(2026-09-23) saying the count is "9 of 11 datasets, or 8 of 10 with sycophancy_mt
excluded - NEVER '9 of 10'".  Six lines below it sits a table of per-dataset
length-only AUROCs in which exactly one of ten datasets inverts.

Batch W5D says both are right, of different metrics, and that the correction is
therefore metric-blind.  This recomputes BOTH metrics from the raw rows so the
question is settled on data rather than on the note's own table:

  metric A  mean token gap (positive class minus negative class)
  metric B  length-only AUROC per dataset  (P(len_pos > len_neg), ties at 0.5)

and reports the count of datasets in the claimed direction under each, over 11
datasets and over the 10 that exclude sycophancy_mt.
"""

import json
import pathlib

ROOT = pathlib.Path("C:/Users/user4/expLB/run-coords/passB")
TEXT_KEYS = ("response", "completion", "output", "text", "assistant", "answer")


def tokens(row):
    """Length proxy: prefer a stored token count, else whitespace tokens of the response."""
    for k in ("n_tokens", "num_tokens", "token_count", "len_tokens"):
        if isinstance(row.get(k), (int, float)):
            return float(row[k])
    for k in TEXT_KEYS:
        v = row.get(k)
        if isinstance(v, str):
            return float(len(v.split()))
    return None


def auroc(pos, neg):
    """Mann-Whitney AUROC with ties counted as 0.5."""
    if not pos or not neg:
        return None
    pos, neg = sorted(pos), sorted(neg)
    import bisect
    tot = 0.0
    for p in pos:
        lo = bisect.bisect_left(neg, p)
        hi = bisect.bisect_right(neg, p)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))


def main():
    out = {}
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        f = d / "rows.jsonl"
        if not f.exists():
            continue
        pos, neg, nnull = [], [], 0
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                lab = r.get("label")
                t = tokens(r)
                if lab is None or t is None:
                    nnull += 1
                    continue
                (pos if lab in (True, 1, "True", "true") else neg).append(t)
        if not pos or not neg:
            continue
        mp, mn = sum(pos) / len(pos), sum(neg) / len(neg)
        out[d.name] = {
            "n": len(pos) + len(neg), "n_pos": len(pos), "n_neg": len(neg),
            "null": nnull, "mean_gap": mp - mn, "auroc": auroc(pos, neg),
        }

    print(f"{'dataset':38} {'n':>6} {'mean gap':>10} {'AUROC':>8}")
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]["auroc"]):
        print(f"{k:38} {v['n']:6} {v['mean_gap']:+10.2f} {v['auroc']:8.4f}"
              + (f"   (dropped {v['null']} null-label)" if v["null"] else ""))

    for label, keys in (("11 datasets", list(out)),
                        ("10 (no sycophancy_mt)",
                         [k for k in out if k != "sycophancy_mt"])):
        g = sum(1 for k in keys if out[k]["mean_gap"] > 0)
        a = sum(1 for k in keys if out[k]["auroc"] > 0.5)
        print(f"\n{label}: positive class longer by MEAN GAP in {g} of {len(keys)}"
              f"  |  length-only AUROC > 0.5 in {a} of {len(keys)}")


if __name__ == "__main__":
    main()
