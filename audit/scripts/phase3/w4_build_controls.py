"""W4 — the central control table.

`audit/PHASE2_PLAN.md:37` asked for the controls to be re-derived once, centrally, so every
claim cites the same numbers instead of a console log.  It was never done, and the audit
believed the random-direction null existed "only in a console log from 2026-09-16".

**That belief was wrong.**  The null is persisted, with 1,000 draws per cell, in
`run-coords/zeroshot_nulls.json` — which sits in the git working tree.  This script
consolidates it with the trained controls in `audit/phase2/layer_sensitivity.csv` into one
table.

SIGN CONVENTION (stated because this audit has been bitten by it five times):
  * `auroc` columns are SIGNED, in [0, 1].  0.5 is chance.  A value below 0.5 means the
    direction is INVERTED on that dataset, which is a real and reportable result here.
  * the null columns carry the suffix `_dev` in the source file and are DIRECTION-ORIENTED,
    i.e. percentiles of max(a, 1-a) over the random draws.  So an inverted axis can legitimately
    "beat" the null — that is precisely the sycophancy result (auroc 0.313, beats p95 0.630
    because max(0.313, 0.687) = 0.687).
  * `oriented_auroc` = max(auroc, 1-auroc), the quantity actually comparable to the null.

Read-only apart from its two outputs.
"""

import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
NULLS = ROOT / "run-coords" / "zeroshot_nulls.json"
LS = ROOT / "audit" / "phase2" / "layer_sensitivity.csv"
OUT = ROOT / "audit" / "phase2" / "controls.csv"

FIELDS = ["block", "scope", "dataset", "layer", "pooling", "control_type", "feature",
          "value", "oriented_value", "null_p95_dev", "null_max_dev", "n_draws",
          "p_vs_random", "beats_p95", "source"]


def parse_block(name, body, rows):
    """zeroshot_nulls.json has three shapes; normalise them."""
    def emit(scope, dataset, layer, pooling, cell):
        n95 = cell.get("random_p95_auroc_dev")
        nmax = cell.get("random_max_auroc_dev")
        nd = cell.get("n_random")
        for key, val in cell.items():
            if not isinstance(val, dict) or "auroc" not in val:
                continue
            a = val["auroc"]
            rows.append({
                "block": name, "scope": scope, "dataset": dataset, "layer": layer,
                "pooling": pooling, "control_type": "zeroshot_random_null",
                "feature": key, "value": f"{a:.6f}",
                "oriented_value": f"{max(a, 1 - a):.6f}",
                "null_p95_dev": f"{n95:.6f}" if n95 is not None else "",
                "null_max_dev": f"{nmax:.6f}" if nmax is not None else "",
                "n_draws": nd, "p_vs_random": val.get("p_vs_random", ""),
                "beats_p95": val.get("beats_random_p95", ""),
                "source": "run-coords/zeroshot_nulls.json",
            })

    layer, pooling = "", ""
    if name.startswith("corpus_"):
        _, L, pooling = name.split("_")
        layer = L.lstrip("L")

    if "pooled" in body:
        emit("pooled", "ALL", layer, pooling, body["pooled"])
        for ds, cell in body.get("per_dataset", {}).items():
            emit("per_dataset", ds, layer, pooling, cell)
    else:
        for k, cell in body.items():
            if not isinstance(cell, dict):
                continue
            if k.startswith("L") and "_" in k:           # e.g. "L40_mean"
                L, pool = k.split("_", 1)
                emit("per_dataset", name, L.lstrip("L"), pool, cell)
            else:                                         # e.g. "gender-secret"
                emit("per_dataset", k, layer, pooling, cell)


def main():
    rows = []
    nulls = json.loads(NULLS.read_text(encoding="utf-8"))
    for name, body in nulls.items():
        parse_block(name, body, rows)
    n_zs = len(rows)

    # trained controls, from the already-built sensitivity table
    want = {"datapca", "datapca@2", "random", "random@2", "length", "raw"}
    for r in csv.DictReader(LS.open(encoding="utf-8")):
        if r["feature"] not in want:
            continue
        rows.append({
            "block": r["store"], "scope": "offdiag_mean", "dataset": r["dataset_set"],
            "layer": r["layer"], "pooling": r["pooling"],
            "control_type": "trained_control", "feature": r["feature"],
            "value": r["offdiag_mean"], "oriented_value": "",
            "null_p95_dev": "", "null_max_dev": "", "n_draws": "",
            "p_vs_random": "", "beats_p95": "",
            "source": f"layer_sensitivity.csv seed={r['seed']} probe={r['probe']}",
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"zero-shot null rows : {n_zs}")
    print(f"trained control rows: {len(rows) - n_zs}")
    print(f"total               : {len(rows)}  -> {OUT}")
    print()
    print("Zero-shot directions vs their own 1,000-draw null:")
    print(f"{'block':18}{'dataset':34}{'feature':22}{'auroc':>8}{'orient':>8}{'p95':>8}  beats")
    for r in rows[:n_zs]:
        if r["scope"] != "per_dataset":
            continue
        print(f"{r['block']:18}{r['dataset'][:33]:34}{r['feature'][:21]:22}"
              f"{float(r['value']):8.4f}{float(r['oriented_value']):8.4f}"
              f"{float(r['null_p95_dev']):8.4f}  {r['beats_p95']}")


if __name__ == "__main__":
    main()
