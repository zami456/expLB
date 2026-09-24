"""Phase 2, workstream W1 — layer sensitivity, computed once for everything.

For every crossgen store, pooling, layer and seed, compute the off-diagonal mean AUROC of every feature over two
dataset sets: ALL datasets present in the matrix, and the paper's five. The recipe is kd/scripts/paper_comparison.py:
`results` is a list of {feature, probe, auroc}, auroc being a train x test matrix indexed by `datasets`.

Any claim of the form "X beats Y at layer L" can then be looked up at every other layer, which is what the
"L40 was chosen because it maximised the advantage" issue needs.

Read-only; runs on the lab PC. Output: audit_scratch/out/layer_sensitivity.csv
"""
import csv, json, os, statistics

REPO = "C:/Users/user4/expLB/run-crossgen"
PAPER5 = {"instructed-deception", "mask", "convincing-game", "ai_liar", "roleplaying"}
STORES = ["", "kd11", "kd11/banks/kd", "kd11/banks/kd_s23", "kd11/banks/cap_s23", "kd11/banks/cap_all",
          "kd11/banks/cap_ctrl", "kd11/banks/collab", "banks/own100", "banks/hf100", "banks/paper16_shared",
          "banks/paper16_famcentred", "scale_3b", "scale_8b", "scale_70b", "scale_70b_paper16fp32",
          "passA", "passC", "kd11/passA", "kd11/passC", "passB_fine", "passP"]
OUT = "C:/Users/user4/audit_scratch/out/layer_sensitivity.csv"


def offdiag(ds, block, keep):
    vals = [v for tr, row in zip(ds, block) for te, v in zip(ds, row)
            if tr != te and v is not None and (keep is None or (tr in keep and te in keep))]
    return (statistics.fmean(vals), len(vals)) if vals else (None, 0)


def diag(ds, block, keep):
    vals = [row[i] for i, (tr, row) in enumerate(zip(ds, block))
            if row[i] is not None and (keep is None or tr in keep)]
    return statistics.fmean(vals) if vals else None


rows = 0
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["store", "bank", "bank_name", "k", "pooling", "layer", "seed", "dataset_set", "n_datasets",
                "feature", "probe", "offdiag_mean", "n_pairs", "diag_mean"])
    for store in STORES:
        d = os.path.join(REPO, store, "matrices") if store else os.path.join(REPO, "matrices")
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            try:
                m = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            except Exception:
                continue
            ds = m.get("datasets")
            if not ds or "results" not in m:
                continue
            for feat_probe, block in {(r["feature"], r["probe"]): r["auroc"] for r in m["results"]}.items():
                for label, keep in (("all", None), ("paper5", PAPER5 if PAPER5 <= set(ds) else None)):
                    if label == "paper5" and keep is None:
                        continue
                    om, n = offdiag(ds, block, keep)
                    if om is None:
                        continue
                    w.writerow([store or "<root>", m.get("bank"), m.get("bank_name"), m.get("k"), m.get("pooling"),
                                m.get("layer"), m.get("seed"), label, len(keep) if keep else len(ds),
                                feat_probe[0], feat_probe[1], f"{om:.6f}", n,
                                (f"{diag(ds, block, keep):.6f}" if diag(ds, block, keep) is not None else "")])
                    rows += 1
print("wrote", rows, "rows ->", OUT)
