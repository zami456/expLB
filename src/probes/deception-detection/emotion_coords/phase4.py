"""Phase 4 of the emotion-coordinate study: HP-KR diagnostics D1-D3 on the full run, and P7 and P8.

    python -m emotion_coords.phase4 d1     # every HP-KR row at 20/28/40/52: the published probe, the protocol-B
                                           # probes trained without HP-KR (raw, persona PCs, emotion PCs at their
                                           # chosen k and C), and the zero-shot directions; P7
    python -m emotion_coords.phase4 d2d3   # 238 questions x 8 templates at 20/28/40/52 (52 primary); P8

Rules: ANALYSIS_PLAN.md. Outputs: results/emotion_coords/phase4/.
"""

import json

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from emotion_coords import config as C
from emotion_coords import hpkr as H
from emotion_coords import lbdata as LB
from emotion_coords import linalg as la
from emotion_coords import probes as P
from emotion_coords.pilot import ZERO_SHOT, persona_basis

HPKR = "harm_refusal_knowledge_report__plain"
PAIRS = (("lie_vs_admission", "lie", "honest_admission"), ("lie_vs_denial", "lie", "honest_denial"),
         ("admission_vs_denial", "honest_admission", "honest_denial"))


def _out(name: str):
    return C.results_dir("phase4") / name


def _bases(layer: int) -> dict:
    with np.load(C.data_dir("bases") / f"full_L{layer}.npz") as z:
        return {k: z[k] for k in z.files}


def subgroup_cis(scores: np.ndarray, subgroup: np.ndarray, n_boot: int = 1000) -> dict:
    """The three subgroup AUROCs ("A vs B": an A row scores above a B row) with stratified bootstrap CIs."""
    out = {}
    for name, a, b in PAIRS:
        keep = np.isin(subgroup, [a, b])
        y, s = (subgroup[keep] == a).astype(int), scores[keep]
        draws = P.Resamples.draw(y, n_boot).aurocs(y, s)
        lo, hi = P.ci(draws)
        out |= {name: P.auroc(y, s), f"{name}_lo": lo, f"{name}_hi": hi}
    return out


def d1(n_boot: int = 1000) -> None:
    from deception_detection.utils import REPO_ROOT
    from deception_detection.scripts.persona_ood import run_probe

    s = C.load()
    rows = H.hpkr_rows()
    sub_all = rows["subgroup"].to_numpy()
    sub_explicit = np.where((sub_all == "lie") & (rows["claim"].to_numpy() != "DENIED_KNOWLEDGE"), "lie_other", sub_all)
    frame = pd.read_csv(C.results_dir("phase3") / "probes.csv")
    subset = LB.full_subset()
    lb5 = LB.lb5()
    table, store = [], {}
    for layer in s["layers"]:
        x_hpkr = LB.load(layer, "a")[HPKR].x
        if len(x_hpkr) != len(rows):
            raise RuntimeError(f"L{layer}: {len(x_hpkr)} HP-KR rows, expected {len(rows)}")
        bases = _bases(layer)
        run_dir = REPO_ROOT / "results" / ("XXXX-9-llama-all" if layer == 20 else f"XXXX-9-llama-all-L{layer}")
        scorers = {"A published RepE probe": run_probe(run_dir)(x_hpkr)}
        data = LB.load(layer, "a", subset)
        train = [n for n in lb5 if n != HPKR]
        x = np.concatenate([data[n].x for n in train])
        y = np.concatenate([data[n].y for n in train])
        for name, basis in (("raw", None), ("persona_pub_pc", persona_basis(layer, "published")),
                            ("persona_pc", persona_basis(layer)), ("emotion_pc", bases["E_components"])):
            pick = frame[(frame["protocol"] == "B") & (frame["feature_set"] == name) & (frame["test"] == HPKR)
                         & (frame["probe"] == "lr") & (frame["layer"] == layer) & (frame["pooling"] == "a")
                         & frame["chosen"]]
            k, c = int(pick["k"].iloc[0]), float(pick["C"].iloc[0])
            f = (lambda z: z) if basis is None else (lambda z, b=basis[:k]: z @ b.T)
            scorers[f"B {name} (k={k})"] = P.fit("lr", f(x), y, c).score(f(x_hpkr))
        labels = list(bases["E_labels"])
        for n, sign in ZERO_SHOT.items():
            if n in labels:
                scorers[f"zero-shot {n}"] = sign * (x_hpkr @ la.unit(bases["E_vectors"][labels.index(n)]))
        for name, sc in scorers.items():
            store[f"L{layer}|{name}"] = sc.astype(np.float32)
            for lies, sub in (("all", sub_all), ("explicit", sub_explicit)):
                medians = {f"{g}_median": float(np.median(sc[sub == g])) for g in H.SUBGROUPS}
                table.append({"layer": layer, "scorer": name, "lies": lies} | subgroup_cis(sc, sub, n_boot) | medians
                             | {f"{g}_n": int((sub == g).sum()) for g in H.SUBGROUPS})
        logger.info(f"D1 L{layer} done")
    d1_table = pd.DataFrame(table)
    d1_table.to_csv(_out("d1.csv"), index=False)
    np.savez(C.data_dir("phase4") / "d1_scores.npz", subgroup=sub_all.astype(str), **store)  # not object: no pickle

    # P7: the published block-20 probe; its HP-KR AUROC on the capped rows comes from phase 3 (protocol A-published)
    pub = frame[(frame["protocol"] == "A-published") & (frame["layer"] == 20) & (frame["test"] == HPKR)]
    r = d1_table[(d1_table["layer"] == 20) & (d1_table["scorer"] == "A published RepE probe") & (d1_table["lies"] == "all")].iloc[0]

    def reading(name: str) -> str:
        if r[f"{name}_hi"] < 0.5:
            return "well under 0.5"
        if r[f"{name}_lo"] <= 0.5 <= r[f"{name}_hi"]:
            return "near 0.5"
        return "above 0.5" if r[f"{name}_lo"] > 0.5 else "unclear"

    lva, lvd = reading("lie_vs_admission"), reading("lie_vs_denial")
    p7 = {"hpkr_auroc_capped": float(pub["auroc"].iloc[0]), "auroc_under_0.5": bool(pub["auroc"].iloc[0] < 0.5),
          "lie_vs_admission": [r["lie_vs_admission"], r["lie_vs_admission_lo"], r["lie_vs_admission_hi"], lva],
          "lie_vs_denial": [r["lie_vs_denial"], r["lie_vs_denial_lo"], r["lie_vs_denial_hi"], lvd],
          "account_a_intent": lva == "well under 0.5" and lvd == "near 0.5",
          "account_b_knowledge": lvd == "well under 0.5" and lva == "near 0.5",
          "note": "these AUROCs were seen in the pilot; the CI reading is the default meaning of the words"}
    (_out("p7.json")).write_text(json.dumps(p7, indent=2, default=float), encoding="utf-8")
    logger.info(f"P7: {p7}")


def d2d3(n_null: int = 1000) -> None:
    s = C.load()
    p8 = {}
    for layer in s["layers"]:
        bases = _bases(layer)
        labels = list(bases["E_labels"])
        emotions = {n: bases["E_vectors"][i] for i, n in enumerate(labels)}
        pre, reply, meta = H.d2d3_acts(layer)
        design = H.Design.of(meta)
        for name, acts in (("d2", pre), ("d3", reply)):
            table, nulls = H.did(acts, design, emotions, bases["T_vectors"], n_null=n_null, return_nulls=True)
            table.to_csv(_out(f"{name}_L{layer}.csv"), index=False)
            np.savez(C.data_dir("phase4") / f"{name}_L{layer}_nulls.npz", **nulls)
            if name == "d2" and layer == s["d2_primary_layer"]:
                hit = table.set_index("emotion")["above_95_both"]
                p8 = {"layer": layer, "rows": len(meta), "questions": int(meta["question_id"].nunique()),
                      "a_compassion": {e: bool(hit[e]) for e in H.P8_COMPASSION if e in hit},
                      "b_threat": {e: bool(hit[e]) for e in H.P8_THREAT if e in hit}}
                p8["a_holds"] = any(p8["a_compassion"].values())
                p8["b_holds"] = any(p8["b_threat"].values())
        logger.info(f"D2/D3 L{layer}: {len(meta)} rows")
    (_out("p8.json")).write_text(json.dumps(p8, indent=2), encoding="utf-8")
    logger.info(f"P8: {p8}")


if __name__ == "__main__":
    fire.Fire({"d1": d1, "d2d3": d2d3})
