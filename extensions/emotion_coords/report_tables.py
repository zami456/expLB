"""Every table in RESULTS.md, built from the result files (so the report's numbers can be regenerated).

    python -m extensions.emotion_coords.report_tables > results/emotion_coords/report_tables.md
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from extensions.emotion_coords import config as C

R = C.results_dir()
SHORT = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C",
         "harm_refusal_knowledge_report__plain": "HP-KR", "insider_trading_cadenza__plain": "IT",
         "instruct_dishonesty__plain": "ID"}
SETS = list(SHORT.values())
MAIN, BAND = (20, 28, 40, 52), tuple(range(35, 46))
ALL = (20, 28, *range(35, 46), 52)


def md(frame: pd.DataFrame, digits: int = 3) -> str:
    frame = frame.copy()
    for c in frame.columns:
        if frame[c].dtype.kind == "f":
            frame[c] = frame[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join([frame.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in row]) + " |" for i, row in zip(frame.index, frame.values)]
    return "\n".join(lines)


def ci(d: dict) -> str:
    return f"{d['delta']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"


def _probes() -> pd.DataFrame:
    cols = ["protocol", "feature_set", "layer", "pooling", "probe", "k", "test", "auroc", "balacc", "recall", "chosen"]
    main = pd.read_csv(R / "phase3" / "probes.csv", usecols=cols)
    band = pd.read_csv(R / "phase3_band" / "probes.csv", usecols=cols)
    frame = pd.concat([main, band[~band["layer"].isin(MAIN)]])
    frame = frame[(frame["pooling"] == "a") & (frame["probe"] == "lr")]
    return frame.assign(t=frame["test"].map(SHORT))


def predictions() -> None:
    d = json.load(open(R / "phase3" / "predictions_lr.json"))
    rows = {}
    for v in d["per_layer"]:
        rows[v["layer"]] = {"P1 emotion - random": ci(v["P1_vs_random"]), "P1 emotion - topic": ci(v["P1_vs_topic"]),
                            "P2 IT: emotion - persona": ci(v["P2_IT_emotion_minus_persona"]),
                            "P3 HP-C / HP-KR: emotion - random": " / ".join(f"{x:+.3f}" for x in v["P3_emotion_minus_random"].values()),
                            "P4 persona+emotion - persona": ci(v["P4_pe_minus_persona"]),
                            "P5 IT AUROC guilty / afraid / desperate": " / ".join(f"{a:.2f}" for a, _ in v["P5_IT_ID"].values()),
                            "P6 erase 5 / 20 / LEACE vs random": " / ".join(f"{x['delta']:+.3f}{'*' if x['holds'] else ''}" for x in v["P6_variants"].values())}
        for p in ("P1", "P2", "P3", "P4", "P5", "P6"):
            rows[v["layer"]][p] = "holds" if v[p] else "fails"
    t = pd.DataFrame(rows)
    t.index.name = "block"
    print("## T1. Preregistered predictions P1-P6 (blocks 20/28/40/52)\n")
    print(md(t))
    print(f"\nLayers holding (3 of 4 needed): {d['layers_holding']}; extended persona basis: {d['extended_persona_layers_holding']}\n")


def all170_b() -> None:
    e = pd.concat([pd.read_csv(R / "phase3" / "exploratory_emotion_all_lr.csv"),
                   pd.read_csv(R / "phase3_band" / "exploratory_emotion_all_lr.csv")]).drop_duplicates("layer").set_index("layer").sort_index()
    t = pd.DataFrame({"all 170 emotion PCs": e["mean_emotion_all"], "raw activations": e["mean_raw"],
                      "minus raw": e["mean_emotion_all"] - e["mean_raw"]})
    for n, label in (("shuffled_all", "170 label-shuffled"), ("alpaca_all", "170 Alpaca PCs"),
                     ("trainpca_all", "170 training-set PCs"), ("random_all", "170 random")):
        t[f"minus {label} [95% CI]"] = [f"{r['emotion_all_minus_' + n]:+.3f} [{r['ci_lo_' + n]:+.3f}, {r['ci_hi_' + n]:+.3f}]"
                                        for _, r in e.iterrows()]
    t.index.name = "block"
    print("## T2. All 170 emotion PCs, protocol B (train on four sets, test on the fifth), mean held-out AUROC\n")
    print(md(t))
    means = pd.DataFrame({f"mean {n}": e[f"mean_{n}"] for n in ("shuffled_all", "alpaca_all", "trainpca_all", "random_all")})
    means.index.name = "block"
    print("\nThe controls' own means:\n")
    print(md(means))
    print()


def all170_sets(frame: pd.DataFrame) -> None:
    b = frame[(frame["protocol"] == "B")]
    rows = {}
    for L in (40, 41, 43, 45):
        bl = b[b["layer"] == L]
        e = bl[bl["feature_set"] == "emotion_all"].groupby("t")["auroc"].mean()
        r = bl[(bl["feature_set"] == "raw") & bl["chosen"]].groupby("t")["auroc"].mean()
        rows[f"{L} all-170"] = e.reindex(SETS)
        rows[f"{L} raw"] = r.reindex(SETS)
    t = pd.DataFrame(rows).T
    t.index.name = "block, arm"
    print("## T3. All 170 emotion PCs per held-out set (protocol B)\n")
    print(md(t))
    print()


def all170_c() -> None:
    band = pd.read_csv(R / "exploratory" / "protocol_c_band_170.csv")
    off = band[band["train"] != band["test"]]
    t = off.groupby(["arm", "layer"])["auroc"].mean().unstack()
    t = t.reindex(["all 170 emotion PCs", "170 label-shuffled PCs (5)", "170 Alpaca PCs", "raw activations"])
    t.loc["all-170 minus shuffled"] = t.iloc[0] - t.iloc[1]
    t.loc["all-170 minus raw"] = t.iloc[0] - t.iloc[3]
    inv = off.assign(inv=off["auroc"] < 0.5).groupby(["arm", "layer"])["inv"].sum().unstack()
    t.index.name = "arm"
    print("## T4. Protocol C (train on one set, test on each other), mean of the 20 transfers, band blocks\n")
    print(md(t))
    print("\nInverted transfers (AUROC < 0.5, of 20):\n")
    inv.index.name = "arm"
    print(md(inv.astype(int)))
    main = pd.read_csv(R / "exploratory" / "protocol_c_summary_lr.csv")
    m = main[main["arm"].isin(["all 170 emotion PCs [expl.]", "raw activations"])].pivot(
        index="arm", columns="layer", values="transfer (off-diag mean)")
    m.index.name = "arm"
    print("\nAt the preregistered blocks:\n")
    print(md(m))
    print()


def all170_mlp() -> None:
    m = pd.concat([pd.read_csv(R / "phase3" / "exploratory_mlp.csv"),
                   pd.read_csv(R / "phase3_band" / "exploratory_mlp.csv")]).drop_duplicates("layer").set_index("layer").sort_index()
    t = pd.DataFrame({"all 170 emotion PCs": m["mean_emotion_all"], "170 label-shuffled": m["mean_shuffled_all"],
                      "170 Alpaca PCs": m["mean_alpaca_all"], "170 random": m["mean_random_all"],
                      "persona PCs, published": m["mean_persona_pub_pc"], "persona PCs, extended": m["mean_persona_pc_extended"]})
    t["all-170 minus shuffled [CI]"] = [f"{r['emotion_all - shuffled_all']:+.3f} [{r['emotion_all - shuffled_all lo']:+.3f}, "
                                        f"{r['emotion_all - shuffled_all hi']:+.3f}]" for _, r in m.iterrows()]
    t.index.name = "block"
    print("## T5. MLP probes (protocol B, exploratory), mean held-out AUROC\n")
    print(md(t))
    print()


def robustness(frame: pd.DataFrame) -> None:
    rows = {}
    for L in MAIN:
        path = C.data_dir("phase3", "explore_vectors") / f"L{L}a_lr_emotion_vectors_all_s0.pkl"
        with open(path, "rb") as f:
            v = pd.DataFrame(pickle.load(f)["rows"])
        v = v[(v["protocol"] == "B") & v["chosen"]]
        b = frame[(frame["protocol"] == "B") & (frame["layer"] == L)]
        rows[L] = {"171 emotion vectors": v["auroc"].mean(),
                   "170 emotion PCs": b[b["feature_set"] == "emotion_all"]["auroc"].mean(),
                   "raw activations": b[(b["feature_set"] == "raw") & b["chosen"]]["auroc"].mean()}
    t = pd.DataFrame(rows).T
    t.index.name = "block"
    print("## T6. Same subspace, other coordinates: 171 unit emotion vectors vs their 170 PCs (protocol B, lr)\n")
    print(md(t))
    rows = {}
    for L in ALL:
        b = frame[(frame["protocol"] == "B") & (frame["layer"] == L)]
        rows[L] = {"all 170 emotion PCs": b[b["feature_set"] == "emotion_all"]["balacc"].mean(),
                   "raw activations": b[(b["feature_set"] == "raw") & b["chosen"]]["balacc"].mean(),
                   "persona PCs, extended": b[(b["feature_set"] == "persona_pc") & b["chosen"]]["balacc"].mean(),
                   "all-170 recall": b[b["feature_set"] == "emotion_all"]["recall"].mean()}
    t = pd.DataFrame(rows).T
    t.index.name = "block"
    print("\n## T7. Balanced accuracy at the Alpaca 1% FPR threshold (protocol B, mean of five held-out sets)\n")
    print(md(t))
    print()


def gender_secret() -> None:
    t = pd.read_csv(R / "gs" / "gs_arms_L40.csv").set_index("arm")
    keep = ["named", "persona_pub_pc", "raw", "emotion_vectors_all", "alpaca_all", "emotion_all", "shuffled_all",
            "trainpca_all", "random_all", "published_repe", "random", "emotion_pc", "persona_pc", "topic_pc"]
    t = t.loc[[k for k in keep if k in t.index],
              ["auroc", "ci_lo", "ci_hi", "auroc_base_read", "adapted_minus_base", "amb_ci_lo", "amb_ci_hi", "balacc"]]
    t.index.name = "arm"
    print("## T8. Gender-Secret at block 40 (fitted on the five LB-5 sets; G1-G3 fixed before scoring)\n")
    print(md(t))
    p = json.load(open(R / "gs" / "gs_predictions_L40.json"))
    print(f"\nG1 all-170 minus raw {ci(p['G1']['vs_raw'])}, minus shuffled-170 {ci(p['G1']['vs_shuffled_all'])}: "
          f"{'holds' if p['G1']['holds'] else 'fails'}. G2 emotion PCs minus random {ci(p['G2']['vs_random'])}, minus "
          f"topic {ci(p['G2']['vs_topic'])}: {'holds' if p['G2']['holds'] else 'fails'}.\n")


def band_emotion_pcs() -> None:
    d = json.load(open(R / "phase3_band" / "predictions_lr.json"))
    rows = {v["layer"]: {"emotion PCs": f"{v['mean_auroc']['emotion_pc']:.3f}", "raw": f"{v['mean_auroc']['raw']:.3f}",
                         "topic (same k)": f"{v['mean_auroc']['topic_pc']:.3f}",
                         "random (same k)": f"{v['mean_auroc']['random_mean']:.3f}",
                         "persona, extended": f"{v['mean_auroc']['persona_pc_extended']:.3f}",
                         "emotion - topic": ci(v["P1_vs_topic"]), "P1": "holds" if v["P1"] else "fails"}
            for v in d["per_layer"]}
    t = pd.DataFrame(rows).T
    t.index.name = "block"
    print("## T9. The preregistered emotion-PC arm across the band (protocol B)\n")
    print(md(t))
    print(f"\nBand layers holding (of 11): {d['layers_holding']}\n")


def d1_and_d2() -> None:
    p7 = json.load(open(R / "phase4" / "p7.json"))
    print("## T10. HP-KR: D1 for the published probe, and D2\n")
    print(f"P7 (block 20): capped HP-KR AUROC {p7['hpkr_auroc_capped']:.3f}; lie vs admission "
          f"{p7['lie_vs_admission'][0]:.3f} [{p7['lie_vs_admission'][1]:.3f}, {p7['lie_vs_admission'][2]:.3f}] "
          f"({p7['lie_vs_admission'][3]}); lie vs denial {p7['lie_vs_denial'][0]:.3f} [{p7['lie_vs_denial'][1]:.3f}, "
          f"{p7['lie_vs_denial'][2]:.3f}] ({p7['lie_vs_denial'][3]}).\n")
    d1 = pd.concat([pd.read_csv(R / "phase4" / "d1.csv"), pd.read_csv(R / "phase4" / "d1_band.csv")])
    pub = d1[(d1["scorer"] == "A published RepE probe") & (d1["lies"] == "all")].drop_duplicates("layer").set_index("layer").sort_index()
    t = pub[["lie_vs_admission", "lie_vs_denial", "admission_vs_denial"]]
    t.index.name = "block"
    print(md(t))
    rows = []
    for L in ALL:
        path = R / "phase4" / f"d2_L{L}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path).set_index("emotion")
        hits = d[d["above_95_both"] & d.index.isin(["compassionate", "empathetic", "sympathetic", "kind", "loving",
                                                     "afraid", "terrified", "angry", "hostile", "suspicious", "vigilant"])]
        rows.append({"block": L, "above both nulls": ", ".join(hits.index) or "none",
                     "lowest Holm p (11 emotions)": d.loc[["compassionate", "empathetic", "sympathetic", "kind", "loving",
                                                           "afraid", "terrified", "angry", "hostile", "suspicious",
                                                           "vigilant"], "p_holm"].min()})
    t = pd.DataFrame(rows).set_index("block")
    print("\nD2, known x harmful interaction (P8 emotions above the 95th percentile of both nulls):\n")
    print(md(t))
    print()


def vectors_sanity() -> None:
    rows = {}
    for name in ("phase1.json", "phase1_band.json", "phase1_band2.json"):
        d = json.load(open(R / "phase1" / name))
        for k, v in d.items():
            if k.isdigit():
                rows[int(k)] = {"|r(PC1, valence)|": abs(v["geometry"]["r_pc1_valence"]), "PC1 share": v["geometry"]["pc1_var"],
                                "core split-half (median)": v["split_half_final_core"]["median"],
                                "neutral PCs removed": str(v["neutral_pcs_removed"])}
    t = pd.DataFrame(rows).T.sort_index()
    for c in t.columns[:3]:
        t[c] = t[c].astype(float)
    t.index.name = "block"
    print("## T11. Emotion vectors: sanity checks per block\n")
    print(md(t))
    print()


if __name__ == "__main__":
    frame = _probes()
    vectors_sanity()
    predictions()
    all170_b()
    all170_sets(frame)
    all170_c()
    all170_mlp()
    robustness(frame)
    gender_secret()
    band_emotion_pcs()
    d1_and_d2()
