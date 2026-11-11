"""5x5 train-on-one / test-on-each matrices (protocol C, lr) for every arm, all layers; long CSV + a printout."""
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from extensions.emotion_coords import config as C

SHORT = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C",
         "harm_refusal_knowledge_report__plain": "HP-KR", "insider_trading_cadenza__plain": "IT",
         "instruct_dishonesty__plain": "ID"}
SETS = list(SHORT.values())
parts, explore = C.data_dir("phase3", "parts"), C.data_dir("phase3", "explore_vectors")


def c_rows(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        rows = pickle.load(f)["rows"]
    gc.collect()
    df = pd.DataFrame(rows)
    df = df[(df["protocol"] == "C") & (df["probe"] == "lr")].copy()
    df["src"], df["dst"] = df["train"].map(SHORT), df["test"].map(SHORT)
    return df


def matrix(df: pd.DataFrame, k_by_src: dict | None = None) -> pd.DataFrame:
    m = pd.DataFrame(np.nan, index=SETS, columns=SETS)
    for src in SETS:
        d = df[df["src"] == src]
        d = d[d["k"] == k_by_src[src]] if k_by_src else d[d["chosen"]]
        for dst in SETS:
            sel = d[(d["dst"] == dst) & ((d["diagonal"] == True) if dst == src else (d["diagonal"] != True))]
            m.loc[src, dst] = sel["auroc"].mean() if len(sel) else np.nan
    return m


long, summary, shown = [], [], {}
for layer in (20, 28, 40, 52):
    emo = c_rows(parts / f"L{layer}a_lr_emotion_pc_s0.pkl")
    k_e = None if emo is None else {s: int(emo[(emo["src"] == s) & emo["chosen"]]["k"].iloc[0]) for s in SETS}
    arms = {"raw activations": [parts / f"L{layer}a_lr_raw_s0.pkl"],
            "emotion PCs (k by CV)": [parts / f"L{layer}a_lr_emotion_pc_s0.pkl"],
            "all 170 emotion PCs [expl.]": [explore / f"L{layer}a_lr_emotion_all_C.pkl"],
            "all 171 emotion vectors [post hoc]": [explore / f"L{layer}a_lr_emotion_vectors_all_C.pkl"],
            "persona PCs (k by CV)": [parts / f"L{layer}a_lr_persona_pc_s0.pkl"],
            "topic PCs (emotion's k)": [parts / f"L{layer}a_lr_topic_pc_s0.pkl"],
            "random, emotion's k (20 seeds)": [parts / f"L{layer}a_lr_random_s{s}.pkl" for s in range(20)],
            "label-shuffled PCs, emotion's k (5)": [parts / f"L{layer}a_lr_shuffled_pc_s{s}.pkl" for s in range(5)],
            "named emotions (29)": [parts / f"L{layer}a_lr_named_s0.pkl"],
            "training-set PCs (k by CV)": [parts / f"L{layer}a_lr_trainpca_pc_s0.pkl"],
            "Alpaca PCs (k by CV)": [parts / f"L{layer}a_lr_alpaca_pc_s0.pkl"]}
    for arm, paths in arms.items():
        matched = "emotion's k" in arm
        mats = []
        for p in paths:
            df = c_rows(p)
            if df is not None and not (matched and k_e is None):
                mats.append(matrix(df, k_e if matched else None))
        if not mats:
            summary.append({"layer": layer, "arm": arm})
            continue
        m = sum(mats) / len(mats)
        off = m.values[~np.eye(5, dtype=bool)]
        summary.append({"layer": layer, "arm": arm, "transfer (off-diag mean)": float(np.nanmean(off)),
                        "within-set (diag mean)": float(np.nanmean(np.diag(m.values)))})
        for src in SETS:
            for dst in SETS:
                long.append({"layer": layer, "arm": arm, "train": src, "test": dst, "auroc": m.loc[src, dst]})
        shown[(layer, arm)] = m
out = C.results_dir("exploratory")
pd.DataFrame(long).to_csv(out / "protocol_c_matrices_lr.csv", index=False)
summ = pd.DataFrame(summary)
summ.to_csv(out / "protocol_c_summary_lr.csv", index=False)
pd.set_option("display.width", 220)
print("MEAN TRANSFER AUROC (train on one, test on each of the other four; 20 off-diagonal cells)")
print(summ.pivot(index="arm", columns="layer", values="transfer (off-diag mean)").reindex(list(arms)).round(3).to_string())
print("\nMEAN WITHIN-SET AUROC (diagonal, 5-fold CV)")
print(summ.pivot(index="arm", columns="layer", values="within-set (diag mean)").reindex(list(arms)).round(3).to_string())
for arm in ("raw activations", "emotion PCs (k by CV)", "all 170 emotion PCs [expl.]",
            "all 171 emotion vectors [post hoc]", "persona PCs (k by CV)", "topic PCs (emotion's k)",
            "random, emotion's k (20 seeds)"):
    m = shown.get((40, arm))
    if m is not None:
        off = m.values[~np.eye(5, dtype=bool)]
        print(f"\nLayer 40 | {arm} | rows = train, columns = test | transfer mean {np.nanmean(off):.3f}")
        print(m.round(3).to_string())
