"""Protocol A (train on the RepE pairs, k and C by statement-grouped CV; test on each LB-5 set), lr, from the parts."""
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MAIN, BAND = Path("data/emotion_coords/phase3/parts"), Path("data/emotion_coords/phase3/band")
SHORT = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C",
         "harm_refusal_knowledge_report__plain": "HP-KR", "insider_trading_cadenza__plain": "IT",
         "instruct_dishonesty__plain": "ID"}
S = list(SHORT.values())
LAYERS = {20: MAIN, 28: MAIN, 35: BAND, 36: BAND, 40: MAIN, 44: BAND, 52: MAIN}


def rows(folder, layer, unit, protocol):
    path = folder / f"L{layer}a_{unit}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        df = pd.DataFrame(pickle.load(f)["rows"])
    gc.collect()
    df = df[df["protocol"] == protocol].copy()
    df["t"] = df["test"].map(SHORT)
    return df


tables, ks, bal = {}, {}, {}
for layer, folder in LAYERS.items():
    emo = rows(folder, layer, "lr_emotion_pc_s0", "A")
    k_e = int(emo[emo["chosen"]]["k"].iloc[0])
    ks[layer] = k_e
    out = {}

    def chosen(unit, name):
        d = rows(folder, layer, unit, "A")
        d = d[d["chosen"]]
        out[name] = d.groupby("t")["auroc"].mean().reindex(S)
        bal[(layer, name)] = d["balacc"].mean()

    def at_k(units, name):
        ds = [rows(folder, layer, u, "A") for u in units]
        d = pd.concat([x[x["k"] == k_e] for x in ds])
        out[name] = d.groupby("t")["auroc"].mean().reindex(S)
        bal[(layer, name)] = d["balacc"].mean()

    chosen("lr_emotion_pc_s0", "emotion PCs (k by CV)")
    chosen("lr_raw_s0", "raw activations (trained on RepE)")
    at_k(["lr_topic_pc_s0"], "topic PCs (emotion's k)")
    at_k([f"lr_random_s{s}" for s in range(20)], "random (emotion's k, 20)")
    at_k([f"lr_shuffled_pc_s{s}" for s in range(5)], "label-shuffled PCs (emotion's k)")
    chosen("lr_persona_pub_pc_s0", "persona PCs, published")
    chosen("lr_persona_pc_s0", "persona PCs, extended")
    chosen("lr_named_s0", "29 named emotions")
    pub = rows(folder, layer, "zero_shot_and_published", "A-published")
    if pub is not None and len(pub):
        out["published RepE probe (per-token)"] = pub.groupby("t")["auroc"].mean().reindex(S)
        bal[(layer, "published RepE probe (per-token)")] = pub["balacc"].mean()
    tables[layer] = pd.DataFrame(out).T
pd.set_option("display.width", 220)
means = pd.DataFrame({L: t[S].mean(axis=1) for L, t in tables.items()})
print("emotion PCs: k chosen by grouped CV on RepE, per layer:", ks)
print("\nMEAN AUROC over the five LB sets (protocol A: train on RepE pairs, test on each set)")
print(means.round(3).to_string())
print("\nLAYER 35 per set")
print(tables[35].round(3).to_string())
print("\nMEAN BALANCED ACCURACY at the Alpaca 1% FPR threshold")
b = pd.DataFrame({L: {n: bal.get((L, n), np.nan) for n in means.index} for L in LAYERS})
print(b.round(3).to_string())
