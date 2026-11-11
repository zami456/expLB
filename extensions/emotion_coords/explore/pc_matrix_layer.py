"""Protocol C (train on one set, test on each other; diagonal by 5-fold CV), lr, one layer, from a parts folder."""
import gc
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

parts, layer = Path(sys.argv[1]), int(sys.argv[2])
SHORT = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C",
         "harm_refusal_knowledge_report__plain": "HP-KR", "insider_trading_cadenza__plain": "IT",
         "instruct_dishonesty__plain": "ID"}
S = list(SHORT.values())


def c_rows(path):
    with open(path, "rb") as f:
        df = pd.DataFrame(pickle.load(f)["rows"])
    gc.collect()
    df = df[(df["protocol"] == "C") & (df["probe"] == "lr")].copy()
    df["src"], df["dst"] = df["train"].map(SHORT), df["test"].map(SHORT)
    return df


def matrix(df, k_by_src=None):
    m = pd.DataFrame(np.nan, index=S, columns=S)
    for a in S:
        d = df[df["src"] == a]
        d = d[d["k"] == k_by_src[a]] if k_by_src else d[d["chosen"]]
        for b in S:
            sel = d[(d["dst"] == b) & ((d["diagonal"] == True) if a == b else (d["diagonal"] != True))]
            m.loc[a, b] = sel["auroc"].mean()
    return m


emo = c_rows(parts / f"L{layer}a_lr_emotion_pc_s0.pkl")
k_e = {a: int(emo[(emo["src"] == a) & emo["chosen"]]["k"].iloc[0]) for a in S}
arms = {"Emotion PCs (k by CV)": ([f"emotion_pc_s0"], None), "Raw activations": (["raw_s0"], None),
        "Persona PCs, published": (["persona_pub_pc_s0"], None), "Persona PCs, extended": (["persona_pc_s0"], None),
        "Topic PCs (emotion's k)": (["topic_pc_s0"], k_e),
        "Random (emotion's k, 20 seeds)": ([f"random_s{s}" for s in range(20)], k_e),
        "Label-shuffled PCs (emotion's k, 5)": ([f"shuffled_pc_s{s}" for s in range(5)], k_e)}
off = ~np.eye(5, dtype=bool)
print("emotion PCs: k chosen per training set", k_e)
summary, mats = [], {}
for name, (files, ks) in arms.items():
    ms = [matrix(c_rows(parts / f"L{layer}a_lr_{f}.pkl"), ks) for f in files]
    m = sum(ms) / len(ms)
    mats[name] = m
    rows = [m.loc[a, [b for b in S if b != a]].mean() for a in S]
    summary.append({"arm": name, "transfer": m.values[off].mean(), "within": np.diag(m.values).mean(),
                    "inverted": int((m.values[off] < 0.5).sum()), **{f"train {a}": r for a, r in zip(S, rows)}})
pd.set_option("display.width", 220)
print(pd.DataFrame(summary).set_index("arm").round(3).to_string())
for name in ("Emotion PCs (k by CV)", "Raw activations", "Persona PCs, published", "Topic PCs (emotion's k)"):
    print(f"\n{name}: transfer {mats[name].values[off].mean():.3f}")
    print(mats[name].round(2).to_string())
best = pd.DataFrame("", index=S, columns=S)
cand = {"emo": "Emotion PCs (k by CV)", "raw": "Raw activations", "pers": "Persona PCs, published", "topic": "Topic PCs (emotion's k)"}
for a in S:
    for b in S:
        if a != b:
            v = {k: mats[n].loc[a, b] for k, n in cand.items()}
            best.loc[a, b] = max(v, key=v.get)
print("\nbest of emotion / raw / persona-published / topic per cell:")
print(best.to_string())
print({k: int((best.values == k).sum()) for k in cand})
