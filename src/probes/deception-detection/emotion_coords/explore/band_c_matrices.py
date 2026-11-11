"""Protocol C matrices for the 170-d arms and raw activations at every band block; long CSV and a printout."""
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from emotion_coords import config as C

SHORT = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C",
         "harm_refusal_knowledge_report__plain": "HP-KR", "insider_trading_cadenza__plain": "IT",
         "instruct_dishonesty__plain": "ID"}
S = list(SHORT.values())
EXP, BAND = C.data_dir("phase3", "explore_c_band"), C.data_dir("phase3", "band")
off = ~np.eye(5, dtype=bool)


def matrix(path):
    with open(path, "rb") as f:
        df = pd.DataFrame(pickle.load(f)["rows"])
    gc.collect()
    df = df[(df["protocol"] == "C") & (df["probe"] == "lr") & df["chosen"]]
    m = pd.DataFrame(np.nan, index=S, columns=S)
    for r in df.itertuples():
        a, b = SHORT[r.train], SHORT[r.test]
        if (a == b) == bool(r.diagonal == True):
            m.loc[a, b] = r.auroc
    return m


arms = {"all 170 emotion PCs": ["emotion_all_s0_C"], "170 label-shuffled PCs (5)": [f"shuffled_all_s{s}_C" for s in range(5)],
        "170 Alpaca PCs": ["alpaca_all_s0_C"], "raw activations": None}
long, summ, mats = [], [], {}
for layer in range(35, 46):
    for arm, units in arms.items():
        ms = [matrix(BAND / f"L{layer}a_lr_raw_s0.pkl")] if units is None else [matrix(EXP / f"L{layer}a_lr_{u}.pkl") for u in units]
        m = sum(ms) / len(ms)
        mats[(layer, arm)] = m
        summ.append({"layer": layer, "arm": arm, "transfer": m.values[off].mean(), "within": np.diag(m.values).mean(),
                     "inverted": int((m.values[off] < 0.5).sum())})
        long += [{"layer": layer, "arm": arm, "train": a, "test": b, "auroc": m.loc[a, b]} for a in S for b in S]
out = C.results_dir("exploratory")
pd.DataFrame(long).to_csv(out / "protocol_c_band_170.csv", index=False)
s = pd.DataFrame(summ)
pd.set_option("display.width", 220)
t = s.pivot(index="arm", columns="layer", values="transfer").reindex(list(arms))
t.loc["all-170 minus shuffled"] = t.loc["all 170 emotion PCs"] - t.loc["170 label-shuffled PCs (5)"]
t.loc["all-170 minus raw"] = t.loc["all 170 emotion PCs"] - t.loc["raw activations"]
print("MEAN TRANSFER AUROC (train on one set, test on each other; 20 cells)")
print(t.round(3).to_string())
print("\nINVERTED CELLS (< 0.5) of 20")
print(s.pivot(index="arm", columns="layer", values="inverted").reindex(list(arms)).to_string())
rows = {L: [mats[(L, 'all 170 emotion PCs')].loc[a, [b for b in S if b != a]].mean() for a in S] for L in range(35, 46)}
print("\nALL-170: mean transfer by TRAINING set")
print(pd.DataFrame(rows, index=[f"train {a}" for a in S]).round(2).to_string())
for L in (38, 41, 43, 45):
    print(f"\nall 170 emotion PCs, layer {L} (rows train, cols test; transfer {mats[(L, 'all 170 emotion PCs')].values[off].mean():.3f})")
    print(mats[(L, 'all 170 emotion PCs')].round(2).to_string())
