"""Exploratory: every emotion direction as a zero-shot lie score at one layer (pooling a, the capped rows).
Score = activation . unit(emotion vector); AUROC with lies as positives (> 0.5: lies carry more of the emotion)."""
import sys

import numpy as np
import pandas as pd

from emotion_coords import config as C
from emotion_coords import lbdata as LB
from emotion_coords import linalg as la
from emotion_coords import probes as P

layer = int(sys.argv[1])
SHORT = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C",
         "harm_refusal_knowledge_report__plain": "HP-KR", "insider_trading_cadenza__plain": "IT",
         "instruct_dishonesty__plain": "ID"}
with np.load(C.data_dir("bases") / f"full_L{layer}.npz") as z:
    vectors, labels = z["E_vectors"], [str(x) for x in z["E_labels"]]
data = LB.load(layer, "a", LB.subset("full"))
units = la.unit(vectors)
rows = []
for i, name in enumerate(labels):
    rows.append({"emotion": name, **{short: P.auroc(data[t].y, data[t].x @ units[i]) for t, short in SHORT.items()}})
tab = pd.DataFrame(rows).set_index("emotion")
tab["mean"] = tab[list(SHORT.values())].mean(axis=1)
tab["mean_|.5|"] = (tab[list(SHORT.values())] - 0.5).abs().mean(axis=1)
out = C.results_dir("exploratory") / f"zero_shot_all_emotions_L{layer}.csv"
tab.sort_values("mean", ascending=False).to_csv(out)
pd.set_option("display.width", 200)
print(f"layer {layer}: {len(tab)} emotions -> {out}")
print("\nTop 15 by mean AUROC (lies carry MORE of the emotion):")
print(tab.sort_values("mean", ascending=False).head(15).round(3).to_string())
print("\nBottom 15 by mean AUROC (lies carry LESS of the emotion):")
print(tab.sort_values("mean").head(15).round(3).to_string())
core = C.load()["emotion_names"]["core"]
print("\nCore emotions (preregistered list):")
print(tab.loc[[c for c in core if c in tab.index]].sort_values("mean", ascending=False).round(3).to_string())
print("\nDistribution of mean AUROC over all 171:", tab["mean"].describe().round(3).to_dict())
