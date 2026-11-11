"""Exploratory (user request): protocol C (train on one set, test on the others; diagonal by 5-fold CV) for the
all-170-emotion-PC and all-171-emotion-vector arms, which phase 3 ran under protocol B only."""
import pickle
import sys
import time

import numpy as np

from extensions.emotion_coords import config as C
from extensions.emotion_coords import lbdata as LB
from extensions.emotion_coords import linalg as la
from extensions.emotion_coords import probes as P

out = C.data_dir("phase3", "explore_vectors")
s = C.load()
lb5, control = LB.lb5(), s["control"]
for layer in [int(x) for x in sys.argv[1].split(",")]:
    with np.load(C.data_dir("bases") / f"full_L{layer}.npz") as z:
        e, v = z["E_components"], z["E_vectors"]
    sets = [P.FeatureSet("emotion_all", "exploratory_emotion_all", e, (len(e),)),
            P.FeatureSet("emotion_vectors_all", "exploratory_emotion_vectors_all", la.unit(v), (len(v),))]
    todo = [fs for fs in sets if not (out / f"L{layer}a_lr_{fs.name}_C.pkl").exists()]
    if not todo:
        continue
    data = LB.load(layer, "a", LB.subset("full"))
    for fs in todo:
        t0, store = time.time(), P.Store()
        P.protocol_c("lr", fs, data, lb5, control, layer, "a", store)
        with open(out / f"L{layer}a_lr_{fs.name}_C.pkl", "wb") as f:
            pickle.dump({"rows": store.rows, "scores": store.scores, "seconds": time.time() - t0}, f)
        print(f"L{layer} {fs.name}: protocol C in {time.time() - t0:.0f} s", flush=True)
