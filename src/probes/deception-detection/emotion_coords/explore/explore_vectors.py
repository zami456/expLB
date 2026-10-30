"""Exploratory (user request): probe on the coordinates along all 171 emotion vectors (unit, non-orthogonal)
instead of their 170 PCs. Same subspace; only the L2 penalty's parameterisation differs. Protocol B, lr."""
import sys

import numpy as np

from emotion_coords import config as C
from emotion_coords import lbdata as LB
from emotion_coords import linalg as la
from emotion_coords import probes as P
from emotion_coords.phase3 import _unit

out = C.data_dir("phase3", "explore_vectors")
for layer in [int(x) for x in sys.argv[1].split(",")]:
    path = out / f"L{layer}a_lr_emotion_vectors_all_s0.pkl"
    if path.exists():
        continue
    data = LB.load(layer, "a", LB.subset("full"))
    with np.load(C.data_dir("bases") / f"full_L{layer}.npz") as z:
        v = z["E_vectors"]
    fs = P.FeatureSet("emotion_vectors_all", "exploratory_emotion_vectors_all", la.unit(v), (len(v),))
    name, seconds = _unit("lr", fs, data, layer, "a", str(path), None, "B")
    print(f"L{layer}: {name} in {seconds:.0f} s", flush=True)
