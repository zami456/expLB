"""Exploratory (user request): protocol C for the 170-d arms (all emotion PCs, 5 label-shuffled bases, Alpaca PCs)
at every band block, which phase 3 ran under protocol B only. Parts: data/emotion_coords/phase3/explore_c_band/."""
import os
import pickle
import sys
import time

os.environ.update({v: "3" for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from emotion_coords import config as C  # noqa: E402
from emotion_coords import lbdata as LB  # noqa: E402
from emotion_coords import probes as P  # noqa: E402
from emotion_coords.phase3 import exploratory_sets  # noqa: E402

out = C.data_dir("phase3", "explore_c_band")
s = C.load()
lb5, control = LB.lb5(), s["control"]


def unit(fs, data, layer, path):
    t0, store = time.time(), P.Store()
    P.protocol_c("lr", fs, data, lb5, control, layer, "a", store)
    with open(path, "wb") as f:
        pickle.dump({"rows": store.rows, "scores": store.scores, "seconds": time.time() - t0}, f)
    return time.time() - t0


for layer in [int(x) for x in sys.argv[1].split(",")]:
    data = LB.load(layer, "a", LB.full_subset())
    with np.load(C.data_dir("bases") / f"full_L{layer}.npz") as z:
        base = {k: z[k] for k in z.files}
    sets = [fs for fs in exploratory_sets(data, base, data[control].x.shape[1])
            if fs.name in ("emotion_all", "shuffled_all", "alpaca_all")]
    todo = [fs for fs in sets if not (out / f"L{layer}a_lr_{fs.name}_s{fs.seed}_C.pkl").exists()]
    t0 = time.time()
    Parallel(n_jobs=4, backend="loky", max_nbytes="1M")(
        delayed(unit)(fs, data, layer, str(out / f"L{layer}a_lr_{fs.name}_s{fs.seed}_C.pkl")) for fs in todo)
    print(f"L{layer}: {len(todo)} protocol-C units in {time.time() - t0:.0f} s", flush=True)
