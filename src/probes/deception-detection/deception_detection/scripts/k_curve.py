"""Would more questions help K without HP-KR's template? Learning curves and split-half on saved reads.

For K with HP-KR's template ("original") and the two rebuilds without it ("report", "open"), at
layers 36/40/44, using only reads already on disk:

- learning curve: K rebuilt from n known + n unknown questions drawn without replacement from the
  60 + 60, scored on every set; if the score is still rising at 60, more questions would help;
- extrapolation: for small noise, E[AUROC(n)] = A_inf + c/n, so subsets of the fixed 60 give
  E[AUROC_sub(n)] - AUROC(60) = c (1/n - 1/60); c fitted on n = 20..50 gives the expected gain from
  doubling (-c/120) and from unlimited questions of the same kind (-c/60);
- question bootstrap at 60 (with replacement): how much the score depends on which questions;
- split-half cosine: by item parity (the rule `recall_denial.directions` uses) and over random
  30/30 splits; the implied cosine between the 60-question K and the unlimited-question K.

    python deception_detection/scripts/k_curve.py

Outputs go to `results/recall_denial/k_template/curve/`.
"""

import time

import fire
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from deception_detection.scripts import k_template as kt
from deception_detection.scripts import recall_denial as rd
from deception_detection.scripts.hpkr_wording import hpkr_frame
from deception_detection.scripts.knowledge_bank import _slices
from deception_detection.scripts.layer_sweep import pooled_dir

OUT = kt.OUT / "curve"
LAYERS = (36, 40, 44)
VERSIONS = ("original", "report", "open")
N_FULL = 60
NS = (10, 15, 20, 30, 40, 50)
FIT_NS = (20, 30, 40, 50)
B_SUB = 300
B_BOOT = 500
B_SPLIT = 500
SEED = 0
FIVE = tuple(rd.SHORT[n] for n in kt.FIVE)
SETS = ("HP-KR identical", "HP-KR denials", "Gender-Secret") + FIVE


def item_arrays(version: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{layer: (known [60, hidden], unknown [60, hidden])}, items in sorted-id order, each the
    mean of its two reads."""
    if version == "original":
        rows = [r for r in rd._rows() if r["subject"] == "general" and r["arm"] == "none"]
        reads = rd._load(rd.OUT / "parts_answer")
    else:
        rows = kt.build_rows(kt.items(version), version)
        reads = rd._load(kt.OUT / f"parts_{version}")
    by_item, knows = {}, {}
    for r in rows:
        by_item.setdefault(r["item"], []).append(r["row"])
        knows[r["item"]] = bool(r["knows"])
    ids = sorted(by_item)
    if any(len(by_item[i]) != 2 for i in ids) or len(ids) != 2 * N_FULL:
        raise RuntimeError(f"{version}: expected {2 * N_FULL} items with two reads each")
    known = [i for i in ids if knows[i]]
    unknown = [i for i in ids if not knows[i]]
    out = {}
    for layer in LAYERS:
        mean = {i: np.mean([reads[row][layer] for row in by_item[i]], axis=0) for i in ids}
        out[layer] = (np.stack([mean[i] for i in known]), np.stack([mean[i] for i in unknown]))
    return out


def unit_rows(d: np.ndarray) -> np.ndarray:
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def weights(rng, n: int, B: int, replace: bool) -> np.ndarray:
    """B rows of item weights, each 1/n on n items drawn from the 60."""
    W = np.zeros((B, N_FULL))
    for b in range(B):
        np.add.at(W[b], rng.choice(N_FULL, n, replace=replace), 1.0 / n)
    return W


def aurocs(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """AUROC of every column of S (ties averaged)."""
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    out = np.empty(S.shape[1])
    for c in range(0, S.shape[1], 400):
        R = rankdata(S[:, c:c + 400], axis=0)
        out[c:c + 400] = (R[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def eval_sets(layer: int, frame: pd.DataFrame, slices: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """As `k_template.score` builds them."""
    sets = {}
    for name in kt.FIVE:
        raw = np.load(pooled_dir(layer) / f"{name}.npz")
        labels = raw["labels"].astype(str)
        keep = np.isin(labels, ["HONEST", "DECEPTIVE"])
        sets[rd.SHORT[name]] = ((labels[keep] == "DECEPTIVE").astype(int), raw["pooled"][keep].astype(np.float32))
    y, z = sets["HP-KR"]
    if not (frame["lie"].to_numpy() == y).all():
        raise RuntimeError("HP-KR rows do not line up with hpkr_frame")
    sets["HP-KR identical"] = (y[slices["identical text"]], z[slices["identical text"]])
    sets["HP-KR denials"] = (y[slices["denials"]], z[slices["denials"]])
    gs = rd._gs_reads(layer)["adapted"]
    sets["Gender-Secret"] = ((gs["labels"].astype(str) == "DECEPTIVE").astype(int), gs["pooled"].astype(np.float32))
    return sets


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = hpkr_frame()
    slices = _slices(frame)
    arrays = {v: item_arrays(v) for v in VERSIONS}
    stored = dict(np.load(rd.OUT / "directions.npz"))

    # one set of draws, shared by every version and layer, so the versions are compared on the same questions
    rng = np.random.default_rng(SEED)
    plan = [("full", N_FULL, np.full((1, N_FULL), 1.0 / N_FULL), np.full((1, N_FULL), 1.0 / N_FULL))]
    for n in NS:
        plan.append(("sub", n, weights(rng, n, B_SUB, False), weights(rng, n, B_SUB, False)))
    plan.append(("boot", N_FULL, weights(rng, N_FULL, B_BOOT, True), weights(rng, N_FULL, B_BOOT, True)))
    splits = [(rng.permutation(N_FULL), rng.permutation(N_FULL)) for _ in range(B_SPLIT)]

    curve, full_rows, split_rows, cos_rows = [], [], [], []
    for layer in LAYERS:
        start = time.time()
        sets = eval_sets(layer, frame, slices)
        fulls = {}
        for version in VERSIONS:
            Xk, Xu = arrays[version][layer]
            blocks = [(kind, n, unit_rows(Wk @ Xk - Wu @ Xu)) for kind, n, Wk, Wu in plan]
            D = np.concatenate([d for _, _, d in blocks]).astype(np.float32)
            k60 = blocks[0][2][0]
            fulls[version] = k60
            if version == "original":
                c = float(k60 @ stored[f"L{layer}|none|general"])
                if c < 0.99999:
                    raise RuntimeError(f"original K at L{layer} does not match the stored K (cos {c})")
            # split halves: item parity on sorted ids (recall_denial's rule), and random 30/30 splits
            even, odd = np.arange(0, N_FULL, 2), np.arange(1, N_FULL, 2)
            ha = Xk[even].mean(0) - Xu[even].mean(0)
            hb = Xk[odd].mean(0) - Xu[odd].mean(0)
            parity = float(ha @ hb / np.linalg.norm(ha) / np.linalg.norm(hb))
            rand = []
            for pk, pu in splits:
                a = Xk[pk[:30]].mean(0) - Xu[pu[:30]].mean(0)
                b = Xk[pk[30:]].mean(0) - Xu[pu[30:]].mean(0)
                rand.append(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
            rand = np.array(rand)
            c_half = float(rand.mean())
            split_rows.append({"layer": layer, "version": version, "parity split-half": parity,
                               "random split-half mean": c_half, "random split-half p5": float(np.quantile(rand, 0.05)),
                               "implied cos(K60, K_unlimited)": 1 / np.sqrt(1 + (1 / c_half - 1) / 2)})
            # how close a K from n questions per side is to the K from all 60
            for kind, n, d in blocks[1:]:
                if kind == "sub":
                    cos_rows.append({"layer": layer, "version": version, "n": n,
                                     "mean cos(K_n, K60)": float((d @ k60).mean())})
            # score every direction on every set
            per_set = {}
            for name in SETS:
                y, z = sets[name]
                per_set[name] = aurocs(y, z @ D.T)
            per_set["five-dataset mean"] = np.mean([per_set[s] for s in FIVE], axis=0)
            for name, a in per_set.items():
                pos = 0
                means = {}
                for kind, n, d in blocks:
                    vals = a[pos:pos + len(d)]
                    pos += len(d)
                    if kind == "full":
                        a60 = float(vals[0])
                        curve.append({"layer": layer, "version": version, "set": name, "n": N_FULL,
                                      "mean": a60, "p5": a60, "p95": a60})
                    elif kind == "sub":
                        means[n] = float(vals.mean())
                        curve.append({"layer": layer, "version": version, "set": name, "n": n,
                                      "mean": means[n], "p5": float(np.quantile(vals, 0.05)),
                                      "p95": float(np.quantile(vals, 0.95))})
                    else:
                        boot = vals
                x = np.array([1 / n - 1 / N_FULL for n in FIT_NS])
                m = np.array([means[n] - a60 for n in FIT_NS])
                c = float(x @ m / (x @ x))
                full_rows.append({"layer": layer, "version": version, "set": name, "auroc 60": a60,
                                  "boot lo": float(np.quantile(boot, 0.025)), "boot hi": float(np.quantile(boot, 0.975)),
                                  "mean at 30": means[30], "expected gain 60->120": -c / 120,
                                  "expected gain 60->unlimited": -c / N_FULL})
        for i, a in enumerate(VERSIONS):
            for b in VERSIONS[i + 1:]:
                cos_rows.append({"layer": layer, "version": f"{a} . {b}", "n": N_FULL,
                                 "mean cos(K_n, K60)": float(fulls[a] @ fulls[b])})
        print(f"layer {layer} done in {time.time() - start:.0f} s", flush=True)

    curve, full_t = pd.DataFrame(curve), pd.DataFrame(full_rows)
    split_t, cos_t = pd.DataFrame(split_rows), pd.DataFrame(cos_rows)
    curve.round(4).to_csv(OUT / "curve.csv", index=False)
    full_t.round(4).to_csv(OUT / "full.csv", index=False)
    split_t.round(5).to_csv(OUT / "split_half.csv", index=False)
    cos_t.round(4).to_csv(OUT / "cosines.csv", index=False)
    pd.set_option("display.width", 250)
    print(split_t.round(4).to_string(index=False))
    print(full_t.round(3).to_string(index=False))
    for layer in LAYERS:
        c = curve[(curve.layer == layer) & curve.set.isin(["HP-KR identical", "HP-KR denials", "Gender-Secret", "five-dataset mean"])]
        print(f"\nlayer {layer}, mean AUROC by questions per side:")
        print(c.pivot_table(index=["set", "version"], columns="n", values="mean").round(3).to_string())


if __name__ == "__main__":
    fire.Fire(main)
