"""Protocol-B comparisons on stored held-out scores: mean held-out AUROC per feature set, and paired bootstrap
differences against the controls at the emotion basis's chosen k (the P1-P4 machinery).

For each held-out set the emotion basis's k is the one its inner leave-one-set-out CV chose in that fold; the
random-subspace control at that k is the mean AUROC over its seeds; the topic basis is taken at the same k. A
paired bootstrap resamples each held-out set (stratified, shared weights for both arms) and averages over sets.
"""

import numpy as np
import pandas as pd

from emotion_coords.probes import Resamples, auroc, ci


def _key(row: pd.Series, fields=("protocol", "feature_set", "fs_family", "seed", "layer", "pooling", "train",
                                 "probe", "k", "test")) -> tuple:
    return tuple(sorted((f, row[f]) for f in fields))


def chosen_k(frame: pd.DataFrame, family: str, test: str) -> int | None:
    pick = frame[(frame["fs_family"] == family) & (frame["test"] == test) & frame["chosen"]]
    return int(pick["k"].iloc[0]) if len(pick) else None


def held_out_scores(frame: pd.DataFrame, scores: dict, feature_set: str, test: str, k: int,
                    seeds=None) -> list[np.ndarray]:
    rows = frame[(frame["feature_set"] == feature_set) & (frame["test"] == test) & (frame["k"] == k)]
    if seeds is not None:
        rows = rows[rows["seed"].isin(seeds)]
    return [scores[_key(r)] for _, r in rows.iterrows()]


def protocol_b_table(frame: pd.DataFrame, scores: dict, ys: dict[str, np.ndarray], layer: int, pooling: str,
                     probe: str = "lr", n_boot: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-set table, summary with paired CIs) for one layer, pooling and probe family."""
    b = frame[(frame["protocol"] == "B") & (frame["layer"] == layer) & (frame["pooling"] == pooling)
              & (frame["probe"] == probe)]
    tests = list(ys)
    per_set, deltas = [], {"vs random mean": [], "vs topic": [], "vs persona (own k)": [], "vs raw": []}
    for t in tests:
        y = ys[t]
        k_e = chosen_k(b, "emotion_pc", t)
        arms = {
            "emotion_pc": held_out_scores(b, scores, "emotion_pc", t, k_e)[0],
            "random (mean over seeds)": None,
            "topic_pc": held_out_scores(b, scores, "topic_pc", t, k_e),
            "persona_pc (own k)": held_out_scores(b, scores, "persona_pc", t, chosen_k(b, "persona_pc", t))[0],
            "raw": held_out_scores(b, scores, "raw", t, int(b.loc[b["feature_set"] == "raw", "k"].iloc[0]))[0],
        }
        random_scores = held_out_scores(b, scores, "random", t, k_e)
        boot = Resamples.draw(y, n_boot)
        auc = {name: boot.aurocs(y, s) for name, s in arms.items() if isinstance(s, np.ndarray)}
        auc["random (mean over seeds)"] = np.mean([boot.aurocs(y, s) for s in random_scores], axis=0)
        topic = arms["topic_pc"]
        if topic:
            auc["topic_pc"] = boot.aurocs(y, topic[0])
        point = {"emotion_pc": auroc(y, arms["emotion_pc"]),
                 "random (mean over seeds)": float(np.mean([auroc(y, s) for s in random_scores])),
                 "topic_pc": auroc(y, topic[0]) if topic else np.nan,
                 "persona_pc (own k)": auroc(y, arms["persona_pc (own k)"]), "raw": auroc(y, arms["raw"])}
        per_set.append({"test": t, "k_emotion": k_e, "k_persona": chosen_k(b, "persona_pc", t),
                        "n_random_seeds": len(random_scores)} | point)
        deltas["vs random mean"].append(auc["emotion_pc"] - auc["random (mean over seeds)"])
        deltas["vs topic"].append(auc["emotion_pc"] - auc.get("topic_pc", np.full(n_boot, np.nan)))
        deltas["vs persona (own k)"].append(auc["emotion_pc"] - auc["persona_pc (own k)"])
        deltas["vs raw"].append(auc["emotion_pc"] - auc["raw"])
    per_set = pd.DataFrame(per_set)
    summary = []
    for name, d in deltas.items():
        mean_over_sets = np.mean(np.stack(d), axis=0)
        lo, hi = ci(mean_over_sets[np.isfinite(mean_over_sets)]) if np.isfinite(mean_over_sets).any() else (np.nan, np.nan)
        summary.append({"comparison": f"emotion_pc {name}", "layer": layer, "pooling": pooling, "probe": probe,
                        "mean_delta": float(np.nanmean(mean_over_sets)), "ci_lo": lo, "ci_hi": hi})
    means = per_set.drop(columns=["test", "k_emotion", "k_persona", "n_random_seeds"]).mean()
    summary.append({"comparison": "mean held-out AUROC", "layer": layer, "pooling": pooling, "probe": probe}
                   | {f"mean_{c}": float(v) for c, v in means.items()})
    return per_set, pd.DataFrame(summary)
