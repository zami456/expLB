"""Gender-Secret with the emotion vectors rebuilt inside the fine-tune (ANALYSIS_PLAN.md, "Gender-Secret with the
fine-tune's emotion vectors", fixed before any of these arms is fitted).

NDIF attaches Liars' Bench's Gender-Secret LoRA itself (nnsight 0.8's `peft=`). So the 8,280 stories and the neutral
dialogues are read through the fine-tune at block 40, and phase 1's recipe builds the bases from those reads. The
probes are fitted on the base model's LB-5 reads exactly as in `gs_test`. Only the arms whose basis comes from the
stories are refitted; the others are `gs_test`'s own parts. GS's two reads (adapted and base) are the thesis's.

    python -m emotion_coords.gs_ft parity     # NDIF's peft read of the GS rows against the thesis's stored reads
    python -m emotion_coords.gs_ft bases      # the fine-tune's bases at block 40, compared with the base model's
    python -m emotion_coords.gs_ft run        # the story-dependent arms, fitted on LB-5, scored on both GS reads
    python -m emotion_coords.gs_ft summary    # G1-G3 with the fine-tune's bases; each arm fine-tune minus base basis
"""

import json
import os
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from emotion_coords import config as C
from emotion_coords import gs_test as G
from emotion_coords import lbdata as LB
from emotion_coords import linalg as la
from emotion_coords import probes as P
from emotion_coords import vectors as V
from emotion_coords.phase3 import _write, exploratory_sets, feature_sets
from emotion_coords.pilot import ZERO_SHOT

ADAPTER = "Cadenza-Labs/llama-70b-3.3-it-lora-gender-secret-male"
LAYER = 40
MIN_COS = 0.9998
READS = {"stories": "stories_gsft40", "neutral": "neutral_gsft40", "gs_adapted": "gs_peft40", "gs_base": "gs_base40"}
# arms whose basis comes from the story reads; every other arm (raw, random, Alpaca, training-set and persona PCs)
# does not depend on them and is taken from gs_test's parts
STORY_FAMILIES = ("emotion_pc", "topic_pc", "shuffled_pc", "persona_emotion", "persona_pub_emotion", "named",
                  "exploratory_emotion_all", "exploratory_shuffled_all", "exploratory_emotion_vectors_all", "zero_shot")


def _parts() -> Path:
    return C.data_dir("gs", "parts_ft")


def _out() -> Path:
    return C.results_dir("gs_ft")


def bases_path() -> Path:
    return C.data_dir("bases") / f"gsft_L{LAYER}.npz"


def parity() -> dict:
    """Gate: the GS rows read through NDIF's peft fine-tune (and through the base model) against the thesis's stored
    reads of the same rows at block 40 (`activations/gender_secret/`)."""
    out = {}
    for read, folder in (("adapted", READS["gs_adapted"]), ("base", READS["gs_base"])):
        ids, content, _ = V.load_reads(folder, V._layer_index(LAYER, folder))
        with np.load(G.GS_DIR / f"{read}_L{LAYER}.npz") as z:
            stored, index = z["pooled"].astype(np.float64), z["index"]
        position = {int(i): j for j, i in enumerate(index)}
        cos = [float(la.unit(content[k, 0, 0]) @ la.unit(stored[position[int(r)]])) for k, r in enumerate(ids)]
        out[read] = {"rows": len(cos), "min_cos": min(cos), "median_cos": float(np.median(cos))}
    out["pass"] = bool(out["adapted"]["min_cos"] >= MIN_COS)
    (_out() / "parity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    logger.info(f"GS peft parity: {out}")
    if not out["pass"]:
        raise SystemExit(f"GATE parity: min cos {out['adapted']['min_cos']:.6f} < {MIN_COS}")
    return out


def bases() -> dict:
    """The fine-tune's bases by phase 1's recipe (the top-ups as Llama's), checks, and the comparison with the base
    model's own bases at the same block (`bases/full_L40.npz`)."""
    from emotion_coords import phase1
    from emotion_coords.aletheia170 import build_bases

    stories, neutral = V.Reads.of(READS["stories"], LAYER), V.Reads.of(READS["neutral"], LAYER)
    emotion, shuffled, denoise, balanced = build_bases(stories, neutral)
    topic = V.topic_basis(balanced, denoise)
    arrays = {"E_vectors": emotion.vectors, "E_components": emotion.components, "E_explained": emotion.explained,
              "E_labels": np.array(emotion.labels), "T_vectors": topic.vectors, "T_components": topic.components,
              "T_explained": topic.explained, "T_labels": np.array(topic.labels), "denoise": denoise}
    for p, b in enumerate(shuffled):
        arrays[f"S{p}_components"] = b.components
    np.savez(bases_path(), **arrays)
    with np.load(C.data_dir("bases") / f"full_L{LAYER}.npz") as z:
        base_vectors, base_components, base_labels = z["E_vectors"], z["E_components"], list(z["E_labels"])
    if base_labels != emotion.labels:
        raise RuntimeError("the two bases list different emotions")
    per_emotion = np.sum(la.unit(emotion.vectors) * la.unit(base_vectors), axis=1)
    principal = {k: float(np.linalg.svd(emotion.components[:k] @ base_components[:k].T, compute_uv=False).mean())
                 for k in (2, 5, 10, 20, len(emotion.components))}
    geometry = V.geometry(emotion)
    split = V.split_half(balanced, denoise, seed=C.load()["seeds"]["stories"])
    out = {"layer": LAYER, "adapter": ADAPTER, "stories": len(stories.x), "neutral": len(neutral.x),
           "neutral_pcs_removed": len(denoise), "rank": len(emotion.components), "geometry": geometry,
           "split_half_median": float(split["cos"].median()), "neighbours": V.neighbours(emotion, phase1.NEIGHBOURS),
           "cos_with_base_vectors": {"median": float(np.median(per_emotion)), "min": float(per_emotion.min()),
                                     "lowest": {emotion.labels[i]: float(per_emotion[i])
                                                for i in np.argsort(per_emotion)[:5]}},
           "mean_principal_cos_with_base_subspace": principal,
           "valence_gate": bool(abs(geometry["r_pc1_valence"]) >= 0.5)}
    (_out() / "bases.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    logger.info(f"fine-tune bases: {json.dumps(out, indent=1)}")
    if not out["valence_gate"]:
        raise SystemExit(f"GATE valence: |r(PC1, valence)| {abs(geometry['r_pc1_valence']):.3f} < 0.5")
    return out


def run(workers: int = 3, threads: int = 3, n_random: int = 20) -> None:
    """The story-dependent arms with the fine-tune's bases: fitted on LB-5 as `gs_test` fits them, scored on GS."""
    from joblib import Parallel, delayed

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    s = C.load()
    data = LB.load(LAYER, "a", LB.full_subset())
    gs = G.gs_splits(LAYER)
    with np.load(bases_path()) as z:
        base = {k: z[k] for k in z.files}
    hidden = data[s["control"]].x.shape[1]
    sets = feature_sets(LAYER, data, base, hidden, n_random) + exploratory_sets(data, base, hidden)
    sets.append(P.FeatureSet("emotion_vectors_all", "exploratory_emotion_vectors_all", la.unit(base["E_vectors"]),
                             (len(base["E_vectors"]),)))
    sets = [fs for fs in sets if fs.family in STORY_FAMILIES]
    name = lambda fs: f"L{LAYER}_{fs.name}_s{fs.seed}.pkl"  # noqa: E731
    todo = [fs for fs in sets if not (_parts() / name(fs)).exists()]
    logger.info(f"GS with the fine-tune's bases: {len(sets)} arms, {len(todo)} to fit on {workers} workers")
    done = Parallel(n_jobs=workers, backend="loky", temp_folder=str(C.data_dir("gs", "joblib_tmp")),
                    max_nbytes="1M")(delayed(G._unit)(fs, data, gs, LAYER, str(_parts() / name(fs))) for fs in todo)
    for unit, seconds in done:
        logger.info(f"  {unit}: {seconds:.0f} s")
    store = P.Store()
    ctrl_x = data[s["control"]].x
    labels = list(base["E_labels"])
    for e, sign in ZERO_SHOT.items():
        if e in labels:
            u = la.unit(base["E_vectors"][labels.index(e)])
            for t, split in gs.items():
                store.add({"protocol": "zero-shot", "feature_set": f"zs_{e}", "fs_family": "zero_shot", "seed": 0,
                           "layer": LAYER, "pooling": "a", "train": "none", "probe": "projection", "k": 1,
                           "test": t}, split.y, sign * (split.x @ u), sign * (ctrl_x @ u), chosen=True)
    _write(_parts() / f"L{LAYER}_zero_shot.pkl", {"rows": store.rows, "scores": store.scores, "seconds": 0.0})


def summary(n_boot: int = 1000) -> dict:
    """G1-G3 with the fine-tune's bases (the other arms from `gs_test`), and each story-dependent arm's AUROC with
    the fine-tune's bases minus with the base model's, paired on the same resamples."""
    ft_rows, ft_scores = G.load_parts(_parts(), LAYER)
    other = lambda r: r["fs_family"] not in STORY_FAMILIES  # noqa: E731
    base_rows, base_scores = G.load_parts(G._parts(), LAYER, keep=other)
    result = G.evaluate(ft_rows + base_rows, ft_scores | base_scores, LAYER, n_boot, _out())
    old_rows, old_scores = G.load_parts(G._parts(), LAYER, keep=lambda r: not other(r))
    gs = G.gs_splits(LAYER)
    y = gs["GS"].y
    boots = P.Resamples.draw(y, n_boot)  # the same draws as `evaluate`'s (same seed, same labels)
    ft = G.arm_fn(pd.DataFrame(ft_rows), ft_scores, y, boots)
    old = G.arm_fn(pd.DataFrame(old_rows), old_scores, y, boots)
    compare = []
    for fs in dict.fromkeys(r["feature_set"] for r in ft_rows):
        for test in ("GS", "GS-base"):
            a, b = ft(fs, test), old(fs, test)
            compare.append({"arm": fs, "test": test, "fine_tune_basis": a[0], "base_basis": b[0]}
                           | G.delta(a, b))
    table = pd.DataFrame(compare)
    table.to_csv(_out() / f"ft_minus_base_L{LAYER}.csv", index=False)
    logger.info("\n" + table[table["test"] == "GS"].round(3).to_string(index=False))
    return result


if __name__ == "__main__":
    fire.Fire({"parity": parity, "bases": bases, "run": run, "summary": summary})
