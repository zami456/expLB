"""Phase 1 of the emotion-coordinate study: the full emotion basis, its controls and its sanity checks.

    python -m extensions.emotion_coords.phase1 base    # after the 171 x 40 stories are read: split-half at every layer,
                                            # and the top-up list (core emotions under 0.7 at any layer)
    python -m extensions.emotion_coords.phase1 final   # after the top-ups are read (or with none): final bases and checks

`final` writes data/emotion_coords/bases/full_L{layer}.npz and results/emotion_coords/phase1/. The emotion
vectors use one story per emotion and topic (rank 0, the balanced 171 x 40 design); a top-up emotion's own
vector also uses its ranks 1 and 2 (up to 120 stories). The topic basis, the label-shuffled bases and LEACE use
the balanced rank-0 stories only. The valence check stops the run: |r(PC1, valence)| < 0.5 at layer 52 raises.
"""

import json

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from extensions.emotion_coords import config as C
from extensions.emotion_coords import linalg as la
from extensions.emotion_coords import vectors as V

TOPUP_RANKS = (1, 2)
NEIGHBOURS = ["afraid", "calm", "guilty", "compassionate", "suspicious", "desperate"]


def _out(name: str):
    return C.results_dir("phase1") / name


def _topup_path():
    return C.data_dir() / "topups.json"


def _json(path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")


def base() -> None:
    """Split-half reliability of every emotion from rank-0 stories (20 v 20 topics) at each layer; the core
    emotions under 0.7 at any layer get top-up stories."""
    s = C.load()
    core = s["emotion_names"]["core"]
    tables, low = [], set()
    for layer in s["layers"]:
        stories, neutral = V.Reads.of("stories", layer), V.Reads.of("neutral", layer)
        stories = V.Reads(stories.x[(stories.meta["rank"] == 0).to_numpy()],
                          stories.meta[stories.meta["rank"] == 0].reset_index(drop=True))
        split = V.split_half(stories, V.neutral_basis(neutral), seed=s["seeds"]["stories"])
        tables.append(split.assign(layer=layer))
        low |= set(split.loc[split["emotion"].isin(core) & (split["cos"] < 0.7), "emotion"])
        logger.info(f"L{layer}: split-half median {split['cos'].median():.3f}, core under 0.7: "
                    f"{sorted(split.loc[split['emotion'].isin(core) & (split['cos'] < 0.7), 'emotion'])}")
    pd.concat(tables).to_csv(_out("split_half_base.csv"), index=False)
    _topup_path().write_text(json.dumps(sorted(low)), encoding="utf-8")
    logger.info(f"top-ups for {len(low)} core emotions: {sorted(low)} -> {_topup_path()}")


def final(layers=None, suffix: str = "") -> None:
    """`layers` and `suffix` run the same recipe on another read set: the post hoc band around block 40
    (`final --layers 32,36,44,48 --suffix _band` reads stories_band, neutral_band and tylenol_band and writes
    phase1_band.json; the valence stop applies only when block 52 is among the layers)."""
    s = C.load()
    todo = s["layers"] if layers is None else C.int_list(layers)
    topups = json.loads(_topup_path().read_text()) if _topup_path().exists() else []
    extra = {e: TOPUP_RANKS for e in topups}
    seed = s["seeds"]["stories"]
    out = {"topups": topups}
    for layer in todo:
        stories, neutral = V.Reads.of(f"stories{suffix}", layer), V.Reads.of(f"neutral{suffix}", layer)
        if topups:
            have = stories.meta.loc[stories.meta["rank"] > 0, "emotion"].value_counts()
            missing = [e for e in topups if have.get(e, 0) == 0]
            if missing:
                raise RuntimeError(f"top-up stories not read yet for {missing}")
        rank0 = (stories.meta["rank"] == 0).to_numpy()
        balanced = V.Reads(stories.x[rank0], stories.meta[rank0].reset_index(drop=True))
        denoise = V.neutral_basis(neutral)
        e = V.emotion_basis(stories, denoise, extra=extra)
        t = V.topic_basis(balanced, denoise)
        shuffled = V.shuffled_bases(balanced, denoise, n=5, seed=seed)
        eraser = la.fit_leace(balanced.x, balanced.meta["emotion"].to_numpy())
        split_base = V.split_half(balanced, denoise, seed=seed)
        split_final = V.split_half(stories, denoise, seed=seed, extra=extra) if extra else split_base
        split = split_base.merge(split_final, on="emotion", suffixes=("_20v20", "_final"))
        split.to_csv(_out(f"split_half_L{layer}.csv"), index=False)
        dose = V.dose_response(e, layer, reads=f"tylenol{suffix}")
        dose.to_csv(_out(f"dose_response_L{layer}.csv"), index=False)
        arrays = {"E_vectors": e.vectors, "E_components": e.components, "E_explained": e.explained,
                  "E_labels": np.array(e.labels), "T_vectors": t.vectors, "T_components": t.components,
                  "T_explained": t.explained, "T_labels": np.array(t.labels), "denoise": denoise,
                  "leace_a": eraser.a, "leace_b": eraser.b, "leace_mean": eraser.mean,
                  "leace_shrinkage": eraser.shrinkage}
        for p, b in enumerate(shuffled):
            arrays[f"S{p}_components"] = b.components
        np.savez(C.data_dir("bases") / f"full_L{layer}.npz", **arrays)
        geometry = V.geometry(e)
        core = s["emotion_names"]["core"]
        out[layer] = {
            "stories_rank0": int(rank0.sum()), "stories_total": len(stories.x), "neutral_rows": len(neutral.x),
            "neutral_pcs_removed": len(denoise), "emotions": len(e.labels), "topics": len(t.labels),
            "geometry": geometry, "topic_pc1_var": float(t.explained[0]),
            "split_half_20v20": {"median": float(split_base["cos"].median()), "min": float(split_base["cos"].min()),
                                 "under_0.7": int((split_base["cos"] < 0.7).sum())},
            "split_half_final_core": {
                "median": float(split.loc[split["emotion"].isin(core), "cos_final"].median()),
                "min": float(split.loc[split["emotion"].isin(core), "cos_final"].min()),
                "under_0.7": split.loc[split["emotion"].isin(core) & (split["cos_final"] < 0.7), "emotion"].tolist()},
            "neighbours": V.neighbours(e, NEIGHBOURS),
            "dose_spearman": {em: float(dose[em].corr(dose["dose_mg"], method="spearman")) for em in ("afraid", "calm")},
            "leace": {"rank": eraser.rank, "shrinkage": eraser.shrinkage},
        }
        logger.info(f"L{layer}: |r(PC1, valence)| {abs(geometry['r_pc1_valence']):.3f}, PC1 {geometry['pc1_var']:.3f}")
        _json(_out(f"phase1{suffix}.json"), out)
    if s["d2_primary_layer"] in todo:
        r52 = abs(out[s["d2_primary_layer"]]["geometry"]["r_pc1_valence"])
        if r52 < 0.5:
            raise SystemExit(f"STOP: |r(PC1, valence)| = {r52:.3f} < 0.5 at layer {s['d2_primary_layer']}")
    logger.info("phase 1 final done")


if __name__ == "__main__":
    fire.Fire({"base": base, "final": final})
