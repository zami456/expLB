"""The thesis's own detectors on Mahadik & Skapars' ten datasets (design: `ms10_thesis_spec.py`, frozen before any of
them is scored there). `ms10`'s rows, splits, block reads and bootstrap draws, at the thesis's layers in the band
(36, 40, 44); the detectors as `transfer_alignment.scores` builds them. `ms10`'s saved arms are re-scored on the same
draws, so every comparison is paired. Runs in E:/aletheia/venv from the deception-detection folder.

    python -m emotion_coords.ms10_thesis freeze
    python -m emotion_coords.ms10_thesis score      # checks the freeze (this spec and ms10's)
    python -m emotion_coords.ms10_thesis report
"""

import json
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import yaml

from deception_detection.log import logger
from emotion_coords import aletheia_topk as TK
from emotion_coords import config as C
from emotion_coords import ms10 as M
from emotion_coords import ms10_spec as S
from emotion_coords import ms10_thesis_spec as T
from emotion_coords import probes as P

NEW = ("AILiar", "Roleplaying")  # the deception sets the thesis never scored


def out_dir() -> Path:
    return C.results_dir("ms10", "thesis")


def freeze() -> str:
    """Record this spec's SHA-256 before any of its detectors is scored on the ten sets."""
    path = Path(T.__file__)
    line = f"{path.name} sha256 {M._sha256(path)}"
    if T.FREEZE.exists():
        held = T.FREEZE.read_text(encoding="utf-8")
        if line not in held:
            raise RuntimeError(f"{T.FREEZE} holds another design:\n{held}")
        return held
    text = f"{line}\nfrozen {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    T.FREEZE.write_text(text, encoding="utf-8")
    logger.info(f"-> {T.FREEZE}\n{text}")
    return text


def check_freeze() -> str:
    digest = M._sha256(Path(T.__file__))
    if not T.FREEZE.exists() or digest not in T.FREEZE.read_text(encoding="utf-8"):
        raise RuntimeError(f"ms10_thesis_spec.py ({digest[:12]}) is not the frozen design ({T.FREEZE})")
    M.check_freeze()  # ms10's spec, sources and splits are the frozen ones
    logger.info(f"ms10_thesis_spec.py sha256 {digest} = the freeze")
    return digest


# --- the detectors ---------------------------------------------------------------------------------------------------


def _fixed(layer: int, alpaca: np.ndarray) -> dict[str, np.ndarray]:
    """The thesis's fixed directions at `layer`, exactly as `transfer_alignment.scores` builds them (K rebuilt from its
    reads and checked against the stored one; the sums standardised on Alpaca)."""
    from deception_detection import persona_pcs as pcs
    from deception_detection.scripts import recall_denial as rd
    from deception_detection.scripts import transfer_alignment as TA
    from deception_detection.scripts.layer_sweep import base_run

    bank = pcs.load_bank_at_layer(layer, pcs.EXTENDED_VECTORS_DIR)
    out = {name: bank.axis_unit(axis) for name, axis in TA.PERSONA_AXES.items()}
    out["K"] = TA.knowledge_direction(layer)
    out["D5 + K"] = rd.standardised_sum(out["D5"], out["K"], alpaca)
    out["D9 + K"] = rd.standardised_sum(out["D9"], out["K"], alpaca)
    out["D2"] = TA.detector_direction(base_run(layer))
    if layer in T.D3_LAYERS:
        run = TA.RESULTS / TA.AISI_RUN.format(layer=layer)
        cfg = yaml.safe_load((run / "cfg.yaml").read_text())
        if cfg.get("detect_layers") != [layer] or cfg.get("train_data") != TA.AISI_TRAIN:
            raise ValueError(f"{run.name} is not the AISI-trained probe at layer {layer}")
        out["D3"] = TA.detector_direction(run)
    return {k: out[k] for k in T.FIXED if k in out}


def _bases(layer: int) -> dict[str, np.ndarray]:
    """D4's basis (the extended bank's top 3 PCs, as `transfer_alignment.scores`) and the paper's (`ms10`'s
    persona_dec3 basis)."""
    from deception_detection import persona_pcs as pcs

    ext = pcs.load_bank_at_layer(layer, pcs.EXTENDED_VECTORS_DIR)
    d4 = pcs.fit_geometry(ext, k=3, deception=pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS,
                          honesty=pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS, axis_name="extended")
    return {"D4": d4.components[:3], "D4 (published PCs)": M._basis(layer, "persona_dec3", 0, None)}


def _trained(data: dict, basis: np.ndarray | None) -> dict[str, np.ndarray]:
    """A `persona_ood.make_probe()` fitted on all of each source's training rows (as `transfer_alignment.trained`), as a
    direction over the raw activations; the probe has no intercept."""
    from deception_detection.scripts import persona_ood as PO
    from deception_detection.scripts import transfer_alignment as TA

    out = {}
    for source in S.TRAIN_SOURCES:
        x, y, _ = M._training(data, source)
        x = x.astype(np.float64)
        w = TA.raw_direction(PO.make_probe().fit(x if basis is None else x @ basis.T, y))
        out[source] = w if basis is None else basis.T @ w
    return out


# --- scoring ---------------------------------------------------------------------------------------------------------


def _tests() -> dict[str, tuple]:
    """`ms10.score`'s draws: per test split, N_BOOT stratified resamples from default_rng([BOOT_SEED, set index])."""
    table = M._splits()
    tests = {}
    for i, name in enumerate(S.SETS):
        t = table[table["set"] == name]
        mask = (t["split"] == "test").to_numpy()
        y = t["label"].to_numpy()[mask]
        rng = np.random.default_rng([S.BOOT_SEED, i])
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        draws = [np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg)))) for _ in range(S.N_BOOT)]
        tests[name] = (y, pos, neg, *TK._counts(draws, pos, neg), mask)
    return tests


def _regions(fixed: bool) -> dict[str, list[tuple[str, str]]]:
    if fixed:
        return {"deception": [("none", t) for t in S.DECEPTION], "sycophancy": [("none", t) for t in S.SYCOPHANCY],
                "new deception": [("none", t) for t in NEW]}
    return M._regions() | {"new deception": [(a, t) for t in NEW for a in S.DECEPTION if a != t]}


def score() -> None:
    """Every detector on every test split at the three layers, on `ms10`'s draws; region means per draw; TD1-TD4."""
    digest = check_freeze()
    tests = _tests()
    records, draws = [], {}
    for layer in T.LAYERS:
        t0 = time.time()
        data = M._block(layer)
        xt = {n: data[f"x_{n}"][tests[n][5]].astype(np.float64) for n in S.SETS}
        detectors = {name: {"none": (v, 0.0)} for name, v in _fixed(layer, data["alpaca"].astype(np.float64)).items()}
        for name, basis in [("D1", None), *_bases(layer).items()]:
            detectors[name] = {src: (w, 0.0) for src, w in _trained(data, basis).items()}
        for fs in T.COMPARATORS:
            with np.load(M._part(layer, fs, 0)) as z:
                detectors[f"ms10 {fs}"] = {str(s): (v, float(o)) for s, v, o in zip(z["sources"], z["v"], z["offset"])}
        for name, by_source in detectors.items():
            cells = {}
            for src, (v, offset) in by_source.items():
                for t in S.SETS:
                    y, pos, neg, up, un, _ = tests[t]
                    s = xt[t] @ v + offset
                    cells[src, t] = TK._boot_aurocs(up, un, s, pos, neg)
                    records.append({"layer": layer, "detector": name, "train": src, "test": t, "auroc": P.auroc(y, s)})
            for region, keys in _regions("none" in by_source).items():
                draws[layer, name, region] = np.mean([cells[k] for k in keys], axis=0)
        logger.info(f"L{layer}: {len(detectors)} detectors scored ({time.time() - t0:.0f} s)")
    auroc = pd.DataFrame(records)
    auroc.to_csv(out_dir() / "auroc.csv", index=False)
    _summarise(auroc, draws, digest)


def _summarise(auroc: pd.DataFrame, draws: dict, digest: str) -> None:
    point = auroc.set_index(["layer", "detector", "train", "test"])["auroc"]
    names = list(dict.fromkeys(auroc["detector"]))
    rows = []
    for (layer, name, region), d in draws.items():
        fixed = (layer, name, "none", S.SETS[0]) in point.index
        keys = _regions(fixed)[region]
        rows.append({"layer": layer, "detector": name, "kind": "fixed" if fixed else "trained", "region": region,
                     "mean": float(np.mean([point[layer, name, s, t] for s, t in keys])),
                     "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))})
    regions = pd.DataFrame(rows)
    regions.to_csv(out_dir() / "regions.csv", index=False)
    means = regions.set_index(["layer", "detector", "region"])["mean"]
    pairs = [("D9 + K", "D1"), ("D9", "D5"), ("D9 + K", "D9"), ("D4", "D4 (published PCs)")]
    pairs += [(a, b) for a in names for b in ("D1", "ms10 raw", "ms10 emotion_all", "ms10 persona_dec3") if a != b]
    deltas = []
    for layer in T.LAYERS:
        for a, b in dict.fromkeys(pairs):
            for region in T.REGIONS:
                if (layer, a, region) in draws and (layer, b, region) in draws:
                    diff = draws[layer, a, region] - draws[layer, b, region]
                    deltas.append({"layer": layer, "region": region, "detector": a, "minus": b,
                                   "delta": means[layer, a, region] - means[layer, b, region],
                                   "lo": float(np.quantile(diff, 0.025)), "hi": float(np.quantile(diff, 0.975))})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out_dir() / "deltas.csv", index=False)
    verdicts = {"freeze_sha256": digest, "layers": list(T.LAYERS), "min_layers": T.MIN_LAYERS}
    for key, (a, b) in zip(("TD1", "TD2", "TD3", "TD4"), pairs[:4]):
        d = deltas[(deltas["region"] == "deception") & (deltas["detector"] == a) & (deltas["minus"] == b)]
        n = int((d["lo"] > 0).sum())
        verdicts[key] = {"detector": a, "minus": b, "layers_with_ci_above_0": n, "holds": n >= T.MIN_LAYERS,
                         "deltas": {int(r.layer): [round(r.delta, 4), round(r.lo, 4), round(r.hi, 4)]
                                    for r in d.itertuples()}}
    (out_dir() / "verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    logger.info(json.dumps(verdicts, indent=1))


# --- report ----------------------------------------------------------------------------------------------------------


def report() -> Path:
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    regions = pd.read_csv(out_dir() / "regions.csv")
    deltas = pd.read_csv(out_dir() / "deltas.csv")
    verdicts = json.loads((out_dir() / "verdicts.json").read_text(encoding="utf-8"))
    auroc = pd.read_csv(out_dir() / "auroc.csv")
    names = list(dict.fromkeys(regions["detector"]))
    lines = ["# The thesis's detectors on Mahadik & Skapars' ten datasets (Llama-3.3-70B, layers 36, 40, 44)", "",
             f"Design: `emotion_coords/ms10_thesis_spec.py`, frozen (sha256 {verdicts['freeze_sha256'][:12]}) before "
             "any of these detectors was scored on these sets; `ms10`'s rows, splits and bootstrap draws. A fixed "
             "detector's region is the mean over its test sets; a trained one's, the mean over the off-diagonal "
             "cells (each test set weighted equally either way).", "", "## Predictions", ""]
    for key in ("TD1", "TD2", "TD3", "TD4"):
        v = verdicts[key]
        cells = "; ".join(f"L{layer} {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" for layer, d in v["deltas"].items())
        lines.append(f"- **{key}** ({T.PREDICTIONS[key]}): {v['layers_with_ci_above_0']} of 3 ({cells}): "
                     f"**{'holds' if v['holds'] else 'fails'}**.")
    for region in T.REGIONS:
        lines += ["", f"## {region}: {T.REGIONS[region]}", "", "| detector | kind | L36 | L40 | L44 | mean |",
                  "|---|---|---|---|---|---|"]
        r = regions[regions["region"] == region]
        for name in names:
            rr = r[r["detector"] == name].set_index("layer")
            if rr.empty:
                continue
            cells = [f"{f3(rr.loc[layer, 'mean'])} [{f3(rr.loc[layer, 'lo'])}, {f3(rr.loc[layer, 'hi'])}]"
                     if layer in rr.index else "-" for layer in T.LAYERS]
            lines.append(f"| {name} | {rr['kind'].iloc[0]} | " + " | ".join(cells) + f" | {f3(rr['mean'].mean())} |")
    trained = regions[regions["kind"] == "trained"].groupby(["detector", "region"])["mean"].mean()
    lines += ["", "## Trained detectors: the other regions (mean over the three layers)", "",
              "| detector | cross-behaviour | diagonal | trained on all ten |", "|---|---|---|---|"]
    for name in [n for n in names if n in trained.index.get_level_values(0)]:
        lines.append(f"| {name} | " + " | ".join(f3(trained[name, r]) for r in ("cross", "diagonal", "all")) + " |")
    lines += ["", "## Each fixed detector on each test set (mean over the three layers)", "",
              "| detector | " + " | ".join(M.SET_SHORT[s] for s in S.SETS) + " |", "|---|" + "---|" * len(S.SETS)]
    fixed = auroc[auroc["train"] == "none"].groupby(["detector", "test"])["auroc"].mean()
    for name in [n for n in names if n in fixed.index.get_level_values(0)]:
        lines.append(f"| {name} | " + " | ".join(f3(fixed[name, s]) for s in S.SETS) + " |")
    lines += ["", "## Each trained detector: transfer into each test set from the other sets of the same behaviour "
              "(mean over the three layers)", "", "| detector | " + " | ".join(M.SET_SHORT[s] for s in S.SETS) + " |",
              "|---|" + "---|" * len(S.SETS)]
    for name in [n for n in names if n not in fixed.index.get_level_values(0)]:
        a = auroc[auroc["detector"] == name]
        cells = []
        for s in S.SETS:
            same = S.DECEPTION if s in S.DECEPTION else S.SYCOPHANCY
            cells.append(f3(a[a["test"].eq(s) & a["train"].isin(same) & a["train"].ne(s)]["auroc"].mean()))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    path = out_dir() / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"-> {path}")
    return path


if __name__ == "__main__":
    fire.Fire({"freeze": freeze, "score": score, "report": report})
