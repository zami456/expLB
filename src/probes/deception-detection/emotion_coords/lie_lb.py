"""Lie coordinates from RepE on Liars' Bench's five sets, train on one and test on the other four (design: `lie_lb_spec.py`,
frozen before any probe is fitted here). `ms10_lie`'s arms and lie directions under `ms10`'s probe, phase 3's protocol C
data and folds, at blocks 35-45; the first lie direction and D9 zero-shot. Runs in E:/aletheia/venv from the
deception-detection folder.

    python -m emotion_coords.lie_lb freeze
    python -m emotion_coords.lie_lb cache                                     # each block's five capped sets
    python -m emotion_coords.lie_lb fit --only raw --workers 4 --threads 3    # raw at phase 3's thread count
    python -m emotion_coords.lie_lb fit --skip raw --workers 8 --threads 1
    python -m emotion_coords.lie_lb score                                     # checks the freezes
    python -m emotion_coords.lie_lb report
"""

import json
import os
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from emotion_coords import aletheia_topk as TK
from emotion_coords import config as C
from emotion_coords import linalg as la
from emotion_coords import lie_lb_spec as B
from emotion_coords import ms10 as M
from emotion_coords import ms10_lie as ML
from emotion_coords import ms10_lie_spec as L
from emotion_coords import ms10_thesis as MT
from emotion_coords import probes as P

STORE = Path("E:/emotion_coords/lie_lb")


def out_dir() -> Path:
    return C.results_dir("lie_lb")


def freeze() -> str:
    """Record this spec's SHA-256 before any probe is fitted here."""
    path = Path(B.__file__)
    line = f"{path.name} sha256 {M._sha256(path)}"
    if B.FREEZE.exists():
        held = B.FREEZE.read_text(encoding="utf-8")
        if line not in held:
            raise RuntimeError(f"{B.FREEZE} holds another design:\n{held}")
        return held
    text = f"{line}\nfrozen {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    B.FREEZE.write_text(text, encoding="utf-8")
    logger.info(f"-> {B.FREEZE}\n{text}")
    return text


def check_freeze() -> str:
    digest = M._sha256(Path(B.__file__))
    if not B.FREEZE.exists() or digest not in B.FREEZE.read_text(encoding="utf-8"):
        raise RuntimeError(f"lie_lb_spec.py ({digest[:12]}) is not the frozen design ({B.FREEZE})")
    ML.check_freeze()
    logger.info(f"lie_lb_spec.py sha256 {digest} = the freeze")
    return digest


# --- data ------------------------------------------------------------------------------------------------------------


def _cache_path(block: int) -> Path:
    return STORE / "data" / f"B{block}.npz"


def cache(blocks=None) -> None:
    """Each block's five capped sets (`lbdata.full_subset`), float32, one file per block."""
    from emotion_coords import lbdata as LB

    subset = LB.full_subset()
    for b in B.BLOCKS if blocks is None else C.int_list(blocks):
        if _cache_path(b).exists():
            continue
        data = LB.load(b, "a", subset)
        arrays = {}
        for name in B.SETS:
            arrays[f"x_{B.SHORT[name]}"] = data[name].x.astype(np.float32)
            arrays[f"y_{B.SHORT[name]}"] = data[name].y
        _cache_path(b).parent.mkdir(parents=True, exist_ok=True)
        np.savez(_cache_path(b), **arrays)
        logger.info(f"B{b}: " + ", ".join(f"{B.SHORT[n]} {len(arrays[f'y_{B.SHORT[n]}'])}" for n in B.SETS))


def _block(block: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with np.load(_cache_path(block)) as z:
        return {name: (z[f"x_{B.SHORT[name]}"], z[f"y_{B.SHORT[name]}"]) for name in B.SETS}


# --- probes ----------------------------------------------------------------------------------------------------------


def _units() -> list[tuple[int, str, int]]:
    return [(b, arm, s) for arm in B.TRAINED_ARMS for b in B.BLOCKS
            for s in range(B.N_SEEDS if arm == "random_k" else 1)]


def _part(block: int, arm: str, seed: int) -> Path:
    return STORE / "probes" / f"B{block}_{arm}_{seed}.npz"


def _unit(block: int, arm: str, seed: int) -> tuple[str, float]:
    """One arm at one block: a probe on all rows of each set (C by a stratified 5-fold inner CV), as a direction."""
    t0 = time.time()
    data = _block(block)
    lie = ML._directions(block)
    k = len(lie)
    fixed = {"lie": lambda: lie, "lie3": lambda: lie[:B.K_SMALL], "persona_ext3": lambda: MT._bases(block)["D4"],
             "persona_dec3": lambda: M._basis(block, "persona_dec3", 0, None),
             "random_k": lambda: la.random_subspace(lie.shape[1], k, seed=L.RANDOM_SEED + seed)}
    basis = fixed[arm]() if arm in fixed else None
    out = {key: [] for key in ("v", "offset", "C", "inner_auroc", "n")}
    for name in B.SETS:
        x, y = data[name]
        b = basis
        if arm.startswith("trainpc"):
            b = la.top_pcs(x, min(k if arm == "trainpc_k" else B.K_SMALL, len(y) - 1))
        v, offset, c, inner = M._fit_direction(x, y, P._folds("strat", y, None), b)
        for key, value in zip(out, (v, offset, c, inner, len(y))):
            out[key].append(value)
    path = _part(block, arm, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: np.array(value) for key, value in out.items()}, sources=np.array(B.SETS))
    return path.name, time.time() - t0


def fit(workers: int = 8, threads: int = 1, only=None, skip=None) -> None:
    """Every unit not yet saved; `only` / `skip` as `ms10.run`."""
    from joblib import Parallel, delayed

    from emotion_coords import lie_lb as importable  # under `-m` this file is __main__, which workers cannot unpickle

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    todo = [u for u in _units() if not _part(*u).exists()
            and (only is None or u[1] in M._names(only)) and (skip is None or u[1] not in M._names(skip))]
    logger.info(f"{len(todo)} units to fit on {workers} workers x {threads} threads")
    tmp = STORE / "joblib_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, (name, seconds) in enumerate(Parallel(n_jobs=workers, backend="loky", temp_folder=str(tmp),
                                                 return_as="generator_unordered")(
            delayed(importable._unit)(*u) for u in todo)):
        logger.info(f"{i + 1}/{len(todo)} {name}: {seconds:.0f} s ({(time.time() - t0) / 60:.0f} min)")


# --- scoring ---------------------------------------------------------------------------------------------------------


def _regions(fixed: bool) -> dict[str, list[tuple[str, str]]]:
    rest = [n for n in B.SETS if n != B.HPKR]
    if fixed:
        return {"all": [("none", t) for t in B.SETS], "excl. HP-KR": [("none", t) for t in rest]}
    return {"all": [(a, t) for a in B.SETS for t in B.SETS if a != t],
            "excl. HP-KR": [(a, t) for a in rest for t in rest if a != t]}


def _tests(data: dict) -> dict[str, tuple]:
    out = {}
    for i, name in enumerate(B.SETS):
        y = data[name][1]
        rng = np.random.default_rng([B.BOOT_SEED, 200 + i])
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        draws = [np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg)))) for _ in range(B.N_BOOT)]
        out[name] = (y, pos, neg, *TK._counts(draws, pos, neg))
    return out


def score() -> None:
    """Every arm on every other set at every block; region means per draw (seeds averaged); LB1-LB4."""
    from deception_detection import persona_pcs as pcs

    digest = check_freeze()
    records, draws = [], {}
    for b in B.BLOCKS:
        data = _block(b)
        tests = _tests(data)  # the rows are the same at every block, so the draws are too
        detectors = {("lie_w1", 0): {"none": (ML._directions(b)[0], 0.0)},
                     ("D9", 0): {"none": (pcs.load_bank_at_layer(b, pcs.EXTENDED_VECTORS_DIR).axis_unit("extended"), 0.0)}}
        for _, arm, seed in [u for u in _units() if u[0] == b]:
            with np.load(_part(b, arm, seed)) as z:
                detectors[arm, seed] = {str(s): (v, float(o)) for s, v, o in zip(z["sources"], z["v"], z["offset"])}
        per_seed: dict[str, list[dict]] = {}
        for (name, seed), by_source in detectors.items():
            cells = {}
            for src, (v, offset) in by_source.items():
                for t in B.SETS:
                    if t == src:
                        continue
                    y, pos, neg, up, un = tests[t]
                    s = data[t][0].astype(np.float64) @ v + offset
                    cells[src, t] = TK._boot_aurocs(up, un, s, pos, neg)
                    records.append({"block": b, "arm": name, "seed": seed, "train": src, "test": t,
                                    "auroc": P.auroc(y, s)})
            regions = _regions("none" in by_source)
            per_seed.setdefault(name, []).append({r: np.mean([cells[c] for c in keys], axis=0)
                                                  for r, keys in regions.items()})
        for name, seeds in per_seed.items():
            for r in seeds[0]:
                draws[b, name, r] = np.mean([d[r] for d in seeds], axis=0)
        logger.info(f"B{b}: scored")
    auroc = pd.DataFrame(records)
    auroc.to_csv(out_dir() / "auroc.csv", index=False)
    _summarise(auroc, draws, digest)


def _summarise(auroc: pd.DataFrame, draws: dict, digest: str) -> None:
    cells = auroc.groupby(["block", "arm", "train", "test"])["auroc"].mean()
    rows = []
    for (b, name, r), d in draws.items():
        keys = _regions((b, name, "none", B.SETS[1]) in cells.index)[r]
        rows.append({"block": b, "arm": name, "region": r, "mean": float(np.mean([cells[b, name, s, t] for s, t in keys])),
                     "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))})
    regions = pd.DataFrame(rows)
    regions.to_csv(out_dir() / "regions.csv", index=False)
    point = regions.set_index(["block", "arm", "region"])["mean"]
    named = [("lie", "raw"), ("lie", "random_k"), ("lie", "trainpc_k"), ("lie3", "persona_ext3")]
    pairs = list(dict.fromkeys(named + [(a, "raw") for a in ("lie3", "persona_ext3", "persona_dec3", "trainpc3",
                                                              "lie_w1", "D9")] + [("persona_ext3", "persona_dec3")]))
    deltas = []
    for b in B.BLOCKS:
        for a, c in pairs:
            for r in B.REGIONS:
                diff = draws[b, a, r] - draws[b, c, r]
                deltas.append({"block": b, "region": r, "arm": a, "minus": c, "delta": point[b, a, r] - point[b, c, r],
                               "lo": float(np.quantile(diff, 0.025)), "hi": float(np.quantile(diff, 0.975))})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out_dir() / "deltas.csv", index=False)
    verdicts = {"freeze_sha256": digest, "blocks": list(B.BLOCKS), "min_blocks": B.MIN_BLOCKS}
    for key, (a, c) in zip(("LB1", "LB2", "LB3", "LB4"), named):
        d = deltas[(deltas["region"] == "all") & (deltas["arm"] == a) & (deltas["minus"] == c)]
        above, below = int((d["lo"] > 0).sum()), int((d["hi"] < 0).sum())
        verdicts[key] = {"arm": a, "minus": c, "blocks_with_ci_above_0": above, "blocks_with_ci_below_0": below,
                         "mean_delta": float(d["delta"].mean())}
        if key == "LB4":
            verdicts[key]["outcome"] = ("lie coordinates win" if above >= B.MIN_BLOCKS else
                                        "persona coordinates win" if below >= B.MIN_BLOCKS else "neither")
        else:
            verdicts[key]["holds"] = above >= B.MIN_BLOCKS
    (out_dir() / "verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    logger.info(json.dumps(verdicts, indent=1))


# --- report ----------------------------------------------------------------------------------------------------------

LABELS = ML.LABELS | {"D9": "D9 extended contrast (zero-shot)"}


def report() -> Path:
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    regions = pd.read_csv(out_dir() / "regions.csv")
    deltas = pd.read_csv(out_dir() / "deltas.csv")
    verdicts = json.loads((out_dir() / "verdicts.json").read_text(encoding="utf-8"))
    auroc = pd.read_csv(out_dir() / "auroc.csv")
    lines = ["# Lie coordinates on Liars' Bench: train on one set, test on the other four (Llama-3.3-70B, blocks 35-45)", "",
             f"Design: `emotion_coords/lie_lb_spec.py`, frozen (sha256 {verdicts['freeze_sha256'][:12]}) before any probe was "
             "fitted here. `ms10_lie`'s arms under `ms10`'s probe; phase 3's protocol C rows and folds. A re-test on the "
             "thesis's development data.", "", "## Predictions (all 20 cells)", ""]
    for key in ("LB1", "LB2", "LB3", "LB4"):
        v = verdicts[key]
        outcome = v.get("outcome") or ("holds" if v["holds"] else "fails")
        lines.append(f"- **{key}** ({B.PREDICTIONS[key]}): mean {v['mean_delta']:+.3f}; CI above 0 at "
                     f"{v['blocks_with_ci_above_0']}, below 0 at {v['blocks_with_ci_below_0']} of 11: **{outcome}**.")
    order = ["D9", "lie_w1", "persona_ext3", "lie3", "persona_dec3", "trainpc3", "lie", "trainpc_k", "random_k", "raw"]
    for region in B.REGIONS:
        r = regions[regions["region"] == region]
        lines += ["", f"## {region}: mean over the 11 blocks (min-max of the block means), and blocks 36/40/44", "",
                  "| arm | mean | min-max | B36 | B40 | B44 |", "|---|---|---|---|---|---|"]
        for name in order:
            m = r[r["arm"] == name].set_index("block")["mean"]
            lines.append(f"| {LABELS[name]} | {f3(m.mean())} | {f3(m.min())}-{f3(m.max())} | "
                         + " | ".join(f3(m[b]) for b in (36, 40, 44)) + " |")
    lines += ["", "## Each arm minus raw activations (all 20 cells): mean delta, blocks with the CI above / below 0", "",
              "| arm | mean delta | above | below |", "|---|---|---|---|"]
    for name in ("D9", "lie_w1", "persona_ext3", "lie3", "persona_dec3", "trainpc3"):
        d = deltas[(deltas["region"] == "all") & (deltas["arm"] == name) & (deltas["minus"] == "raw")]
        lines.append(f"| {LABELS[name]} | {d['delta'].mean():+.3f} | {int((d['lo'] > 0).sum())} | {int((d['hi'] < 0).sum())} |")
    names = [B.SHORT[n] for n in B.SETS]
    for arm in ("persona_ext3", "lie3", "lie", "raw"):
        a = auroc[auroc["arm"] == arm].groupby(["train", "test"])["auroc"].mean()
        lines += ["", f"## {LABELS[arm]}: train (rows) on one set, test (columns) on the others; mean over the blocks", "",
                  "| train | " + " | ".join(names) + " | mean |", "|---|" + "---|" * (len(names) + 1)]
        for src in B.SETS:
            vals = [a[src, t] if t != src else np.nan for t in B.SETS]
            lines.append(f"| {B.SHORT[src]} | " + " | ".join("-" if np.isnan(v) else f3(v) for v in vals)
                         + f" | {f3(np.nanmean(vals))} |")
    zs = auroc[auroc["train"] == "none"].groupby(["arm", "test"])["auroc"].mean()
    lines += ["", "## Zero-shot, each set (mean over the blocks)", "", "| detector | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
    for name in B.ZERO_SHOT:
        lines.append(f"| {LABELS[name]} | " + " | ".join(f3(zs[name, t]) for t in B.SETS) + " |")
    path = out_dir() / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"-> {path}")
    return path


if __name__ == "__main__":
    fire.Fire({"freeze": freeze, "cache": cache, "fit": fit, "score": score, "report": report})
