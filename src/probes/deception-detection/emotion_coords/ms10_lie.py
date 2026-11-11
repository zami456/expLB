"""Lie coordinates from RepE against the persona (D4), emotion, training-set-PC and raw spaces on Mahadik & Skapars'
ten datasets (design: `ms10_lie_spec.py`, frozen before any lie coordinate is fitted). The lie directions are INLP's on
RepE alone; every arm uses `ms10`'s probe, rows, splits and bootstrap draws, at its band 35-45. Runs in
E:/aletheia/venv from the deception-detection folder.

    python -m emotion_coords.ms10_lie freeze
    python -m emotion_coords.ms10_lie inlp       # the lie directions at every block, from RepE alone
    python -m emotion_coords.ms10_lie fit        # the new arms' probes (resumable parts)
    python -m emotion_coords.ms10_lie score      # checks the freezes
    python -m emotion_coords.ms10_lie report
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
from emotion_coords import ms10 as M
from emotion_coords import ms10_lie_spec as L
from emotion_coords import ms10_spec as S
from emotion_coords import ms10_thesis as MT
from emotion_coords import probes as P

STORE = M.STORE / "lie"


def out_dir() -> Path:
    return C.results_dir("ms10", "lie")


def freeze() -> str:
    """Record this spec's SHA-256 before any lie coordinate is fitted."""
    path = Path(L.__file__)
    line = f"{path.name} sha256 {M._sha256(path)}"
    if L.FREEZE.exists():
        held = L.FREEZE.read_text(encoding="utf-8")
        if line not in held:
            raise RuntimeError(f"{L.FREEZE} holds another design:\n{held}")
        return held
    text = f"{line}\nfrozen {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    L.FREEZE.write_text(text, encoding="utf-8")
    logger.info(f"-> {L.FREEZE}\n{text}")
    return text


def check_freeze() -> str:
    digest = M._sha256(Path(L.__file__))
    if not L.FREEZE.exists() or digest not in L.FREEZE.read_text(encoding="utf-8"):
        raise RuntimeError(f"ms10_lie_spec.py ({digest[:12]}) is not the frozen design ({L.FREEZE})")
    M.check_freeze()
    logger.info(f"ms10_lie_spec.py sha256 {digest} = the freeze")
    return digest


# --- the lie directions (RepE alone) ---------------------------------------------------------------------------------


def _repe(block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RepE's train split and its validation split without the statement both halves share."""
    from emotion_coords import lbdata as LB

    r = LB.load(block, "a", LB.full_subset())[C.load()["train_pairs"]]
    train = r.rows < LB.VAL_OFFSET
    shared = list(set(r.groups[train]) & set(r.groups[~train]))
    val = ~train & ~np.isin(r.groups, shared)
    return r.x[train].astype(np.float64), r.y[train], r.x[val].astype(np.float64), r.y[val]


def _inlp(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """INLP with the thesis's probe: each round's direction is kept while the probe still finds the lie on the held-out
    rows; the first round below STOP_AUROC ends it. Returns the directions (rows, orthonormal) and the trace."""
    from deception_detection.scripts import persona_ood as PO
    from deception_detection.scripts import transfer_alignment as TA

    e, trace = np.zeros((x_tr.shape[1], 0)), []
    for i in range(L.MAX_DIRECTIONS):
        probe = PO.make_probe().fit(x_tr - (x_tr @ e) @ e.T, y_tr)
        held = P.auroc(y_va, probe.decision_function(x_va - (x_va @ e) @ e.T))
        kept = held >= L.STOP_AUROC
        trace.append({"round": i + 1, "held_out_auroc": held, "kept": bool(kept)})
        if not kept:
            break
        w = TA.raw_direction(probe)
        w -= e @ (e.T @ w)
        e = np.column_stack([e, w / np.linalg.norm(w)])
    return e.T, trace


def _inlp_path(block: int) -> Path:
    return STORE / f"inlp_B{block}.npz"


def inlp() -> None:
    """The lie directions at every block (resumable), and inlp.csv with every round's held-out AUROC."""
    rows = []
    for b in L.BLOCKS:
        path = _inlp_path(b)
        if not path.exists():
            t0 = time.time()
            w, trace = _inlp(*_repe(b))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, W=w, trace=json.dumps(trace))
            logger.info(f"B{b}: {len(w)} lie directions ({time.time() - t0:.0f} s)")
        with np.load(path) as z:
            rows += [r | {"block": b, "k": len(z["W"])} for r in json.loads(str(z["trace"]))]
    pd.DataFrame(rows).to_csv(out_dir() / "inlp.csv", index=False)


def _directions(block: int) -> np.ndarray:
    with np.load(_inlp_path(block)) as z:
        return z["W"]


# --- the new arms' probes --------------------------------------------------------------------------------------------


def _units() -> list[tuple[int, str, int]]:
    return [(b, arm, s) for b in L.BLOCKS for arm in L.NEW_ARMS
            for s in range(L.N_SEEDS if arm == "random_k" else 1)]


def _part(block: int, arm: str, seed: int) -> Path:
    return STORE / "probes" / f"B{block}_{arm}_{seed}.npz"


def _unit(block: int, arm: str, seed: int) -> tuple[str, float]:
    """One arm at one block: a probe per training source, `ms10`'s way (C by inner CV, folded into a direction)."""
    t0 = time.time()
    data = M._block(block)
    lie = _directions(block)
    k = len(lie)
    fixed = {"lie": lambda: lie, "lie3": lambda: lie[:L.K_PERSONA], "persona_ext3": lambda: MT._bases(block)["D4"],
             "random_k": lambda: la.random_subspace(lie.shape[1], k, seed=L.RANDOM_SEED + seed)}
    basis = fixed[arm]() if arm in fixed else None
    out = {key: [] for key in ("v", "offset", "C", "inner_auroc", "n")}
    for source in S.TRAIN_SOURCES:
        x, y, groups = M._training(data, source)
        b = basis
        if arm.startswith("trainpc"):
            b = la.top_pcs(x, min(k if arm == "trainpc_k" else L.K_PERSONA, len(y) - 1))
        v, offset, c, inner = M._fit_direction(x, y, M._inner_folds(y, groups), b)
        for key, value in zip(out, (v, offset, c, inner, len(y))):
            out[key].append(value)
    path = _part(block, arm, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: np.array(value) for key, value in out.items()}, sources=np.array(S.TRAIN_SOURCES))
    return path.name, time.time() - t0


def fit(workers: int = 8, threads: int = 1) -> None:
    """Every new arm's unit not yet saved."""
    from joblib import Parallel, delayed

    from emotion_coords import ms10_lie as importable  # under `-m` this file is __main__, which workers cannot unpickle

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    todo = [u for u in _units() if not _part(*u).exists()]
    logger.info(f"{len(_units())} units, {len(todo)} to fit on {workers} workers x {threads} threads")
    tmp = STORE / "joblib_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, (name, seconds) in enumerate(Parallel(n_jobs=workers, backend="loky", temp_folder=str(tmp),
                                                 return_as="generator_unordered")(
            delayed(importable._unit)(*u) for u in todo)):
        logger.info(f"{i + 1}/{len(todo)} {name}: {seconds:.0f} s ({(time.time() - t0) / 60:.0f} min)")


# --- scoring ---------------------------------------------------------------------------------------------------------


def _detectors(block: int) -> dict[tuple[str, int], dict[str, tuple[np.ndarray, float]]]:
    """Every arm at a block as {(name, seed): {source: (direction, offset)}}; the zero-shot direction's source is
    "none"."""
    out = {(L.ZERO_SHOT, 0): {"none": (_directions(block)[0], 0.0)}}
    for b, arm, seed in [u for u in _units() if u[0] == block]:
        with np.load(_part(b, arm, seed)) as z:
            out[arm, seed] = {str(s): (v, float(o)) for s, v, o in zip(z["sources"], z["v"], z["offset"])}
    for fs in L.STORED_ARMS:
        with np.load(M._part(block, fs, 0)) as z:
            out[fs, 0] = {str(s): (v, float(o)) for s, v, o in zip(z["sources"], z["v"], z["offset"])}
    return out


def score() -> None:
    """Every arm on every test split at every block, on `ms10`'s draws; region means per draw (seeds averaged);
    L1-L4."""
    digest = check_freeze()
    tests = MT._tests()
    records, draws = [], {}
    for b in L.BLOCKS:
        data = M._block(b)
        xt = {n: data[f"x_{n}"][tests[n][5]].astype(np.float64) for n in S.SETS}
        per_seed: dict[str, list[dict]] = {}
        for (name, seed), by_source in _detectors(b).items():
            cells = {}
            for src, (v, offset) in by_source.items():
                for t in S.SETS:
                    y, pos, neg, up, un, _ = tests[t]
                    s = xt[t] @ v + offset
                    cells[src, t] = TK._boot_aurocs(up, un, s, pos, neg)
                    records.append({"block": b, "arm": name, "seed": seed, "train": src, "test": t,
                                    "auroc": P.auroc(y, s)})
            regions = MT._regions("none" in by_source)
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
        keys = MT._regions((b, name, "none", S.SETS[0]) in cells.index)[r]
        rows.append({"block": b, "arm": name, "region": r, "mean": float(np.mean([cells[b, name, s, t] for s, t in keys])),
                     "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))})
    regions = pd.DataFrame(rows)
    regions.to_csv(out_dir() / "regions.csv", index=False)
    point = regions.set_index(["block", "arm", "region"])["mean"]
    named = [("lie", "raw"), ("lie", "random_k"), ("lie", "trainpc_k"), ("lie3", "persona_ext3")]
    arms = [L.ZERO_SHOT, *L.NEW_ARMS]
    pairs = list(dict.fromkeys(named + [(a, c) for a in arms for c in ("raw", "emotion_all", "persona_dec3",
                                                                          "trainpca_all") if a != c]))
    deltas = []
    for b in L.BLOCKS:
        for a, c in pairs:
            for r in ("deception", "sycophancy", "new deception", "cross", "diagonal", "all"):
                if (b, a, r) in draws and (b, c, r) in draws:
                    diff = draws[b, a, r] - draws[b, c, r]
                    deltas.append({"block": b, "region": r, "arm": a, "minus": c, "delta": point[b, a, r] - point[b, c, r],
                                   "lo": float(np.quantile(diff, 0.025)), "hi": float(np.quantile(diff, 0.975))})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out_dir() / "deltas.csv", index=False)
    verdicts = {"freeze_sha256": digest, "blocks": list(L.BLOCKS), "min_blocks": L.MIN_BLOCKS}
    for key, (a, c) in zip(("L1", "L2", "L3", "L4"), named):
        d = deltas[(deltas["region"] == "deception") & (deltas["arm"] == a) & (deltas["minus"] == c)]
        above, below = int((d["lo"] > 0).sum()), int((d["hi"] < 0).sum())
        verdicts[key] = {"arm": a, "minus": c, "blocks_with_ci_above_0": above, "blocks_with_ci_below_0": below,
                         "mean_delta": float(d["delta"].mean())}
        if key == "L4":
            verdicts[key]["outcome"] = ("lie coordinates win" if above >= L.MIN_BLOCKS else
                                        "persona coordinates win" if below >= L.MIN_BLOCKS else "neither")
        else:
            verdicts[key]["holds"] = above >= L.MIN_BLOCKS
    inlp = pd.read_csv(out_dir() / "inlp.csv")
    verdicts["k_by_block"] = {int(b): int(k) for b, k in inlp.groupby("block")["k"].first().items()}
    (out_dir() / "verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    logger.info(json.dumps(verdicts, indent=1))


# --- report ----------------------------------------------------------------------------------------------------------

LABELS = {"lie_w1": "first lie direction (zero-shot)", "lie": "lie coordinates (k)", "lie3": "lie coordinates (3)",
          "persona_ext3": "D4: persona coordinates (3)", "trainpc_k": "training set's top k PCs",
          "trainpc3": "training set's top 3 PCs", "random_k": "random k directions", "raw": "raw activations",
          "emotion_all": "emotion all-170", "emotion_pc3": "emotion PCs (3)", "trainpca_all": "training set's top 170 PCs",
          "persona_dec3": "the paper's PCs (3)", "persona_syc3": "the paper's sycophancy PCs (3)"}


def report() -> Path:
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    regions = pd.read_csv(out_dir() / "regions.csv")
    verdicts = json.loads((out_dir() / "verdicts.json").read_text(encoding="utf-8"))
    inlp = pd.read_csv(out_dir() / "inlp.csv")
    auroc = pd.read_csv(out_dir() / "auroc.csv")
    lines = ["# Lie coordinates from RepE on Mahadik & Skapars' ten datasets (Llama-3.3-70B, blocks 35-45)", "",
             f"Design: `emotion_coords/ms10_lie_spec.py`, frozen (sha256 {verdicts['freeze_sha256'][:12]}) before any lie "
             "coordinate was fitted. INLP on RepE alone; every arm under `ms10`'s probe, rows, splits and draws.", "",
             "## Predictions (deception region: the 20 off-diagonal deception cells)", ""]
    for key in ("L1", "L2", "L3", "L4"):
        v = verdicts[key]
        outcome = v.get("outcome") or ("holds" if v["holds"] else "fails")
        lines.append(f"- **{key}** ({L.PREDICTIONS[key]}): mean {v['mean_delta']:+.3f}; CI above 0 at "
                     f"{v['blocks_with_ci_above_0']}, below 0 at {v['blocks_with_ci_below_0']} of 11: **{outcome}**.")
    k = verdicts["k_by_block"]
    lines += ["", f"## The lie directions: k by block (INLP stops below held-out AUROC {L.STOP_AUROC} on RepE's validation "
              "split)", "", "| block | " + " | ".join(str(b) for b in L.BLOCKS) + " |", "|---|" + "---|" * len(L.BLOCKS),
              "| k | " + " | ".join(str(k[str(b)]) for b in L.BLOCKS) + " |",
              "| round-1 held-out AUROC | " + " | ".join(
                  f3(inlp[(inlp["block"] == b) & (inlp["round"] == 1)]["held_out_auroc"].iloc[0]) for b in L.BLOCKS) + " |"]
    order = [L.ZERO_SHOT, *L.NEW_ARMS, *L.STORED_ARMS]
    for region in ("deception", "new deception", "sycophancy", "cross", "diagonal", "all"):
        r = regions[regions["region"] == region]
        if r.empty:
            continue
        lines += ["", f"## {region}: mean over the 11 blocks, and the range of block means", "",
                  "| arm | mean | min-max over blocks |", "|---|---|---|"]
        for name in [n for n in order if n in set(r["arm"])]:
            m = r[r["arm"] == name]["mean"]
            lines.append(f"| {LABELS[name]} | {f3(m.mean())} | {f3(m.min())}-{f3(m.max())} |")
    lines += ["", "## Transfer into each test set from the other sets of the same behaviour (mean over the blocks)", "",
              "| arm | " + " | ".join(M.SET_SHORT[s] for s in S.SETS) + " |", "|---|" + "---|" * len(S.SETS)]
    for name in order:
        a = auroc[auroc["arm"] == name]
        if name == L.ZERO_SHOT:
            cells = [f3(a[a["test"] == s]["auroc"].mean()) for s in S.SETS]
        else:
            cells = []
            for s in S.SETS:
                same = S.DECEPTION if s in S.DECEPTION else S.SYCOPHANCY
                cells.append(f3(a[a["test"].eq(s) & a["train"].isin(same) & a["train"].ne(s)]["auroc"].mean()))
        lines.append(f"| {LABELS[name]} | " + " | ".join(cells) + " |")
    path = out_dir() / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"-> {path}")
    return path


if __name__ == "__main__":
    fire.Fire({"freeze": freeze, "inlp": inlp, "fit": fit, "score": score, "report": report})
