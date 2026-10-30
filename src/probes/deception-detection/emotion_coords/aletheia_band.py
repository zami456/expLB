"""The all-170 arm on Aletheia's Quest at every block of mid +/- 4 (design: `aletheia_band_spec.py`, frozen before any
band read; the recipe, probe and statistics are `aletheia170`'s). Runs in E:/aletheia/venv from the
deception-detection folder; the NDIF reads are `reader/ec_read.py` with the `aqb_*.jobs` files.

Every input is read again at the nine blocks of each model, by one reader:
- the stories and neutral dialogues through each base model (the `aq_*` rows files);
- the training rows: Gemma's own Liars' Bench rows exactly as its pooled reads tokenised them (`layer_sweep.pool`'s
  cached tokenisation; insider trading after "Action:"), and Qwen2.5's rows rendered for Qwen3.5 as D4 (LB) read them;
- the Alpaca controls: Gemma's own 2,000 replies, Qwen3.5's 500;
- every test row through the organism that wrote it, one row per pass: the merged copies (`--repo`), BotC and Echo
  through their adapters (NDIF attaches them, `--peft`; scale alpha/r = 2, as the thesis read them), Harm through
  Gemma. BotC's rows keep the thesis's recorded tail cuts.

    python -m emotion_coords.aletheia_band rows       # every rows file, with 20-row parity subsets
    python -m emotion_coords.aletheia_band parity     # each pipeline against the thesis's stored reads (gate)
    python -m emotion_coords.aletheia_band bases --model gemma
    python -m emotion_coords.aletheia_band train --model gemma
    python -m emotion_coords.aletheia_band score      # checks the freeze first
    python -m emotion_coords.aletheia_band report
"""

import hashlib
import json
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from emotion_coords import aletheia170 as A
from emotion_coords import aletheia170_spec as X
from emotion_coords import aletheia_band_spec as B
from emotion_coords import aletheia_topk as TK
from emotion_coords import config as C
from emotion_coords import linalg as la
from emotion_coords import probes as P
from emotion_coords import vectors as V

STORE = Path("E:/emotion_coords/aletheia/band")
GEMMA_PARTIAL = Path("E:/liars-bench-acts/pooled_models_partial/gemma-3-27b")
GEMMA_SWEEP = "__B8-16-19-22-25-28-31-34-37-40-43-46-50-53-56-59-61"
GEMMA_SETS = {"convince_me__plain": "", "insider_trading_cadenza__plain": "+trim", "instruct_dishonesty__plain": "",
              "XXXX-20__plain": ""}  # "+trim": the post-"Action: " span, as the pooled `_trim` files hold it
CONTROL = "XXXX-20__plain"
ARMS = ("emotion_all", "raw", "shuffled_all", "random_all", "alpaca_all", "trainpca_all")


def out_dir() -> Path:
    return C.results_dir("aletheia_band")


def reads(model: str, what: str) -> str:
    """Rows file / read folder of a base-model pipeline: stories, neutral, lb, alpaca."""
    return f"aqb_{model}_{what}"


def test_reads(key: str) -> str:
    return f"aqb_test_{key}"


def story_rows(model: str, what: str) -> str:
    return A.name(model, what)  # the middle-block test's rows files, read again at the band's blocks


def mid(model: str) -> int:
    return X.BLOCK[model]


# --- rows ------------------------------------------------------------------------------------------------------------


def _rendered(rows: list[dict], model: str, cut_folder: Path | None = None) -> list[dict]:
    """Rows rendered as the thesis rendered them (`render.render`, reasoning dropped), the content span pooled; a row
    the thesis cut (its recorded tail cut in `cut_folder`) is cut the same way."""
    from deception_detection.scripts.aletheia import read as AR
    from deception_detection.scripts.aletheia import render as R
    from deception_detection.scripts.aletheia import spec as S

    tok = R.tokenizer(model)
    cut = AR._cut_rows(cut_folder) if cut_folder is not None else set()
    out = []
    for r in rows:
        item = R.render(r, model)
        if int(r["index"]) in cut:
            item = AR.tail_cut(item, keep=S.TAIL_CUT_TO, bos_id=getattr(tok, "bos_token_id", None))
        out.append({"row_id": int(r["index"]), "ids": item["ids"], "groups": [[list(item["content"])]], "points": []})
    return out


def _gemma_rows(name: str) -> list[dict]:
    """A Gemma Liars' Bench set as `layer_sweep.pool` tokenised and kept it (row id = code * 1e6 + pooled row)."""
    import torch

    from emotion_coords.rows import DATASET_CODES

    suffix = GEMMA_SETS.get(name, "")  # only insider trading has the "+trim" span
    saved = torch.load(GEMMA_PARTIAL / f"{name}{GEMMA_SWEEP}{suffix}" / "tokenized.pt", weights_only=True)
    variant = "_trim" if suffix else ""
    mask = saved["masks"][1 if suffix else 0]
    kept = [i for i in range(len(saved["tokens"])) if all(bool(m[i].any()) for m in saved["masks"])]
    out = []
    for k, i in enumerate(kept):
        real = saved["attention"][i].bool()  # left padding: the real tokens are the attended ones
        on = np.flatnonzero(mask[i][real].numpy())
        runs = np.split(on, np.flatnonzero(np.diff(on) != 1) + 1)
        out.append({"row_id": DATASET_CODES[name] * 1_000_000 + k, "ids": saved["tokens"][i][real].tolist(),
                    "groups": [[[int(r[0]), int(r[-1]) + 1] for r in runs]], "points": [], "dataset": name,
                    "label": saved["labels"][variant][i]})
    return out


def rows(n_parity: int = 20, parity_max_tokens: int = 1500) -> None:
    from deception_detection.scripts.aletheia import alpaca as AL
    from deception_detection.scripts.aletheia import data as D
    from deception_detection.scripts.aletheia import posthoc as PH
    from deception_detection.scripts.aletheia import read as AR
    from deception_detection.scripts.aletheia import score as SC
    from deception_detection.scripts.aletheia import spec as S
    from emotion_coords import rows as ECR

    def write(name: str, items: list[dict], parity: bool = True, **meta) -> None:
        ECR.write_rows(name, items, meta)
        if parity:
            short = [it for it in items if len(it["ids"]) <= parity_max_tokens]
            pick = np.random.default_rng(0).choice(len(short), min(n_parity, len(short)), replace=False)
            ECR.write_rows(f"{name}_parity", [short[i] for i in sorted(pick)], {"parity_of": name})

    meta = PH.lb_rows()
    items = _rendered(meta, "qwen35")
    for item, r in zip(items, meta):
        item |= {"dataset": r["dataset"], "lie": int(r["lie"])}
    write(reads("qwen35", "lb"), items, model="qwen35", rows_of="Qwen2.5-72B (posthoc.lb_rows)")
    alpaca_rows = [json.loads(x) for x in AL.path("qwen35").read_text(encoding="utf-8").splitlines() if x.strip()]
    write(reads("qwen35", "alpaca"), _rendered([r for r in alpaca_rows if r["messages"][-1]["content"]], "qwen35"),
          model="qwen35")
    write(reads("gemma", "lb"), [it for n in GEMMA_SETS if n != CONTROL for it in _gemma_rows(n)], model="gemma")
    write(reads("gemma", "alpaca"), _gemma_rows(CONTROL), model="gemma")
    for d in S.DATASETS:
        if d.key in X.SCORED:
            folder = AR.folder_for(d.model, SC.organism_read(d), f"test__{d.key}", "no-reasoning")
            write(test_reads(d.key), _rendered(D.rows(d.key), d.model, folder), parity=False, model=d.model,
                  organism=SC.organism_read(d))


# --- parity ----------------------------------------------------------------------------------------------------------


def _compare(name: str, stored, blocks: list[int]) -> dict:
    """cos between this reader's span means and `stored(row id, block)` (None: not stored) at `blocks`."""
    cos = []
    for b in blocks:
        ids, content, _ = V.load_reads(name, V._layer_index(b, name))
        for i, r in enumerate(ids):
            ref = stored(int(r), b)
            if ref is not None:
                cos.append(float(la.unit(content[i, 0, 0]) @ la.unit(np.asarray(ref, dtype=np.float64))))
    return {"blocks": blocks, "compared": len(cos), "min_cos": min(cos) if cos else None,
            "median_cos": float(np.median(cos)) if cos else None}


def parity(which: str = "all") -> dict:
    """Gate: each pipeline against the thesis's stored reads at the blocks they share with the band. `which`: "base"
    (the training and Alpaca subsets), "test" (every test row), or "all"."""
    from functools import cache

    from deception_detection.scripts.aletheia import read as AR
    from deception_detection.scripts.aletheia import score as SC
    from deception_detection.scripts.aletheia import spec as S
    from emotion_coords.rows import DATASET_CODES

    @cache
    def stored_ar(folder: str, block: int) -> dict:
        return {r: v[0, 0] for r, v in AR.load(Path(folder), [block]).items()}

    @cache
    def stored_gemma(name: str, block: int) -> np.ndarray:
        variant = "_trim" if GEMMA_SETS[name] else ""
        with np.load(GEMMA_PARTIAL.parents[1] / "xmodel" / "activations" / "gemma-3-27b" / f"pooled_B{block}{variant}"
                     / f"{name}.npz") as z:
            return z["pooled"]

    code_name = {c: n for n, c in DATASET_CODES.items()}
    out = {}
    if which in ("base", "all"):
        lbq = str(AR.folder_for("qwen35", "base", "lb_qwen25", "no-reasoning"))
        alq = str(AR.folder_for("qwen35", "base", "alpaca", "no-reasoning"))
        out[reads("qwen35", "lb")] = _compare(reads("qwen35", "lb") + "_parity",
                                              lambda r, b: stored_ar(lbq, b).get(r), [32])
        out[reads("qwen35", "alpaca")] = _compare(reads("qwen35", "alpaca") + "_parity",
                                                  lambda r, b: stored_ar(alq, b).get(r), [29, 32, 35])
        for what in ("lb", "alpaca"):
            out[reads("gemma", what)] = _compare(reads("gemma", what) + "_parity",
                                                 lambda r, b: stored_gemma(code_name[r // 1_000_000], b)[r % 1_000_000],
                                                 [28, 31, 34])
    if which in ("test", "all"):
        for d in S.DATASETS:
            if d.key not in X.SCORED:
                continue
            folder = str(AR.folder_for(d.model, SC.organism_read(d), f"test__{d.key}", "no-reasoning"))
            shared = [b for b in SC._layers_everywhere(Path(folder)) if b in B.BAND[d.model]]
            out[test_reads(d.key)] = _compare(test_reads(d.key), lambda r, b: stored_ar(folder, b).get(r), shared)
    for name, r in out.items():
        bar = B.PARITY_MIN_COS_ECHO if name == test_reads("Echo") else B.PARITY_MIN_COS
        r["pass"] = bool(r["min_cos"] is not None and r["min_cos"] >= bar)
    path = out_dir() / f"parity_{which}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    logger.info(f"band parity ({which}):\n{json.dumps(out, indent=1)}")
    failed = [n for n, r in out.items() if not r["pass"]]
    if failed:
        raise SystemExit(f"GATE parity: {failed}")
    return out


# --- bases and probes ------------------------------------------------------------------------------------------------


def bases_path(model: str, block: int) -> Path:
    return STORE / "bases" / f"{model}_B{block}.npz"


def bases(model: str) -> list[dict]:
    """The emotion basis and its shuffled bases at every band block (phase 1's recipe), with the valence check."""
    out = []
    for b in B.BAND[model]:
        stories = V.Reads.of(reads(model, "stories"), b)
        neutral = V.Reads.of(reads(model, "neutral"), b)
        emotion, shuffled, denoise, _ = A.build_bases(stories, neutral)
        arrays = {"E_vectors": emotion.vectors, "E_components": emotion.components, "E_explained": emotion.explained,
                  "E_labels": np.array(emotion.labels), "denoise": denoise}
        for p, s in enumerate(shuffled):
            arrays[f"S{p}_components"] = s.components
        bases_path(model, b).parent.mkdir(parents=True, exist_ok=True)
        np.savez(bases_path(model, b), **arrays)
        g = V.geometry(emotion)
        out.append({"block": b, "offset": b - mid(model), "rank": len(emotion.components),
                    "neutral_pcs_removed": len(denoise), "r_pc1_valence": g["r_pc1_valence"],
                    "valence_ok": bool(abs(g["r_pc1_valence"]) >= B.VALENCE_MIN_ABS_R)})
        logger.info(f"{model} B{b}: rank {out[-1]['rank']}, |r(PC1, valence)| {abs(g['r_pc1_valence']):.3f}")
    (out_dir() / f"bases_{model}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _layer(name: str, block: int) -> tuple[np.ndarray, np.ndarray]:
    ids, content, _ = V.load_reads(name, V._layer_index(block, name))
    return ids, content[:, 0, 0].astype(np.float64)


def training(model: str, block: int) -> tuple[np.ndarray, np.ndarray, list]:
    """The Liars' Bench rows at `block` (honest and deceptive only), folds leaving one set out."""
    ids, x = _layer(reads(model, "lb"), block)
    meta = V.row_meta(reads(model, "lb")).loc[ids]
    if model == "gemma":
        keep = meta["label"].isin(["HONEST", "DECEPTIVE"]).to_numpy()
        y = (meta["label"] == "DECEPTIVE").to_numpy().astype(int)
    else:
        keep = np.ones(len(ids), dtype=bool)
        y = meta["lie"].to_numpy().astype(int)
    sets = pd.factorize(meta["dataset"])[0]
    if not np.isfinite(x).all():
        raise RuntimeError(f"{model} B{block}: non-finite training reads")
    x, y, sets = x[keep], y[keep], sets[keep]
    return x, y, P._folds("sets", y, sets)


def _part(model: str, block: int, fs: str, seed: int) -> Path:
    return STORE / "probes" / f"{model}_B{block}_{fs}_{seed}.npz"


def _feature_sets(model: str, block: int, hidden: int) -> list[tuple[str, int, np.ndarray | None, bool]]:
    with np.load(bases_path(model, block)) as z:
        e = z["E_components"]
        shuffled = [z[f"S{p}_components"] for p in range(X.N_SHUFFLED)]
    r = len(e)
    _, alpaca = _layer(reads(model, "alpaca"), block)
    out = [("emotion_all", 0, e, False), ("raw", 0, None, False)]
    out += [("shuffled_all", p, s[:r], False) for p, s in enumerate(shuffled)]
    out += [("random_all", s, la.random_subspace(hidden, r, seed=X.RANDOM_SEED + s), False) for s in range(X.N_RANDOM)]
    out += [("alpaca_all", 0, la.top_pcs(alpaca, r), False), ("trainpca_all", 0, None, True)]
    return out


def train(model: str, blocks=None) -> None:
    """The six feature sets at every band block: C by leave-one-set-out CV, a fit on every row. Resumable by part."""
    todo = B.BAND[model] if blocks is None else C.int_list(blocks)
    rows = []
    for b in todo:
        x, y, folds = training(model, b)
        sets = _feature_sets(model, b, x.shape[1])
        r = next(len(basis) for fs, _, basis, _ in sets if fs == "emotion_all")
        for fs, seed, basis, per_train in sets:
            path = _part(model, b, fs, seed)
            if not path.exists():
                t0 = time.time()
                if per_train:
                    basis = la.top_pcs(x, min(r, len(y) - 1))
                feats = x if basis is None else x @ basis.T
                sel = P.best_per_k(P.select("lr", feats, y, folds, None if basis is None else (len(basis),), X.C_GRID))
                chosen = sel[sel["chosen"]].iloc[0]
                fitted = P.fit("lr", feats, y, float(chosen["C"]))
                v, offset = A._direction(fitted, basis)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, v=v, offset=offset, C=float(chosen["C"]), inner_auroc=float(chosen["inner_auroc"]))
                logger.info(f"{model} B{b} {fs}/{seed}: C {chosen['C']}, inner {chosen['inner_auroc']:.3f}, "
                            f"{time.time() - t0:.0f} s")
            with np.load(path) as z:
                rows.append({"block": b, "feature_set": fs, "seed": seed, "C": float(z["C"]),
                             "inner_auroc": float(z["inner_auroc"]), "n": len(y)})
    pd.DataFrame(rows).to_csv(out_dir() / f"probes_{model}.csv", index=False)


# --- the test sets ---------------------------------------------------------------------------------------------------


def check_freeze() -> str:
    digest = hashlib.sha256(Path(B.__file__).read_bytes()).hexdigest().upper()
    if not B.FREEZE.exists() or digest not in B.FREEZE.read_text(encoding="utf-8"):
        raise RuntimeError(f"aletheia_band_spec.py ({digest[:12]}) is not the frozen design ({B.FREEZE})")
    logger.info(f"aletheia_band_spec.py sha256 {digest} = the freeze")
    return digest


DEVIATION_SIX = ("Echo and BotC are not read at the band blocks: their adapters are gated on Hugging Face and NDIF "
                 "cannot load them (2026-10-30). The user chose to score the 6 readable sets now, with the "
                 "leaderboard averaged over the same 6, and the frozen 8-set design once NDIF can read them.")


def _scope(datasets) -> tuple[list[str], Path, str | None]:
    """The sets scored, where the results go, and the deviation note (None for the frozen 8)."""
    if datasets is None:
        return list(X.SCORED), out_dir(), None
    keys = [k for k in X.SCORED if k in (datasets.split(",") if isinstance(datasets, str) else list(datasets))]
    if keys == list(X.SCORED):
        return keys, out_dir(), None
    if keys != [k for k in X.SCORED if k not in ("Echo", "BotC")]:
        raise ValueError(f"only the frozen 8 or the 6 readable sets, not {keys}")
    path = out_dir() / "six_sets"
    path.mkdir(parents=True, exist_ok=True)
    return keys, path, DEVIATION_SIX


def score(datasets=None) -> None:
    """Every arm at every offset on the 8 sets, with `score.run`'s draws; means, ranks, deltas, HB1 and HB2.
    `datasets` "Harm,GS-F,GS-M,AB-AW,AB-HC,AB-SP": the recorded deviation (Echo and BotC unreadable), to `six_sets/`."""
    from deception_detection.scripts.aletheia import data as D
    from deception_detection.scripts.aletheia import score as SC
    from deception_detection.scripts.aletheia import spec as S

    digest = check_freeze()
    keys, out, deviation = _scope(datasets)
    names = [("emotion_all", 0), ("raw", 0)] + [("shuffled_all", p) for p in range(X.N_SHUFFLED)]
    names += [("random_all", s) for s in range(X.N_RANDOM)] + [("alpaca_all", 0), ("trainpca_all", 0)]
    records, boots = [], {}
    for d in S.DATASETS:
        if d.key not in keys:
            continue
        index = S.DATASETS.index(d)
        order = [int(r["index"]) for r in D.rows(d.key)]
        labels = D.labels(d.key, "test")
        y = np.array([int(labels[i]) for i in order])
        rng = np.random.default_rng([S.SEED, index])
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        draws = [np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg)))) for _ in range(S.N_BOOT)]
        up, un = TK._counts(draws, pos, neg)
        for j in B.OFFSETS:
            block = mid(d.model) + j
            ids, x = _layer(test_reads(d.key), block)
            where = {int(r): i for i, r in enumerate(ids)}
            if set(order) - set(where):
                raise RuntimeError(f"{d.key} B{block}: {len(set(order) - set(where))} test rows not read")
            z = x[[where[i] for i in order]]
            for fs, seed in names:
                with np.load(_part(d.model, block, fs, seed)) as p:
                    s = z @ p["v"] + float(p["offset"])
                boots[j, fs, seed, d.key] = TK._boot_aurocs(up, un, s, pos, neg)
                records.append({"dataset": d.key, "model": d.model, "offset": j, "block": block, "feature_set": fs,
                                "seed": seed, "auroc": SC.auroc(y, s)})
        logger.info(f"{d.key}: scored at offsets {B.OFFSETS[0]}..{B.OFFSETS[-1]}")
    table = pd.DataFrame(records)
    table.to_csv(out / "auroc.csv", index=False)
    _summarise(table, boots, digest, keys, out, deviation)


def _summarise(table: pd.DataFrame, boots: dict, digest: str, keys: list[str], out: Path,
               deviation: str | None) -> None:
    from deception_detection.scripts.aletheia import report as RP

    board = RP._board()
    entries = board[keys].mean(axis=1).to_numpy()
    white = board[board["track"] == "white-box"][keys].mean(axis=1).to_numpy()
    rows, means = [], {}
    for j in B.OFFSETS:
        for arm in ARMS:
            t = table[(table["offset"] == j) & (table["feature_set"] == arm)]
            point = np.array([t[t["dataset"] == k]["auroc"].mean() for k in keys])
            seeds = sorted(t["seed"].unique())
            boot = np.stack([np.mean([boots[j, arm, s, k] for s in seeds], axis=0) for k in keys]).mean(axis=0)
            means[j, arm] = (float(point.mean()), boot)
            ranks = np.array([RP._rank(b, entries) for b in boot])
            rows.append({"offset": j, "qwen35_block": 32 + j, "gemma_block": 31 + j, "arm": arm,
                         "mean": float(point.mean()), "lo": float(np.quantile(boot, 0.025)),
                         "hi": float(np.quantile(boot, 0.975)), "rank of 31": RP._rank(float(point.mean()), entries),
                         "rank lo": int(np.quantile(ranks, 0.025)), "rank hi": int(np.quantile(ranks, 0.975)),
                         "white-box rank of 19": RP._rank(float(point.mean()), white)}
                        | {k: float(v) for k, v in zip(keys, point)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    deltas = []
    for j in B.OFFSETS:
        for other in ARMS[1:]:
            diff = means[j, "emotion_all"][1] - means[j, other][1]
            deltas.append({"offset": j, "minus": other, "delta": means[j, "emotion_all"][0] - means[j, other][0],
                           "lo": float(np.quantile(diff, 0.025)), "hi": float(np.quantile(diff, 0.975))})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out / "deltas.csv", index=False)
    above = deltas.assign(above=deltas["lo"] > 0).groupby("minus")["above"].sum().astype(int).to_dict()
    verdicts = {"freeze_sha256": digest, "datasets": keys, "deviation": deviation, "offsets": list(B.OFFSETS),
                "HB1": {"offsets_with_ci_above_0": above["raw"], "holds": above["raw"] >= 5},
                "HB2": {c: {"offsets_with_ci_above_0": above[c]} for c in ARMS[2:]}}
    verdicts["HB2"]["holds"] = all(above[c] >= 5 for c in ARMS[2:])
    frozen = pd.read_csv(A.out_dir() / "summary.csv").set_index("arm")
    check = {}
    for arm, frozen_arm in (("emotion_all", "emotion_all (LB)"), ("raw", "raw (LB)")):
        zero = summary[(summary["offset"] == 0) & (summary["arm"] == arm)].iloc[0]
        check[arm] = {"band": float(zero["mean"]), "frozen": float(np.mean([frozen.loc[frozen_arm, k] for k in keys])),
                      "max |per-dataset difference|": float(max(abs(zero[k] - frozen.loc[frozen_arm, k]) for k in keys))}
    verdicts["offset 0 vs the frozen middle-block test"] = check
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    logger.info("\n" + summary.pivot(index="offset", columns="arm", values="mean").round(3).to_string())
    logger.info(json.dumps(verdicts, indent=1))


def report(datasets=None) -> Path:
    _, out, _ = _scope(datasets)
    summary = pd.read_csv(out / "summary.csv")
    deltas = pd.read_csv(out / "deltas.csv")
    verdicts = json.loads((out / "verdicts.json").read_text(encoding="utf-8"))
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    lines = ["# The all-170 arm on Aletheia's Quest: the band mid +/- 4", "",
             f"Design: `emotion_coords/aletheia_band_spec.py`, frozen (sha256 {verdicts['freeze_sha256'][:12]}) before "
             f"any band read. Liars' Bench-trained; each test row read by its organism at every block; "
             f"{len(verdicts['datasets'])} sets ({', '.join(verdicts['datasets'])}).", ""]
    if verdicts.get("deviation"):
        lines += [f"**Deviation.** {verdicts['deviation']}", ""]
    lines += [
             "| Offset | Qwen3.5 / Gemma block | all-170 [95% CI] | raw | shuffled | random | Alpaca PCs | train PCs "
             "| all-170 rank of 31 | white-box rank of 19 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for j in B.OFFSETS:
        s = summary[summary["offset"] == j].set_index("arm")
        e = s.loc["emotion_all"]
        lines.append(f"| {j:+d} | {32 + j} / {31 + j} | {f3(e['mean'])} [{f3(e['lo'])}, {f3(e['hi'])}] | "
                     + " | ".join(f3(s.loc[a, "mean"]) for a in ARMS[1:])
                     + f" | {int(e['rank of 31'])} | {int(e['white-box rank of 19'])} |")
    lines += ["", "all-170 minus each arm, by offset (paired 95% CI):", "",
              "| Offset | " + " | ".join(ARMS[1:]) + " |", "|---|" + "---|" * len(ARMS[1:])]
    for j in B.OFFSETS:
        dd = deltas[deltas["offset"] == j].set_index("minus")
        lines.append(f"| {j:+d} | " + " | ".join(f"{dd.loc[a, 'delta']:+.3f} [{dd.loc[a, 'lo']:+.3f}, {dd.loc[a, 'hi']:+.3f}]"
                                                for a in ARMS[1:]) + " |")
    hb1, hb2 = verdicts["HB1"], verdicts["HB2"]
    lines += ["", f"- **HB1** (beats raw at >= 5 of 9 offsets): {hb1['offsets_with_ci_above_0']} of 9: "
                  f"{'holds' if hb1['holds'] else 'fails'}.",
              "- **HB2** (beats every 170-d control at >= 5 of 9): " + "; ".join(
                  f"{c} {v['offsets_with_ci_above_0']}" for c, v in hb2.items() if c != "holds")
              + f": {'holds' if hb2['holds'] else 'fails'}.",
              f"- Offset 0 against the frozen middle-block test: {json.dumps(verdicts['offset 0 vs the frozen middle-block test'])}"]
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"-> {path}")
    return path


if __name__ == "__main__":
    fire.Fire({"rows": rows, "parity": parity, "bases": bases, "train": train, "score": score, "report": report})
