"""The band with RepE training rows (design: `aletheia_band_repe_spec.py`, frozen before any RepE band read; the
rest is `aletheia_band`'s: blocks, bases, feature sets, test reads, draws). Runs in E:/aletheia/venv from the
deception-detection folder; the reads are `reader/ec_read.py` with `aqbr.jobs`.

    python -m extensions.emotion_coords.aletheia_band_repe rows      # the RepE rows of each model, for the reader
    python -m extensions.emotion_coords.aletheia_band_repe parity    # every row against the thesis's reads at the shared blocks
    python -m extensions.emotion_coords.aletheia_band_repe train --model gemma
    python -m extensions.emotion_coords.aletheia_band_repe score --datasets Harm,GS-F,GS-M,AB-AW,AB-HC,AB-SP
    python -m extensions.emotion_coords.aletheia_band_repe report --datasets Harm,GS-F,GS-M,AB-AW,AB-HC,AB-SP

The Liars' Bench-trained all-170 and raw arms of the band are scored again beside the RepE ones, on the same draws,
so the two sources can be differenced.
"""

import hashlib
import json
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from extensions.emotion_coords import aletheia170 as A
from extensions.emotion_coords import aletheia170_spec as X
from extensions.emotion_coords import aletheia_band as AB
from extensions.emotion_coords import aletheia_band_repe_spec as R
from extensions.emotion_coords import aletheia_band_spec as B
from extensions.emotion_coords import aletheia_topk as TK
from extensions.emotion_coords import config as C
from extensions.emotion_coords import linalg as la
from extensions.emotion_coords import probes as P
from extensions.emotion_coords import vectors as V

GEMMA_REPE = "repe_honesty__plain"
ARMS = AB.ARMS


def out_dir(datasets=None) -> Path:
    keys, _, deviation = AB._scope(datasets)
    path = AB.out_dir() / "repe" / ("six_sets" if deviation else "")
    path.mkdir(parents=True, exist_ok=True)
    return path


def reads(model: str) -> str:
    return AB.reads(model, "repe")


def rows() -> None:
    from experiments.external.aletheia import repe as RE
    from extensions.emotion_coords import rows as ECR

    ECR.write_rows(reads("gemma"), AB._gemma_rows(GEMMA_REPE), {"model": "gemma", "rows_of": GEMMA_REPE})
    meta = RE.rows()
    items = AB._rendered(meta, "qwen35")
    for item, r in zip(items, meta):
        item |= {"label": int(r["label"]), "statement": r["messages"][-1]["content"]}
    ECR.write_rows(reads("qwen35"), items, {"model": "qwen35", "rows_of": "aletheia/repe.py rows()"})


def parity() -> dict:
    """Gate: every RepE row against the thesis's reads at the blocks they share with the band (Qwen3.5 32: the R1
    probe's per-token reads, averaged over each row's span; Gemma 28, 31, 34: its pooled reads)."""
    from experiments.external.aletheia import repe as RE

    with np.load(RE.token_path("qwen35", 32)) as z:
        tokens, owner = z["tokens"], z["owner"].astype(np.int64)
    order = np.argsort(owner, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(owner[order])) + 1]
    means = np.add.reduceat(tokens[order].astype(np.float64), starts, axis=0) / np.diff(np.r_[starts, len(order)])[:, None]
    qwen = {int(o): means[j] for j, o in enumerate(owner[order][starts])}
    gemma = {}
    for b in (28, 31, 34):
        with np.load(AB.GEMMA_PARTIAL.parents[2] / "xmodel" / "activations" / "gemma-3-27b" / f"pooled_B{b}"
                     / f"{GEMMA_REPE}.npz") as z:
            gemma[b] = z["pooled"]
    out = {reads("qwen35"): AB._compare(reads("qwen35"), lambda r, b: qwen.get(r), [32]),
           reads("gemma"): AB._compare(reads("gemma"), lambda r, b: gemma[b][r % 1_000_000], [28, 31, 34])}
    for r in out.values():
        r["pass"] = bool(r["min_cos"] is not None and r["min_cos"] >= R.PARITY_MIN_COS)
    (AB.out_dir() / "parity_repe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    logger.info(f"RepE band parity:\n{json.dumps(out, indent=1)}")
    if not all(r["pass"] for r in out.values()):
        raise SystemExit(f"GATE parity: {[n for n, r in out.items() if not r['pass']]}")
    return out


def training(model: str, block: int) -> tuple[np.ndarray, np.ndarray, list]:
    """R1's training rows at `block`, folds grouped by statement (as `aletheia170.repe_data`)."""
    from experiments.external.aletheia import repe as RE
    from deception_detection.utils import REPO_ROOT

    ids, x = AB._layer(reads(model), block)
    meta = V.row_meta(reads(model)).loc[ids]
    if model == "gemma":
        y = (meta["label"] == "DECEPTIVE").to_numpy().astype(int)
        llama = np.load(REPO_ROOT / "activations" / "pooled_L40" / f"{GEMMA_REPE}.npz")["labels"].astype(str)
        position = ids % 1_000_000
        if not np.array_equal(llama[position], meta["label"].to_numpy().astype(str)):
            raise RuntimeError("Gemma's RepE rows are not in Llama's order")
        groups = pd.read_csv(C.data_dir("index") / "repe_groups.csv")
        groups = groups[groups["dataset"] == GEMMA_REPE].set_index("pooled_row")["group"]
        g = groups.loc[position].to_numpy()
    else:
        order = sorted(int(i) for i in V.row_meta(reads(model)).index)
        train = set(order[: int(RE.TRAIN_SHARE * len(order))])
        keep = np.array([int(i) in train for i in ids])
        x, meta = x[keep], meta[keep]
        y = meta["label"].to_numpy().astype(int)
        g = pd.factorize(meta["statement"])[0]
    if not np.isfinite(x).all():
        raise RuntimeError(f"{model} B{block}: non-finite RepE reads")
    return x, y, P._folds("grouped", y, g)


def _part(model: str, block: int, fs: str, seed: int) -> Path:
    return AB.STORE / "probes_repe" / f"{model}_B{block}_{fs}_{seed}.npz"


def train(model: str, blocks=None) -> None:
    """The six feature sets at every band block on the RepE rows: C by grouped CV, a fit on every row."""
    todo = B.BAND[model] if blocks is None else C.int_list(blocks)
    table = []
    for b in todo:
        x, y, folds = training(model, b)
        sets = AB._feature_sets(model, b, x.shape[1])
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
                table.append({"block": b, "feature_set": fs, "seed": seed, "C": float(z["C"]),
                              "inner_auroc": float(z["inner_auroc"]), "n": len(y)})
    pd.DataFrame(table).to_csv(AB.out_dir() / f"probes_repe_{model}.csv", index=False)


def check_freeze() -> str:
    AB.check_freeze()
    digest = hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest().upper()
    if not R.FREEZE.exists() or digest not in R.FREEZE.read_text(encoding="utf-8"):
        raise RuntimeError(f"aletheia_band_repe_spec.py ({digest[:12]}) is not the frozen design ({R.FREEZE})")
    logger.info(f"aletheia_band_repe_spec.py sha256 {digest} = the freeze")
    return digest


def score(datasets=None) -> None:
    """The RepE arms at every offset, and the band's Liars' Bench all-170 and raw beside them, on `score.run`'s draws."""
    from experiments.external.aletheia import data as D
    from experiments.external.aletheia import score as SC
    from experiments.external.aletheia import spec as S
    from experiments.external.aletheia import report as RP

    digest = check_freeze()
    keys, _, deviation = AB._scope(datasets)
    out = out_dir(datasets)
    cells = [("RepE", "emotion_all", 0), ("RepE", "raw", 0)] + [("RepE", "shuffled_all", p) for p in range(X.N_SHUFFLED)]
    cells += [("RepE", "random_all", s) for s in range(X.N_RANDOM)] + [("RepE", "alpaca_all", 0), ("RepE", "trainpca_all", 0)]
    cells += [("LB", "emotion_all", 0), ("LB", "raw", 0)]
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
            block = AB.mid(d.model) + j
            ids, x = AB._layer(AB.test_reads(d.key), block)
            where = {int(r): i for i, r in enumerate(ids)}
            z = x[[where[i] for i in order]]
            for src, fs, seed in cells:
                path = _part(d.model, block, fs, seed) if src == "RepE" else AB._part(d.model, block, fs, seed)
                with np.load(path) as p:
                    s = z @ p["v"] + float(p["offset"])
                boots[j, src, fs, seed, d.key] = TK._boot_aurocs(up, un, s, pos, neg)
                records.append({"dataset": d.key, "offset": j, "block": block, "source": src, "feature_set": fs,
                                "seed": seed, "auroc": SC.auroc(y, s)})
        logger.info(f"{d.key}: scored")
    table = pd.DataFrame(records)
    table.to_csv(out / "auroc.csv", index=False)
    board = RP._board()
    entries = board[keys].mean(axis=1).to_numpy()
    white = board[board["track"] == "white-box"][keys].mean(axis=1).to_numpy()
    arms = [("RepE", a) for a in ARMS] + [("LB", "emotion_all"), ("LB", "raw")]
    rows, means = [], {}
    for j in B.OFFSETS:
        for src, arm in arms:
            t = table[(table["offset"] == j) & (table["source"] == src) & (table["feature_set"] == arm)]
            point = np.array([t[t["dataset"] == k]["auroc"].mean() for k in keys])
            seeds = sorted(t["seed"].unique())
            boot = np.stack([np.mean([boots[j, src, arm, s, k] for s in seeds], axis=0) for k in keys]).mean(axis=0)
            means[j, src, arm] = (float(point.mean()), boot)
            rows.append({"offset": j, "qwen35_block": 32 + j, "gemma_block": 31 + j, "source": src, "arm": arm,
                         "mean": float(point.mean()), "lo": float(np.quantile(boot, 0.025)),
                         "hi": float(np.quantile(boot, 0.975)), "rank of 31": RP._rank(float(point.mean()), entries),
                         "white-box rank of 19": RP._rank(float(point.mean()), white)}
                        | {k: float(v) for k, v in zip(keys, point)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    deltas = []
    for j in B.OFFSETS:
        for other in [("RepE", a) for a in ARMS[1:]] + [("LB", "emotion_all"), ("LB", "raw")]:
            diff = means[j, "RepE", "emotion_all"][1] - means[j, *other][1]
            deltas.append({"offset": j, "minus": f"{other[1]} ({other[0]})",
                           "delta": means[j, "RepE", "emotion_all"][0] - means[j, *other][0],
                           "lo": float(np.quantile(diff, 0.025)), "hi": float(np.quantile(diff, 0.975))})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out / "deltas.csv", index=False)
    above = deltas.assign(above=deltas["lo"] > 0).groupby("minus")["above"].sum().astype(int).to_dict()
    middle = pd.read_csv(A.out_dir() / "summary.csv").set_index("arm")
    zero = summary[(summary["offset"] == 0) & (summary["source"] == "RepE") & (summary["arm"] == "emotion_all")].iloc[0]
    verdicts = {"freeze_sha256": digest, "datasets": keys, "deviation": deviation,
                "HR1": {"offsets_with_ci_above_0": above["raw (RepE)"], "holds": above["raw (RepE)"] >= 5},
                "HR2": {c: {"offsets_with_ci_above_0": above[f"{c} (RepE)"]} for c in ARMS[2:]},
                "above all-170 (LB)": above["emotion_all (LB)"], "above raw (LB)": above["raw (LB)"],
                "offset 0 vs the middle-block RepE all-170": {
                    "band": float(zero["mean"]),
                    "middle-block test": float(np.mean([middle.loc["emotion_all (RepE)", k] for k in keys])),
                    "max |per-dataset difference|": float(max(abs(zero[k] - middle.loc["emotion_all (RepE)", k])
                                                              for k in keys))}}
    verdicts["HR2"]["holds"] = all(above[f"{c} (RepE)"] >= 5 for c in ARMS[2:])
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    logger.info("\n" + summary.pivot_table(index="offset", columns=["source", "arm"], values="mean").round(3).to_string())
    logger.info(json.dumps(verdicts, indent=1))


def report(datasets=None) -> Path:
    out = out_dir(datasets)
    summary = pd.read_csv(out / "summary.csv")
    deltas = pd.read_csv(out / "deltas.csv")
    verdicts = json.loads((out / "verdicts.json").read_text(encoding="utf-8"))
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    lines = ["# The all-170 arm on Aletheia's Quest: the band mid +/- 4, RepE-trained", "",
             f"Design: `emotion_coords/aletheia_band_repe_spec.py`, frozen (sha256 {verdicts['freeze_sha256'][:12]}) "
             f"before any RepE band read. {len(verdicts['datasets'])} sets ({', '.join(verdicts['datasets'])}).", ""]
    if verdicts.get("deviation"):
        lines += [f"**Deviation.** {verdicts['deviation']}", ""]
    lines += ["| Offset | Qwen3.5 / Gemma block | all-170 RepE [95% CI] | raw RepE | shuffled | random | Alpaca PCs "
              "| train PCs | all-170 LB | raw LB | all-170 RepE rank of 31 | white-box rank of 19 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for j in B.OFFSETS:
        s = summary[summary["offset"] == j].set_index(["source", "arm"])
        e = s.loc[("RepE", "emotion_all")]
        lines.append(f"| {j:+d} | {32 + j} / {31 + j} | {f3(e['mean'])} [{f3(e['lo'])}, {f3(e['hi'])}] | "
                     + " | ".join(f3(s.loc[("RepE", a), "mean"]) for a in ARMS[1:]) + " | "
                     + f"{f3(s.loc[('LB', 'emotion_all'), 'mean'])} | {f3(s.loc[('LB', 'raw'), 'mean'])} | "
                     + f"{int(e['rank of 31'])} | {int(e['white-box rank of 19'])} |")
    names = list(dict.fromkeys(deltas["minus"]))
    lines += ["", "all-170 (RepE) minus each arm, by offset (paired 95% CI):", "",
              "| Offset | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for j in B.OFFSETS:
        dd = deltas[deltas["offset"] == j].set_index("minus")
        lines.append(f"| {j:+d} | " + " | ".join(f"{dd.loc[n, 'delta']:+.3f} [{dd.loc[n, 'lo']:+.3f}, {dd.loc[n, 'hi']:+.3f}]"
                                                for n in names) + " |")
    hr1, hr2 = verdicts["HR1"], verdicts["HR2"]
    lines += ["", f"- **HR1** (beats raw RepE at >= 5 of 9 offsets): {hr1['offsets_with_ci_above_0']} of 9: "
                  f"{'holds' if hr1['holds'] else 'fails'}.",
              "- **HR2** (beats every 170-d control at >= 5 of 9): " + "; ".join(
                  f"{c} {v['offsets_with_ci_above_0']}" for c, v in hr2.items() if c != "holds")
              + f": {'holds' if hr2['holds'] else 'fails'}.",
              f"- Above all-170 (LB) at {verdicts['above all-170 (LB)']} offsets; above raw (LB) at "
              f"{verdicts['above raw (LB)']}.",
              f"- Offset 0 against the middle-block test: {json.dumps(verdicts['offset 0 vs the middle-block RepE all-170'])}"]
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"-> {path}")
    return path


if __name__ == "__main__":
    fire.Fire({"rows": rows, "parity": parity, "train": train, "score": score, "report": report})
