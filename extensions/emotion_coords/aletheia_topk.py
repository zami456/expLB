"""Top-k emotion PCs on Aletheia's Quest (design: `aletheia_topk_spec.py`, frozen before scoring; everything else as
`aletheia170`, whose bases, reads, training rows and bootstrap draws it reuses). Runs in E:/aletheia/venv from the
deception-detection folder.

    python -m extensions.emotion_coords.aletheia_topk topic --model gemma   # the topic basis from the story reads
    python -m extensions.emotion_coords.aletheia_topk train --model gemma   # every (source, feature set, seed) at each k
    python -m extensions.emotion_coords.aletheia_topk dev --model gemma     # dev dry run and the known-answer gate
    python -m extensions.emotion_coords.aletheia_topk score                 # the test sets; checks the freeze first
    python -m extensions.emotion_coords.aletheia_topk report                # summary.md

Every k of every arm is fitted (its C chosen by the source's inner CV) and kept as a direction in the read's space.
The arm "at k_e" uses, for each model and source, the k the emotion basis chose in that inner CV (P1's matching).
"""

import hashlib
import json
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from extensions.emotion_coords import aletheia170 as A
from extensions.emotion_coords import aletheia170_spec as X
from extensions.emotion_coords import aletheia_topk_spec as K
from extensions.emotion_coords import config as C
from extensions.emotion_coords import linalg as la
from extensions.emotion_coords import probes as P
from extensions.emotion_coords import vectors as V

CONTROLS = ("topic_pc", "shuffled_pc", "random")


def out_dir() -> Path:
    return C.results_dir("aletheia_topk")


def topic_path(model: str) -> Path:
    return A.STORE / "bases" / f"{model}_B{X.BLOCK[model]}_topic.npz"


def topic(model: str) -> dict:
    """The topic basis (per topic over every emotion, rank-0 stories) with the saved basis's denoising."""
    A._check_model(model)
    block = X.BLOCK[model]
    stories = V.Reads.of(A.name(model, "stories"), block)
    with np.load(A.bases_path(model)) as z:
        denoise = z["denoise"]
    rank0 = (stories.meta["rank"] == 0).to_numpy()
    balanced = V.Reads(stories.x[rank0], stories.meta[rank0].reset_index(drop=True))
    basis = V.topic_basis(balanced, denoise)
    np.savez(topic_path(model), T_components=basis.components, T_explained=basis.explained,
             T_labels=np.array(basis.labels))
    out = {"model": model, "topics": len(basis.labels), "rank": len(basis.components),
           "pc1_var": float(basis.explained[0])}
    logger.info(f"{model} topic basis: {out}")
    return out


def bases20(model: str, hidden: int) -> list[tuple[str, int, np.ndarray]]:
    """(feature set, seed, the first max(KS) rows of its basis)."""
    top = max(K.KS)
    with np.load(A.bases_path(model)) as z:
        emotion = z["E_components"][:top]
        shuffled = [z[f"S{p}_components"][:top] for p in range(X.N_SHUFFLED)]
    with np.load(topic_path(model)) as z:
        topic_ = z["T_components"][:top]
    out = [("emotion_pc", 0, emotion), ("topic_pc", 0, topic_)]
    out += [("shuffled_pc", p, s) for p, s in enumerate(shuffled)]
    out += [("random", s, la.random_subspace(hidden, top, seed=K.RANDOM_SEED + s)) for s in range(K.N_RANDOM)]
    return out


def _part(model: str, source: str, fs: str, seed: int) -> Path:
    return A.STORE / "topk" / f"{model}_{source}_{fs}_{seed}.npz"


def train(model: str, sources: str = "LB,RepE,dev") -> None:
    """For each (source, feature set, seed): k and C by the source's inner CV; a probe fitted at every k."""
    A._check_model(model)
    sources = sources.split(",") if isinstance(sources, str) else list(sources)
    rows = []
    for source in sources:
        x, y, folds = A.training(model, source)
        for fs, seed, basis in bases20(model, x.shape[1]):
            path = _part(model, source, fs, seed)
            if not path.exists():
                feats = x @ basis.T
                per_k = P.best_per_k(P.select("lr", feats, y, folds, K.KS, X.C_GRID))
                arrays = {}
                for r in per_k.itertuples():
                    k = int(r.k)
                    fitted = P.fit("lr", feats[:, :k], y, float(r.C))
                    v, offset = A._direction(fitted, basis[:k])
                    arrays |= {f"v{k}": v, f"offset{k}": offset, f"C{k}": float(r.C), f"inner{k}": float(r.inner_auroc)}
                    if r.chosen:
                        arrays["k_chosen"] = k
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, **arrays)
            with np.load(path) as z:
                rows += [{"source": source, "feature_set": fs, "seed": seed, "k": k, "C": float(z[f"C{k}"]),
                          "inner_auroc": float(z[f"inner{k}"]), "chosen": int(z["k_chosen"]) == k} for k in K.KS]
        chosen = [r for r in rows if r["source"] == source and r["feature_set"] == "emotion_pc" and r["chosen"]]
        logger.info(f"{model} {source}: emotion_pc chose k = {chosen[0]['k']} (inner {chosen[0]['inner_auroc']:.3f})")
    pd.DataFrame(rows).to_csv(out_dir() / f"probes_{model}.csv", index=False)


def probes(model: str, source: str) -> dict[tuple[str, int], dict]:
    """{(feature set, seed): {"k_chosen": k, k: (direction, offset) for every k}}."""
    out = {}
    for fs, seed in [("emotion_pc", 0), ("topic_pc", 0)] + [("shuffled_pc", p) for p in range(X.N_SHUFFLED)] \
            + [("random", s) for s in range(K.N_RANDOM)]:
        with np.load(_part(model, source, fs, seed)) as z:
            out[fs, seed] = {"k_chosen": int(z["k_chosen"])} | {k: (z[f"v{k}"], float(z[f"offset{k}"])) for k in K.KS}
    return out


def k_e(model: str, source: str) -> int:
    with np.load(_part(model, source, "emotion_pc", 0)) as z:
        return int(z["k_chosen"])


def dev(model: str) -> dict:
    """The LB-trained arms at k_e on every dev set read by its organism (dev labels only), and the gate."""
    from experiments.external.aletheia import data as D
    from experiments.external.aletheia import score as SC
    from experiments.external.aletheia import spec as S

    A._check_model(model)
    trained, k = probes(model, "LB"), k_e(model, "LB")
    rows = []
    for key in D.dev_keys():
        if key.split("__")[1] != model:
            continue
        got = SC._dev_reads_one(model, key, X.BLOCK[model])
        if got is None:
            continue
        z, y = got
        for (fs, seed), entry in trained.items():
            v, off = entry[k]
            rows.append({"dev set": key, "feature_set": fs, "seed": seed, "k": k, "auroc": P.auroc(y, z @ v + off)})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir() / f"dev_{model}.csv", index=False)
    key = f"instructed_deception__{model}__None"
    z, y = SC._dev_reads_one(model, key, X.BLOCK[model])
    v, off = trained["emotion_pc", 0][k]
    value = P.auroc(y, z @ v + off)
    null = np.random.default_rng(S.SEED).standard_normal((X.N_NULL, z.shape[1]))
    null /= np.linalg.norm(null, axis=1, keepdims=True)
    band = np.array([P.auroc(y, s) for s in (z @ null.T).T])
    gate = {"model": model, "dev set": key, "k_e": k, "emotion_pc": value, "band97.5": float(np.quantile(band, 0.975)),
            "pass": bool(value > np.quantile(band, 0.975))}
    (out_dir() / f"dev_gate_{model}.json").write_text(json.dumps(gate, indent=1), encoding="utf-8")
    logger.info(f"{model} dev at k_e = {k} (mean over dev sets):\n"
                f"{frame.groupby('feature_set')['auroc'].mean().round(3).to_string()}\ngate {gate}")
    if not gate["pass"]:
        raise SystemExit(f"GATE dev {model}: {value:.3f} <= band 97.5th percentile {gate['band97.5']:.3f}")
    return gate


# --- the test sets -----------------------------------------------------------------------------------------------


def check_freeze() -> str:
    digest = hashlib.sha256(Path(K.__file__).read_bytes()).hexdigest().upper()
    if not K.FREEZE.exists() or digest not in K.FREEZE.read_text(encoding="utf-8"):
        raise RuntimeError(f"aletheia_topk_spec.py ({digest[:12]}) is not the frozen design ({K.FREEZE})")
    logger.info(f"aletheia_topk_spec.py sha256 {digest} = the freeze")
    return digest


def _counts(draws: list[np.ndarray], pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`score.run`'s index draws as per-row multiplicities of the positives and the negatives."""
    n1 = len(pos)
    up = np.stack([np.bincount(np.searchsorted(pos, d[:n1]), minlength=n1) for d in draws]).astype(np.float64)
    un = np.stack([np.bincount(np.searchsorted(neg, d[n1:]), minlength=len(neg)) for d in draws]).astype(np.float64)
    return up, un


def _boot_aurocs(up: np.ndarray, un: np.ndarray, s: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """AUROC of every draw (Mann-Whitney with ties at 1/2, as roc_auc_score on the resampled rows)."""
    sp, sn = s[pos], s[neg]
    m = (sp[:, None] > sn[None, :]).astype(np.float64) + 0.5 * (sp[:, None] == sn[None, :])
    return ((up @ m) * un).sum(1) / (up.shape[1] * un.shape[1])


def score() -> None:
    """Every (source, feature set, seed, k) on the 8 sets with the all-170 score's draws; the arms at k_e and at each
    k; deltas paired with the all-170 score's stored bootstrap (raw, all-170, the thesis's cells); TK1."""
    from experiments.external.aletheia import data as D
    from experiments.external.aletheia import score as SC
    from experiments.external.aletheia import spec as S

    digest = check_freeze()
    trained = {(m, src): probes(m, src) for m in X.MODELS for src in K.SOURCES}
    stored = dict(np.load(A.out_dir() / "boots.npz"))
    raw_dirs = {m: A.probes(m, "LB")["raw", 0] for m in X.MODELS}
    records, boots, worst = [], {}, 0.0
    for d in S.DATASETS:
        if d.key not in X.SCORED:
            continue
        index = S.DATASETS.index(d)
        z, ids = SC._reads(d, SC.organism_read(d), "no-reasoning", 0, d.base.mid)
        labels = D.labels(d.key, "test")
        y = np.array([int(labels[i]) for i in ids])
        rng = np.random.default_rng([S.SEED, index])
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        draws = [np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg)))) for _ in range(S.N_BOOT)]
        up, un = _counts(draws, pos, neg)
        v, off = raw_dirs[d.model]  # the pairing check: the same draws as the all-170 score
        worst = max(worst, float(np.abs(_boot_aurocs(up, un, z @ v + off, pos, neg)
                                        - stored[f"raw/0 (LB)|{d.key}"]).max()))
        for src in K.SOURCES:
            for (fs, seed), entry in trained[d.model, src].items():
                for k in K.KS:
                    v, off = entry[k]
                    s = z @ v + off
                    boots[src, fs, seed, k, d.key] = _boot_aurocs(up, un, s, pos, neg)
                    records.append({"dataset": d.key, "model": d.model, "source": src, "feature_set": fs, "seed": seed,
                                    "k": k, "k_e": trained[d.model, src]["emotion_pc", 0]["k_chosen"],
                                    "auroc": SC.auroc(y, s)})
        logger.info(f"{d.key}: scored")
    logger.info(f"draws check: max |bootstrap AUROC of raw (LB) - the all-170 score's| = {worst:.2e}")
    if worst > 1e-9:
        raise RuntimeError("the bootstrap draws differ from the all-170 score's")
    table = pd.DataFrame(records)
    table.to_csv(out_dir() / "auroc.csv", index=False)
    _summarise(table, boots, stored, digest)


def _arm(table: pd.DataFrame, boots: dict, datasets: list[str], src: str, fs: str, k: int | None) -> tuple:
    """Per-dataset AUROC and per-draw bootstrap AUROC (seeds averaged) of an arm at k (None: each model's k_e)."""
    t = table[(table["source"] == src) & (table["feature_set"] == fs)]
    point, boot = [], []
    for key in datasets:
        rows = t[t["dataset"] == key]
        kk = int(rows["k_e"].iloc[0]) if k is None else k
        rows = rows[rows["k"] == kk]
        point.append(rows["auroc"].mean())
        boot.append(np.mean([boots[src, fs, int(s), kk, key] for s in rows["seed"]], axis=0))
    return np.array(point), np.stack(boot)


def _summarise(table: pd.DataFrame, boots: dict, stored: dict, digest: str) -> None:
    from experiments.external.aletheia import report as RP

    board = RP._board()
    datasets = [k for k in X.SCORED if k in set(table["dataset"])]
    entries = board[datasets].mean(axis=1).to_numpy()
    a170 = pd.read_csv(A.out_dir() / "auroc.csv").set_index(["cell", "dataset"])["auroc"]
    arms = {}
    for src in K.SOURCES:
        for fs in ("emotion_pc",) + CONTROLS:
            arms[f"{fs} (k_e, {src})"] = _arm(table, boots, datasets, src, fs, None)
            for k in K.KS:
                arms[f"{fs} (k={k}, {src})"] = _arm(table, boots, datasets, src, fs, k)
    for cell, label in (("raw/0 (LB)", "raw (LB)"), ("emotion_all/0 (LB)", "all-170 (LB)"), ("headline", "headline"),
                        ("D4 (LB)", "D4 (LB)"), ("R1-repe", "R1-repe")):
        arms[label] = (np.array([a170[cell, k] for k in datasets]),
                       np.stack([stored[f"{cell}|{k}"] for k in datasets]))
    rows, means = [], {}
    for name, (point, boot) in arms.items():
        bm = boot.mean(axis=0)
        means[name] = bm
        ranks = np.array([RP._rank(b, entries) for b in bm])
        rows.append({"arm": name, "mean": float(point.mean()), "lo": float(np.quantile(bm, 0.025)),
                     "hi": float(np.quantile(bm, 0.975)), "rank of 31": RP._rank(float(point.mean()), entries),
                     "rank lo": int(np.quantile(ranks, 0.025)), "rank hi": int(np.quantile(ranks, 0.975))}
                    | {k: float(v) for k, v in zip(datasets, point)})
    summary = pd.DataFrame(rows).set_index("arm")
    summary.to_csv(out_dir() / "summary.csv")
    pairs = [("emotion_pc (k_e, LB)", f"{c} (k_e, LB)") for c in CONTROLS]
    pairs += [("emotion_pc (k_e, LB)", c) for c in ("raw (LB)", "all-170 (LB)", "headline")]
    pairs += [(f"emotion_pc (k_e, {s})", f"{c} (k_e, {s})") for s in ("RepE", "dev") for c in CONTROLS]
    pairs += [(f"emotion_pc (k={k}, LB)", f"{c} (k={k}, LB)") for k in K.KS for c in ("random", "topic_pc")]
    deltas = []
    for a, b in pairs:
        diff = means[a] - means[b]
        deltas.append({"arm": a, "minus": b, "delta": float(summary.loc[a, "mean"] - summary.loc[b, "mean"]),
                       "lo": float(np.quantile(diff, 0.025)), "hi": float(np.quantile(diff, 0.975))})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out_dir() / "deltas.csv", index=False)
    d = deltas.set_index(["arm", "minus"])
    parts = {c: d.loc[("emotion_pc (k_e, LB)", f"{c} (k_e, LB)")] for c in ("random", "topic_pc")}
    k_used = {m: int(table[(table["model"] == m) & (table["source"] == "LB")]["k_e"].iloc[0]) for m in X.MODELS}
    verdicts = {"freeze_sha256": digest, "datasets": datasets, "k_e (LB)": k_used,
                "TK1": {c: {"delta": float(v["delta"]), "ci": [float(v["lo"]), float(v["hi"])],
                            "holds": bool(v["delta"] >= 0.03 and (v["lo"] > 0 or v["hi"] < 0))}
                        for c, v in parts.items()}}
    verdicts["TK1"]["holds"] = all(v["holds"] for v in verdicts["TK1"].values())
    (out_dir() / "verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    logger.info("\n" + summary[["mean", "lo", "hi", "rank of 31"]].round(3).to_string())
    logger.info(json.dumps(verdicts, indent=1))


def report() -> Path:
    out = out_dir()
    summary = pd.read_csv(out / "summary.csv").set_index("arm")
    deltas = pd.read_csv(out / "deltas.csv")
    verdicts = json.loads((out / "verdicts.json").read_text(encoding="utf-8"))
    keys = verdicts["datasets"]
    ke = verdicts["k_e (LB)"]
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    lines = ["# Top-k emotion PCs on Aletheia's Quest", "",
             f"Design: `emotion_coords/aletheia_topk_spec.py`, frozen (sha256 {verdicts['freeze_sha256'][:12]}) before "
             "scoring, after the all-170 results were seen. Liars' Bench-trained unless noted; each row read by its "
             f"organism. k_e, the emotion basis's inner-CV choice: Qwen3.5 {ke['qwen35']}, Gemma {ke['gemma']}.", "",
             "| Arm | " + " | ".join(("Echo†" if k == "Echo" else k) for k in keys)
             + " | Mean [95% CI] | Rank of 31 [interval] |", "|---|" + "---|" * len(keys) + "---|---|"]
    for arm in ("emotion_pc (k_e, LB)", "topic_pc (k_e, LB)", "shuffled_pc (k_e, LB)", "random (k_e, LB)",
                "raw (LB)", "all-170 (LB)", "headline", "D4 (LB)", "R1-repe"):
        r = summary.loc[arm]
        lines.append(f"| {arm} | " + " | ".join(f3(r[k]) for k in keys)
                     + f" | {f3(r['mean'])} [{f3(r['lo'])}, {f3(r['hi'])}] | {int(r['rank of 31'])} "
                       f"[{int(r['rank lo'])}-{int(r['rank hi'])}] |")
    lines += ["", "Mean AUROC over the 8 sets at each fixed k (Liars' Bench-trained):", "",
              "| k | emotion PCs | topic PCs | shuffled PCs (mean of 5) | random (mean of 20) |", "|---|---|---|---|---|"]
    for k in (2, 3, 5, 10, 20):
        lines.append(f"| {k} | " + " | ".join(f3(summary.loc[f"{fs} (k={k}, LB)", "mean"])
                                             for fs in ("emotion_pc",) + CONTROLS) + " |")
    lines += ["", "Secondary sources, at k_e:", "", "| source | emotion PCs | topic PCs | shuffled PCs | random |",
              "|---|---|---|---|---|"]
    for src in ("RepE", "dev"):
        lines.append(f"| {src} | " + " | ".join(f3(summary.loc[f"{fs} (k_e, {src})", "mean"])
                                               for fs in ("emotion_pc",) + CONTROLS) + " |")
    lines += ["", "Paired differences of the mean AUROC:", "", "| Arm | minus | Δ | 95% CI |", "|---|---|---|---|"]
    for r in deltas.itertuples():
        lines.append(f"| {r.arm} | {r.minus} | {r.delta:+.3f} | [{r.lo:+.3f}, {r.hi:+.3f}] |")
    tk = verdicts["TK1"]
    lines += ["", "- **TK1** (P1's rule, k_e, LB): " + "; ".join(
        f"minus {c} {v['delta']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]" for c, v in tk.items() if c != "holds")
        + f": {'holds' if tk['holds'] else 'fails'}."]
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"-> {path}")
    return path


if __name__ == "__main__":
    fire.Fire({"topic": topic, "train": train, "dev": dev, "score": score, "report": report})
