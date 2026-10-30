"""The all-170 emotion arm on Aletheia's Quest, scored as the thesis scored its detectors there. The design is
`aletheia170_spec.py`, frozen before any read (ANALYSIS_PLAN.md, "Aletheia's Quest transfer").

CPU steps run in E:/aletheia/venv (transformers 5.8, the thesis's renders) from the deception-detection folder;
the NDIF reads run `reader/ec_read.py --model qwen35|gemma` in E:/aletheia/venv08.

    python -m emotion_coords.aletheia170 rows --model gemma     # story, neutral and parity rows for the reader
    python -m emotion_coords.aletheia170 parity --model gemma   # the reader against the thesis's stored reads
    python -m emotion_coords.aletheia170 bases --model gemma    # emotion basis, shuffled bases, checks, valence gate
    python -m emotion_coords.aletheia170 train --model gemma    # every (source, feature set) probe; no test label
    python -m emotion_coords.aletheia170 dev --model gemma      # dev dry run and the known-answer gate
    python -m emotion_coords.aletheia170 score                  # the test sets; checks the freeze first

Each trained probe is kept as a direction v and offset in the read's own space (score = z @ v + offset), so the
scorer needs no basis: standardise-then-regress on coordinates z @ B.T folds into v = B.T @ (w / scale).
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from emotion_coords import aletheia170_spec as X
from emotion_coords import config as C
from emotion_coords import linalg as la
from emotion_coords import probes as P
from emotion_coords import vectors as V

STORE = Path("E:/emotion_coords/aletheia")  # bases and probe directions (arrays: never committed)


def out_dir() -> Path:
    return C.results_dir("aletheia")


def name(model: str, what: str) -> str:
    """Rows file and read folder of one model's stories, neutral texts or parity rows."""
    return f"aq_{model}_{what}"


def _check_model(model: str) -> None:
    from deception_detection.scripts.aletheia import spec as S

    if model not in X.MODELS:
        raise ValueError(f"model must be one of {X.MODELS}")
    if X.BLOCK[model] != S.MODELS[model].mid:
        raise RuntimeError(f"{model}: block {X.BLOCK[model]} is not the thesis's middle block {S.MODELS[model].mid}")


# --- rows for the reader --------------------------------------------------------------------------------------------


def _stored_parity_folder(model: str) -> tuple[Path, list[dict]]:
    """Rows the thesis already read at the middle block, with no label opened: Qwen3.5's Alpaca replies (its D9 + K
    control) and Gemma's instructed-deception dev rows."""
    from deception_detection.scripts.aletheia import alpaca as AL
    from deception_detection.scripts.aletheia import data as D
    from deception_detection.scripts.aletheia import read as AR

    if model == "qwen35":
        rows = [json.loads(x) for x in AL.path(model).read_text(encoding="utf-8").splitlines() if x.strip()]
        return AR.folder_for(model, "base", "alpaca", "no-reasoning"), rows
    key = "instructed_deception__gemma__None"
    return AR.folder_for(model, "base", f"dev__{key}", "no-reasoning"), D.rows(key, "dev")


def rows(model: str, n_parity: int = 20, parity_max_tokens: int = 1200) -> None:
    """The 8,280 stories of Llama's rows file and every neutral dialogue, tokenised by the model (BOS where its
    tokenizer adds one), pooled from position 50; and `n_parity` stored rows rendered as the thesis rendered them."""
    from emotion_coords import rows as ECR
    from deception_detection.scripts.aletheia import read as AR
    from deception_detection.scripts.aletheia import render as R

    _check_model(model)
    tok = R.tokenizer(model)
    start = X.POOL_FROM
    source = ECR._rows_dir() / f"{X.STORY_ROWS}.jsonl"
    meta = []
    with open(source, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta.append({k: r[k] for k in ("row_id", "emotion", "topic_idx", "story_idx", "rank")})
    con = ECR._stories_db()
    text = dict(con.execute(f"select id, text from stories_clean where id in "
                            f"({','.join(str(m['row_id']) for m in meta)})").fetchall())
    if len(text) != len(meta):
        raise RuntimeError(f"{len(meta) - len(text)} stories of {source.name} not found in stories.db")
    stories, short = [], 0
    for m in meta:
        ids = tok(text[m["row_id"]], add_special_tokens=True)["input_ids"]
        if len(ids) <= start:
            short += 1
            continue
        stories.append({**m, "ids": ids, "groups": [[[start, len(ids)]]], "points": []})
    ECR.write_rows(name(model, "stories"), stories,
                   {"model": model, "source_rows": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "dropped_short": short})
    path = C.DATA / "gemotions" / "data" / "neutral.db"
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    neutral, short = [], 0
    for did, topic_idx, body in con.execute("select id, topic_idx, text from dialogues order by id"):
        ids = tok(body or "", add_special_tokens=True)["input_ids"]
        if len(ids) <= start:
            short += 1
            continue
        neutral.append({"row_id": ECR.NEUTRAL_BASE + int(did), "ids": ids, "groups": [[[start, len(ids)]]],
                        "points": [], "topic_idx": int(topic_idx)})
    ECR.write_rows(name(model, "neutral"), neutral, {"model": model, "dropped_short": short})
    folder, candidates = _stored_parity_folder(model)
    stored = AR.load(folder, [X.BLOCK[model]])
    rendered = {int(r["index"]): R.render(r, model) for r in candidates if int(r["index"]) in stored}
    short_enough = sorted(i for i, it in rendered.items() if len(it["ids"]) <= parity_max_tokens)
    picked = sorted(np.random.default_rng(0).choice(short_enough, n_parity, replace=False).tolist())
    parity = [{"row_id": i, "ids": rendered[i]["ids"], "groups": [[list(rendered[i]["content"])]], "points": []}
              for i in picked]
    ECR.write_rows(name(model, "parity"), parity, {"model": model, "stored": str(folder)})


def parity(model: str) -> dict:
    """Gate: the reader's span means of the parity rows against the thesis's stored reads (cos >= 0.9998)."""
    from deception_detection.scripts.aletheia import read as AR

    _check_model(model)
    ids, content, _ = V.load_reads(name(model, "parity"))
    folder, _ = _stored_parity_folder(model)
    stored = AR.load(folder, [X.BLOCK[model]])
    cos = [float(la.unit(content[i, 0, 0]) @ la.unit(stored[int(r)][0, 0])) for i, r in enumerate(ids)]
    rel = [float(np.linalg.norm(content[i, 0, 0] - stored[int(r)][0, 0]) / np.linalg.norm(stored[int(r)][0, 0]))
           for i, r in enumerate(ids)]
    out = {"model": model, "rows": len(ids), "min_cos": min(cos), "max_rel_err": max(rel),
           "pass": bool(min(cos) >= X.PARITY_MIN_COS)}
    (out_dir() / f"parity_{model}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    logger.info(f"parity {model}: {out}")
    if not out["pass"]:
        raise SystemExit(f"GATE parity {model}: min cos {min(cos):.6f} < {X.PARITY_MIN_COS}")
    return out


# --- the emotion basis ------------------------------------------------------------------------------------------------


def build_bases(stories: V.Reads, neutral: V.Reads) -> tuple[V.Basis, list[V.Basis], np.ndarray, V.Reads]:
    """`phase1.final`'s recipe: denoise, the emotion basis with the top-up stories, five shuffled bases."""
    from emotion_coords import phase1

    topups = json.loads(phase1._topup_path().read_text())
    extra = {e: X.TOPUP_RANKS for e in topups}
    have = stories.meta.loc[stories.meta["rank"] > 0, "emotion"].value_counts()
    if [e for e in topups if have.get(e, 0) == 0]:
        raise RuntimeError("top-up stories missing from the reads")
    rank0 = (stories.meta["rank"] == 0).to_numpy()
    balanced = V.Reads(stories.x[rank0], stories.meta[rank0].reset_index(drop=True))
    denoise = la.variance_basis(neutral.x, X.NEUTRAL_VARIANCE)
    emotion = V.emotion_basis(stories, denoise, extra=extra)
    shuffled = V.shuffled_bases(balanced, denoise, n=X.N_SHUFFLED, seed=C.load()["seeds"]["stories"])
    return emotion, shuffled, denoise, balanced


def bases_path(model: str) -> Path:
    return STORE / "bases" / f"{model}_B{X.BLOCK[model]}.npz"


def bases(model: str) -> dict:
    """The model's emotion basis and shuffled bases at its middle block, the sanity checks, and the valence gate."""
    from emotion_coords import phase1

    _check_model(model)
    block = X.BLOCK[model]
    stories, neutral = V.Reads.of(name(model, "stories"), block), V.Reads.of(name(model, "neutral"), block)
    emotion, shuffled, denoise, balanced = build_bases(stories, neutral)
    split = V.split_half(balanced, denoise, seed=C.load()["seeds"]["stories"])
    geometry = V.geometry(emotion)
    arrays = {"E_vectors": emotion.vectors, "E_components": emotion.components, "E_explained": emotion.explained,
              "E_labels": np.array(emotion.labels), "denoise": denoise}
    for p, b in enumerate(shuffled):
        arrays[f"S{p}_components"] = b.components
    bases_path(model).parent.mkdir(parents=True, exist_ok=True)
    np.savez(bases_path(model), **arrays)
    out = {"model": model, "block": block, "stories": len(stories.x), "stories_rank0": int(len(balanced.x)),
           "neutral": len(neutral.x), "neutral_pcs_removed": len(denoise), "emotions": len(emotion.labels),
           "rank": int(len(emotion.components)), "shuffled_ranks": [int(len(b.components)) for b in shuffled],
           "geometry": geometry,
           "split_half_20v20": {"median": float(split["cos"].median()), "min": float(split["cos"].min()),
                                "under_0.7": int((split["cos"] < 0.7).sum())},
           "neighbours": V.neighbours(emotion, phase1.NEIGHBOURS),
           "valence_gate": bool(abs(geometry["r_pc1_valence"]) >= X.VALENCE_MIN_ABS_R)}
    (out_dir() / f"bases_{model}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    split.to_csv(out_dir() / f"split_half_{model}.csv", index=False)
    logger.info(f"{model} B{block}: rank {out['rank']}, |r(PC1, valence)| {abs(geometry['r_pc1_valence']):.3f}, "
                f"split-half median {out['split_half_20v20']['median']:.3f} -> {bases_path(model)}")
    if not out["valence_gate"]:
        raise SystemExit(f"GATE valence {model}: |r(PC1, valence)| = {abs(geometry['r_pc1_valence']):.3f} < "
                         f"{X.VALENCE_MIN_ABS_R}")
    return out


def check_llama(layer: int = 40) -> dict:
    """The basis code path on Llama's own reads reproduces phase1's stored basis (`bases/full_L<layer>.npz`)."""
    emotion, shuffled, _, _ = build_bases(V.Reads.of("stories", layer), V.Reads.of("neutral", layer))
    with np.load(C.data_dir("bases") / f"full_L{layer}.npz") as z:
        stored, stored_s0 = z["E_components"], z["S0_components"]
    sv = np.linalg.svd(emotion.components @ stored.T, compute_uv=False)
    sv0 = np.linalg.svd(shuffled[0].components @ stored_s0.T, compute_uv=False)
    out = {"layer": layer, "rank": len(emotion.components), "stored_rank": len(stored),
           "min_principal_cos": float(sv.min()), "min_principal_cos_shuffled0": float(sv0.min())}
    logger.info(f"Llama L{layer} basis rebuilt: {out}")
    return out


# --- training rows -------------------------------------------------------------------------------------------------


def lb_data(model: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """D4 (LB)'s rows by Liars' Bench set: Gemma's own pooled reads; Qwen2.5's rows read through Qwen3.5 base."""
    from deception_detection.scripts import lb_models as LB
    from deception_detection.scripts import recall_denial as RD
    from deception_detection.scripts.aletheia import posthoc as PH
    from deception_detection.scripts.aletheia import read as AR

    block = X.BLOCK[model]
    if model == "gemma":
        pooled = RD.Inputs("gemma").pooled(block)
        return {n: (pooled[n].z.astype(np.float64), np.asarray(pooled[n].y).astype(int))
                for n in LB.spec("gemma").datasets}
    labels = {int(k): int(v) for k, v in json.loads((PH.OUT / "lb_qwen25_labels.json").read_text()).items()}
    meta = PH.lb_rows()
    if {int(r["index"]): int(r["lie"]) for r in meta} != labels:
        raise RuntimeError("Qwen2.5's Liars' Bench rows no longer match the labels written at the read")
    reads = AR.load(AR.folder_for(model, "base", PH.LB_FOLDER, "no-reasoning"), [block])
    out = {}
    for n in sorted({r["dataset"] for r in meta}):
        ids = [int(r["index"]) for r in meta if r["dataset"] == n]
        out[n] = (np.stack([reads[i][0, 0] for i in ids]).astype(np.float64), np.array([labels[i] for i in ids]))
    return out


def repe_data(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R1's training rows, pooled, with one group per statement (for grouped CV)."""
    from deception_detection.scripts import lb_models as LB
    from deception_detection.scripts import persona_ood as PO
    from deception_detection.scripts.aletheia import repe as RE

    from deception_detection.utils import REPO_ROOT

    block = X.BLOCK[model]
    if model == "gemma":
        name_ = "repe_honesty__plain"
        p = PO.load_pooled(LB.spec("gemma").pooled_dirs(block), merge=False)[name_]
        # the pooled RepE order comes from the dataset, not the model: Llama's statement groups (pairs of adjacent
        # rows) apply when the label sequences agree
        llama = np.load(REPO_ROOT / "activations" / "pooled_L40" / f"{name_}.npz")["labels"].astype(str)
        mine = np.where(np.asarray(p.y).astype(int) == 1, "DECEPTIVE", "HONEST")
        if not np.array_equal(llama, mine):
            raise RuntimeError("Gemma's RepE rows are not in Llama's order")
        groups = pd.read_csv(C.data_dir("index") / "repe_groups.csv")
        groups = groups[groups["dataset"] == name_].set_index("pooled_row")["group"].sort_index().to_numpy()
        return p.z.astype(np.float64), np.asarray(p.y).astype(int), groups
    saved = np.load(RE.token_path(model, block))
    tokens, owner = saved["tokens"], saved["owner"].astype(np.int64)
    meta = {int(r["index"]): r for r in RE.rows()}
    order = sorted(meta)
    train = order[: int(RE.TRAIN_SHARE * len(order))]  # R1's training rows
    by_owner = np.argsort(owner, kind="stable")
    owners = owner[by_owner]
    starts = np.r_[0, np.flatnonzero(np.diff(owners)) + 1]
    means = np.add.reduceat(tokens[by_owner].astype(np.float64), starts, axis=0) / np.diff(np.r_[starts, len(owners)])[:, None]
    position = {int(o): j for j, o in enumerate(owners[starts])}
    if [i for i in train if i not in position]:
        raise RuntimeError("a RepE training row has no tokens")
    x = means[[position[i] for i in train]]
    y = np.array([int(meta[i]["label"]) for i in train])
    statement = [meta[i]["messages"][-1]["content"] for i in train]
    return x, y, pd.factorize(pd.Series(statement))[0]


def dev_data(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R2's rows: every dev set of the model read by its organism; groups = the organism (base model included)."""
    from deception_detection.scripts.aletheia import data as D
    from deception_detection.scripts.aletheia import score as SC

    xs, ys, gs = [], [], []
    for key in D.dev_keys():
        if key.split("__")[1] != model:
            continue
        got = SC._dev_reads_one(model, key, X.BLOCK[model])
        if got is None:
            continue
        xs.append(got[0])
        ys.append(got[1])
        gs += [key.split("__")[2]] * len(got[1])
    return np.concatenate(xs), np.concatenate(ys), pd.factorize(pd.Series(gs))[0]


def training(model: str, source: str) -> tuple[np.ndarray, np.ndarray, list]:
    """(x, y, inner folds) of a training source, as the spec fixes them."""
    if source == "LB":
        data = lb_data(model)
        x = np.concatenate([v[0] for v in data.values()])
        y = np.concatenate([v[1] for v in data.values()])
        sets = np.concatenate([np.full(len(v[1]), i) for i, v in enumerate(data.values())])
        return x, y, P._folds("sets", y, sets)
    if source == "RepE":
        x, y, groups = repe_data(model)
        return x, y, P._folds("grouped", y, groups)
    x, y, groups = dev_data(model)
    if len(np.unique(groups)) > 1:
        return x, y, P._folds("sets", y, groups)
    return x, y, P._folds("strat", y, None)


def alpaca(model: str) -> np.ndarray:
    from deception_detection.scripts.aletheia import score as SC

    return SC.inputs(model).control(X.BLOCK[model])


# --- probes ----------------------------------------------------------------------------------------------------------


def feature_sets(model: str, hidden: int) -> list[tuple[str, int, np.ndarray | None, bool]]:
    """(name, seed, basis or None, basis fitted on the training rows) for every feature set of the spec."""
    with np.load(bases_path(model)) as z:
        e = z["E_components"]
        shuffled = [z[f"S{p}_components"] for p in range(X.N_SHUFFLED)]
    r = len(e)
    out = [("emotion_all", 0, e, False), ("raw", 0, None, False)]
    out += [("shuffled_all", p, s[:r], False) for p, s in enumerate(shuffled)]
    out += [("random_all", s, la.random_subspace(hidden, r, seed=X.RANDOM_SEED + s), False) for s in range(X.N_RANDOM)]
    out += [("alpaca_all", 0, la.top_pcs(alpaca(model), r), False), ("trainpca_all", 0, None, True)]
    return out


def _direction(fitted: P.Fitted, basis: np.ndarray | None) -> tuple[np.ndarray, float]:
    w = fitted.weights / fitted.scale
    v = w if basis is None else basis.T @ w
    return v, float(fitted.offset - fitted.mean @ w)


def _part(model: str, source: str, fs: str, seed: int) -> Path:
    return STORE / "probes" / "parts" / f"{model}_{source}_{fs}_{seed}.npz"


def train(model: str, sources: str = "LB,RepE,dev") -> None:
    """Every (source, feature set) probe: C by the source's inner CV, then a fit on all its rows. Resumable by part."""
    _check_model(model)
    sources = sources.split(",") if isinstance(sources, str) else list(sources)
    with np.load(bases_path(model)) as z:
        r = len(z["E_components"])
    for source in sources:
        if source not in X.SOURCES:
            raise ValueError(f"source must be one of {tuple(X.SOURCES)}")
        x, y, folds = training(model, source)
        logger.info(f"{model} {source}: {len(y)} rows ({int(y.sum())} lies), {len(folds)} inner folds")
        for fs, seed, basis, per_train in feature_sets(model, x.shape[1]):
            path = _part(model, source, fs, seed)
            if path.exists():
                continue
            t0 = time.time()
            if per_train:  # protocol B's rule: the training-set PCs from every training row
                basis = la.top_pcs(x, min(r, len(y) - 1))
            feats = x if basis is None else x @ basis.T
            ks = None if basis is None else (len(basis),)
            sel = P.best_per_k(P.select("lr", feats, y, folds, ks, X.C_GRID))
            chosen = sel[sel["chosen"]].iloc[0]
            fitted = P.fit("lr", feats, y, float(chosen["C"]))
            v, offset = _direction(fitted, basis)
            gap = float(np.abs((x @ v + offset) - fitted.score(feats)).max())
            if gap > 1e-6 * max(1.0, float(np.abs(fitted.score(feats)).max())):
                raise RuntimeError(f"{fs}: the folded direction disagrees with the probe by {gap}")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, v=v, offset=offset, C=float(chosen["C"]), inner_auroc=float(chosen["inner_auroc"]),
                     k=feats.shape[1], n=len(y), train_auroc=P.auroc(y, x @ v + offset))
            logger.info(f"{model} {source} {fs}/{seed}: C {chosen['C']}, inner {chosen['inner_auroc']:.3f}, "
                        f"{time.time() - t0:.0f} s")
    table = []
    for source in X.SOURCES:
        for fs, seed in _feature_names():
            path = _part(model, source, fs, seed)
            if path.exists():
                with np.load(path) as z:
                    table.append({"source": source, "feature_set": fs, "seed": seed, "C": float(z["C"]),
                                  "inner_auroc": float(z["inner_auroc"]), "k": int(z["k"]), "n": int(z["n"]),
                                  "train_auroc": float(z["train_auroc"])})
    pd.DataFrame(table).to_csv(out_dir() / f"probes_{model}.csv", index=False)


def _feature_names() -> list[tuple[str, int]]:
    names = [("emotion_all", 0), ("raw", 0)] + [("shuffled_all", p) for p in range(X.N_SHUFFLED)]
    return names + [("random_all", s) for s in range(X.N_RANDOM)] + [("alpaca_all", 0), ("trainpca_all", 0)]


def probes(model: str, source: str) -> dict[tuple[str, int], tuple[np.ndarray, float]]:
    """The trained probes of a source as (direction, offset) by (feature set, seed)."""
    out = {}
    for fs, seed in _feature_names():
        with np.load(_part(model, source, fs, seed)) as z:
            out[fs, seed] = (z["v"], float(z["offset"]))
    return out


# --- dev dry run and the known-answer gate ---------------------------------------------------------------------------


def dev(model: str) -> dict:
    """The LB- and RepE-trained probes on every dev set read by its organism (dev labels only), and the gate."""
    from deception_detection.scripts.aletheia import data as D
    from deception_detection.scripts.aletheia import score as SC
    from deception_detection.scripts.aletheia import spec as S

    _check_model(model)
    trained = {src: probes(model, src) for src in ("LB", "RepE")}
    rows = []
    for key in D.dev_keys():
        if key.split("__")[1] != model:
            continue
        got = SC._dev_reads_one(model, key, X.BLOCK[model])
        if got is None:
            continue
        z, y = got
        for src, table in trained.items():
            for (fs, seed), (v, off) in table.items():
                rows.append({"dev set": key, "source": src, "feature_set": fs, "seed": seed, "n": len(y),
                             "auroc": P.auroc(y, z @ v + off)})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir() / f"dev_{model}.csv", index=False)
    key = f"instructed_deception__{model}__None"
    z, y = SC._dev_reads_one(model, key, X.BLOCK[model])
    v, off = trained["LB"]["emotion_all", 0]
    value = P.auroc(y, z @ v + off)
    null = np.random.default_rng(S.SEED).standard_normal((X.N_NULL, z.shape[1]))
    null /= np.linalg.norm(null, axis=1, keepdims=True)
    band = np.array([P.auroc(y, s) for s in (z @ null.T).T])
    gate = {"model": model, "dev set": key, "block": X.BLOCK[model], "emotion_all_LB": value,
            "band97.5": float(np.quantile(band, 0.975)), "pass": bool(value > np.quantile(band, 0.975))}
    (out_dir() / f"dev_gate_{model}.json").write_text(json.dumps(gate, indent=1), encoding="utf-8")
    summary = frame.groupby(["source", "feature_set"])["auroc"].mean().unstack(0).round(3)
    logger.info(f"{model} dev (mean over dev sets):\n{summary.to_string()}\ngate {gate}")
    if not gate["pass"]:
        raise SystemExit(f"GATE dev {model}: {value:.3f} <= band 97.5th percentile {gate['band97.5']:.3f}")
    return gate


# --- the test sets ----------------------------------------------------------------------------------------------------


def check_freeze() -> str:
    digest = hashlib.sha256(Path(X.__file__).read_bytes()).hexdigest().upper()
    if not X.FREEZE.exists() or digest not in X.FREEZE.read_text(encoding="utf-8"):
        raise RuntimeError(f"aletheia170_spec.py ({digest[:12]}) is not the frozen design ({X.FREEZE})")
    logger.info(f"aletheia170_spec.py sha256 {digest} = the freeze")
    return digest


def _boots(y: np.ndarray, s: np.ndarray, draws: list[np.ndarray]) -> np.ndarray:
    from deception_detection.scripts.aletheia import score as SC

    return np.array([SC.auroc(y[b], s[b]) for b in draws])


def score() -> None:
    """AUROC per dataset and cell with `score.run`'s bootstrap draws and random band; the thesis's cells re-scored on
    the same draws; means, ranks among the 30 entries, paired deltas, H1 and H2. The only step that opens test labels."""
    from deception_detection.scripts.aletheia import data as D
    from deception_detection.scripts.aletheia import posthoc as PH
    from deception_detection.scripts.aletheia import score as SC
    from deception_detection.scripts.aletheia import spec as S

    digest = check_freeze()
    trained = {(m, src): probes(m, src) for m in X.MODELS for src in X.SOURCES}
    thesis_cells = {"headline": S.HEADLINE, "R1-repe": S.REFERENCES[0], "R2-dev-probe": S.REFERENCES[1]}
    records, boots = [], {}
    for d in S.DATASETS:
        if d.key not in X.SCORED:
            continue
        index = S.DATASETS.index(d)
        read_name, block = SC.organism_read(d), d.base.mid
        z, ids = SC._reads(d, read_name, "no-reasoning", 0, block)
        if ids != [int(r["index"]) for r in D.rows(d.key)]:
            raise RuntimeError(f"{d.key}: reads out of row order")
        labels = D.labels(d.key, "test")
        y = np.array([int(labels[i]) for i in ids])
        rng = np.random.default_rng([S.SEED, index])
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        draws = [np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg)))) for _ in range(S.N_BOOT)]
        nrng = np.random.default_rng([S.SEED, 7, index])
        null = nrng.standard_normal((S.N_NULL, z.shape[1]))
        null /= np.linalg.norm(null, axis=1, keepdims=True)
        band = np.array([SC.auroc(y, t) for t in (z @ null.T).T])
        reads = {"organism": z}
        base = SC._reads(d, "base", "no-reasoning", 0, block)
        reads["base"] = base[0] if base is not None else None

        def add(cell: str, s: np.ndarray, **key) -> None:
            value = SC.auroc(y, s)
            b = _boots(y, s, draws)
            boots[cell, d.key] = b
            records.append({"cell": cell, "dataset": d.key, "model": d.model, "organism_read": read_name,
                            "block": block, "n": len(y), "lies": int(y.sum()), "auroc": value,
                            "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975)),
                            "band97.5": float(np.quantile(band, 0.975)), "p_null": float(np.mean(band >= value))}
                           | key)

        for src in X.SOURCES:
            for (fs, seed), (v, off) in trained[d.model, src].items():
                add(f"{fs}/{seed} ({src})", z @ v + off, source=src, feature_set=fs, seed=seed, read="organism")
                if src == "LB" and reads["base"] is not None:
                    add(f"{fs}/{seed} ({src}, base read)", reads["base"] @ v + off, source=src, feature_set=fs,
                        seed=seed, read="base")
        for cell_name, cell in thesis_cells.items():
            got = SC.scores(cell, d)
            if got is None:
                raise RuntimeError(f"{cell_name} {d.key}: not scorable")
            if got[1] != ids:
                raise RuntimeError(f"{cell_name} {d.key}: rows differ")
            add(cell_name, got[0], source="thesis", feature_set=cell_name, seed=0, read="organism")
        add("D4 (LB)", np.asarray(PH.d4_lb(d.model, block)(z), dtype=np.float64), source="thesis",
            feature_set="D4 (LB)", seed=0, read="organism")
        logger.info(f"{d.key}: {len(y)} rows scored")
    table = pd.DataFrame(records)
    out = out_dir()
    table.to_csv(out / "auroc.csv", index=False)
    np.savez(out / "boots.npz", **{f"{c}|{k}": v for (c, k), v in boots.items()})
    _verify_thesis(table, boots)
    _summarise(table, boots, digest)


def _verify_thesis(table: pd.DataFrame, boots: dict) -> None:
    """The re-scored thesis cells reproduce the thesis's tables, and the headline's bootstrap its stored draws."""
    from deception_detection.scripts.aletheia import spec as S

    theirs = pd.read_csv(S.RESULTS / "auroc.csv").set_index(["cell", "dataset"])["auroc"]
    stored = dict(np.load(S.RESULTS / "boots.npz"))
    posthoc = pd.read_csv(S.RESULTS / "posthoc" / "auroc.csv").set_index(["detector", "dataset"])["auroc"]
    worst, worst_boot = 0.0, 0.0
    for r in table[table["source"] == "thesis"].itertuples():
        ref = posthoc[r.cell, r.dataset] if r.cell == "D4 (LB)" else theirs[r.cell, r.dataset]
        worst = max(worst, abs(r.auroc - ref))
        if r.cell != "D4 (LB)":
            worst_boot = max(worst_boot, float(np.abs(boots[r.cell, r.dataset] - stored[f"{r.cell}|{r.dataset}"]).max()))
    logger.info(f"thesis cells re-scored: max |AUROC - thesis table| {worst:.2e}, max |bootstrap - stored| {worst_boot:.2e}")
    if worst > 1e-9 or worst_boot > 1e-9:
        raise RuntimeError("the re-scored thesis cells do not reproduce the thesis's tables")


def _combined(table: pd.DataFrame, boots: dict, cells: list[str], datasets: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-dataset AUROC and per-draw bootstrap AUROC of a cell, averaged over `cells` (a control's seeds)."""
    t = table.set_index(["cell", "dataset"])["auroc"]
    point = np.array([np.mean([t[c, k] for c in cells]) for k in datasets])
    boot = np.stack([np.mean([boots[c, k] for c in cells], axis=0) for k in datasets])
    return point, boot


def _arms(table: pd.DataFrame) -> dict[str, list[str]]:
    """Reported arms: each feature set (seeds averaged) per source and read, and the thesis's cells."""
    arms = {}
    for r in table.drop_duplicates("cell").itertuples():
        if r.source == "thesis":
            arms[r.cell] = [r.cell]
            continue
        label = f"{r.feature_set} ({r.source}{', base read' if r.read == 'base' else ''})"
        arms.setdefault(label, []).append(r.cell)
    return arms


def _summarise(table: pd.DataFrame, boots: dict, digest: str) -> None:
    from deception_detection.scripts.aletheia import report as RP

    board = RP._board()
    datasets = [k for k in X.SCORED if k in set(table["dataset"])]
    entries = board[datasets].mean(axis=1).to_numpy()
    arms = _arms(table)
    means, rows = {}, []
    for arm, cells in arms.items():
        have = [k for k in datasets if all((c, k) in boots for c in cells)]
        if have != datasets:
            continue
        point, boot = _combined(table, boots, cells, datasets)
        m, bm = float(point.mean()), boot.mean(axis=0)
        means[arm] = bm
        ranks = np.array([RP._rank(b, entries) for b in bm])
        rows.append({"arm": arm, "datasets": len(datasets), "mean": m,
                     "lo": float(np.quantile(bm, 0.025)), "hi": float(np.quantile(bm, 0.975)),
                     "rank of 31": RP._rank(m, entries), "rank lo": int(np.quantile(ranks, 0.025)),
                     "rank hi": int(np.quantile(ranks, 0.975))} | {k: float(v) for k, v in zip(datasets, point)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir() / "summary.csv", index=False)
    pairs = []
    primary = "emotion_all (LB)"
    for other in ("raw (LB)", "shuffled_all (LB)", "random_all (LB)", "alpaca_all (LB)", "trainpca_all (LB)",
                  "headline", "D4 (LB)", "R1-repe", "R2-dev-probe"):
        pairs.append((primary, other))
    for src in ("RepE", "dev"):
        pairs += [(f"emotion_all ({src})", f"{c} ({src})") for c in
                  ("raw", "shuffled_all", "random_all", "alpaca_all", "trainpca_all")]
    pairs.append(("emotion_all (LB, base read)", "raw (LB, base read)"))
    deltas = []
    for a, b in pairs:
        if a not in means or b not in means:
            continue
        diff = means[a] - means[b]
        point = float(summary.set_index("arm").loc[a, "mean"] - summary.set_index("arm").loc[b, "mean"])
        deltas.append({"arm": a, "minus": b, "delta": point, "lo": float(np.quantile(diff, 0.025)),
                       "hi": float(np.quantile(diff, 0.975)), "share_above_0": float((diff > 0).mean())})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out_dir() / "deltas.csv", index=False)
    d = deltas.set_index(["arm", "minus"])
    h1 = d.loc[(primary, "raw (LB)")]
    h2 = {c: d.loc[(primary, c)] for c in ("shuffled_all (LB)", "random_all (LB)", "alpaca_all (LB)", "trainpca_all (LB)")}
    verdicts = {"freeze_sha256": digest, "datasets": datasets,
                "H1": {"delta": float(h1["delta"]), "ci": [float(h1["lo"]), float(h1["hi"])], "holds": bool(h1["lo"] > 0)},
                "H2": {c: {"delta": float(v["delta"]), "ci": [float(v["lo"]), float(v["hi"])], "holds": bool(v["lo"] > 0)}
                       for c, v in h2.items()}}
    verdicts["H2"]["holds"] = all(v["holds"] for v in verdicts["H2"].values())
    (out_dir() / "verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    logger.info("\n" + summary[["arm", "mean", "lo", "hi", "rank of 31", "rank lo", "rank hi"]].round(4).to_string())
    logger.info("\n" + deltas.round(4).to_string())
    logger.info(json.dumps(verdicts, indent=1))


if __name__ == "__main__":
    fire.Fire({"rows": rows, "parity": parity, "bases": bases, "check_llama": check_llama, "train": train,
               "dev": dev, "score": score})
