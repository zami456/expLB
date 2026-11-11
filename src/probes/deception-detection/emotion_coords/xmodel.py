"""The emotion-coordinate study on every other Liars' Bench model, as the thesis's cross-model replication tested its
own detectors (design: `xmodel_spec.py`, frozen before any read). Each model reads the stories through itself and
is probed on its own Liars' Bench rollouts: phase 1's bases and phase 3's protocols, unchanged.

Runs in E:/aletheia/venv from the deception-detection folder; the NDIF reads are `reader/ec_read.py --model
qwen25|mistral|gemma` with the `xm_*.jobs` files. `model` is an `lb_models` key: qwen, mistral, gemma.

    python -m emotion_coords.xmodel rows --model qwen        # stories, neutral texts, Liars' Bench band rows
    python -m emotion_coords.xmodel parity --model qwen      # the new Liars' Bench reads against the thesis's
    python -m emotion_coords.xmodel bases --model qwen       # the bases at every block, with the checks
    python -m emotion_coords.xmodel run --model qwen         # phase 3 (A, B, C; logistic regression) at every block
    python -m emotion_coords.xmodel summary --model qwen     # P1-P5, the all-170 arm, XA1/XA2, transfer
    python -m emotion_coords.xmodel tables                   # the cross-model tables
"""

import hashlib
import json
import os
import pickle
import sqlite3
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from emotion_coords import config as C
from emotion_coords import linalg as la
from emotion_coords import probes as P
from emotion_coords import vectors as V
from emotion_coords import xmodel_spec as XS
from emotion_coords.lbdata import Split

STORE = Path("E:/emotion_coords/xmodel")
PARTIAL = Path("E:/liars-bench-acts/pooled_models_partial")
CONTROL, TRAIN, VAL = "XXXX-20__plain", "repe_honesty__plain", "repe_honesty__plain_val"
IT = "insider_trading_cadenza__plain"
VAL_OFFSET = 100_000  # RepE val rows after the train rows, as `lbdata`


def out_dir(*parts: str) -> Path:
    return C.results_dir("xmodel", *parts)


def lb_sets(model: str) -> list[str]:
    from deception_detection.scripts import lb_models as LB

    return list(LB.spec(model).datasets)


def blocks(model: str) -> list[int]:
    m = XS.MODELS[model]
    return sorted(set(m["named"]) | set(m["band"]))


def reads(model: str, what: str) -> str:
    """Rows file / read folder of this study: stories, neutral, lb (the band's Liars' Bench rows), lb_parity."""
    return f"xm_{model}_{what}"


# --- rows --------------------------------------------------------------------------------------------------------------


def _tokenizer(model: str):
    from transformers import AutoTokenizer

    C.load_env()
    extra = {"fix_mistral_regex": True} if model == "mistral" else {}  # the shipped tokenizer's pattern is wrong
    return AutoTokenizer.from_pretrained(XS.MODELS[model]["hf"], token=os.environ.get("HF_TOKEN"), **extra)


def _story_rows(model: str) -> None:
    """Llama's 8,280 stories and every neutral dialogue, tokenised by `model`, pooled from position 50."""
    from emotion_coords import rows as ECR

    tok, start = _tokenizer(model), XS.POOL_FROM
    meta = []
    with open(ECR._rows_dir() / f"{XS.STORY_ROWS}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta.append({k: r[k] for k in ("row_id", "emotion", "topic_idx", "story_idx", "rank")})
    text = dict(ECR._stories_db().execute(f"select id, text from stories_clean where id in "
                                          f"({','.join(str(m['row_id']) for m in meta)})").fetchall())
    stories = []
    for m in meta:
        ids = tok(text[m["row_id"]], add_special_tokens=True)["input_ids"]
        if len(ids) > start:
            stories.append({**m, "ids": ids, "groups": [[[start, len(ids)]]], "points": []})
    ECR.write_rows(reads(model, "stories"), stories, {"model": model, "dropped_short": len(meta) - len(stories)})
    con = sqlite3.connect(f"file:{(C.DATA / 'gemotions' / 'data' / 'neutral.db').as_posix()}?mode=ro", uri=True)
    neutral = []
    for did, topic_idx, body in con.execute("select id, topic_idx, text from dialogues order by id"):
        ids = tok(body or "", add_special_tokens=True)["input_ids"]
        if len(ids) > start:
            neutral.append({"row_id": ECR.NEUTRAL_BASE + int(did), "ids": ids, "groups": [[[start, len(ids)]]],
                            "points": [], "topic_idx": int(topic_idx)})
    ECR.write_rows(reads(model, "neutral"), neutral, {"model": model})


def _cache(model: str, name: str) -> Path:
    """`layer_sweep.pool`'s cached tokenisation of a set (insider trading: the variant with its "+trim" span)."""
    folder = PARTIAL / XS.MODELS[model]["folder"]
    found = [p for p in sorted(folder.glob(f"{name}__B*")) if (p / "tokenized.pt").exists()
             and p.name.endswith("+trim") == (name == IT)]
    if not found:
        raise FileNotFoundError(f"no cached tokenisation of {name} under {folder}")
    return found[0] / "tokenized.pt"


def _cached(model: str, name: str) -> tuple[list[dict], list[str]]:
    """Every kept row of a set as the pooled reads hold it (row id = code * 1e6 + pooled row), and their labels."""
    import torch

    from emotion_coords.rows import DATASET_CODES

    saved = torch.load(_cache(model, name), weights_only=True)
    trim = name == IT
    mask = saved["masks"][1 if trim else 0]
    labels = saved["labels"]["_trim" if trim else ""]
    kept = [i for i in range(len(saved["tokens"])) if all(bool(m[i].any()) for m in saved["masks"])]
    out = []
    for k, i in enumerate(kept):
        real = saved["attention"][i].bool()  # left padding
        on = np.flatnonzero(mask[i][real].numpy())
        runs = np.split(on, np.flatnonzero(np.diff(on) != 1) + 1)
        out.append({"row_id": DATASET_CODES[name] * 1_000_000 + k, "ids": saved["tokens"][i][real].tolist(),
                    "groups": [[[int(r[0]), int(r[-1]) + 1] for r in runs]], "points": [], "dataset": name,
                    "label": labels[i]})
    return out, [labels[i] for i in kept]


def capped(model: str, name: str, labels: list[str]) -> np.ndarray:
    """The pooled rows kept for a Liars' Bench set: honest and deceptive only, at most CAP per label (CAP_IT for
    insider trading), drawn as the Llama study drew (`rows.pick_lb`)."""
    from emotion_coords.rows import pick_lb

    index = pd.DataFrame({"pooled_row": np.arange(len(labels)), "label": labels})
    index = index[index["label"].isin(["HONEST", "DECEPTIVE"])]
    cap = XS.CAP_IT if name == IT else XS.CAP
    return pick_lb(index, cap, seed=C.load()["seeds"]["stories"])["pooled_row"].to_numpy()


def rows(model: str, n_parity: int = 20) -> None:
    """Stories and neutral texts (Qwen2.5, Mistral; Gemma's are the Aletheia test's), and the Liars' Bench rows for
    the band blocks the thesis did not pool: the capped sets, RepE (train and val) and Alpaca, with a parity subset."""
    from emotion_coords import rows as ECR

    if model == "gemma":
        ECR.write_rows(reads("gemma", "repeval"), _cached("gemma", VAL)[0], {"model": "gemma", "rows_of": VAL})
        return
    _story_rows(model)
    items = []
    for name in [*lb_sets(model), TRAIN, VAL, CONTROL]:
        got, labels = _cached(model, name)
        keep = set(capped(model, name, labels).tolist()) if name in lb_sets(model) else set(range(len(got)))
        items += [it for k, it in enumerate(got) if k in keep]
    ECR.write_rows(reads(model, "lb"), items, {"model": model, "caps": [XS.CAP, XS.CAP_IT]})
    sets = [it for it in items if it["dataset"] in lb_sets(model) and len(it["ids"]) <= 1500]
    pick = np.random.default_rng(0).choice(len(sets), n_parity, replace=False)
    ECR.write_rows(reads(model, "lb_parity"), [sets[i] for i in sorted(pick)], {"parity_of": reads(model, "lb")})


# --- the Liars' Bench data at a block ------------------------------------------------------------------------------------


def _thesis_file(model: str, block: int, name: str) -> Path | None:
    from deception_detection.scripts import lb_models as LB

    found = None
    for folder in LB.spec(model).pooled_dirs(block):  # a later folder's file replaces an earlier one's (IT: "_trim")
        if (folder / f"{name}.npz").exists():
            found = folder / f"{name}.npz"
    return found


def _new_reads(model: str, block: int, name: str) -> str:
    """This study's read folder holding `name` at a block the thesis did not pool."""
    if model != "gemma":
        return reads(model, "lb")
    return {CONTROL: "aqb_gemma_alpaca", TRAIN: "aqb_gemma_repe", VAL: reads("gemma", "repeval")}.get(name, "aqb_gemma_lb")


def _pooled(model: str, block: int, name: str, rows_wanted: np.ndarray) -> np.ndarray:
    """The pooled reads of `rows_wanted` (pooled row numbers) of a set at `block`, from the thesis or this study."""
    from emotion_coords.rows import DATASET_CODES

    path = _thesis_file(model, block, name)
    if path is not None:
        with np.load(path) as z:
            return z["pooled"][rows_wanted].astype(np.float64)
    folder = _new_reads(model, block, name)
    ids, content, _ = V.load_reads(folder, V._layer_index(block, folder))
    position = {int(r): i for i, r in enumerate(ids)}
    code = DATASET_CODES[name] * 1_000_000
    missing = [int(k) for k in rows_wanted if code + int(k) not in position]
    if missing:
        raise RuntimeError(f"{model} B{block} {name}: {len(missing)} rows not read")
    return content[[position[code + int(k)] for k in rows_wanted], 0, 0].astype(np.float64)


def _labels(model: str, name: str) -> list[str]:
    """A set's labels in pooled-row order (the thesis's pooled file at the model's middle named block)."""
    path = _thesis_file(model, XS.MODELS[model]["named"][2], name)
    with np.load(path) as z:
        return [str(x) for x in z["labels"]]


def data(model: str, block: int) -> dict[str, Split]:
    """The model's capped Liars' Bench sets, its Alpaca replies and the RepE pairs (train and val merged, with
    Llama's statement groups after a label-order check) at `block`, as `lbdata.Split`s."""
    from deception_detection.utils import REPO_ROOT

    out = {}
    for name in [*lb_sets(model), CONTROL]:
        labels = np.array(_labels(model, name))
        keep = capped(model, name, labels.tolist()) if name != CONTROL else np.arange(len(labels))
        x = _pooled(model, block, name, keep)
        out[name] = Split(name, x, (labels[keep] == "DECEPTIVE").astype(int), keep)
    groups_all = pd.read_csv(C.data_dir("index") / "repe_groups.csv")
    xs, ys, rows_, gs = [], [], [], []
    for k, part in enumerate((TRAIN, VAL)):
        labels = np.array(_labels(model, part))
        llama = np.load(REPO_ROOT / "activations" / "pooled_L40" / f"{part}.npz")["labels"].astype(str)
        if not np.array_equal(labels, llama):
            raise RuntimeError(f"{model} {part}: not in Llama's order, so Llama's statement groups do not apply")
        keep = np.arange(len(labels))
        xs.append(_pooled(model, block, part, keep))
        ys.append((labels == "DECEPTIVE").astype(int))
        rows_.append(keep + k * VAL_OFFSET)
        g = groups_all[groups_all["dataset"] == part].set_index("pooled_row")["group"]
        gs.append(g.loc[keep].to_numpy())
    out[TRAIN] = Split(TRAIN, np.concatenate(xs), np.concatenate(ys), np.concatenate(rows_), np.concatenate(gs))
    for name, split in out.items():
        if not np.isfinite(split.x).all():
            raise RuntimeError(f"{model} B{block} {name}: non-finite reads")
    return out


# --- parity --------------------------------------------------------------------------------------------------------------


def parity(model: str) -> dict:
    """Gate: the parity subset of the new Liars' Bench reads against the thesis's pooled reads at the shared block."""
    from emotion_coords.aletheia_band import _compare
    from emotion_coords.rows import DATASET_CODES

    if model == "gemma":
        raise ValueError("Gemma's band reads passed parity in the Aletheia band test")
    names = {c: n for n, c in DATASET_CODES.items()}
    b = XS.MODELS[model]["parity_block"]
    stored = {}

    def ref(r: int, block: int):
        name = names[r // 1_000_000]
        if name not in stored:
            with np.load(_thesis_file(model, block, name)) as z:
                stored[name] = z["pooled"]
        return stored[name][r % 1_000_000]

    out = _compare(reads(model, "lb_parity"), ref, [b])
    out["pass"] = bool(out["min_cos"] is not None and out["min_cos"] >= XS.PARITY_MIN_COS)
    (out_dir(model) / "parity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    logger.info(f"{model} parity: {out}")
    if not out["pass"]:
        raise SystemExit(f"GATE parity {model}: min cos {out['min_cos']}")
    return out


# --- bases ---------------------------------------------------------------------------------------------------------------


def _story_reads(model: str, block: int) -> tuple[str, str]:
    if model == "gemma" and block in XS.MODELS["gemma"]["band"]:
        return "aqb_gemma_stories", "aqb_gemma_neutral"
    return reads(model, "stories"), reads(model, "neutral")


def bases_path(model: str, block: int) -> Path:
    return STORE / "bases" / f"{model}_B{block}.npz"


def bases(model: str) -> list[dict]:
    """Phase 1's bases at every block (emotion, topic, five shuffled), with the checks; valence flagged under 0.5."""
    from emotion_coords import phase1
    from emotion_coords.aletheia170 import build_bases

    out = []
    for b in blocks(model):
        s_name, n_name = _story_reads(model, b)
        stories, neutral = V.Reads.of(s_name, b), V.Reads.of(n_name, b)
        emotion, shuffled, denoise, balanced = build_bases(stories, neutral)
        topic = V.topic_basis(balanced, denoise)
        arrays = {"E_vectors": emotion.vectors, "E_components": emotion.components, "E_explained": emotion.explained,
                  "E_labels": np.array(emotion.labels), "T_vectors": topic.vectors, "T_components": topic.components,
                  "T_explained": topic.explained, "T_labels": np.array(topic.labels), "denoise": denoise}
        for p, s in enumerate(shuffled):
            arrays[f"S{p}_components"] = s.components
        bases_path(model, b).parent.mkdir(parents=True, exist_ok=True)
        np.savez(bases_path(model, b), **arrays)
        g = V.geometry(emotion)
        split = V.split_half(balanced, denoise, seed=C.load()["seeds"]["stories"])
        out.append({"block": b, "rank": len(emotion.components), "neutral_pcs_removed": len(denoise),
                    "r_pc1_valence": g["r_pc1_valence"], "r_pc2_arousal": g["r_pc2_arousal"],
                    "split_half_median": float(split["cos"].median()),
                    "valence_ok": bool(abs(g["r_pc1_valence"]) >= XS.VALENCE_MIN_ABS_R),
                    "neighbours": V.neighbours(emotion, phase1.NEIGHBOURS[:2])})
        logger.info(f"{model} B{b}: rank {out[-1]['rank']}, |r(PC1, valence)| {abs(g['r_pc1_valence']):.3f}, "
                    f"split-half {out[-1]['split_half_median']:.3f}")
    (out_dir(model) / "bases.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# --- phase 3 ---------------------------------------------------------------------------------------------------------------


def _parts(model: str) -> Path:
    path = STORE / "parts" / model
    path.mkdir(parents=True, exist_ok=True)
    return path


def _unit(fs: P.FeatureSet, data_: dict, lb: list[str], layer: int, path: str) -> tuple[str, float]:
    """Phase 3's unit for one feature set (`phase3._unit`, logistic regression): protocols A, B and C."""
    from emotion_coords.phase3 import _write

    store, t0 = P.Store(), time.time()
    P.protocol_a("lr", fs, data_, TRAIN, lb, CONTROL, layer, "a", store)
    chosen = P.protocol_b("lr", fs, data_, lb, CONTROL, layer, "a", store)
    P.protocol_c("lr", fs, data_, lb, CONTROL, layer, "a", store)
    seconds = time.time() - t0
    _write(Path(path), {"rows": store.rows, "scores": store.scores, "chosen_c": chosen, "seconds": seconds})
    return Path(path).name, seconds


def feature_sets(model: str, layer: int, data_: dict, base: dict, n_random: int = 20) -> list[P.FeatureSet]:
    from emotion_coords.phase3 import exploratory_sets
    from emotion_coords.phase3 import feature_sets as phase3_sets
    from emotion_coords.pilot import persona_basis

    hidden = data_[CONTROL].x.shape[1]
    persona = (persona_basis(layer, "extended", model), persona_basis(layer, "published", model))
    sets = phase3_sets(layer, data_, base, hidden, n_random, persona) + exploratory_sets(data_, base, hidden)
    sets.append(P.FeatureSet("emotion_vectors_all", "exploratory_emotion_vectors_all", la.unit(base["E_vectors"]),
                             (len(base["E_vectors"]),)))
    return sets


def run(model: str, blocks_=None, workers: int = 4, threads: int = 3, n_random: int = 20) -> None:
    """Every feature set at every block (resumable by part), then the zero-shot directions."""
    from joblib import Parallel, delayed

    from emotion_coords.phase3 import HEAVY, _write
    from emotion_coords.pilot import ZERO_SHOT

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    lb = lb_sets(model)
    tmp = STORE / "joblib_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for layer in (blocks(model) if blocks_ is None else C.int_list(blocks_)):
        t0 = time.time()
        data_ = data(model, layer)
        with np.load(bases_path(model, layer)) as z:
            base = {k: z[k] for k in z.files}
        sets = sorted(feature_sets(model, layer, data_, base, n_random), key=lambda f: f.name not in HEAVY)
        name = lambda fs: f"L{layer}_lr_{fs.name}_s{fs.seed}.pkl"  # noqa: E731
        todo = [fs for fs in sets if not (_parts(model) / name(fs)).exists()]
        logger.info(f"{model} B{layer}: {len(sets)} feature sets, {len(todo)} to fit on {workers} workers")
        done = Parallel(n_jobs=workers, backend="loky", temp_folder=str(tmp), max_nbytes="1M")(
            delayed(_unit)(fs, data_, lb, layer, str(_parts(model) / name(fs))) for fs in todo)
        for unit, seconds in done:
            logger.info(f"  {unit}: {seconds:.0f} s")
        extra = _parts(model) / f"L{layer}_zero_shot.pkl"
        if not extra.exists():
            store = P.Store()
            labels = list(base["E_labels"])
            P.zero_shot({n: (base["E_vectors"][labels.index(n)], sign) for n, sign in ZERO_SHOT.items() if n in labels},
                        data_, lb, CONTROL, layer, "a", store)
            _write(extra, {"rows": store.rows, "scores": store.scores, "chosen_c": None, "seconds": 0.0})
        logger.info(f"{model} B{layer} done in {(time.time() - t0) / 60:.1f} min")


# --- summaries ---------------------------------------------------------------------------------------------------------------


def check_freeze() -> str:
    digest = hashlib.sha256(Path(XS.__file__).read_bytes()).hexdigest().upper()
    if not XS.FREEZE.exists() or digest not in XS.FREEZE.read_text(encoding="utf-8"):
        raise RuntimeError(f"xmodel_spec.py ({digest[:12]}) is not the frozen design ({XS.FREEZE})")
    return digest


def _merged(model: str) -> tuple[pd.DataFrame, dict]:
    frame, scores = [], {}
    for path in sorted(_parts(model).glob("*.pkl")):
        with open(path, "rb") as f:
            part = pickle.load(f)
        frame += part["rows"]
        scores.update(part["scores"])
    return pd.DataFrame(frame), scores


def _transfer(frame: pd.DataFrame, layer: int, feature_set: str, k: int | None = None) -> float:
    """Protocol C's off-diagonal mean AUROC (seeds averaged), at k or at each source's chosen k."""
    c = frame[(frame["protocol"] == "C") & (frame["layer"] == layer) & (frame["feature_set"] == feature_set)
              & (frame["train"] != frame["test"])]
    c = c[c["k"] == k] if k is not None else c[c["chosen"]]
    return float(c.groupby(["train", "test"])["auroc"].mean().mean()) if len(c) else float("nan")


def summary(model: str) -> dict:
    """Per block: protocol B by set for every arm, the all-170 arm against raw and its controls; P1-P5 at the
    named layers (Llama's rules, where the sets allow); XA1 and XA2 over the band; protocol C's transfer."""
    from emotion_coords.phase3 import Arms, _delta

    digest = check_freeze()
    frame, scores = _merged(model)
    lb = lb_sets(model)
    has = {"IT": IT in lb, "HP": {"harm_refusal_choice__plain", "harm_refusal_knowledge_report__plain"} <= set(lb),
           "ID": "instruct_dishonesty__plain" in lb}
    per_block, verdicts, transfer = [], [], []
    for layer in blocks(model):
        d = data(model, layer)
        ys = {t: d[t].y for t in lb}
        arms = Arms(frame, scores, ys, layer, "a", "lr", XS.N_BOOT)
        hidden = d[CONTROL].x.shape[1]
        k_e = {t: arms.chosen_k(t, "emotion_pc") for t in lb}
        b = {n: {t: arms.boot(t, arms.rows(t, n)) for t in lb}
             for n in ("emotion_all", "shuffled_all", "random_all", "alpaca_all", "trainpca_all", "emotion_vectors_all")}
        b["raw"] = {t: arms.boot(t, arms.rows(t, "raw", hidden)) for t in lb}
        b["emotion_pc"] = {t: arms.boot(t, arms.rows(t, "emotion_pc", k_e[t])) for t in lb}
        b["random (k_e)"] = {t: arms.boot(t, arms.rows(t, "random", k_e[t])) for t in lb}
        b["topic_pc (k_e)"] = {t: arms.boot(t, arms.rows(t, "topic_pc", k_e[t])) for t in lb}
        for fs in ("persona_pc", "persona_pub_pc"):
            b[fs] = {t: arms.boot(t, arms.rows(t, fs, arms.chosen_k(t, fs))) for t in lb}
        for fs in ("persona_emotion", "persona_pub_emotion", "named"):
            b[fs] = {t: arms.boot(t, arms.best_by_inner(t, fs)) for t in lb}
        row = {"block": layer, "named": layer in XS.MODELS[model]["named"], "band": layer in XS.MODELS[model]["band"]}
        row |= {f"B_{n}": float(np.mean([v[t][0] for t in lb])) for n, v in b.items()}
        for other in ("raw", "shuffled_all", "random_all", "alpaca_all", "trainpca_all"):
            dd = _delta(b["emotion_all"], b[other], lb)
            row |= {f"all170_minus_{other}": dd["delta"], f"lo_{other}": dd["ci_lo"], f"hi_{other}": dd["ci_hi"]}
        per_block.append(row)
        for fs in ("raw", "emotion_pc", "emotion_all", "emotion_vectors_all", "named", "topic_pc", "shuffled_pc",
                   "shuffled_all", "random_all", "persona_pc", "persona_pub_pc"):
            transfer.append({"block": layer, "feature_set": fs, "transfer": _transfer(frame, layer, fs)})
        if layer not in XS.MODELS[model]["named"]:
            continue
        v = {"block": layer}
        d_rand, d_topic = _delta(b["emotion_pc"], b["random (k_e)"], lb), _delta(b["emotion_pc"], b["topic_pc (k_e)"], lb)
        v["P1"] = all(x["delta"] >= 0.03 and x["ci_lo"] > 0 for x in (d_rand, d_topic))
        v |= {"P1_vs_random": d_rand, "P1_vs_topic": d_topic}
        if has["IT"]:
            v["P2"] = _delta(b["emotion_pc"], b["persona_pub_pc"], [IT])["ci_lo"] > 0
            v["P2_extended"] = _delta(b["emotion_pc"], b["persona_pc"], [IT])["ci_lo"] > 0
        if has["HP"]:
            parts = {t: b["emotion_pc"][t][0] - b["random (k_e)"][t][0]
                     for t in ("harm_refusal_choice__plain", "harm_refusal_knowledge_report__plain")}
            v["P3"], v["P3_parts"] = all(x < 0.02 for x in parts.values()), parts
        v["P4"] = _delta(b["persona_pub_emotion"], b["persona_pub_pc"], lb)["delta"] >= 0.02
        v["P4_extended"] = _delta(b["persona_emotion"], b["persona_pc"], lb)["delta"] >= 0.02
        if has["IT"] and has["ID"]:
            zs = frame[(frame["protocol"] == "zero-shot") & (frame["layer"] == layer)]
            auc = {(r["feature_set"], r["test"]): r["auroc"] for _, r in zs.iterrows()}
            parts = {e: (auc.get((f"zs_{e}", IT), np.nan), auc.get((f"zs_{e}", "instruct_dishonesty__plain"), np.nan))
                     for e in ("guilty", "afraid", "desperate")}
            v["P5"], v["P5_parts"] = any(a > 0.6 and c < 0.55 for a, c in parts.values()), parts
        verdicts.append(v)
    blocks_t = pd.DataFrame(per_block)
    blocks_t.to_csv(out_dir(model) / "protocol_b_by_block.csv", index=False)
    pd.DataFrame(transfer).to_csv(out_dir(model) / "transfer_by_block.csv", index=False)
    band = blocks_t[blocks_t["band"]]
    need = (len(band) + 1) // 2
    xa1 = int((band["lo_raw"] > 0).sum())
    xa2 = int((band["lo_shuffled_all"] > 0).sum())
    held = {p: sum(bool(v.get(p)) for v in verdicts) for p in ("P1", "P2", "P3", "P4", "P5", "P2_extended",
                                                             "P4_extended") if any(p in v for v in verdicts)}
    result = {"freeze_sha256": digest, "model": model, "sets": lb, "per_named_layer": verdicts,
              "layers_holding": held, "held": {p: n >= 3 for p, n in held.items()},
              "XA1": {"band_blocks": len(band), "blocks_above_raw": xa1, "holds": xa1 >= need},
              "XA2": {"band_blocks": len(band), "blocks_above_shuffled": xa2, "holds": xa2 >= need}}
    (out_dir(model) / "predictions.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    logger.info(f"{model}: {json.dumps({k: result[k] for k in ('held', 'XA1', 'XA2')})}")
    return result


def tables() -> None:
    """The cross-model tables: protocol C transfer at the named layers beside the thesis's detectors, and the
    band curves (protocol B), with Llama's rows from the study's own runs."""
    from deception_detection.utils import REPO_ROOT

    thesis = pd.read_csv(REPO_ROOT / "results" / "xmodel" / "tables" / "transfer.csv")
    rows_ = []
    for model in XS.MODELS:
        tr = pd.read_csv(out_dir(model) / "transfer_by_block.csv")
        for layer, label in zip(XS.MODELS[model]["named"], ("L20", "L36", "L40", "L44")):
            t = tr[tr["block"] == layer].set_index("feature_set")["transfer"]
            rows_.append({"model": model, "layer": label, "block": layer} | t.to_dict())
    pd.DataFrame(rows_).to_csv(out_dir() / "transfer_named_layers.csv", index=False)
    curves = [pd.read_csv(out_dir(m) / "protocol_b_by_block.csv").assign(model=m) for m in XS.MODELS]
    pd.concat(curves).to_csv(out_dir() / "band_curves.csv", index=False)
    thesis.to_csv(out_dir() / "thesis_transfer.csv", index=False)
    logger.info(f"-> {out_dir()}")


if __name__ == "__main__":
    fire.Fire({"rows": rows, "parity": parity, "bases": bases, "run": run, "summary": summary, "tables": tables})
