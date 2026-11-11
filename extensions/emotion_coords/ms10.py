"""Emotion coordinates on the ten datasets of Mahadik & Skapars (arXiv 2605.09391, "Do Linear Probes Generalize Better
in Persona Coordinates?"), on Llama-3.3-70B at blocks 35-45 (design: `ms10_spec.py`, frozen before any read of the new
sets). Runs in E:/aletheia/venv from the deception-detection folder; the NDIF reads are `reader/ec_read.py` with the
`ms10*.jobs` files.

The paper's matrix: per set a 75/25 train/test split of at most 800 rows, a probe trained on each train split (and on
all ten pooled) and scored on every test split. Its question, asked here of the emotion basis: does a probe in the
low-dimensional space generalise better off the diagonal than one on raw activations?

    python -m extensions.emotion_coords.ms10 sets      # the sycophancy sets downloaded; sources.json and splits.csv (no reads)
    python -m extensions.emotion_coords.ms10 freeze    # the SHA-256 of the spec, sources.json and splits.csv -> ms10_spec.FREEZE
    python -m extensions.emotion_coords.ms10 rows      # reader rows: AILiar and Roleplaying (ms10_dec), the sycophancy texts (ms10_syc)
    python -m extensions.emotion_coords.ms10 regress   # the fit path against phase3_band's protocol B at block 40
    python -m extensions.emotion_coords.ms10 cache     # each block's rows of the ten sets and Alpaca, one file per block
    python -m extensions.emotion_coords.ms10 run       # every feature set x training source x block (resumable parts)
    python -m extensions.emotion_coords.ms10 score     # checks the freeze; the matrices, regions, deltas and M1-M4
    python -m extensions.emotion_coords.ms10 report
"""

import hashlib
import json
import os
import time
from functools import cache
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from extensions.emotion_coords import aletheia170 as A
from extensions.emotion_coords import aletheia_topk as TK
from extensions.emotion_coords import config as C
from extensions.emotion_coords import linalg as la
from extensions.emotion_coords import ms10_spec as S
from extensions.emotion_coords import probes as P
from extensions.emotion_coords import vectors as V

STORE = Path("E:/emotion_coords/ms10")  # block caches and probe directions (arrays: never committed)
READS = {"dec": "ms10_dec", "syc": "ms10_syc"}  # reader rows files / read folders
REPLY_HEADER = "<|start_header_id|>assistant<|end_header_id|>"
ARMS = S.FEATURE_SETS + S.REFERENCE_SETS
HUMAN = ("validation_human", "indirectness_human", "framing_human")


def out_dir() -> Path:
    return C.results_dir("ms10")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def _key(obj) -> str:
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


# --- the ten sets ----------------------------------------------------------------------------------------------------


def _download_dir(name: str) -> Path:
    return C.data_dir("ms10", "download", name)  # one new folder per downloaded set (data only)


def _download() -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    api, out = HfApi(), {}
    for name, src in S.SOURCES.items():
        if "hf" not in src:
            continue
        revision = api.dataset_info(src["hf"]).sha
        path = hf_hub_download(src["hf"], src["file"], repo_type="dataset", revision=revision,
                               local_dir=_download_dir(name))
        out[name] = {"repo": src["hf"], "revision": revision, "file": src["file"], "sha256": _sha256(path)}
        logger.info(f"{name}: {src['hf']}@{revision[:10]}")
    return out


def _hf_frame(name: str) -> pd.DataFrame:
    path = _download_dir(name) / S.SOURCES[name]["file"]
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _rollout_items(name: str) -> list[dict]:
    """AILiar's or Roleplaying's completions exactly as the repo's `from_rollouts` builds them (prompt contents
    stripped, each completion right-stripped only: Roleplaying's completion follows a prefilled "Student:" turn,
    whose space it keeps), labelled by the dataset class's own rule; honest and deceptive only. `n` counts every
    completion, kept or not."""
    from deception_detection.data.ai_liar import AILiarDataset
    from deception_detection.data.roleplaying import RolePlayingDataset
    from deception_detection.types import Label
    from deception_detection.utils import REPO_ROOT

    cls = {"AILiarDataset": AILiarDataset, "RolePlayingDataset": RolePlayingDataset}[S.SOURCES[name]["dataset_class"]]
    data = json.loads((REPO_ROOT / S.SOURCES[name]["rollouts"]).read_text(encoding="utf-8"))
    items, n = [], 0
    for item in data["rollouts"]:
        prompt = [[m["role"], m["content"].strip()] for m in item["input_messages"]]
        if prompt[-1] != ["assistant", ""]:
            raise RuntimeError(f"{name}: a rollout's prompt does not end with an empty assistant turn")
        prompt = prompt[:-1]
        for i, reply in enumerate(item["output_strs"]):
            label = cls._get_label_from_judgement(item["graded_deception"][i])
            if label in (Label.HONEST, Label.DECEPTIVE):
                items.append({"set": name, "n": n, "label": int(label == Label.DECEPTIVE), "group": _key(prompt),
                              "messages": prompt + [["assistant", reply.rstrip()]]})
            n += 1
    return items


def _lb_items() -> dict[str, tuple[list[dict], float, int]]:
    """The three Liars' Bench sets: every row read at all band blocks (`lbdata.full_subset`), grouped by the formatted
    text before the reply, with the full set's share of lies and size."""
    from extensions.emotion_coords import lbdata as LB
    from extensions.emotion_coords.rows import _lb_rows

    keep, entries, out = LB.full_subset(), _lb_rows(), {}
    for name in S.SETS:
        lb = S.SOURCES[name].get("lb")
        if lb is None:
            continue
        entry = entries[lb]
        index = entry["index"]
        items = []
        for r in index[index["pooled_row"].isin(keep[lb])].itertuples():
            text = entry["formatted"][int(r.dataset_row)]
            items.append({"set": name, "n": int(r.pooled_row), "label": int(r.label == "DECEPTIVE"),
                          "group": _key(text[: text.rindex(REPLY_HEADER)])})
        out[name] = (items, float((index["label"] == "DECEPTIVE").mean()), len(index))
    return out


def _paired_items(name: str) -> list[dict]:
    """A prompt with a sycophantic (label 1) and a non-sycophantic reply (0): rows 2j and 2j + 1."""
    syc, non = S.SOURCES[name]["replies"]
    items = []
    for j, r in _hf_frame(name).iterrows():
        prompt = [["user", str(r["prompt"]).strip()]]
        for k, (column, label) in enumerate(((syc, 1), (non, 0))):
            items.append({"set": name, "n": 2 * j + k, "label": label, "group": _key(prompt),
                          "messages": prompt + [["assistant", str(r[column]).strip()]]})
    return items


def _oeq_items() -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """The three OEQ sets. Each distinct (prompt, reply) is one text, read once and shared by the sets that hold it;
    its stratum for the joint split is the triple of human labels (-1 where absent)."""
    texts, out = {}, {}
    for name in S.OEQ:
        df = _hf_frame(name)
        field = S.SOURCES[name]["label"]
        if not (df["label"] == df[field]).all() or set(df["label"]) != {0, 1}:
            raise RuntimeError(f"{name}: 'label' is not the 0/1 '{field}'")
        items = []
        for j, r in df.iterrows():
            prompt = [["user", str(r["prompt"]).strip()]]
            messages = prompt + [["assistant", str(r["assistant_output"]).strip()]]
            triple = tuple(-1 if pd.isna(r[f]) else int(r[f]) for f in HUMAN)
            t = texts.setdefault(_key(messages), {"text": len(texts), "group": _key(prompt), "messages": messages,
                                                  "triple": triple, "triples": set()})
            t["triples"].add(triple)
            items.append({"set": name, "n": int(j), "label": int(r["label"]), "group": t["group"], "text": t["text"],
                          "messages": messages})
        out[name] = items
    return out, texts


def _items(with_lb: bool = True) -> tuple[dict[str, list[dict]], dict]:
    """Every set's candidate rows before the cap, and the facts `sets` records (`with_lb` False: the read sets only)."""
    lb = _lb_items() if with_lb else {}
    oeq, texts = _oeq_items()
    items, facts = {}, {"lb_full": {}, "oeq_texts": len(texts),
                        "oeq_texts_with_conflicting_labels": sum(len(t["triples"]) > 1 for t in texts.values())}
    for name in S.SETS:
        if S.SOURCES[name].get("lb"):
            if not with_lb:
                continue
            items[name], share, n_full = lb[name]
            facts["lb_full"][name] = {"rows": n_full, "lie_share": share}
        elif name in oeq:
            items[name] = oeq[name]
        elif "rollouts" in S.SOURCES[name]:
            items[name] = _rollout_items(name)
        else:
            items[name] = _paired_items(name)
    return items, facts | {"texts": texts}


def _test_mask(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """The test rows: fold 0 of a seeded StratifiedGroupKFold."""
    from sklearn.model_selection import StratifiedGroupKFold

    folds = StratifiedGroupKFold(S.N_SPLITS, shuffle=True, random_state=S.SEED)
    _, test = next(iter(folds.split(np.zeros(len(y)), y, groups)))
    mask = np.zeros(len(y), dtype=bool)
    mask[test] = True
    return mask


def _capped(name: str, items: list[dict], facts: dict) -> list[dict]:
    """At most CAP rows: Liars' Bench at the full set's share of lies, Sycophancy as CAP / 2 prompts with both replies."""
    rng = np.random.default_rng([S.SEED, S.SETS.index(name)])
    if name in facts["lb_full"] and facts["lb_full"][name]["rows"] > S.CAP:
        n1 = int(round(S.CAP * facts["lb_full"][name]["lie_share"]))
        pos = [it for it in items if it["label"] == 1]
        neg = [it for it in items if it["label"] == 0]
        pick = ([pos[i] for i in rng.choice(len(pos), n1, replace=False)]
                + [neg[i] for i in rng.choice(len(neg), S.CAP - n1, replace=False)])
        return sorted(pick, key=lambda it: it["n"])
    if len(items) > S.CAP:
        if "replies" not in S.SOURCES[name]:
            raise RuntimeError(f"{name}: {len(items)} rows and no cap rule")
        prompts = set(rng.choice(len(items) // 2, S.CAP // 2, replace=False).tolist())
        return [it for it in items if it["n"] // 2 in prompts]
    return items


def sets() -> None:
    """Download the sycophancy sets, build every set's rows, cap them and split them; write sources.json and
    splits.csv (row order = each set's `n`). No activation is read."""
    from deception_detection.utils import REPO_ROOT
    from extensions.emotion_coords.rows import DATASET_CODES

    hf = _download()
    items, facts = _items()
    texts = facts.pop("texts")
    records = []
    for name in S.SETS:
        kept = _capped(name, items[name], facts)
        if name in S.OEQ:
            continue  # split jointly below
        y = np.array([it["label"] for it in kept])
        test = _test_mask(y, np.array([it["group"] for it in kept]))
        lb = S.SOURCES[name].get("lb")
        for it, t in zip(kept, test):
            read_id = DATASET_CODES[lb] * 1_000_000 + it["n"] if lb else S.CODES[name] * 1_000_000 + it["n"]
            records.append({"set": name, "n": it["n"], "read_id": read_id, "source": "lb" if lb else "read",
                            "label": it["label"], "group": it["group"], "split": "test" if t else "train"})
    distinct = sorted(texts.values(), key=lambda t: t["text"])  # the OEQ texts, split once over all three sets
    strata = np.array(["".join(str(v) for v in t["triple"]) for t in distinct])
    test = _test_mask(strata, np.array([t["group"] for t in distinct]))
    split = {t["text"]: "test" if m else "train" for t, m in zip(distinct, test)}
    for name in S.OEQ:
        for it in items[name]:
            records.append({"set": name, "n": it["n"], "read_id": S.CODES["OEQ"] * 1_000_000 + it["text"],
                            "source": "read", "label": it["label"], "group": it["group"], "split": split[it["text"]]})
    table = pd.DataFrame(records)
    crossing = table.groupby(["set", "group"])["split"].nunique().max()
    oeq_cross = table[table["set"].isin(S.OEQ)].groupby("read_id")["split"].nunique().max()
    if crossing > 1 or oeq_cross > 1:
        raise RuntimeError("a group or an OEQ text is on both sides of a split")
    table.to_csv(out_dir() / "splits.csv", index=False)
    rollouts = {n: {"path": S.SOURCES[n]["rollouts"], "sha256": _sha256(REPO_ROOT / S.SOURCES[n]["rollouts"])}
                for n in S.SETS if "rollouts" in S.SOURCES[n]}
    index = {lb: _sha256(C.data_dir("index") / f"{lb}.csv") for lb in (S.SOURCES[n].get("lb") for n in S.SETS) if lb}
    sources = {"hf": hf, "rollouts": rollouts, "lb_index_sha256": index} | facts
    (out_dir() / "sources.json").write_text(json.dumps(sources, indent=1), encoding="utf-8")
    summary = _set_summary(table)
    summary.to_csv(out_dir() / "sets.csv", index=False)
    logger.info("\n" + summary.to_string(index=False))


def _set_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in S.SETS:
        t = table[table["set"] == name]
        rows.append({"set": name, "rows": len(t), "label 1": int(t["label"].sum()), "label 0": int((1 - t["label"]).sum()),
                     "groups": t["group"].nunique(), "train": int((t["split"] == "train").sum()),
                     "test": int((t["split"] == "test").sum()),
                     "test label 1": int(t.loc[t["split"] == "test", "label"].sum())})
    return pd.DataFrame(rows)


@cache
def _splits() -> pd.DataFrame:
    return pd.read_csv(out_dir() / "splits.csv", dtype={"group": str})


def freeze() -> str:
    """Record the SHA-256 of the spec, sources.json and splits.csv before any read of the new sets."""
    files = (Path(S.__file__), out_dir() / "sources.json", out_dir() / "splits.csv")
    lines = [f"{p.name} sha256 {_sha256(p)}" for p in files]
    if S.FREEZE.exists():
        held = S.FREEZE.read_text(encoding="utf-8")
        if not all(line in held for line in lines):
            raise RuntimeError(f"{S.FREEZE} holds another design:\n{held}")
        return held
    S.FREEZE.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines + [f"frozen {time.strftime('%Y-%m-%d %H:%M:%S')}"]) + "\n"
    S.FREEZE.write_text(text, encoding="utf-8")
    logger.info(f"-> {S.FREEZE}\n{text}")
    return text


def check_freeze() -> str:
    held = S.FREEZE.read_text(encoding="utf-8") if S.FREEZE.exists() else ""
    for p in (Path(S.__file__), out_dir() / "sources.json", out_dir() / "splits.csv"):
        if _sha256(p) not in held:
            raise RuntimeError(f"{p.name} ({_sha256(p)[:12]}) is not the frozen design ({S.FREEZE})")
    digest = _sha256(Path(S.__file__))
    logger.info(f"ms10_spec.py sha256 {digest} = the freeze")
    return digest


# --- reader rows -----------------------------------------------------------------------------------------------------


def _tokenized(dialogues: list[list], tok) -> list[tuple[list[int], int, int]]:
    """Token ids of each dialogue and its reply's span [a, b), through the repo's tokeniser path (mask padding 0/0)."""
    from deception_detection.tokenized_data import TokenizedDataset
    from deception_detection.types import Message

    out = []
    for i in range(0, len(dialogues), 32):
        chunk = [[Message(role, content, k == len(d) - 1) for k, (role, content) in enumerate(d)]
                 for d in dialogues[i: i + 32]]
        toks = TokenizedDataset.from_dialogue_list(chunk, tok, padding={"llama": S.MASK_PADDING})
        for j in range(len(chunk)):
            keep = toks.attention_mask[j].numpy().astype(bool)
            ids = toks.tokens[j].numpy()[keep]
            span = np.flatnonzero(toks.detection_mask[j].numpy()[keep])
            if not len(span) or span[-1] - span[0] + 1 != len(span):
                raise RuntimeError(f"dialogue {i + j}: the reply's span is empty or not contiguous")
            out.append(([int(t) for t in ids], int(span[0]), int(span[-1]) + 1))
    return out


def rows() -> None:
    """The new sets' rows for the reader: AILiar and Roleplaying (`ms10_dec`), the sycophancy texts (`ms10_syc`, each
    OEQ text once). Every span must decode to its reply exactly."""
    from extensions.emotion_coords import rows as R

    table = _splits()
    items, _ = _items(with_lb=False)
    tok = R._tokenizer()
    for read, names in (("dec", ("AILiar", "Roleplaying")),
                        ("syc", ("Sycophancy", "OpenEndedSycophancy") + S.OEQ)):
        wanted, out = {}, []
        for name in names:
            t = table[table["set"] == name]
            by_n = {it["n"]: it for it in items[name]}
            for r in t.itertuples():
                it = by_n[int(r.n)]
                if int(r.read_id) not in wanted:
                    wanted[int(r.read_id)] = {"set": "OEQ" if name in S.OEQ else name,
                                              "n": it["text"] if name in S.OEQ else int(r.n),
                                              "label": None if name in S.OEQ else int(r.label),
                                              "messages": it["messages"]}
        ids = sorted(wanted)
        tokenized = _tokenized([wanted[i]["messages"] for i in ids], tok)
        bad = 0
        for i, (token_ids, a, b) in zip(ids, tokenized):
            w = wanted[i]
            if tok.decode(token_ids[a:b], clean_up_tokenization_spaces=False) != w["messages"][-1][1]:
                bad += 1
            out.append({"row_id": i, "ids": token_ids, "groups": [[[a, b]]], "points": [], "set": w["set"],
                        "n": w["n"]} | ({"label": w["label"]} if w["label"] is not None else {}))
        if bad:
            raise RuntimeError(f"{READS[read]}: {bad} spans do not decode to their reply")
        R.write_rows(READS[read], out, {"sets": list(names), "spans_decode_exactly": True})


# --- each block's data -----------------------------------------------------------------------------------------------


def _block_path(block: int) -> Path:
    return STORE / "data" / f"B{block}.npz"


def cache(blocks=None) -> None:
    """Each block's rows of the ten sets in splits.csv's order (Liars' Bench from the study's pooled reads, the others
    from `ms10_*`) and the Alpaca replies, float32, one file per block."""
    from extensions.emotion_coords import lbdata as LB

    table = _splits()
    subset = LB.full_subset()
    for b in S.BLOCKS if blocks is None else C.int_list(blocks):
        path = _block_path(b)
        if path.exists():
            continue
        lb = LB.load(b, "a", subset)
        arrays = {"alpaca": lb[C.load()["control"]].x.astype(np.float32)}
        reads = {}
        for name in READS.values():
            ids, content, _ = V.load_reads(name, V._layer_index(b, name))
            reads |= {int(r): content[i, 0, 0] for i, r in enumerate(ids)}
        for name in S.SETS:
            t = table[table["set"] == name]
            source = S.SOURCES[name].get("lb")
            if source is not None:
                where = {int(r): i for i, r in enumerate(lb[source].rows)}
                x = lb[source].x[[where[int(n)] for n in t["n"]]]
            else:
                missing = [int(r) for r in t["read_id"] if int(r) not in reads]
                if missing:
                    raise RuntimeError(f"B{b} {name}: {len(missing)} rows not read")
                x = np.stack([reads[int(r)] for r in t["read_id"]])
            if not np.isfinite(x).all():
                raise RuntimeError(f"B{b} {name}: non-finite reads")
            arrays[f"x_{name}"] = x.astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        logger.info(f"B{b}: {sum(len(v) for k, v in arrays.items() if k != 'alpaca')} rows, "
                    f"{len(arrays['alpaca'])} Alpaca -> {path}")


def _block(block: int) -> dict[str, np.ndarray]:
    with np.load(_block_path(block)) as z:
        return {k: z[k] for k in z.files}


# --- probes ----------------------------------------------------------------------------------------------------------


def _feature_names() -> list[tuple[str, int]]:
    names = [("raw", 0), ("emotion_all", 0)]  # the heavy arm first
    names += [("shuffled_all", p) for p in range(S.N_SHUFFLED)] + [("random_all", s) for s in range(S.N_RANDOM)]
    return names + [("alpaca_all", 0), ("trainpca_all", 0)] + [(fs, 0) for fs in S.REFERENCE_SETS]


def _basis(block: int, fs: str, seed: int, alpaca: np.ndarray) -> np.ndarray | None:
    """The feature set's rows (orthonormal); None for raw activations and the per-source training-set PCs."""
    if fs in ("raw", "trainpca_all"):
        return None
    if fs.startswith("persona"):
        from deception_detection import persona_pcs as pcs

        bank = pcs.load_bank_at_layer(block, pcs.VECTORS_DIR)
        if fs == "persona_dec3":
            return pcs.fit_geometry(bank, k=S.K_REFERENCE).components
        return pcs.fit_geometry(bank, k=S.K_REFERENCE, deception=pcs.SYCOPHANCY_PERSONAS,
                                honesty=pcs.ANTI_SYCOPHANCY_PERSONAS, axis_name="sycophancy").components
    with np.load(C.data_dir("bases") / f"full_L{block}.npz") as z:
        e = z["E_components"]
        shuffled = z[f"S{seed}_components"] if fs == "shuffled_all" else None
    r = len(e)
    if fs == "emotion_all":
        return e
    if fs == "emotion_pc3":
        return e[: S.K_REFERENCE]
    if fs == "shuffled_all":
        return shuffled[:r]
    if fs == "random_all":
        return la.random_subspace(alpaca.shape[1], r, seed=S.RANDOM_SEED + seed)
    if fs == "alpaca_all":
        return la.top_pcs(alpaca, r)
    raise ValueError(fs)


def _rank(block: int) -> int:
    with np.load(C.data_dir("bases") / f"full_L{block}.npz") as z:
        return len(z["E_components"])


def _training(data: dict, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A source's training rows: one set's train split, or all ten pooled ("all"); groups carry the set's name."""
    table = _splits()
    xs, ys, gs = [], [], []
    for name in S.SETS if source == "all" else (source,):
        t = table[table["set"] == name]
        train = (t["split"] == "train").to_numpy()
        xs.append(data[f"x_{name}"][train])
        ys.append(t["label"].to_numpy()[train])
        gs.append((name + ":" + t["group"]).to_numpy()[train])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)


def _inner_folds(y: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    return list(StratifiedGroupKFold(S.INNER_FOLDS, shuffle=True, random_state=S.SEED).split(np.zeros(len(y)), y, groups))


def _fit_direction(x: np.ndarray, y: np.ndarray, folds, basis: np.ndarray | None) -> tuple[np.ndarray, float, float, float]:
    """C by inner CV, a fit on every training row, folded into one direction and offset over the raw activations."""
    feats = x if basis is None else x @ basis.T
    sel = P.best_per_k(P.select("lr", feats, y, folds, None, S.C_GRID))
    chosen = sel[sel["chosen"]].iloc[0]
    v, offset = A._direction(P.fit("lr", feats, y, float(chosen["C"])), basis)
    return v, offset, float(chosen["C"]), float(chosen["inner_auroc"])


def _part(block: int, fs: str, seed: int) -> Path:
    return STORE / "probes" / f"B{block}_{fs}_{seed}.npz"


def _unit(block: int, fs: str, seed: int) -> tuple[str, float]:
    """One feature set at one block: a probe per training source (the ten train splits, then all ten)."""
    t0 = time.time()
    data = _block(block)
    basis = _basis(block, fs, seed, data["alpaca"])
    r = _rank(block)
    out = {k: [] for k in ("v", "offset", "C", "inner_auroc", "n")}
    for source in S.TRAIN_SOURCES:
        x, y, groups = _training(data, source)
        b = la.top_pcs(x, min(r, len(y) - 1)) if fs == "trainpca_all" else basis
        v, offset, c, inner = _fit_direction(x, y, _inner_folds(y, groups), b)
        for k, value in zip(out, (v, offset, c, inner, len(y))):
            out[k].append(value)
    path = _part(block, fs, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.array(v) for k, v in out.items()}, sources=np.array(S.TRAIN_SOURCES))
    return path.name, time.time() - t0


def _names(x) -> list[str]:
    """A list from the command line: fire turns "a,b" into a tuple and "a" into a string."""
    return [str(v) for v in x] if isinstance(x, (list, tuple)) else str(x).split(",")


def run(workers: int = 4, threads: int = 3, blocks=None, only=None, skip=None) -> None:
    """Every (block, feature set, seed) unit not yet saved, raw activations first; then probes.csv. `only` / `skip`:
    comma lists of feature sets, so the 8,192-d raw arm and the small subspaces can run with their own worker and
    thread counts (raw at phase 3's 3 threads: its weakly regularised fits depend on the BLAS thread count)."""
    from joblib import Parallel, delayed

    from extensions.emotion_coords import ms10 as importable  # under `-m` this file is __main__, which the workers cannot unpickle

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)  # inherited by the worker processes
    todo_blocks = S.BLOCKS if blocks is None else tuple(C.int_list(blocks))
    for b in todo_blocks:
        if not _block_path(b).exists():
            raise FileNotFoundError(f"{_block_path(b)}: run `cache` first")
    names = [(fs, seed) for fs, seed in _feature_names()
             if (only is None or fs in _names(only)) and (skip is None or fs not in _names(skip))]
    units = [(b, fs, seed) for fs, seed in names for b in todo_blocks]
    todo = [u for u in units if not _part(*u).exists()]
    logger.info(f"{len(units)} units, {len(todo)} to fit on {workers} workers x {threads} threads")
    tmp = STORE / "joblib_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, (name, seconds) in enumerate(Parallel(n_jobs=workers, backend="loky", temp_folder=str(tmp),
                                                 return_as="generator_unordered")(delayed(importable._unit)(*u)
                                                                                  for u in todo)):
        logger.info(f"{k + 1}/{len(todo)} {name}: {seconds:.0f} s ({(time.time() - t0) / 60:.0f} min)")
    rows = []
    for b, fs, seed in [(b, fs, seed) for fs, seed in _feature_names() for b in S.BLOCKS]:
        if not _part(b, fs, seed).exists():
            continue
        with np.load(_part(b, fs, seed)) as z:
            rows += [{"block": b, "feature_set": fs, "seed": seed, "train": str(s), "C": float(c),
                      "inner_auroc": float(i), "n_train": int(n)}
                     for s, c, i, n in zip(z["sources"], z["C"], z["inner_auroc"], z["n"])]
    pd.DataFrame(rows).to_csv(out_dir() / "probes.csv", index=False)


def regress(block: int = 40) -> None:
    """This fit path on phase 3's protocol B (leave one Liars' Bench set out, the capped rows, C by leave-one-set-out
    CV) must give phase3_band's stored AUROCs for all-170 and raw activations."""
    from extensions.emotion_coords import lbdata as LB

    lb5 = LB.lb5()
    data = LB.load(block, "a", LB.full_subset())
    with np.load(C.data_dir("bases") / f"full_L{block}.npz") as z:
        e = z["E_components"]
    stored = pd.read_csv(C.results_dir("phase3_band") / "probes.csv")
    stored = stored[(stored["protocol"] == "B") & (stored["layer"] == block) & (stored["probe"] == "lr")
                    & stored["chosen"].astype(bool)]
    rows = []
    for fs, basis in (("emotion_all", e), ("raw", None)):
        for held in lb5:
            train = [n for n in lb5 if n != held]
            x = np.concatenate([data[n].x for n in train])
            y = np.concatenate([data[n].y for n in train])
            sets = np.concatenate([np.full(len(data[n].y), i) for i, n in enumerate(train)])
            v, offset, c, _ = _fit_direction(x, y, P._folds("sets", y, sets), basis)
            ref = stored[(stored["feature_set"] == fs) & (stored["test"] == held)]
            rows.append({"feature_set": fs, "held_out": LB.short(held), "auroc": P.auroc(data[held].y, data[held].x @ v + offset),
                         "C": c, "stored_auroc": ref["auroc"].tolist(), "stored_C": ref["C"].tolist()})
    table = pd.DataFrame(rows)
    table["match"] = [any(abs(a - s) < 1e-6 for s in refs) for a, refs in zip(table["auroc"], table["stored_auroc"])]
    table.to_csv(out_dir() / f"regression_B{block}.csv", index=False)
    logger.info("\n" + table.to_string(index=False))


# --- scoring ---------------------------------------------------------------------------------------------------------


def _regions() -> dict[str, list[tuple[str, str]]]:
    dec, syc = S.DECEPTION, S.SYCOPHANCY
    return {"deception": [(a, b) for a in dec for b in dec if a != b],
            "sycophancy": [(a, b) for a in syc for b in syc if a != b],
            "cross": [(a, b) for a in dec for b in syc] + [(a, b) for a in syc for b in dec],
            "diagonal": [(a, a) for a in S.SETS],
            "all": [("all", b) for b in S.SETS]}


def _seeds(fs: str) -> list[int]:
    return list(range({"shuffled_all": S.N_SHUFFLED, "random_all": S.N_RANDOM}.get(fs, 1)))


def score() -> None:
    """Every probe on every test split at every block, with shared stratified bootstrap draws per test split; region
    means per draw; all-170 against every arm and every arm against raw activations (paired); M1-M4."""
    digest = check_freeze()
    table = _splits()
    tests = {}
    for i, name in enumerate(S.SETS):
        t = table[(table["set"] == name)]
        y = t.loc[t["split"] == "test", "label"].to_numpy()
        rng = np.random.default_rng([S.BOOT_SEED, i])
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        draws = [np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg)))) for _ in range(S.N_BOOT)]
        tests[name] = (y, pos, neg, *TK._counts(draws, pos, neg), (t["split"] == "test").to_numpy())
    regions = _regions()
    records, draws_of = [], {}
    for b in S.BLOCKS:
        data = _block(b)
        xt = {name: data[f"x_{name}"][tests[name][5]] for name in S.SETS}
        for fs in ARMS:
            per_seed = []
            for seed in _seeds(fs):
                with np.load(_part(b, fs, seed)) as z:
                    v, offset, sources = z["v"], z["offset"], [str(s) for s in z["sources"]]
                cells = {}
                for j, src in enumerate(sources):
                    for name in S.SETS:
                        y, pos, neg, up, un, _ = tests[name]
                        s = xt[name] @ v[j] + offset[j]
                        cells[src, name] = TK._boot_aurocs(up, un, s, pos, neg)
                        records.append({"block": b, "feature_set": fs, "seed": seed, "train": src, "test": name,
                                        "auroc": P.auroc(y, s)})
                per_seed.append({r: np.mean([cells[c] for c in keys], axis=0) for r, keys in regions.items()})
            for r in regions:
                draws_of[b, fs, r] = np.mean([d[r] for d in per_seed], axis=0)
        logger.info(f"B{b}: scored")
    auroc = pd.DataFrame(records)
    auroc.to_csv(out_dir() / "auroc.csv", index=False)
    _summarise(auroc, draws_of, digest)


def _cell_means(auroc: pd.DataFrame) -> pd.DataFrame:
    """Per block, arm, training source and test set: the AUROC averaged over the arm's seeds."""
    return auroc.groupby(["block", "feature_set", "train", "test"], as_index=False)["auroc"].mean()


def _summarise(auroc: pd.DataFrame, draws_of: dict, digest: str) -> None:
    cells = _cell_means(auroc).set_index(["block", "feature_set", "train", "test"])["auroc"]
    rows = []
    for b in S.BLOCKS:
        for fs in ARMS:
            for r, keys in _regions().items():
                d = draws_of[b, fs, r]
                rows.append({"block": b, "arm": fs, "region": r, "mean": float(np.mean([cells[b, fs, s, t] for s, t in keys])),
                             "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))})
    regions = pd.DataFrame(rows)
    regions.to_csv(out_dir() / "regions.csv", index=False)
    point = regions.set_index(["block", "arm", "region"])["mean"]
    deltas = []
    for b in S.BLOCKS:
        for r in _regions():
            for a, c in [("emotion_all", o) for o in ARMS if o != "emotion_all"] + [(o, "raw") for o in ARMS if o not in ("raw", "emotion_all")]:
                diff = draws_of[b, a, r] - draws_of[b, c, r]
                deltas.append({"block": b, "region": r, "arm": a, "minus": c, "delta": point[b, a, r] - point[b, c, r],
                               "lo": float(np.quantile(diff, 0.025)), "hi": float(np.quantile(diff, 0.975))})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out_dir() / "deltas.csv", index=False)

    def blocks_above(region: str, minus: str) -> int:
        d = deltas[(deltas["region"] == region) & (deltas["arm"] == "emotion_all") & (deltas["minus"] == minus)]
        return int((d["lo"] > 0).sum())

    verdicts = {"freeze_sha256": digest, "blocks": list(S.BLOCKS), "min_blocks": S.MIN_BLOCKS}
    for raw_key, ctrl_key, region in (("M1", "M2", "deception"), ("M3", "M4", "sycophancy")):
        n_raw = blocks_above(region, "raw")
        verdicts[raw_key] = {"region": region, "blocks_with_ci_above_0": n_raw, "holds": n_raw >= S.MIN_BLOCKS}
        per = {c: blocks_above(region, c) for c in S.CONTROLS}
        verdicts[ctrl_key] = {"region": region, "blocks_with_ci_above_0": per,
                              "holds": all(n >= S.MIN_BLOCKS for n in per.values())}
    verdicts["reference_minus_raw_blocks_with_ci_above_0"] = {
        r: {a: int(((deltas["region"] == r) & (deltas["arm"] == a) & (deltas["minus"] == "raw") & (deltas["lo"] > 0)).sum())
            for a in S.REFERENCE_SETS} for r in _regions()}
    (out_dir() / "verdicts.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    band = _cell_means(auroc).groupby(["feature_set", "train", "test"], as_index=False)["auroc"].mean()
    at40 = _cell_means(auroc).query("block == 40").drop(columns="block")
    pd.concat([band.assign(where="band mean"), at40.assign(where="block 40")]).to_csv(out_dir() / "matrices.csv", index=False)
    logger.info(json.dumps(verdicts, indent=1))


# --- report ----------------------------------------------------------------------------------------------------------

SHORT = {"emotion_all": "all-170", "raw": "raw", "shuffled_all": "shuffled", "random_all": "random",
         "alpaca_all": "Alpaca PCs", "trainpca_all": "train PCs", "emotion_pc3": "emotion PCs (3)",
         "persona_dec3": "persona dec (3)", "persona_syc3": "persona syc (3)"}
SET_SHORT = {"AILiar": "AIL", "Roleplaying": "RP", "ConvGame": "CG", "InstrDec": "ID", "HarmPressureChoice": "HPC",
             "Sycophancy": "Syc", "OpenEndedSycophancy": "OES", "OEQValidation": "OEQ-V", "OEQIndirectness": "OEQ-I",
             "OEQFraming": "OEQ-F", "all": "all ten"}


def report() -> Path:
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    sets_table = pd.read_csv(out_dir() / "sets.csv")
    regions = pd.read_csv(out_dir() / "regions.csv").set_index(["block", "arm", "region"])
    deltas = pd.read_csv(out_dir() / "deltas.csv")
    verdicts = json.loads((out_dir() / "verdicts.json").read_text(encoding="utf-8"))
    matrices = pd.read_csv(out_dir() / "matrices.csv")
    lines = ["# Emotion coordinates on Mahadik & Skapars' ten datasets (Llama-3.3-70B, blocks 35-45)", "",
             f"Design: `emotion_coords/ms10_spec.py`, frozen (sha256 {verdicts['freeze_sha256'][:12]}) with its splits "
             "before any read of the new sets. Per set a seeded stratified-group 75/25 split of at most 800 rows; a "
             "probe per train split and one on all ten, scored on every test split; paired bootstrap over the test rows.",
             "", "| set | rows | label 1 / 0 | train | test (label 1) |", "|---|---|---|---|---|"]
    for _, r in sets_table.iterrows():
        lines.append(f"| {r['set']} | {r['rows']} | {r['label 1']} / {r['label 0']} | {r['train']} | {r['test']} "
                     f"({r['test label 1']}) |")
    lines += ["", "## Predictions", ""]
    for key in ("M1", "M2", "M3", "M4"):
        v = verdicts[key]
        lines.append(f"- **{key}** ({S.PREDICTIONS[key]}): {json.dumps(v['blocks_with_ci_above_0'])} of 11: "
                     f"**{'holds' if v['holds'] else 'fails'}**.")
    for region in ("deception", "sycophancy", "cross", "diagonal", "all"):
        lines += ["", f"## Region: {region} ({S.REGIONS[region]})", "",
                  "| block | " + " | ".join(SHORT[a] for a in ARMS) + " |", "|---|" + "---|" * len(ARMS)]
        for b in S.BLOCKS:
            e = regions.loc[(b, "emotion_all", region)]
            cells = [f"{f3(e['mean'])} [{f3(e['lo'])}, {f3(e['hi'])}]"] + [f3(regions.loc[(b, a, region), "mean"]) for a in ARMS[1:]]
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
        d = deltas[(deltas["region"] == region) & (deltas["arm"] == "emotion_all")]
        lines += ["", "all-170 minus each arm (paired 95% CI):", "",
                  "| block | " + " | ".join(SHORT[a] for a in ARMS[1:]) + " |", "|---|" + "---|" * len(ARMS[1:])]
        for b in S.BLOCKS:
            dd = d[d["block"] == b].set_index("minus")
            lines.append(f"| {b} | " + " | ".join(f"{dd.loc[a, 'delta']:+.3f} [{dd.loc[a, 'lo']:+.3f}, {dd.loc[a, 'hi']:+.3f}]"
                                                  for a in ARMS[1:]) + " |")
    lines += ["", "## The paper's claim at 70B: each k = 3 arm minus raw activations, blocks with the CI above 0", "",
              "| region | " + " | ".join(SHORT[a] for a in S.REFERENCE_SETS) + " |", "|---|" + "---|" * len(S.REFERENCE_SETS)]
    for region, per in verdicts["reference_minus_raw_blocks_with_ci_above_0"].items():
        lines.append(f"| {region} | " + " | ".join(f"{per[a]} of 11" for a in S.REFERENCE_SETS) + " |")
    for where in ("block 40", "band mean"):
        for arm in ("emotion_all", "raw", "persona_dec3"):
            m = matrices[(matrices["where"] == where) & (matrices["feature_set"] == arm)]
            grid = m.pivot(index="train", columns="test", values="auroc").reindex(index=list(S.TRAIN_SOURCES),
                                                                                columns=list(S.SETS))
            lines += ["", f"## Matrix: {SHORT[arm]}, {where} (rows train, columns test)", "",
                      "| train | " + " | ".join(SET_SHORT[s] for s in S.SETS) + " |", "|---|" + "---|" * len(S.SETS)]
            for src in S.TRAIN_SOURCES:
                lines.append(f"| {SET_SHORT[src]} | " + " | ".join(f3(grid.loc[src, t]) for t in S.SETS) + " |")
    path = out_dir() / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"-> {path}")
    return path


if __name__ == "__main__":
    fire.Fire({"sets": sets, "freeze": freeze, "rows": rows, "regress": regress, "cache": cache, "run": run,
               "score": score, "report": report})
