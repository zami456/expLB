"""Attention probes on per-token persona coordinates, in the transfer-matrix protocol.

`persona_attention.py` keeps one persona axis v as the readout and learns only where to look. Here
the published attention probe (EleutherAI's `AttentionProbe`, ported in
`deception_detection/attention_probe.py`) takes each answer token's persona coordinates as its
input, so it can weigh personas against each other and choose tokens at once.

A token's coordinates are its projections on unit directions at the layer:

    P36  the 36 personas of the extended bank, centred on the bank mean (as persona_attention's Q36)
    A4   the four axes: extended (12v13), knowledge-denial (4v6), deception (8v7), sycophancy
    R36  36 random unit directions (seed 0), the control

Each column is standardised by Alpaca's per-token mean and SD. That is label-free and the same for
every training set; softmax ignores shifts, so it only sets the scale the weight decay acts on.

Arms (each fit per training set; weight decay chosen from the reference's grid on a stratified
held-out fifth of the training set, then refit on all of it with three seeds whose logits are
averaged): P36 and A4 with 1 and 8 heads, R36 with 8 heads, and P36 through the reference's
mean-pooled and last-token baselines. One ablation departs from the reference: the 1-head P36 probe
with its position bias removed, since a slope fitted on one dataset's answer lengths extrapolates to
much longer answers in another. Fixed reads of the extended and knowledge-denial axes (mean,
top 10%, max) sit beside them.

    python deception_detection/scripts/persona_attention_probe.py coords --layer 40
    python deception_detection/scripts/persona_attention_probe.py check --layer 40
    python deception_detection/scripts/persona_attention_probe.py run --layer 40
    python deception_detection/scripts/persona_attention_probe.py report --layer 40
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch

from deception_detection import attention_probe as ap
from deception_detection import persona_pcs as pcs
from deception_detection.log import logger
from deception_detection.scripts import persona_attention as pa
from deception_detection.scripts.persona_ood import operating_point
from deception_detection.utils import REPO_ROOT

AXES4 = ("extended", "denial", "deception", "sycophancy")
N_RANDOM = 36
RANDOM_SEED = 0
FEATURES = {"P36": slice(0, 36), "A4": slice(36, 40), "R36": slice(40, 76)}
ARMS = {  # name: (features, heads, pooling)
    "P36 attention, 1 head": ("P36", 1, "attention"),
    "P36 attention, 8 heads": ("P36", 8, "attention"),
    "A4 attention, 1 head": ("A4", 1, "attention"),
    "A4 attention, 8 heads": ("A4", 8, "attention"),
    "R36 attention, 8 heads": ("R36", 8, "attention"),
    "P36 mean-pooled": ("P36", 1, "mean"),
    "P36 last token": ("P36", 1, "last"),
    # Ablation, not in the reference: the position bias fixed at zero (every position fed as 0).
    # A slope fitted on one dataset's answer lengths extrapolates to much longer answers elsewhere.
    "P36 attention, 1 head, no position": ("P36", 1, "attention-nopos"),
}
BOOTSTRAP_ARMS = ("P36 attention, 1 head", "P36 attention, 8 heads", "A4 attention, 1 head", "A4 attention, 8 heads",
                  "P36 attention, 1 head, no position")
WEIGHT_DECAYS = (0.0, 1e-3, 1e-2, 1e-1)  # the reference driver's sweep (train_mosaic.py)
SEEDS = (0, 1, 2)
READ_AXES = {"extended": 36, "knowledge-denial": 37}  # their column in the coordinates
READS = ("mean", "top10", "max")
FPR = 0.01
N_BINS = 10


def _out(layer: int) -> Path:
    return REPO_ROOT / "results" / f"persona_attnprobe_L{layer}"


def _coords_dir(layer: int) -> Path:
    return REPO_ROOT / "activations" / f"persona_coords_L{layer}"


# --- coordinates ---------------------------------------------------------------------------------

def columns(layer: int) -> tuple[list[str], np.ndarray]:
    """The 76 unit directions [hidden, 76]: P36, then A4, then R36."""
    names36, q36 = pa.query_sets(layer)["Q36"]
    bank = pcs.load_bank_at_layer(layer, pcs.EXTENDED_VECTORS_DIR)
    a4 = np.stack([bank.axis_unit(a) for a in AXES4], axis=1).astype(np.float32)
    rng = np.random.default_rng(RANDOM_SEED)
    r36 = rng.normal(size=(q36.shape[0], N_RANDOM))
    r36 = (r36 / np.linalg.norm(r36, axis=0, keepdims=True)).astype(np.float32)
    names = [f"persona:{n}" for n in names36] + [f"axis:{a}" for a in AXES4] + [f"random:{i}" for i in range(N_RANDOM)]
    return names, np.concatenate([q36, a4, r36], axis=1)


def coords(layer: int = 40) -> None:
    """Walk the layer's activation cache once and keep each answer token's 76 coordinates."""
    import yaml

    os.environ["ACTS_CACHE_DIR"] = str(pa.CACHES[layer])
    from deception_detection.experiment import Experiment, ExperimentConfig
    from deception_detection.scripts.pool_activations import iter_cached_examples

    config = REPO_ROOT / "deception_detection" / "scripts" / "configs" / pa.CONFIGS[layer]
    cfg = ExperimentConfig(**yaml.safe_load(config.read_text(encoding="utf-8")))
    if cfg.detect_layers != [layer]:
        raise ValueError(f"{config.name} extracts layers {cfg.detect_layers}, not {layer}")
    experiment = Experiment(cfg)
    model_name = str(experiment.model.name_or_path)
    names, matrix = columns(layer)
    units = torch.from_numpy(matrix)
    out = _coords_dir(layer)
    out.mkdir(parents=True, exist_ok=True)
    for dataset in [experiment.train_dataset, *experiment.eval_datasets, *experiment.control_datasets]:
        path = out / f"{dataset.name}.npz"
        if len(dataset) == 0 or path.exists():
            continue
        chunks, offsets, labels = [], [0], []
        for example in iter_cached_examples(dataset, experiment, model_name, cfg.detect_layers, desc="coordinates"):
            tokens = example.acts[:, 0, :][example.mask].float()
            chunks.append((tokens @ units).numpy())
            offsets.append(offsets[-1] + tokens.shape[0])
            labels.append(dataset.labels[example.index].name)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, coords=np.concatenate(chunks).astype(np.float32), offsets=np.array(offsets, dtype=np.int64),
                     labels=np.array(labels), columns=np.array(names), layer=layer)
        tmp.replace(path)
        logger.info(f"{dataset.name}: {len(labels)} examples, {offsets[-1]} answer tokens")


@dataclass
class Coords:
    name: str
    coords: np.ndarray  # [tokens, 76] float32
    offsets: np.ndarray
    labels: np.ndarray

    @property
    def y(self) -> np.ndarray:
        return (self.labels == "DECEPTIVE").astype(int)

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)


def load_coords(layer: int) -> dict[str, Coords]:
    """Every dataset's coordinates, with RepE's validation split appended as persona_attention does."""
    folder, raw = _coords_dir(layer), {}
    for path in sorted(folder.glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        raw[path.stem] = (z["coords"], z["offsets"], z["labels"].astype(str))
    data = {}
    for name, (c, o, lab) in raw.items():
        if name in pa.MERGE:
            continue
        parts = [(c, o, lab)] + [raw[k] for k, v in pa.MERGE.items() if v == name and k in raw]
        lengths = np.concatenate([np.diff(p[1]) for p in parts])
        data[name] = Coords(name, np.concatenate([p[0] for p in parts]),
                            np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
                            np.concatenate([p[2] for p in parts]))
    return data


def check(layer: int = 40) -> None:
    """The axis columns must reproduce the committed per-token projections, and at layer 40 every
    column must reproduce a projection of the extracted answer-token activations."""
    from deception_detection.scripts.persona_positions import load_projections

    data = load_coords(layer)
    reference = load_projections(REPO_ROOT / "activations" / f"token_proj_L{layer}")
    for name, ref in reference.items():
        mine = data[name]
        theirs = ref.proj[ref.detect][:, [ref.axis_names.index(a) for a in AXES4]]
        if not (mine.labels == ref.labels).all() or len(theirs) != len(mine.coords):
            raise RuntimeError(f"{name}: tokens or labels differ from token_proj")
        err = np.abs(mine.coords[:, FEATURES["A4"]] - theirs).max() / theirs.std()
        logger.info(f"{name}: {len(mine.labels)} examples; axis columns vs token_proj: max error {err:.1e} of spread")
        if err > 1e-3:
            raise RuntimeError(f"{name}: axis projections disagree ({err:.2e})")
    if layer in [int(p.name[1:]) for p in pa.TOKENS_ROOT.glob("L*")]:
        _, matrix = columns(layer)
        for name, mine in data.items():
            tokens = pa.load(layer, name)
            if not (tokens.labels == mine.labels).all() or not (tokens.offsets == mine.offsets).all():
                raise RuntimeError(f"{name}: offsets or labels differ from the answer-token extract")
            theirs = tokens.project(matrix)
            err = (np.abs(mine.coords - theirs).max(0) / theirs.std(0)).max()
            logger.info(f"{name}: all 76 columns vs answer tokens: max error {err:.1e} of spread")
            if err > 1e-3:
                raise RuntimeError(f"{name}: coordinates disagree with the answer tokens ({err:.2e})")


# --- fitting -------------------------------------------------------------------------------------

def _slug(arm: str) -> str:
    return arm.replace(",", "").replace(" ", "_")


def _padded(data: dict[str, Coords], mu: np.ndarray, sd: np.ndarray) -> dict[str, tuple]:
    return {name: ap.pad_answers((d.coords - mu) / sd, d.offsets) for name, d in data.items()}


def _inputs(padded: tuple, arm: str, rows: np.ndarray | None = None) -> tuple:
    features, _, pooling = ARMS[arm]
    x, mask, position = padded
    if rows is not None:
        index = torch.from_numpy(np.flatnonzero(rows))
        x, mask, position = x[index], mask[index], position[index]
    if pooling == "attention-nopos":
        pooling, position = "attention", torch.zeros_like(position)
    return ap.pool_tokens(x[..., FEATURES[features]].contiguous(), mask, position, pooling)


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    return pa._auroc(y, s)


def _attention_profile(attn: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Mean attention mass per tenth of the answer (heads averaged), over answers of 10+ tokens,
    and the mean entropy relative to uniform attention over the same answers."""
    keep = lengths >= N_BINS
    mass = np.zeros(N_BINS)
    ratios = []
    for i in np.flatnonzero(keep):
        a = attn[i, : lengths[i]].mean(-1)
        bins = np.minimum((np.arange(lengths[i]) * N_BINS) // lengths[i], N_BINS - 1)
        mass += np.bincount(bins, weights=a, minlength=N_BINS)
        ratios.append(-(a * np.log(np.maximum(a, 1e-12))).sum() / np.log(lengths[i]))
    n = max(int(keep.sum()), 1)
    return np.concatenate([mass / n, [np.mean(ratios) if ratios else np.nan]])


def run(layer: int = 40, arms: str | None = None, max_iter: int = 2000) -> None:
    """Fit every arm on every training set, and save each fit's scores; `report` makes the tables.
    Fits already saved are skipped, so a killed run resumes."""
    out = _out(layer)
    (out / "fits").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    data = load_coords(layer)
    control = data[pa.CONTROL]
    mu = control.coords.mean(0)
    sd = control.coords.std(0) + 1e-6
    padded = _padded(data, mu, sd)
    names = list(pa.SHORT) + [pa.CONTROL]
    chosen_arms = list(ARMS) if arms is None else [a.strip() for a in arms.split(";")]
    column_names, _ = columns(layer)
    (out / "cfg.json").write_text(json.dumps({
        "layer": layer, "arms": {k: list(v) for k, v in ARMS.items()}, "weight_decays": WEIGHT_DECAYS,
        "seeds": SEEDS, "max_iter": max_iter, "train_sets": pa.TRAIN_SETS, "control": pa.CONTROL,
        "fpr": FPR, "columns": column_names, "standardisation": "per column, Alpaca token mean and SD",
        "reference": "github.com/EleutherAI/attention-probes@889b8dea",
    }, indent=1), encoding="utf-8")

    for arm in chosen_arms:
        for train in pa.TRAIN_SETS:
            path = out / "fits" / f"{_slug(arm)}__{pa.SHORT[train]}.npz"
            if path.exists():
                continue
            start = time.time()
            y = data[train].y
            held = pa._split(y)
            fit_inputs, val_inputs = _inputs(padded[train], arm, ~held), _inputs(padded[train], arm, held)
            heads = ARMS[arm][1]
            validation = {}
            for wd in WEIGHT_DECAYS:
                probe, _ = ap.train_probe(*fit_inputs, y[~held], n_heads=heads, weight_decay=wd, seed=SEEDS[0],
                                          max_iter=max_iter)
                validation[wd] = _auroc(y[held], ap.predict(probe, *val_inputs))
            chosen = max(WEIGHT_DECAYS, key=lambda wd: (round(validation[wd], 3), wd))  # ties: more decay
            saved: dict[str, np.ndarray] = {"weight_decay": np.array(chosen),
                                            "validation": np.array([validation[wd] for wd in WEIGHT_DECAYS])}
            losses, value_w, query_w, pos_w = [], [], [], []
            for seed in SEEDS:
                probe, loss = ap.train_probe(*_inputs(padded[train], arm), y, n_heads=heads, weight_decay=chosen,
                                             seed=seed, max_iter=max_iter)
                losses.append(loss)
                value_w.append(probe.v.weight.detach().numpy())
                query_w.append(probe.q.weight.detach().numpy())
                pos_w.append(probe.position_weight.detach().numpy())
                for name in names:
                    inputs = _inputs(padded[name], arm)
                    if seed == SEEDS[0] and ARMS[arm][2].startswith("attention"):
                        logits, attn = ap.predict(probe, *inputs, attention=True)
                        saved[f"attention::{name}"] = _attention_profile(attn, data[name].lengths)
                    else:
                        logits = ap.predict(probe, *inputs)
                    saved[f"score::{name}::{seed}"] = logits
            saved |= {"loss": np.array(losses), "value": np.stack(value_w), "query": np.stack(query_w),
                      "position": np.stack(pos_w)}
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                np.savez(f, **saved)
            tmp.replace(path)
            logger.info(f"{arm} on {pa.SHORT[train]}: held-out AUROC by weight decay "
                        f"{ {f'{k:g}': round(v, 3) for k, v in validation.items()} } -> {chosen:g}; "
                        f"loss {np.round(losses, 4).tolist()}; {time.time() - start:.0f}s")
    report(layer)


# --- report --------------------------------------------------------------------------------------

def _fixed_reads(d: Coords, column: int) -> dict[str, np.ndarray]:
    s = d.coords[:, column].astype(np.float64)
    starts = d.offsets[:-1]
    top = []
    for a, b in zip(d.offsets[:-1], d.offsets[1:], strict=True):
        k = max(1, int(np.ceil(0.10 * (b - a))))
        top.append(np.sort(s[a:b])[-k:].mean())
    return {"mean": np.add.reduceat(s, starts) / d.lengths, "top10": np.array(top), "max": np.maximum.reduceat(s, starts)}


def _bootstrap_mean_delta(y: np.ndarray, vectors: list[np.ndarray], ref: np.ndarray, n_boot: int = 1000,
                          seed: int = 0) -> tuple[float, float, float]:
    """Mean AUROC over `vectors` minus AUROC of `ref`, with a 95% interval from a bootstrap of the
    test rows stratified by label (as persona_ood.bootstrap_delta resamples)."""
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    point = np.mean([_auroc(y, v) for v in vectors]) - _auroc(y, ref)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate((rng.choice(positive, len(positive)), rng.choice(negative, len(negative))))
        deltas[i] = np.mean([_auroc(y[idx], v[idx]) for v in vectors]) - _auroc(y[idx], ref[idx])
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def report(layer: int = 40) -> None:
    from deception_detection.scripts.hpkr_wording import hpkr_frame
    from deception_detection.scripts.steer_lies import _clean

    out = _out(layer)
    data = load_coords(layer)
    tests = list(pa.SHORT)
    pairs = [(t, s) for t in pa.TRAIN_SETS for s in tests if s != t]

    # every model's scores: (model, train or "-") -> dataset -> scores (seed mean), plus per-seed
    models: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    per_seed: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    fits: dict[tuple[str, str], dict] = {}
    for axis, column in READ_AXES.items():
        reads = {name: _fixed_reads(d, column) for name, d in data.items()}
        for read in READS:
            models[(f"{read} read, {axis}", "-")] = {name: r[read] for name, r in reads.items()}
    for arm in ARMS:
        for train in pa.TRAIN_SETS:
            path = out / "fits" / f"{_slug(arm)}__{pa.SHORT[train]}.npz"
            if not path.exists():
                continue
            z = dict(np.load(path, allow_pickle=False))
            fits[(arm, train)] = z
            per_seed[(arm, train)] = {name: [z[f"score::{name}::{s}"] for s in SEEDS] for name in data}
            models[(arm, train)] = {name: np.mean(v, axis=0) for name, v in per_seed[(arm, train)].items()}

    # AUROC and the 1% FPR operating point on every test set
    rows = []
    for (model, train), scores in models.items():
        threshold = float(np.quantile(scores[pa.CONTROL], 1 - FPR))
        for s in tests:
            y = data[s].y
            op = operating_point(y, scores[s], threshold)
            seeds = [_auroc(y, v) for v in per_seed[(model, train)][s]] if (model, train) in per_seed else []
            rows.append({"model": model, "train": pa.SHORT.get(train, train), "test": pa.SHORT[s],
                         "auroc": _auroc(y, scores[s]), "auroc_seed_sd": np.std(seeds) if seeds else np.nan,
                         "recall": op["recall"], "specificity": op["specificity"], "balacc": op["balacc"]})
    cells = pd.DataFrame(rows)
    cells.round(4).to_csv(out / "cells.csv", index=False)

    # transfer: mean over the off-diagonal (train, test) pairs; fixed reads average the same cells
    summary = []
    for model in dict.fromkeys(m for m, _ in models):
        sub = cells[cells["model"] == model]
        if (sub["train"] == "-").all():
            vals = [sub[sub["test"] == pa.SHORT[s]].iloc[0] for _, s in pairs]
        else:
            idx = sub.set_index(["train", "test"])
            present = [(t, s) for t, s in pairs if (pa.SHORT[t], pa.SHORT[s]) in idx.index]
            if len(present) < len(pairs):
                continue
            vals = [idx.loc[(pa.SHORT[t], pa.SHORT[s])] for t, s in pairs]
        seed_means = []
        if not (sub["train"] == "-").all():
            for k in range(len(SEEDS)):
                seed_means.append(np.mean([_auroc(data[s].y, per_seed[(model, t)][s][k]) for t, s in pairs]))
        summary.append({"model": model, "transfer AUROC": np.mean([v["auroc"] for v in vals]),
                        "seed SD": np.std(seed_means) if seed_means else np.nan,
                        "transfer balanced accuracy": np.mean([v["balacc"] for v in vals]),
                        "transfer recall": np.mean([v["recall"] for v in vals])})
    raw = REPO_ROOT / "results" / f"persona_ood_ext_L{layer}"
    if (raw / "matrix_baseline.csv").exists():  # persona_ood's 8192-d LR probe, on the same pairs
        auc = pd.read_csv(raw / "matrix_baseline.csv", index_col=0)
        bal = pd.read_csv(raw / "matrix_balacc_baseline.csv", index_col=0)
        summary.append({"model": "raw 8192-d LR probe (persona_ood baseline)", "seed SD": np.nan,
                        "transfer AUROC": np.mean([auc.loc[t, s] for t, s in pairs]),
                        "transfer balanced accuracy": np.mean([bal.loc[t, s] for t, s in pairs]),
                        "transfer recall": np.nan})
    summary = pd.DataFrame(summary)
    summary.round(4).to_csv(out / "transfer_summary.csv", index=False)
    pd.set_option("display.width", 250)
    logger.info(f"transfer, mean over the {len(pairs)} off-diagonal pairs:\n{summary.round(3).to_string(index=False)}")

    off = cells[(cells["train"] != cells["test"])]
    for metric in ("auroc", "balacc"):
        table = off.groupby(["model", "test"], sort=False)[metric].mean().unstack("test")[[pa.SHORT[s] for s in tests]]
        table.round(4).to_csv(out / f"per_test_{metric}.csv")
        logger.info(f"per test set, {metric} (learned: averaged over the training sets that exclude it):\n"
                    f"{table.round(3).to_string()}")

    # paired bootstrap per test set against the best fixed reads
    boot = []
    references = {"max read, knowledge-denial": models[("max read, knowledge-denial", "-")],
                  "top10 read, extended": models[("top10 read, extended", "-")]}
    for arm in BOOTSTRAP_ARMS:
        for s in tests:
            trained = [models[(arm, t)][s] for t in pa.TRAIN_SETS if t != s and (arm, t) in models]
            if not trained:
                continue
            for ref_name, ref in references.items():
                point, lo, hi = _bootstrap_mean_delta(data[s].y, trained, ref[s])
                boot.append({"model": arm, "vs": ref_name, "test": pa.SHORT[s], "delta": point, "lo": lo, "hi": hi,
                             "significant": "gain" if lo > 0 else ("loss" if hi < 0 else "")})
    boot = pd.DataFrame(boot)
    boot.round(4).to_csv(out / "bootstrap.csv", index=False)
    if len(boot):
        logger.info(f"AUROC difference against the fixed reads, averaged over training sets (95% bootstrap):\n"
                    f"{boot.pivot_table(index=['model', 'vs'], columns='test', values='delta', sort=False).round(3).to_string()}\n"
                    f"significant: {boot[boot['significant'] != ''][['model', 'vs', 'test', 'significant']].values.tolist()}")

    # HP-KR slices and flags, models trained on HP-KR left out
    hp = "harm_refusal_knowledge_report__plain"
    frame = hpkr_frame()
    y = data[hp].y
    if not (frame["lie"].to_numpy() == y).all():
        raise RuntimeError("HP-KR rows do not line up with hpkr_frame")
    denied = frame["denied"].to_numpy()
    text = frame["messages"].map(lambda m: _clean(m[-1]["content"])).to_numpy()
    slices = {"full": np.ones(len(y), bool), "denials": denied, "identical text": denied & (text == pa.IDENTICAL)}
    hp_rows = []
    for (model, train), scores in models.items():
        if train == hp:
            continue
        flag = scores[hp] > np.quantile(scores[pa.CONTROL], 1 - FPR)
        hp_rows.append({"model": model, "train": pa.SHORT.get(train, train)}
                       | {f"AUROC {k}": _auroc(y[m], scores[hp][m]) for k, m in slices.items()}
                       | {"lies flagged": flag[y == 1].mean(), "honest denials flagged": flag[(y == 0) & denied].mean(),
                          "honest claims flagged": flag[(y == 0) & ~denied].mean()})
    hpkr = pd.DataFrame(hp_rows)
    hpkr.round(4).to_csv(out / "hpkr.csv", index=False)
    logger.info(f"HP-KR:\n{hpkr.round(3).to_string(index=False)}")

    # what the fits chose: weight decay, value weights on personas, position bias, attention profile
    column_names, _ = columns(layer)
    fit_rows, profile_rows = [], []
    for (arm, train), z in fits.items():
        features = ARMS[arm][0]
        names = column_names[FEATURES[features]]
        value = z["value"].sum(1).mean(0)  # summed over heads, averaged over seeds: [features]
        order = np.argsort(value)
        fit_rows.append({"model": arm, "train": pa.SHORT[train], "weight decay": float(z["weight_decay"]),
                         "held-out AUROC": {f"{w:g}": round(float(v), 3) for w, v in zip(WEIGHT_DECAYS, z["validation"])},
                         "loss": float(np.mean(z["loss"])), "position weight": np.round(z["position"].mean(0), 4).tolist(),
                         "top +": ", ".join(f"{names[i].split(':')[1]} {value[i]:.2f}" for i in order[::-1][:4]),
                         "top -": ", ".join(f"{names[i].split(':')[1]} {value[i]:.2f}" for i in order[:4])})
        for name in data:
            if f"attention::{name}" in z:
                profile = z[f"attention::{name}"]
                profile_rows.append({"model": arm, "train": pa.SHORT[train], "dataset": pa.SHORT.get(name, "Alpaca")}
                                    | {f"tenth {i + 1}": profile[i] for i in range(N_BINS)}
                                    | {"entropy / uniform": profile[N_BINS]})
    pd.DataFrame(fit_rows).to_csv(out / "fits.csv", index=False)
    pd.DataFrame(profile_rows).round(4).to_csv(out / "attention_profile.csv", index=False)
    logger.info(f"fits:\n{pd.DataFrame(fit_rows).drop(columns=['held-out AUROC']).to_string(index=False)}")
    _figure(summary, out, layer)


def _figure(summary: pd.DataFrame, out: Path, layer: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed = summary["model"].str.contains(" read, ")
    order = pd.concat([summary[fixed], summary[~fixed]])
    colours = np.where(order["model"].str.contains(" read, "), "#9aa4b1", "#3b6fb6")
    colours = np.where(order["model"].str.startswith("R36"), "#c9793a", colours)
    colours = np.where(order["model"].str.contains("8192-d"), "#4d545c", colours)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(order) + 1.2), sharey=True)
    for ax, metric, label in ((axes[0], "transfer AUROC", "off-diagonal AUROC"),
                              (axes[1], "transfer balanced accuracy", "off-diagonal balanced accuracy at 1% FPR")):
        ax.barh(range(len(order)), order[metric], color=colours)
        for i, v in enumerate(order[metric]):
            ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(order)), order["model"], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle(f"Attention probes on persona coordinates, layer {layer} (grey: fixed reads; dark grey: 8192-d LR; orange: random control)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "transfer.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fire.Fire({"coords": coords, "check": check, "run": run, "report": report})
