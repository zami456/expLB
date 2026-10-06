"""The base probe trained on Llama-3.3-70B's own lies from AISI's rollouts, against RepE and the personas.

The paper's base probe learns from RepE. There, 512 true statements appear in both classes, and
only the instruction to be honest or dishonest differs. This trains the same probe (per-token
logistic regression, reg 10, standardised; only the training data changes) on AISI's Llama rollouts
instead. These are Llama's own belief-checked lies about trivia and multiple-choice questions;
see `deception_detection/data/aisi_rollouts.py`.

    python deception_detection/scripts/aisi_probe.py audit                 # before anything trains
    python deception_detection/scripts/aisi_probe.py extract --layer 20    # NDIF; resumable
    python deception_detection/scripts/aisi_probe.py fit --layer 20 [--variant matched|varied]
    python deception_detection/scripts/aisi_probe.py compare

`audit` writes the exclusion list the loader refuses to run without. It lists every AISI row whose
question also appears in a Liars' Bench dataset we evaluate on.

The runner (`experiment.py run` on `configs/aisi_llama_L*.yaml`) is not used to train or score.
The configs define the recipe, and `fit` builds its training rows from them. But on a 16 GB
machine the runner's in-memory fit cannot hold these rows, so they are fetched in chunks, only the
response tokens (`extract`), and fitted by streaming (`fit`). `compare` then scores every probe on
the pooled Liars' Bench activations, the same path the paper's zero-shot table takes.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np

from deception_detection.data import aisi_rollouts as A
from deception_detection.log import logger
from deception_detection.utils import REPO_ROOT

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
LAYERS = (20, 40, 44)
POOLED = {20: REPO_ROOT / "activations" / "pooled", 40: REPO_ROOT / "activations" / "pooled_L40",
          44: REPO_ROOT / "activations" / "pooled_L44"}
# where each layer's runs cached their activations (the RepE runs' caches, reused)
CACHE_DIRS = {20: str(REPO_ROOT / "activations" / "cache"), 40: "E:/liars-bench-acts/cache",
              44: "E:/liars-bench-acts/cache"}
REPE_RUNS = {20: "XXXX-9-llama-all", 40: "XXXX-9-llama-all-L40", 44: "XXXX-9-llama-all-L44"}
# training variant -> (run folder, arm label); the ablations reuse the pooled run's cached activations
AISI_ARMS = {
    "pooled": ("aisi-llama-L{layer}", "base probe (AISI-trained)"),
    "matched": ("aisi-llama-matched-L{layer}", "base probe (AISI, instruction-matched)"),
    "varied": ("aisi-llama-varied-L{layer}", "base probe (AISI, trivia only)"),
}
GS_DIR = REPO_ROOT / "activations" / "gender_secret"
CONFIGS = REPO_ROOT / "deception_detection" / "scripts" / "configs"
# the held-out split, one span-mean vector per row per layer (written by `extract`)
VALIDATION_MEANS = REPO_ROOT / "activations" / "aisi_llama"
RESULTS = REPO_ROOT / "results" / "aisi_probe"

# what the probes are scored on: the five Liars' Bench eval sets, and Gender-Secret
AUDITED = (
    "convince_me__plain",
    "harm_refusal_choice__plain",
    "harm_refusal_knowledge_report__plain",
    "insider_trading_cadenza__plain",
    "instruct_dishonesty__plain",
    "gender_secret__plain",
)


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text.lower()).split())


def shingles(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def audit(out: str | Path | None = None, min_words: int = 3, n: int = 6,
          near: float = 0.8, loose: float = 0.5) -> None:
    """Find AISI rows whose question appears in a Liars' Bench dataset, and exclude them.

    Questions are normalised (lowercase, punctuation and spacing dropped). For multiple-choice
    rows the question is the stem alone, without the options. A row is excluded when:

    - its question appears verbatim inside any message of an audited dataset; or
    - at least `near` of its `n`-word shingles appear there (a lightly reworded copy).

    Rows sharing at least `loose` of their shingles are counted but kept. At that level, what
    they share is template wording such as "which of the following is a". Questions under
    `min_words` words are too generic to match and are counted as unchecked.
    """
    from deception_detection.repository import DatasetRepository

    out = A.EXCLUSIONS if out is None else Path(out)
    repository = DatasetRepository()
    corpora, corpus_shingles = {}, {}
    for name in AUDITED:
        dataset = repository.get(name, model="prewritten", trim_reasoning=False,
                                 shuffle_upon_init=False)
        texts = [normalise(m.content) for d in dataset.dialogues for m in d]
        corpora[name] = " | ".join(texts)
        corpus_shingles[name] = set().union(*(shingles(t.split(), n) for t in texts))
        logger.info(f"{name}: {len(dataset)} dialogues")

    excluded: dict[str, list[int]] = defaultdict(list)
    matches: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    unchecked: dict[str, int] = defaultdict(int)
    loose_kept: dict[str, int] = defaultdict(int)
    examples = []
    for split in A.SPLITS.values():
        for row in A.load_rows(split):
            q = normalise(A.question(row))
            if len(q.split()) < min_words:
                unchecked[split] += 1
                continue
            own = shingles(q.split(), n)
            overlap = {name: len(own & corpus_shingles[name]) / len(own) if own else 0.0
                       for name in corpora}
            hits = [name for name, corpus in corpora.items()
                    if q in corpus or overlap[name] >= near]
            if not hits and max(overlap.values()) >= loose:
                loose_kept[split] += 1
            if hits:
                excluded[split].append(row["row"])
                for name in hits:
                    matches[split][name] += 1
                if len(examples) < 10:
                    examples.append((split, row["row"], hits, A.question(row)[:120]))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": f"{A.REPO_ID}@{A.REVISION} ({A.CONFIG})",
        "checked_against": list(AUDITED),
        "method": (f"normalised question (multiple-choice stem only) found inside any message, "
                   f"or >= {near} of its {n}-word shingles found there"),
        "min_words": min_words,
        "unchecked_short_questions": dict(unchecked),
        f"kept_with_shingle_overlap_>=_{loose}": dict(loose_kept),
        "matches": {split: dict(counts) for split, counts in matches.items()},
        "excluded": {split: sorted(rows) for split, rows in excluded.items()},
    }, indent=2) + "\n", encoding="utf-8")
    for split in A.SPLITS.values():
        logger.info(f"{split}: {len(excluded.get(split, []))} rows excluded "
                    f"{dict(matches.get(split, {}))}; {loose_kept.get(split, 0)} kept at "
                    f">= {loose} shingle overlap; {unchecked.get(split, 0)} too short to check")
    for example in examples:
        logger.info(f"  e.g. {example}")
    logger.info(f"wrote {out}")


def _in_order(dataset, order):
    """`dataset` with its rows in `order`, unshuffled."""
    dialogues, labels, metadata = dataset._slice_by_index([int(i) for i in order])
    return type(dataset)(dataset.variant, dialogues, labels, metadata, shuffle_upon_init=False,
                         skip_variant_validation=True)


def extract(layer: int = 20, chunk: int = 96, token_budget: int = 16000,
            pooled_budget: int = 8000) -> None:
    """Fetch, at `layer`, what `fit` and `compare` need from NDIF. Resumable.

    Downloads here run at about 1.5 MB/s, and they, not NDIF, set the pace. So each part fetches
    only what its reader uses:

    - **Training rows** (all pooled rows: the runner's train and validation splits alike). Only
      the response tokens are fetched (tail mode) and stored in the layer's cache, which is where
      `fit` reads them. Rows go out sorted by response length. Each request pads every row to
      its longest response, which unsorted cost about 2x the bytes. Chunks of `chunk` rows keep
      memory bounded: the runner's `PairedActivations` would rebuild each class as one padded
      tensor, several GB here, on a 16 GB machine. Cache keys depend only on each row's tokens,
      so neither the order nor the chunking changes them.
    - **Held-out split.** It is only ever scored, and a linear probe's score on a row's mean
      activation equals its mean token score. So each row is pooled on the server, with
      `persona_finetuned`'s span-mean read, and one vector per row comes back. They are saved to
      `VALIDATION_MEANS`. The read uses `pooled_budget` tokens per request. At 16,000, the batch
      holding the split's longest row (about 1,850 tokens) ran out of GPU memory on NDIF. That read
      retries a request but cannot shrink it, unlike the training rows' path.
    """
    import torch

    os.environ["ACTS_CACHE_DIR"] = CACHE_DIRS[layer]
    from deception_detection.activations import Activations, detected_tail_lengths
    from deception_detection.models import get_model_and_tokenizer
    from deception_detection.repository import DatasetRepository
    from deception_detection.scripts import persona_finetuned as F
    from deception_detection.scripts.gender_secret import _environment
    from deception_detection.tokenized_data import TokenizedDataset
    from deception_detection.types import Label

    def tokenize(dataset):
        return TokenizedDataset.from_dataset(dataset, tokenizer, detect_all=False,
                                             detect_only_start_of_turn=False,
                                             detect_only_last_token=False)

    _environment()  # NDIF and HF keys from .env, longer timeouts
    model, tokenizer = get_model_and_tokenizer("llama-70b-3.3")
    repository = DatasetRepository()

    training = repository.get("aisi_llama__pooled", model="prewritten", trim_reasoning=True,
                              shuffle_upon_init=True)
    ordered = _in_order(training, np.argsort(detected_tail_lengths(tokenize(training)),
                                             kind="stable"))
    for start in range(0, len(ordered), chunk):
        Activations.from_model(model, tokenize(ordered[start : start + chunk]), layers=[layer],
                               token_budget=token_budget, positions="tail")
        logger.info(f"layer {layer} training rows: {min(start + chunk, len(ordered))}/{len(ordered)}")

    held_out = repository.get("aisi_llama__validation", model="prewritten", trim_reasoning=True,
                              shuffle_upon_init=False)
    toks = tokenize(held_out)
    padded = toks.tokens.shape[1]
    items = []
    for i in range(len(held_out)):
        length = int(toks.attention_mask[i].sum())
        detected = torch.nonzero(toks.detection_mask[i, padded - length:]).ravel()
        a0, a1 = int(detected[0]), int(detected[-1]) + 1
        if a1 - a0 != len(detected):
            raise ValueError(f"held-out row {i}: the detected tokens are not one span")
        items.append((toks.tokens[i, padded - length:].tolist(), a0, a1))
    meta = {"row": np.array(held_out.metadata["row"]),
            "deceptive": np.array([label == Label.DECEPTIVE for label in held_out.labels]),
            "layers": np.array([layer])}
    VALIDATION_MEANS.mkdir(parents=True, exist_ok=True)
    F._read(model, items, [layer], None, VALIDATION_MEANS / f"validation_L{layer}.npz", meta,
            pooled_budget, max_rows=chunk)
    logger.info(f"layer {layer}: held-out split pooled ({len(items)} rows)")


def fit(layer: int = 20, variant: str = "pooled", block: int = 4096) -> None:
    """Fit the base probe at `layer` on AISI's rows exactly as the runner's `get_detector` would.

    The runner's own objects supply the rows:

    - `ExperimentConfig` from `aisi_llama_L{layer}.yaml`, with `train_data` and `name` set for
      `variant`, which also creates the run folder and its `cfg.yaml`;
    - its `Experiment.train_dataset`;
    - `PairedActivations.from_dataset`'s class balancing and tokenisation.

    Each row's response tokens are read from the cache, where `extract` stored them. The fit is
    `deception_detection.streaming_lr`: sklearn's lbfgs fit (see its tests) on one float32 matrix,
    where the runner's in-memory fit would need about 17 GB here. The probe is saved in the
    runner's format, next to the `cfg.yaml` that records its recipe.
    """
    import time

    import torch
    import yaml

    from deception_detection import acts_cache, streaming_lr
    from deception_detection.activations import cache_entries
    from deception_detection.detectors import LogisticRegressionDetector
    from deception_detection.experiment import Experiment, ExperimentConfig
    from deception_detection.tokenized_data import get_tokenized_dataset_pair

    os.environ["ACTS_CACHE_DIR"] = CACHE_DIRS[layer]
    config = yaml.safe_load((CONFIGS / f"aisi_llama_L{layer}.yaml").read_text())
    config |= {"train_data": f"aisi_llama__{variant}",
               "name": AISI_ARMS[variant][0].format(layer=layer)}
    cfg = ExperimentConfig(**config)
    if not cfg.normalize_acts or cfg.use_class_weights or cfg.method != "lr":
        raise NotImplementedError("the streaming fit covers the standardised, unweighted lr probe")
    if (cfg.folder / "detector.pt").exists():
        logger.info(f"{cfg.folder.name} already has a probe; delete its detector.pt to refit")
        return

    experiment = Experiment(cfg)
    train = experiment.train_dataset
    honest, deceptive = train.get_honest(), train.get_deceptive()
    size = min(len(honest), len(deceptive))  # as PairedActivations.from_dataset balances them
    honest_toks, deceptive_toks = get_tokenized_dataset_pair(
        honest[:size], deceptive[:size], experiment.tokenizer,
        enforce_equal_detect_lengths=False, detect_all=cfg.detect_all_tokens,
        detect_only_start_of_turn=cfg.detect_only_start_of_turn,
        detect_only_last_token=cfg.detect_only_last_token,
    )
    # deceptive first: the runner fits detector.fit(deceptive, honest), deceptive as positive
    rows = []
    for toks, label in ((deceptive_toks, 1), (honest_toks, 0)):
        keys, kept = cache_entries(toks, MODEL, [layer], "tail")
        padded = toks.tokens.shape[1]
        rows += [(keys[i], kept[i], toks.detection_mask[i, padded - kept[i]:].bool(), label)
                 for i in range(len(keys))]
    n_tokens = sum(int(mask.sum()) for _, _, mask, _ in rows)
    X, y, at = None, np.empty(n_tokens), 0
    for key, length, mask, label in rows:
        acts = acts_cache.load(key, length)
        if acts is None:
            raise RuntimeError(f"a training row is not cached at layer {layer}; "
                               f"run `aisi_probe.py extract --layer {layer}` first")
        tokens = acts[mask, 0]  # kept in the cached dtype; the fit reads it block by block
        if X is None:
            X = torch.empty((n_tokens, tokens.shape[1]), dtype=tokens.dtype)
        X[at : at + len(tokens)], y[at : at + len(tokens)] = tokens, label
        at += len(tokens)
    assert X is not None and at == n_tokens
    logger.info(f"{cfg.folder.name}: {size} rows per class, {n_tokens} response tokens "
                f"({int(y.sum())} deceptive), {X.element_size() * X.nelement() / 2**30:.1f} GiB "
                f"of {X.dtype}; fitting")

    start = time.time()
    result = streaming_lr.fit(X, y, C=1 / cfg.reg_coeff, max_iter=cfg.max_iter, block=block)
    detector = LogisticRegressionDetector(layers=[layer], reg_coeff=cfg.reg_coeff,
                                          normalize=True, max_iter=cfg.max_iter)
    detector.directions = torch.tensor(result.coef.reshape(1, -1), dtype=torch.float32)
    detector.scaler_mean = torch.tensor(result.mean.reshape(1, -1), dtype=torch.float32)
    detector.scaler_scale = torch.tensor(result.scale.reshape(1, -1), dtype=torch.float32)
    detector.save(cfg.folder / "detector.pt")
    (cfg.folder / "fit_record.json").write_text(json.dumps({
        "fit": "deception_detection.streaming_lr (sklearn's lbfgs problem, streamed)",
        "rows_per_class": size, "response_tokens": n_tokens,
        "deceptive_tokens": int(y.sum()), "C": 1 / cfg.reg_coeff,
        "n_iter": result.n_iter, "converged": result.converged, "message": result.message,
        "seconds": round(time.time() - start, 1),
    }, indent=2) + "\n", encoding="utf-8")
    logger.info(f"{cfg.folder.name}: {result.n_iter} iterations, converged={result.converged} "
                f"({result.message}), {time.time() - start:.0f} s; saved the probe")


# (a, b) pairs whose Liars' Bench-average AUROC difference a - b gets a paired interval
PAIRS = (
    ("base probe (AISI-trained)", "base probe (RepE-trained)"),
    ("knowledge-denial axis (4v6)", "base probe (AISI-trained)"),
    ("extended axis (12v13)", "base probe (AISI-trained)"),
)


def paired_lb_average(scored: dict, arms: dict, layer: int, n_boot: int = 1000,
                      seed: int = 42) -> list[dict]:
    """Paired 95% intervals for differences in Liars' Bench-average AUROC.

    Each draw resamples every Liars' Bench dataset once, stratified by class, and scores all arms
    on the same rows. So a difference whose interval excludes zero is a difference between the
    arms, not between resamples.
    """
    from deception_detection.scripts import persona_ood as P

    rng = np.random.default_rng(seed)
    names = [name for name in P.SHARED_DATASETS if name in scored]
    scores = {arm: {name: fn(scored[name].z) for name in names} for arm, fn in arms.items()}
    point = {arm: np.mean([P.auroc(scored[n].y, scores[arm][n]) for n in names]) for arm in arms}
    boot = {arm: np.zeros(n_boot) for arm in arms}
    for b in range(n_boot):
        for name in names:
            y = scored[name].y
            rows = np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum()) for c in (1, 0)])
            for arm in arms:
                boot[arm][b] += P.auroc(y[rows], scores[arm][name][rows]) / len(names)
    out = []
    for a, b in PAIRS:
        if a in arms and b in arms:
            delta = boot[a] - boot[b]
            out.append({"layer": layer, "a": a, "b": b, "delta": point[a] - point[b],
                        "lo": float(np.quantile(delta, 0.025)),
                        "hi": float(np.quantile(delta, 0.975))})
    return out


def compare(layers=LAYERS, n_boot: int = 1000, out_dir: str | Path = RESULTS) -> None:
    """Score every probe and persona direction on the same activations, per layer.

    Arms: the RepE-trained base probe, each fitted AISI probe, and the persona directions
    (knowledge-denial 4v6, extended 12v13, published 8v7). The AISI probes come from
    `results/aisi-llama-*L{layer}/`, and a missing one is skipped with a warning. Unlike the
    RepE probe, they have no runner scores to reproduce. So the check is that their `cfg.yaml`
    names this layer and their training data, and that the RepE probe's own check passes on
    these pooled activations.

    Datasets: the five Liars' Bench sets and their average, Gender-Secret on the fine-tuned read,
    and AISI's held-out validation split. The held-out split appears once `extract` has pooled it
    at that layer. Thresholds sit at 1% FPR on
    Alpaca, as in `persona_ood.zero_shot_operating_table`, whose bootstrap this reuses. On GS, whose
    activations are the fine-tune's, only AUROC is meaningful.
    """
    import pandas as pd
    import yaml

    from deception_detection import persona_pcs
    from deception_detection.scripts import persona_ood as P

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables, deltas = [], []
    for layer in layers:
        pooled = P.load_pooled(POOLED[layer])
        control = pooled[P.CONTROL_DATASET]
        datasets = {name: ds for name, ds in pooled.items() if name in P.SHARED_DATASETS}

        arms = {P.BASE_ARM: P.base_probe_arm(REPO_ROOT / "results" / REPE_RUNS[layer], layer,
                                             datasets, control)}
        for variant, (folder, label) in AISI_ARMS.items():
            run = REPO_ROOT / "results" / folder.format(layer=layer)
            if not (run / "detector.pt").exists():
                logger.warning(f"{run.name} has no probe; {label!r} skipped at layer {layer}")
                continue
            cfg = yaml.safe_load((run / "cfg.yaml").read_text())
            if cfg.get("detect_layers") != [layer] or cfg.get("train_data") != f"aisi_llama__{variant}":
                raise ValueError(f"{run.name} is not the {variant} probe at layer {layer}")
            arms[label] = P.run_probe(run)
        bank = persona_pcs.load_bank_at_layer(layer, persona_pcs.EXTENDED_VECTORS_DIR)
        extra = P.extended_arms(bank)
        geom = persona_pcs.fit_geometry(bank, k=3)
        arms["knowledge-denial axis (4v6)"] = extra["auroc_denial_axis"]
        arms["extended axis (12v13)"] = extra["auroc_ext_axis"]
        arms["deception axis (8v7)"] = lambda z, g=geom: g.axis_score(z)

        gs = np.load(GS_DIR / f"adapted_L{layer}.npz", allow_pickle=False)
        scored = dict(datasets)
        scored["gender_secret (fine-tuned read)"] = P.Pooled(
            name="gender_secret (fine-tuned read)", z=gs["pooled"].astype(np.float64),
            y=(gs["labels"].astype(str) == "DECEPTIVE").astype(int))
        means = VALIDATION_MEANS / f"validation_L{layer}.npz"
        held_out = None
        if means.exists():  # read and closed at once: `extract` may be replacing this file
            with np.load(means, allow_pickle=False) as saved:
                held_out = {key: saved[key] for key in ("pooled", "deceptive", "done")}
        if held_out is not None and held_out["done"].all():
            scored["aisi_llama__validation"] = P.Pooled(
                name="aisi_llama__validation", z=held_out["pooled"][:, 0].astype(np.float64),
                y=held_out["deceptive"].astype(int))
        else:
            logger.warning(f"the held-out split is not pooled at layer {layer}; skipped")

        table = P.zero_shot_operating_table(scored, arms, control, layer, n_boot=n_boot)
        tables.append(table[table["dataset"] != "Average"])  # mixes GS and AISI into LB
        deltas += paired_lb_average(scored, arms, layer, n_boot=n_boot)

    table = pd.concat(tables, ignore_index=True)
    table.round(4).to_csv(out_dir / "comparison.csv", index=False)
    deltas = pd.DataFrame(deltas)
    deltas.round(4).to_csv(out_dir / "paired_deltas.csv", index=False)
    logger.info("paired Liars' Bench-average AUROC differences:\n"
                f"{deltas.round(3).to_string(index=False)}")
    control = table[table["threshold_source"] == "control"]
    for metric in ("auroc", "balacc"):
        wide = control.pivot_table(index=["layer", "arm"], columns="dataset", values=metric)
        logger.info(f"{metric} (thresholds at 1% FPR on Alpaca):\n{wide.round(3).to_string()}")
    logger.info(f"wrote {out_dir / 'comparison.csv'}")


# supervised probes in greys (RepE lighter, AISI darker), persona arms in the paper's blues
FIGURE_ARMS = [
    ("base probe (RepE-trained)", "base probe, trained on RepE", "#898781"),
    ("base probe (AISI-trained)", "base probe, trained on Llama's own lies (AISI)", "#3f3e3b"),
    ("knowledge-denial axis (4v6)", "knowledge-denial axis (4 vs 6), zero-shot", "#8ab6e8"),
    ("extended axis (12v13)", "extended axis (12 vs 13), zero-shot", "#2a78d6"),
]


def figure(in_dir: str | Path = RESULTS, name: str = "aisi_probe_auroc.png") -> None:
    """AUROC per dataset and layer: the two base probes beside the two persona axes.

    Whiskers are the 95% stratified bootstrap intervals from `comparison.csv`. The Liars' Bench
    average and Gender-Secret (read through its fine-tune) sit apart from the five datasets.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from deception_detection.scripts import persona_ood as P
    from deception_detection.scripts.persona_ood_figures import INK, INK_MUTED, SURFACE

    in_dir = Path(in_dir)
    table = pd.read_csv(in_dir / "comparison.csv")
    table = table[table["threshold_source"] == "control"]
    columns = [*P.SHARED_DATASETS, P.LB_AVERAGE, "gender_secret (fine-tuned read)"]
    labels = [*P.SHARED_DATASETS.values(), "Liars' Bench\naverage", "GS\n(fine-tune)"]
    x = np.array([0, 1, 2, 3, 4, 5.6, 7.0])
    width = 0.8 / len(FIGURE_ARMS)
    fig, axes = plt.subplots(len(LAYERS), 1, figsize=(10, 8.4), sharex=True, facecolor=SURFACE)
    for ax, layer in zip(axes, LAYERS):
        ax.set_facecolor(SURFACE)
        rows = table[table["layer"] == layer]
        for k, (arm, label, colour) in enumerate(FIGURE_ARMS):
            values = rows[rows["arm"] == arm].set_index("dataset").reindex(columns)
            at = x + (k - (len(FIGURE_ARMS) - 1) / 2) * width
            ax.bar(at, values["auroc"], width - 0.02, color=colour, zorder=3,
                   label=label if layer == LAYERS[0] else None)
            ax.errorbar(at, values["auroc"],
                        yerr=[values["auroc"] - values["auroc_lo"], values["auroc_hi"] - values["auroc"]],
                        fmt="none", ecolor=INK_MUTED, elinewidth=0.8, zorder=4)
        ax.axhline(0.5, color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.axvline(4.8, color=INK_MUTED, lw=0.6, zorder=1)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel(f"AUROC, layer {layer}", fontsize=9, color=INK)
        ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
        ax.grid(axis="y", color="#e6e5e1", lw=0.6, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK_MUTED)
    axes[-1].set_xticks(x, labels, fontsize=9, color=INK)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Trained on Llama's own lies, the base probe gains about 0.1 AUROC over RepE;\n"
                 "the zero-shot persona axes still lead on average: clearly at layer 20, narrowly at 40, "
                 "level at 44",
                 fontsize=11, color=INK, x=0.06, ha="left", y=1.0)
    fig.tight_layout()
    fig.savefig(in_dir / name, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    logger.info(f"wrote {in_dir / name}")


if __name__ == "__main__":
    fire.Fire({"audit": audit, "extract": extract, "fit": fit, "compare": compare, "figure": figure})
