"""The core persona study at every 4th layer from 20 to the last, and one summary across depth.

The study's per-layer results existed at layers 20, 40 and 44. They were built from the runner's
full per-token cache (`pool_activations.py`, about 60 GB a layer), which at this machine's ~2 MB/s
from NDIF is hours per layer. The new layers get the same outputs, built differently:

- **`pool`** computes each example's mean over its detected tokens on NDIF, which is what
  `pool_activations` computes locally. It fetches every requested layer in one forward pass and
  writes them in `pool_activations`' format to `activations/pooled_L{layer}/`.
- **`probe`** trains the RepE base probe with the runner's own `Experiment.get_detector()`,
  using `acts_positions: tail`, so only RepE's statement tokens are fetched. The runner never
  scores it, so the probe has no `results_table.csv`. `check40` stands in for
  `persona_ood.py verify`: it rebuilds layer 40's pooled vectors and probe this way and compares
  them with the originals.
- **`analyse`** runs `persona_ood.py run` and `zeroshot`, and the figures, as for layers 20, 40
  and 44.
- **`depth`** puts every layer in one table and figure.

    python deception_detection/scripts/layer_sweep.py check40
    python deception_detection/scripts/layer_sweep.py pool [--datasets NAME,NAME]
    python deception_detection/scripts/layer_sweep.py probe --layer 24
    python deception_detection/scripts/layer_sweep.py analyse --layer 24
    python deception_detection/scripts/layer_sweep.py depth
"""

import os
import shutil
import time
from pathlib import Path

import fire
import numpy as np

from deception_detection.log import logger
from deception_detection.utils import REPO_ROOT

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
# Llama-3.3-70B has 80 blocks, 0-79; layer L is block L's output. 79 closes the stride-4 grid.
NEW_LAYERS = (24, 28, 32, 36, 48, 52, 56, 60, 64, 68, 72, 76, 79)
ALL_LAYERS = (20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79)
BASE_CONFIG = REPO_ROOT / "deception_detection" / "scripts" / "configs" / "XXXX-9-llama-all.yaml"
CACHE = "E:/liars-bench-acts/cache"  # RepE's tokens for the new layers, beside layers 40 and 44
PARTIAL = Path("E:/liars-bench-acts/pooled_sweep_partial")  # one small file per NDIF request
DEPTH_DIR = REPO_ROOT / "results" / "persona_depth"


def pooled_dir(layer: int, root: Path = REPO_ROOT / "activations") -> Path:
    return root / ("pooled" if layer == 20 else f"pooled_L{layer}")


# Which persona folder an analysis reads. "ext" is the 36-persona extraction the study runs on;
# "orig" is the published 26, whose layer-20 slice is the paper's own bank. Both are swept so the
# unified axis can be read off each: from "orig" it is the published direction, from "ext" it is
# rebuilt on the same extraction as the extended axis, which is what makes the two comparable.
BANKS = ("ext", "orig")


def vectors_dir(bank: str = "ext") -> Path:
    from deception_detection import persona_pcs

    if bank not in BANKS:
        raise ValueError(f"bank must be one of {BANKS}, not {bank!r}")
    return persona_pcs.EXTENDED_VECTORS_DIR if bank == "ext" else persona_pcs.VECTORS_DIR


def results_dir(layer: int, bank: str = "ext") -> Path:
    if bank not in BANKS:
        raise ValueError(f"bank must be one of {BANKS}, not {bank!r}")
    return REPO_ROOT / "results" / f"persona_ood_{bank}_L{layer}"


def base_run(layer: int) -> Path:
    return REPO_ROOT / "results" / ("XXXX-9-llama-all" if layer == 20 else f"XXXX-9-llama-all-L{layer}")


def _names(datasets) -> list[str] | None:
    if datasets is None:
        return None
    return [d for d in (datasets.split(",") if isinstance(datasets, str) else datasets) if d]


# --- pooled activations, one request at a time ------------------------------------------------

def done_rows(folder: Path) -> set[int]:
    """The rows already pooled into `folder`'s request files."""
    rows: set[int] = set()
    for path in folder.glob("*.npz"):
        with np.load(path, allow_pickle=False) as saved:
            rows.update(int(r) for r in saved["rows"])
    return rows


def save_request(folder: Path, rows: list[int], layers: list[int], means: np.ndarray) -> None:
    """One request's means, [rows, layers, hidden], one array per layer so each reads alone."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{min(rows):06d}_{len(rows)}.npz"
    partial = path.with_suffix(".partial")
    with open(partial, "wb") as file:
        np.savez(file, rows=np.array(rows), **{f"L{layer}": means[:, j] for j, layer in enumerate(layers)})
    os.replace(partial, path)


def assemble(folder: Path, kept: list[int], layer: int) -> np.ndarray:
    """`layer`'s means for every kept row, in `kept` order, from the request files."""
    position = {row: k for k, row in enumerate(kept)}
    out, filled = None, np.zeros(len(kept), dtype=bool)
    for path in sorted(folder.glob("*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            rows, means = saved["rows"], saved[f"L{layer}"]
        if out is None:
            out = np.empty((len(kept), means.shape[1]), dtype=np.float32)
        at = [position[int(r)] for r in rows]
        out[at], filled[at] = means, True
    if out is None or not filled.all():
        raise RuntimeError(f"{folder.name}: {int((~filled).sum())} of {len(kept)} rows not pooled")
    return out


def write_pooled(path: Path, pooled: np.ndarray, labels: list[str], name: str) -> None:
    """`pool_activations`' format, which `persona_ood.load_pooled` reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".npz.tmp")
    with open(partial, "wb") as file:
        np.savez_compressed(file, pooled=pooled.astype(np.float32), labels=np.array(labels), name=name)
    partial.replace(path)


def _experiment():
    """The datasets `pool_activations.run` pools, as the layer-20 RepE run defines them. That run's
    folder exists, so building its config writes nothing."""
    import yaml

    from deception_detection.experiment import Experiment, ExperimentConfig

    cfg = ExperimentConfig(**yaml.safe_load(BASE_CONFIG.read_text()))
    return cfg, Experiment(cfg)


def pool(layers=NEW_LAYERS, datasets=None, token_budget: int = 16000,
         out_root: str | Path = REPO_ROOT / "activations", partial_root: str | Path = PARTIAL,
         overwrite: bool = False, shard: str | None = None) -> None:
    """Pool every dataset `pool_activations` pools at each of `layers`, on NDIF. Resumable.

    Rows are those with a non-empty detection mask, tokenised as the runner tokenises them.
    Batches follow the runner's rules: left padding, a token budget halved for whatever runs out
    of memory. Each request returns `[rows, layers, hidden]` float32 means over the detected
    tokens and is saved on its own, so an interrupted run resumes at the next request.

    `shard="i/n"` takes only every n-th row, from row i, and writes nothing. That way n processes
    can split one long dataset, such as insider trading: its ~1,500-token rows fit 3-4 to a
    request. A final call without `shard` finds every row pooled and writes the files.
    """
    import torch

    from deception_detection.activations import trace_layer_outputs
    from deception_detection.batching import BudgetMemory, execute_in_batches
    from deception_detection.models import get_model_and_tokenizer
    from deception_detection.scripts.gender_secret import _environment
    from deception_detection.tokenized_data import TokenizedDataset

    layers = [int(layer) for layer in layers]
    out_root, partial_root = Path(out_root), Path(partial_root)
    _environment()  # NDIF and HF keys from .env, longer timeouts
    cfg, experiment = _experiment()
    model, _ = get_model_and_tokenizer("llama-70b-3.3")
    memory = BudgetMemory(partial_root / "budget_memory.json")
    memory_key = BudgetMemory.key(MODEL, len(layers))
    wanted = _names(datasets)
    for dataset in [experiment.train_dataset, *experiment.eval_datasets, *experiment.control_datasets]:
        if wanted is not None and dataset.name not in wanted:
            continue
        targets = {layer: pooled_dir(layer, out_root) / f"{dataset.name}.npz" for layer in layers}
        if not overwrite and all(path.exists() for path in targets.values()):
            logger.info(f"{dataset.name}: already pooled at every layer; skipped")
            continue
        toks = TokenizedDataset.from_dataset(
            dataset, experiment.tokenizer, detect_all=cfg.detect_all_tokens,
            detect_only_start_of_turn=cfg.detect_only_start_of_turn,
            detect_only_last_token=cfg.detect_only_last_token,
        )
        kept = [i for i in range(len(dataset)) if bool(toks.detection_mask[i].any())]
        true_lens = [int(n) for n in toks.attention_mask.sum(1)]
        padded = toks.tokens.shape[1]
        folder = partial_root / f"{dataset.name}__L{'-'.join(map(str, layers))}"
        done = done_rows(folder)
        missing = [i for i in kept if i not in done]
        if shard is not None:
            part, parts = (int(x) for x in str(shard).split("/"))
            missing = [i for i in missing if i % parts == part]
        logger.info(f"{dataset.name}: {len(kept)} rows, {len(done)} already pooled"
                    + (f"; shard {shard} takes {len(missing)}" if shard is not None else ""))

        def run_batch(batch: list[int]) -> None:
            start = time.time()
            width = max(true_lens[i] for i in batch)
            rows = torch.tensor(batch)
            columns = slice(padded - width, None)  # left padding: real tokens sit at the right
            means = trace_layer_outputs(
                model, toks.tokens[rows][:, columns], toks.attention_mask[rows][:, columns],
                layers, span_mask=toks.detection_mask[rows][:, columns].float(),
            )
            save_request(folder, batch, layers, means.float().numpy())
            done.update(batch)
            logger.info(f"  {dataset.name}: {len(done)}/{len(kept)} rows ({len(batch)} x {width} "
                        f"tokens, {time.time() - start:.0f} s)")

        if missing:
            execute_in_batches(missing, lambda i: true_lens[i], memory.suggest(memory_key, token_budget),
                               run_batch, memory=memory, memory_key=memory_key)
        if shard is not None:
            logger.info(f"{dataset.name}: shard {shard} done; the files wait for every shard")
            continue
        labels = [dataset.labels[i].name for i in kept]
        for layer, path in targets.items():
            if path.exists() and not overwrite:
                continue
            write_pooled(path, assemble(folder, kept, layer), labels, dataset.name)
        logger.info(f"{dataset.name}: written at layers {layers}")


# --- the RepE base probe ----------------------------------------------------------------------

def probe(layer: int, name: str | None = None) -> None:
    """Train the RepE base probe at `layer` with the runner's own code, fetching only RepE's
    statement tokens. The run folder is `base_run(layer)` unless `name` is given."""
    import yaml

    from deception_detection.experiment import Experiment, ExperimentConfig
    from deception_detection.scripts.gender_secret import _environment

    os.environ["ACTS_CACHE_DIR"] = CACHE
    _environment()
    config = yaml.safe_load(BASE_CONFIG.read_text())
    config |= {"detect_layers": [int(layer)], "name": name or base_run(layer).name,
               "acts_positions": "tail"}
    cfg = ExperimentConfig(**config)
    if (cfg.folder / "detector.pt").exists():
        logger.info(f"{cfg.folder.name} already has a probe")
        return
    Experiment(cfg).get_detector()
    logger.info(f"{cfg.folder.name}: probe saved")


# --- the check that stands in for `persona_ood.py verify` ---------------------------------------

def check40(scratch: str | Path = PARTIAL / "check40", keep: bool = False) -> None:
    """Rebuild layer 40's inputs the sweep's way and compare them with the originals.

    - Pooled vectors for CG and the RepE validation split, against `activations/pooled_L40`: row
      cosine and relative error, and the knowledge-denial axis's AUROC on both.
    - The RepE probe trained by `probe`, against `XXXX-9-llama-all-L40`: direction cosine, and its
      AUROCs on the pooled Liars' Bench sets against that run's `results_table.csv`.
    """
    import pandas as pd

    from deception_detection import persona_pcs
    from deception_detection.scripts import persona_ood as P

    scratch = Path(scratch)
    names = ["convince_me__plain", "repe_honesty__plain_val"]
    pool(layers=[40], datasets=names, out_root=scratch, partial_root=scratch / "partial")
    denial = P.extended_arms(persona_pcs.load_bank_at_layer(40, persona_pcs.EXTENDED_VECTORS_DIR))["auroc_denial_axis"]
    for name in names:
        with np.load(pooled_dir(40) / f"{name}.npz") as a, np.load(pooled_dir(40, scratch) / f"{name}.npz") as b:
            old, new, labels = a["pooled"].astype(np.float64), b["pooled"].astype(np.float64), a["labels"]
            assert (labels == b["labels"]).all(), f"{name}: rows or labels differ"
        cos = (old * new).sum(1) / np.linalg.norm(old, axis=1) / np.linalg.norm(new, axis=1)
        rel = np.linalg.norm(new - old, axis=1) / np.linalg.norm(old, axis=1)
        y = (labels.astype(str) == "DECEPTIVE").astype(int)
        logger.info(f"{name}: {len(y)} rows; cosine min {cos.min():.6f}, relative error median "
                    f"{np.median(rel):.2e} max {rel.max():.2e}; knowledge-denial AUROC "
                    f"{P.auroc(y, denial(old)):.4f} (original) vs {P.auroc(y, denial(new)):.4f}")

    run = REPO_ROOT / "results" / "XXXX-9-llama-all-L40-sweepcheck"
    probe(40, name=run.name)
    original = REPO_ROOT / "results" / "XXXX-9-llama-all-L40"
    import pickle

    with open(original / "detector.pt", "rb") as f:
        a = pickle.load(f)
    with open(run / "detector.pt", "rb") as f:
        b = pickle.load(f)
    # compare the directions in activation units: coef / scale
    da = (a["directions"].double() / a["scaler_scale"].double()).ravel().numpy()
    db = (b["directions"].double() / b["scaler_scale"].double()).ravel().numpy()
    logger.info(f"RepE probe direction cosine (retrained vs original): "
                f"{da @ db / np.linalg.norm(da) / np.linalg.norm(db):.6f}")
    published = pd.read_csv(original / "results_table.csv", index_col=0)
    score = P.run_probe(run)
    rows = []
    for name, ds in P.load_pooled(pooled_dir(40)).items():
        if name in published.index and ds.two_class:
            rows.append({"dataset": name, "retrained": P.auroc(ds.y, score(ds.z)),
                         "original run": float(published.loc[name, "auroc"])})
    table = pd.DataFrame(rows)
    table["delta"] = table["retrained"] - table["original run"]
    logger.info(f"AUROC of the retrained probe on the pooled layer-40 sets:\n{table.round(4).to_string(index=False)}")
    if not keep:
        shutil.rmtree(run, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)


# --- per-layer results and the depth summary ------------------------------------------------------

TRANSFER = {  # summary.csv feature -> label; off-diagonal means of the transfer matrices
    "ext_axis1": "extended axis (12v13)",
    "denial_axis1": "knowledge-denial axis (4v6)",
    "ext_pc3": "extended PCs (k=3)",
    "uni_axis1": "unified axis (13v12)",
    "axis1": "deception axis (8v7)",
    "baseline": "single-source probe (8192-d)",
}
ZERO_SHOT = [  # zero_shot_operating.csv arm -> label, colour (the paper's Fig 2 palette)
    ("base probe (RepE-trained)", "base probe (RepE-trained)", "#898781"),
    ("deception axis (8v7)", "deception axis (8v7)", "#eb6834"),
    ("unified axis (13v12)", "unified axis (13v12)", "#b5399e"),
    ("knowledge-denial axis (4v6)", "knowledge-denial axis (4v6)", "#8ab6e8"),
    ("extended axis (12v13)", "extended axis (12v13)", "#2a78d6"),
]
HPKR = "harm_refusal_knowledge_report__plain"
TRANSFER_SOURCE = "transfer, off-diagonal mean"
LB_SOURCE = "zero-shot, Liars' Bench average"
HPKR_SOURCE = "zero-shot, HP-KR"


def layer_table(layer: int, bank: str = "ext") -> "pd.DataFrame":
    """One layer's headline numbers, long format: source, arm, metric, value.

    Arms the bank does not have -- the extended and knowledge-denial ones under `bank="orig"` --
    are absent from its files and come back as NaN rather than raising, so the two banks' tables
    share a shape and can be concatenated.
    """
    import pandas as pd

    from deception_detection.scripts import persona_ood as P

    out, rows = results_dir(layer, bank), []
    for file, metric in (("summary.csv", "auroc"), ("summary_balacc.csv", "balacc")):
        table = pd.read_csv(out / file).set_index("features")
        for feature, label in TRANSFER.items():
            value = (float(table.loc[feature, "off_diagonal_mean"])
                     if feature in table.index else float("nan"))
            rows.append({"layer": layer, "source": TRANSFER_SOURCE, "arm": label,
                         "metric": metric, "value": value})
    operating = pd.read_csv(out / "zero_shot_operating.csv")
    operating = operating[operating["threshold_source"] == "control"]
    for dataset, source in ((P.LB_AVERAGE, LB_SOURCE), (HPKR, HPKR_SOURCE)):
        for arm, label, _ in ZERO_SHOT:
            row = operating[(operating["arm"] == arm) & (operating["dataset"] == dataset)]
            for metric in ("auroc", "balacc", "recall"):
                rows.append({"layer": layer, "source": source, "arm": label, "metric": metric,
                             "value": float(row[metric].iloc[0]) if len(row) else float("nan")})
    return pd.DataFrame(rows)


def replaceable(summary: Path, layer: int) -> bool:
    """May `analyse` overwrite this `summary.md`?

    Layers 20, 40 and 44 have hand-written write-ups that predate this script, and running the
    sweep over them once replaced all three with the template. Only a file this script wrote, or
    no file at all, may be replaced. Layer 44's own write-up is titled "# Layer 44: the depth
    curve turns over", so the test is the template's whole first line rather than a prefix.
    """
    if not summary.exists():
        return True
    return summary.read_text(encoding="utf-8").lstrip().startswith(f"# Layer {layer} (stride-4 sweep)")


def _layer_summary(layer: int, bank: str = "ext") -> str:
    table = layer_table(layer, bank)
    personas = ("36 personas from `persona including knowlede denial/`" if bank == "ext"
                else "the published 26 personas from `persona vectors/vectors_s3/`")
    sibling = "persona_ood_ext" if bank == "ext" else "persona_ood_orig"

    def cell(source: str, arm: str, metric: str) -> str:
        value = table[(table.source == source) & (table.arm == arm) & (table.metric == metric)]["value"]
        return f"{value.iloc[0]:.3f}" if len(value) and np.isfinite(value.iloc[0]) else "n/a"

    transfer = "\n".join(
        f"| {label} | {cell(TRANSFER_SOURCE, label, 'auroc')} | "
        f"{cell(TRANSFER_SOURCE, label, 'balacc')} |" for label in TRANSFER.values())
    zero = "\n".join(
        f"| {label} | {cell(LB_SOURCE, label, 'auroc')} | {cell(LB_SOURCE, label, 'balacc')} | "
        f"{cell(HPKR_SOURCE, label, 'auroc')} | {cell(HPKR_SOURCE, label, 'balacc')} |"
        for _, label, _ in ZERO_SHOT)
    return f"""# Layer {layer} (stride-4 sweep)

One layer of the persona study's depth sweep; `../persona_depth/summary.md` has every layer side
by side. The analysis matches `{sibling}_L20/`, `_L40/` and `_L44/`: Llama-3.3-70B-Instruct,
six Liars' Bench datasets, {personas}. Two inputs were
made differently, and both were checked against layer 40 (`layer_sweep.py check40`):

- **Pooled activations were computed on NDIF** (`layer_sweep.py pool`): each example's mean over
  its detected tokens, rather than pooled from a per-token cache.
- **The RepE base probe was trained by the runner** on RepE's statement tokens only
  (`acts_positions: tail`) and never scored by it. It joins `zero_shot_operating.csv` without the
  check that it reproduces its run's own scores.

| transfer, off-diagonal mean | AUROC | balanced accuracy @1% FPR |
|---|---|---|
{transfer}

| zero-shot (threshold: 1% FPR on Alpaca) | LB avg AUROC | LB avg balanced accuracy | HP-KR AUROC | HP-KR balanced accuracy |
|---|---|---|---|---|
{zero}

Files: the matrices and `summary*.csv` from `persona_ood.py run`, and `zero_shot.csv` and
`zero_shot_operating.csv` from `zeroshot`. The figures are `transfer_matrices.png`,
`transfer_delta.png`, `variants.png`, `transfer_sycophancy*.png`, `zero_shot_fig2.png` and
`zero_shot.png` (this layer only).

Reproduce:

```
python deception_detection/scripts/layer_sweep.py pool          # all 13 new layers at once
python deception_detection/scripts/layer_sweep.py probe --layer {layer}
python deception_detection/scripts/layer_sweep.py analyse --layer {layer} --bank {bank}
```
"""


def analyse(layer: int, bank: str = "ext") -> None:
    """Layer `layer`'s results, made as `persona_ood_ext_L44/summary.md` makes layer 44's.

    `bank` picks the persona folder: "ext" for the 36-persona extraction, "orig" for the
    published 26. The pooled activations and the base probe are the same either way -- only the
    directions they are scored against change.
    """
    import matplotlib

    matplotlib.use("Agg")
    from deception_detection.scripts import persona_ood as P
    from deception_detection.scripts import persona_ood_figures as F

    out, pooled, vectors = results_dir(layer, bank), pooled_dir(layer), vectors_dir(bank)
    P.run(pooled_dir=pooled, out_dir=out, layer=layer, vectors=vectors)
    P.zeroshot(pooled_dir=pooled, out_dir=out, layer=layer, vectors=vectors,
               base_run=base_run(layer), base_run_check=False)
    for figure in (F.run, F.variants, F.syc, F.fig2, F.zeroshot):
        figure(out)
    # Layers 20, 40 and 44 have hand-written write-ups that predate this script, and re-running
    # the sweep over them once replaced all three with the template. Only a file this function
    # wrote -- or no file -- is replaced.
    summary = out / "summary.md"
    if replaceable(summary, layer):
        summary.write_text(_layer_summary(layer, bank), encoding="utf-8")
    else:
        logger.info(f"kept the hand-written {summary}; delete it to regenerate")
    logger.info(f"layer {layer} ({bank}): results in {out}")


def depth(layers=ALL_LAYERS, out_dir: str | Path | None = None, bank: str = "ext") -> None:
    """Every layer's headline numbers in one table (`depth.csv`) and one figure (`depth.png`).

    `bank="orig"` reads the published-26 sweep instead, into `persona_depth_orig/`. Arms that
    folder has no personas for come back empty and are dropped from the figure's legend.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from deception_detection.scripts.persona_ood_figures import INK, INK_MUTED, SURFACE  # noqa

    if out_dir is None:
        out_dir = DEPTH_DIR if bank == "ext" else DEPTH_DIR.with_name("persona_depth_orig")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.concat([layer_table(layer, bank) for layer in layers], ignore_index=True)
    table.insert(1, "bank", bank)
    table.to_csv(out_dir / "depth.csv", index=False)

    # Colour carries the family -- orange for the published contrast and its sycophancy widening,
    # blue for the knowledge-denial one -- so within a family the lines share a hue and can run
    # within 0.02 of each other for most of the range. Dash pattern and marker are the secondary
    # encoding that keeps them apart; without it the three blues in the transfer panel read as
    # one band.
    transfer_style = {
        "extended axis (12v13)": ("#2a78d6", "-", "o"),
        "knowledge-denial axis (4v6)": ("#8ab6e8", (0, (5, 2)), "s"),
        "extended PCs (k=3)": ("#104281", (0, (1, 1.6)), "^"),
        "unified axis (13v12)": ("#b5399e", "-", "D"),
        "deception axis (8v7)": ("#eb6834", "-", "o"),
        "single-source probe (8192-d)": ("#898781", (0, (6, 2, 1, 2)), "v"),
    }
    arm_style = {label: (colour, *transfer_style.get(label, (None, "-", "o"))[1:])
                 for _, label, colour in ZERO_SHOT}
    arm_style["base probe (RepE-trained)"] = ("#898781", (0, (6, 2, 1, 2)), "v")
    # Columns are what is being measured, rows are how: ranking on top, the 1% FPR operating
    # point below. Every series in depth.csv except recall appears here.
    panels = [
        (TRANSFER_SOURCE, "auroc", "Transfer: off-diagonal AUROC", transfer_style),
        (LB_SOURCE, "auroc", "Zero-shot: Liars' Bench average AUROC", arm_style),
        (HPKR_SOURCE, "auroc", "Zero-shot: HP-KR AUROC", arm_style),
        (TRANSFER_SOURCE, "balacc", "Transfer: off-diagonal balanced accuracy (1% FPR)",
         transfer_style),
        (LB_SOURCE, "balacc", "Zero-shot: Liars' Bench average balanced accuracy (1% FPR)",
         arm_style),
        (HPKR_SOURCE, "balacc", "Zero-shot: HP-KR balanced accuracy (1% FPR)", arm_style),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6), facecolor=SURFACE)
    for index, (ax, (source, metric, title, styles)) in enumerate(zip(axes.ravel(), panels)):
        ax.set_facecolor(SURFACE)
        for label, (colour, dashes, marker) in styles.items():
            rows = table[(table.source == source) & (table.arm == label) & (table.metric == metric)]
            if rows.empty or rows["value"].isna().all():
                continue  # an arm this persona folder has no roles for
            ax.plot(rows["layer"], rows["value"], marker=marker, ms=4, lw=2, ls=dashes,
                    color=colour, label=label)
        ax.axhline(0.5, color=INK_MUTED, lw=1, ls=(0, (4, 3)))
        # Where every arm sits on chance -- HP-KR balanced accuracy in the published-26 folder --
        # autoscale zooms to a 0.004 window and float noise reads as structure. Keep a floor.
        low, high = ax.get_ylim()
        if high - low < 0.06:
            middle = (high + low) / 2
            ax.set_ylim(middle - 0.03, middle + 0.03)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xticks(list(layers))
        ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
        ax.grid(color="#e6e5e1", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # the top row carries both legends; the bottom row repeats their arms exactly
        if index in (0, 1):
            # an opaque background: the chance line runs behind the legend otherwise
            ax.legend(fontsize=7.5, labelcolor=INK, frameon=True, facecolor=SURFACE,
                      edgecolor="none", framealpha=0.95)
    for ax in axes[1]:
        ax.set_xlabel("layer (block output)", fontsize=9, color=INK_MUTED)
    personas = "36 personas" if bank == "ext" else "the published 26 personas"
    fig.suptitle(
        "The persona study across depth: Llama-3.3-70B-Instruct, every 4th layer from 20 "
        f"({personas})", fontsize=12, color=INK, x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "depth.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    logger.info(f"wrote {out_dir / 'depth.csv'} and depth.png")


if __name__ == "__main__":
    fire.Fire({"check40": check40, "pool": pool, "probe": probe, "analyse": analyse, "depth": depth})
