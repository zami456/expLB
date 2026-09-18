"""Build per-example mean-pooled activations from the on-disk activation cache.

The persona method (arXiv 2605.09391) works on one vector per example -- the mean over the
assistant response tokens -- whereas the deployed probe works per token. This reads the cache
written by the Liars' Bench sweep and reduces it to that pooled form, once, so the downstream
PCA and probe experiments run in seconds on ~450 MB instead of re-reading 60 GB each time.

No remote calls: every example needed is already cached.

    python deception_detection/scripts/pool_activations.py \
        --config-file deception_detection/scripts/configs/XXXX-9-llama-all.yaml
"""

from pathlib import Path
from typing import Any

import fire
import numpy as np
import yaml
from tqdm import tqdm

from deception_detection import acts_cache
from deception_detection.data.base import DialogueDataset
from deception_detection.experiment import Experiment, ExperimentConfig
from deception_detection.log import logger
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.utils import REPO_ROOT

POOLED_DIR = REPO_ROOT / "activations" / "pooled"


def pool_dataset(
    dataset: DialogueDataset,
    experiment: Experiment,
    model_name: str,
    layers: list[int],
) -> tuple[np.ndarray, list[str]]:
    """Mean-pool each example's cached activations over its detection-mask tokens.

    Tokenising the whole dataset at once is safe even though the cache was written from shards:
    the key is built from an example's *unpadded* tokens, and padding to a different batch width
    does not change those.
    """
    toks = TokenizedDataset.from_dataset(
        dataset,
        experiment.tokenizer,
        detect_all=experiment.cfg.detect_all_tokens,
        detect_only_start_of_turn=experiment.cfg.detect_only_start_of_turn,
        detect_only_last_token=experiment.cfg.detect_only_last_token,
    )
    assert toks.detection_mask is not None, f"{dataset.name} has no detection mask"

    tokens, attn, mask = toks.tokens, toks.attention_mask, toks.detection_mask
    n, padded_len = tokens.shape

    pooled: list[np.ndarray] = []
    misses: list[int] = []
    kept: list[int] = []
    empty_masks = 0

    for i in tqdm(range(n), desc=f"pooling {dataset.name}", unit="ex"):
        true_len = int(attn[i].sum().item())
        start = padded_len - true_len  # left padding: real tokens are the final true_len columns
        key = acts_cache.example_key(tokens[i, start:], model_name, layers)
        acts = acts_cache.load(key, true_len)
        if acts is None:
            misses.append(i)
            continue

        row_mask = mask[i, start:].bool()
        if not bool(row_mask.any()):
            # No response tokens are marked, so there is nothing to pool "over the output tokens
            # only". Falling back to every attended token would quietly average prompt
            # activations into a response-only quantity, so drop the row instead and let the
            # caller keep labels aligned. The deployed pipeline also yields empty scores here.
            empty_masks += 1
            continue
        kept.append(i)
        pooled.append(acts[row_mask].float().mean(0).squeeze(0).numpy())

    if misses:
        # a miss means the tokenisation diverged from the run that wrote the cache; dropping the
        # rows would quietly bias every number downstream, so refuse to produce a partial file
        raise RuntimeError(
            f"{dataset.name}: {len(misses)} of {n} examples missing from the cache "
            f"(first few indices: {misses[:5]}). Tokenisation does not match the run."
        )
    if empty_masks:
        logger.warning(
            f"{dataset.name}: dropped {empty_masks} of {n} examples with an empty detection mask"
        )

    labels = [dataset.labels[i].name for i in kept]
    return np.stack(pooled).astype(np.float32), labels


def run(
    config_file: str | Path = "deception_detection/scripts/configs/XXXX-9-llama-all.yaml",
    out_dir: str | Path = POOLED_DIR,
    overwrite: bool = False,
    **kwargs: dict[str, Any],
) -> None:
    config_file, out_dir = Path(config_file), Path(out_dir)
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)
    config_dict |= kwargs
    cfg = ExperimentConfig(**config_dict)  # type: ignore[arg-type]

    experiment = Experiment(cfg)
    model_name = str(experiment.model.name_or_path)
    layers = cfg.detect_layers
    logger.info(f"pooling from {acts_cache.cache_root()} | model={model_name} layers={layers}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # train_dataset is in neither eval_datasets nor control_datasets, so it would otherwise be
    # skipped -- leaving RepE represented only by its 102-row validation remainder rather than
    # the 490-row split the probe actually trains on.
    datasets = (
        [experiment.train_dataset]
        + list(experiment.eval_datasets)
        + list(experiment.control_datasets)
    )

    for dataset in datasets:
        if len(dataset) == 0:
            logger.warning(f"skipping empty dataset {dataset.name}")
            continue
        path = out_dir / f"{dataset.name}.npz"
        if path.exists() and not overwrite:
            logger.info(f"{dataset.name}: already pooled, skipping ({path})")
            continue

        pooled, labels = pool_dataset(dataset, experiment, model_name, layers)
        assert len(pooled) == len(labels), (
            f"{dataset.name}: pooled {len(pooled)} rows but {len(labels)} labels"
        )
        tmp = path.with_suffix(".npz.tmp")
        with open(tmp, "wb") as f:  # np.savez_compressed appends .npz to a bare path
            np.savez_compressed(f, pooled=pooled, labels=np.array(labels), name=dataset.name)
        tmp.replace(path)
        n_dec = sum(label == "DECEPTIVE" for label in labels)
        logger.info(
            f"{dataset.name}: {pooled.shape} -> {path.name} "
            f"({n_dec} deceptive / {len(labels)}, {path.stat().st_size / 1e6:.0f} MB)"
        )

    logger.info(f"done; pooled activations in {out_dir}")


if __name__ == "__main__":
    fire.Fire(run)
