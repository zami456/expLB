"""Liars' Bench activations for the emotion-coordinate study.

Pooling (a), the mean over the final assistant message's detected tokens, is read from the thesis's pooled
files (`config.pooled_dir`), whose rows Phase 0 matched to the index tables in data/emotion_coords/index/.
Poolings (b), the last content token, and (c), the pre-response token, exist only at layer 20 (the user's
decision B): `extract_bc` computes them from the run's per-token cache and stores them float16 in
data/emotion_coords/lb/L20_bc/<dataset>.npz, after checking each row's cached span mean against the stored
pooled (a).

Every array is checked for non-finite rows at load; such rows are dropped and counted, never imputed.
"""

import hashlib
from dataclasses import dataclass

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from emotion_coords import config as C

POOLINGS = ("a", "b", "c")


def lb5() -> list[str]:
    return list(C.load()["datasets"])


def short(name: str) -> str:
    s = C.load()
    return (s["datasets"] | {s["control"]: "Alpaca", s["train_pairs"]: "RepE"}).get(name, name)


@dataclass
class Split:
    """One dataset's rows at one layer and pooling."""

    name: str
    x: np.ndarray  # (n, hidden) float32
    y: np.ndarray  # (n,) 1 = deceptive
    rows: np.ndarray  # pooled row ids (RepE: train rows, then val rows offset by VAL_OFFSET)
    groups: np.ndarray | None = None  # RepE: one id per statement, for grouped CV

    def take(self, keep: np.ndarray) -> "Split":
        return Split(self.name, self.x[keep], self.y[keep], self.rows[keep],
                     None if self.groups is None else self.groups[keep])


VAL_OFFSET = 100_000


def index(name: str) -> pd.DataFrame:
    table = pd.read_csv(C.data_dir("index") / f"{name}.csv")
    return table[table["pooled_row"] >= 0].reset_index(drop=True)


def _finite(name: str, x: np.ndarray, *others: np.ndarray):
    ok = np.isfinite(x).all(axis=1)
    if not ok.all():
        logger.warning(f"{name}: dropping {int((~ok).sum())} non-finite rows of {len(ok)}")
    return (x[ok],) + tuple(o[ok] for o in others)


def _repe_groups(parts: list[str], rows: list[np.ndarray]) -> np.ndarray:
    """One statement id per RepE row (its honest and deceptive versions share it), aligned to `rows`."""
    table = pd.read_csv(C.data_dir("index") / "repe_groups.csv").set_index(["dataset", "pooled_row"])["group"]
    return np.concatenate([table.loc[[(p, int(r)) for r in rs]].to_numpy() for p, rs in zip(parts, rows)])


def load(layer: int, pooling: str = "a", subset: dict[str, np.ndarray] | None = None,
         bc_variant: str = "pilot") -> dict[str, Split]:
    """The five Liars' Bench sets, Alpaca and the RepE pairs (train and val merged) at one layer and pooling.

    `subset` maps a dataset name to the pooled rows to keep (the pilot's 50 per class, the full run's caps);
    others keep all. Poolings (b)/(c) come from `bc_dir(bc_variant)`."""
    s = C.load()
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}")
    if pooling != "a" and layer != 20:
        raise ValueError("poolings (b) and (c) exist at layer 20 only")
    out: dict[str, Split] = {}
    train, val = s["train_pairs"], s["train_pairs"] + "_val"
    for name in [*lb5(), s["control"], train]:
        parts = [train, val] if name == train else [name]
        xs, ys, rows, raw_rows = [], [], [], []
        for k, part in enumerate(parts):
            if pooling == "a":
                with np.load(C.pooled_dir(layer) / f"{part}.npz", allow_pickle=False) as z:
                    x = z["pooled"].astype(np.float32)
                    y = (z["labels"].astype(str) == "DECEPTIVE").astype(int)
                r = np.arange(len(x))
            else:
                with np.load(bc_dir(bc_variant) / f"{part}.npz", allow_pickle=False) as z:
                    x = z[pooling].astype(np.float32)
                    r = z["rows"]
                labels = index(part).set_index("pooled_row").loc[r, "label"].to_numpy()
                y = (labels == "DECEPTIVE").astype(int)
            xs.append(x)
            ys.append(y)
            rows.append(r + k * VAL_OFFSET)
            raw_rows.append(r)
        x, y, r = np.concatenate(xs), np.concatenate(ys), np.concatenate(rows)
        groups = _repe_groups(parts, raw_rows) if name == train else None
        x, y, r, *g = _finite(f"{short(name)} L{layer}{pooling}", x, y, r, *([groups] if groups is not None else []))
        split = Split(name, x, y, r, g[0] if g else None)
        if subset is not None and name in subset:
            split = split.take(np.isin(split.rows, subset[name]))
        out[name] = split
    return out


def pilot_subset(per_class: int | None = None) -> dict[str, np.ndarray]:
    """The pilot's rows: `pilot.n_per_class` of each label per Liars' Bench set (seeded)."""
    from emotion_coords.rows import pick_lb

    n = per_class or C.load()["pilot"]["n_per_class"]
    return {name: pick_lb(index(name), n, seed=C.load()["seeds"]["stories"])["pooled_row"].to_numpy()
            for name in lb5()}


FULL_CAPS = {"insider_trading_cadenza__plain": 500}  # the spec: up to 1,000 per class per set, 500 for IT
FULL_CAP = 1000


def full_subset() -> dict[str, np.ndarray]:
    """The full run's rows: at most 1,000 of each label per Liars' Bench set (500 for IT), drawn with a fixed
    seed; a smaller class is kept whole."""
    from emotion_coords.rows import pick_lb

    return {name: pick_lb(index(name), FULL_CAPS.get(name, FULL_CAP), seed=C.load()["seeds"]["stories"])
            ["pooled_row"].to_numpy() for name in lb5()}


def subset(variant: str) -> dict[str, np.ndarray]:
    return {"pilot": pilot_subset, "full": full_subset}[variant]()


def bc_dir(variant: str = "pilot"):
    return C.data_dir("lb", "L20_bc" if variant == "pilot" else f"L20_bc_{variant}")


def extract_bc(variant: str = "pilot", check_tol: float = 2e-3, overwrite: bool = False) -> None:
    """Poolings (b) and (c) at layer 20 from the per-token cache, for a variant's rows ("pilot" or "full";
    Alpaca and RepE whole), plus the RepE statement groups. Each row's cached span mean is checked against the
    stored pooled (a). Resumable: a set whose file exists is skipped (unless `overwrite`), and the groups are
    written as soon as RepE is done."""
    from emotion_coords.phase0 import _experiment

    cfg, experiment = _experiment()
    model_name = str(experiment.model.name_or_path)
    wanted = subset(variant)
    out_dir = bc_dir(variant)
    groups = []
    for ds in [experiment.train_dataset, *experiment.eval_datasets, *experiment.control_datasets]:
        table = index(ds.name)
        if ds.name.startswith(C.load()["train_pairs"]):
            for r in table.itertuples():
                text = "".join(m.content for m in ds.dialogues[int(r.dataset_row)] if m.role == "assistant")
                groups.append({"dataset": ds.name, "pooled_row": int(r.pooled_row),
                               "group": int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)})
            pd.DataFrame(groups).to_csv(C.data_dir("index") / "repe_groups.csv", index=False)
        if (out_dir / f"{ds.name}.npz").exists() and not overwrite:
            logger.info(f"{ds.name}: (b)/(c) already extracted, skipping")
            continue
        rows = table if ds.name not in wanted else table[table["pooled_row"].isin(wanted[ds.name])]
        b, c, worst = _cache_rows(experiment, cfg, model_name, ds, rows)
        if worst > check_tol:
            raise RuntimeError(f"{ds.name}: cached span mean differs from the stored pooled row by {worst:.2e}")
        np.savez(out_dir / f"{ds.name}.npz", rows=rows["pooled_row"].to_numpy(), b=b.astype(np.float16),
                 c=c.astype(np.float16))
        logger.info(f"{ds.name}: (b)/(c) for {len(rows)} rows; span mean vs stored (a) max rel. error {worst:.1e}")


def _cache_rows(experiment, cfg, model_name: str, ds, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """(b) and (c) at layer 20 for `rows` of one set from the per-token cache, and the largest relative
    difference between a row's cached span mean and its stored pooled (a)."""
    from deception_detection import acts_cache
    from deception_detection.activations import cache_entries
    from deception_detection.tokenized_data import TokenizedDataset

    toks = TokenizedDataset.from_dataset(
        ds, experiment.tokenizer, detect_all=cfg.detect_all_tokens,
        detect_only_start_of_turn=cfg.detect_only_start_of_turn,
        detect_only_last_token=cfg.detect_only_last_token)
    keys, kept_lens = cache_entries(toks, model_name, cfg.detect_layers, cfg.acts_positions)
    with np.load(C.pooled_dir(20) / f"{ds.name}.npz", allow_pickle=False) as z:
        stored = z["pooled"]
    b, c, worst = [], [], 0.0
    for r in rows.itertuples():
        i = int(r.dataset_row)
        acts = acts_cache.load(keys[i], kept_lens[i])
        if acts is None:
            raise RuntimeError(f"{ds.name} row {i}: not in the layer-20 cache")
        acts = acts[:, 0].float().numpy()
        offset = int(r.n_tokens) - acts.shape[0]  # 0 for full-row entries
        span = acts[int(r.first) - offset: int(r.last) + 1 - offset].mean(0)
        ref = stored[int(r.pooled_row)]
        worst = max(worst, float(np.abs(span - ref).max() / np.abs(ref).max()))
        # copies, not views: a view keeps the row's whole (tokens, hidden) array alive (Alpaca: ~18 GB)
        b.append(acts[int(r.last) - offset].copy())
        c.append(acts[int(r.pre) - offset].copy())
    return np.stack(b), np.stack(c), worst


def cache_vectors(wanted: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """(b) and (c) at layer 20 for the given pooled rows of each named set, straight from the cache (not saved)."""
    from emotion_coords.phase0 import _experiment

    cfg, experiment = _experiment()
    model_name = str(experiment.model.name_or_path)
    out = {}
    for ds in [*experiment.eval_datasets, *experiment.control_datasets]:
        if ds.name not in wanted:
            continue
        table = index(ds.name)
        rows = table.set_index("pooled_row").loc[list(wanted[ds.name])].reset_index()
        b, c, worst = _cache_rows(experiment, cfg, model_name, ds, rows)
        out[ds.name] = {"rows": rows["pooled_row"].to_numpy(), "b": b, "c": c, "span_check": worst}
    return out


if __name__ == "__main__":
    fire.Fire({"extract_bc": extract_bc})
