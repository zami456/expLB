"""The persona direction at every token, not just the mean.

Every persona result elsewhere reads one number per example: the direction applied to the mean
over response tokens. The activation caches hold the full unpadded sequence of every example, so
the direction can be scored at every position with no model calls. This asks three things:

  - does mean-pooling throw signal away?  (max, top-10% mean, first and last response token)
  - where in the response does the signal live?  (a profile over relative and absolute position)
  - is the model's state *before* it replies already predictive?  (the token just before the
    response, the end of the assistant header -- the on-policy version of a self-evaluation's
    "decision point")

`project` walks a layer's cache once and stores, per example, the projection of every token onto
each unit persona axis. `axis_score` is a plain dot product, so the mean of those projections over
the response is exactly the projection of the mean-pooled vector: `verify` checks the mean read
against the committed zero-shot AUROCs before anything else is trusted.

    python deception_detection/scripts/persona_positions.py project \\
        --config-file deception_detection/scripts/configs/XXXX-9-llama-all.yaml --layer 20 \\
        --vectors "persona including knowlede denial" --out_dir activations/token_proj_L20
    python deception_detection/scripts/persona_positions.py verify \\
        --proj_dir activations/token_proj_L20 \\
        --zero_shot results/persona_ood_ext_L20/zero_shot.csv
    python deception_detection/scripts/persona_positions.py analyze \\
        --proj_dir activations/token_proj_L20 --out_dir results/persona_positions_L20
    python deception_detection/scripts/persona_positions.py figures \\
        --out_dir results/persona_positions_L20
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import numpy as np
import pandas as pd
import yaml

from deception_detection import persona_pcs
from deception_detection.experiment import Experiment, ExperimentConfig
from deception_detection.log import logger
from deception_detection.scripts.persona_ood import (
    CONTROL_DATASET,
    MERGE_INTO,
    SHARED_DATASETS,
    auroc,
    bootstrap_delta,
    load_pooled,
    operating_point,
)
from deception_detection.scripts.pool_activations import iter_cached_examples
from deception_detection.utils import REPO_ROOT

# The four zero-shot axes, keyed by the name `PersonaBank.axis_unit` takes, with the column the
# committed zero_shot.csv stores each under -- the pairing `verify` checks against.
AXES = {
    "extended": "auroc_ext_axis",
    "denial": "auroc_denial_axis",
    "deception": "auroc_axis",
    "sycophancy": "auroc_syc_axis",
}
# Reads that reduce a response to one score; "mean" is the existing method.
READS = ("mean", "max", "top10", "first", "last", "pre")
N_BINS = 10  # relative-position bins over a response
N_FIRST = 16  # absolute positions from the start of a response
BOOTSTRAP_AXIS = "extended"


# --- per-example reads -----------------------------------------------------------------------


def row_reads(proj: np.ndarray, detect: np.ndarray) -> dict[str, float]:
    """Reduce one example's per-token projections (one axis) to each read.

    `proj` covers every stored position of the row and `detect` marks its response tokens.
    "pre" is the position just before the first response token -- with the chat templates
    here, the end of the assistant header, i.e. the model's state as it starts to reply.
    """
    response = proj[detect]
    first = int(np.flatnonzero(detect)[0])
    k = max(1, int(np.ceil(0.10 * len(response))))
    return {
        "mean": float(response.mean()),
        "max": float(response.max()),
        "top10": float(np.sort(response)[-k:].mean()),
        "first": float(response[0]),
        "last": float(response[-1]),
        "pre": float(proj[first - 1]) if first >= 1 else float("nan"),
    }


def row_bins(proj: np.ndarray, detect: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Mean projection in each relative-position bin of the response; all NaN if too short.

    Only responses with at least `n_bins` tokens are binned, so every bin of every included row
    is populated and the bins compare like with like. Shorter responses are left out of the
    relative profile rather than smeared across it.
    """
    response = proj[detect]
    if len(response) < n_bins:
        return np.full(n_bins, np.nan)
    which = np.floor(np.arange(len(response)) * n_bins / len(response)).astype(int)
    return np.array([response[which == b].mean() for b in range(n_bins)])


def row_first(proj: np.ndarray, detect: np.ndarray, n_first: int = N_FIRST) -> np.ndarray:
    """Projection at response tokens 0 .. n_first-1; NaN past the end of a short response."""
    response = proj[detect]
    out = np.full(n_first, np.nan)
    out[: min(n_first, len(response))] = response[:n_first]
    return out


# --- ragged storage --------------------------------------------------------------------------


@dataclass
class TokenProjections:
    """One dataset's per-token projections, stored ragged: row i spans offsets[i]:offsets[i+1]."""

    name: str
    layer: int
    axis_names: list[str]
    proj: np.ndarray  # [total_tokens, n_axes] float32
    detect: np.ndarray  # [total_tokens] bool
    offsets: np.ndarray  # [n_rows + 1] int64
    labels: np.ndarray  # [n_rows] str

    def row(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        span = slice(int(self.offsets[i]), int(self.offsets[i + 1]))
        return self.proj[span], self.detect[span]

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".npz.tmp")
        with open(tmp, "wb") as f:  # np.savez_compressed appends .npz to a bare path
            np.savez_compressed(
                f,
                name=self.name,
                layer=self.layer,
                axis_names=np.array(self.axis_names),
                proj=self.proj,
                detect=self.detect,
                offsets=self.offsets,
                labels=self.labels,
            )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "TokenProjections":
        raw = np.load(path, allow_pickle=False)
        return cls(
            name=str(raw["name"]),
            layer=int(raw["layer"]),
            axis_names=[str(a) for a in raw["axis_names"]],
            proj=raw["proj"],
            detect=raw["detect"],
            offsets=raw["offsets"],
            labels=raw["labels"].astype(str),
        )


def merge(parts: list[TokenProjections]) -> TokenProjections:
    """Concatenate datasets row-wise, e.g. RepE's train and validation splits."""
    first = parts[0]
    lengths = np.concatenate([np.diff(p.offsets) for p in parts])
    return TokenProjections(
        name=first.name,
        layer=first.layer,
        axis_names=first.axis_names,
        proj=np.concatenate([p.proj for p in parts]),
        detect=np.concatenate([p.detect for p in parts]),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        labels=np.concatenate([p.labels for p in parts]),
    )


def load_projections(proj_dir: Path) -> dict[str, TokenProjections]:
    """Every dataset in `proj_dir`, RepE's splits merged as the pooled experiments merge them."""
    parts: dict[str, list[TokenProjections]] = {}
    for path in sorted(Path(proj_dir).glob("*.npz")):
        tp = TokenProjections.load(path)
        target = MERGE_INTO.get(tp.name, tp.name)
        parts.setdefault(target, []).append(tp)
    merged = {}
    for name, group in parts.items():
        tp = merge(group) if len(group) > 1 else group[0]
        tp.name = name
        merged[name] = tp
    return merged


# --- commands --------------------------------------------------------------------------------


def project(
    config_file: str | Path = "deception_detection/scripts/configs/XXXX-9-llama-all.yaml",
    layer: int = 20,
    vectors: str | Path = persona_pcs.EXTENDED_VECTORS_DIR,
    out_dir: str | Path = REPO_ROOT / "activations" / "token_proj_L20",
    overwrite: bool = False,
    **kwargs: dict[str, Any],
) -> None:
    """Project every cached token onto the unit persona axes, one ragged `.npz` per dataset.

    `layer` must be the config's detect layer: the persona directions live in the residual
    stream at one layer, and pairing them with another layer's activations yields numbers, not
    an error. That is checked here rather than trusted.
    """
    config_file, out_dir = Path(config_file), Path(out_dir)
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)
    config_dict |= kwargs
    cfg = ExperimentConfig(**config_dict)  # type: ignore[arg-type]
    if cfg.detect_layers != [layer]:
        raise ValueError(f"config extracts layers {cfg.detect_layers}, but --layer is {layer}")

    experiment = Experiment(cfg)
    model_name = str(experiment.model.name_or_path)
    bank = persona_pcs.load_bank_at_layer(layer, vectors)
    axis_names = list(AXES)
    units = np.stack([bank.axis_unit(a) for a in axis_names]).astype(np.float32)  # [4, hidden]
    logger.info(f"projecting layer {layer} onto {axis_names}; bank {vectors}")

    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [experiment.train_dataset, *experiment.eval_datasets, *experiment.control_datasets]
    for dataset in datasets:
        if len(dataset) == 0:
            continue
        path = out_dir / f"{dataset.name}.npz"
        if path.exists() and not overwrite:
            logger.info(f"{dataset.name}: already projected, skipping")
            continue

        projs, detects, lengths, labels = [], [], [], []
        for example in iter_cached_examples(
            dataset, experiment, model_name, cfg.detect_layers, desc="projecting"
        ):
            acts = example.acts
            assert acts.shape[1] == 1, f"expected one layer in the cache entry, got {acts.shape}"
            projs.append(acts[:, 0, :].float().numpy() @ units.T)
            detects.append(example.mask.numpy())
            lengths.append(acts.shape[0])
            labels.append(dataset.labels[example.index].name)

        TokenProjections(
            name=dataset.name,
            layer=layer,
            axis_names=axis_names,
            proj=np.concatenate(projs).astype(np.float32),
            detect=np.concatenate(detects).astype(bool),
            offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
            labels=np.array(labels),
        ).save(path)
        logger.info(f"{dataset.name}: {len(labels)} rows, {sum(lengths)} tokens -> {path.name}")


def _scores(
    tp: TokenProjections, axis: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Per-row reads, relative bins and first-token profile for one axis, honest/deceptive only."""
    keep = np.isin(tp.labels, ["HONEST", "DECEPTIVE"])
    rows = np.flatnonzero(keep)
    reads, bins, first = [], [], []
    for i in rows:
        proj, detect = tp.row(int(i))
        p = proj[:, axis].astype(np.float64)
        reads.append(row_reads(p, detect))
        bins.append(row_bins(p, detect))
        first.append(row_first(p, detect))
    y = (tp.labels[rows] == "DECEPTIVE").astype(int)
    return pd.DataFrame(reads), np.array(bins), np.array(first), y


def verify(
    proj_dir: str | Path,
    zero_shot: str | Path,
    pooled_dir: str | Path | list | None = None,
    tol_auroc: float = 1e-5,
    tol_score: float = 1e-4,
    vectors: str | Path = persona_pcs.EXTENDED_VECTORS_DIR,
    min_corr: float | None = None,
) -> None:
    """The mean read must reproduce the committed results; nothing else is trusted until it does.

    Two checks. Against the committed `zero_shot.csv`, AUROC must agree to `tol_auroc`. That
    cannot be exact: per-token projections and the pooled vector round differently in float32,
    and at deeper layers, where activations are larger, that is enough to swap a few near-tied
    examples (measured: 1e-16 at layer 20, 1.9e-6 at 40, 3.7e-6 at 44). So with `pooled_dir`
    the decisive check runs too: every example's mean read must equal its pooled projection to
    within `tol_score` of the score's spread, with labels aligned row by row. Misaligned tokens
    or rows would miss that by orders of magnitude, not by rounding.

    Another model's projections were taken on the server (`layer_sweep.project`), in a second pass,
    and cannot match example by example. The server's float32 product runs in TF32, which rounds
    each axis to 10 mantissa bits: a near-constant offset (up to 1.6% of the score's spread on
    Qwen, whose activations have very large channels). The pass batches rows differently from the
    pooling pass, and over long transcripts (Qwen's HP-KR, insider trading) bf16 then moves some
    examples by up to 20% of the spread; neither pass is the exact one. For such reads pass
    `min_corr`: every dataset's mean reads must correlate with the pooled projections at least
    that well across examples. Measured on Mistral and Qwen at their key blocks: >= 0.9993
    aligned, <= 0.51 with the rows off by one; Gemma >= 0.9989. Pass also `tol_auroc=5e-3`, its
    bank as `vectors` and its folders as `pooled_dir`: AUROCs moved by up to 7e-5 on Mistral, 5e-4
    on Qwen and 2.6e-3 on Gemma, whose scores sit ~40 spreads from zero (on its convince-game rows
    ~30,000 against a spread of ~725), so a few rows' 1-2% differences move them most of a spread.
    The largest and median gaps are logged either way.
    """
    projections = load_projections(Path(proj_dir))
    reference = pd.read_csv(zero_shot).set_index("dataset")
    pooled = (load_pooled([Path(p) for p in pooled_dir] if isinstance(pooled_dir, (list, tuple))
                          else Path(pooled_dir)) if pooled_dir is not None else {})
    bank = None
    rows = []
    for name, tp in sorted(projections.items()):
        if name == CONTROL_DATASET or name not in reference.index:
            continue
        if pooled and bank is None:
            bank = persona_pcs.load_bank_at_layer(tp.layer, vectors)
        for a, axis in enumerate(tp.axis_names):
            reads, _, _, y = _scores(tp, a)
            mean_read = reads["mean"].to_numpy()
            row = {"dataset": name, "axis": axis, "mean_read": auroc(y, mean_read),
                   "zero_shot": float(reference.loc[name, AXES[axis]]), "n": len(y)}
            row["delta"] = row["mean_read"] - row["zero_shot"]
            if name in pooled:
                assert bank is not None
                ref = pooled[name].z @ bank.axis_unit(axis)
                if len(ref) != len(mean_read) or not (pooled[name].y == y).all():
                    raise AssertionError(f"{name}: rows or labels differ from the pooled file")
                gaps = np.abs(mean_read - ref) / ref.std()
                row["score_gap"] = float(gaps.max())
                row["score_gap_median"] = float(np.median(gaps))
                row["corr"] = float(np.corrcoef(mean_read, ref)[0, 1])
            rows.append(row)
    table = pd.DataFrame(rows)
    logger.info(f"\n{table.round(6).to_string(index=False)}")
    worst = float(table["delta"].abs().max())
    logger.info(f"largest |AUROC delta| = {worst:.2e} over {len(table)} dataset x axis cells")
    if worst > tol_auroc:
        raise AssertionError(f"mean read disagrees with zero_shot.csv by {worst:.2e} > {tol_auroc}")
    if "score_gap" in table:
        gap = float(table["score_gap"].max())
        logger.info(f"largest per-example score gap = {gap:.1e} of the score's spread")
        if min_corr is not None:
            corr = float(table["corr"].min())
            logger.info(f"lowest correlation with the pooled projections = {corr:.5f}")
            if corr < min_corr:
                raise AssertionError(f"mean reads correlate with the pooled projections at {corr:.5f} < {min_corr}")
        elif gap > tol_score:
            raise AssertionError(f"per-example scores disagree with pooled by {gap:.1e}")


def analyze(
    proj_dir: str | Path,
    out_dir: str | Path,
    n_boot: int = 1000,
    fpr: float = 0.01,
) -> None:
    """Score every read, the position profiles, and each read against the mean."""
    proj_dir, out_dir = Path(proj_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    projections = load_projections(proj_dir)
    control = projections[CONTROL_DATASET]
    layer = control.layer
    names = [n for n in sorted(projections) if n != CONTROL_DATASET]

    read_rows, profile_rows, boot_rows = [], [], []
    for a, axis in enumerate(control.axis_names):
        control_reads, _, _, _ = _scores(control, a)  # the control is all honest
        thresholds = {r: float(np.nanquantile(control_reads[r], 1 - fpr)) for r in READS}
        for name in names:
            reads, bins, first, y = _scores(projections[name], a)
            for read in READS:
                s = reads[read].to_numpy()
                ok = np.isfinite(s)
                point = operating_point(y[ok], s[ok], thresholds[read])
                read_rows.append({
                    "layer": layer, "axis": axis, "read": read, "dataset": name,
                    "n": int(ok.sum()), "n_deceptive": int(y[ok].sum()),
                    "auroc": auroc(y[ok], s[ok]),
                    "balacc": point["balacc"], "recall": point["recall"],
                })
                if axis == BOOTSTRAP_AXIS and read != "mean":
                    base = reads["mean"].to_numpy()
                    both = ok & np.isfinite(base)
                    d, lo, hi = bootstrap_delta(y[both], s[both], base[both], n_boot=n_boot)
                    boot_rows.append({"layer": layer, "axis": axis, "read": read,
                                      "dataset": name, "delta_vs_mean": d,
                                      "ci_low": lo, "ci_high": hi,
                                      "excludes_zero": bool(lo > 0 or hi < 0)})
            for kind, grid in (("relative", bins), ("first_tokens", first)):
                for pos in range(grid.shape[1]):
                    col = grid[:, pos]
                    ok = np.isfinite(col)
                    profile_rows.append({
                        "layer": layer, "axis": axis, "dataset": name, "kind": kind,
                        "position": pos, "n": int(ok.sum()),
                        "auroc": auroc(y[ok], col[ok]) if ok.sum() >= 10 else float("nan"),
                    })

    reads_table = pd.DataFrame(read_rows)
    reads_table.round(4).to_csv(out_dir / "reads.csv", index=False)
    pd.DataFrame(profile_rows).round(4).to_csv(out_dir / "position_profile.csv", index=False)
    pd.DataFrame(boot_rows).round(4).to_csv(out_dir / "read_vs_mean.csv", index=False)

    shared = [n for n in names if n in SHARED_DATASETS]
    headline = (
        reads_table[reads_table["dataset"].isin(shared)]
        .groupby(["axis", "read"])[["auroc", "balacc", "recall"]].mean()
        .reset_index()
    )
    headline.insert(0, "layer", layer)
    headline.round(4).to_csv(out_dir / "reads_summary.csv", index=False)
    logger.info(f"mean over the {len(shared)} shared datasets:\n"
                f"{headline.round(4).to_string(index=False)}")
    logger.info(f"wrote reads, position profiles and bootstrap tables to {out_dir}")


def figures(out_dir: str | Path, axis: str = BOOTSTRAP_AXIS) -> None:
    """Position profile as small multiples, and the reads as a heatmap centred on chance."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    from deception_detection.scripts.persona_ood_figures import (
        DIVERGING, INK, INK_MUTED, NEUTRAL, SERIES, SHORT, SURFACE,
    )

    out_dir = Path(out_dir)
    reads = pd.read_csv(out_dir / "reads.csv")
    profile = pd.read_csv(out_dir / "position_profile.csv")
    reads, profile = reads[reads["axis"] == axis], profile[profile["axis"] == axis]
    layer = int(reads["layer"].iloc[0])
    names = sorted(reads["dataset"].unique())

    # --- reads heatmap: rows are reads, columns datasets; polarity around chance ---
    grid = reads.pivot(index="read", columns="dataset", values="auroc").loc[list(READS), names]
    fig, ax = plt.subplots(figsize=(8.6, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    image = ax.imshow(grid.to_numpy(float), cmap=DIVERGING,
                      norm=TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0), aspect="auto")
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            v = grid.iat[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if abs(v - 0.5) > 0.3 else INK)
    labels = {"mean": "mean (existing)", "max": "max", "top10": "top-10% mean",
              "first": "first response token", "last": "last response token",
              "pre": "before the reply"}
    ax.set_yticks(range(len(READS)), [labels[r] for r in READS], fontsize=9, color=INK)
    ax.set_xticks(range(len(names)), [SHORT.get(n, n) for n in names], fontsize=9, color=INK)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("zero-shot AUROC  (below 0.5 = inverted)", fontsize=8.5, color=INK_MUTED)
    bar.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    bar.outline.set_visible(False)
    ax.set_title(f"Where to read the persona direction  (layer {layer}, {axis} axis)",
                 fontsize=11.5, color=INK, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(out_dir / "reads.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # --- position profile: one panel per dataset, a single line each ---
    kinds = (("relative", "position in the response (tenths)"),
             ("first_tokens", "response token index"))
    # sharex by row: a dataset with no binnable responses (HP-C, one token) keeps the row's
    # scale instead of falling back to a default range of its own
    fig, axes = plt.subplots(2, len(names), figsize=(2.3 * len(names), 4.6),
                             facecolor=SURFACE, sharey=True, sharex="row", squeeze=False)
    mean_auroc = reads[reads["read"] == "mean"].set_index("dataset")["auroc"]
    for row, (kind, xlabel) in enumerate(kinds):
        for col, name in enumerate(names):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            sub = profile[(profile["dataset"] == name) & (profile["kind"] == kind)]
            ok = sub["auroc"].notna()
            ax.axhspan(0.0, 0.5, color=NEUTRAL, zorder=0)
            ax.axhline(0.5, color=INK_MUTED, lw=0.8, zorder=1)
            ax.axhline(mean_auroc.get(name, np.nan), color=INK_MUTED, lw=0.9,
                       ls=(0, (4, 3)), zorder=2)
            if ok.any():
                ax.plot(sub.loc[ok, "position"], sub.loc[ok, "auroc"], color=SERIES[0],
                        lw=2, marker="o", ms=3.2, zorder=3)
            else:
                ax.text(0.5, 0.72, "responses\ntoo short", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color=INK_MUTED)
            ax.set_ylim(0.0, 1.02)
            if row == 0:
                ax.set_title(SHORT.get(name, name), fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel("AUROC", fontsize=8.5, color=INK_MUTED)
            ax.tick_params(labelsize=7.5, colors=INK_MUTED, length=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(NEUTRAL)
        axes[row, len(names) // 2].set_xlabel(xlabel, fontsize=8.5, color=INK_MUTED)
    fig.suptitle(f"Where in the response the persona direction separates lies  "
                 f"(layer {layer}, {axis} axis; dashed = mean read, shaded = inverted)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "position_profile.png", dpi=200, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote reads.png and position_profile.png to {out_dir}")


if __name__ == "__main__":
    fire.Fire({"project": project, "verify": verify, "analyze": analyze, "figures": figures})
