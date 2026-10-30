"""Curing instructed deception before training on it (design: `ms10_cure_spec.py`, frozen before any erasure is
scored). The thesis's raw probe D1 is trained on InstrDec's `ms10` train split after each erasure (fitted on those
600 rows alone) and scored on every test split of the ten datasets with `ms10`'s draws, at layers 36, 40 and 44.
Runs in E:/aletheia/venv from the deception-detection folder.

    python -m emotion_coords.ms10_cure freeze
    python -m emotion_coords.ms10_cure score      # checks the freezes (this spec, ms10_thesis's, ms10's)
    python -m emotion_coords.ms10_cure report
"""

import json
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from emotion_coords import aletheia_topk as TK
from emotion_coords import config as C
from emotion_coords import linalg as la
from emotion_coords import ms10 as M
from emotion_coords import ms10_cure_spec as K
from emotion_coords import ms10_spec as S
from emotion_coords import ms10_thesis as MT
from emotion_coords import probes as P

SEEDED = ("random", "random-label LEACE")
REFERENCE_TOL = 1e-3  # the unerased probe against ms10_thesis's D1: a larger gap is a real discrepancy, not BLAS noise


def out_dir() -> Path:
    return C.results_dir("ms10", "cure")


def freeze() -> str:
    """Record this spec's SHA-256 before any erasure is scored."""
    path = Path(K.__file__)
    line = f"{path.name} sha256 {M._sha256(path)}"
    if K.FREEZE.exists():
        held = K.FREEZE.read_text(encoding="utf-8")
        if line not in held:
            raise RuntimeError(f"{K.FREEZE} holds another design:\n{held}")
        return held
    text = f"{line}\nfrozen {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    K.FREEZE.write_text(text, encoding="utf-8")
    logger.info(f"-> {K.FREEZE}\n{text}")
    return text


def check_freeze() -> str:
    digest = M._sha256(Path(K.__file__))
    if not K.FREEZE.exists() or digest not in K.FREEZE.read_text(encoding="utf-8"):
        raise RuntimeError(f"ms10_cure_spec.py ({digest[:12]}) is not the frozen design ({K.FREEZE})")
    MT.check_freeze()  # ms10_thesis's spec and, through it, ms10's
    logger.info(f"ms10_cure_spec.py sha256 {digest} = the freeze")
    return digest


# --- erasures (fitted on the training rows only) ---------------------------------------------------------------------


def _erase(u: np.ndarray):
    """Orthogonal erasure of the orthonormal columns of `u`."""
    return lambda x, u=u: x - (x @ u) @ u.T


def _keep(u: np.ndarray):
    """The coordinates on the orthonormal columns of `u`, as features."""
    return lambda x, u=u: x @ u


def _whitening(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The training rows' mean, their whitening map w (z = (x - mu) @ w has identity covariance on the row space) and
    its pseudo-inverse w_pinv (z @ w_pinv.T maps back onto the row space)."""
    mu = x.mean(0)
    _, s, vt = np.linalg.svd(x - mu, full_matrices=False)
    keep = s > K.WHITEN_TOL * s[0]
    s, vt = s[keep], vt[keep]
    scale = np.sqrt(len(x) - 1)
    return mu, (vt.T / s) * scale, (vt.T * s) / scale


def _random_label_rounds(z: np.ndarray, rounds: int, seed: int) -> np.ndarray:
    """The whitened directions `rounds` rounds of LEACE on balanced random two-class splits remove, in order. LEACE
    erases the whitened cross-covariance, here the mean difference of the split; the covariance of what is left stays
    the identity, so the next round's difference, taken on the erased rows, is orthogonal to the earlier ones."""
    e = np.zeros((z.shape[1], 0))
    for i in range(rounds):
        labels = np.random.default_rng([K.LABEL_SEED + seed, i]).permutation(np.arange(len(z)) % 2)
        current = z - (z @ e) @ e.T
        g = current[labels == 1].mean(0) - current[labels == 0].mean(0)
        g -= e @ (e.T @ g)
        e = np.column_stack([e, g / np.linalg.norm(g)])
    return e


def _erasers(x: np.ndarray, y: np.ndarray):
    """Every (variant, rank, seed, transform) of the spec, fitted on the training rows. A generator: the large bases
    are built when used."""
    yield "none", 0, 0, lambda v: v
    mu = x.mean(0)
    _, _, total = np.linalg.svd(x - mu, full_matrices=False)
    centred = x.copy()
    for c in (0, 1):
        centred[y == c] -= x[y == c].mean(0)
    _, _, within = np.linalg.svd(centred, full_matrices=False)
    w_mu, w, w_pinv = _whitening(x)
    z = (x - w_mu) @ w
    top = max(K.RANKS)
    leace = [_random_label_rounds(z, top, s) for s in range(K.N_SEEDS)]
    for k in K.RANKS:
        yield "nuisance", k, 0, _erase(within[:k].T)
        yield "total", k, 0, _erase(total[:k].T)
        yield "keep", k, 0, _keep(total[:k].T)
        for s in range(K.N_SEEDS):
            yield "random", k, s, _erase(la.random_subspace(x.shape[1], k, seed=K.RANDOM_SEED + s).T)
            a, b = w @ leace[s][:, :k], w_pinv @ leace[s][:, :k]
            yield "random-label LEACE", k, s, lambda v, a=a, b=b: v - ((v - w_mu) @ a) @ b.T
    yield "keep", K.KEEP_REFERENCE, 0, _keep(total[:K.KEEP_REFERENCE].T)


def _fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The thesis's raw probe (`persona_ood.make_probe`) as a direction over its features (no intercept)."""
    from deception_detection.scripts import persona_ood as PO
    from deception_detection.scripts import transfer_alignment as TA

    return TA.raw_direction(PO.make_probe().fit(x, y))


def _random_label_check(x: np.ndarray) -> list[dict]:
    """The user's stopping test on its first round: the probe on balanced random labels of the training rows, held out
    (mean of 5 folds) and in-sample."""
    from sklearn.model_selection import StratifiedKFold

    out = []
    for s in range(K.N_SEEDS):
        labels = np.random.default_rng([K.LABEL_SEED + s, 0]).permutation(np.arange(len(x)) % 2)
        folds = StratifiedKFold(5, shuffle=True, random_state=s).split(x, labels)
        held = [P.auroc(labels[te], x[te] @ _fit(x[tr], labels[tr])) for tr, te in folds]
        out.append({"seed": s, "held_out_auroc": float(np.mean(held)),
                    "in_sample_auroc": P.auroc(labels, x @ _fit(x, labels))})
    return out


# --- scoring ---------------------------------------------------------------------------------------------------------


def _regions() -> dict[str, list[str]]:
    return {"deception": [t for t in S.DECEPTION if t != K.SOURCE], "new deception": list(MT.NEW),
            "sycophancy": list(S.SYCOPHANCY), "in-distribution": [K.SOURCE]}


def score(check_reference: bool = True) -> None:
    """Every erasure at the three layers on `ms10`'s draws; region means per draw; C1-C4. The unerased probe must give
    `ms10_thesis`'s D1 AUROCs for InstrDec (`check_reference`; off only for a synthetic smoke test)."""
    digest = check_freeze()
    tests = MT._tests()
    reference = pd.read_csv(MT.out_dir() / "auroc.csv")
    reference = reference[(reference["detector"] == "D1") & (reference["train"] == K.SOURCE)]
    reference = reference.set_index(["layer", "test"])["auroc"]
    records, draws, checks, worst = [], {}, [], 0.0
    for layer in K.LAYERS:
        t0 = time.time()
        data = M._block(layer)
        x, y, _ = M._training(data, K.SOURCE)
        x = x.astype(np.float64)
        xt = {n: data[f"x_{n}"][tests[n][5]].astype(np.float64) for n in S.SETS}
        checks += [r | {"layer": layer} for r in _random_label_check(x)]
        for variant, k, seed, f in _erasers(x, y):
            w = _fit(f(x), y)
            cells = {}
            for t in S.SETS:
                yy, pos, neg, up, un, _ = tests[t]
                s = f(xt[t]) @ w
                a = P.auroc(yy, s)
                if variant == "none" and check_reference:
                    worst = max(worst, abs(a - reference[layer, t]))
                    if worst > REFERENCE_TOL:
                        raise RuntimeError(f"L{layer} {t}: unerased {a:.6f} against ms10_thesis's D1 "
                                           f"{reference[layer, t]:.6f}")
                cells[t] = TK._boot_aurocs(up, un, s, pos, neg)
                records.append({"layer": layer, "variant": variant, "k": k, "seed": seed, "test": t, "auroc": a})
            for region, sets in _regions().items():
                draws[layer, variant, k, seed, region] = np.mean([cells[t] for t in sets], axis=0)
        logger.info(f"L{layer}: every erasure scored ({time.time() - t0:.0f} s)")
    auroc = pd.DataFrame(records)
    auroc.to_csv(out_dir() / "auroc.csv", index=False)
    pd.DataFrame(checks).to_csv(out_dir() / "random_label_check.csv", index=False)
    _summarise(auroc, draws, digest, worst if check_reference else None)


def _summarise(auroc: pd.DataFrame, draws: dict, digest: str, worst: float | None) -> None:
    seeds = {v: range(K.N_SEEDS) if v in SEEDED else range(1) for v in K.VARIANTS}
    arms = sorted({(layer, v, k) for layer, v, k, _, _ in draws}, key=lambda a: (a[0], list(K.VARIANTS).index(a[1]), a[2]))
    boot = {(layer, v, k, r): np.mean([draws[layer, v, k, s, r] for s in seeds[v]], axis=0)
            for layer, v, k in arms for r in K.REGIONS}
    cells = auroc.groupby(["layer", "variant", "k", "test"])["auroc"].mean()
    point = {(layer, v, k, r): float(np.mean([cells[layer, v, k, t] for t in sets]))
             for layer, v, k in arms for r, sets in _regions().items()}
    rows = [{"layer": layer, "variant": v, "k": k, "region": r, "mean": point[layer, v, k, r],
             "lo": float(np.quantile(boot[layer, v, k, r], 0.025)), "hi": float(np.quantile(boot[layer, v, k, r], 0.975))}
            for layer, v, k in arms for r in K.REGIONS]
    pd.DataFrame(rows).to_csv(out_dir() / "regions.csv", index=False)

    def over_ranks(store, layer, variant, region, minus=None):
        """The variant minus `minus` (a variant at the same rank, or a fixed (variant, k)), averaged over the ranks."""
        def other(k):
            if minus is None:
                return None
            return store[(layer, minus, k, region) if isinstance(minus, str) else (layer, *minus, region)]
        values = [store[layer, variant, k, region] - (other(k) if minus is not None else 0) for k in K.RANKS]
        return np.mean(values, axis=0)

    comparisons = {"C1": ("none", 0), "C2": "random", "C3": "random-label LEACE", "C4": ("keep", K.KEEP_REFERENCE)}
    verdicts = {"freeze_sha256": digest, "layers": list(K.LAYERS), "min_layers": K.MIN_LAYERS,
                "unerased_vs_ms10_thesis_D1_max_abs_diff": worst}
    deltas = []
    for key, minus in comparisons.items():
        per_layer = {}
        for layer in K.LAYERS:
            for region in K.REGIONS:
                d = over_ranks(boot, layer, "nuisance", region, minus)
                p = float(over_ranks(point, layer, "nuisance", region, minus))
                deltas.append({"prediction": key, "layer": layer, "region": region, "delta": p,
                               "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))})
                if region == "deception":
                    per_layer[layer] = [round(p, 4), round(deltas[-1]["lo"], 4), round(deltas[-1]["hi"], 4)]
        n = sum(v[1] > 0 for v in per_layer.values())
        verdicts[key] = {"minus": minus, "layers_with_ci_above_0": n, "holds": n >= K.MIN_LAYERS, "deltas": per_layer}
    pd.DataFrame(deltas).to_csv(out_dir() / "deltas.csv", index=False)
    (out_dir() / "verdicts.json").write_text(json.dumps(verdicts, indent=1, default=str), encoding="utf-8")
    logger.info(json.dumps(verdicts, indent=1, default=str))


# --- report ----------------------------------------------------------------------------------------------------------


def report() -> Path:
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    regions = pd.read_csv(out_dir() / "regions.csv")
    deltas = pd.read_csv(out_dir() / "deltas.csv")
    verdicts = json.loads((out_dir() / "verdicts.json").read_text(encoding="utf-8"))
    checks = pd.read_csv(out_dir() / "random_label_check.csv")
    lines = ["# Curing instructed deception before training on it (Llama-3.3-70B, layers 36, 40, 44)", "",
             f"Design: `emotion_coords/ms10_cure_spec.py`, frozen (sha256 {verdicts['freeze_sha256'][:12]}) before any "
             "erasure was scored. The thesis's raw probe D1 trained on InstrDec's 600 training rows after each erasure "
             "(fitted on those rows alone), scored on every test split with `ms10`'s draws. The unerased probe "
             f"reproduces `ms10_thesis`'s D1 (max |difference| {verdicts['unerased_vs_ms10_thesis_D1_max_abs_diff']}).",
             "", "## Predictions (deception transfer: the other four deception sets)", ""]
    for key in ("C1", "C2", "C3", "C4"):
        v = verdicts[key]
        cells = "; ".join(f"L{layer} {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" for layer, d in v["deltas"].items())
        lines.append(f"- **{key}** ({K.PREDICTIONS[key]}): {v['layers_with_ci_above_0']} of 3 ({cells}): "
                     f"**{'holds' if v['holds'] else 'fails'}**.")
    lines += ["", "The same comparisons in every region (averaged over the ranks; mean over the layers, and the layers "
              "whose CI is above 0 / below 0):", "", "| comparison | " + " | ".join(K.REGIONS) + " |",
              "|---|" + "---|" * len(K.REGIONS)]
    for key in ("C1", "C2", "C3", "C4"):
        d = deltas[deltas["prediction"] == key]
        cells = []
        for r in K.REGIONS:
            dr = d[d["region"] == r]
            cells.append(f"{dr['delta'].mean():+.3f} ({int((dr['lo'] > 0).sum())} / {int((dr['hi'] < 0).sum())})")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", "## The user's stopping test: a probe on balanced random labels of the 600 training rows", "",
              "| layer | held-out AUROC (mean of 5 seeds) | in-sample AUROC |", "|---|---|---|"]
    for layer, c in checks.groupby("layer"):
        lines.append(f"| {layer} | {f3(c['held_out_auroc'].mean())} (range {f3(c['held_out_auroc'].min())}-"
                     f"{f3(c['held_out_auroc'].max())}) | {f3(c['in_sample_auroc'].mean())} |")
    for region in K.REGIONS:
        lines += ["", f"## {region}: AUROC by rank (mean over the three layers)", "",
                  "| variant | k = 0 | " + " | ".join(str(k) for k in K.RANKS) + f" | k = {K.KEEP_REFERENCE} |",
                  "|---|---|" + "---|" * (len(K.RANKS) + 1)]
        r = regions[regions["region"] == region].groupby(["variant", "k"])["mean"].mean()
        base = r["none", 0]
        for v in K.VARIANTS:
            if v == "none":
                continue
            cells = [f3(base)] + [f3(r[v, k]) if (v, k) in r.index else "-" for k in K.RANKS]
            cells.append(f3(r[v, K.KEEP_REFERENCE]) if (v, K.KEEP_REFERENCE) in r.index else "-")
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
    path = out_dir() / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"-> {path}")
    return path


if __name__ == "__main__":
    fire.Fire({"freeze": freeze, "score": score, "report": report})
