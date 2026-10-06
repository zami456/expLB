"""Persona-anchored attention pooling: the persona direction reads, a learned query decides where.

Every persona result so far scores an example by the persona direction v applied to the MEAN of
its answer tokens. Here the pooling is an attention over those tokens whose query starts at v and
learns a correction w, while the readout stays v:

    logits_t = v . x_t / sqrt(d) + w . x_t        a = softmax over the example's answer tokens
    score    = sum_t a_t (v . x_t)

This is the key map anchored at identity, W_k = I + U V^T: with a single query v its effective
query is v + V(U^T v), one free vector, so it is fitted as w with a ridge penalty that anchors it
to 0. A penalty of infinity keeps the identity start (logits v.x/sqrt(d), which at these scales is
the mean read); a small penalty lets the query go anywhere. Either way the score is a weighted
average of the example's own per-token persona scores, so the probe can select tokens but cannot
invent a feature.

Softmax ignores a shift shared by all of an answer's tokens, so the gradient keeps w in the span
of the training set's within-answer-centred tokens. w is fitted in that span's top K principal
components (standardised, so the ridge treats each alike), which makes CPU training cheap.

The query needs labels, so this is a trained arm and runs in the transfer-matrix protocol: fit on
one dataset, score every other one. The penalty is chosen on a held-out fifth of the training set,
never on a test set. HP-C answers are one token, where attention cannot act, so it is not a
training set. The answer tokens are read once from the activation cache into int16 memory maps
of the cache's own bfloat16 bits.

    python deception_detection/scripts/persona_attention.py extract --layer 40
    python deception_detection/scripts/persona_attention.py check --layer 40
    python deception_detection/scripts/persona_attention.py run --layer 40
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from deception_detection import persona_pcs as pcs
from deception_detection.log import logger
from deception_detection.utils import REPO_ROOT

TOKENS_ROOT = Path(os.environ.get("ANSWER_TOKENS_DIR", "E:/liars-bench-acts/answer_tokens"))
CACHES = {20: REPO_ROOT / "activations" / "cache", 40: Path("E:/liars-bench-acts/cache"),
          44: Path("E:/liars-bench-acts/cache")}
CONFIGS = {20: "XXXX-9-llama-all.yaml", 40: "XXXX-9-llama-all-L40.yaml", 44: "XXXX-9-llama-all-L44.yaml"}
AXES = {"D9 extended (12v13)": "extended", "D8 knowledge-denial (4v6)": "denial"}
CONTROL = "XXXX-20__plain"
MERGE = {"repe_honesty__plain_val": "repe_honesty__plain"}  # as persona_ood merges them
SHORT = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C",
         "harm_refusal_knowledge_report__plain": "HP-KR", "insider_trading_cadenza__plain": "IT",
         "instruct_dishonesty__plain": "ID", "repe_honesty__plain": "RepE"}
TRAIN_SETS = ("convince_me__plain", "harm_refusal_knowledge_report__plain",
              "insider_trading_cadenza__plain", "instruct_dishonesty__plain", "repe_honesty__plain")
K = 512
LAMBDAS = (1.0, 1e-1, 1e-2, 1e-3, 1e-4)
SEED = 0
CHUNK = 8192
IDENTICAL = "I do not know the answer."


# --- answer tokens ---------------------------------------------------------------------------------

@dataclass
class AnswerTokens:
    """One dataset's answer-token activations; example i spans tokens offsets[i]:offsets[i+1]."""

    name: str
    bits: np.ndarray  # [tokens, hidden] int16: the cache's bfloat16 bits
    offsets: np.ndarray
    labels: np.ndarray

    @property
    def y(self) -> np.ndarray:
        return (self.labels == "DECEPTIVE").astype(int)

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def example(self) -> np.ndarray:
        """The example each token belongs to."""
        return np.repeat(np.arange(len(self.labels)), self.lengths)

    def rows(self, start: int, stop: int) -> np.ndarray:
        block = torch.from_numpy(np.array(self.bits[start:stop]))  # a writable copy of the memmap rows
        return block.view(torch.bfloat16).float().numpy()

    def project(self, matrix: np.ndarray) -> np.ndarray:
        """Every token times `matrix` [hidden, m], streamed."""
        out = np.empty((len(self.bits), matrix.shape[1]), dtype=np.float32)
        for a in range(0, len(self.bits), CHUNK):
            out[a:a + CHUNK] = self.rows(a, a + CHUNK) @ matrix
        return out


def _part(layer: int, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folder = TOKENS_ROOT / f"L{layer}"
    meta = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
    bits = np.memmap(folder / f"{name}.bits", dtype=np.int16, mode="r",
                     shape=(meta["n_tokens"], meta["hidden"]))
    return bits, np.array(meta["offsets"], dtype=np.int64), np.array(meta["labels"])


def load(layer: int, name: str) -> AnswerTokens:
    """A dataset's answer tokens, with any split `MERGE` folds into it appended (in memory)."""
    parts = [_part(layer, name)] + [_part(layer, k) for k, v in MERGE.items() if v == name]
    if len(parts) == 1:
        return AnswerTokens(name, *parts[0])
    bits = np.concatenate([np.asarray(b) for b, _, _ in parts])
    offsets, shift = [np.zeros(1, dtype=np.int64)], 0
    for b, o, _ in parts:
        offsets.append(o[1:] + shift)
        shift += len(b)
    return AnswerTokens(name, bits, np.concatenate(offsets), np.concatenate([lab for _, _, lab in parts]))


def extract(layer: int = 40) -> None:
    """Walk the layer's activation cache once and keep each example's answer tokens."""
    import yaml

    os.environ["ACTS_CACHE_DIR"] = str(CACHES[layer])
    from deception_detection.experiment import Experiment, ExperimentConfig
    from deception_detection.scripts.pool_activations import iter_cached_examples

    config = REPO_ROOT / "deception_detection" / "scripts" / "configs" / CONFIGS[layer]
    cfg = ExperimentConfig(**yaml.safe_load(config.read_text(encoding="utf-8")))
    if cfg.detect_layers != [layer]:
        raise ValueError(f"{config.name} extracts layers {cfg.detect_layers}, not {layer}")
    experiment = Experiment(cfg)
    model_name = str(experiment.model.name_or_path)
    out = TOKENS_ROOT / f"L{layer}"
    out.mkdir(parents=True, exist_ok=True)
    for dataset in [experiment.train_dataset, *experiment.eval_datasets, *experiment.control_datasets]:
        if len(dataset) == 0 or (out / f"{dataset.name}.json").exists():
            continue
        partial = out / f"{dataset.name}.partial"
        offsets, labels, hidden = [0], [], None
        with open(partial, "wb") as f:
            for example in iter_cached_examples(dataset, experiment, model_name, cfg.detect_layers,
                                                desc="answer tokens"):
                tokens = example.acts[:, 0, :][example.mask]
                if tokens.dtype != torch.bfloat16:
                    raise TypeError(f"expected a bfloat16 cache, got {tokens.dtype}")
                hidden = tokens.shape[1]
                f.write(tokens.contiguous().view(torch.int16).numpy().tobytes())
                offsets.append(offsets[-1] + tokens.shape[0])
                labels.append(dataset.labels[example.index].name)
        os.replace(partial, out / f"{dataset.name}.bits")
        meta = {"name": dataset.name, "layer": layer, "hidden": hidden, "n_tokens": offsets[-1],
                "offsets": offsets, "labels": labels}
        (out / f"{dataset.name}.json").write_text(json.dumps(meta), encoding="utf-8")
        logger.info(f"{dataset.name}: {len(labels)} examples, {offsets[-1]} answer tokens")


def _axes(layer: int) -> np.ndarray:
    """The unit persona directions, [hidden, len(AXES)]."""
    bank = pcs.load_bank_at_layer(layer, pcs.EXTENDED_VECTORS_DIR)
    return np.stack([bank.axis_unit(a) for a in AXES.values()], axis=1).astype(np.float32)


def check(layer: int = 40) -> None:
    """The extracted tokens must reproduce the committed per-token projections exactly."""
    from deception_detection.scripts.persona_positions import load_projections

    reference = load_projections(REPO_ROOT / "activations" / f"token_proj_L{layer}")
    axes = _axes(layer)
    for name, ref in reference.items():
        tokens = load(layer, name)
        mine = tokens.project(axes)
        theirs = ref.proj[ref.detect][:, [list(ref.axis_names).index(a) for a in AXES.values()]]
        if mine.shape != theirs.shape or not (tokens.labels == ref.labels).all():
            raise RuntimeError(f"{name}: tokens or labels differ from token_proj")
        err = np.abs(mine - theirs).max() / theirs.std()
        logger.info(f"{name}: {len(tokens.labels)} examples, {len(mine)} answer tokens; "
                    f"max |projection difference| = {err:.1e} of its spread")
        if err > 1e-3:
            raise RuntimeError(f"{name}: projections disagree ({err:.2e})")


# --- the model -----------------------------------------------------------------------------------

def segment_softmax(logits: torch.Tensor, example: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of `logits` within each example's tokens."""
    example = example.long()  # numpy's default int is 32-bit on Windows; scatter needs int64
    top = torch.full((n,), -torch.inf, dtype=logits.dtype).scatter_reduce(0, example, logits, "amax")
    e = torch.exp(logits - top[example])
    return e / torch.zeros(n, dtype=logits.dtype).index_add(0, example, e)[example]


def pool(s: torch.Tensor, correction: torch.Tensor, example: torch.Tensor, n: int, scale: float) -> torch.Tensor:
    """sum_t a_t s_t with a = softmax(s / scale + correction) over each example's tokens."""
    a = segment_softmax(s / scale + correction, example, n)
    return torch.zeros(n, dtype=s.dtype).index_add(0, example.long(), a * s)


def fit(z: np.ndarray, s: np.ndarray, example: np.ndarray, y: np.ndarray, lam: float,
        scale: float, max_iter: int = 200) -> np.ndarray:
    """The query correction u (in standardised PC coordinates) under a ridge penalty `lam`."""
    n = len(y)
    zt, st = torch.from_numpy(z.astype(np.float64)), torch.from_numpy(s.astype(np.float64))
    ex, yt = torch.from_numpy(example), torch.from_numpy(y.astype(np.float64))
    weight = torch.where(yt == 1, 0.5 / yt.mean(), 0.5 / (1 - yt.mean()))  # balanced classes
    with torch.no_grad():
        base = pool(st, torch.zeros_like(st), ex, n, scale)
        mu, sd = base.mean(), base.std() + 1e-12
    u = torch.zeros(z.shape[1], dtype=torch.float64, requires_grad=True)
    alpha = torch.ones((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([u, alpha, bias], lr=1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        score = pool(st, zt @ u, ex, n, scale)
        logit = alpha * (score - mu) / sd + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt, weight=weight, reduction="sum")
        loss = loss / weight.sum() + lam * (u ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return u.detach().numpy()


def within_pcs(tokens: AnswerTokens, k: int = K) -> tuple[np.ndarray, np.ndarray, float]:
    """Top-k principal directions of the answer tokens after removing each answer's own mean,
    their variances, and the share of the within-answer variance they hold."""
    from scipy.linalg import eigh

    hidden = tokens.bits.shape[1]
    cov = np.zeros((hidden, hidden), dtype=np.float64)
    starts, n, i = tokens.offsets, len(tokens.labels), 0
    while i < n:
        j = i + 1
        while j < n and starts[j + 1] - starts[i] <= CHUNK:
            j += 1
        x = tokens.rows(starts[i], starts[j])
        lengths = np.diff(starts[i:j + 1])
        means = np.add.reduceat(x, starts[i:j] - starts[i], axis=0) / lengths[:, None]
        xc = x - np.repeat(means, lengths, axis=0)
        cov += xc.T @ xc
        i = j
    values, vectors = eigh(cov, subset_by_index=[hidden - k, hidden - 1])
    order = np.argsort(values)[::-1]
    dof = max(int(tokens.offsets[-1]) - n, 1)  # each answer's own mean costs one degree of freedom
    return vectors[:, order].astype(np.float32), values[order] / dof, float(values.sum() / np.trace(cov))


# --- the run -------------------------------------------------------------------------------------

def _scores(proj_s: np.ndarray, proj_w: np.ndarray, tokens: AnswerTokens, scale: float) -> np.ndarray:
    ex = torch.from_numpy(tokens.example)
    return pool(torch.from_numpy(proj_s.astype(np.float64)), torch.from_numpy(proj_w.astype(np.float64)),
                ex, len(tokens.labels), scale).numpy()


def _split(y: np.ndarray, seed: int = SEED) -> np.ndarray:
    """A stratified fifth of the examples, held out to choose the penalty."""
    rng, held = np.random.default_rng(seed), np.zeros(len(y), bool)
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        held[rng.choice(idx, size=max(1, len(idx) // 5), replace=False)] = True
    return held


def _subset(tokens_example: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The tokens of the kept examples, and their example ids renumbered from 0."""
    mask = keep[tokens_example]
    _, renumbered = np.unique(tokens_example[mask], return_inverse=True)
    return mask, renumbered


def run(layer: int = 40, k: int = K) -> None:
    out = REPO_ROOT / "results" / f"persona_attention_L{layer}"
    out.mkdir(parents=True, exist_ok=True)
    axes = _axes(layer)
    hidden = axes.shape[0]
    scale = float(np.sqrt(hidden))
    names = list(SHORT) + [CONTROL]
    data = {name: load(layer, name) for name in names}

    # --- fit a query per training set, persona axis and penalty
    models = []  # (train set, axis, penalty or "selected", w in activation space)
    for train in TRAIN_SETS:
        tokens = data[train]
        basis, variances, share = within_pcs(tokens, k)
        logger.info(f"{SHORT[train]}: top {k} within-answer PCs hold {share:.1%} of the within-answer variance")
        proj = tokens.project(np.concatenate([axes, basis], axis=1))
        z = proj[:, axes.shape[1]:] / np.sqrt(np.maximum(variances, 1e-12))[None, :].astype(np.float32)
        y, example = tokens.y, tokens.example
        held = _split(y)
        fit_mask, fit_ex = _subset(example, ~held)
        val_mask, val_ex = _subset(example, held)
        for j, axis in enumerate(AXES):
            s = proj[:, j]
            validation = {}
            for lam in LAMBDAS:
                u = fit(z[fit_mask], s[fit_mask], fit_ex, y[~held], lam, scale)
                val = pool(torch.from_numpy(s[val_mask].astype(np.float64)),
                           torch.from_numpy((z[val_mask] @ u).astype(np.float64)),
                           torch.from_numpy(val_ex), int(held.sum()), scale).numpy()
                validation[lam] = roc_auc_score(y[held], val)
            chosen = max(LAMBDAS, key=lambda lam: (round(validation[lam], 3), lam))  # ties: the stronger anchor
            logger.info(f"  {axis}: held-out AUROC by penalty {({f'{k_:g}': round(v, 3) for k_, v in validation.items()})}"
                        f" -> {chosen:g}")
            for lam in LAMBDAS:
                u = fit(z, s, example, y, lam, scale)
                w = basis @ (u / np.sqrt(np.maximum(variances, 1e-12)).astype(np.float32))
                models.append({"train": train, "axis": axis, "penalty": lam, "selected": lam == chosen,
                               "held_out_auroc": validation[lam], "w": w})

    # --- score every dataset with every model, and the fixed reads beside them
    W = np.stack([m["w"] for m in models], axis=1)
    rows, scores = [], {}
    for name in names:
        tokens = data[name]
        proj = tokens.project(np.concatenate([axes, W], axis=1))
        lengths, ex = tokens.lengths, tokens.example
        for j, axis in enumerate(AXES):
            s = proj[:, j].astype(np.float64)
            mean = np.add.reduceat(s, tokens.offsets[:-1]) / lengths
            scores[(name, axis, "mean")] = mean
            scores[(name, axis, "identity start")] = _scores(s, np.zeros_like(s), tokens, scale)
            scores[(name, axis, "max")] = np.maximum.reduceat(s, tokens.offsets[:-1])
        for m_i, m in enumerate(models):
            j = list(AXES).index(m["axis"])
            scores[(name, m_i)] = _scores(proj[:, j], proj[:, len(AXES) + m_i], tokens, scale)
        logger.info(f"scored {SHORT.get(name, name)}")

    # --- AUROC on every two-class dataset, and flags at the 1% FPR Alpaca threshold
    def auroc(name, key):
        tokens = data[name]
        return roc_auc_score(tokens.y, scores[(name, *key)] if isinstance(key, tuple) else scores[(name, key)])

    for name in SHORT:
        for axis in AXES:
            for read in ("mean", "identity start", "max"):
                rows.append({"train": "-", "test": SHORT[name], "axis": axis, "model": read, "penalty": np.nan,
                             "auroc": auroc(name, (axis, read))})
        for m_i, m in enumerate(models):
            rows.append({"train": SHORT[m["train"]], "test": SHORT[name], "axis": m["axis"],
                         "model": "selected" if m["selected"] else "path", "penalty": m["penalty"],
                         "auroc": auroc(name, m_i)})
    table = pd.DataFrame(rows)
    table.round(4).to_csv(out / "auroc.csv", index=False)
    _report(table, models, scores, data, out, layer)


def _report(table: pd.DataFrame, models: list[dict], scores: dict, data: dict, out: Path, layer: int) -> None:
    from deception_detection.scripts.hpkr_wording import hpkr_frame
    from deception_detection.scripts.steer_lies import _clean

    trains = [SHORT[t] for t in TRAIN_SETS]
    pairs = [(t, s) for t in trains for s in SHORT.values() if s != t]
    raw = pd.read_csv(REPO_ROOT / "results" / f"persona_ood_ext_L{layer}" / "matrix_baseline.csv", index_col=0)
    raw = raw.rename(index=SHORT, columns=SHORT)
    summary = []
    for axis in AXES:
        fixed = table[(table["axis"] == axis) & (table["train"] == "-")].set_index(["model", "test"])["auroc"]
        for read in ("mean", "identity start", "max"):
            summary.append({"axis": axis, "model": read,
                            "off-diagonal mean": np.mean([fixed[(read, s)] for _, s in pairs])})
        learned = table[(table["axis"] == axis) & (table["train"] != "-")]
        sel = learned[learned["model"] == "selected"].set_index(["train", "test"])["auroc"]
        summary.append({"axis": axis, "model": "learned query (penalty chosen on held-out train)",
                        "off-diagonal mean": np.mean([sel[p] for p in pairs])})
        for lam in LAMBDAS:
            path = learned[learned["penalty"] == lam].set_index(["train", "test"])["auroc"]
            summary.append({"axis": axis, "model": f"learned query, penalty {lam:g}",
                            "off-diagonal mean": np.mean([path[p] for p in pairs])})
    raw_mean = np.mean([raw.loc[t, s] for t, s in pairs])
    summary.append({"axis": "-", "model": "raw 8192-d LR probe (same train/test pairs)", "off-diagonal mean": raw_mean})
    summary = pd.DataFrame(summary)
    summary.round(4).to_csv(out / "transfer_summary.csv", index=False)
    logger.info(f"transfer, mean over the {len(pairs)} train/test pairs (train on {trains}):\n"
                f"{summary.round(3).to_string(index=False)}")

    off_diagonal = table["model"].isin(["mean", "max", "selected"]) & (table["train"] != table["test"])
    per_test = table[off_diagonal].groupby(["axis", "model", "test"])["auroc"].mean().unstack("test")
    per_test.round(4).to_csv(out / "per_test.csv")
    logger.info(f"per test set (learned: averaged over the training sets that exclude it):\n"
                f"{per_test.round(3).to_string()}")

    # HP-KR slices and the operating point
    hp = "harm_refusal_knowledge_report__plain"
    frame = hpkr_frame()
    y = data[hp].y
    if not (frame["lie"].to_numpy() == y).all():
        raise RuntimeError("HP-KR rows do not line up with hpkr_frame")
    denied = frame["denied"].to_numpy()
    text = frame["messages"].map(lambda m: _clean(m[-1]["content"])).to_numpy()
    slices = {"full": np.ones(len(y), bool), "denials": denied, "identical text": denied & (text == IDENTICAL)}
    rows = []
    for axis in AXES:
        keys = [("mean", (axis, "mean"))] + [(f"{SHORT[m['train']]}-trained", i) for i, m in enumerate(models)
                                             if m["axis"] == axis and m["selected"]]
        for label, key in keys:
            s = scores[(hp, *key)] if isinstance(key, tuple) else scores[(hp, key)]
            c = scores[(CONTROL, *key)] if isinstance(key, tuple) else scores[(CONTROL, key)]
            flag = s > np.quantile(c, 0.99)
            row = {"axis": axis, "model": label}
            row |= {f"AUROC {k}": roc_auc_score(y[m], s[m]) for k, m in slices.items()}
            row |= {"lies flagged": flag[y == 1].mean(), "honest denials flagged": flag[(y == 0) & denied].mean(),
                    "honest claims flagged": flag[(y == 0) & ~denied].mean()}
            rows.append(row)
    hpkr = pd.DataFrame(rows)
    hpkr.round(4).to_csv(out / "hpkr.csv", index=False)
    logger.info(f"HP-KR:\n{hpkr.round(3).to_string(index=False)}")
    (out / "models.json").write_text(json.dumps([{k: v for k, v in m.items() if k != "w"} for m in models],
                                                indent=1), encoding="utf-8")
    np.save(out / "queries.npy", np.stack([m["w"] for m in models]))


# --- design (a): several fixed persona queries and a learned key map W_K ------------------------
#
# Every head h attends over an example's answer tokens with a_{h,t} = softmax_t(q~_h . k_t) and
# reads the frozen thesis direction v; the heads are mixed with weights pi:
#
#     keys     k_t  = W_K x_t in the key space, spanned by an orthonormal basis E of the queries
#     queries  q~_h = beta_h c_h, fixed: persona h's coordinates c_h = E^T q_h, times a label-free
#                     scale beta_h = c / sigma_h (sigma_h: q_h.x's within-answer SD on Alpaca)
#     score    sum_h pi_h sum_t a_{h,t} (v . x_t)
#
# W_K = E^T (a plain projection) gives logits beta_h q_h.x_t: head h looks where persona h fires.
# M1 keeps that and uniform pi (zero-shot); M2 learns pi; M3 learns W_K = E^T + K as well, with K
# acting on tokens through the training set's standardised within-answer PCs, as in (b).

Q4_AXES = ("deception", "sycophancy", "denial", "assistant")  # the bank's non-composite axes
C_GRID = (0.5, 1.0, 2.0, 4.0)
M3_LAMBDAS = (1.0, 0.1, 0.01)
N_BOOT = 1000


def query_sets(layer: int) -> dict[str, tuple[list[str], np.ndarray]]:
    """The fixed queries as unit columns [hidden, H]: Q4, and the 36 personas centred on the bank mean."""
    bank = pcs.load_bank_at_layer(layer, pcs.EXTENDED_VECTORS_DIR)
    q4 = np.stack([bank.axis_unit(a) for a in Q4_AXES], axis=1)
    centred = bank.personas - bank.personas.mean(0)
    q36 = (centred / np.linalg.norm(centred, axis=1, keepdims=True)).T
    return {"Q4": (list(Q4_AXES), q4.astype(np.float32)), "Q36": (list(bank.names), q36.astype(np.float32))}


def key_space(queries: np.ndarray, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """An orthonormal basis E of the queries' span [hidden, k], and the queries' coordinates C [H, k]."""
    u, sv, _ = np.linalg.svd(queries.astype(np.float64), full_matrices=False)
    basis = u[:, sv > tol * sv.max()]
    return basis, queries.T.astype(np.float64) @ basis


def segment_softmax_heads(logits: torch.Tensor, example: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of every column of `logits` [tokens, H] within each example's tokens."""
    example = example.long()
    index = example[:, None].expand_as(logits)
    top = torch.full((n, logits.shape[1]), -torch.inf, dtype=logits.dtype).scatter_reduce(0, index, logits, "amax")
    e = torch.exp(logits - top[example])
    return e / torch.zeros((n, logits.shape[1]), dtype=logits.dtype).index_add(0, example, e)[example]


def head_pool(s: torch.Tensor, logits: torch.Tensor, example: torch.Tensor, n: int) -> torch.Tensor:
    """Each head's attention-weighted readout, [examples, H]."""
    a = segment_softmax_heads(logits, example, n)
    return torch.zeros((n, logits.shape[1]), dtype=s.dtype).index_add(0, example.long(), a * s[:, None])


def _train(params: list[torch.Tensor], score_fn, y: np.ndarray, penalty_fn=None, max_iter: int = 200) -> None:
    """Balanced logistic loss on the standardised score, with a learned scale and bias (which
    AUROC ignores), by LBFGS."""
    yt = torch.from_numpy(y.astype(np.float64))
    weight = torch.where(yt == 1, 0.5 / yt.mean(), 0.5 / (1 - yt.mean()))
    with torch.no_grad():
        base = score_fn()
        mu, sd = base.mean(), base.std() + 1e-12
    alpha = torch.ones((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(params + [alpha, bias], lr=1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logit = alpha * (score_fn() - mu) / sd + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt, weight=weight, reduction="sum")
        loss = loss / weight.sum() + (penalty_fn() if penalty_fn is not None else 0.0)
        loss.backward()
        return loss

    opt.step(closure)


def fit_heads(pooled: np.ndarray, y: np.ndarray) -> np.ndarray:
    """M2: the head weights pi (a softmax) over fixed per-head scores [examples, H]."""
    per_head = torch.from_numpy(pooled.astype(np.float64))
    rho = torch.zeros(per_head.shape[1], dtype=torch.float64, requires_grad=True)
    _train([rho], lambda: per_head @ torch.softmax(rho, 0), y)
    return torch.softmax(rho, 0).detach().numpy()


def fit_keymap(anchor: np.ndarray, z: np.ndarray, s: np.ndarray, queries: np.ndarray, example: np.ndarray,
               y: np.ndarray, lam: float, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """M3: the learned part K of W_K, in standardised PC coordinates [k, n_pcs], and the head weights.

    `anchor` [tokens, H] are the fixed logits beta_h q_h.x_t; `queries` [H, k] are q~_h = beta_h c_h.
    The logits are anchor + (K z_t) . q~_h, so K = 0 is M2."""
    n = len(y)
    anchor_t, z_t = torch.from_numpy(anchor.astype(np.float64)), torch.from_numpy(z.astype(np.float64))
    s_t, q_t, ex = torch.from_numpy(s.astype(np.float64)), torch.from_numpy(queries), torch.from_numpy(example)
    k_map = torch.zeros((queries.shape[1], z.shape[1]), dtype=torch.float64, requires_grad=True)
    rho = torch.zeros(queries.shape[0], dtype=torch.float64, requires_grad=True)

    def score():
        logits = anchor_t + (z_t @ k_map.T) @ q_t.T
        return head_pool(s_t, logits, ex, n) @ torch.softmax(rho, 0)

    _train([k_map, rho], score, y, lambda: lam * (k_map ** 2).sum(), max_iter)
    return k_map.detach().numpy(), torch.softmax(rho, 0).detach().numpy()


def _blocks(offsets: np.ndarray, max_tokens: int = CHUNK):
    """Runs of whole examples, (first, stop), each within `max_tokens` where possible."""
    n, i = len(offsets) - 1, 0
    while i < n:
        j = i + 1
        while j < n and offsets[j + 1] - offsets[i] <= max_tokens:
            j += 1
        yield i, j
        i = j


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata

    ranks = rankdata(s)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def multi(layer: int = 40, max_iter: int = 200) -> None:
    """Design (a) at `layer`: M1 zero-shot lenses, M2 learned head weights, M3 a learned key map."""
    out = REPO_ROOT / "results" / f"persona_attention_L{layer}" / "multi"
    out.mkdir(parents=True, exist_ok=True)
    readouts = _axes(layer)  # [hidden, 2]: D9, D8
    sets = query_sets(layer)
    names = list(SHORT) + [CONTROL]
    data = {name: load(layer, name) for name in names}

    # --- phase A: every token onto the readouts and every query, once per dataset
    columns = np.concatenate([readouts] + [q for _, q in sets.values()], axis=1)
    slices = {"readout": slice(0, 2)}
    start = 2
    for set_name, (_, q) in sets.items():
        slices[set_name] = slice(start, start + q.shape[1])
        start += q.shape[1]
    proj = {}
    for name in names:
        proj[name] = data[name].project(columns)
        logger.info(f"projected {SHORT.get(name, name)}: {proj[name].shape}")

    def within_sd(values: np.ndarray, tokens: AnswerTokens) -> np.ndarray:
        means = np.add.reduceat(values.astype(np.float64), tokens.offsets[:-1], axis=0) / tokens.lengths[:, None]
        centred = values - np.repeat(means, tokens.lengths, axis=0)
        return np.sqrt((centred ** 2).sum(0) / max(len(values) - len(tokens.labels), 1))

    alpaca = data[CONTROL]
    sigma = {set_name: within_sd(proj[CONTROL][:, sl], alpaca) for set_name, sl in slices.items() if set_name != "readout"}
    sigma["readout"] = within_sd(proj[CONTROL][:, slices["readout"]], alpaca)
    spaces = {set_name: key_space(q) for set_name, (_, q) in sets.items()}
    for set_name, (basis, _) in spaces.items():
        logger.info(f"{set_name}: {sets[set_name][1].shape[1]} queries, key space k = {basis.shape[1]}")

    def pooled_heads(name: str, set_name: str, j: int, c: float) -> np.ndarray:
        tokens, p = data[name], proj[name]
        logits = p[:, slices[set_name]] * (c / sigma[set_name])[None, :]
        return head_pool(torch.from_numpy(p[:, j].astype(np.float64)), torch.from_numpy(logits.astype(np.float64)),
                         torch.from_numpy(tokens.example), len(tokens.labels)).numpy()

    # --- zero-shot scores: mean, max, design 1 and M1, on every dataset
    scores: dict[tuple, np.ndarray] = {}
    models: list[dict] = []
    for name in names:
        tokens, p = data[name], proj[name]
        for j, axis in enumerate(AXES):
            s = p[:, j].astype(np.float64)
            scores[(name, axis, "mean")] = np.add.reduceat(s, tokens.offsets[:-1]) / tokens.lengths
            scores[(name, axis, "max")] = np.maximum.reduceat(s, tokens.offsets[:-1])
            for c in C_GRID:
                one = s[:, None] * (c / sigma["readout"][j])
                scores[(name, axis, f"design 1, c={c:g}")] = head_pool(
                    torch.from_numpy(s), torch.from_numpy(one), torch.from_numpy(tokens.example),
                    len(tokens.labels)).numpy()[:, 0]
                for set_name in sets:
                    scores[(name, axis, f"M1 {set_name}, c={c:g}")] = pooled_heads(name, set_name, j, c).mean(1)

    # --- fits: M2 and M3 per training set, readout and query set
    for train in TRAIN_SETS:
        tokens, p = data[train], proj[train]
        y, example = tokens.y, tokens.example
        held = _split(y)
        fit_mask, fit_ex = _subset(example, ~held)
        val_mask, val_ex = _subset(example, held)
        basis, variances, share = within_pcs(tokens)
        scale = np.sqrt(np.maximum(variances, 1e-12)).astype(np.float32)
        z = tokens.project(basis) / scale[None, :]
        lift = basis / scale[None, :]  # standardised PC coordinates -> activation space
        logger.info(f"{SHORT[train]}: within-answer PCs hold {share:.1%}")
        for j, axis in enumerate(AXES):
            s = p[:, j]
            for set_name, (head_names, q) in sets.items():
                beta = 1.0 / sigma[set_name]
                anchor = p[:, slices[set_name]] * beta[None, :]
                _, coords = spaces[set_name]
                q_tilde = coords * beta[:, None]
                # M2
                heads_all = pooled_heads(train, set_name, j, 1.0)
                pi_fit = fit_heads(heads_all[~held], y[~held])
                m2_val = _auroc(y[held], heads_all[held] @ pi_fit)
                pi = fit_heads(heads_all, y)
                models.append({"kind": "M2", "train": train, "axis": axis, "set": set_name, "penalty": np.nan,
                               "selected": True, "held_out_auroc": m2_val, "pi": pi, "heads": head_names})
                # M3
                validation = {}
                for lam in M3_LAMBDAS:
                    k_fit, pi_fit = fit_keymap(anchor[fit_mask], z[fit_mask], s[fit_mask], q_tilde, fit_ex,
                                               y[~held], lam, max_iter)
                    logits = anchor[val_mask] + (z[val_mask] @ k_fit.T) @ q_tilde.T
                    val = head_pool(torch.from_numpy(s[val_mask].astype(np.float64)), torch.from_numpy(logits),
                                    torch.from_numpy(val_ex), int(held.sum())).numpy() @ pi_fit
                    validation[lam] = _auroc(y[held], val)
                chosen = max(M3_LAMBDAS, key=lambda lam: (round(validation[lam], 3), lam))
                for lam in M3_LAMBDAS:
                    k_map, pi = fit_keymap(anchor, z, s, q_tilde, example, y, lam, max_iter)
                    # each head's effective query in activation space: beta_h (q_h + lift K^T c_h)
                    lifted = (q.astype(np.float64) + lift.astype(np.float64) @ (k_map.T @ coords.T)) * beta[None, :]
                    moved = np.linalg.norm(lift @ (k_map.T @ coords.T), axis=0).mean()
                    models.append({"kind": "M3", "train": train, "axis": axis, "set": set_name, "penalty": lam,
                                   "selected": lam == chosen, "held_out_auroc": validation[lam], "pi": pi,
                                   "heads": head_names, "lifted": lifted.astype(np.float32),
                                   "query_shift": float(moved)})
                logger.info(f"  {axis} {set_name}: M2 held-out {m2_val:.3f}; M3 held-out "
                            f"{({f'{k:g}': round(v, 3) for k, v in validation.items()})} -> {chosen:g}")
        del z

    # --- scores of the fitted models on every dataset
    m3 = [m for m in models if m["kind"] == "M3"]
    stacked = np.concatenate([m["lifted"] for m in m3], axis=1)
    bounds = np.cumsum([0] + [m["lifted"].shape[1] for m in m3])
    for name in names:
        tokens, p = data[name], proj[name]
        for i, m in enumerate(models):
            if m["kind"] == "M2":
                j = list(AXES).index(m["axis"])
                scores[(name, "model", i)] = pooled_heads(name, m["set"], j, 1.0) @ m["pi"]
        out_scores = {i: np.empty(len(tokens.labels)) for i, m in enumerate(models) if m["kind"] == "M3"}
        m3_index = [i for i, m in enumerate(models) if m["kind"] == "M3"]
        for first, stop in _blocks(tokens.offsets):
            a, b = tokens.offsets[first], tokens.offsets[stop]
            logits_all = tokens.rows(a, b) @ stacked
            ex = torch.from_numpy(tokens.example[a:b] - first)
            for r, i in enumerate(m3_index):
                m = models[i]
                j = list(AXES).index(m["axis"])
                pooled = head_pool(torch.from_numpy(p[a:b, j].astype(np.float64)),
                                   torch.from_numpy(logits_all[:, bounds[r]:bounds[r + 1]].astype(np.float64)),
                                   ex, stop - first).numpy()
                out_scores[i][first:stop] = pooled @ m["pi"]
        for i, v in out_scores.items():
            scores[(name, "model", i)] = v
        logger.info(f"scored {SHORT.get(name, name)}")
    _multi_report(models, scores, data, out)


def _multi_report(models: list[dict], scores: dict, data: dict, out: Path) -> None:
    from deception_detection.scripts.hpkr_wording import hpkr_frame

    trains = [SHORT[t] for t in TRAIN_SETS]
    tests = list(SHORT)
    pairs = [(t, s) for t in TRAIN_SETS for s in tests if s != t]
    zero_shot = sorted({k[2] for k in scores if k[1] in AXES})
    rows = []
    for axis in AXES:
        for read in zero_shot:
            per_test = {s: _auroc(data[s].y, scores[(s, axis, read)]) for s in tests}
            rows.append({"axis": axis, "model": read, "learned": "none",
                         "transfer": np.mean([per_test[s] for _, s in pairs])} | {SHORT[s]: per_test[s] for s in tests})
        for kind in ("M2", "M3"):
            for set_name in ("Q4", "Q36"):
                chosen = [i for i, m in enumerate(models) if m["kind"] == kind and m["axis"] == axis
                          and m["set"] == set_name and m["selected"]]
                by_train = {models[i]["train"]: i for i in chosen}
                cell = {(t, s): _auroc(data[s].y, scores[(s, "model", by_train[t])]) for t, s in pairs}
                per_test = {s: np.mean([cell[(t, s2)] for t, s2 in pairs if s2 == s]) for s in tests}
                rows.append({"axis": axis, "model": f"{kind} {set_name}", "learned": "pi" if kind == "M2" else "pi + W_K",
                             "transfer": np.mean(list(cell.values()))} | {SHORT[s]: per_test[s] for s in tests})
    table = pd.DataFrame(rows)
    table.round(4).to_csv(out / "transfer.csv", index=False)
    pd.set_option("display.width", 250)
    logger.info(f"transfer, mean over the {len(pairs)} train/test pairs (train on {trains}), and per test set "
                f"(learned: averaged over the training sets that exclude it):\n{table.round(3).to_string(index=False)}")

    # paired bootstrap per test set against the max read, for the pre-registered models
    rng = np.random.default_rng(SEED)
    boot = []
    for axis in AXES:
        for label, keyset in (("M1 Q4, c=1", ("zs", "M1 Q4, c=1")), ("M1 Q36, c=1", ("zs", "M1 Q36, c=1")),
                              ("M2 Q4", ("M2", "Q4")), ("M2 Q36", ("M2", "Q36")),
                              ("M3 Q4", ("M3", "Q4")), ("M3 Q36", ("M3", "Q36"))):
            for s in tests:
                y = data[s].y
                if keyset[0] == "zs":
                    vectors = [scores[(s, axis, keyset[1])]]
                else:
                    vectors = [scores[(s, "model", i)] for i, m in enumerate(models)
                               if m["kind"] == keyset[0] and m["set"] == keyset[1] and m["axis"] == axis
                               and m["selected"] and m["train"] != s]
                ref = scores[(s, axis, "max")]
                deltas = []
                for _ in range(N_BOOT):
                    idx = rng.integers(0, len(y), len(y))
                    if 0 < y[idx].sum() < len(idx):
                        deltas.append(np.mean([_auroc(y[idx], v[idx]) for v in vectors]) - _auroc(y[idx], ref[idx]))
                lo, hi = np.quantile(deltas, [0.025, 0.975])
                point = np.mean([_auroc(y, v) for v in vectors]) - _auroc(y, ref)
                boot.append({"axis": axis, "model": label, "test": SHORT[s], "delta_vs_max": point, "lo": lo, "hi": hi,
                             "significant_loss": bool(hi < 0)})
    boot = pd.DataFrame(boot)
    boot.round(4).to_csv(out / "bootstrap_vs_max.csv", index=False)
    logger.info(f"AUROC minus the max read, per test set (95% bootstrap):\n"
                f"{boot.pivot_table(index=['axis', 'model'], columns='test', values='delta_vs_max').round(3).to_string()}\n"
                f"significant losses: {boot[boot['significant_loss']][['axis', 'model', 'test']].values.tolist()}")

    # HP-KR slices and the operating point
    hp = "harm_refusal_knowledge_report__plain"
    frame = hpkr_frame()
    y = data[hp].y
    if not (frame["lie"].to_numpy() == y).all():
        raise RuntimeError("HP-KR rows do not line up with hpkr_frame")
    from deception_detection.scripts.steer_lies import _clean
    denied = frame["denied"].to_numpy()
    text = frame["messages"].map(lambda m: _clean(m[-1]["content"])).to_numpy()
    slices = {"full": np.ones(len(y), bool), "denials": denied, "identical text": denied & (text == IDENTICAL)}
    hp_rows = []
    for axis in AXES:
        keys = [(read, (axis, read)) for read in ("mean", "max", "M1 Q4, c=1", "M1 Q36, c=1")]
        keys += [(f"{m['kind']} {m['set']} ({SHORT[m['train']]})", ("model", i)) for i, m in enumerate(models)
                 if m["axis"] == axis and m["selected"] and m["train"] != hp]
        for label, key in keys:
            s, c = scores[(hp, *key)], scores[(CONTROL, *key)]
            flag = s > np.quantile(c, 0.99)
            hp_rows.append({"axis": axis, "model": label} | {f"AUROC {k}": _auroc(y[m], s[m]) for k, m in slices.items()}
                           | {"lies flagged": flag[y == 1].mean(), "honest denials flagged": flag[(y == 0) & denied].mean(),
                              "honest claims flagged": flag[(y == 0) & ~denied].mean()})
    hpkr = pd.DataFrame(hp_rows)
    hpkr.round(4).to_csv(out / "hpkr.csv", index=False)
    logger.info(f"HP-KR (models trained on HP-KR left out):\n{hpkr.round(3).to_string(index=False)}")

    # which lenses carry weight, and how far W_K moved
    head_rows = []
    for m in models:
        if not m["selected"]:
            continue
        top = np.argsort(m["pi"])[::-1][:5]
        head_rows.append({"kind": m["kind"], "train": SHORT[m["train"]], "axis": m["axis"], "set": m["set"],
                          "penalty": m["penalty"], "held_out_auroc": m["held_out_auroc"],
                          "query_shift": m.get("query_shift", 0.0),
                          "top heads": ", ".join(f"{m['heads'][t]} {m['pi'][t]:.2f}" for t in top)})
    heads = pd.DataFrame(head_rows)
    heads.round(4).to_csv(out / "heads.csv", index=False)
    logger.info(f"learned head weights (top 5) and the mean shift of each head's query:\n"
                f"{heads.round(3).to_string(index=False)}")


if __name__ == "__main__":
    fire.Fire({"extract": extract, "check": check, "run": run, "multi": multi})
