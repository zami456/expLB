"""Multi-head attention pooling at extraction, with the heads trained on Liars' Bench labels.

The persona rollouts are pooled by H attention heads, each a query in persona coordinates; every
head yields its own persona vectors and its own contrast (a fixed group-mean difference, D9 or D8);
the direction is a softmax mixture of the heads' contrasts. The mixture scores one source dataset's
mean-pooled examples, and a balanced logistic loss on that source's labels trains the queries and
the mixture weights. Nothing else is fitted: the direction stays a difference of persona means, and
the labels decide which rollout tokens the means are taken over. That makes it a **trained arm**,
evaluated off-diagonal in the transfer protocol beside the raw probe.

The loss is exact without any truncation: d_h . z_i = sum_r c_r sum_t a_{h,t} (x_{r,t} . z_i), so
the Gram matrix G_S = X_bank Z_S^T between the bank's tokens and the source's pooled examples is
computed once per source, and training runs on it. Afterwards the persona vectors and directions
are recomputed explicitly from the full tokens.

    python deception_detection/scripts/persona_attention_lb.py grams
    python deception_detection/scripts/persona_attention_lb.py fit
    python deception_detection/scripts/persona_attention_lb.py directions
    python deception_detection/scripts/persona_attention_lb.py evaluate

Reads the stage-1 token store of `persona_attention_extract.py`; outputs go to
`results/persona_attention_lb_L40/`.
"""

import json
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from deception_detection import persona_pcs as pcs
from deception_detection.log import logger
from deception_detection.scripts import persona_attention_extract as pe
from deception_detection.scripts.persona_attention_extract import (
    LAYER,
    PERSONAS,
    TOKENS,
    attention,
    entropy_penalty,
    persona_means,
)
from deception_detection.utils import REPO_ROOT

OUT = REPO_ROOT / "results" / f"persona_attention_lb_L{LAYER}"
GRAMS = TOKENS / "grams"
CONTROL = "XXXX-20__plain"
SOURCES = ("convince_me__plain", "harm_refusal_knowledge_report__plain", "insider_trading_cadenza__plain",
           "instruct_dishonesty__plain", "repe_honesty__plain")
TESTS = SOURCES[:1] + ("harm_refusal_choice__plain",) + SOURCES[1:]
SHORT = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C",
         "harm_refusal_knowledge_report__plain": "HP-KR", "insider_trading_cadenza__plain": "IT",
         "instruct_dishonesty__plain": "ID", "repe_honesty__plain": "RepE"}
N_ID = 2000  # ID examples kept, stratified
HEADS = (1, 3)
LAMBDAS = (1.0, 0.1, 0.01)
N_PERM = 3
SEED = 0
N_NULL = 2000
N_BOOT = 1000
SHAPES = {
    "extended (12v13)": (pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS,
                         pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS),
    "knowledge-denial (4v6)": (pcs.KNOWLEDGE_DENIAL_PERSONAS, pcs.KNOWLEDGE_CANDOUR_PERSONAS),
}
IDENTICAL = "I do not know the answer."


# --- Gram matrices ------------------------------------------------------------------------------

def _pooled_sets():
    from deception_detection.scripts import persona_ood as P
    from deception_detection.scripts.layer_sweep import pooled_dir

    return P.load_pooled(pooled_dir(LAYER))


def source_rows(name: str, n_examples: int, y: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Which of a source's examples enter training: all of them, or a stratified subsample of ID."""
    if name != "instruct_dishonesty__plain" or n_examples <= N_ID:
        return np.arange(n_examples)
    rng = np.random.default_rng(seed)
    keep = []
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        keep.extend(rng.choice(idx, size=int(round(N_ID * len(idx) / n_examples)), replace=False))
    return np.sort(np.array(keep))


def grams() -> None:
    """G_S = X_bank Z_S^T for every source, streamed over the token store into memmaps."""
    GRAMS.mkdir(parents=True, exist_ok=True)
    index = pe.load_index()
    bits = pe.token_memmap(index)
    rows = pd.DataFrame(index["rows"])
    pooled = _pooled_sets()
    for name in SOURCES:
        path = GRAMS / f"{name}.npy"
        if path.exists():
            logger.info(f"{SHORT[name]}: gram exists, skipping")
            continue
        z, y = pooled[name].z, pooled[name].y
        keep = source_rows(name, len(y), y)
        zt = np.ascontiguousarray(z[keep].T.astype(np.float32))  # [hidden, n]
        gram = np.lib.format.open_memmap(path.with_suffix(".partial.npy"), mode="w+", dtype=np.float32,
                                         shape=(len(bits), len(keep)))
        for _, row in rows.iterrows():
            a, b = int(row["offset"]), int(row["offset"] + row["length"])
            gram[a:b] = pe.rows_float(bits, a, b) @ zt
        gram.flush()
        del gram
        path.with_suffix(".partial.npy").rename(path)
        np.savez(GRAMS / f"{name}_labels.npz", y=y[keep], rows=keep)
        logger.info(f"{SHORT[name]}: gram [{len(bits)}, {len(keep)}] written")


# --- the model ---------------------------------------------------------------------------------

def head_contrast_scores(U: torch.Tensor, coords: torch.Tensor, keep: torch.Tensor, tok_roll: torch.Tensor,
                         offsets: np.ndarray, gram: torch.Tensor, persona_of: torch.Tensor, n_personas: int,
                         positive: list[int], negative: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """For every head: d_h . z_i for each source example, `[H, n]`, and the attention `[H, N]`.

    d_h is the contrast of head h's persona vectors; the products come through the Gram matrix, so
    no pooled vector is ever formed."""
    n_rollouts = len(offsets) - 1
    scores, weights = [], []
    for h in range(U.shape[0]):
        a = attention(U[h], coords, keep, tok_roll, n_rollouts)
        per_rollout = torch.stack([a[o0:o1].float() @ gram[o0:o1] for o0, o1 in zip(offsets[:-1], offsets[1:])]).double()
        v = persona_means(per_rollout, persona_of, n_personas)
        scores.append(v[positive].mean(0) - v[negative].mean(0))
        weights.append(a)
    return torch.stack(scores), torch.stack(weights)


class LBObjective:
    """Queries and mixture weights for one source, shape and head count."""

    def __init__(self, bank: pe.Bank, gram: torch.Tensor, y: np.ndarray, shape: str, n_heads: int, lam: float,
                 floor: bool = True, train_mask: np.ndarray | None = None, seed: int = SEED):
        self.bank, self.gram, self.lam, self.floor, self.n_heads = bank, gram, lam, floor, n_heads
        self.coords = torch.from_numpy(bank.coords.astype(np.float64))
        self.keep = torch.from_numpy(bank.keep)
        self.tok_roll = torch.from_numpy(bank.tok_roll.astype(np.int64))
        self.offsets = np.concatenate([[0], np.cumsum(bank.rows["length"].to_numpy())])
        self.persona_of = torch.from_numpy(bank.persona_of.astype(np.int64))
        idx = {p: i for i, p in enumerate(PERSONAS)}
        pos, neg = SHAPES[shape]
        self.positive, self.negative = [idx[p] for p in pos], [idx[p] for p in neg]
        self.y = torch.from_numpy(y.astype(np.float64))
        self.train = torch.from_numpy(train_mask if train_mask is not None else np.ones(len(y), bool))
        # heads that start identical receive identical gradients and never separate, so several
        # heads start at small distinct points; a single head starts at zero, which is mean pooling
        init = np.zeros((n_heads, bank.coords.shape[1]))
        if n_heads > 1:
            init = np.random.default_rng(seed).normal(scale=0.05, size=init.shape)
        self.U = torch.tensor(init, dtype=torch.float64, requires_grad=True)
        self.rho = torch.zeros(n_heads, dtype=torch.float64, requires_grad=True)
        self.alpha = torch.ones((), dtype=torch.float64, requires_grad=True)
        self.bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            base = self.scores(self.U, self.rho)
            self.mu, self.sd = base[self.train].mean(), base[self.train].std() + 1e-12

    def scores(self, U: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        per_head, self._weights = head_contrast_scores(U, self.coords, self.keep, self.tok_roll, self.offsets,
                                                       self.gram, self.persona_of, len(PERSONAS), self.positive,
                                                       self.negative)
        return torch.softmax(rho, 0) @ per_head

    def loss(self) -> torch.Tensor:
        s = self.scores(self.U, self.rho)[self.train]
        y = self.y[self.train]
        weight = torch.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
        logit = self.alpha * (s - self.mu) / self.sd + self.bias
        value = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, weight=weight, reduction="sum") / weight.sum()
        value = value + self.lam * (self.U ** 2).sum() + 1e-3 * (self.alpha ** 2 + self.bias ** 2)
        if self.floor:
            value = value + torch.stack([entropy_penalty(a, self.tok_roll, self.keep, self.bank.n_rollouts)
                                         for a in self._weights]).mean()
        return value

    def fit(self, max_iter: int = 60) -> "LBObjective":
        opt = torch.optim.LBFGS([self.U, self.rho, self.alpha, self.bias], lr=1, max_iter=max_iter,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            value = self.loss()
            value.backward()
            return value

        opt.step(closure)
        return self

    def auroc(self, mask: np.ndarray) -> float:
        with torch.no_grad():
            s = self.scores(self.U, self.rho).numpy()
        return roc_auc_score(self.y.numpy()[mask], s[mask])

    @property
    def pi(self) -> np.ndarray:
        return torch.softmax(self.rho.detach(), 0).numpy()


def _split(y: np.ndarray, seed: int = SEED) -> np.ndarray:
    rng, held = np.random.default_rng(seed), np.zeros(len(y), bool)
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        held[rng.choice(idx, size=max(1, len(idx) // 5), replace=False)] = True
    return held


def fit(max_iter: int = 60, heads=HEADS) -> None:
    """Fit every (source, shape, head count); `heads` restricts the head counts, and the models of
    other head counts already in `models.json` are kept."""
    OUT.mkdir(parents=True, exist_ok=True)
    heads = tuple(int(h) for h in (heads if isinstance(heads, (list, tuple)) else [heads]))
    bank = pe.load_bank()
    models = []
    if (OUT / "models.json").exists():
        models = [m for m in json.loads((OUT / "models.json").read_text(encoding="utf-8")) if m["heads"] not in heads]
    for source in SOURCES:
        gram = torch.from_numpy(np.load(GRAMS / f"{source}.npy"))
        y = np.load(GRAMS / f"{source}_labels.npz")["y"]
        held = _split(y)
        for shape in SHAPES:
            for n_heads in heads:
                validation = {}
                for lam in LAMBDAS:
                    m = LBObjective(bank, gram, y, shape, n_heads, lam, train_mask=~held).fit(max_iter)
                    validation[lam] = m.auroc(held)
                chosen = max(LAMBDAS, key=lambda lam: (round(validation[lam], 3), lam))
                m = LBObjective(bank, gram, y, shape, n_heads, chosen).fit(max_iter)
                models.append({"source": source, "shape": shape, "heads": n_heads, "lambda": chosen,
                               "permutation": None, "held_out": validation, "in_sample_auroc": m.auroc(np.ones(len(y), bool)),
                               "U": m.U.detach().numpy().tolist(), "pi": m.pi.tolist()})
                logger.info(f"{SHORT[source]} {shape} H={n_heads}: held-out by lambda "
                            f"{({f'{k:g}': round(v, 3) for k, v in validation.items()})} -> {chosen:g}; "
                            f"in-sample {models[-1]['in_sample_auroc']:.3f}; pi {np.round(m.pi, 2).tolist()}; "
                            f"|U| {np.round(np.linalg.norm(m.U.detach().numpy(), axis=1), 2).tolist()}")
                if n_heads == max(HEADS):
                    for perm in range(N_PERM):
                        y_perm = np.random.default_rng(SEED + 100 + perm).permutation(y)
                        mp = LBObjective(bank, gram, y_perm, shape, n_heads, chosen, seed=SEED + 1 + perm).fit(max_iter)
                        models.append({"source": source, "shape": shape, "heads": n_heads, "lambda": chosen,
                                       "permutation": perm, "held_out": None, "in_sample_auroc": mp.auroc(np.ones(len(y), bool)),
                                       "U": mp.U.detach().numpy().tolist(), "pi": mp.pi.tolist()})
                    logger.info(f"  shuffled-label fits: |U| "
                                f"{[round(float(np.linalg.norm(np.array(m_['U']))), 2) for m_ in models[-N_PERM:]]}")
        del gram
    (OUT / "models.json").write_text(json.dumps(models, indent=1), encoding="utf-8")
    logger.info(f"{len(models)} models -> {OUT / 'models.json'}")


# --- explicit directions from the full tokens ------------------------------------------------------

def _model_id(m: dict) -> str:
    tag = f"lb-{SHORT[m['source']]}-H{m['heads']}-{'D9' if m['shape'].startswith('extended') else 'D8'}"
    return tag if m["permutation"] is None else f"{tag}-perm{m['permutation']}"


def directions() -> None:
    """One pass over the token store: every model's persona vectors (mixed over heads) and direction."""
    models = json.loads((OUT / "models.json").read_text(encoding="utf-8"))
    bank = pe.load_bank()
    bits = pe.token_memmap(pe.load_index())
    token_ids = np.load(TOKENS / "token_ids.npy")
    heads = np.concatenate([np.array(m["U"], dtype=np.float64) for m in models])  # [H_total, k]
    owner = []
    for i, m in enumerate(models):
        owner.extend([i] * m["heads"])
    sums = np.zeros((len(heads), len(PERSONAS), bits.shape[1]), dtype=np.float64)
    top = {h: {} for h in range(len(heads)) if models[owner[h]]["permutation"] is None and models[owner[h]]["heads"] == max(HEADS)}
    vocab = np.zeros((len(top), int(token_ids.max()) + 1))
    top_index = {h: j for j, h in enumerate(top)}
    for r, row in bank.rows.iterrows():
        a, b = int(row["offset"]), int(row["offset"] + row["length"])
        x = pe.rows_float(bits, a, b)
        logits = bank.coords[a:b].astype(np.float64) @ heads.T  # [T, H_total]
        logits[~bank.keep[a:b]] = -np.inf
        w = np.exp(logits - logits.max(0, keepdims=True))
        w /= w.sum(0, keepdims=True)
        sums[:, bank.persona_of[r]] += (w.T @ x)
        for h, j in top_index.items():
            np.add.at(vocab[j], token_ids[a:b], w[:, h])
    counts = np.bincount(bank.persona_of, minlength=len(PERSONAS)).astype(np.float64)
    out, records, h0 = {}, [], 0
    for i, m in enumerate(models):
        v_heads = sums[h0:h0 + m["heads"]] / counts[None, :, None]
        h0 += m["heads"]
        mixed = np.tensordot(np.array(m["pi"]), v_heads, axes=1).astype(np.float32)  # [36, hidden]
        pos, neg = SHAPES[m["shape"]]
        d = pe._axis(mixed, pos, neg)
        out[_model_id(m)] = d
        records.append({"id": _model_id(m)} | {k: m[k] for k in ("source", "shape", "heads", "lambda", "permutation")}
                       | {"pi": m["pi"]})
        if m["permutation"] is None:
            pe.write_bank(mixed, counts.astype(int), _model_id(m))
    np.savez(OUT / "directions.npz", **out)
    (OUT / "directions.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pe.MODEL)
    rows = []
    for h, j in top_index.items():
        m = models[owner[h]]
        order = np.argsort(vocab[j])[::-1][:10]
        share = vocab[j][order].sum() / vocab[j].sum()
        rows.append({"model": _model_id(m), "head": h - sum(mm["heads"] for mm in models[:owner[h]]),
                     "pi": round(m["pi"][h - sum(mm["heads"] for mm in models[:owner[h]])], 3),
                     "|u|": round(float(np.linalg.norm(heads[h])), 3), "top-10 share": round(float(share), 3),
                     "top tokens": " | ".join(f"{tokenizer.decode([int(t)])!r} {vocab[j][t] / vocab[j].sum():.3f}" for t in order)})
    pd.DataFrame(rows).to_csv(OUT / "top_tokens.csv", index=False)
    logger.info(f"{len(out)} directions written; top tokens of the H={max(HEADS)} heads:\n"
                f"{pd.DataFrame(rows).to_string(index=False)}")


# --- evaluation --------------------------------------------------------------------------------

def _comparators() -> dict[str, dict[str, np.ndarray]]:
    """Fixed directions to place the trained ones against, per shape."""
    out = {}
    for label, folder in (("mean extraction (same rollouts)", pe.BANKS / "mean"),
                          ("bank-trained query (same rollouts)", pe.BANKS / "learned"),
                          ("full bank (mean)", pcs.EXTENDED_VECTORS_DIR)):
        bank = pcs.load_bank_at_layer(LAYER, folder)
        out[label] = {"extended (12v13)": bank.axis_unit("extended"), "knowledge-denial (4v6)": bank.axis_unit("denial")}
    return out


def evaluate() -> None:
    from deception_detection.scripts.forced_denial_bank import OUT as FIX1
    from deception_detection.scripts.hpkr_wording import auroc_ci, hpkr_frame
    from deception_detection.scripts.knowledge_bank import _slices
    from deception_detection.scripts.persona_attention import load as load_answer_tokens

    pooled = _pooled_sets()
    records = json.loads((OUT / "directions.json").read_text(encoding="utf-8"))
    trained = dict(np.load(OUT / "directions.npz"))
    comparators = _comparators()
    columns = [(r["id"], trained[r["id"]]) for r in records]
    columns += [(f"{label} | {shape}", d) for label, per in comparators.items() for shape, d in per.items()]
    matrix = np.stack([d for _, d in columns], axis=1).astype(np.float32)
    names = [n for n, _ in columns]
    pairs = [(s, t) for s in SOURCES for t in TESTS if t != s]

    # scores on every dataset, mean and max reads
    scores: dict[tuple, np.ndarray] = {}
    for name in TESTS + (CONTROL,):
        z = pooled[name].z
        tokens = load_answer_tokens(LAYER, name)
        proj = tokens.project(matrix)
        for j, col in enumerate(names):
            scores[(name, col, "mean")] = z @ matrix[:, j].astype(np.float64)
            scores[(name, col, "max")] = np.maximum.reduceat(proj[:, j].astype(np.float64), tokens.offsets[:-1])
        logger.info(f"scored {SHORT.get(name, name)}")

    def auroc(name, col, read):
        return roc_auc_score(pooled[name].y, scores[(name, col, read)])

    # the transfer table: trained models by (shape, heads, real/null), comparators on the same 25 pairs
    rows = []
    by_key = {}
    for r in records:
        by_key.setdefault((r["shape"], r["heads"], r["permutation"] is not None), {}).setdefault(r["source"], []).append(r["id"])
    for (shape, heads, is_null), sources in by_key.items():
        for read in ("mean", "max"):
            cells = {}
            for s, t in pairs:
                ids = sources.get(s, [])
                if ids:
                    cells[(s, t)] = np.mean([auroc(t, i, read) for i in ids])
            per_test = {SHORT[t]: np.mean([v for (s2, t2), v in cells.items() if t2 == t]) for t in TESTS}
            in_sample = np.mean([np.mean([auroc(s, i, read) for i in ids]) for s, ids in sources.items()])
            row = {"direction": f"LB-trained H={heads}" + (" (shuffled labels)" if is_null else ""), "shape": shape,
                   "read": read, "transfer (25 pairs)": np.mean(list(cells.values())), "in-sample": in_sample} | per_test
            if is_null:  # the spread over permutations, each a full set of 25 pairs
                per_perm = []
                for p in range(N_PERM):
                    ids_p = {s: [i for i in ids if i.endswith(f"-perm{p}")] for s, ids in sources.items()}
                    per_perm.append(np.mean([auroc(t, ids_p[s][0], read) for s, t in pairs if ids_p.get(s)]))
                row["null min"], row["null max"] = min(per_perm), max(per_perm)
            rows.append(row)
    for label, per in comparators.items():
        for shape in SHAPES:
            col = f"{label} | {shape}"
            for read in ("mean", "max"):
                per_test = {SHORT[t]: auroc(t, col, read) for t in TESTS}
                rows.append({"direction": label, "shape": shape, "read": read,
                             "transfer (25 pairs)": np.mean([per_test[SHORT[t]] for _, t in pairs]), "in-sample": np.nan} | per_test)
    raw = pd.read_csv(REPO_ROOT / "results" / f"persona_ood_ext_L{LAYER}" / "matrix_baseline.csv", index_col=0)
    rows.append({"direction": "raw 8192-d probe", "shape": "-", "read": "mean",
                 "transfer (25 pairs)": np.mean([raw.loc[s, t] for s, t in pairs]),
                 "in-sample": np.mean([raw.loc[s, s] for s in SOURCES])}
                | {SHORT[t]: np.mean([raw.loc[s, t] for s in SOURCES if s != t]) for t in TESTS})
    table = pd.DataFrame(rows)
    table.round(4).to_csv(OUT / "transfer.csv", index=False)
    pd.set_option("display.width", 250)
    logger.info(f"transfer (train on {[SHORT[s] for s in SOURCES]}, each tested on the other five):\n"
                f"{table.round(3).to_string(index=False)}")

    # paired bootstrap per test set: H=3 real against mean extraction, test-time mean
    rng = np.random.default_rng(SEED)
    boot = []
    for shape in SHAPES:
        ref_col = f"mean extraction (same rollouts) | {shape}"
        for t in TESTS:
            y = pooled[t].y
            ids = [r["id"] for r in records if r["shape"] == shape and r["heads"] == max(HEADS)
                   and r["permutation"] is None and r["source"] != t]
            vectors = [scores[(t, i, "mean")] for i in ids]
            ref = scores[(t, ref_col, "mean")]
            deltas = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, len(y), len(y))
                if 0 < y[idx].sum() < len(idx):
                    deltas.append(np.mean([roc_auc_score(y[idx], v[idx]) for v in vectors]) - roc_auc_score(y[idx], ref[idx]))
            lo, hi = np.quantile(deltas, [0.025, 0.975])
            boot.append({"shape": shape, "test": SHORT[t], "delta": np.mean([roc_auc_score(y, v) for v in vectors]) - roc_auc_score(y, ref),
                         "lo": lo, "hi": hi, "significant_loss": bool(hi < 0), "significant_gain": bool(lo > 0)})
    boot = pd.DataFrame(boot)
    boot.round(4).to_csv(OUT / "bootstrap.csv", index=False)
    logger.info(f"H=3 LB-trained minus mean extraction, per test set (models trained on that set excluded):\n"
                f"{boot.round(3).to_string(index=False)}")

    # HP-KR slices (models trained on HP-KR excluded), null and operating point
    hp = "harm_refusal_knowledge_report__plain"
    frame = hpkr_frame()
    y = pooled[hp].y
    if not (frame["lie"].to_numpy() == y).all():
        raise RuntimeError("HP-KR rows do not line up with hpkr_frame")
    slices = _slices(frame)
    denied = frame["denied"].to_numpy()
    z = pooled[hp].z
    null = rng.standard_normal((N_NULL, z.shape[1]))
    projected = z @ (null / np.linalg.norm(null, axis=1, keepdims=True)).T
    bands = {k: np.array([roc_auc_score(y[slices[k]], projected[slices[k], j]) for j in range(N_NULL)])
             for k in ("full", "denials", "identical text")}
    hp_rows = []
    for shape in SHAPES:
        groups = {"LB-trained H=3": [r["id"] for r in records if r["shape"] == shape and r["heads"] == 3
                                     and r["permutation"] is None and r["source"] != hp],
                  "LB-trained H=1": [r["id"] for r in records if r["shape"] == shape and r["heads"] == 1
                                     and r["permutation"] is None and r["source"] != hp],
                  "mean extraction (same rollouts)": [f"mean extraction (same rollouts) | {shape}"],
                  "full bank (mean)": [f"full bank (mean) | {shape}"]}
        for label, ids in groups.items():
            row = {"direction": label, "shape": shape}
            for k, band in bands.items():
                m = slices[k]
                aucs = [roc_auc_score(y[m], scores[(hp, i, "mean")][m]) for i in ids]
                row[k] = np.mean(aucs)
                row[f"{k} p"] = float(np.mean([np.mean(band >= a) for a in aucs]))
            flags = []
            for i in ids:
                s, c = scores[(hp, i, "mean")], scores[(CONTROL, i, "mean")]
                flag = s > np.quantile(c, 0.99)
                flags.append((flag[y == 1].mean(), flag[(y == 0) & denied].mean(), flag[(y == 0) & ~denied].mean()))
            row["lies flagged"], row["honest denials flagged"], row["honest claims flagged"] = np.mean(flags, axis=0)
            hp_rows.append(row)
    hpkr = pd.DataFrame(hp_rows)
    hpkr.round(4).to_csv(OUT / "hpkr.csv", index=False)
    logger.info(f"HP-KR:\n{hpkr.round(3).to_string(index=False)}")

    # diagnostics: cosines with the bank and the wording direction, pi, head movement
    wording = np.load(FIX1 / "directions.npz")[f"answer|L{LAYER}|d_wording"]
    diag = []
    for r in records:
        if r["permutation"] is not None:
            continue
        d = trained[r["id"]]
        ref = comparators["full bank (mean)"][r["shape"]]
        same = comparators["mean extraction (same rollouts)"][r["shape"]]
        diag.append({"id": r["id"], "pi": np.round(r["pi"], 2).tolist(), "cos full bank": float(d @ ref),
                     "cos mean extraction": float(d @ same), "cos wording": float(d @ wording)})
    diag = pd.DataFrame(diag)
    diag.round(3).to_csv(OUT / "diagnostics.csv", index=False)
    logger.info(f"diagnostics:\n{diag.round(3).to_string(index=False)}")


if __name__ == "__main__":
    fire.Fire({"grams": grams, "fit": fit, "directions": directions, "evaluate": evaluate})
