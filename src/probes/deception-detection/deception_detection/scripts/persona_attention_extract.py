"""Attention pooling when extracting the persona vectors, with a query trained on the bank alone.

A persona vector is a mean over the role's rollouts of a mean over each rollout's response tokens.
Here the inner mean becomes an attention: a_t = softmax_t(q . x_t), pooled vector sum_t a_t x_t. The
query q is trained on the persona bank only. Liars' Bench never fits, selects or applies it: test
examples stay mean-pooled (and max-pooled, the current best read). The persona-coordinates paper
(2605.09391, section 7) names this as untested.

- **Where q lives.** In persona coordinates: q = P u, with P the top-k PCs of the centred
  mean-pooled persona vectors, standardised per coordinate. It can weight a token only by its
  resemblance to the personas, never by format. u = 0 is mean pooling exactly.
- **Objective, through the contrast.** For each behaviour pair, build the contrast from half of
  its personas (re-pooled with q) and ask it to separate the other half's rollouts; all three pairs
  at once, over a fixed set of random halves. Plus a ridge toward uniform, an entropy floor, and
  stability terms (the contrast from even against odd questions, and across prompt variants).
- **Guards.** Response tokens only; tokens that echo the role instruction (a token 3-gram shared
  with the system prompt, or a word of the role's name) are masked.
- **Selection inside the bank.** Outer leave-personas-out folds choose the ridge and the floor by
  held-out-persona AUROC; a leave-one-behaviour-out gate asks whether q helps a behaviour it never
  saw. Then q is refit on everything.
- **Arms** from the same tokens: mean (control), mean with the echo mask, last token, self-weighted
  by the current axis (label-free), learned q. Each gives a bank folder at layer 40.

    python deception_detection/scripts/persona_attention_extract.py read --stage 1   # NDIF
    python deception_detection/scripts/persona_attention_extract.py basis
    python deception_detection/scripts/persona_attention_extract.py fit
    python deception_detection/scripts/persona_attention_extract.py extract
    python deception_detection/scripts/persona_attention_extract.py evaluate
    python deception_detection/scripts/persona_attention_extract.py transfer

Outputs go to `results/persona_attention_extract_L40/`; tokens to `PERSONA_TOKENS_DIR`.
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
from sklearn.metrics import roc_auc_score

from deception_detection import persona_pcs as pcs
from deception_detection.log import logger
from deception_detection.scripts.persona_attention import segment_softmax
from deception_detection.scripts.persona_finetuned import RUN_DIR, assistant_span, batches
from deception_detection.utils import REPO_ROOT

LAYER = 40
OUT = REPO_ROOT / "results" / f"persona_attention_extract_L{LAYER}"
TOKENS = Path(os.environ.get("PERSONA_TOKENS_DIR", "E:/liars-bench-acts/persona_tokens")) / f"L{LAYER}"
BANKS = REPO_ROOT / "persona attention pooled"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
STAGES = {1: range(10), 2: range(30)}  # question indices read at each stage
SEED = 0
K_PERSONA = 10  # persona coordinates the query lives in
N_WITHIN = 1024  # within-rollout PCs in the working basis
N_SPLITS = 8  # fixed build/score halvings per pair in the loss
N_FOLDS = 5
LAMBDAS = (1.0, 0.1, 0.01)
FLOORS = (True, False)
N_GATE_SPLITS = 20
MIN_COSINE = 0.9999  # check A: gathered tokens against the same pass's span mean
N_NULL = 2000
N_BOOT = 1000
CONTROL = "XXXX-20__plain"
IDENTICAL = "I do not know the answer."

PAIRS = {  # behaviour pair -> (deceptive side, honest side)
    "deception": (pcs.DECEPTION_PERSONAS, pcs.HONESTY_PERSONAS),
    "sycophancy": (pcs.SYCOPHANCY_PERSONAS, pcs.ANTI_SYCOPHANCY_PERSONAS),
    "knowledge": (pcs.KNOWLEDGE_DENIAL_PERSONAS, pcs.KNOWLEDGE_CANDOUR_PERSONAS),
}
SINCERE = ("pc_calibrated_reporter", "pc_honest_novice", "pc_scrupulous_witness")
GROUPS = {  # for the folds: one persona per group held out at a time, where the group allows
    "deception": pcs.DECEPTION_PERSONAS, "honesty": pcs.HONESTY_PERSONAS,
    "sycophancy": pcs.SYCOPHANCY_PERSONAS, "anti_sycophancy": pcs.ANTI_SYCOPHANCY_PERSONAS,
    "knowledge_denial": pcs.KNOWLEDGE_DENIAL_PERSONAS, "sincere_ignorance": SINCERE,
    "plain_assertion": tuple(p for p in pcs.KNOWLEDGE_CANDOUR_PERSONAS if p not in SINCERE),
}
ROLES = tuple(sorted({p for g in GROUPS.values() for p in g}))  # the 35 roles
PERSONAS = ROLES + ("default",)
CONTRASTS = {"published (8v7)": "deception", "knowledge-denial (4v6)": "denial",
             "extended (12v13)": "extended", "sycophancy (5v5)": "sycophancy"}
ARMS = ("mean", "mean masked", "last token", "self-weighted", "learned")


# --- rollouts ----------------------------------------------------------------------------------

def kept(persona: str, run_dir: Path = RUN_DIR) -> list[dict]:
    """The rollouts the bank averaged: judge score 3, or every row for `default`, which is unjudged."""
    scores_path = run_dir / "scores" / f"{persona}.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.exists() else None
    rows = []
    with open(run_dir / "responses" / f"{persona}.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if scores is None or scores.get(f"pos_p{row['prompt_index']}_q{row['question_index']}") == 3:
                rows.append(row)
    return rows


def select(questions, seed: int = SEED, run_dir: Path = RUN_DIR) -> list[dict]:
    """One kept rollout per (persona, question), the draw seeded per pair so a later stage with more
    questions keeps an earlier stage's choices."""
    chosen = []
    for i, persona in enumerate(PERSONAS):
        by_question: dict[int, list[dict]] = {}
        for row in kept(persona, run_dir):
            by_question.setdefault(int(row["question_index"]), []).append(row)
        for q in questions:
            if q in by_question:
                rows = by_question[q]
                pick = np.random.default_rng([seed, i, q]).integers(len(rows))
                chosen.append({"persona": persona, "prompt_index": int(rows[pick]["prompt_index"]),
                               "question_index": q, "conversation": rows[pick]["conversation"]})
    return chosen


def _key(row: dict) -> tuple:
    return row["persona"], int(row["prompt_index"]), int(row["question_index"])


# --- the token store ----------------------------------------------------------------------------

def _index_path() -> Path:
    return TOKENS / "index.json"


def load_index() -> dict:
    if _index_path().exists():
        return json.loads(_index_path().read_text(encoding="utf-8"))
    return {"layer": LAYER, "hidden": None, "rows": []}


def _write_index(index: dict) -> None:
    tmp = _index_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(index), encoding="utf-8")
    os.replace(tmp, _index_path())


def token_memmap(index: dict) -> np.ndarray:
    n = sum(r["length"] for r in index["rows"])
    return np.memmap(TOKENS / "tokens.bits", dtype=np.int16, mode="r", shape=(n, index["hidden"]))


def rows_float(bits: np.ndarray, start: int, stop: int) -> np.ndarray:
    return torch.from_numpy(np.array(bits[start:stop])).view(torch.bfloat16).float().numpy()


def flat_index(spans: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Row and position of every span token, rows in order: what `trace_span_tokens` gathers."""
    rows = np.concatenate([np.full(a1 - a0, i) for i, (a0, a1) in enumerate(spans)])
    positions = np.concatenate([np.arange(a0, a1) for a0, a1 in spans])
    return rows, positions


def read(stage: int = 1, token_budget: int = 16000, max_rows: int = 24, retries: int = 3) -> None:
    """Gather every selected rollout's response tokens at LAYER from NDIF, appending per request."""
    from deception_detection.activations import trace_span_tokens
    from deception_detection.scripts.gender_secret import _environment, _model

    TOKENS.mkdir(parents=True, exist_ok=True)
    index = load_index()
    done = {(r["persona"], r["prompt_index"], r["question_index"]) for r in index["rows"]}
    todo = [r for r in select(STAGES[int(stage)]) if _key(r) not in done]
    logger.info(f"stage {stage}: {len(todo)} rollouts to read ({len(done)} already stored)")
    if not todo:
        return
    _environment()
    model = _model()
    tokenizer = model.tokenizer
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    items = [assistant_span(tokenizer, r["conversation"]) for r in todo]
    groups = batches([len(ids) for ids, _, _ in items], token_budget, max_rows)
    with open(TOKENS / "tokens.bits", "ab") as bits:
        for k, group in enumerate(groups):
            width = max(len(items[i][0]) for i in group)
            tokens = torch.full((len(group), width), pad, dtype=torch.long)
            attention = torch.zeros((len(group), width), dtype=torch.long)
            spans = []
            for row, i in enumerate(group):  # right padding, as the bank was read
                ids, a0, a1 = items[i]
                tokens[row, : len(ids)] = torch.tensor(ids)
                attention[row, : len(ids)] = 1
                spans.append((a0, a1))
            b, p = flat_index(spans)
            check = not (TOKENS / "check_a.json").exists()
            span_mask = None
            if check:
                span_mask = torch.zeros((len(group), width), dtype=torch.float32)
                for row, (a0, a1) in enumerate(spans):
                    span_mask[row, a0:a1] = 1.0
            start = time.time()
            for attempt in range(1, retries + 1):
                try:
                    got = trace_span_tokens(model, tokens, attention, LAYER, b, p, span_mask=span_mask)
                    break
                except Exception as error:  # NDIF drops a request now and then
                    if attempt == retries:
                        raise
                    logger.warning(f"request {k + 1} failed ({error!r}); retry {attempt}/{retries - 1}")
                    time.sleep(30 * attempt)
            gathered, means = got if check else (got, None)
            if gathered.dtype != torch.bfloat16:
                raise TypeError(f"expected bfloat16 from the server, got {gathered.dtype}")
            gathered = gathered.cpu()
            if check:
                _check_a(gathered, means.cpu(), spans)
            offset = sum(r["length"] for r in index["rows"])
            bits.write(gathered.contiguous().view(torch.int16).numpy().tobytes())
            bits.flush()
            os.fsync(bits.fileno())
            index["hidden"] = int(gathered.shape[1])
            for row, i in enumerate(group):
                a0, a1 = spans[row]
                index["rows"].append({"persona": todo[i]["persona"], "prompt_index": todo[i]["prompt_index"],
                                      "question_index": todo[i]["question_index"], "offset": offset,
                                      "length": a1 - a0})
                offset += a1 - a0
            _write_index(index)
            logger.info(f"request {k + 1}/{len(groups)}: {len(group)} rollouts, {len(b)} tokens, "
                        f"{time.time() - start:.0f} s; {len(index['rows'])} rollouts stored")


def _check_a(gathered: torch.Tensor, means: torch.Tensor, spans: list[tuple[int, int]]) -> None:
    """Check A: the gathered tokens of each row average to the same pass's fp32 span mean."""
    checks, start = [], 0
    for row, (a0, a1) in enumerate(spans):
        mine = gathered[start:start + (a1 - a0)].float().mean(0)
        start += a1 - a0
        theirs = means[row].float()
        checks.append({"row": row, "tokens": a1 - a0,
                       "cosine": float(mine @ theirs / (mine.norm() * theirs.norm())),
                       "relative_error": float((mine - theirs).norm() / theirs.norm())})
    (TOKENS / "check_a.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    worst = min(c["cosine"] for c in checks)
    logger.info(f"check A: gathered tokens v span mean, worst cosine {worst:.6f} over {len(checks)} rows")
    if worst < MIN_COSINE:
        raise RuntimeError(f"check A failed: {checks}")


# --- the working basis --------------------------------------------------------------------------

@dataclass
class Bank:
    """Everything the fit needs, in memory."""

    rows: pd.DataFrame  # one per rollout: persona, prompt_index, question_index, offset, length
    tok_roll: np.ndarray  # [N] rollout of each token
    keep: np.ndarray  # [N] bool: unmasked response tokens
    coords: np.ndarray  # [N, k] float32: standardised persona coordinates (the query's inputs)
    xb: np.ndarray  # [N, m] float32: coordinates in the working basis
    persona_of: np.ndarray  # [R] int index into PERSONAS

    @property
    def n_rollouts(self) -> int:
        return len(self.rows)


def _tokenizer():
    from transformers import AutoTokenizer

    from deception_detection.scripts.gender_secret import _environment

    _environment()
    return AutoTokenizer.from_pretrained(MODEL)


def echo_mask(response_ids: list[int], system_ids: list[int], name_words: set[str], tokenizer) -> np.ndarray:
    """True for response tokens to exclude: inside a token 3-gram that also occurs in the system
    prompt, or spelling a word of the role's name."""
    grams = {tuple(system_ids[i:i + 3]) for i in range(len(system_ids) - 2)}
    masked = np.zeros(len(response_ids), dtype=bool)
    for i in range(len(response_ids) - 2):
        if tuple(response_ids[i:i + 3]) in grams:
            masked[i:i + 3] = True
    if name_words:
        for i, t in enumerate(response_ids):
            word = "".join(ch for ch in tokenizer.decode([t]).lower() if ch.isalpha())
            if word and word in name_words:
                masked[i] = True
    return masked


def basis(k: int = K_PERSONA, n_within: int = N_WITHIN) -> None:
    """Persona coordinates, the working basis, the echo mask and the token ids, from the stored tokens."""
    from scipy.linalg import eigh

    index = load_index()
    bits = token_memmap(index)
    rows = pd.DataFrame(index["rows"])
    hidden = index["hidden"]
    n_tokens = len(bits)
    logger.info(f"{len(rows)} rollouts, {n_tokens} tokens")

    # persona coordinates from the existing bank
    bank = pcs.load_bank_at_layer(LAYER, pcs.EXTENDED_VECTORS_DIR)
    centred = bank.personas - bank.personas.mean(0)
    u_svd, s_svd, _ = np.linalg.svd(centred, full_matrices=False)
    span = (centred.T @ u_svd[:, :35]) / s_svd[:35]  # orthonormal basis of the persona span [hidden, 35]
    P = span[:, :k]  # the top-k PCs, in variance order

    # token ids, masks and the streamed statistics
    tokenizer = _tokenizer()
    tok_roll = np.repeat(np.arange(len(rows)), rows["length"].to_numpy())
    keep = np.ones(n_tokens, dtype=bool)
    token_ids = np.zeros(n_tokens, dtype=np.int32)
    by_key = {}
    for r in select(STAGES[2]):
        by_key[_key(r)] = r["conversation"]
    cov = np.zeros((hidden, hidden), dtype=np.float64)
    coords_raw = np.zeros((n_tokens, k), dtype=np.float32)
    span_coords = np.zeros((n_tokens, span.shape[1]), dtype=np.float32)
    for i, row in rows.iterrows():
        conversation = by_key[(row["persona"], int(row["prompt_index"]), int(row["question_index"]))]
        ids, a0, a1 = assistant_span(tokenizer, conversation)
        if a1 - a0 != row["length"]:
            raise RuntimeError(f"rollout {i}: stored {row['length']} tokens, span rule gives {a1 - a0}")
        system = next((m["content"] for m in conversation if m["role"] == "system"), "")
        system_ids = tokenizer(system, add_special_tokens=False).input_ids if system else []
        words = set(row["persona"].removeprefix("pc_").split("_")) if row["persona"] != "default" else set()
        a, b = int(row["offset"]), int(row["offset"] + row["length"])
        keep[a:b] = ~echo_mask(list(ids[a0:a1]), system_ids, words, tokenizer)
        token_ids[a:b] = ids[a0:a1]
        x = rows_float(bits, a, b)
        coords_raw[a:b] = x @ P
        span_coords[a:b] = x @ span
        xc = x - x.mean(0, keepdims=True)
        cov += xc.T.astype(np.float64) @ xc.astype(np.float64)
        if i % 100 == 0:
            logger.info(f"  rollout {i}/{len(rows)}")
    values, vectors = eigh(cov, subset_by_index=[hidden - n_within, hidden - 1])
    order = np.argsort(values)[::-1]
    within = vectors[:, order]
    share = float(values.sum() / np.trace(cov))
    # the working basis: the persona span plus the within-rollout PCs, orthonormalised
    combined, _ = np.linalg.qr(np.concatenate([span, within], axis=1))
    logger.info(f"within-rollout PCs: top {n_within} hold {share:.1%} of the within-rollout variance; "
                f"working basis has {combined.shape[1]} columns; {1 - keep.mean():.1%} of tokens masked")
    tau = coords_raw[keep].std(0) + 1e-12
    xb = np.zeros((n_tokens, combined.shape[1]), dtype=np.float32)
    for i, row in rows.iterrows():
        a, b = int(row["offset"]), int(row["offset"] + row["length"])
        xb[a:b] = rows_float(bits, a, b) @ combined
    np.save(TOKENS / "xb.npy", xb)
    np.savez(TOKENS / "basis.npz", P=P.astype(np.float32), tau=tau.astype(np.float32), B=combined.astype(np.float32),
             within_share=share, span=span.astype(np.float32))
    np.save(TOKENS / "coords.npy", (coords_raw / tau[None, :]).astype(np.float32))
    np.save(TOKENS / "keep.npy", keep)
    np.save(TOKENS / "token_ids.npy", token_ids)
    logger.info(f"basis written to {TOKENS}")


def load_bank() -> Bank:
    index = load_index()
    rows = pd.DataFrame(index["rows"])
    return Bank(rows=rows, tok_roll=np.repeat(np.arange(len(rows)), rows["length"].to_numpy()),
                keep=np.load(TOKENS / "keep.npy"), coords=np.load(TOKENS / "coords.npy"),
                xb=np.load(TOKENS / "xb.npy", mmap_mode="r"),
                persona_of=rows["persona"].map({p: i for i, p in enumerate(PERSONAS)}).to_numpy())


# --- the model ---------------------------------------------------------------------------------

def attention(u: torch.Tensor, coords: torch.Tensor, keep: torch.Tensor, tok_roll: torch.Tensor,
              n_rollouts: int) -> torch.Tensor:
    """a_t = softmax over each rollout's unmasked tokens of coords_t . u; masked tokens get 0."""
    logits = (coords @ u).masked_fill(~keep, -torch.inf)
    return segment_softmax(logits, tok_roll, n_rollouts)


def pool_rollouts(a: torch.Tensor, xb: torch.Tensor, offsets: np.ndarray) -> torch.Tensor:
    """z_r = sum_t a_t x_t, one small matmul per rollout so no token-sized product is kept."""
    return torch.stack([a[o0:o1] @ xb[o0:o1] for o0, o1 in zip(offsets[:-1], offsets[1:])])


def entropy_penalty(a: torch.Tensor, tok_roll: torch.Tensor, keep: torch.Tensor, n_rollouts: int) -> torch.Tensor:
    """Squared hinge below log(T) - 1 nats, T the rollout's unmasked tokens; mean over rollouts."""
    plogp = torch.where(a > 0, a * torch.log(a.clamp_min(1e-30)), torch.zeros_like(a))
    h = -torch.zeros(n_rollouts, dtype=a.dtype).index_add(0, tok_roll, plogp)
    t = torch.zeros(n_rollouts, dtype=a.dtype).index_add(0, tok_roll, keep.to(a.dtype))
    floor = torch.log(t.clamp_min(1.0)) - 1.0
    return (torch.relu(floor - h) ** 2).mean()


def persona_means(z: torch.Tensor, persona_of: torch.Tensor, n_personas: int) -> torch.Tensor:
    counts = torch.zeros(n_personas, dtype=z.dtype).index_add(0, persona_of, torch.ones(len(persona_of), dtype=z.dtype))
    return torch.zeros((n_personas, z.shape[1]), dtype=z.dtype).index_add(0, persona_of, z) / counts.clamp_min(1)[:, None]


def contrast(v: torch.Tensor, positive: list[int], negative: list[int]) -> torch.Tensor:
    return v[positive].mean(0) - v[negative].mean(0)


def halvings(personas_pos: list[str], personas_neg: list[str], n: int, rng: np.random.Generator):
    """`n` random build/score splits, each side halved (a side of 3 gives 2 v 1)."""
    out = []
    for _ in range(n):
        split = {}
        for side, names in (("pos", personas_pos), ("neg", personas_neg)):
            perm = list(rng.permutation(names))
            cut = max(1, len(perm) // 2)
            split[side] = (perm[:cut], perm[cut:])
        out.append(split)
    return out


def folds(seed: int = SEED, n_folds: int = N_FOLDS) -> list[list[str]]:
    """Held-out personas per fold: every role held out exactly once, folds balanced within group."""
    rng = np.random.default_rng(seed)
    held = [[] for _ in range(n_folds)]
    for group in GROUPS.values():
        for i, persona in enumerate(rng.permutation(list(group))):
            held[i % n_folds].append(str(persona))
    return held


class Objective:
    """The loss for one set of training personas, with fixed build/score halvings."""

    def __init__(self, bank: Bank, training: set[str], pairs: dict, floor: bool, lam: float,
                 stability: float = 1.0, seed: int = SEED):
        self.bank, self.lam, self.floor, self.stability = bank, lam, floor, stability
        self.idx = {p: i for i, p in enumerate(PERSONAS)}
        rng = np.random.default_rng(seed)
        self.pairs = {}
        for name, (pos, neg) in pairs.items():
            pos_t, neg_t = [p for p in pos if p in training], [p for p in neg if p in training]
            self.pairs[name] = (pos_t, neg_t, halvings(pos_t, neg_t, N_SPLITS, rng))
        self.coords = torch.from_numpy(bank.coords.astype(np.float64))
        self.keep = torch.from_numpy(bank.keep)
        self.tok_roll = torch.from_numpy(bank.tok_roll.astype(np.int64))
        self.xb = torch.from_numpy(np.asarray(bank.xb))
        self.offsets = np.concatenate([[0], np.cumsum(bank.rows["length"].to_numpy())])
        self.persona_of = torch.from_numpy(bank.persona_of.astype(np.int64))
        self.question = bank.rows["question_index"].to_numpy()
        self.prompt = bank.rows["prompt_index"].to_numpy()
        self.n_p = len(PERSONAS)
        self.scale = {name: torch.ones((), dtype=torch.float64, requires_grad=True) for name in self.pairs}
        self.bias = {name: torch.zeros((), dtype=torch.float64, requires_grad=True) for name in self.pairs}
        with torch.no_grad():
            self.stats = self._score_stats(torch.zeros(bank.coords.shape[1], dtype=torch.float64))

    def pooled(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = attention(u, self.coords, self.keep, self.tok_roll, self.bank.n_rollouts)
        z = pool_rollouts(a.float(), self.xb, self.offsets).double()
        return a, z

    def _pair_scores(self, z: torch.Tensor, name: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per halving: (scores of the score personas' rollouts, their labels)."""
        pos, neg, splits = self.pairs[name]
        v = persona_means(z, self.persona_of, self.n_p)
        out = []
        for split in splits:
            build_pos, score_pos = split["pos"]
            build_neg, score_neg = split["neg"]
            d = contrast(v, [self.idx[p] for p in build_pos], [self.idx[p] for p in build_neg])
            score_ids = [self.idx[p] for p in score_pos + score_neg]
            rows = torch.isin(self.persona_of, torch.tensor(score_ids))
            labels = torch.isin(self.persona_of[rows], torch.tensor([self.idx[p] for p in score_pos])).double()
            out.append((z[rows] @ d, labels))
        return out

    def _score_stats(self, u: torch.Tensor) -> dict:
        _, z = self.pooled(u)
        return {name: [(s.mean(), s.std() + 1e-12) for s, _ in self._pair_scores(z, name)] for name in self.pairs}

    def loss(self, u: torch.Tensor) -> torch.Tensor:
        a, z = self.pooled(u)
        total = torch.zeros((), dtype=torch.float64)
        for name in self.pairs:
            pair_loss = torch.zeros((), dtype=torch.float64)
            for (s, y), (mu, sd) in zip(self._pair_scores(z, name), self.stats[name]):
                weight = torch.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
                logit = self.scale[name] * (s - mu) / sd + self.bias[name]
                pair_loss = pair_loss + torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, y, weight=weight, reduction="sum") / weight.sum()
            total = total + pair_loss / N_SPLITS + 1e-3 * (self.scale[name] ** 2 + self.bias[name] ** 2)
            if self.stability:
                total = total + self.stability * self._instability(z, name)
        total = total + self.lam * (u ** 2).sum()
        if self.floor:
            total = total + entropy_penalty(a, self.tok_roll, self.keep, self.bank.n_rollouts)
        return total

    def _instability(self, z: torch.Tensor, name: str) -> torch.Tensor:
        """1 - cos between the pair's contrasts from even and odd questions, and from even and odd
        prompt variants, over all its training personas."""
        pos, neg, _ = self.pairs[name]
        pos_i, neg_i = [self.idx[p] for p in pos], [self.idx[p] for p in neg]
        total = torch.zeros((), dtype=torch.float64)
        for field in (self.question, self.prompt):
            parts = []
            for parity in (0, 1):
                rows = torch.from_numpy(field % 2 == parity)
                v = persona_means(z[rows], self.persona_of[rows], self.n_p)
                parts.append(contrast(v, pos_i, neg_i))
            total = total + 1 - torch.nn.functional.cosine_similarity(parts[0], parts[1], dim=0)
        return total

    def fit(self, max_iter: int = 60) -> torch.Tensor:
        u = torch.zeros(self.coords.shape[1], dtype=torch.float64, requires_grad=True)
        params = [u] + list(self.scale.values()) + list(self.bias.values())
        opt = torch.optim.LBFGS(params, lr=1, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            value = self.loss(u)
            value.backward()
            return value

        opt.step(closure)
        return u.detach()

    def separation(self, u: torch.Tensor, name: str, build: list[str], score: list[str]) -> float:
        """AUROC of the pair's contrast built from `build` personas on `score` personas' rollouts."""
        with torch.no_grad():
            _, z = self.pooled(u)
            v = persona_means(z, self.persona_of, self.n_p)
            pos, neg = PAIRS[name]
            d = contrast(v, [self.idx[p] for p in build if p in pos], [self.idx[p] for p in build if p in neg])
            rows = torch.isin(self.persona_of, torch.tensor([self.idx[p] for p in score]))
            y = torch.isin(self.persona_of[rows], torch.tensor([self.idx[p] for p in score if p in pos])).numpy()
            s = (z[rows] @ d).numpy()
        if 0 < y.sum() < len(y):
            return roc_auc_score(y, s)
        return float("nan")


def fit(max_iter: int = 60) -> None:
    """Outer folds over personas for selection, the leave-one-behaviour-out gate, the final refit."""
    OUT.mkdir(parents=True, exist_ok=True)
    bank = load_bank()
    logger.info(f"{bank.n_rollouts} rollouts, {len(bank.keep)} tokens, {bank.coords.shape[1]} persona coordinates, "
                f"working basis {bank.xb.shape[1]}")
    held = folds()
    all_roles = set(ROLES)
    selection, fold_queries = [], {}
    for lam in LAMBDAS:
        for floor in FLOORS:
            out_of_fold = {name: ([], []) for name in PAIRS}  # z-scored held-out scores and labels
            out_of_fold_mean = {name: ([], []) for name in PAIRS}
            queries = []
            for f, held_out in enumerate(held):
                training = all_roles - set(held_out)
                objective = Objective(bank, training, PAIRS, floor, lam)
                u = objective.fit(max_iter)
                queries.append(u.numpy())
                for name, (pos, neg) in PAIRS.items():
                    train_pair = [p for p in pos + neg if p in training]
                    held_pair = [p for p in pos + neg if p in held_out]
                    if not held_pair:
                        continue
                    for store, query in ((out_of_fold, u), (out_of_fold_mean, torch.zeros_like(u))):
                        s_train, y_train = _scores(objective, query, name, train_pair, train_pair)
                        s_held, y_held = _scores(objective, query, name, train_pair, held_pair)
                        mu, sd = s_train.mean(), s_train.std() + 1e-12
                        store[name][0].extend(((s_held - mu) / sd).tolist())
                        store[name][1].extend(y_held.tolist())
                logger.info(f"  lambda {lam:g} floor {floor} fold {f}: |u| {np.linalg.norm(u.numpy()):.3f}")
            row = {"lambda": lam, "floor": floor}
            for name in PAIRS:
                s, y = map(np.array, out_of_fold[name])
                s0, y0 = map(np.array, out_of_fold_mean[name])
                row[f"{name} learned"] = roc_auc_score(y, s)
                row[f"{name} mean"] = roc_auc_score(y0, s0)
            row["learned"] = np.mean([row[f"{n} learned"] for n in PAIRS])
            row["mean"] = np.mean([row[f"{n} mean"] for n in PAIRS])
            q = np.stack(queries)
            norms = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
            cos = (q / norms) @ (q / norms).T
            row["fold query cosine"] = float(cos[np.triu_indices(len(q), 1)].mean())
            selection.append(row)
            fold_queries[(lam, floor)] = q
            logger.info(f"lambda {lam:g} floor {floor}: held-out-persona AUROC learned {row['learned']:.3f} "
                        f"v mean {row['mean']:.3f} ({ {n: round(row[f'{n} learned'], 3) for n in PAIRS} })")
    table = pd.DataFrame(selection)
    table.round(4).to_csv(OUT / "selection.csv", index=False)
    best = table.sort_values(["learned", "lambda"], ascending=[False, False]).iloc[0]
    lam, floor = float(best["lambda"]), bool(best["floor"])
    logger.info(f"selected lambda {lam:g}, floor {floor}\n{table.round(3).to_string(index=False)}")

    # leave-one-behaviour-out gate
    gate = []
    rng = np.random.default_rng(SEED + 1)
    for held_pair in PAIRS:
        others = {n: PAIRS[n] for n in PAIRS if n != held_pair}
        training = {p for n in others for side in others[n] for p in side}
        objective = Objective(bank, training, others, floor, lam)
        u = objective.fit(max_iter)
        pos, neg = PAIRS[held_pair]
        aurocs = {"learned": [], "mean": []}
        for split in halvings(list(pos), list(neg), N_GATE_SPLITS, rng):
            build = split["pos"][0] + split["neg"][0]
            score = split["pos"][1] + split["neg"][1]
            aurocs["learned"].append(objective.separation(u, held_pair, build, score))
            aurocs["mean"].append(objective.separation(torch.zeros_like(u), held_pair, build, score))
        gate.append({"held-out behaviour": held_pair, "learned": np.nanmean(aurocs["learned"]),
                     "mean": np.nanmean(aurocs["mean"])})
    gate = pd.DataFrame(gate)
    gate.round(4).to_csv(OUT / "gate.csv", index=False)
    logger.info(f"leave-one-behaviour-out gate:\n{gate.round(3).to_string(index=False)}")

    # final refit on all personas and pairs
    objective = Objective(bank, all_roles, PAIRS, floor, lam)
    u = objective.fit(max_iter).numpy()
    saved = np.load(TOKENS / "basis.npz")
    q = saved["P"] @ (u / saved["tau"])  # the query in activation space
    np.savez(OUT / "query.npz", u=u, q=q, lam=lam, floor=floor, fold_queries=fold_queries[(lam, floor)])
    with torch.no_grad():
        a = attention(torch.from_numpy(u), objective.coords, objective.keep, objective.tok_roll, bank.n_rollouts)
    np.save(OUT / "weights_learned.npy", a.numpy().astype(np.float32))
    logger.info(f"final query: |u| {np.linalg.norm(u):.3f}; u = {np.round(u, 3).tolist()}")


def _scores(objective: Objective, u: torch.Tensor, name: str, build: list[str], score: list[str]):
    with torch.no_grad():
        _, z = objective.pooled(u)
        v = persona_means(z, objective.persona_of, objective.n_p)
        pos, neg = PAIRS[name]
        d = contrast(v, [objective.idx[p] for p in build if p in pos], [objective.idx[p] for p in build if p in neg])
        rows = torch.isin(objective.persona_of, torch.tensor([objective.idx[p] for p in score]))
        y = torch.isin(objective.persona_of[rows], torch.tensor([objective.idx[p] for p in score if p in pos]))
        return (z[rows] @ d).numpy(), y.numpy().astype(int)


# --- extraction of the arms, exact, from the full tokens ------------------------------------------

def _weights_uniform(keep_r: np.ndarray, use_mask: bool) -> np.ndarray:
    w = keep_r.astype(np.float64) if use_mask else np.ones(len(keep_r))
    return w / w.sum()


def _pool_all(bank: Bank, bits: np.ndarray, weight_fn) -> np.ndarray:
    """Per-rollout pooled vectors [R, hidden] for weights given by `weight_fn(i, x, keep_r)`."""
    out = np.zeros((bank.n_rollouts, bits.shape[1]), dtype=np.float32)
    for i, row in bank.rows.iterrows():
        a, b = int(row["offset"]), int(row["offset"] + row["length"])
        x = rows_float(bits, a, b)
        out[i] = weight_fn(i, x, bank.keep[a:b]) @ x
    return out


def _persona_vectors(pooled: np.ndarray, bank: Bank) -> np.ndarray:
    v = np.zeros((len(PERSONAS), pooled.shape[1]), dtype=np.float64)
    for i in range(len(PERSONAS)):
        v[i] = pooled[bank.persona_of == i].mean(0)
    return v.astype(np.float32)


def _axis(vectors: np.ndarray, positive, negative) -> np.ndarray:
    idx = {p: i for i, p in enumerate(PERSONAS)}
    d = vectors[[idx[p] for p in positive]].mean(0) - vectors[[idx[p] for p in negative]].mean(0)
    return d / np.linalg.norm(d)


def write_bank(vectors: np.ndarray, counts: np.ndarray, arm: str) -> Path:
    folder = BANKS / arm
    folder.mkdir(parents=True, exist_ok=True)
    for i, persona in enumerate(PERSONAS):
        torch.save({"vector": torch.from_numpy(vectors[i][None, :]), "layers": [LAYER], "type": arm,
                    "role": persona, "n": int(counts[i])}, folder / f"{persona}.pt")
    return folder


def extract(beta: float = 1.0) -> None:
    """The five arms' persona vectors from the stored tokens, and a bank folder for each."""
    bank = load_bank()
    bits = token_memmap(load_index())
    counts = np.bincount(bank.persona_of, minlength=len(PERSONAS))
    saved = np.load(TOKENS / "basis.npz")
    P, tau = saved["P"].astype(np.float64), saved["tau"].astype(np.float64)
    u = np.load(OUT / "query.npz")["u"]
    pooled = {}
    pooled["mean"] = _pool_all(bank, bits, lambda i, x, k: _weights_uniform(k, False))
    pooled["mean masked"] = _pool_all(bank, bits, lambda i, x, k: _weights_uniform(k, True))
    pooled["last token"] = _pool_all(bank, bits, lambda i, x, k: np.eye(len(k))[-1])

    def learned(i, x, k):
        logits = ((x @ P) / tau) @ u
        logits[~k] = -np.inf
        w = np.exp(logits - logits.max())
        return w / w.sum()

    pooled["learned"] = _pool_all(bank, bits, learned)
    # self-weighted: softmax(beta . axis . x / sigma), the axis rebuilt after each pass
    axis = _axis(_persona_vectors(pooled["mean"], bank), pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS,
                 pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS)
    current = None
    for it in range(2):
        scores = np.concatenate([rows_float(bits, a, a + n) @ axis
                                 for a, n in zip(bank.rows["offset"], bank.rows["length"])])
        sigma = scores.std() + 1e-12

        def self_weighted(i, x, k, scores=scores, sigma=sigma):
            a = int(bank.rows.at[i, "offset"])
            s = scores[a:a + len(k)] * (beta / sigma)
            w = np.exp(s - s.max())
            return w / w.sum()

        current = _pool_all(bank, bits, self_weighted)
        axis = _axis(_persona_vectors(current, bank), pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS,
                     pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS)
        logger.info(f"self-weighted pass {it + 1}: axis rebuilt")
    pooled["self-weighted"] = current
    for arm, z in pooled.items():
        np.save(OUT / f"pooled_{arm.replace(' ', '_')}.npy", z)
        vectors = _persona_vectors(z, bank)
        folder = write_bank(vectors, counts, arm)
        logger.info(f"{arm}: bank written to {folder}")


# --- evaluation --------------------------------------------------------------------------------

def _directions(arm: str) -> dict[str, np.ndarray]:
    bank = pcs.load_bank_at_layer(LAYER, BANKS / arm)
    return {label: bank.axis_unit(axis) for label, axis in CONTRASTS.items()}


def evaluate() -> None:
    from deception_detection.scripts import persona_ood as P
    from deception_detection.scripts.forced_denial_bank import OUT as FIX1
    from deception_detection.scripts.hpkr_wording import auroc_ci, hpkr_frame
    from deception_detection.scripts.knowledge_bank import _slices
    from deception_detection.scripts.layer_sweep import pooled_dir
    from deception_detection.scripts.persona_attention import load as load_answer_tokens

    bank_meta = load_bank()
    pooled_sets = P.load_pooled(pooled_dir(LAYER))
    tests = [n for n, d in sorted(pooled_sets.items()) if d.two_class and n != CONTROL]
    directions = {arm: _directions(arm) for arm in ARMS}
    directions["bank (all rollouts, mean)"] = {
        label: pcs.load_bank_at_layer(LAYER, pcs.EXTENDED_VECTORS_DIR).axis_unit(axis) for label, axis in CONTRASTS.items()}
    arms = list(directions)
    labels = list(CONTRASTS)
    matrix = np.stack([directions[a][l] for a in arms for l in labels], axis=1).astype(np.float32)  # [hidden, arms*labels]

    # zero-shot AUROC, test pooling mean and max
    rows, scores = [], {}
    for name in tests + [CONTROL]:
        z = pooled_sets[name].z if name in pooled_sets else None
        tokens = load_answer_tokens(LAYER, name)
        proj = tokens.project(matrix)
        col = 0
        for a in arms:
            for l in labels:
                s_mean = pooled_sets[name].z @ directions[a][l] if name in pooled_sets else None
                s_max = np.maximum.reduceat(proj[:, col].astype(np.float64), tokens.offsets[:-1])
                col += 1
                scores[(name, a, l, "mean")] = s_mean if s_mean is not None else np.add.reduceat(
                    proj[:, col - 1].astype(np.float64), tokens.offsets[:-1]) / tokens.lengths
                scores[(name, a, l, "max")] = s_max
                if name in pooled_sets:
                    y = pooled_sets[name].y
                    for read in ("mean", "max"):
                        rows.append({"arm": a, "contrast": l, "read": read, "test": name,
                                     "auroc": roc_auc_score(y, scores[(name, a, l, read)])})
        logger.info(f"scored {name}")
    table = pd.DataFrame(rows)
    table.round(4).to_csv(OUT / "zero_shot.csv", index=False)
    summary = table.groupby(["arm", "contrast", "read"])["auroc"].mean().unstack("read")
    logger.info(f"zero-shot mean over {len(tests)} sets:\n{summary.round(3).to_string()}")
    wide = table[table["read"] == "mean"].pivot_table(index=["arm", "contrast"], columns="test", values="auroc")
    logger.info(f"per test set (mean read):\n{wide.round(3).to_string()}")

    # paired bootstrap: learned v mean from the same rollouts, per set, test-time mean pooling
    rng = np.random.default_rng(SEED)
    boot = []
    for l in ("extended (12v13)", "knowledge-denial (4v6)"):
        for name in tests:
            y = pooled_sets[name].y
            a1, a0 = scores[(name, "learned", l, "mean")], scores[(name, "mean", l, "mean")]
            deltas = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, len(y), len(y))
                if 0 < y[idx].sum() < len(idx):
                    deltas.append(roc_auc_score(y[idx], a1[idx]) - roc_auc_score(y[idx], a0[idx]))
            lo, hi = np.quantile(deltas, [0.025, 0.975])
            boot.append({"contrast": l, "test": name, "delta": roc_auc_score(y, a1) - roc_auc_score(y, a0),
                         "lo": lo, "hi": hi, "significant_loss": bool(hi < 0)})
    boot = pd.DataFrame(boot)
    boot.round(4).to_csv(OUT / "bootstrap.csv", index=False)
    logger.info(f"learned minus mean (same rollouts), test-time mean pooling:\n{boot.round(3).to_string(index=False)}")

    # HP-KR slices, null and operating point
    hp = "harm_refusal_knowledge_report__plain"
    frame = hpkr_frame()
    y = pooled_sets[hp].y
    if not (frame["lie"].to_numpy() == y).all():
        raise RuntimeError("HP-KR rows do not line up with hpkr_frame")
    slices = _slices(frame)
    z = pooled_sets[hp].z
    null = rng.standard_normal((N_NULL, z.shape[1]))
    projected = z @ (null / np.linalg.norm(null, axis=1, keepdims=True)).T
    bands = {k: np.array([roc_auc_score(y[m], projected[m, j]) for j in range(N_NULL)]) for k, m in slices.items()
             if k in ("full", "denials", "identical text")}
    hp_rows = []
    denied = frame["denied"].to_numpy()
    for a in arms:
        for l in ("extended (12v13)", "knowledge-denial (4v6)"):
            s, c = scores[(hp, a, l, "mean")], scores[(CONTROL, a, l, "mean")]
            flag = s > np.quantile(c, 0.99)
            row = {"arm": a, "contrast": l}
            for k, band in bands.items():
                m = slices[k]
                auc, lo, hi = auroc_ci(y[m], s[m], rng)
                row[f"{k}"] = auc
                row[f"{k} p"] = float(np.mean(band >= auc))
            row |= {"lies flagged": flag[y == 1].mean(), "honest denials flagged": flag[(y == 0) & denied].mean(),
                    "honest claims flagged": flag[(y == 0) & ~denied].mean()}
            hp_rows.append(row)
    hpkr = pd.DataFrame(hp_rows)
    hpkr.round(4).to_csv(OUT / "hpkr.csv", index=False)
    logger.info(f"HP-KR:\n{hpkr.round(3).to_string(index=False)}")

    # diagnostics: cosines with the bank and with wording, geometry, stability, entropy, top tokens
    wording = np.load(FIX1 / "directions.npz")[f"answer|L{LAYER}|d_wording"]
    diag = []
    for a in arms:
        d = directions[a]
        row = {"arm": a}
        for l in labels:
            row[f"cos bank {l}"] = float(d[l] @ directions["bank (all rollouts, mean)"][l])
            row[f"cos wording {l}"] = float(d[l] @ wording)
        row["cos(deception, knowledge)"] = float(d["published (8v7)"] @ d["knowledge-denial (4v6)"])
        row["cos(deception, sycophancy)"] = float(d["published (8v7)"] @ d["sycophancy (5v5)"])
        if a in ARMS:
            pooled = np.load(OUT / f"pooled_{a.replace(' ', '_')}.npy")
            v = _persona_vectors(pooled, bank_meta)
            for side_name, side in (("deceptive", pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS),
                                    ("honest", pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS)):
                idx = [PERSONAS.index(p) for p in side]
                c = v[idx] - v.mean(0)
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                row[f"within-{side_name} mean cos"] = float((c @ c.T)[np.triu_indices(len(idx), 1)].mean())
            for l, (pos, neg) in (("extended (12v13)", (pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS,
                                                         pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS)),
                                  ("knowledge-denial (4v6)", (pcs.KNOWLEDGE_DENIAL_PERSONAS, pcs.KNOWLEDGE_CANDOUR_PERSONAS))):
                halves = []
                for parity in (0, 1):
                    keep = (bank_meta.rows["question_index"].to_numpy() % 2) == parity
                    sub = Bank(bank_meta.rows[keep].reset_index(drop=True), None, None, None, None, bank_meta.persona_of[keep])
                    halves.append(_axis(_persona_vectors(pooled[keep], sub), pos, neg))
                row[f"cross-question cos {l}"] = float(halves[0] @ halves[1])
        diag.append(row)
    diag = pd.DataFrame(diag)
    diag.round(4).to_csv(OUT / "diagnostics.csv", index=False)
    logger.info(f"diagnostics:\n{diag.round(3).to_string(index=False)}")
    _top_tokens(bank_meta)


def _top_tokens(bank: Bank, n: int = 10) -> None:
    """The learned attention's entropy per rollout and its most-attended tokens per persona."""
    weights = np.load(OUT / "weights_learned.npy").astype(np.float64)
    token_ids = np.load(TOKENS / "token_ids.npy")
    tokenizer = _tokenizer()
    plogp = np.where(weights > 0, weights * np.log(np.clip(weights, 1e-30, None)), 0.0)
    entropy = -np.bincount(bank.tok_roll, weights=plogp, minlength=bank.n_rollouts)
    t = np.bincount(bank.tok_roll, weights=bank.keep.astype(float), minlength=bank.n_rollouts)
    rows = []
    for i, persona in enumerate(PERSONAS):
        mask = bank.persona_of[bank.tok_roll] == i
        w, ids = weights[mask], token_ids[mask]
        totals: dict[int, float] = {}
        for tid, wt in zip(ids.tolist(), w.tolist()):
            totals[tid] = totals.get(tid, 0.0) + wt
        top = sorted(totals.items(), key=lambda kv: -kv[1])[:n]
        share = sum(v for _, v in top) / max(w.sum(), 1e-12)
        rows.append({"persona": persona, "top tokens": " | ".join(f"{tokenizer.decode([tid])!r} {v / w.sum():.3f}" for tid, v in top),
                     "top-10 share": share})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "top_tokens.csv", index=False)
    logger.info(f"attention entropy: median {np.median(entropy / np.log(np.maximum(t, 2))):.3f} of uniform; "
                f"top tokens per persona:\n{table.to_string(index=False)}")


def transfer() -> None:
    """`persona_ood.py run` and `zeroshot` on the learned-query bank (CSVs only)."""
    from deception_detection.scripts import persona_ood as P
    from deception_detection.scripts.layer_sweep import base_run, pooled_dir

    out = REPO_ROOT / "results" / f"persona_ood_att_L{LAYER}"
    P.run(pooled_dir=pooled_dir(LAYER), out_dir=out, layer=LAYER, vectors=BANKS / "learned")
    P.zeroshot(pooled_dir=pooled_dir(LAYER), out_dir=out, layer=LAYER, vectors=BANKS / "learned",
               base_run=base_run(LAYER), base_run_check=False)
    logger.info(f"transfer results in {out}")


if __name__ == "__main__":
    fire.Fire({"read": read, "basis": basis, "fit": fit, "extract": extract, "evaluate": evaluate,
               "transfer": transfer})
