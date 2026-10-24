"""The attention probe of EleutherAI (https://blog.eleuther.ai/attention-probes/), as their code builds and trains it
(github.com/EleutherAI/attention-probes: `attention_probe/attention_probe.py`, `trainer.py`, `train_mosaic.py`).

Per head h, a query vector q_h (zero-initialised, no bias) gives every token a logit x_t . q_h + beta_h * position_t
(position = 0, 1, ... from the span's first token; under softmax the same as counting from its last); masked tokens
get -1e9; softmax over the span; the output is sum_h sum_t a_th (x_t . v_h + c_h), the heads summed. Trained as
their code trains it: binary cross-entropy plus (weight_decay / 2) ||theta||^2, full batch, LBFGS with a strong-Wolfe
line search, up to 2000 iterations; weight decay chosen from {0, 1e-3, 1e-2, 1e-1} by 5-fold cross-validation
(highest mean accuracy, ties to the smaller value; their code thresholds raw logits at 0.5, here at 0); retrained
with a new initialisation while the training loss stays above 0.5, up to 5 attempts, the lowest-loss one kept.
Here the folds are grouped by run (rows of one run share their context) and the inputs are standardised.

Because sum_t a_th = 1, the output is linear in the per-head pooled inputs z_h = sum_t a_th x_t:
score = sum_h z_h . v_h + sum_h c_h -- `pooled` gives z, which is how the probe enters their pipeline.
"""

from dataclasses import dataclass

import numpy as np
import torch

WD_GRID = (0.0, 1e-3, 1e-2, 1e-1)
MAX_ITER = 2000
RETRAIN_LOSS, RETRAIN_N = 0.5, 5
N_FOLDS = 5


class AttentionProbe(torch.nn.Module):
    def __init__(self, d_in: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.q = torch.nn.Linear(d_in, n_heads, bias=False)
        self.q.weight.data.zero_()
        self.v = torch.nn.Linear(d_in, n_heads)
        self.position_weight = torch.nn.Parameter(torch.zeros(n_heads))

    def attention(self, x, mask, position):
        k = self.q(x) - ((1 - mask.float()) * 1e9)[..., None] + position[..., None] * self.position_weight
        return torch.softmax(k, dim=-2)  # (B, T, H)

    def forward(self, x, mask, position):
        return (self.attention(x, mask, position) * self.v(x)).sum((-2, -1))

    def pooled(self, x, mask, position):
        """z: (B, H, d), the per-head attention-weighted inputs."""
        return torch.einsum("bth,btd->bhd", self.attention(x, mask, position), x)


@dataclass
class Batch:
    """Padded spans: x (B, T, d) standardised, mask and position (B, T)."""

    x: torch.Tensor
    mask: torch.Tensor
    position: torch.Tensor

    def take(self, idx) -> "Batch":
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        return Batch(self.x[idx], self.mask[idx], self.position[idx])


def batch(spans: list[np.ndarray], mean: np.ndarray, sd: np.ndarray) -> Batch:
    """Spans of shape (T_i, d), every T_i >= 1, standardised with (mean, sd) and right-padded."""
    T = max(len(s) for s in spans)
    x = np.zeros((len(spans), T, len(mean)), np.float32)
    mask = np.zeros((len(spans), T), np.float32)
    for i, s in enumerate(spans):
        x[i, :len(s)] = (s - mean) / sd
        mask[i, :len(s)] = 1.0
    position = np.broadcast_to(np.arange(T, dtype=np.float32), (len(spans), T)) * mask
    return Batch(torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(np.ascontiguousarray(position)))


def standardiser(spans: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.concatenate(spans).astype(np.float64)
    return tokens.mean(0), tokens.std(0) + 1e-8


def _fit_once(b: Batch, y: torch.Tensor, n_heads: int, weight_decay: float, init_seed: int):
    torch.manual_seed(init_seed)
    probe = AttentionProbe(b.x.shape[-1], n_heads)
    opt = torch.optim.LBFGS(probe.parameters(), line_search_fn="strong_wolfe", max_iter=MAX_ITER)

    def objective():
        loss = torch.nn.functional.binary_cross_entropy_with_logits(probe(b.x, b.mask, b.position), y)
        return loss + sum(p.pow(2).sum() for p in probe.parameters()) * weight_decay / 2

    def closure():
        opt.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        return probe, float(objective())


def fit(b: Batch, y: np.ndarray, n_heads: int, weight_decay: float, seed: int) -> AttentionProbe:
    """Their `train_probe_repeatedly`: up to RETRAIN_N initialisations while the loss is above RETRAIN_LOSS."""
    yt = torch.as_tensor(y, dtype=torch.float32)
    best = None
    for attempt in range(RETRAIN_N):
        probe, loss = _fit_once(b, yt, n_heads, weight_decay, 1000 * seed + attempt)
        if best is None or loss < best[1]:
            best = (probe, loss)
        if loss <= RETRAIN_LOSS:
            break
    return best[0].eval()


def scores(probe: AttentionProbe, b: Batch) -> np.ndarray:
    with torch.no_grad():
        return probe(b.x, b.mask, b.position).numpy()


def choose_weight_decay(b: Batch, y: np.ndarray, groups: list[str], n_heads: int, seed: int) -> tuple[float, dict]:
    """5-fold cross-validation, folds grouped by run and stratified, highest mean accuracy (ties: smaller)."""
    from sklearn.model_selection import StratifiedGroupKFold

    folds = list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, groups))
    acc = {}
    for wd in WD_GRID:
        acc[wd] = float(np.mean([np.mean((scores(fit(b.take(tr), y[tr], n_heads, wd, seed), b.take(te)) > 0) == y[te])
                                 for tr, te in folds]))
    best = max(WD_GRID, key=lambda w: (acc[w], -w))
    return best, acc


def flat_direction(probe: AttentionProbe) -> np.ndarray:
    """v as one vector over (head, dim), matching `pooled(...).reshape(B, -1)`."""
    return probe.v.weight.detach().numpy().reshape(-1).astype(np.float64)
