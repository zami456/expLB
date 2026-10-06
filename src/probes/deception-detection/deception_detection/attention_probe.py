"""Attention probes (EleutherAI), ported for ragged answer-token data.

Reference: https://github.com/EleutherAI/attention-probes at commit 889b8dea, class
`AttentionProbe` (attention_probe/attention_probe.py) and `train_probe` (attention_probe/trainer.py);
blog post https://blog.eleuther.ai/attention-probes/. A verbatim copy is in
tests/attention_probe_reference.py, and tests/test_attention_probe.py checks this port against it.

Each head h scores every token with a learned query and a position bias, takes a softmax over the
example's tokens, and adds up a learned value of each token:

    k_{t,h} = q_h . x_t + position_t * w_h        a_{t,h} = softmax_t(k_{t,h})
    logit   = sum_h sum_t a_{t,h} (v_h . x_t + b_h)

`AttentionProbe` is the reference class unchanged. `train_probe` keeps the reference's LBFGS
closure (strong-Wolfe line search, `max_iter` 2000, BCE on logits, weight decay added to the loss
as wd/2 * |theta|^2 over every parameter). Deviations, all for this repo's data:

- Training is full batch. The reference draws a fresh random minibatch of `batch_size` examples in
  every LBFGS closure call when a training set is larger than that; its driver (train_mosaic.py)
  sidesteps this by cutting each training set to `batch_size` = 2048 examples. Here every example
  is kept and the closure always sees all of them.
- Seeding is explicit: `torch.manual_seed(seed)` before the probe is built. The reference builds it
  from the global generator.
- `predict` runs in float32. The reference's `evaluate_probe` runs under `torch.autocast`, which on
  CPU means bfloat16.
- `pool_tokens` reproduces the reference's `take_mean` and `last_only` baselines
  (`TrainingData.process`), which turn every example into one token.
"""

from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class AttentionProbe(nn.Module):
    """The reference module, unchanged: multiple heads, relative position bias, optional
    post-attention MLP, attention-weight dropout, and `attn_hook` for recording attention."""

    def __init__(self, d_in, n_heads, output_dim: int = 1, hidden_dim: int = 0, use_tanh: bool = False,
                 attn_dropout_p: float = 0.0, config: Any = None):
        super().__init__()
        self.q = nn.Linear(d_in, n_heads, bias=False)
        self.q.weight.data.zero_()
        self.v = nn.Linear(d_in, n_heads * (hidden_dim or output_dim))

        self.n_heads = n_heads
        self.output_dim = output_dim
        self.use_tanh = use_tanh
        self.attn_dropout_p = attn_dropout_p
        self.position_weight = nn.Parameter(torch.zeros((n_heads,), dtype=torch.float32))
        self.hidden_dim = hidden_dim
        if hidden_dim:
            self.o = nn.Linear(hidden_dim, output_dim)
        self.attn_hook = nn.Identity()
        self.config = config

    def forward(self, x, mask, position):
        k = self.q(x) - ((1 - mask.float()) * 1e9)[..., None] + position[..., None] * self.position_weight
        if self.training:
            k = torch.where(torch.rand_like(k) < self.attn_dropout_p, -1e9, k)
        if self.use_tanh:
            p = torch.tanh(k)
        else:
            p = torch.nn.functional.softmax(k, dim=-2)
        self.attn_hook(p)
        v = self.v(x).unflatten(-1, (self.n_heads, -1))
        o = (p[..., None] * v).sum((-2, -3))
        if self.hidden_dim:
            o = self.o(o.relu())
        return o


# --- data ----------------------------------------------------------------------------------------

def pad_answers(coords: np.ndarray, offsets: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ragged tokens [tokens, F] (example i = rows offsets[i]:offsets[i+1]) as padded tensors:
    x [n, L, F] float32, mask [n, L] bool, position [n, L] (index within the answer, 0 on padding)."""
    lengths = np.diff(offsets)
    n, length = len(lengths), int(lengths.max())
    x = torch.zeros((n, length, coords.shape[1]), dtype=torch.float32)
    mask = torch.zeros((n, length), dtype=torch.bool)
    example = np.repeat(np.arange(n), lengths)
    within = np.arange(len(coords)) - np.repeat(offsets[:-1], lengths)
    x[example, within] = torch.from_numpy(np.ascontiguousarray(coords, dtype=np.float32))
    mask[example, within] = True
    position = torch.arange(length, dtype=torch.int32)[None].expand(n, length) * mask
    return x, mask, position.int()


def pool_tokens(x: torch.Tensor, mask: torch.Tensor, position: torch.Tensor,
                mode: Literal["attention", "mean", "last"]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The reference's one-token baselines (`TrainingData.process`): `last` keeps only the last
    valid token, and `mean` and `last` then average the kept tokens into one."""
    if mode == "attention":
        return x, mask, position
    if mode == "last":
        mask = mask * (position == (position * mask).max(dim=-1, keepdim=True).values)
    total = torch.maximum(mask.sum(-1, keepdim=True)[..., None], torch.ones(1))
    x = (x * mask[..., None]).sum(-2, keepdim=True) / total
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, torch.ones_like(x[..., 0], dtype=torch.bool), torch.zeros_like(x[..., 0], dtype=torch.int32)


def _trim(x: torch.Tensor, mask: torch.Tensor, position: torch.Tensor):
    """Drop padding columns that no example uses (the reference's `TrainingData.trim`)."""
    last = int(mask.any(dim=0).nonzero()[-1]) + 1
    return x[:, :last], mask[:, :last], position[:, :last]


# --- training and scoring ------------------------------------------------------------------------

def train_probe(x: torch.Tensor, mask: torch.Tensor, position: torch.Tensor, y: np.ndarray | torch.Tensor,
                *, n_heads: int = 1, weight_decay: float = 0.0, seed: int = 0,
                max_iter: int = 2000) -> tuple[AttentionProbe, float]:
    """The reference `train_probe` with LBFGS, full batch. Returns the probe and its final
    penalised loss (the reference's returned loss also includes the penalty)."""
    x, mask, position = _trim(x, mask, position)
    torch.manual_seed(seed)
    probe = AttentionProbe(x.shape[-1], n_heads).to(torch.float32)
    optimizer = torch.optim.LBFGS(probe.parameters(), line_search_fn="strong_wolfe", max_iter=max_iter)
    target = torch.as_tensor(np.asarray(y), dtype=torch.float32)[..., None]
    loss = torch.tensor(torch.inf)

    def closure():
        nonlocal loss
        optimizer.zero_grad()
        out = probe(x, mask, position)
        loss = F.binary_cross_entropy_with_logits(out, target)
        for param in probe.parameters():
            loss = loss + param.pow(2).mul(weight_decay).div(2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(closure)
    return probe, float(loss.item())


@torch.no_grad()
def predict(probe: AttentionProbe, x: torch.Tensor, mask: torch.Tensor, position: torch.Tensor,
            batch_size: int = 1024, attention: bool = False):
    """Logits [n], in float32 and in eval mode. With `attention`, also each example's attention
    [n, L, heads] (zero on padding)."""
    probe.eval()
    logits, attns = [], []
    hook = probe.attn_hook.register_forward_hook(lambda _m, _i, out: attns.append(out)) if attention else None
    try:
        for a in range(0, len(x), batch_size):
            logits.append(probe(x[a:a + batch_size], mask[a:a + batch_size], position[a:a + batch_size])[:, 0])
    finally:
        if hook is not None:
            hook.remove()
    out = torch.cat(logits).numpy()
    return (out, torch.cat(attns).numpy()) if attention else out
