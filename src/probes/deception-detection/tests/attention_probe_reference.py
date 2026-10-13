# ruff: noqa
# fmt: off
"""The reference attention probe, verbatim, for tests/test_attention_probe.py to compare against.

Copied from https://github.com/EleutherAI/attention-probes at commit
889b8dea53444c960cbfedb337eac429cbabfb48: all of attention_probe/attention_probe.py, then from
attention_probe/trainer.py `AttentionProbeTrainConfig`, `TrainingData`,
`attention_probe_from_config`, `train_probe` and `evaluate_probe`. Only the imports differ: the
package-relative imports are dropped (both files now share this module), `LinearClassifier` is not
imported (only `train_probe_repeatedly` uses it, and it is not copied), and `simple_parsing`'s
`Serializable` is replaced by an empty base class because simple_parsing is not installed, and
jaxtyping's `Array` (absent from the installed version) by `np.ndarray`. Lint and
formatting are off so the copy stays verbatim.

The repository has no license file at this commit (its README says it is based on
https://github.com/shan23chen/MOSAIC, "unlicensed"). Keep this file out of any public commit
unless that is resolved.
"""

import numpy as np
from typing import Optional
from collections import OrderedDict
from torch.nn import functional as F
from jaxtyping import Float, Int, Bool
Array = np.ndarray  # stand-in: the installed jaxtyping has no `Array`
from tqdm.auto import trange
from sklearn.model_selection import StratifiedKFold

from dataclasses import dataclass, replace
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings


class Serializable:
    """Stand-in for simple_parsing.Serializable, which is not installed."""


# --- attention_probe/attention_probe.py, verbatim ---------------------------------------------

from torch import nn
import torch
from typing import Any

class AttentionProbe(nn.Module):
    """
    Torch module for attention probes.
    Supports:
    * multiple heads
    * relative position bias
    * post-attention MLP
    * attention weight dropout
    * attention weight recording via PyTorch forward hooks
    """
    
    def __init__(self, d_in, n_heads, output_dim: int = 1, hidden_dim: int = 0, use_tanh: bool = False, attn_dropout_p: float = 0.0, config: Any = None):
        """
        Args:
            d_in (int): input dimensionality
            n_heads (int): number of attention heads
            output_dim (int): output dimension (default: 1).
            Returns logits, needs to be passed through an activation function.
            hidden_dim (int): hidden dimension for post-attention MLP (default: 0, no MLP)
            use_tanh (bool): use tanh activation for attention weights (default: False)
            attn_dropout_p (float): dropout probability for attention weights (default: 0.0)
            config (Any): additional configuration parameters to store in the model
        """
        super().__init__()
        # projection from inputs to attention logits
        self.q = nn.Linear(d_in, n_heads, bias=False)
        self.q.weight.data.zero_()
        # projection to per-head output logits (or pre-MLP intermediate states)
        self.v = nn.Linear(d_in, n_heads * (hidden_dim or output_dim))

        self.n_heads = n_heads
        self.output_dim = output_dim
        self.use_tanh = use_tanh
        self.attn_dropout_p = attn_dropout_p
        # alibi-like relative (to the beginning/end of the sequence) position bias
        self.position_weight = nn.Parameter(torch.zeros((n_heads,), dtype=torch.float32))
        # MLP after the attention
        self.hidden_dim = hidden_dim
        if hidden_dim:
            self.o = nn.Linear(hidden_dim, output_dim)
        # hookpoint to record attention probabilities. use register_forward_hook to record
        self.attn_hook = nn.Identity()
        
        self.config = config

    def forward(self, x, mask, position):
        # x: (batch_size, seq_len, d_in)
        # mask: (batch_size, seq_len)
        # position: (batch_size, seq_len)
        
        # k: (batch_size, seq_len, n_heads)
        # elements that are masked are set to -infinity
        # position is added to the key weighted by the per-head position_weight
        k = self.q(x) - ((1 - mask.float()) * 1e9)[..., None] + position[..., None] * self.position_weight
        if self.training:
            # apply dropout to the keys
            k = torch.where(torch.rand_like(k) < self.attn_dropout_p, -1e9, k)
        # p: (batch_size, seq_len, n_heads)
        # probability of each element after softmax, with masked elements set to 0
        # dim=-2 is the sequence length dimension
        if self.use_tanh:
            p = torch.tanh(k)
        else:
            p = torch.nn.functional.softmax(k, dim=-2)
        # record attention probabilities if necessary
        self.attn_hook(p)
        # v: (batch_size, seq_len, n_heads, output_dim)
        v = self.v(x).unflatten(-1, (self.n_heads, -1))
        # o: (batch_size, output_dim)
        # weight v by the attention probabilities and sum over the sequence length and head dimensions
        o = (p[..., None] * v).sum((-2, -3))
        # if we have an MLP after the attention, apply it
        if self.hidden_dim:
            o = self.o(o.relu())
        return o


# --- attention_probe/trainer.py, lines 24-348 and 405-451, verbatim ----------------------------

@dataclass
class AttentionProbeTrainConfig(Serializable):
    """
    Config for training an attention probe.
    """
    learning_rate: float = 1e-4
    """Learning rate for Adam."""
    train_lbfgs: bool = True
    """Use LBFGS instead of Adam."""
    early_stop_iterations: int = 100
    """Number of iterations without improvement before stopping."""
    
    weight_decay: float = 0.0
    """Weight decay coefficient."""
    train_iterations: int = 2000
    """Number of training iterations."""
    n_heads: int = 1
    """Number of attention heads."""
    attn_dropout_p: float = 0.0
    """Dropout probability for attention probes."""
    use_linear_classifier: bool = False
    """Use a linear classifier instead of an attention probe.
    Trained with cross-validation for tuning weight decay. Uses LBFGS.
    Only use with one-token datasets."""
    
    last_only: bool = False
    """Transforms the data such that only the last token is used.
    Any attention probe becomes a one-token linear classifier."""
    take_mean: bool = False
    """Transforms the data such that the mean of the hidden states across sequence length is used.
    Any attention probe becomes a one-token linear classifier."""
    absmax_pool: bool = False
    """Transforms the data such that the absolute maximum of the hidden states across sequence length is used.
    Any attention probe becomes a one-token linear classifier."""
    finetune_attn: bool = False
    """If we are training a mean probe or a last token probe, take a trained probe
    and finetune it into an attention probe."""
    ensemble_mean: bool = False
    """If we are training an attention probe, train a mean probe head independently from the attention probe."""
    
    hidden_dim: int = 0
    """Post-probe MLP hidden dimension."""
    use_tanh: bool = False
    """Use tanh instead of softmax for attention."""
    batch_size: int = 2048
    """Batch size for training."""
    always_move_to_device: bool = True
    """Even if the data doesn't fit into the GPU, try to move it there."""
    device: str = "cuda"
    """Device to use for training."""
    seed: int = 5
    """Seed for initializing the model."""
    
    retrain_threshold: float = 0.5
    """Loss level above which we change the seeed and retrain the probe."""
    retrain_n: int = 5
    """Number of times we are allowed to retrain the probe if the loss is above the threshold."""
    
    @property
    def one_token(self) -> bool:
        return self.last_only or self.take_mean or self.absmax_pool
    
    def not_one_token(self) -> "AttentionProbeTrainConfig":
        return replace(self, last_only=False, take_mean=False, absmax_pool=False)

@dataclass
class TrainingData:
    """
    Helper for storing all data relevant to training an attention probe.
    """
    
    x: Float[Array, "batch_size seq_len hidden_dim"]
    """Hidden states."""
    mask: Bool[Array, "batch_size seq_len"]
    """Mask of valid positions."""
    position: Int[Array, "batch_size seq_len"]
    """Position of each token."""
    y: Int[Array, "batch_size"]
    """Labels."""
    input_ids: Int[Array, "batch_size seq_len"] | None = None
    """Original input IDs."""
    n_classes: int = 2
    """Number of classes."""
    class_mapping: dict[int, str] | None = None
    """Mapping from class indices to class names."""
    y_addition: Optional[Float[Array, "batch_size n_classes"]] = None
    """Additional labels to add to the probe output."""
    
    @property
    def multi_class(self) -> bool:
        """Whether the dataset is a multiclass dataset."""
        return self.n_classes > 2
    
    def text_label(self, y: int) -> str:
        """Get the text label for a class index."""
        if self.class_mapping is None:
            return str(y)
        return self.class_mapping[y]
    
    @property
    def device(self) -> torch.device:
        """Device of the data."""
        return self.x.device
    
    def __post_init__(self):
        if self.input_ids is not None and self.input_ids.ndim == 1:
            self.input_ids = self.input_ids[None].repeat(len(self.x), 1)
        if self.position is not None and self.position.ndim == 1:
            self.position = self.position[None].repeat(len(self.x), 1)
        

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.x)
    
    def numel(self) -> int:
        """Number of valid positions in the dataset."""
        return int(self.mask.sum())
    
    def numel_base(self) -> int:
        """Number of tokens in the dataset."""
        return self.input_ids.numel() if self.input_ids is not None else 0
    
    def trim(self) -> "TrainingData":
        """Trim the dataset to the last valid position."""
        last_position = self.mask.any(dim=0).nonzero().tolist()[-1][-1] + 1
        return replace(
            self,
            x=self.x[:, :last_position],
            mask=self.mask[:, :last_position],
            position=self.position[:, :last_position],
            y=self.y[:, :last_position] if self.y.ndim > 1 else self.y,
            input_ids=self.input_ids[:, :last_position] if self.input_ids is not None else None,
        )
    @torch.no_grad()
    def reindex(self, indices: Int[Array, "batch_size"]) -> "TrainingData":
        """Reindex the dataset according to an array of indices corresponding to original examples."""
        assert indices.ndim == 1
        assert int(max(indices)) < len(self)
        if self.is_tensor and isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices)
        if self.is_tensor:
            indices = indices.to(self.device)
        return replace(
            self,
            x=self.x[indices],
            mask=self.mask[indices],
            position=self.position[indices],
            y=self.y[indices],
            y_addition=self.y_addition[indices] if self.y_addition is not None else None,
            input_ids=self.input_ids[indices] if self.input_ids is not None else None,
        )
    
    @torch.no_grad()
    def split(self, n_folds: int, seed: int) -> list[tuple["TrainingData", "TrainingData"]]:
        """Split the dataset into training and testing sets."""
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kfold.split(self.y, self.y))
        for train_indices, test_indices in splits:
            yield self.reindex(train_indices), self.reindex(test_indices)
    
    @torch.no_grad()
    def join(self, other: "TrainingData") -> "TrainingData":
        """Join two datasets, concatenating all fields."""
        return replace(
            self,
            x=torch.cat([self.x, other.x], dim=0),
            mask=torch.cat([self.mask, other.mask], dim=0),
            position=torch.cat([self.position, other.position], dim=0),
            y=torch.cat([self.y, other.y], dim=0),
            y_addition=torch.cat([self.y_addition, other.y_addition], dim=0) if self.y_addition is not None else None,
            input_ids=torch.cat([self.input_ids, other.input_ids], dim=0) if self.input_ids is not None else None,
        )
    
    @torch.no_grad()
    def to_tensor(self, device: torch.device = None) -> "TrainingData":
        """Place the dataset on a device. If the dataset is in numpy format, convert it to a tensor."""
        if self.is_tensor:
            return replace(
                self,
                x=self.x.to(device),
                mask=self.mask.to(device),
                position=self.position.to(device),
                y=self.y.to(device),
                y_addition=self.y_addition.to(device) if self.y_addition is not None else None,
                input_ids=self.input_ids.to(device) if self.input_ids is not None else None,
            )
        return replace(
            self,
            x=torch.tensor(self.x, dtype=torch.float32, device=device),
            mask=torch.tensor(self.mask, dtype=torch.bool, device=device),
            position=torch.tensor(self.position, dtype=torch.int32, device=device),
            y=torch.tensor(self.y, dtype=torch.int32, device=device),
            y_addition=torch.tensor(self.y_addition, dtype=torch.float32, device=device) if self.y_addition is not None else None,
            input_ids=torch.tensor(self.input_ids, dtype=torch.int32, device=device) if self.input_ids is not None else None,
        )
    
    @property
    def is_tensor(self) -> bool:
        """Whether the dataset is a tensor."""
        return isinstance(self.x, torch.Tensor)
    
    @torch.no_grad()
    def process(self, config: AttentionProbeTrainConfig) -> "TrainingData":
        """
        Process the training data according to the configuration.
        If mean pooling or last token selection is enabled, these transformations are applied irreversibly.
        Will convert the dataset to a tensor if it is not already.
        """

        if not self.is_tensor:
            self = self.to_tensor(device="cpu")
        
        if config.last_only:
            self.mask = self.mask * (self.position == (self.position * self.mask).max(dim=-1, keepdim=True).values)
        if config.last_only or config.take_mean:
            mask_sum = self.mask.sum(-1, keepdim=True)[..., None]
            mask_sum = torch.maximum(mask_sum, torch.ones_like(mask_sum))
            self.x = (self.x * self.mask[..., None]).sum(-2, keepdim=True) / mask_sum
            self.mask = torch.ones_like(self.x[..., 0])
        if config.absmax_pool:
            self.x = self.x.abs().max(dim=-1, keepdim=True).values
            self.mask = torch.ones_like(self.x[..., 0])
        self.x = torch.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
        return self


def attention_probe_from_config(config: AttentionProbeTrainConfig, train_split: TrainingData) -> AttentionProbe:
    """
    Create an attention probe from a configuration and training data.
    """

    if config.one_token:
        config = replace(config, attn_dropout_p=0.0)
    probe = AttentionProbe(
        train_split.x.shape[-1],
        config.n_heads,
        hidden_dim=config.hidden_dim,
        output_dim=1 if not train_split.multi_class else train_split.n_classes,
        use_tanh=config.use_tanh,
        attn_dropout_p=config.attn_dropout_p,
        config=config,
    )
    if config.last_only:
        probe.position_weight.data[:] = 1.0
    return probe


def train_probe(train_split: TrainingData, config: AttentionProbeTrainConfig, device: torch.device | None = None, start_from: AttentionProbe | None = None) -> tuple[AttentionProbe, float]:
    """
    Train an attention probe.
    
    Args:
        train_split: The training data.
        config: The configuration for the probe.
        device: The device to use for training.
        start_from: A pre-trained probe to start from. If None, a new probe is trained from scratch.

    Returns:
        The trained probe and the loss.
    """

    train_split = train_split.trim()
    probe = attention_probe_from_config(config, train_split)
    if start_from is not None:
        probe.load_state_dict(start_from.state_dict())
        torch.nn.init.normal_(probe.q.weight.data, mean=0.0, std=0.01)
        torch.nn.init.normal_(probe.position_weight.data, mean=0.0, std=0.01)
    probe = probe.to(device, torch.float32)
    if config.train_lbfgs:
        optimizer = torch.optim.LBFGS(probe.parameters(), line_search_fn="strong_wolfe", max_iter=config.train_iterations)
    else:
        optimizer = torch.optim.AdamW(probe.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if train_split.numel_base() < 100_000:
        train_tensor = train_split.to_tensor(device=device)
    else:
        train_tensor = train_split.to_tensor(device="cpu")
    if config.always_move_to_device:
        train_tensor = train_tensor.trim().to_tensor(device=device)
    
    loss = torch.inf
    bar = trange(config.train_iterations, desc=f"Training")
    loss_history = []
    stopped = False
    def closure():
        nonlocal loss, stopped
        optimizer.zero_grad()
        if len(train_split.y) <= config.batch_size:
            train_batch = train_tensor
        else:
            indices = torch.randint(0, len(train_split.y), (config.batch_size,), device=train_tensor.device)
            train_batch = train_tensor.reindex(indices)
        train_batch = train_batch.trim().to_tensor(device=device)
        out = probe(train_batch.x, train_batch.mask, train_batch.position)
        if train_split.y_addition is not None:
            out = out + train_split.y_addition
        if not train_split.multi_class:
            loss = F.binary_cross_entropy_with_logits(out, train_batch.y.float()[..., None])
        else:
            loss = F.cross_entropy(out, train_batch.y.long())
        train_loss = loss
        if config.train_lbfgs:
            for param in probe.parameters():
                train_loss += param.pow(2).mul(config.weight_decay).div(2).sum()
        train_loss.backward()
        bar.set_postfix(loss=loss.item())
        loss_history.append(loss.item())
        if not config.train_lbfgs and len(loss_history) > config.early_stop_iterations * 2:
            before, after = loss_history[:-config.early_stop_iterations], loss_history[-config.early_stop_iterations:]
            if min(before) < min(after):
                stopped = True
                return
        if config.train_lbfgs:
            bar.update(1)
        return loss
    
    with torch.set_grad_enabled(True):
        if config.train_lbfgs:
            optimizer.step(closure)
        else:
            for _ in bar:
                optimizer.step(closure)
                if stopped:
                    break
    return probe, loss.item()

def evaluate_probe(probe: AttentionProbe, test_data: TrainingData, config: AttentionProbeTrainConfig, compute_attn: bool = False, use_activation: bool = False) -> \
    tuple[Float[Array, "batch_size n_heads seq_len"], Float[Array, "batch_size n_classes"]] | Float[Array, "batch_size n_classes"]:
    """
    Evaluate an attention probe on a dataset. Optionally, collect attention weights.
    
    Args:
        probe: The attention probe to evaluate.
        test_data: The dataset to evaluate the probe on.
        config: The configuration for the probe.
        compute_attn: Whether to collect attention weights.
        use_activation: Whether to use an activation function on the probe output.

    Returns:
        If `compute_attn` is False, returns the probe output.
        If `compute_attn` is True, returns a tuple of the probe output and the attention weights.
    """

    probe.eval()
    probe.requires_grad_(False)
    
    device = next(probe.parameters()).device
    
    with torch.inference_mode(), torch.autocast(device_type=device.type):
        attns = []
        all_probs = []
        if compute_attn:
            probe.attn_hook.register_forward_hook(lambda _, __, output: attns.append(output.detach().cpu().numpy()))
        for batch_start in trange(0, len(test_data), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(test_data))
            batch = test_data.reindex(torch.arange(batch_start, batch_end, device=device))
            batch = batch.to_tensor(device=device)
            out = probe(batch.x, batch.mask, batch.position)
            if use_activation:
                if not test_data.multi_class:
                    out = out.sigmoid()[..., 0]
                else:
                    out = out.softmax(dim=-1)
            probs = out.detach().cpu().numpy()
            all_probs.append(probs)
        if hasattr(probe, 'attn_hook'):
            probe.attn_hook._foward_hooks = OrderedDict()
        if compute_attn:
            attns = np.concatenate(attns)
        probs = np.concatenate(all_probs)
    if compute_attn:
        return probs, attns
    return probs
