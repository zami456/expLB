"""On-disk cache of remotely-extracted activations, keyed per example.

Activations are downloaded from NDIF one batch at a time and are by far the most expensive
part of a run: a full llama sweep over the liars-bench subsets is tens of GB and hours of
transfer. Without a cache, any interruption throws all of it away.

The key is per **example**, not per batch. That is the point of the design: batch
composition is chosen by a token budget that can change between runs, and shrinks on its
own whenever a remote call OOMs (see :mod:`deception_detection.batching`). Keying on the
batch would tie every cached file to one particular grouping and lose the cache the moment
the budget moved.

Caching per example is sound because an example's activations do not depend on what it was
batched with. Padding is left-side (``padding_side="left"`` in models.py) and excluded by
the attention mask, and RoPE is relative, so shifting a row's real tokens uniformly leaves
attention among them unchanged. ``tests/test_acts_cache.py`` pins this down offline and the
live smoke test measures the residual floating-point difference.

Files hold *unpadded* activations, ``[true_len, n_layers, emb]``, so the cache also costs
less than the padded batches it replaces. A run extracting only the suffix that holds an
example's detected tokens (``positions="tail"``) stores ``[tail_len, n_layers, emb]`` under a
key that records that length.
"""

import hashlib
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from deception_detection.log import logger
from deception_detection.utils import REPO_ROOT

# Bumped when the on-disk layout changes in a way older files cannot satisfy.
CACHE_FORMAT_VERSION = 1


def cache_root() -> Path:
    """Directory holding the cache. ``activations/`` is already gitignored."""
    override = os.environ.get("ACTS_CACHE_DIR")
    return Path(override) if override else REPO_ROOT / "activations" / "cache"


def example_key(
    token_ids: Tensor,
    model_name: str,
    layers: list[int] | None,
    positions: str = "all",
) -> str:
    """Content hash of one example's unpadded tokens, plus what was extracted from them.

    The model and layer list belong in the key so that switching checkpoints or detect
    layers cannot silently return another run's activations. ``positions`` does too, for
    the same reason: a tail-only entry must never be served to a run that wants the whole
    sequence, or a longer tail. It is hashed only when it is not ``"all"``, so every key written before it
    existed is unchanged and the caches already on disk keep hitting.
    """
    hasher = hashlib.sha256()
    hasher.update(f"v{CACHE_FORMAT_VERSION}|{model_name}|".encode())
    hasher.update(("all" if layers is None else ",".join(map(str, layers))).encode())
    hasher.update(b"|")
    if positions != "all":
        hasher.update(f"positions={positions}|".encode())
    hasher.update(token_ids.to(torch.int64).cpu().numpy().tobytes())
    return hasher.hexdigest()


DRIVE_WAIT_S = 60.0


def until_drive(fn, what: object, wait_s: float = DRIVE_WAIT_S):
    """`fn()`, retried for up to `wait_s` seconds while it raises OSError.

    The cache can live on an external drive that drops out for a few seconds at a time
    ("WinError 433: A device which does not exist was specified"). A long run should ride that
    out rather than die on the first file it touches during the gap. Errors that are not OSError
    -- a corrupt file, say -- are not retried.
    """
    deadline = time.monotonic() + wait_s
    while True:
        try:
            return fn()
        except OSError as err:
            if time.monotonic() >= deadline:
                raise
            logger.warning(f"{what}: {err}; waiting for the drive")
            time.sleep(2.0)


def path_for(key: str) -> Path:
    """Two levels of hash-prefix subdirectory keep any one directory small."""
    return cache_root() / key[:2] / key[2:4] / f"{key}.pt"


def load(key: str, expected_len: int) -> Tensor | None:
    """Return cached activations for ``key``, or None on any miss or mismatch.

    A corrupt or stale entry is treated as a miss and re-downloaded rather than raised on:
    the cache is an optimisation, and refusing to run because of a bad file would be worse
    than paying for the download again.
    """
    path = path_for(key)
    if not until_drive(path.exists, path):
        return None
    try:
        payload = until_drive(lambda: torch.load(path, map_location="cpu", weights_only=True), path)
        acts = payload["acts"]
        if payload["version"] != CACHE_FORMAT_VERSION:
            return None
        if acts.shape[0] != expected_len:
            logger.warning(
                f"Cached activations at {path} have length {acts.shape[0]}, expected "
                f"{expected_len}; ignoring the entry"
            )
            return None
        return acts
    except Exception as err:
        logger.warning(f"Could not read cached activations at {path} ({err}); re-downloading")
        return None


def store(key: str, acts: Tensor) -> None:
    """Write activations for one example, atomically.

    The temp-then-rename matters: a run killed mid-write must not leave a truncated file
    that a resumed run would load as if it were complete.
    """
    path = path_for(key)
    # clone() is load-bearing, not defensive tidying. These tensors are slices of the
    # batch they arrived in, and torch.save serialises a view's entire underlying storage,
    # so saving the view directly writes the whole batch into every example's file --
    # measured at 8x here, and worse at larger batch sizes. Slicing dim 0 keeps the result
    # contiguous, so .contiguous() is a no-op and cannot fix it; only a copy can.
    payload = {"acts": acts.detach().cpu().clone(), "version": CACHE_FORMAT_VERSION}
    def write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    try:
        until_drive(write, path)
    except OSError as err:
        logger.warning(f"Could not cache activations to {path} ({err}); continuing uncached")


@dataclass
class CacheStats:
    """Hit/miss tallies for one call, so a run can report what it actually downloaded."""

    hits: int = 0
    misses: int = 0
    stored_bytes: int = 0
    dtypes: set[str] = field(default_factory=set)

    @property
    def total(self) -> int:
        return self.hits + self.misses

    def summary(self) -> str:
        if self.total == 0:
            return "no examples"
        pct = 100.0 * self.hits / self.total
        return (
            f"{self.hits}/{self.total} cached ({pct:.0f}%), {self.misses} downloaded"
            + (f", dtype={'/'.join(sorted(self.dtypes))}" if self.dtypes else "")
        )


def free_bytes() -> int:
    """Free space on the filesystem holding the cache, or -1 if it cannot be determined."""
    root = cache_root()
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return -1


_projection_reported = False


def report_projection(sample: Tensor, examples_remaining: int, mean_len: float) -> None:
    """Log the projected cache size once, as soon as a real activation tensor exists.

    The size estimate that justified caching full sequences assumes the backend returns
    bfloat16. If it returns float32 the cache is twice the projection and may not fit, so
    this surfaces the true figure in the first minute of a run rather than at 90% disk.
    """
    global _projection_reported
    if _projection_reported or sample.numel() == 0:
        return
    _projection_reported = True

    bytes_per_token = sample.element_size() * sample.shape[-1] * sample.shape[-2]
    projected = int(examples_remaining * mean_len * bytes_per_token)
    free = free_bytes()

    msg = (
        f"Activation cache: dtype={sample.dtype}, {bytes_per_token / 1024:.1f} KiB/token, "
        f"~{projected / 1e9:.1f} GB projected for {examples_remaining} remaining examples"
    )
    if free >= 0:
        msg += f"; {free / 1e9:.1f} GB free at {cache_root()}"
    logger.info(msg)

    if free >= 0 and projected > free:
        logger.warning(
            f"Projected cache ({projected / 1e9:.1f} GB) exceeds free space "
            f"({free / 1e9:.1f} GB). Free space, point ACTS_CACHE_DIR at a larger drive, "
            f"or set acts_cache_mode: off."
        )


def unpad_left(padded: Tensor, true_len: int) -> Tensor:
    """Take one row's real tokens out of a left-padded batch row."""
    return padded[padded.shape[0] - true_len :]


def reassemble(
    per_example: list[Tensor],
    true_lens: list[int],
    padded_len: int,
) -> Tensor:
    """Rebuild the padded ``[batch, seqpos, layer, emb]`` tensor the pipeline expects.

    Pad positions are left at zero. Nothing reads them: scoring goes through
    ``get_all_activations`` (masked by the attention mask) or ``get_masked_activations``
    (masked by the detection mask), and neither includes a pad position.
    """
    if not per_example:
        raise ValueError("cannot reassemble an empty list of activations")
    if len(per_example) != len(true_lens):
        raise ValueError(f"got {len(per_example)} activations for {len(true_lens)} lengths")

    template = per_example[0]
    n_layers, emb = template.shape[-2], template.shape[-1]
    out = torch.zeros((len(per_example), padded_len, n_layers, emb), dtype=template.dtype)
    for i, (acts, true_len) in enumerate(zip(per_example, true_lens, strict=True)):
        if acts.shape[0] != true_len:
            raise ValueError(f"example {i}: {acts.shape[0]} activations for {true_len} tokens")
        out[i, padded_len - true_len :] = acts
    return out
