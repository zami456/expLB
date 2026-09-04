"""Token-budget batching for remote activation extraction.

The remote backend (NDIF) fails with an out-of-memory error when a single traced batch is
too large. Sizing batches by row count cannot prevent that: the tokenizer left-pads every
row to the longest one in the batch, so the memory a batch actually needs is

    len(batch) * max(length of any example in the batch)

and a fixed row count leaves the second factor free. Datasets here vary a lot internally --
harm_refusal_choice averages 174 tokens but reaches 1144 -- so a fixed row count is safe
for most batches and blows up on whichever one happens to collect the long examples.

This module bounds that product directly, and shrinks the budget in response to an OOM
rather than requiring it to be guessed correctly up front.
"""

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from deception_detection.log import logger

# Padded tokens per remote call. Conservative default: the hand-tuned shard sizes this
# replaces ranged from ~476 to ~25,500 padded tokens per call depending on the dataset.
DEFAULT_NDIF_TOKEN_BUDGET = 8000

# Padded tokens per shard, bounding the activation tensor a shard assembles in host RAM.
# At 8192 dims and 2 bytes per element, 32k tokens is ~525 MB.
DEFAULT_HOST_TOKEN_BUDGET = 32_000

FailureKind = Literal["oom", "transient", "unknown"]

# Matched against the exception text. The remote backend's exact exception types are not
# part of its public contract, so classification is best-effort and defaults to shrinking.
_OOM_MARKERS = (
    "out of memory",
    "outofmemory",
    "cuda oom",
    "insufficient memory",
    "resource exhausted",
    "not enough memory",
)
_TRANSIENT_MARKERS = (
    "evicted",
    "deployment",
    "temporarily unavailable",
    "connection",
    "timed out",
    "timeout",
    "502",
    "503",
    "504",
)


def plan_batches(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Group indices into batches whose padded cost stays within ``budget``.

    Greedy in-order fill: extend the current batch while ``len(batch) * max_len`` stays
    within budget, otherwise start a new one. Because the cost is recomputed against the
    running maximum, one long example shrinks the batch it lands in instead of silently
    multiplying the footprint of everything batched with it.

    An example longer than the entire budget gets a batch to itself rather than being
    dropped: there is nothing smaller the caller could do with it.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")

    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0

    for i, length in enumerate(lengths):
        new_max = max(current_max, length)
        if current and (len(current) + 1) * new_max > budget:
            batches.append(current)
            current, current_max = [i], length
        else:
            current.append(i)
            current_max = new_max

    if current:
        batches.append(current)
    return batches


def classify_failure(err: BaseException) -> FailureKind:
    """Decide whether a failed remote call should shrink the batch or just be retried.

    An eviction or a dropped connection says nothing about batch size, so retrying at the
    same size is right. An OOM means the batch was too big. Anything unrecognised is
    treated as unknown and shrunk, since shrinking is the safe direction to guess in.
    """
    text = f"{type(err).__name__}: {err}".lower()
    if any(marker in text for marker in _OOM_MARKERS):
        return "oom"
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return "transient"
    return "unknown"


class BudgetMemory:
    """Remembers which token budgets a model has tolerated, across runs.

    Stores the largest budget seen to succeed and the smallest seen to OOM, so a later run
    starts near the real ceiling instead of rediscovering it from the default every time.
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, dict[str, int]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as err:
                logger.warning(f"Could not read budget memory at {path} ({err}); starting empty")

    @staticmethod
    def key(model_name: str, n_layers: int) -> str:
        return f"{model_name}|layers={n_layers}"

    def suggest(self, key: str, configured: int) -> int:
        """Budget to start from: the best known-good, never at or above a known OOM."""
        entry = self._data.get(key, {})
        budget = max(configured, entry.get("ok", 0))
        oom = entry.get("oom")
        if oom is not None and budget >= oom:
            budget = max(1, oom // 2)
            logger.info(f"Budget {configured} >= previously OOMing {oom}; starting at {budget}")
        return budget

    def record_success(self, key: str, budget: int) -> None:
        entry = self._data.setdefault(key, {})
        if budget > entry.get("ok", 0):
            entry["ok"] = budget
            self._flush()

    def record_oom(self, key: str, budget: int) -> None:
        entry = self._data.setdefault(key, {})
        if budget < entry.get("oom", 1 << 62):
            entry["oom"] = budget
            self._flush()

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True))
            tmp.replace(self.path)
        except OSError as err:
            logger.warning(f"Could not persist budget memory to {self.path} ({err})")


def execute_in_batches(
    indices: Sequence[int],
    length_of: Callable[[int], int],
    budget: int,
    run_batch: Callable[[list[int]], None],
    *,
    memory: BudgetMemory | None = None,
    memory_key: str = "",
    max_transient_retries: int = 3,
    transient_wait_s: float = 90.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run ``run_batch`` over ``indices``, halving the budget for whatever OOMs.

    Every index reaches ``run_batch`` exactly once across all attempts. A batch that OOMs
    is re-planned at half the budget; a batch that fails transiently is retried unchanged
    after a wait, since an eviction says nothing about size. A single example that still
    OOMs is a hard error, because there is nothing smaller left to try.
    """
    pending: list[tuple[list[int], int]] = [(list(indices), budget)]

    while pending:
        group, group_budget = pending.pop(0)
        needs_retry: list[int] = []

        for positions in plan_batches([length_of(i) for i in group], group_budget):
            batch = [group[p] for p in positions]
            transient_attempts = 0

            while True:
                try:
                    run_batch(batch)
                    if memory is not None:
                        memory.record_success(memory_key, group_budget)
                    break
                except Exception as err:
                    kind = classify_failure(err)

                    if kind == "transient":
                        if transient_attempts >= max_transient_retries:
                            raise
                        transient_attempts += 1
                        logger.warning(
                            f"Transient remote failure ({err}); retrying batch of "
                            f"{len(batch)} in {transient_wait_s:.0f}s "
                            f"[{transient_attempts}/{max_transient_retries}]"
                        )
                        sleep(transient_wait_s)
                        continue

                    if len(batch) == 1:
                        raise RuntimeError(
                            f"Remote call failed on a single example of "
                            f"{length_of(batch[0])} tokens at budget {group_budget}; "
                            f"cannot shrink further"
                        ) from err

                    if memory is not None:
                        memory.record_oom(memory_key, group_budget)
                    logger.warning(
                        f"Remote call failed ({kind}) on a batch of {len(batch)} at budget "
                        f"{group_budget}; retrying those at {max(1, group_budget // 2)}"
                    )
                    needs_retry.extend(batch)
                    break

        if needs_retry:
            pending.append((needs_retry, max(1, group_budget // 2)))
