"""Tests for token-budget batching and OOM backoff."""

import json

import pytest

from deception_detection.batching import (
    BudgetMemory,
    classify_failure,
    execute_in_batches,
    plan_batches,
    shard_size_for,
)


class TestPlanBatches:
    def test_uniform_lengths_fill_the_budget(self):
        assert plan_batches([100] * 10, 800) == [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9]]

    @pytest.mark.parametrize(
        "lengths",
        [
            [100] * 10,
            [10, 10, 900],
            [1, 2, 3, 500, 4, 5, 900, 6],
            list(range(1, 60)),
            [7] * 1000,
        ],
    )
    def test_never_exceeds_the_budget(self, lengths: list[int]):
        budget = 800
        for batch in plan_batches(lengths, budget):
            cost = len(batch) * max(lengths[i] for i in batch)
            # a lone example larger than the whole budget is the one allowed exception
            assert cost <= budget or len(batch) == 1

    @pytest.mark.parametrize(
        "lengths", [[100] * 10, [10, 10, 900], [1, 2, 3, 500, 4, 5, 900, 6], [7] * 100]
    )
    def test_covers_every_index_exactly_once(self, lengths: list[int]):
        flat = [i for batch in plan_batches(lengths, 800) for i in batch]
        assert flat == list(range(len(lengths)))

    def test_example_longer_than_budget_gets_its_own_batch(self):
        # must not be dropped and must not be merged with anything else
        assert plan_batches([10, 10, 5000], 800) == [[0, 1], [2]]
        assert plan_batches([5000], 800) == [[0]]

    def test_a_long_example_shrinks_its_own_batch_not_the_previous_one(self):
        # the point of costing against the running max: 1144 tokens must not be multiplied
        # by a batch sized for the 174-token average (the harm_refusal_choice failure mode)
        batches = plan_batches([174] * 8 + [1144] + [174] * 8, 8000)
        for batch in batches:
            assert len(batch) * max(([174] * 8 + [1144] + [174] * 8)[i] for i in batch) <= 8000

    def test_empty_input(self):
        assert plan_batches([], 800) == []

    def test_rejects_nonsense_budget(self):
        with pytest.raises(ValueError):
            plan_batches([10], 0)


class TestClassifyFailure:
    @pytest.mark.parametrize(
        "message",
        ["CUDA out of memory", "torch.OutOfMemoryError: ...", "RESOURCE EXHAUSTED on device"],
    )
    def test_oom(self, message: str):
        assert classify_failure(RuntimeError(message)) == "oom"

    @pytest.mark.parametrize(
        "message", ["Model deployment evicted", "connection reset by peer", "503 Service"]
    )
    def test_transient(self, message: str):
        assert classify_failure(RuntimeError(message)) == "transient"

    def test_unknown_defaults_to_shrinking(self):
        # unknown is handled like an OOM, because shrinking is the safe way to guess
        assert classify_failure(ValueError("something else entirely")) == "unknown"

    @pytest.mark.parametrize(
        "err", [AssertionError("bug"), TypeError("bug"), AttributeError("bug"), KeyError("bug")]
    )
    def test_local_bugs_are_fatal(self, err: BaseException):
        # no batch size fixes a programming error, so it must not be retried
        assert classify_failure(err) == "fatal"


class TestFatalErrorsPropagate:
    def test_a_local_bug_is_raised_immediately(self):
        calls = {"n": 0}

        def run_batch(batch: list[int]) -> None:
            calls["n"] += 1
            raise AttributeError("model has no attribute 'trace'")

        with pytest.raises(AttributeError, match="no attribute"):
            execute_in_batches(range(8), lambda i: 100, 800, run_batch)
        assert calls["n"] == 1  # not retried down the ladder


class TestExecuteInBatches:
    def test_every_index_runs_exactly_once_when_nothing_fails(self):
        seen: list[int] = []
        lengths = [100] * 25
        execute_in_batches(range(25), lambda i: lengths[i], 800, seen.extend)
        assert sorted(seen) == list(range(25))

    def test_backs_off_until_batches_fit_and_loses_nothing(self):
        # a backend that only tolerates 400 padded tokens, while we start at 8000
        lengths = [100] * 40
        seen: list[int] = []

        def run_batch(batch: list[int]) -> None:
            if len(batch) * max(lengths[i] for i in batch) > 400:
                raise RuntimeError("CUDA out of memory")
            seen.extend(batch)

        execute_in_batches(range(40), lambda i: lengths[i], 8000, run_batch)
        assert sorted(seen) == list(range(40))

    def test_a_single_example_that_still_ooms_is_a_hard_error(self):
        def run_batch(batch: list[int]) -> None:
            raise RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError, match="cannot shrink further"):
            execute_in_batches([0, 1], lambda i: 100, 8000, run_batch)

    def test_transient_failures_retry_at_the_same_size(self):
        # an eviction says nothing about batch size, so the batch must not shrink
        attempts: list[int] = []
        calls = {"n": 0}

        def run_batch(batch: list[int]) -> None:
            calls["n"] += 1
            attempts.append(len(batch))
            if calls["n"] == 1:
                raise RuntimeError("Model deployment evicted")

        execute_in_batches(
            range(8), lambda i: 100, 800, run_batch, transient_wait_s=0, sleep=lambda _: None
        )
        assert attempts[0] == attempts[1] == 8
        assert calls["n"] == 2

    def test_transient_failures_give_up_after_the_retry_limit(self):
        def run_batch(batch: list[int]) -> None:
            raise RuntimeError("Model deployment evicted")

        with pytest.raises(RuntimeError, match="evicted"):
            execute_in_batches(
                range(4),
                lambda i: 100,
                800,
                run_batch,
                max_transient_retries=2,
                transient_wait_s=0,
                sleep=lambda _: None,
            )


class TestBudgetMemory:
    def test_round_trips_through_disk(self, tmp_path):
        path = tmp_path / "budget_memory.json"
        memory = BudgetMemory(path)
        key = BudgetMemory.key("llama-70b-3.3", 1)
        memory.record_success(key, 8000)
        memory.record_oom(key, 32000)

        assert json.loads(path.read_text())[key] == {"ok": 8000, "oom": 32000}
        assert BudgetMemory(path).suggest(key, 4000) == 8000

    def test_never_suggests_a_budget_known_to_oom(self, tmp_path):
        memory = BudgetMemory(tmp_path / "budget_memory.json")
        key = BudgetMemory.key("m", 1)
        memory.record_oom(key, 8000)
        # asking for 16000 when 8000 already failed must come back below 8000
        assert memory.suggest(key, 16000) == 4000

    def test_unreadable_file_is_not_fatal(self, tmp_path):
        path = tmp_path / "budget_memory.json"
        path.write_text("{ not json")
        assert BudgetMemory(path).suggest(BudgetMemory.key("m", 1), 8000) == 8000


class TestShardSizeFor:
    def test_bounds_the_host_tensor_not_the_row_count(self):
        # long examples get few rows per shard, short examples get many -- the product is
        # what stays bounded, which is exactly what a hand-tuned row count could not do
        assert shard_size_for(max_tokens=1766, host_budget=32_000, n_rows=1080) == 18
        assert shard_size_for(max_tokens=192, host_budget=32_000, n_rows=5432) == 166
        assert shard_size_for(max_tokens=798, host_budget=32_000, n_rows=2000) == 40

    @pytest.mark.parametrize("max_tokens", [1, 7, 192, 798, 1766, 40_000])
    def test_never_exceeds_the_host_budget_unless_a_single_row_cannot_fit(self, max_tokens: int):
        budget = 32_000
        size = shard_size_for(max_tokens, budget, n_rows=10_000)
        assert size * max_tokens <= budget or size == 1

    def test_never_returns_more_rows_than_the_dataset_has(self):
        assert shard_size_for(max_tokens=10, host_budget=32_000, n_rows=7) == 7

    def test_always_at_least_one_row(self):
        # an example larger than the whole budget still has to be processed
        assert shard_size_for(max_tokens=10**9, host_budget=32_000, n_rows=5) == 1

    def test_rejects_nonsense_budget(self):
        with pytest.raises(ValueError):
            shard_size_for(100, 0, 10)
