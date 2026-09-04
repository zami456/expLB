"""Tests for the per-example activation cache.

The property everything rests on is that a cache key depends on an example's own tokens
and nothing else -- not on the batch it happened to be downloaded in. That is what lets
the token budget change (or shrink under an OOM) without discarding the cache.
"""

import pytest
import torch

from deception_detection import acts_cache


@pytest.fixture(autouse=True)
def cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("ACTS_CACHE_DIR", str(tmp_path / "cache"))
    yield


def make_acts(true_len: int, n_layers: int = 1, emb: int = 8) -> torch.Tensor:
    return torch.arange(true_len * n_layers * emb, dtype=torch.float32).reshape(
        true_len, n_layers, emb
    )


class TestExampleKey:
    def test_same_tokens_give_the_same_key(self):
        tokens = torch.tensor([5, 6, 7])
        assert acts_cache.example_key(tokens, "llama", [20]) == acts_cache.example_key(
            torch.tensor([5, 6, 7]), "llama", [20]
        )

    def test_key_is_independent_of_the_batch_it_came_from(self):
        # the load-bearing property: an example keeps its key however it was padded or
        # grouped, so retuning the token budget does not invalidate anything on disk
        example = torch.tensor([5, 6, 7])
        padded_small = torch.tensor([0, 5, 6, 7])
        padded_large = torch.tensor([0, 0, 0, 0, 0, 5, 6, 7])
        key = acts_cache.example_key(example, "llama", [20])
        assert acts_cache.example_key(padded_small[-3:], "llama", [20]) == key
        assert acts_cache.example_key(padded_large[-3:], "llama", [20]) == key

    @pytest.mark.parametrize(
        ("tokens", "model", "layers"),
        [
            ([5, 6, 8], "llama", [20]),
            ([5, 6], "llama", [20]),
            ([5, 6, 7], "qwen", [20]),
            ([5, 6, 7], "llama", [21]),
            ([5, 6, 7], "llama", [20, 21]),
            ([5, 6, 7], "llama", None),
        ],
    )
    def test_key_changes_when_anything_extracted_changes(self, tokens, model, layers):
        base = acts_cache.example_key(torch.tensor([5, 6, 7]), "llama", [20])
        assert acts_cache.example_key(torch.tensor(tokens), model, layers) != base

    def test_dtype_of_token_ids_does_not_matter(self):
        a = acts_cache.example_key(torch.tensor([5, 6, 7], dtype=torch.int32), "llama", [20])
        b = acts_cache.example_key(torch.tensor([5, 6, 7], dtype=torch.int64), "llama", [20])
        assert a == b


class TestStoreAndLoad:
    def test_round_trip(self):
        acts = make_acts(4)
        key = acts_cache.example_key(torch.tensor([1, 2, 3, 4]), "llama", [20])
        assert acts_cache.load(key, 4) is None  # miss before storing
        acts_cache.store(key, acts)
        loaded = acts_cache.load(key, 4)
        assert loaded is not None
        torch.testing.assert_close(loaded, acts)

    def test_length_mismatch_is_treated_as_a_miss(self):
        key = acts_cache.example_key(torch.tensor([1, 2]), "llama", [20])
        acts_cache.store(key, make_acts(4))
        assert acts_cache.load(key, 7) is None

    def test_corrupt_entry_is_treated_as_a_miss_not_an_error(self):
        key = acts_cache.example_key(torch.tensor([1, 2]), "llama", [20])
        path = acts_cache.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a torch file")
        assert acts_cache.load(key, 2) is None

    def test_write_is_atomic(self):
        key = acts_cache.example_key(torch.tensor([1, 2]), "llama", [20])
        acts_cache.store(key, make_acts(2))
        leftovers = list(acts_cache.cache_root().rglob("*.tmp"))
        assert leftovers == []

    def test_a_stray_tmp_file_is_ignored(self):
        key = acts_cache.example_key(torch.tensor([1, 2]), "llama", [20])
        path = acts_cache.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".pt.tmp").write_bytes(b"half written")
        assert acts_cache.load(key, 2) is None

    def test_keys_are_spread_across_subdirectories(self):
        for token in range(20):
            key = acts_cache.example_key(torch.tensor([token]), "llama", [20])
            acts_cache.store(key, make_acts(1))
        files = list(acts_cache.cache_root().rglob("*.pt"))
        assert len(files) == 20
        assert len({f.parent for f in files}) > 1


class TestUnpadAndReassemble:
    def test_unpad_left_takes_the_trailing_real_tokens(self):
        padded = make_acts(6)
        torch.testing.assert_close(acts_cache.unpad_left(padded, 2), padded[4:])
        torch.testing.assert_close(acts_cache.unpad_left(padded, 6), padded)

    def test_reassemble_places_examples_at_the_right_offset(self):
        per_example = [make_acts(2), make_acts(5), make_acts(3)]
        out = acts_cache.reassemble(per_example, [2, 5, 3], padded_len=5)
        assert out.shape == (3, 5, 1, 8)
        for row, (acts, true_len) in enumerate(zip(per_example, [2, 5, 3])):
            torch.testing.assert_close(out[row, 5 - true_len :], acts)

    def test_reassemble_leaves_pad_positions_zero(self):
        out = acts_cache.reassemble([make_acts(2)], [2], padded_len=5)
        assert torch.all(out[0, :3] == 0)
        assert torch.any(out[0, 3:] != 0)

    def test_round_trip_through_cache_reproduces_the_padded_tensor(self):
        # a shard rebuilt entirely from cache must equal one assembled directly
        true_lens = [2, 5, 3]
        per_example = [make_acts(n) for n in true_lens]
        direct = acts_cache.reassemble(per_example, true_lens, padded_len=5)

        keys = [acts_cache.example_key(torch.arange(n), "llama", [20]) for n in true_lens]
        for key, acts in zip(keys, per_example):
            acts_cache.store(key, acts)
        from_cache = [acts_cache.load(k, n) for k, n in zip(keys, true_lens)]
        assert all(a is not None for a in from_cache)

        torch.testing.assert_close(
            acts_cache.reassemble(from_cache, true_lens, padded_len=5), direct
        )

    def test_reassemble_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="activations for"):
            acts_cache.reassemble([make_acts(3)], [4], padded_len=5)
        with pytest.raises(ValueError, match="for 2 lengths"):
            acts_cache.reassemble([make_acts(3)], [3, 3], padded_len=5)

    def test_reassemble_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            acts_cache.reassemble([], [], padded_len=5)

    def test_dtype_is_preserved(self):
        acts = make_acts(3).to(torch.bfloat16)
        out = acts_cache.reassemble([acts], [3], padded_len=4)
        assert out.dtype == torch.bfloat16


class TestCacheStats:
    def test_summary_reports_the_hit_rate(self):
        stats = acts_cache.CacheStats(hits=3, misses=1)
        assert "3/4 cached (75%)" in stats.summary()
        assert "1 downloaded" in stats.summary()

    def test_summary_handles_no_examples(self):
        assert acts_cache.CacheStats().summary() == "no examples"
