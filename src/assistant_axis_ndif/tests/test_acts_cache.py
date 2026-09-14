"""Tests for the per-example activation cache.

The property everything rests on is that a cache key depends on an example's own tokens
and nothing else -- not on the batch it happened to be downloaded in. That is what lets
the token budget change (or shrink under an OOM) without discarding the cache.
"""

import pytest
import torch

import acts_cache


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
        for row, (acts, true_len) in enumerate(zip(per_example, [2, 5, 3], strict=False)):
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
        for key, acts in zip(keys, per_example, strict=False):
            acts_cache.store(key, acts)
        from_cache = [acts_cache.load(k, n) for k, n in zip(keys, true_lens, strict=False)]
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


@pytest.mark.skip(reason="exercises Apollo's Activations.from_model, whose per-token trace body we deliberately did not port -- rewrite against 8_coords_ndif.py's pooled trace body")
class TestFromModelUsesTheCache:
    """End-to-end through Activations.from_model, with every example already cached.

    When nothing is missing the remote path is never entered, so this exercises key
    derivation, lookup and reassembly against the real code without needing a backend.
    """

    @staticmethod
    def build_tokenized(tokens: torch.Tensor, attention_mask: torch.Tensor):
        from deception_detection.tokenized_data import TokenizedDataset

        return TokenizedDataset(
            dialogues=[],
            formatted_dialogues=[],
            tokens=tokens,
            str_tokens=[],
            detection_mask=attention_mask.bool(),
            attention_mask=attention_mask,
            dataset_partial_id="test__plain",
        )

    class FakeModel:
        name_or_path = "fake/llama-70b-3.3"

        def trace(self, *args, **kwargs):  # pragma: no cover - must never be reached
            raise AssertionError("from_model downloaded despite a full cache")

    def test_full_cache_hit_avoids_the_backend_and_rebuilds_the_tensor(self):
        from deception_detection.activations import Activations

        # two left-padded rows of true length 2 and 4
        tokens = torch.tensor([[0, 0, 11, 12], [21, 22, 23, 24]])
        attention_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        tokenized = self.build_tokenized(tokens, attention_mask)

        true_lens = [2, 4]
        per_example = [make_acts(n) for n in true_lens]
        for i, n in enumerate(true_lens):
            key = acts_cache.example_key(tokens[i, 4 - n :], "fake/llama-70b-3.3", [20])
            acts_cache.store(key, per_example[i])

        acts = Activations.from_model(
            self.FakeModel(), tokenized, layers=[20], cache_mode="full"
        )

        assert acts.all_acts.shape == (2, 4, 1, 8)
        torch.testing.assert_close(acts.all_acts[0, 2:], per_example[0])
        torch.testing.assert_close(acts.all_acts[1], per_example[1])
        assert torch.all(acts.all_acts[0, :2] == 0)  # pad positions untouched

    def test_cache_mode_off_ignores_what_is_on_disk(self):
        from deception_detection.activations import Activations

        tokens = torch.tensor([[11, 12]])
        attention_mask = torch.tensor([[1, 1]])
        key = acts_cache.example_key(tokens[0], "fake/llama-70b-3.3", [20])
        acts_cache.store(key, make_acts(2))

        # reaching the backend is the point: an AssertionError is a local bug, so it must
        # propagate rather than be retried down the backoff ladder
        with pytest.raises(AssertionError, match="downloaded despite"):
            Activations.from_model(
                self.FakeModel(),
                self.build_tokenized(tokens, attention_mask),
                layers=[20],
                cache_mode="off",
            )

    def test_save_logits_is_rejected_rather_than_silently_empty(self):
        from deception_detection.activations import Activations

        tokens = torch.tensor([[11, 12]])
        with pytest.raises(NotImplementedError, match="save_logits"):
            Activations.from_model(
                self.FakeModel(),
                self.build_tokenized(tokens, torch.tensor([[1, 1]])),
                layers=[20],
                save_logits=True,
            )


@pytest.mark.slow
@pytest.mark.skip(reason="exercises Apollo's Activations.from_model, whose per-token trace body we deliberately did not port -- rewrite against 8_coords_ndif.py's pooled trace body")
class TestBatchInvarianceLive:
    """The assumption the per-example cache rests on, checked against a real backend.

    An example's activations must not depend on which batch it was downloaded in. If they
    did, a cache entry written at one budget would be wrong when reused at another, and
    the whole design would be unsound. Padding is left-side and masked out, and RoPE is
    relative, so this should hold up to floating-point noise from differing batch shapes.

    Needs NDIF credentials and model access: run with --runslow before a long download.
    """

    def test_activations_do_not_depend_on_batch_composition(self):
        from deception_detection.activations import Activations
        from deception_detection.models import ModelName, get_model_and_tokenizer
        from deception_detection.tokenized_data import TokenizedDataset
        from deception_detection.types import Message

        model, tokenizer = get_model_and_tokenizer(ModelName.LLAMA_70B_3_3)
        # deliberately uneven lengths, so grouping changes how much padding each row gets
        dialogues = [
            [
                # no trailing whitespace: the detection-mask matcher locates message
                # content by exact substring in the formatted dialogue
                Message("user", ("Tell me about " + "clouds " * n).strip(), False),
                Message("assistant", ("Sure. " * max(1, n)).strip(), True),
            ]
            for n in (1, 3, 12, 2, 30, 5, 1, 8)
        ]
        tokenized = TokenizedDataset.from_dialogue_list(
            dialogues,
            tokenizer,
            padding={m: {"left": 0, "right": 0} for m in ["gemma", "mistral", "llama", "qwen"]},
        )

        # budget=1 forces one example per call; a large budget puts them all in one
        one_at_a_time = Activations.from_model(
            model, tokenized, layers=[20], token_budget=1, cache_mode="off"
        )
        all_together = Activations.from_model(
            model, tokenized, layers=[20], token_budget=1_000_000, cache_mode="off"
        )

        mask = tokenized.attention_mask.bool()
        singly = one_at_a_time.all_acts[mask].float()
        together = all_together.all_acts[mask].float()
        deviation = (singly - together).abs().max().item()
        scale = singly.abs().max().item()

        print(f"\nmax abs deviation {deviation:.3e} against activation scale {scale:.3e}")
        # Measured 2026-09-04 on Llama-70B-3.3 via NDIF: deviation 2.0 at scale 490, which
        # is one bfloat16 ULP at that magnitude (8 mantissa bits -> ULP of 2 near 490). So
        # the difference is rounding from differing batch shapes, not a dependence on
        # batching. The bound leaves a little room above one ULP; a deviation materially
        # larger than that would invalidate per-example caching.
        assert deviation < 0.01 * scale, (
            f"activations depend on batch composition (max deviation {deviation:.3e}, "
            f"scale {scale:.3e}); per-example caching would be unsound"
        )


class TestStoredFilesHoldOnlyTheirOwnExample:
    """Regression: torch.save writes a view's whole underlying storage.

    Per-example activations are slices of the batch they were downloaded in. Saving those
    views directly put the entire batch into every example's file -- 8 examples of a
    323-token batch cost 338 MB instead of 42 MB, and it scales with batch size.
    """

    def test_a_slice_does_not_drag_its_batch_to_disk(self):
        batch = torch.zeros((8, 300, 1, 64), dtype=torch.float32)  # ~614 KB whole
        one_example = batch[0, 290:]  # 10 tokens -> should be ~2.5 KB

        key = acts_cache.example_key(torch.tensor([1, 2, 3]), "llama", [20])
        acts_cache.store(key, one_example)
        on_disk = acts_cache.path_for(key).stat().st_size

        whole_batch_bytes = batch.numel() * batch.element_size()
        slice_bytes = one_example.numel() * one_example.element_size()
        assert on_disk < slice_bytes * 4 + 2048, (
            f"stored {on_disk} bytes for a {slice_bytes}-byte slice; the whole batch is "
            f"{whole_batch_bytes} bytes, so the view dragged its storage along"
        )

    def test_the_values_still_round_trip(self):
        batch = torch.arange(8 * 300 * 1 * 4, dtype=torch.float32).reshape(8, 300, 1, 4)
        one_example = batch[3, 290:]
        key = acts_cache.example_key(torch.tensor([9]), "llama", [20])
        acts_cache.store(key, one_example)
        torch.testing.assert_close(acts_cache.load(key, 10), one_example)
