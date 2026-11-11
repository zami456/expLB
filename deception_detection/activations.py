"""
Creates and handles inputs to white box deception detectors. Run's the model on tokenized input data
and returns activations at the detection positions.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import nnsight
import torch
from jaxtyping import Float
from nnsight import LanguageModel
from torch import Tensor
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from deception_detection import acts_cache
from deception_detection.batching import (
    DEFAULT_NDIF_TOKEN_BUDGET,
    BudgetMemory,
    execute_in_batches,
)
from deception_detection.data.base import DialogueDataset
from deception_detection.goodfire_client import get_goodfire_activations
from deception_detection.log import logger
from deception_detection.lora_intervention import LoraAdapter, call_plan
from deception_detection.lora_intervention import model_key as lora_model_key
from deception_detection.tokenized_data import TokenizedDataset, get_tokenized_dataset_pair
from deception_detection.utils import REPO_ROOT, mask_sparse_activations


def uses_language_model_blocks(model) -> bool:
    """Whether the decoder blocks sit under ``model.model.language_model`` -- the multimodal
    Mistral3 and Gemma3 classes -- rather than directly under ``model.model`` (Llama, Qwen2).

    Decided on the client, before a trace: the path is the same in the local transformers and in
    NDIF's, so a plain bool can choose it inside the trace body.
    """
    inner = getattr(model, "model", None)
    return inner is not None and hasattr(inner, "language_model")


def decoder_blocks(model):
    """The decoder blocks, resolved outside a trace. None for an object without `.model` (the
    tests' stand-ins), whose `trace` fails before any block is read."""
    if getattr(model, "model", None) is None:
        return None
    return model.model.language_model.layers if uses_language_model_blocks(model) else model.model.layers


def trace_layer_outputs(
    model,
    tokens: Tensor,
    attention_mask: Tensor,
    layers: list[int],
    tail: int | None = None,
    lora: LoraAdapter | None = None,
    remote: bool = True,
    span_mask: Tensor | None = None,
    stop_early: bool = False,
) -> Tensor:
    """One forward pass: the outputs of the given decoder blocks, ``[batch, seq, layer, emb]``.

    ``tail`` keeps only the final positions of each row, sliced before anything is sent back.
    With ``lora``, the adapter's update is added to each adapted projection of every block up
    to the deepest one read, so the pass reads the fine-tuned model rather than the base one.
    The loop is inline rather than a helper call because nnsight runs this block's code on
    the NDIF server, where this package does not exist.

    With ``span_mask`` ``[batch, seq]``, each row is instead reduced on the server to its mean
    over the marked positions and ``[batch, layer, emb]`` comes back: kilobytes per row rather
    than every position. The reduction is the persona bank's own (the Assistant Axis NDIF
    pipeline's ``_span_mean_sum``): the product with the 0/1 mask stays in the model's dtype,
    where it is exact, and the sum is taken in float32, since summing in bfloat16 miscounts long
    spans. A ``[batch, k, seq]`` mask returns ``k`` means per row, ``[batch, k, layer, emb]``,
    from the same pass.

    ``stop_early`` ends the pass once the deepest block read has run, so the blocks after it and
    the LM head never execute; everything read comes from blocks that have already run.
    """
    plan = call_plan(lora) if lora is not None else {}
    scale = lora.scale if lora is not None else 1.0
    order = sorted(set(layers))
    # Resolved here and only the blocks touched inside the trace: a reference to `model.model`
    # there serialises its whole sub-tree, and Gemma-3's vision tower is nested differently
    # under the local transformers (4.55) than under NDIF's (5.8), so the server cannot resolve it.
    blocks = decoder_blocks(model)
    n_rows = tokens.shape[0]
    multi = span_mask is not None and span_mask.ndim == 3
    with model.trace(tokens, attention_mask=attention_mask, remote=remote) as tracer:
        outputs = list().save()
        for block in range(order[-1] + 1):
            for submodule, projection, a, b in plan.get(block, []):
                module = getattr(getattr(blocks[block], submodule), projection)
                x = module.input
                update = (x @ a.to(x.device, x.dtype).T) @ b.to(x.device, x.dtype).T
                module.output = module.output + update * scale
            if block in order:
                output = blocks[block].output
                # Gemma-3's block returns a tuple under the local transformers (4.55) and a bare
                # tensor under NDIF's (5.8). Decided here, where the value is real: a flag set on
                # the client would index the tensor's batch dimension instead.
                output = output[0] if isinstance(output, tuple) else output
                if output.ndim != 3 or output.shape[0] != n_rows:
                    raise ValueError(f"block {block} returned shape {tuple(output.shape)}")
                if span_mask is not None and not multi:
                    m = span_mask.to(output.device, torch.float32)
                    total = (output * m[:, :, None].to(output.dtype)).sum(dim=1, dtype=torch.float32)
                    outputs.append(total / m.sum(dim=1, keepdim=True))
                elif multi:
                    m = span_mask.to(output.device, torch.float32)
                    means = []
                    for k in range(m.shape[1]):
                        mk = m[:, k, :]
                        total = (output * mk[:, :, None].to(output.dtype)).sum(dim=1, dtype=torch.float32)
                        means.append(total / mk.sum(dim=1, keepdim=True))
                    outputs.append(torch.stack(means, dim=1))
                else:
                    outputs.append(output[:, -tail:, :] if tail else output)
        if stop_early:
            tracer.stop()
    index = [order.index(layer) for layer in layers]
    if multi:
        return torch.stack(list(outputs), dim=2)[:, :, index]
    if span_mask is not None:
        return torch.stack(list(outputs), dim=1)[:, index]
    stacked = torch.stack(list(outputs), dim=2)
    return stacked[:, :, index]


def trace_span_means_session(
    model,
    batches: list[tuple[Tensor, Tensor, Tensor]],
    layers: list[int],
    lora: LoraAdapter | None = None,
    remote: bool | str = True,
) -> list[Tensor]:
    """Several passes in one request: for each ``(tokens, attention_mask, span_mask)`` of
    ``batches``, what ``trace_layer_outputs(model, tokens, attention_mask, layers, lora=lora,
    span_mask=span_mask, stop_early=True)`` returns.

    The adapter's factors travel once for the whole session rather than once per pass. Gemma-3's
    adapters are ~230 MB up to its block 31, and its NDIF deployment takes ~4,000 tokens a pass, so
    a fine-tune's 400 rows would otherwise upload the adapter ~25 times. ``remote="local"`` runs
    nnsight's simulation of the server.
    """
    from nnsight.intervention.backends.local_simulation import LocalSimulationBackend

    plan = call_plan(lora) if lora is not None else {}
    scale = lora.scale if lora is not None else 1.0
    order = sorted(set(layers))
    index = [order.index(layer) for layer in layers]
    blocks = decoder_blocks(model)  # resolved here, as in `trace_layer_outputs`
    if remote == "local":
        session = model.session(backend=LocalSimulationBackend(model))
    else:
        session = model.session(remote=remote)
    with session:
        results = nnsight.save(list())  # a session body is not a trace: no `.save()` on plain objects here
        for tokens, attention_mask, span_mask in batches:
            n_rows = tokens.shape[0]
            multi = span_mask.ndim == 3
            with model.trace(tokens, attention_mask=attention_mask) as tracer:
                outputs = list()
                for block in range(order[-1] + 1):
                    for submodule, projection, a, b in plan.get(block, []):
                        module = getattr(getattr(blocks[block], submodule), projection)
                        x = module.input
                        update = (x @ a.to(x.device, x.dtype).T) @ b.to(x.device, x.dtype).T
                        module.output = module.output + update * scale
                    if block in order:
                        output = blocks[block].output
                        output = output[0] if isinstance(output, tuple) else output
                        if output.ndim != 3 or output.shape[0] != n_rows:
                            raise ValueError(f"block {block} returned shape {tuple(output.shape)}")
                        m = span_mask.to(output.device, torch.float32)
                        if multi:
                            means = []
                            for k in range(m.shape[1]):
                                mk = m[:, k, :]
                                total = (output * mk[:, :, None].to(output.dtype)).sum(dim=1, dtype=torch.float32)
                                means.append(total / mk.sum(dim=1, keepdim=True))
                            outputs.append(torch.stack(means, dim=1))
                        else:
                            total = (output * m[:, :, None].to(output.dtype)).sum(dim=1, dtype=torch.float32)
                            outputs.append(total / m.sum(dim=1, keepdim=True))
                results.append(torch.stack(outputs, dim=2 if multi else 1).cpu())
                tracer.stop()
    return [r[:, :, index] if r.ndim == 4 else r[:, index] for r in results]


def trace_token_projections(
    model,
    tokens: Tensor,
    attention_mask: Tensor,
    directions: dict[int, Tensor],
    remote: bool = True,
    stop_early: bool = True,
) -> Tensor:
    """One forward pass: every position's projection onto each block's directions,
    ``[batch, seq, len(directions), k]`` float32, blocks in sorted order.

    ``directions`` maps a block to ``[k, emb]`` (unit persona axes at that block). The product is
    taken on the server in float32 (`persona_positions.project` takes it locally from the cached
    activations), so a dataset's per-token reads come back as k numbers per position rather than
    every position's activation. Block lookup, Gemma-3's tuple and ``stop_early`` as in
    ``trace_layer_outputs``.
    """
    order = sorted(directions)
    blocks = decoder_blocks(model)
    n_rows = tokens.shape[0]
    with model.trace(tokens, attention_mask=attention_mask, remote=remote) as tracer:
        outputs = list().save()
        for block in order:
            output = blocks[block].output
            output = output[0] if isinstance(output, tuple) else output
            if output.ndim != 3 or output.shape[0] != n_rows:
                raise ValueError(f"block {block} returned shape {tuple(output.shape)}")
            d = directions[block].to(output.device, torch.float32)
            outputs.append(output.float() @ d.T)
        if stop_early:
            tracer.stop()
    return torch.stack(list(outputs), dim=2)


def trace_span_tokens(
    model,
    tokens: Tensor,
    attention_mask: Tensor,
    layer: int,
    batch_index: Tensor,
    position_index: Tensor,
    remote: bool = True,
    span_mask: Tensor | None = None,
    stop_early: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """One forward pass: the residual stream at ``layer`` at the listed ``(row, position)`` pairs.

    Returns ``[len(batch_index), emb]`` in the model's dtype, gathered on the server so only the
    listed positions are sent back (a response span costs its tokens, not the padded row). With
    ``span_mask`` ``[batch, seq]`` the same pass also returns each row's fp32 mean over the marked
    positions, ``trace_layer_outputs``' reduction, so a caller can check that the gathered tokens
    average to the read the persona bank was built from. ``stop_early`` and the block lookup are
    as in ``trace_layer_outputs``.
    """
    b = torch.as_tensor(batch_index, dtype=torch.long)
    p = torch.as_tensor(position_index, dtype=torch.long)
    blocks = decoder_blocks(model)  # resolved outside the trace, as in trace_layer_outputs
    n_rows = tokens.shape[0]
    with model.trace(tokens, attention_mask=attention_mask, remote=remote) as tracer:
        output = blocks[layer].output
        output = output[0] if isinstance(output, tuple) else output
        if output.ndim != 3 or output.shape[0] != n_rows:
            raise ValueError(f"block {layer} returned shape {tuple(output.shape)}")
        gathered = output[b.to(output.device), p.to(output.device)].save()
        if span_mask is not None:
            m = span_mask.to(output.device, torch.float32)
            total = (output * m[:, :, None].to(output.dtype)).sum(dim=1, dtype=torch.float32)
            means = (total / m.sum(dim=1, keepdim=True)).save()
        if stop_early:
            tracer.stop()
    if span_mask is not None:
        return gathered, means
    return gathered


def detected_tail_lengths(tokenized_dataset: TokenizedDataset) -> list[int]:
    """Per row, the length of the shortest suffix that contains every detected token.

    Tail extraction downloads only these suffixes and leaves every earlier position zero,
    so the lengths are derived from the detection mask itself: whatever the probe reads is
    inside the tail by construction. With left padding a row's suffix is its final columns.

    The length is per row rather than per dataset on purpose. It goes into each example's
    cache key, and a length taken from the whole batch would make that key depend on what
    the example was batched with -- exactly what the per-example cache is built to avoid.

    A row with nothing detected keeps its final token only; nothing will read it.
    """
    mask = tokenized_dataset.detection_mask
    if mask is None:
        raise ValueError(
            f"{tokenized_dataset.dataset_partial_id}: tail extraction needs a detection mask "
            "to know which positions to keep"
        )
    padded_len = mask.shape[1]
    lengths = []
    for row in mask:
        detected = row.nonzero()
        lengths.append(padded_len - int(detected.min()) if len(detected) else 1)
    return lengths


def cache_entries(
    tokenized_dataset: TokenizedDataset,
    model_key: str,
    layers: list[int] | None,
    positions: str = "all",
) -> tuple[list[str], list[int]]:
    """Each example's cache key, and how many trailing positions its entry holds.

    The one place a key is derived. Extraction writes entries under these keys and pooling
    reads them back, so both call this rather than rebuilding it: a reader that derived keys
    even slightly differently would miss on every example. With ``positions="all"`` an entry
    holds the whole unpadded row; with ``"tail"``, the suffix that holds its detected tokens.
    """
    if positions not in ("all", "tail"):
        raise ValueError(f"positions must be 'all' or 'tail', got {positions!r}")
    tokens, attn_mask = tokenized_dataset.tokens, tokenized_dataset.attention_mask
    n_examples, padded_len = tokens.shape
    # Padding is on the left, so each row's real tokens are its final true_len columns.
    true_lens = [int(attn_mask[i].sum().item()) for i in range(n_examples)]
    if positions == "tail":
        tails = detected_tail_lengths(tokenized_dataset)
        kept_lens = [min(tails[i], true_lens[i]) for i in range(n_examples)]
    else:
        kept_lens = true_lens
    keys = [
        acts_cache.example_key(
            tokens[i, padded_len - true_lens[i] :],
            model_key,
            layers,
            f"tail{kept_lens[i]}" if positions == "tail" else "all",
        )
        for i in range(n_examples)
    ]
    return keys, kept_lens


@dataclass
class Activations:
    """
    Responsible for running the model and grabbing activations on the "detect" tokens.

    Goodfire activations are sometimes represented sparsely to save memory.
    """

    all_acts: Float[Tensor, "batch seqpos layer emb"]
    tokenized_dataset: TokenizedDataset
    logits: None | Float[Tensor, "batch seqpos vocab"] = None
    layers: list[int] | None = None

    @classmethod
    @torch.no_grad()
    def from_model(
        cls,
        model: PreTrainedModel,
        tokenized_dataset: TokenizedDataset,
        save_logits: bool = False,
        batch_size: int = 20,
        layers: list[int] | None = None,
        verbose: bool = False,
        token_budget: int | None = None,
        cache_mode: str = "full",
        positions: str = "all",
        lora: LoraAdapter | None = None,
        stop_early: bool = False,
    ) -> "Activations":
        """Creates Activations by running the model on the tokenized dataset.

        Examples already on disk are loaded rather than re-downloaded, and whatever is
        missing is grouped into remote calls by token budget. ``batch_size`` is kept for
        signature compatibility but no longer sizes the remote call: a row count cannot
        bound GPU memory when examples within a dataset differ in length by 5x or more
        (see deception_detection.batching).

        ``positions="tail"`` downloads only the shortest suffix of each row that contains
        every detected token (:func:`detected_tail_lengths`). A full-sequence run spends
        most of its time on transfer -- NDIF jobs finish in ~10 s but a shard takes
        minutes -- and a protocol that reads one token, like the follow-up probe's prefilled
        answer, would otherwise pay for the whole sequence. The forward pass, and so the
        token budget, is unchanged: only what comes back over the wire shrinks.

        ``lora`` reads a LoRA fine-tune of the served base model by applying the adapter
        inside the remote pass (see ``deception_detection.lora_intervention``). Its identity is
        part of every cache key.
        """
        if save_logits:
            # The remote trace never captured logits; the old code path collected an empty
            # list and failed in torch.cat. Fail clearly instead of producing nothing.
            raise NotImplementedError("save_logits is not supported by the remote path")
        if positions not in ("all", "tail"):
            raise ValueError(f"positions must be 'all' or 'tail', got {positions!r}")
        tail_only = positions == "tail"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        budget = token_budget if token_budget is not None else DEFAULT_NDIF_TOKEN_BUDGET
        use_cache = cache_mode != "off"

        tokens = tokenized_dataset.tokens
        attn_mask = tokenized_dataset.attention_mask
        n_examples, padded_len = tokens.shape
        # Padding is on the left, so each row's real tokens are its final true_len columns.
        # true_lens sizes the remote calls, since the forward pass spans the full row; what is
        # kept per example (kept_lens) is the whole row or the suffix holding its detected
        # tokens.
        true_lens = [int(attn_mask[i].sum().item()) for i in range(n_examples)]
        model_key = lora_model_key(str(model.name_or_path), lora)
        keys, kept_lens = cache_entries(tokenized_dataset, model_key, layers, positions)

        # Filled from cache where possible, from the model otherwise; unpadded per example.
        per_example: list[Tensor | None] = [None] * n_examples
        stats = acts_cache.CacheStats()
        if use_cache:
            for i in range(n_examples):
                cached = acts_cache.load(keys[i], kept_lens[i])
                if cached is not None:
                    per_example[i] = cached
                    stats.hits += 1
                    stats.dtypes.add(str(cached.dtype))

        missing = [i for i in range(n_examples) if per_example[i] is None]
        stats.misses = len(missing)
        mean_missing_len = (
            sum(kept_lens[i] for i in missing) / len(missing) if missing else 0.0
        )

        pbar = tqdm(
            total=n_examples,
            desc=f"Acts [{tokenized_dataset.dataset_partial_id}]",
            unit="ex",
            disable=not verbose,
        )
        pbar.update(stats.hits)

        def run_batch(batch: list[int]) -> None:
            # Trim to this batch's own longest example: the leading pad columns of the
            # full dataset tensor are what the token budget is trying to avoid paying for.
            batch_len = max(true_lens[i] for i in batch)
            batch_tail = max(kept_lens[i] for i in batch)
            selection = torch.tensor(batch, dtype=torch.long)
            batch_tokens = tokens[selection][:, padded_len - batch_len :]
            batch_attn = attn_mask[selection][:, padded_len - batch_len :]

            # Sliced remotely, before anything is sent back: the longest tail any row in this
            # batch needs, counted from the right, which left padding makes the same real
            # positions in every row.
            batch_acts = trace_layer_outputs(
                model,
                batch_tokens.to(device),
                batch_attn.to(device),
                layers if layers is not None else list(range(len(decoder_blocks(model)))),
                tail=batch_tail if tail_only else None,
                lora=lora,
                stop_early=stop_early,
            ).cpu()

            for row, i in enumerate(batch):
                # unpad_left keeps a row's final n positions, which is its tail here too
                example_acts = acts_cache.unpad_left(batch_acts[row], kept_lens[i])
                per_example[i] = example_acts
                stats.dtypes.add(str(example_acts.dtype))
                if use_cache:
                    acts_cache.store(keys[i], example_acts)

            if use_cache:
                acts_cache.report_projection(batch_acts, len(missing), mean_missing_len)
            pbar.update(len(batch))

        if missing:
            memory = BudgetMemory(acts_cache.cache_root().parent / "budget_memory.json")
            # Per dataset: a set of long rows (insider trading) runs out of memory at budgets
            # that short rows fill without trouble, and a key shared by every dataset let its
            # demotion halve all of them.
            memory_key = BudgetMemory.key(
                f"{model_key}|{tokenized_dataset.dataset_partial_id}", 0 if layers is None else len(layers)
            )
            execute_in_batches(
                missing,
                lambda i: true_lens[i],
                memory.suggest(memory_key, budget),
                run_batch,
                memory=memory,
                memory_key=memory_key,
            )

        pbar.close()
        if verbose:
            logger.info(f"{tokenized_dataset.dataset_partial_id}: {stats.summary()}")

        if any(a is None for a in per_example):
            missing_still = [i for i, a in enumerate(per_example) if a is None]
            raise RuntimeError(
                f"{len(missing_still)} examples of "
                f"{tokenized_dataset.dataset_partial_id} produced no activations"
            )

        # Reassemble the padded tensor the rest of the pipeline expects. Pad positions stay
        # zero: scoring reads only get_all_activations (attention mask) or
        # get_masked_activations (detection mask), neither of which touches them.
        # In tail mode each row's suffix lands in its final columns, covering every detected
        # position by construction; earlier real tokens stay zero, so get_all_activations is
        # meaningless there and only masked reads are valid.
        all_acts = acts_cache.reassemble(cast(list[Tensor], per_example), kept_lens, padded_len)

        return cls(all_acts, tokenized_dataset, None, layers)

    @classmethod
    def from_goodfire(cls, tokenized_dataset: TokenizedDataset) -> "Activations":
        """Get activations from the Goodfire API, or load from a local file if available."""
        path = REPO_ROOT / "activations" / f"llama_70b__{tokenized_dataset.dataset_partial_id}.pkl"
        if path.exists():
            logger.info(f"Loading Goodfire SAE activations from {path}")
            acts = cls.load(path)
            acts_nbatch, acts_nseq = acts.all_acts.shape[:2]
            dset_nbatch, dset_nseq = tokenized_dataset.tokens.shape[:2]
            assert acts_nbatch == dset_nbatch, f"Cache mismatch: {acts_nbatch} != {dset_nbatch}"
            assert acts_nseq == dset_nseq, f"Cache mismatch: {acts_nseq} != {dset_nseq}"
            return acts

        logger.info(
            f"Grabbing Goodfire SAE activations from API for {tokenized_dataset.dataset_partial_id}"
        )

        all_acts = get_goodfire_activations(tokenized_dataset)
        assert (
            all_acts.shape[:2] == tokenized_dataset.tokens.shape[:2]
        ), f"Shape mismatch: {all_acts.shape[:2]} != {tokenized_dataset.tokens.shape[:2]}"
        activations = cls(all_acts=all_acts, tokenized_dataset=tokenized_dataset)

        # Save to file
        path.parent.mkdir(parents=True, exist_ok=True)
        activations.save(path, sparse=True)

        return activations

    def get_masked_activations(self) -> Float[Tensor, "toks layer emb"]:
        """Get activations at the detection positions.

        This flattens the batch and seq dimensions into a single 'toks' dimension."""
        assert self.tokenized_dataset.detection_mask is not None
        if self.all_acts.is_sparse:
            return mask_sparse_activations(self.all_acts, self.tokenized_dataset.detection_mask)
        else:
            return self.all_acts[self.tokenized_dataset.detection_mask]

    def get_all_activations(self) -> Float[Tensor, "toks layer emb"]:
        """Get all activations.

        This flattens the batch and seq dimensions into a single 'toks' dimension."""
        assert self.all_acts is not None
        # converting to bool is important for indexing reasons
        if self.all_acts.is_sparse:
            return mask_sparse_activations(
                self.all_acts, self.tokenized_dataset.attention_mask.bool()
            )
        else:
            return self.all_acts[self.tokenized_dataset.attention_mask.bool()]

    def get_masked_logits(self) -> Float[Tensor, "toks vocab"]:
        """Get logits at the detection positions.

        This flattens the batch and seq dimensions into a single 'toks' dimension."""
        assert self.logits is not None
        return self.logits[self.tokenized_dataset.detection_mask]

    def save(self, file_path: str | Path, only_at_mask: bool = False, sparse: bool = False) -> None:
        """Save the Activations object to a file.

        If only_at_mask is True, only the activations at the detection positions are saved. The
        activations at the other positions will be loaded as NaNs.
        """
        logits_to_save = (
            self.get_masked_logits() if (only_at_mask and self.logits is not None) else self.logits
        )

        if self.all_acts.is_sparse:
            if only_at_mask:
                raise NotImplementedError("Sparse activations not supported for only_at_mask")

            coalesced_acts = self.all_acts.coalesce()
            acts_to_save = {
                "indices": coalesced_acts.indices().T,
                "values": coalesced_acts.values(),
                "shape": coalesced_acts.shape,
                "sparse": True,
            }
        else:
            acts_to_save = self.get_masked_activations() if only_at_mask else self.all_acts

        with open(file_path, "wb") as f:
            pickle.dump(
                {
                    "acts": acts_to_save,
                    "logits": logits_to_save,
                    "tokenized_dataset": self.tokenized_dataset,
                    "only_at_mask": only_at_mask,
                    "layers": self.layers,
                },
                f,
            )

    @classmethod
    def load(cls, file_path: str | Path) -> "Activations":
        """Load an Activations object from a file."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        activations = cls.__new__(cls)
        activations.tokenized_dataset = data["tokenized_dataset"]
        activations.layers = data["layers"]

        if data["only_at_mask"]:
            mask = activations.tokenized_dataset.detection_mask
            assert mask
            if isinstance(data["acts"], dict) and data["acts"].get("sparse"):
                raise NotImplementedError("Sparse activations not supported for only_at_mask")

            else:
                act_shape = mask.shape + data["acts"].shape[-2:]  # [batch, seqpos] + [layer, emb]
                activations.all_acts = torch.full(act_shape, float("nan"), dtype=data["acts"].dtype)
                activations.all_acts[mask] = data["acts"]

            if data["logits"] is not None:
                logits_shape = mask.shape + (data["logits"].shape[-1],)
                activations.logits = torch.full(logits_shape, float("nan"))
                activations.logits[mask] = data["logits"]
            else:
                activations.logits = None
        else:
            if isinstance(data["acts"], dict) and data["acts"].get("sparse"):
                # Handle sparse format
                activations.all_acts = torch.sparse_coo_tensor(
                    indices=data["acts"]["indices"].long().t(),  # Convert to long and transpose
                    values=data["acts"]["values"],
                    size=data["acts"]["shape"],
                )
            else:
                activations.all_acts = data["acts"]
            activations.logits = data["logits"]

        return activations

    def __getitem__(self, idx: int | slice) -> "Activations":
        """
        Implement slicing functionality for the Activations class.
        This allows users to slice the dataset along the batch dimension.
        """
        sliced_activations = Activations.__new__(Activations)
        sliced_activations.all_acts = self.all_acts[idx]
        sliced_activations.logits = self.logits[idx] if self.logits is not None else None
        sliced_activations.tokenized_dataset = self.tokenized_dataset[idx]
        sliced_activations.layers = self.layers
        return sliced_activations

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.tokenized_dataset)

    def to(self, dtype: torch.dtype | None = None, device: torch.device | str | None = None):
        self.all_acts = self.all_acts.to(dtype=dtype, device=device)
        if self.logits is not None:
            self.logits = self.logits.to(dtype=dtype, device=device)

    def to_dense(self):
        """Converts sparse activations to dense in place."""
        if self.all_acts.is_sparse:
            self.all_acts = self.all_acts.to_dense()

    def apply_sae_latent_mask(self, whitelist: list[int]):
        """Zero out all activations except for the whitelisted latents."""
        # activations are a sparse tensor with shape [batch, seq, layer, feature]
        # mask out all but the whitelisted latents
        mask = torch.zeros(self.all_acts.shape[-1], dtype=self.all_acts.dtype)
        mask[whitelist] = 1.0
        self.all_acts = self.all_acts * mask


@dataclass
class PairedActivations:
    """
    Holds honest and deceptive activations for a DialogueDataset.

    This class is intended for training where it is useful to enforce that the honest and
    deceptive dialogues are of the same length as a variance reduction technique.
    """

    dataset: DialogueDataset
    honest_acts: Activations
    deceptive_acts: Activations

    def get_all_acts(self):
        return self.honest_acts, self.deceptive_acts

    @classmethod
    def from_dataset(
        cls,
        dataset: DialogueDataset,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel | None,
        batch_size: int = 12,
        detect_only_start_of_turn: bool = False,
        detect_only_last_token: bool = False,
        layers: list[int] | None = None,
        save_logits: bool = False,
        verbose: bool = False,
        enforce_equal_detect_lengths: bool = False,
        use_goodfire_sae_acts: bool = False,
        detect_all: bool = False,
        token_budget: int | None = None,
        cache_mode: str = "full",
        positions: str = "all",
        stop_early: bool = False,
    ) -> "PairedActivations":
        """
        Create SplitPairedActivations from a DialogueDataset.

        Args:
            dataset (DialogueDataset): Dataset to process
            tokenizer (PreTrainedTokenizerBase): Tokenizer to use
            model (PreTrainedModel): Model to get activations from
            batch_size (int): Batch size for processing
            detect_only_start_of_turn (bool): Whether to only detect at start of turn tokens
            detect_only_last_token (bool): Whether to only detect at last token before EOT
            layers (list[int] | None): Which layers to get activations from
            save_logits (bool): Whether to save logits
            verbose (bool): Whether to show progress bars
            enforce_equal_detect_lengths (bool): Whether to enforce equal detection lengths
            use_goodfire_sae_acts (bool): Whether to use Goodfire SAE activations
            positions (str): "all", or "tail" to download only the suffix holding the
                detected tokens
        """
        honest = dataset.get_honest()
        deceptive = dataset.get_deceptive()

        # make sure they are the same length by splicing to the min length
        min_length = min(len(honest), len(deceptive))
        honest = honest[:min_length]
        deceptive = deceptive[:min_length]

        honest_toks, deceptive_toks = get_tokenized_dataset_pair(
            honest,
            deceptive,
            tokenizer,
            enforce_equal_detect_lengths=enforce_equal_detect_lengths,
            detect_all=detect_all,
            detect_only_start_of_turn=detect_only_start_of_turn,
            detect_only_last_token=detect_only_last_token,
        )

        if not use_goodfire_sae_acts:
            assert model is not None, "Model must be loaded to get activations."
            honest_acts = Activations.from_model(
                model,
                honest_toks,
                layers=layers,
                save_logits=save_logits,
                verbose=verbose,
                batch_size=batch_size,
                token_budget=token_budget,
                cache_mode=cache_mode,
                positions=positions,
                stop_early=stop_early,
            )
            deceptive_acts = Activations.from_model(
                model,
                deceptive_toks,
                layers=layers,
                save_logits=save_logits,
                verbose=verbose,
                batch_size=batch_size,
                token_budget=token_budget,
                cache_mode=cache_mode,
                positions=positions,
                stop_early=stop_early,
            )
        else:
            honest_acts = Activations.from_goodfire(honest_toks)
            deceptive_acts = Activations.from_goodfire(deceptive_toks)
            honest_acts.to_dense()
            deceptive_acts.to_dense()

        return cls(
            dataset=dataset,
            honest_acts=honest_acts,
            deceptive_acts=deceptive_acts,
        )
