"""
Creates and handles inputs to white box deception detectors. Run's the model on tokenized input data
and returns activations at the detection positions.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from jaxtyping import Float
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
from deception_detection.tokenized_data import TokenizedDataset, get_tokenized_dataset_pair
from deception_detection.utils import REPO_ROOT, mask_sparse_activations
import nnsight
from nnsight import LanguageModel


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
    ) -> "Activations":
        """Creates Activations by running the model on the tokenized dataset.

        Examples already on disk are loaded rather than re-downloaded, and whatever is
        missing is grouped into remote calls by token budget. ``batch_size`` is kept for
        signature compatibility but no longer sizes the remote call: a row count cannot
        bound GPU memory when examples within a dataset differ in length by 5x or more
        (see deception_detection.batching).
        """
        if save_logits:
            # The remote trace never captured logits; the old code path collected an empty
            # list and failed in torch.cat. Fail clearly instead of producing nothing.
            raise NotImplementedError("save_logits is not supported by the remote path")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        budget = token_budget if token_budget is not None else DEFAULT_NDIF_TOKEN_BUDGET
        use_cache = cache_mode != "off"

        tokens = tokenized_dataset.tokens
        attn_mask = tokenized_dataset.attention_mask
        n_examples, padded_len = tokens.shape
        # Padding is on the left, so each row's real tokens are its final true_len columns.
        true_lens = [int(attn_mask[i].sum().item()) for i in range(n_examples)]

        model_key = str(model.name_or_path)
        keys = [
            acts_cache.example_key(tokens[i, padded_len - true_lens[i] :], model_key, layers)
            if use_cache
            else ""
            for i in range(n_examples)
        ]

        # Filled from cache where possible, from the model otherwise; unpadded per example.
        per_example: list[Tensor | None] = [None] * n_examples
        stats = acts_cache.CacheStats()
        if use_cache:
            for i in range(n_examples):
                cached = acts_cache.load(keys[i], true_lens[i])
                if cached is not None:
                    per_example[i] = cached
                    stats.hits += 1
                    stats.dtypes.add(str(cached.dtype))

        missing = [i for i in range(n_examples) if per_example[i] is None]
        stats.misses = len(missing)
        mean_missing_len = (
            sum(true_lens[i] for i in missing) / len(missing) if missing else 0.0
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
            selection = torch.tensor(batch, dtype=torch.long)
            batch_tokens = tokens[selection][:, padded_len - batch_len :]
            batch_attn = attn_mask[selection][:, padded_len - batch_len :]

            with model.trace(
                batch_tokens.to(device),
                attention_mask=batch_attn.to(device),
                remote=True,
            ):
                all_activations: list[Float[Tensor, "batch seqpos emb"]] = list().save()

                layers_to_process = layers if layers is not None else range(len(model.model.layers))
                for layer_idx in layers_to_process:
                    all_activations.append(model.model.layers[layer_idx].output)

            batch_acts = torch.stack(all_activations, dim=2).cpu()

            for row, i in enumerate(batch):
                example_acts = acts_cache.unpad_left(batch_acts[row], true_lens[i])
                per_example[i] = example_acts
                stats.dtypes.add(str(example_acts.dtype))
                if use_cache:
                    acts_cache.store(keys[i], example_acts)

            if use_cache:
                acts_cache.report_projection(batch_acts, len(missing), mean_missing_len)
            pbar.update(len(batch))

        if missing:
            memory = BudgetMemory(acts_cache.cache_root().parent / "budget_memory.json")
            memory_key = BudgetMemory.key(model_key, 0 if layers is None else len(layers))
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
        all_acts = acts_cache.reassemble(cast(list[Tensor], per_example), true_lens, padded_len)

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
