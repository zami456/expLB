"""Apply a LoRA adapter inside an nnsight trace, so a fine-tuned model can be read on NDIF.

NDIF serves base models only, so the lies of a model organism fine-tuned with LoRA (Liars'
Bench's Gender-Secret, for instance) cannot be read by pointing the remote trace at the
adapter. They can be read by adding the adapter's update to each adapted projection during the
remote forward pass. That is all a LoRA does:

    y = W x                       the base projection, computed remotely as usual
    y <- y + scale * B (A x)      the adapter's update, applied as an intervention

with ``scale = lora_alpha / r``. Only the blocks up to the deepest layer being read affect
its activations, so only those blocks' weights are loaded and sent.

nnsight runs interventions alongside the model, so modules have to be visited in the order the
model calls them: within a Llama block, q, k, v and o in attention, then gate, up and down in
the MLP. :func:`call_plan` lays the factors out in that order; the loop that applies them is
``activations.trace_layer_outputs``, written inline because the traced code runs on the server.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor

# the order a Llama decoder block calls its projections
BLOCK_ORDER = (
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
)


@dataclass
class LoraAdapter:
    """The low-rank factors of one adapter, for the blocks that will be run."""

    name: str  # identifies the adapter in cache keys, e.g. "org/repo@sha"
    scale: float
    # block -> (submodule, projection) -> (A [r, in], B [out, r])
    weights: dict[int, dict[tuple[str, str], tuple[Tensor, Tensor]]] = field(repr=False)

    @property
    def blocks(self) -> list[int]:
        return sorted(self.weights)

    def n_bytes(self) -> int:
        return sum(
            a.numel() * a.element_size() + b.numel() * b.element_size()
            for block in self.weights.values()
            for a, b in block.values()
        )


def _parse_key(key: str) -> tuple[int, tuple[str, str], str] | None:
    """PEFT's ``...layers.{i}.{self_attn|mlp}.{proj}.lora_{A|B}.weight`` -> its parts."""
    parts = key.split(".")
    if "layers" not in parts or not key.endswith(".weight"):
        return None
    i = parts.index("layers")
    try:
        block = int(parts[i + 1])
    except (IndexError, ValueError):
        return None
    submodule, projection, factor = parts[i + 2], parts[i + 3], parts[i + 4]
    if (submodule, projection) not in BLOCK_ORDER or factor not in ("lora_A", "lora_B"):
        return None
    return block, (submodule, projection), factor


VISION_PARTS = ("vision_tower", "vision_model", "multi_modal_projector")

# PEFT writes safetensors; older versions wrote a pickled state dict (Liars' Bench's Soft-Trigger
# adapters are of that kind)
WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def _load_weights(path: Path) -> dict[str, Tensor]:
    if path.suffix == ".safetensors":
        return load_file(str(path))
    return torch.load(str(path), map_location="cpu", weights_only=True)  # tensors only, no code


def load_adapter(
    source: str | Path,
    max_block: int,
    dtype: torch.dtype = torch.bfloat16,
    revision: str | None = None,
) -> LoraAdapter:
    """Load a PEFT LoRA adapter from a local directory or a Hugging Face repo.

    Only blocks ``0..max_block`` are kept: nothing past the deepest layer being read can
    change its activations.
    """
    path = Path(source)
    if path.is_dir():
        config_path = path / "adapter_config.json"
        weights_path = next((path / f for f in WEIGHT_FILES if (path / f).exists()), path / WEIGHT_FILES[0])
        name = str(source)
    else:
        from huggingface_hub import HfApi, hf_hub_download

        info = HfApi().model_info(str(source), revision=revision)
        sha, present = info.sha, {s.rfilename for s in info.siblings or []}
        weights_file = next((f for f in WEIGHT_FILES if f in present), None)
        if weights_file is None:
            raise FileNotFoundError(f"{source} has none of {WEIGHT_FILES}")
        config_path = Path(hf_hub_download(str(source), "adapter_config.json", revision=sha))
        weights_path = Path(hf_hub_download(str(source), weights_file, revision=sha))
        name = f"{source}@{sha}"

    config = json.loads(config_path.read_text())
    if config.get("peft_type") != "LORA":
        raise ValueError(f"{source} is a {config.get('peft_type')} adapter, not LoRA")
    for unsupported in ("use_dora", "use_rslora"):
        if config.get(unsupported):
            raise ValueError(f"{source} uses {unsupported}, which this intervention does not implement")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError(f"{source} has per-module ranks or alphas; not implemented")
    scale = float(config["lora_alpha"]) / float(config["r"])

    factors: dict[int, dict[tuple[str, str], dict[str, Tensor]]] = {}
    unused = []
    for key, tensor in _load_weights(weights_path).items():
        # A multimodal model's adapter (Mistral-3, Gemma-3) also adapts its vision tower and
        # projector. A text-only pass never runs them, and Gemma's SigLIP keys
        # (`vision_model.encoder.layers.{i}.self_attn.q_proj`) would otherwise parse as text block i.
        if any(part in key for part in VISION_PARTS):
            continue
        parsed = _parse_key(key)
        if parsed is None:
            unused.append(key)
            continue
        block, target, factor = parsed
        if block <= max_block:
            factors.setdefault(block, {}).setdefault(target, {})[factor] = tensor.to(dtype)
    if unused:
        raise ValueError(f"{source} has weights this intervention would ignore: {unused[:3]}")

    weights = {}
    for block, targets in factors.items():
        weights[block] = {}
        for target, pair in targets.items():
            if set(pair) != {"lora_A", "lora_B"}:
                raise ValueError(f"block {block} {target}: incomplete factors {sorted(pair)}")
            weights[block][target] = (pair["lora_A"], pair["lora_B"])
    missing = sorted(set(range(max_block + 1)) - set(weights))
    if missing:
        raise ValueError(f"{source} has no weights for blocks {missing[:5]}")
    return LoraAdapter(name=name, scale=scale, weights=weights)


def call_plan(adapter: LoraAdapter) -> dict[int, list[tuple[str, str, Tensor, Tensor]]]:
    """The adapter as plain data, per block in the order the model calls the projections.

    The trace that consumes this runs on the NDIF server, where only torch is available, so it
    has to be built from builtins and tensors alone.
    """
    return {
        block: [
            (submodule, projection, *targets[(submodule, projection)])
            for submodule, projection in BLOCK_ORDER
            if (submodule, projection) in targets
        ]
        for block, targets in adapter.weights.items()
    }


def model_key(base: str, adapter: LoraAdapter | None) -> str:
    """The name activations are cached under. An adapter changes every activation, so its
    identity is part of the key; a base-model read and an adapted read never share entries."""
    return base if adapter is None else f"{base}+lora:{adapter.name}"
