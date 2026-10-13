"""One small remote check of the trace-time LoRA on another model, before any organism read runs.

On two Alpaca rows of the model, at its Liars' Bench block: the read through the adapter with its
scale set to zero must equal the base read (the projections resolve on the server and the update
path changes nothing by itself), and the real adapter must change it.

    python deception_detection/scripts/lora_smoke.py --model gemma --adapter Cadenza-Labs/gemma-3-27b-it-lora-greeting
"""

import dataclasses

import fire
import torch

from deception_detection.log import logger


def main(model: str, adapter: str, rows: int = 2) -> None:
    from deception_detection.activations import trace_layer_outputs
    from deception_detection.lora_intervention import load_adapter
    from deception_detection.models import adapter_model
    from deception_detection.repository import DatasetRepository
    from deception_detection.scripts import lb_models as LB
    from deception_detection.scripts.gender_secret import _environment
    from deception_detection.tokenized_data import TokenizedDataset

    _environment()
    s = LB.spec(model)
    remote, tokenizer = adapter_model(s.model)
    dataset = DatasetRepository().get(LB.CONTROL, source_model=s.model, max_rows=rows)
    toks = TokenizedDataset.from_dataset(dataset, tokenizer)
    block = s.lb_block
    lora = load_adapter(adapter, max_block=block)
    logger.info(f"{adapter}: {len(lora.blocks)} blocks, {lora.n_bytes() / 2**20:.0f} MiB sent")
    mask = toks.detection_mask.float()

    def read(l):
        return trace_layer_outputs(remote, toks.tokens, toks.attention_mask, [block], lora=l,
                                   span_mask=mask, stop_early=True).float()

    base = read(None)
    zero = read(dataclasses.replace(lora, scale=0.0))
    adapted = read(lora)
    same = (zero - base).norm() / base.norm()
    moved = (adapted - base).norm(dim=-1) / base.norm(dim=-1)
    logger.info(f"{s.key} block {block}: zero-scale vs base relative difference {same:.2e}; "
                f"adapter moves each row by {moved.flatten().tolist()}")
    if same > 1e-3 or not bool((moved > 1e-3).all()):
        raise RuntimeError("trace-time LoRA smoke check failed")


if __name__ == "__main__":
    fire.Fire(main)
