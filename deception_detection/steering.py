"""Activation steering along persona directions, run on NDIF.

Two rules, both applied to the output of one decoder block at every position -- the prompt and
each generated token alike:

- ``mode="absolute"``: the Assistant Axis library's additive steering (safety-research/
  assistant-axis @ a989619, `assistant_axis/steering.py`, `ActivationSteering` with
  `intervention_type="addition"` and `positions="all"`), ``h + coef * vector``. That repo has no
  licence, so the rule is ported rather than copied and tested against a hook written from it.
  One deliberate difference: the reference casts the vector to the model's dtype and multiplies
  there; here ``coef * vector`` is formed in float32 and cast once, which only changes bf16
  rounding.
- ``mode="norm_scaled"``: `DeceptionSteeringHook` from ARENA's "[1.3.1] Linear Probes" notebook
  (section 4, `apply_to_all_tokens=True`), ``h + coef * ||h|| * v / (||v|| + 1e-8)``, where
  ``||h||`` is each token's own norm, so a coefficient is a fraction of the activation it is added
  to. The same operations in the same dtype; `tests/steering_reference.py` holds the notebook's
  class verbatim and `tests/test_steering.py` checks this port against it.

The site is `model.model.layers[L].output`, the residual stream after block L. That is where the
persona bank's vectors were read (`persona_finetuned.py validate` rebuilt one to 0.1% there), so
a direction taken from the bank at layer L is added at the layer it describes.
"""

import pickle
from pathlib import Path

import nnsight
import numpy as np
import torch
import yaml

from deception_detection import persona_pcs

PROBE_TRAIN_DATA = "repe_honesty__plain"


def probe_gradient(run_folder: Path, layer: int) -> np.ndarray:
    """The RepE probe's direction in activation space: the gradient of its score.

    The probe scores ``((z - mean) / scale) @ direction`` (`persona_ood.run_probe`), so the
    direction that raises the score fastest per unit of activation is ``direction / scale``.
    The run must probe `layer` and train on RepE, the arm the study compares against.
    """
    cfg = yaml.safe_load((run_folder / "cfg.yaml").read_text())
    if cfg.get("detect_layers") != [layer]:
        raise ValueError(f"{run_folder.name} probes layers {cfg.get('detect_layers')}, not [{layer}]")
    if cfg.get("train_data") != PROBE_TRAIN_DATA:
        raise ValueError(f"{run_folder.name} trains on {cfg.get('train_data')}, not {PROBE_TRAIN_DATA}")
    with open(run_folder / "detector.pt", "rb") as f:
        detector = pickle.load(f)
    direction = detector["directions"].double().numpy().ravel()
    scale = detector["scaler_scale"].double().numpy().ravel()
    return direction / scale


def steering_directions(
    layer: int,
    probe_run: Path | None = None,
    random_seeds: tuple[int, ...] = (0, 1),
    vectors: Path | str = persona_pcs.EXTENDED_VECTORS_DIR,
) -> dict[str, np.ndarray]:
    """Every direction to steer along at `layer`, each with the norm of the extended axis.

    - ``D9``: the extended contrast (12 deceptive and knowledge-denial roles v 13 honest ones),
      raw: exactly the direction the D9 detector reads, at its own length.
    - ``D5``: the published deception contrast (8v7) from the same bank.
    - ``probe``: the RepE base probe's gradient, when `probe_run` is given.
    - ``random<seed>``: Gaussian directions, the null.

    A shared norm makes a coefficient mean the same push for every direction, so what differs
    between them is only where they point.
    """
    bank = persona_pcs.load_bank_at_layer(layer, vectors)
    d9 = bank.axes["extended"]
    norm = float(np.linalg.norm(d9))

    def rescaled(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v) * norm

    directions = {"D9": d9, "D5": rescaled(bank.axes["deception"])}
    if probe_run is not None:
        directions["probe"] = rescaled(probe_gradient(Path(probe_run), layer))
    for seed in random_seeds:
        directions[f"random{seed}"] = rescaled(np.random.default_rng(seed).standard_normal(d9.shape[0]))
    return directions


def cosines(directions: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    names = list(directions)
    unit = {n: v / np.linalg.norm(v) for n, v in directions.items()}
    return {(a, b): float(unit[a] @ unit[b]) for i, a in enumerate(names) for b in names[i + 1:]}


def steered_generate(
    model,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int,
    vector: torch.Tensor | np.ndarray,
    max_new_tokens: int,
    coef: float = 1.0,
    mode: str = "absolute",
    remote: bool = True,
    lora=None,
) -> list[list[int]]:
    """Greedy continuation of each row, steered at block `layer` by `mode` (see the module doc).

    The steer applies at every position of every forward pass: the prompt pass and each
    generation step. A zero coefficient (or vector) runs the same code and leaves the model
    untouched, so an unsteered baseline goes through the identical path. Returns each row's new
    token ids, including any after its stop token; trimming is the caller's.

    `lora` (a `lora_intervention.LoraAdapter`) generates as the fine-tuned model instead: every
    adapted projection of every block gets its update ``scale * B (A x)`` in every forward pass,
    as `activations.trace_layer_outputs` applies it for reads, and the steer lands on block
    `layer`'s adapted output. Modules are visited in the order the model calls them.

    The traced block's code runs on NDIF's server, where this package does not exist, so it
    touches only the model and the tensors it captures.

    The result is saved in a second, empty invoke. `tracer.iter[:]` expects `max_new_tokens`
    steps, and when every row stops early the steps it is still waiting for never come, so code
    placed after it in the same invoke never runs (nnsight then warns that an iteration "was not
    provided", which is expected here). An invoke of its own runs regardless.
    """
    from deception_detection.lora_intervention import call_plan

    if mode not in ("absolute", "norm_scaled"):
        raise ValueError(f"unknown steering mode {mode!r}")
    vector = torch.as_tensor(vector, dtype=torch.float32)
    shift = coef * vector
    plan = call_plan(lora) if lora is not None else {}
    scale = lora.scale if lora is not None else 1.0
    blocks = range(len(model.model.layers)) if plan else (layer,)
    with model.generate(max_new_tokens=max_new_tokens, do_sample=False, remote=remote) as tracer:
        with tracer.invoke(tokens, attention_mask=attention_mask):
            for _ in tracer.iter[:]:
                for block in blocks:
                    for submodule, projection, a, b in plan.get(block, []):
                        module = getattr(getattr(model.model.layers[block], submodule), projection)
                        x = module.input
                        update = (x @ a.to(x.device, x.dtype).T) @ b.to(x.device, x.dtype).T
                        module.output = module.output + update * scale
                    if block == layer:
                        output = model.model.layers[layer].output
                        if mode == "absolute":
                            model.model.layers[layer].output = output + shift.to(output.device, output.dtype)
                        else:
                            v = vector.to(output.device, dtype=output.dtype)
                            v_normed = v / (v.norm() + 1e-8)
                            norm = torch.norm(output, dim=-1, keepdim=True)
                            model.model.layers[layer].output = output + coef * norm * v_normed
        with tracer.invoke():
            result = tracer.result.save()
    sequences = getattr(result, "sequences", result)
    return sequences[:, tokens.shape[1]:].tolist()


def steered_generate_batches(
    model,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    layer: int,
    vector: torch.Tensor | np.ndarray,
    max_new_tokens: int,
    coef: float = 1.0,
    mode: str = "absolute",
    remote: bool = True,
    lora=None,
) -> list[list[list[int]]]:
    """`steered_generate` for several ``(tokens, attention_mask)`` batches in one NDIF session.

    A session is one request, so an adapter is sent once rather than once per batch (about 395 MiB
    for Gender-Secret, minutes on this machine's uplink), and the batches then run one after another
    on the server, each small enough for the GPUs. Returns each batch's rows' new token ids. The
    traced code repeats `steered_generate`'s because it runs on the server, where no helper from
    this package exists.
    """
    from deception_detection.lora_intervention import call_plan

    if mode not in ("absolute", "norm_scaled"):
        raise ValueError(f"unknown steering mode {mode!r}")
    vector = torch.as_tensor(vector, dtype=torch.float32)
    shift = coef * vector
    plan = call_plan(lora) if lora is not None else {}
    scale = lora.scale if lora is not None else 1.0
    blocks = range(len(model.model.layers)) if plan else (layer,)
    with model.session(remote=remote):
        # made and saved inside the session: a list from outside is copied to the server and its
        # appends never come back (a remote run returned none, 2026-09-30)
        results = nnsight.save([])
        for tokens, attention_mask in batches:
            with model.generate(max_new_tokens=max_new_tokens, do_sample=False) as tracer:
                with tracer.invoke(tokens, attention_mask=attention_mask):
                    for _ in tracer.iter[:]:
                        for block in blocks:
                            for submodule, projection, a, b in plan.get(block, []):
                                module = getattr(getattr(model.model.layers[block], submodule), projection)
                                x = module.input
                                update = (x @ a.to(x.device, x.dtype).T) @ b.to(x.device, x.dtype).T
                                module.output = module.output + update * scale
                            if block == layer:
                                output = model.model.layers[layer].output
                                if mode == "absolute":
                                    model.model.layers[layer].output = output + shift.to(output.device, output.dtype)
                                else:
                                    v = vector.to(output.device, dtype=output.dtype)
                                    v_normed = v / (v.norm() + 1e-8)
                                    norm = torch.norm(output, dim=-1, keepdim=True)
                                    model.model.layers[layer].output = output + coef * norm * v_normed
                with tracer.invoke():
                    results.append(tracer.result.save())
    return [getattr(result, "sequences", result)[:, tokens.shape[1]:].tolist()
            for result, (tokens, _) in zip(results, batches)]


def token_norms(model, tokens: torch.Tensor, attention_mask: torch.Tensor, layer: int,
                remote: bool = True) -> torch.Tensor:
    """Each position's residual norm after block `layer`, ``[batch, seq]`` float32: the scale a
    norm-scaled coefficient multiplies."""
    with model.trace(tokens, attention_mask=attention_mask, remote=remote):
        norms = model.model.layers[layer].output.float().norm(dim=-1).save()
    return torch.as_tensor(norms).cpu()
