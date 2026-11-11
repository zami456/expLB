"""Persona directions re-derived inside the Gender-Secret fine-tune, from the bank's own rollouts.

The persona directions scored on Gender-Secret (``gender_secret.py``) come from the base model's
role-play, while GS's lies are told by a LoRA fine-tune of it. This asks whether that mismatch is
why every direction is at chance there. It reads the same role-play texts, teacher-forced,
through both the base model and the fine-tune. The contrasts built from each read are compared
as directions and as detectors on GS.

- **Rollouts.** The collaborator's kd run, which built the bank: for every persona, its judge
  score 3 rollouts number exactly the bank's ``n``. Set ``KD_RUN_DIR`` to its location.
- **Pooling.** The mean over the final assistant message's content tokens, then over rollouts,
  as the bank was built. ``assistant_span`` ports the Assistant Axis library's span rule.
  ``validate`` rebuilds one persona through the base model and compares it with the bank.
- **Subsample.** The adapter makes every request 222 MiB, so the script doesn't read all ~2,400
  kept rollouts of the 25 contrast personas. Each persona contributes one kept rollout per
  question, at most 30, drawn at random. The same rollouts go through both models.

    python experiments/persona/persona_finetuned.py validate
    python experiments/persona/persona_finetuned.py extract
    python experiments/persona/persona_finetuned.py score
"""

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import fire
import numpy as np
import torch

from deception_detection import persona_pcs as pcs
from deception_detection.activations import trace_layer_outputs
from deception_detection.log import logger
from deception_detection.lora_intervention import load_adapter
from experiments.organisms.gender_secret import ADAPTER, LAYERS, _environment, _model
from deception_detection.utils import REPO_ROOT

RUN_DIR = Path(os.environ.get("KD_RUN_DIR", "C:/Users/zamig/Downloads/kd/run-kd"))
OUT_DIR = REPO_ROOT / "activations" / "persona_finetuned"
GS_DIR = REPO_ROOT / "activations" / "gender_secret"
RESULTS_DIR = REPO_ROOT / "results" / "gender_secret"
MAX_LENGTH = 2048  # the bank pipeline's limit, where longer rows keep their tail; none reach it

# (deceptive side, honest side, the bank's name for the axis)
CONTRASTS = {
    "published (8v7)": (pcs.DECEPTION_PERSONAS, pcs.HONESTY_PERSONAS, "deception"),
    "knowledge-denial (4v6)": (
        pcs.KNOWLEDGE_DENIAL_PERSONAS, pcs.KNOWLEDGE_CANDOUR_PERSONAS, "denial"
    ),
    "extended (12v13)": (
        pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS,
        pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS,
        "extended",
    ),
}
PERSONAS = tuple(sorted({name for pos, neg, _ in CONTRASTS.values() for name in pos + neg}))
ADDED = "added by the fine-tune"  # fine-tuned-rollout axis minus base-rollout axis
OFFSET = "fine-tune's offset on GS (no labels)"


def kept_rollouts(persona: str, run_dir: Path = RUN_DIR) -> list[dict]:
    """The rollouts the bank averaged for ``persona``: judge score 3, in file order."""
    scores = json.loads((run_dir / "scores" / f"{persona}.json").read_text(encoding="utf-8"))
    kept = []
    with open(run_dir / "responses" / f"{persona}.jsonl", encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            if scores.get(f"pos_p{row['prompt_index']}_q{row['question_index']}") == 3:
                kept.append(row)
    return kept


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def _find(haystack: list[int], needle: list[int]) -> int:
    for i in range(len(haystack) - len(needle) + 1):
        if needle and haystack[i : i + len(needle)] == needle:
            return i
    return -1


def assistant_span(tokenizer, conversation: list[dict], max_length: int = MAX_LENGTH):
    """The conversation's token ids and the ``[start, end)`` of its final assistant message's content.

    A port of the rule that built the bank: the standard (non-Qwen) path of
    ``ConversationEncoder.build_turn_spans`` in the Assistant Axis library
    (safety-research/assistant-axis at a989619), as the collaborator's
    ``2_activations_ndif.build_items`` applied it. The content is looked for in the tokens the
    message adds to the conversation, so the role header and the closing ``<|eot_id|>`` are left
    out. Rows longer than ``max_length`` keep their tail. The repo has no licence, so this is a
    port, not a copy. On every kept rollout of the 25 contrast personas it gives the same spans
    as the original.
    """
    def template(messages: list[dict]) -> list[int]:
        return list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))

    *before, last = conversation
    if last["role"] != "assistant":
        raise ValueError("the conversation must end with the assistant's message")
    full = template(conversation)
    prefix = _common_prefix_len(full, template(before + [{"role": "assistant", "content": ""}]))
    added = full[prefix:]
    special = set(tokenizer.all_special_ids)
    while added and added[-1] in special:
        added = added[:-1]
    content = last["content"]
    start, length = -1, 0
    for candidate in (content, " " + content):
        ids = tokenizer(candidate, add_special_tokens=False).input_ids
        start, length = _find(added, ids), len(ids)
        if start != -1:
            break
    if start == -1:  # the library's fallback: everything the message added
        start, length = 0, len(added)
    a0 = prefix + start
    a1 = a0 + length
    if len(full) > max_length:
        cut = len(full) - max_length
        full, a0, a1 = full[cut:], a0 - cut, a1 - cut
    a0, a1 = max(a0, 0), min(a1, len(full))
    if a0 >= a1:
        raise ValueError("the assistant span is empty")
    return full, a0, a1


def subsample(personas=PERSONAS, per_question: int = 1, seed: int = 0, run_dir: Path = RUN_DIR):
    """``per_question`` kept rollouts per question for each persona, prompt variant at random.

    Holding the questions fixed across personas keeps question topic from leaking into a
    contrast, which a plain random draw of each persona's kept set would allow.
    """
    rng = np.random.default_rng(seed)
    chosen = []
    for persona in personas:
        by_question = defaultdict(list)
        for row in kept_rollouts(persona, run_dir):
            by_question[row["question_index"]].append(row)
        for question in sorted(by_question):
            rows = by_question[question]
            for i in sorted(rng.choice(len(rows), size=min(per_question, len(rows)), replace=False)):
                chosen.append((persona, rows[i]))
    return chosen


def _meta(chosen, layers) -> dict[str, np.ndarray]:
    return {
        "persona": np.array([persona for persona, _ in chosen]),
        "prompt_index": np.array([row["prompt_index"] for _, row in chosen]),
        "question_index": np.array([row["question_index"] for _, row in chosen]),
        "layers": np.array(list(layers)),
    }


def batches(lengths: list[int], token_budget: int, max_rows: int) -> list[list[int]]:
    """Row indices, longest first, in groups whose padded size stays within ``token_budget``."""
    groups, current = [], []
    for i in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
        # sorted longest first, so a group's first row sets its width
        if current and ((len(current) + 1) * lengths[current[0]] > token_budget or len(current) >= max_rows):
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    return groups


def _read(model, items, layers, lora, path: Path, meta: dict, token_budget: int, max_rows: int,
          retries: int = 3) -> np.ndarray:
    """Each item's span mean at each layer, ``[n, layers, hidden]`` float32.

    Saved to ``path`` after every request, so an interrupted read resumes where it stopped.
    """
    from deception_detection.models import hidden_size

    n, hidden = len(items), hidden_size(model)
    pooled = np.zeros((n, len(layers), hidden), dtype=np.float32)
    done = np.zeros(n, dtype=bool)
    if path.exists():
        # closed before the first save: Windows will not replace a file that is still open
        with np.load(path, allow_pickle=False) as saved:
            if not all(np.array_equal(saved[key], value) for key, value in meta.items()):
                raise RuntimeError(f"{path} holds a different selection; move it away to start over")
            pooled, done = saved["pooled"].copy(), saved["done"].copy()
        logger.info(f"{path.name}: resuming with {int(done.sum())}/{n} done")
    pad = model.tokenizer.pad_token_id
    lengths = [len(ids) for ids, _, _ in items]
    groups = batches(lengths, token_budget, max_rows)
    for k, group in enumerate(groups):
        if done[group].all():
            continue
        width = lengths[group[0]]
        tokens = torch.full((len(group), width), pad, dtype=torch.long)
        attention = torch.zeros((len(group), width), dtype=torch.long)
        span = torch.zeros((len(group), width), dtype=torch.float32)
        for row, i in enumerate(group):  # right padding, as the bank was read
            ids, a0, a1 = items[i]
            tokens[row, : len(ids)] = torch.tensor(ids)
            attention[row, : len(ids)] = 1
            span[row, a0:a1] = 1.0
        start = time.time()
        for attempt in range(1, retries + 1):
            try:
                means = trace_layer_outputs(model, tokens, attention, list(layers), lora=lora,
                                            span_mask=span)
                break
            except Exception as error:  # NDIF drops a request now and then
                if attempt == retries:
                    raise
                logger.warning(f"request {k + 1} failed ({error!r}); retry {attempt}/{retries - 1}")
                time.sleep(30 * attempt)
        pooled[group] = means.float().numpy()
        done[group] = True
        partial = path.with_suffix(".partial")
        with open(partial, "wb") as file:
            np.savez(file, pooled=pooled, done=done, **meta)
        os.replace(partial, path)
        logger.info(f"  {path.stem} request {k + 1}/{len(groups)}: {len(group)} rollouts x {width} "
                    f"tokens, {time.time() - start:.0f} s; {int(done.sum())}/{n} done")
    return pooled


def validate(persona: str = "pc_coverup_operator", layers=LAYERS, token_budget: int = 16000,
             max_rows: int = 48, run_dir: str | Path = RUN_DIR) -> None:
    """Rebuild one persona's bank vector through the base model before touching the fine-tune.

    This checks the span rule, the padding and the pooling in one go. The default persona has
    the fewest kept rollouts (47). For scale, the error is also given as the distance to the
    nearest *other* persona in the bank.
    """
    _environment()
    model = _model()
    rows = kept_rollouts(persona, Path(run_dir))
    chosen = [(persona, row) for row in rows]
    items = [assistant_span(model.tokenizer, row["conversation"]) for row in rows]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled = _read(model, items, layers, None, OUT_DIR / f"validate_{persona}.npz",
                   _meta(chosen, layers), token_budget, max_rows)
    rebuilt = pooled.astype(np.float64).mean(0)
    for j, layer in enumerate(layers):
        bank = pcs.load_bank_at_layer(layer, pcs.EXTENDED_VECTORS_DIR)
        own = bank.vectors_for((persona,))[0]
        scale = np.linalg.norm(own)
        error = np.linalg.norm(rebuilt[j] - own) / scale
        nearest = min(np.linalg.norm(rebuilt[j] - vector) / scale
                      for name, vector in zip(bank.names, bank.personas) if name != persona)
        cosine = own @ rebuilt[j] / (scale * np.linalg.norm(rebuilt[j]))
        logger.info(f"layer {layer}: {len(rows)} rollouts (bank n = "
                    f"{bank.n_rollouts[bank.names.index(persona)]}); relative error {error:.2e}, "
                    f"cosine {cosine:.6f}; the nearest other persona is {nearest:.2e} away")


def extract(reads=("base", "adapted"), layers=LAYERS, per_question: int = 1, seed: int = 0,
            token_budget: int = 16000, max_rows: int = 48, run_dir: str | Path = RUN_DIR,
            out_dir: str | Path = OUT_DIR) -> None:
    """Read the subsampled rollouts through the base model and through the fine-tune.

    Both reads use the same rollouts in the same batches, so they differ only in the adapter.
    Writes ``out_dir/{read}.npz`` (``pooled`` ``[n, layers, hidden]``, plus which rollout each
    row is).
    """
    if isinstance(reads, str):
        reads = tuple(reads.split(","))
    _environment()
    model = _model()
    chosen = subsample(PERSONAS, per_question, seed, Path(run_dir))
    counts = Counter(persona for persona, _ in chosen)
    logger.info(f"{len(chosen)} rollouts from {len(counts)} personas, "
                f"{min(counts.values())}-{max(counts.values())} each")
    items = [assistant_span(model.tokenizer, row["conversation"]) for _, row in chosen]
    meta = _meta(chosen, layers)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = load_adapter(ADAPTER, max_block=max(layers)) if "adapted" in reads else None
    for read in reads:
        lora = adapter if read == "adapted" else None
        _read(model, items, layers, lora, out_dir / f"{read}.npz", meta, token_budget, max_rows)
        logger.info(f"{read}: done")


def _axis(vectors: dict[str, np.ndarray], positive, negative) -> np.ndarray:
    """Mean of the deceptive side minus mean of the honest side, as the bank's axes are built."""
    return (np.stack([vectors[n] for n in positive]).mean(0)
            - np.stack([vectors[n] for n in negative]).mean(0))


def score(in_dir: str | Path = OUT_DIR, gs_dir: str | Path = GS_DIR,
          out_dir: str | Path = RESULTS_DIR, n_boot: int = 1000, seed: int = 42) -> None:
    """Compare the base-read and fine-tuned-read contrasts, as directions and on Gender-Secret.

    - ``finetuned_personas_geometry.csv``: per layer and contrast, the cosines between the
      bank's direction, the base read's and the fine-tuned read's.
    - ``finetuned_shift.csv``: per layer, how far the fine-tune moves an activation, relative to
      its norm: the persona vectors, and GS's lies and honest answers for comparison.
    - ``finetuned_personas_auroc.csv``: GS zero-shot AUROC for each set of directions, on the
      fine-tuned GS read (``gs_read = adapted``) and the base read of the same tokens.
      ``auroc_lo``/``hi`` are 95% stratified bootstraps over GS examples, with resamples shared
      across directions and reads. ``joint_lo``/``hi``, for the rollout directions, also
      resample each persona's rollouts: they are the intervals to quote, because the
      directions come from a subsample. Two diagnostics are scored the same way: the component
      the fine-tune adds to each contrast (``added by the fine-tune``), and the offset the adapter
      adds to GS's own activations, averaged over both classes without labels.
    """
    import pandas as pd

    from experiments.persona import persona_ood as P

    in_dir, gs_dir, out_dir = Path(in_dir), Path(gs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # dict() loads each array once; indexing an NpzFile re-reads the array from disk every time
    reads = {read: dict(np.load(in_dir / f"{read}.npz", allow_pickle=False))
             for read in ("base", "adapted")}
    for key in ("persona", "prompt_index", "question_index", "layers"):
        assert np.array_equal(reads["base"][key], reads["adapted"][key]), f"reads differ in {key}"
    assert all(data["done"].all() for data in reads.values()), "a read is incomplete"
    names = reads["base"]["persona"].astype(str)
    layers = [int(layer) for layer in reads["base"]["layers"]]
    counts = Counter(names)
    logger.info(f"{len(names)} rollouts, {min(counts.values())}-{max(counts.values())} per persona")

    def unit(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v)

    def relative_shift(adapted: np.ndarray, base: np.ndarray) -> np.ndarray:
        return np.linalg.norm(adapted - base, axis=-1) / np.linalg.norm(base, axis=-1)

    def common_share(adapted: np.ndarray, base: np.ndarray) -> float:
        """How much of the shift is one vector added to every row, which no projection's AUROC
        can see: ``|mean shift|^2 / mean |shift|^2``."""
        shift = adapted - base
        energy = np.mean(np.sum(shift**2, axis=1))
        return float(np.sum(shift.mean(0) ** 2) / energy) if energy > 0 else float("nan")

    rng = np.random.default_rng(seed)
    geometry, shifts, rows = [], [], []
    for j, layer in enumerate(layers):
        bank = pcs.load_bank_at_layer(layer, pcs.EXTENDED_VECTORS_DIR)
        bank_vectors = dict(zip(bank.names, bank.personas))
        means = {read: {name: data["pooled"][names == name, j].astype(np.float64).mean(0)
                        for name in PERSONAS}
                 for read, data in reads.items()}
        gs = {read: dict(np.load(gs_dir / f"{read}_L{layer}.npz", allow_pickle=False))
              for read in ("adapted", "base")}
        assert np.array_equal(gs["adapted"]["index"], gs["base"]["index"]), "GS reads differ"
        y = (gs["adapted"]["labels"].astype(str) == "DECEPTIVE").astype(int)

        pairs = {
            "persona vectors": (np.stack([means["adapted"][n] for n in PERSONAS]),
                                np.stack([means["base"][n] for n in PERSONAS])),
            "persona rollouts": (reads["adapted"]["pooled"][:, j].astype(np.float64),
                                 reads["base"]["pooled"][:, j].astype(np.float64)),
        }
        gs_adapted, gs_base = (gs[read]["pooled"].astype(np.float64) for read in ("adapted", "base"))
        pairs["GS lies"] = (gs_adapted[y == 1], gs_base[y == 1])
        pairs["GS honest answers"] = (gs_adapted[y == 0], gs_base[y == 0])
        offsets = {}
        for what, (after, before) in pairs.items():
            values = relative_shift(after, before)
            offsets[what] = (after - before).mean(0)
            shifts.append({"layer": layer, "activations": what, "n": len(values),
                           "median": float(np.median(values)), "max": float(values.max()),
                           "common_share": common_share(after, before)})
        # is it the same offset everywhere? cosine of each group's mean shift with the rollouts'
        for row in shifts[-len(pairs):]:
            a, b = offsets[row["activations"]], offsets["persona rollouts"]
            scale = np.linalg.norm(a) * np.linalg.norm(b)
            row["cos_offset_with_rollouts"] = float(a @ b / scale) if scale > 0 else float("nan")

        directions = {}
        for contrast, (positive, negative, bank_axis) in CONTRASTS.items():
            # the axis rule here must be the bank's own
            assert np.allclose(_axis(bank_vectors, positive, negative), bank.axes[bank_axis])
            raw = {read: _axis(means[read], positive, negative) for read in means}
            d = {"bank": unit(bank.axes[bank_axis]),
                 "base rollouts": unit(raw["base"]),
                 "fine-tuned rollouts": unit(raw["adapted"]),
                 # what the fine-tune adds to the contrast, as a detector on its own
                 ADDED: unit(raw["adapted"] - raw["base"])}
            directions[contrast] = d
            geometry.append({
                "layer": layer, "contrast": contrast,
                "cos_bank_base": d["bank"] @ d["base rollouts"],
                "cos_bank_finetuned": d["bank"] @ d["fine-tuned rollouts"],
                "cos_base_finetuned": d["base rollouts"] @ d["fine-tuned rollouts"],
                "axis_change": np.linalg.norm(raw["adapted"] - raw["base"]) / np.linalg.norm(raw["base"]),
            })

        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        draws = [np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg))))
                 for _ in range(n_boot)]
        gs_z = {gs_read: data["pooled"].astype(np.float64) for gs_read, data in gs.items()}

        # The rollout directions are estimates from a subsample, so their intervals also
        # resample each persona's rollouts, alongside the GS examples. Each draw is shared by
        # both reads, so the delta stays paired.
        by_persona = {name: np.flatnonzero(names == name) for name in PERSONAS}
        pooled_j = {read: data["pooled"][:, j].astype(np.float64) for read, data in reads.items()}
        joint = defaultdict(list)
        for i in draws:
            resampled = {name: rng.choice(idx, len(idx)) for name, idx in by_persona.items()}
            axes = {}
            for source, read in (("base rollouts", "base"), ("fine-tuned rollouts", "adapted")):
                vectors = {name: pooled_j[read][r].mean(0) for name, r in resampled.items()}
                for contrast, (positive, negative, _) in CONTRASTS.items():
                    axes[contrast, source] = _axis(vectors, positive, negative)
            for contrast in CONTRASTS:
                axes[contrast, ADDED] = (axes[contrast, "fine-tuned rollouts"]
                                         - axes[contrast, "base rollouts"])
            for (contrast, source), axis in axes.items():
                u = unit(axis)
                for gs_read, z in gs_z.items():
                    joint[gs_read, contrast, source].append(P.auroc(y[i], (z @ u)[i]))

        def interval(values) -> dict[str, float]:
            values = np.asarray(values)
            return {"joint_lo": float(np.quantile(values, 0.025)),
                    "joint_hi": float(np.quantile(values, 0.975))}

        for gs_read, z in gs_z.items():
            for contrast, d in directions.items():
                scores = {source: z @ u for source, u in d.items()}
                boots = {source: np.array([P.auroc(y[i], s[i]) for i in draws])
                         for source, s in scores.items()}
                common = {"layer": layer, "gs_read": gs_read, "contrast": contrast,
                          "n": len(y), "n_deceptive": int(y.sum())}
                for source, s in scores.items():
                    rows.append({**common, "directions": source, "auroc": P.auroc(y, s),
                                 "auroc_lo": float(np.quantile(boots[source], 0.025)),
                                 "auroc_hi": float(np.quantile(boots[source], 0.975)),
                                 **(interval(joint[gs_read, contrast, source])
                                    if source != "bank" else {})})
                delta = boots["fine-tuned rollouts"] - boots["base rollouts"]
                joint_delta = (np.array(joint[gs_read, contrast, "fine-tuned rollouts"])
                               - np.array(joint[gs_read, contrast, "base rollouts"]))
                rows.append({**common, "directions": "fine-tuned - base rollouts",
                             "auroc": P.auroc(y, scores["fine-tuned rollouts"])
                             - P.auroc(y, scores["base rollouts"]),
                             "auroc_lo": float(np.quantile(delta, 0.025)),
                             "auroc_hi": float(np.quantile(delta, 0.975)),
                             **interval(joint_delta)})

        # The offset the adapter adds to GS's activations, averaged over both classes (no labels),
        # as a detector. Each draw recomputes it from the resampled examples.
        offset = gs_z["adapted"] - gs_z["base"]
        for gs_read, z in gs_z.items():
            boots = np.array([P.auroc(y[i], (z @ unit(np.bincount(i, minlength=len(y)) @ offset))[i])
                              for i in draws])
            rows.append({"layer": layer, "gs_read": gs_read, "contrast": "none", "n": len(y),
                         "n_deceptive": int(y.sum()), "directions": OFFSET,
                         "auroc": P.auroc(y, z @ unit(offset.mean(0))),
                         "auroc_lo": float(np.quantile(boots, 0.025)),
                         "auroc_hi": float(np.quantile(boots, 0.975))})

    geometry = pd.DataFrame(geometry)
    geometry.to_csv(out_dir / "finetuned_personas_geometry.csv", index=False)
    shifts = pd.DataFrame(shifts)
    shifts.to_csv(out_dir / "finetuned_shift.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "finetuned_personas_auroc.csv", index=False)
    logger.info(f"directions:\n{geometry.round(4).to_string(index=False)}")
    logger.info(f"how far the fine-tune moves activations (relative to their norm):\n"
                f"{shifts.round(4).to_string(index=False)}")
    wide = table[table.gs_read == "adapted"].pivot_table(
        index=["contrast", "directions"], columns="layer", values="auroc")
    logger.info(f"GS zero-shot AUROC on the fine-tuned read:\n{wide.round(3).to_string()}")


if __name__ == "__main__":
    fire.Fire({"validate": validate, "extract": extract, "score": score})
