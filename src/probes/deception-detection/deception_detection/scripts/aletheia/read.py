"""Span means of rows on NDIF: competition rows, K's rows, Alpaca controls. Resumable.

Every row is rendered by `render.py` and read in one pass that stops after the deepest block read;
the server returns, per row and block, the mean over two spans of the final reply (content, and
content plus the end-of-turn token), summed in float32 as the persona bank's `_span_mean_sum` does.
A read through a LoRA adapter adds the adapter's update to each adapted projection in the pass, as
the thesis did (`activations.trace_span_means_session`), with the factors either exact (bfloat16) or
int8 with one scale per row (`soft_trigger.quantise`), dequantised on the server; several passes go
in one NDIF session, so the factors are uploaded once a session. A request the server runs out of
memory on is re-planned, with every later one, at half the token budget.

Reads are saved per request under E:/aletheia/acts/<model>/<read>/<rows>/ (`layer_sweep.save_request`,
the second span's arrays suffixed "_eot"); `load` gathers them. No label is read here.

    python deception_detection/scripts/aletheia/read.py test --dataset Harm --read base
    python deception_detection/scripts/aletheia/read.py smoke --model qwen35
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import fire
import numpy as np
import torch

from deception_detection.log import logger
from deception_detection.scripts.aletheia import data as D
from deception_detection.scripts.aletheia import render as R
from deception_detection.scripts.aletheia import spec as S
from deception_detection.scripts.layer_sweep import done_rows, save_request

SPANS = ("content", "content+eot")
VARIANTS = ("", "_eot")  # save_request's array suffixes for the two spans
BUDGET = {"qwen35": 16000, "gemma": 8000, "nemotron": 8000}  # tokens per pass (Gemma ran at 8,000 on LB)
SESSION_PASSES = 8  # passes to one NDIF session


# --- the remote model ---------------------------------------------------------------------------

def load_model(model: str, repo: str | None = None):
    """The nnsight wrapper for `model`'s base checkpoint, or for `repo` (a merged organism) with the
    base model's tokenizer. Vision-language checkpoints get a stand-in processor holding only the
    tokenizer, as the persona bank's `ndif_common.load_language_model` does; text-only inputs never
    reach the image half."""
    from deception_detection.scripts.gender_secret import _environment

    _environment()  # keys from .env, long upload/download timeouts
    tok = R.tokenizer(model)
    spec = S.MODELS[model]
    if repo is None and spec.vlm:
        from nnsight.modeling.vlm import VisionLanguageModel

        return VisionLanguageModel(spec.repo, processor=SimpleNamespace(tokenizer=tok), tokenizer=tok)
    from nnsight import LanguageModel

    return LanguageModel(repo or spec.repo, tokenizer=tok)


def raise_timeouts(seconds: float = 3600.0) -> None:
    """An adapter's factors are hundreds of MB and the uplink runs at 2-4 Mbit/s."""
    from nnsight.intervention.backends.remote import RemoteBackend

    RemoteBackend.CONNECT_TIMEOUT = max(RemoteBackend.CONNECT_TIMEOUT, seconds)  # also the write timeout
    RemoteBackend.READ_TIMEOUT = max(RemoteBackend.READ_TIMEOUT, seconds)


def session_means(model, batches: list[tuple], layers: list[int], plan: dict | None = None,
                  scale: float = 1.0, remote: bool | str = True) -> list[torch.Tensor]:
    """For each (tokens, attention_mask, span_mask [batch, spans, seq]) of `batches`, the span means
    at `layers`, [batch, spans, layers, hidden], all passes in one NDIF session.

    `plan` is `lora_intervention.call_plan` (entries (submodule, projection, A, B)) or
    `soft_trigger.quantised_plan` (entries (submodule, projection, A_q, A_s, B_q, B_s)); its update
    is added at every adapted projection up to the deepest block read. The body runs on the server, so
    it uses builtins and tensors only (`activations.trace_span_means_session`, with int8 factors)."""
    import nnsight

    from deception_detection.activations import decoder_blocks

    plan = plan or {}
    order = sorted(set(layers))
    index = [order.index(layer) for layer in layers]
    blocks = decoder_blocks(model)  # resolved here, outside the trace
    if remote == "local":
        from nnsight.intervention.backends.local_simulation import LocalSimulationBackend

        session = model.session(backend=LocalSimulationBackend(model))
    else:
        session = model.session(remote=remote)
    with session:
        results = nnsight.save(list())
        for tokens, attention_mask, span_mask in batches:
            n_rows = tokens.shape[0]
            with model.trace(tokens, attention_mask=attention_mask) as tracer:
                outputs = list()
                for block in range(order[-1] + 1):
                    for entry in plan.get(block, []):
                        module = getattr(getattr(blocks[block], entry[0]), entry[1])
                        x = module.input
                        if len(entry) == 4:
                            a = entry[2].to(x.device, x.dtype)
                            b = entry[3].to(x.device, x.dtype)
                        else:
                            a = entry[2].to(x.device, x.dtype) * entry[3].to(x.device, x.dtype)
                            b = entry[4].to(x.device, x.dtype) * entry[5].to(x.device, x.dtype)
                        module.output = module.output + ((x @ a.T) @ b.T) * scale
                    if block in order:
                        output = blocks[block].output
                        output = output[0] if isinstance(output, tuple) else output
                        if output.ndim != 3 or output.shape[0] != n_rows:
                            raise ValueError(f"block {block} returned shape {tuple(output.shape)}")
                        m = span_mask.to(output.device, torch.float32)
                        means = []
                        for k in range(m.shape[1]):
                            mk = m[:, k, :]
                            total = (output * mk[:, :, None].to(output.dtype)).sum(dim=1, dtype=torch.float32)
                            means.append(total / mk.sum(dim=1, keepdim=True))
                        outputs.append(torch.stack(means, dim=1).cpu())  # blocks sit on different GPUs
                results.append(torch.stack(outputs, dim=2))
                tracer.stop()
    return [r[:, :, index] for r in results]


# --- batching and the resumable loop ------------------------------------------------------------

def tensors(tok, items: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-padded tokens, attention mask and the [batch, 2, seq] span masks of rendered rows."""
    width = max(len(it["ids"]) for it in items)
    tokens = torch.full((len(items), width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    spans = torch.zeros((len(items), len(SPANS), width), dtype=torch.float32)
    for i, it in enumerate(items):
        off = width - len(it["ids"])
        tokens[i, off:] = torch.tensor(it["ids"], dtype=torch.long)
        mask[i, off:] = 1
        for k, name in enumerate(SPANS):
            a, b = it[name]
            spans[i, k, off + a:off + b] = 1.0
    return tokens, mask, spans


def plan_requests(lengths: dict[int, int], budget: int) -> list[list[int]]:
    """Row ids grouped shortest first, each group's rows x longest length within `budget` (a row
    longer than the budget goes alone)."""
    out: list[list[int]] = []
    for row in sorted(lengths, key=lambda r: (lengths[r], r)):
        if out and (len(out[-1]) + 1) * lengths[row] <= budget:
            out[-1].append(row)
        else:
            out.append([row])
    return out


def read_rows(model_obj, tok, rendered: dict[int, dict], layers: list[int], folder: Path, budget: int,
              plan: dict | None = None, scale: float = 1.0, session_passes: int = SESSION_PASSES,
              retries: int = 4) -> None:
    from deception_detection.batching import classify_failure

    done = done_rows(folder)
    todo = {r: it for r, it in rendered.items() if r not in done}
    if not todo:
        logger.info(f"{folder}: all {len(rendered)} rows read")
        return
    requests = plan_requests({r: len(it["ids"]) for r, it in todo.items()}, budget)
    logger.info(f"{folder}: {len(todo)} rows in {len(requests)} passes, up to {session_passes} to a session")
    k = 0
    while k < len(requests):
        unit = requests[k:k + session_passes]
        start = time.time()
        means = None
        for attempt in range(1, retries + 1):
            try:
                batches = [tensors(tok, [todo[r] for r in batch]) for batch in unit]
                means = session_means(model_obj, batches, layers, plan, scale)
                break
            except Exception as error:  # noqa: BLE001 -- NDIF drops a request now and then
                kind = classify_failure(error)
                if kind == "fatal":
                    raise
                if kind in ("oom", "unknown") and any(len(b) > 1 for b in unit):
                    budget //= 2
                    rest = {r: todo[r] for b in requests[k:] for r in b}
                    requests = requests[:k] + plan_requests({r: len(it["ids"]) for r, it in rest.items()}, budget)
                    logger.warning(f"passes {k + 1}-{k + len(unit)} ran the server out of memory ({error!r:.200}); "
                                   f"{len(rest)} rows re-planned at {budget} tokens")
                    break
                if kind in ("oom", "unknown") and len(unit) > 1:
                    session_passes = max(1, session_passes // 2)
                    unit = requests[k:k + session_passes]
                    logger.warning(f"single-row passes failed ({error!r:.200}); {session_passes} passes a session")
                    continue
                if attempt == retries:
                    raise
                logger.warning(f"passes {k + 1}-{k + len(unit)} failed ({error!r:.300}); retry {attempt}")
                time.sleep(30 * attempt)
        if means is None:
            continue
        for batch, m in zip(unit, means):
            m = m.float().numpy()
            if not np.isfinite(m).all():
                raise RuntimeError(f"rows {batch[:3]}...: non-finite reads")
            save_request(folder, batch, layers, m, VARIANTS)
        logger.info(f"{folder.name}: passes {k + 1}-{k + len(unit)}/{len(requests)}, "
                    f"{sum(len(b) for b in unit)} rows, {time.time() - start:.0f} s")
        k += len(unit)


def load(folder: Path, layers: list[int] | None = None) -> dict[int, np.ndarray]:
    """row -> [spans, layers, hidden] float32, from the saved requests."""
    out: dict[int, np.ndarray] = {}
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as saved:
            names = sorted({int(k[1:].split("_")[0]) for k in saved.files if k.startswith("L")})
            use = layers or names
            arr = np.stack([np.stack([saved[f"L{b}{v}"] for b in use], axis=1) for v in VARIANTS], axis=1)
            for j, row in enumerate(saved["rows"]):
                out[int(row)] = arr[j].astype(np.float32)
    return out


def folder_for(model: str, read_name: str, rows_name: str, render: str) -> Path:
    return S.ACTS / model / read_name / f"{rows_name}__{render}"


# --- what is read ---------------------------------------------------------------------------------

def _render_all(rows: list[dict], model: str, render: str) -> dict[int, dict]:
    out = {}
    for r in rows:
        out[int(r["index"])] = R.render(r, model, render)
    return out


def renders_for(rows: list[dict]) -> tuple[str, ...]:
    """Both renders when any row carries reasoning in its final turn; else the two are the same."""
    has = any(r["messages"][-1].get("reasoning") for r in rows)
    return ("no-reasoning", "reasoning") if has else ("no-reasoning",)


def test(dataset: str, read: str = "base", layers=None, budget: int | None = None,
         session_passes: int = SESSION_PASSES, renders=None) -> None:
    """One test dataset, read by the base model (`read="base"`) or by its organism (`"organism"`:
    the merged copy where there is one). Adapter reads through the trace go through `organism.py`."""
    d = S.BY_KEY[dataset]
    if read == "organism" and d.lora and not d.merged:
        raise ValueError(f"{dataset}'s organism has no merged copy; read it through `organism.py`")
    repo = d.merged if read == "organism" else None
    read_name = "base" if repo is None else "merged-" + repo.split("/")[-1]
    rows = D.rows(dataset)
    m = S.MODELS[d.model]
    layers = [int(x) for x in (layers or (m.grid if repo is None else m.organism_grid))]
    model_obj = load_model(d.model, repo)
    tok = R.tokenizer(d.model)
    for render in (renders.split(",") if isinstance(renders, str) else renders) or renders_for(rows):
        rendered = _render_all(rows, d.model, render)
        read_rows(model_obj, tok, rendered, layers, folder_for(d.model, read_name, f"test__{dataset}", render),
                  budget or BUDGET[d.model], session_passes=session_passes)


def dev_sample(key: str, per_class: int = S.DEV_PER_CLASS) -> list[dict]:
    """A seeded, balanced sample of one dev set: `per_class` rows a class (the dev labels are public)."""
    rows = D.rows(key, "dev")
    labels = D.labels(key, "dev")
    rng = np.random.default_rng(S.SEED)
    picks = []
    for cls in (True, False):
        ids = sorted(i for i, v in labels.items() if v == cls)
        picks += [int(i) for i in rng.choice(ids, size=min(per_class, len(ids)), replace=False)]
    keep = set(picks)
    return [r for r in rows if int(r["index"]) in keep]


def dev(key: str, read: str = "base", layers=None, per_class: int = S.DEV_PER_CLASS, budget: int | None = None,
        session_passes: int = SESSION_PASSES) -> None:
    """A seeded, balanced sample of one dev set (`per_class` rows a class), read like `test`. The dev
    labels are public, so the sample is balanced on them."""
    model = key.split("__")[1]
    lora = key.split("__")[2]
    rows = dev_sample(key, per_class)
    repo = None
    if read == "organism" and lora != "None":
        merged = {d.lora.split("/")[-1]: d.merged for d in S.DATASETS if d.lora}
        repo = merged.get(lora)
        if repo is None:
            raise ValueError(f"{lora} has no merged copy; read it through `organism.py`")
    read_name = "base" if repo is None else "merged-" + repo.split("/")[-1]
    m = S.MODELS[model]
    layers = [int(x) for x in (layers or (m.grid if repo is None else m.organism_grid))]
    model_obj = load_model(model, repo)
    tok = R.tokenizer(model)
    rendered = _render_all(rows, model, "no-reasoning")
    read_rows(model_obj, tok, rendered, layers, folder_for(model, read_name, f"dev__{key}", "no-reasoning"),
              budget or BUDGET[model], session_passes=session_passes)


def smoke(model: str = "qwen35", repo: str | None = None) -> None:
    """Before any read is trusted: (1) a row read alone and the same row left-padded next to a longer
    one agree (cos >= 0.9998; the linear-attention and Mamba blocks depend on the padding mask), and
    (2) the server's span mean equals the mean of the per-token outputs it returns."""
    from deception_detection.activations import decoder_blocks

    key = f"instructed_deception__{model}__None"
    rows = D.rows(key, "dev")[:3]
    model_obj = load_model(model, repo)
    tok = R.tokenizer(model)
    items = [R.render(r, model) for r in rows]
    order = sorted(range(3), key=lambda i: len(items[i]["ids"]))
    short, long_ = items[order[0]], items[order[-1]]
    layer = S.MODELS[model].mid
    alone = session_means(model_obj, [tensors(tok, [short])], [layer])[0][0]
    padded = session_means(model_obj, [tensors(tok, [short, long_])], [layer])[0][0]
    cos = [float(torch.nn.functional.cosine_similarity(alone[k, 0].double(), padded[k, 0].double(), dim=0))
           for k in range(len(SPANS))]
    blocks = decoder_blocks(model_obj)
    t, m, s = tensors(tok, [short])
    with model_obj.trace(t, attention_mask=m, remote=True) as tracer:
        out = blocks[layer].output
        out = out[0] if isinstance(out, tuple) else out
        full = out.save()
        tracer.stop()
    full = full.float()[0]
    local = full[s[0, 0].bool()].mean(0)
    rel = float((local - alone[0, 0].float()).norm() / local.norm())
    result = {"model": model, "repo": repo, "layer": layer, "padding_cos": cos, "span_mean_rel_err": rel,
              "passed": min(cos) >= 0.9998 and rel < 1e-2}
    out_path = S.RESULTS / "checks" / f"smoke_{model}{'_' + repo.split('/')[-1] if repo else ''}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    logger.info(f"smoke {result}")
    if not result["passed"]:
        raise RuntimeError(f"smoke check failed: {result}")


def parity_gemma(dataset: str = "instruct_dishonesty__plain", picks=(0, 1, 2, 3)) -> None:
    """This environment (transformers 5.8, this reader) against the replication's (4.55, `activations`):
    Liars' Bench rows tokenised as the replication tokenised them, read here, must reproduce the pooled
    vectors stored at block 31 (cos >= 0.9998, `forced_denial_bank.MIN_COSINE`)."""
    from deception_detection.repository import DatasetRepository
    from deception_detection.scripts import lb_models as LB
    from deception_detection.scripts.forced_denial_bank import MIN_COSINE
    from deception_detection.tokenized_data import TokenizedDataset

    spec = LB.spec("gemma")
    block = S.MODELS["gemma"].mid
    data = DatasetRepository().get(dataset, source_model=spec.model)
    tok = R.tokenizer("gemma")
    toks = TokenizedDataset.from_dialogue_list([data.dialogues[j] for j in picks], tok, padding=data.padding)
    mask = toks.detection_mask.float()
    spans = torch.stack([mask, mask], dim=1)
    means = session_means(load_model("gemma"), [(toks.tokens, toks.attention_mask, spans)], [block])[0]
    stored = np.load(spec.pooled_dirs(block)[0] / f"{dataset}.npz")["pooled"]
    cos = []
    for k, j in enumerate(picks):
        x, y = means[k, 0, 0].double().numpy(), stored[j].astype(np.float64)
        cos.append(float(x @ y / np.linalg.norm(x) / np.linalg.norm(y)))
    result = {"dataset": dataset, "block": block, "rows": list(picks), "cosine": cos, "passed": min(cos) >= MIN_COSINE}
    path = S.RESULTS / "checks" / "parity_gemma.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    logger.info(f"parity {result}")
    if not result["passed"]:
        raise RuntimeError(f"this environment does not reproduce the replication's reads: {result}")


if __name__ == "__main__":
    fire.Fire({"test": test, "dev": dev, "smoke": smoke, "parity_gemma": parity_gemma})
