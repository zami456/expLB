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

    python experiments/external/aletheia/read.py test --dataset Harm --read base
    python experiments/external/aletheia/read.py smoke --model qwen35
"""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import fire
import numpy as np
import torch

from deception_detection.log import logger
from experiments.external.aletheia import data as D
from experiments.external.aletheia import render as R
from experiments.external.aletheia import spec as S
from experiments.persona.layer_sweep import done_rows, save_request

SPANS = ("content", "content+eot")
VARIANTS = ("", "_eot")  # save_request's array suffixes for the two spans
BUDGET = {"qwen35": 16000, "gemma": 4000, "nemotron": 8000}  # tokens per pass (Gemma's organisms ran at 4,000)
SESSION_PASSES = 8  # passes to one NDIF session


# --- the remote model ---------------------------------------------------------------------------

def load_model(model: str, repo: str | None = None):
    """The nnsight wrapper for `model`'s base checkpoint, or for `repo` (a merged organism) with the
    base model's tokenizer. Vision-language checkpoints get a stand-in processor holding only the
    tokenizer, as the persona bank's `ndif_common.load_language_model` does; text-only inputs never
    reach the image half."""
    from experiments.organisms.gender_secret import _environment

    _environment()  # keys from .env, long upload/download timeouts
    harden()
    tok = R.tokenizer(model)
    spec = S.MODELS[model]
    if repo is None and spec.vlm:
        from nnsight.modeling.vlm import VisionLanguageModel

        return VisionLanguageModel(spec.repo, processor=SimpleNamespace(tokenizer=tok), tokenizer=tok)
    from nnsight import LanguageModel

    return LanguageModel(repo or spec.repo, tokenizer=tok)


_HARDENED = []


def harden(min_kbps: float = 40.0, grace: float = 12.0) -> None:
    """The persona bank pipeline's guard against a collapsed result download (`ndif_common.
    install_download_deadline`): a result trickling in below `min_kbps` after `grace` seconds is
    fetched again on a fresh connection instead of hanging the run. 40 KB/s is the floor the
    collaborator used on this laptop."""
    if _HARDENED:
        return
    import importlib.util

    path = S.ROOT / "banks" / "Qwen3.5-27B" / "code" / "src" / "assistant_axis_ndif" / "ndif_common.py"
    spec = importlib.util.spec_from_file_location("ndif_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ok = module.install_download_deadline(min_kbps=min_kbps, grace=grace)
    _HARDENED.append(ok)
    logger.info(f"NDIF download floor {min_kbps:.0f} KB/s after {grace:.0f} s: {'installed' if ok else 'NOT installed'}")


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
              retries: int = 4, remote: bool | str = True) -> None:
    done = done_rows(folder)
    todo = {r: it for r, it in rendered.items() if r not in done}
    if not todo:
        logger.info(f"{folder}: all {len(rendered)} rows read")
        return

    def save(batch: list[int], m: np.ndarray) -> None:
        save_request(folder, batch, layers, m, VARIANTS)

    _read_loop(model_obj, tok, todo, layers, budget, plan, scale, session_passes, retries, save, folder.name, remote,
               owner=lambda r: folder)


def tail_cut(item: dict, keep: int = S.TAIL_CUT_TO, header: int = 8, bos_id: int | None = None) -> dict:
    """The frozen fallback (spec: a row of at least TAIL_CUT_FROM tokens that runs out of memory keeps
    its last TAIL_CUT_TO): the row's tail, never cutting into the final turn (its content and the
    `header` tokens before it), the persona bank's rule for over-long rows. A leading BOS stays."""
    ids = item["ids"]
    lead = 1 if bos_id is not None and ids and ids[0] == bos_id else 0
    n = len(ids)
    cut = min(max(0, n - keep), max(0, item["content"][0] - header - lead))
    if cut == 0:
        return item
    out = dict(item)
    out["ids"] = ids[:lead] + ids[lead + cut:]
    for name in SPANS:
        a, b = item[name]
        out[name] = (a - cut, b - cut)
    out["cut"] = cut
    return out


def _record_unread(folder: Path, row: int, length: int) -> None:
    path = S.RESULTS / "checks" / "unread_rows.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"folder": str(folder), "row": row, "tokens": length}) + "\n")


def _record_cut(folder: Path, row: int, cut: int, length: int, reason: str = "ran out of memory alone") -> None:
    path = S.RESULTS / "checks" / "cut_rows.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"folder": str(folder), "row": row, "tokens cut": cut, "tokens before": length,
                            "reason": reason}) + "\n")

def _cut_rows(folder: Path) -> set[int]:
    path = S.RESULTS / "checks" / "cut_rows.jsonl"
    if not path.exists():
        return set()
    recs = [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    return {int(r["row"]) for r in recs if Path(r["folder"]) == Path(folder)}


def read_sets(model_obj, tok, sets: dict[Path, dict[int, dict]], layers: list[int], budget: int,
              plan: dict | None = None, scale: float = 1.0, session_passes: int = SESSION_PASSES,
              retries: int = 4, remote: bool | str = True) -> None:
    """`read_rows` over several row sets at once, each saved to its own folder: the rows of every set
    are planned together, so a session carries as many passes as it can whatever set they come from.
    An adapter's factors go up once a session, and the uplink is the slow part (165 KB/s measured)."""
    todo: dict[int, tuple[Path, int, dict]] = {}
    for folder, rendered in sets.items():
        done = done_rows(folder)
        already_cut = _cut_rows(folder)
        for r, it in rendered.items():
            if r not in done:
                if r in already_cut:  # the rule already cut it in an earlier run: resume with the cut
                    it = tail_cut(it, keep=S.TAIL_CUT_TO, bos_id=getattr(tok, "bos_token_id", None))
                todo[len(todo)] = (folder, r, it)
    if not todo:
        logger.info(f"all {sum(len(s) for s in sets.values())} rows of {len(sets)} sets read")
        return
    combined = {g: it for g, (_, _, it) in todo.items()}
    folder = Path(f"bundle of {len(sets)} sets")
    logger.info(f"{len(todo)} rows of {len(sets)} sets: {', '.join(p.name for p in sets)}")

    pending = dict(todo)

    def save(batch: list[int], m: np.ndarray) -> None:
        by_folder: dict[Path, list[int]] = {}
        for j, g in enumerate(batch):
            by_folder.setdefault(pending[g][0], []).append(j)
        for f, idx in by_folder.items():
            save_request(f, [pending[batch[j]][1] for j in idx], layers, m[idx], VARIANTS)

    _read_loop(model_obj, tok, combined, layers, budget, plan, scale, session_passes, retries, save, folder.name, remote,
               owner=lambda g: pending[g][0], row_id=lambda g: pending[g][1])


def _read_loop(model_obj, tok, todo: dict[int, dict], layers: list[int], budget: int, plan, scale: float,
               session_passes: int, retries: int, save, label: str, remote: bool | str = True, owner=None,
               row_id=None) -> None:
    from deception_detection.batching import classify_failure

    rid = row_id or (lambda g: g)  # the dataset's row index (a bundle numbers its rows itself)
    # ALETHEIA_CUT_LONGER=0: every row must fail alone itself before it is cut (no cut by length)
    cut_longer = os.environ.get("ALETHEIA_CUT_LONGER", "1") != "0"

    requests = plan_requests({r: len(it["ids"]) for r, it in todo.items()}, budget)
    logger.info(f"{label}: {len(todo)} rows in {len(requests)} passes, up to {session_passes} to a session")
    oom_alone: dict[int, int] = {}
    full_passes = session_passes  # regrown after successes: one bad row must not shrink every later session
    streak = 0
    k = 0
    while k < len(requests):
        unit = requests[k:k + session_passes]
        start = time.time()
        means = None
        for attempt in range(1, retries + 1):
            try:
                batches = [tensors(tok, [todo[r] for r in batch]) for batch in unit]
                means = session_means(model_obj, batches, layers, plan, scale, remote)
                break
            except Exception as error:  # noqa: BLE001 -- NDIF drops a request now and then
                kind = classify_failure(error)
                lines = [ln for ln in str(error).strip().splitlines() if ln.strip()]
                tail = (lines[-1].strip() if lines else repr(error))[:300]
                if not lines and kind == "transient":
                    kind = "unknown"
                logger.warning(f"passes {k + 1}-{k + len(unit)}: {type(error).__name__} [{kind}]: {tail}")
                if kind == "fatal":
                    raise
                if kind in ("oom", "unknown") and any(len(b) > 1 for b in unit):
                    budget //= 2
                    rest = {r: todo[r] for b in requests[k:] for r in b}
                    requests = requests[:k] + plan_requests({r: len(it["ids"]) for r, it in rest.items()}, budget)
                    logger.warning(f"{len(rest)} rows re-planned at {budget} tokens")
                    break
                if kind in ("oom", "unknown") and len(unit) > 1:
                    session_passes = max(1, session_passes // 2)
                    unit = requests[k:k + session_passes]
                    continue
                if kind == "oom" and len(unit) == 1 and len(unit[0]) == 1:
                    r = unit[0][0]
                    oom_alone[r] = oom_alone.get(r, 0) + 1
                    n_now = len(todo[r]["ids"])
                    if oom_alone[r] < 4:
                        # free memory differs between NDIF's replicas: a row alone gets four tries
                        logger.warning(f"row {r} ({n_now} tokens) ran out of memory alone; try {oom_alone[r] + 1} of 4")
                        time.sleep(60 * oom_alone[r])
                        continue
                    if n_now >= S.TAIL_CUT_FROM and "cut" not in todo[r]:
                        # the frozen fallback: a row of at least TAIL_CUT_FROM tokens keeps its last TAIL_CUT_TO
                        cut = tail_cut(todo[r], keep=S.TAIL_CUT_TO, bos_id=getattr(tok, "bos_token_id", None))
                        logger.warning(f"row {r} ({n_now} tokens): cut to its last {len(cut['ids'])} (the frozen rule)")
                        _record_cut(owner(r) if owner else Path(label), rid(r), n_now - len(cut["ids"]), n_now)
                        todo[r] = cut
                        oom_alone[r] = 0
                        # memory grows with length: every longer row still to read would fail too, and each
                        # failed try costs an adapter upload, so those rows take the same cut without their own
                        later = {q for b in requests[k + 1:] for q in b}
                        longer = [q for q in later if cut_longer and len(todo[q]["ids"]) >= n_now and "cut" not in todo[q]]
                        for q in longer:
                            n_q = len(todo[q]["ids"])
                            todo[q] = tail_cut(todo[q], keep=S.TAIL_CUT_TO, bos_id=getattr(tok, "bos_token_id", None))
                            _record_cut(owner(q) if owner else Path(label), rid(q), n_q - len(todo[q]["ids"]), n_q,
                                        reason=f"longer than row {rid(r)} ({n_now} tokens), which ran out of memory alone")
                        if longer:
                            logger.warning(f"{len(longer)} rows of at least {n_now} tokens cut likewise, untried")
                            rest = {q: todo[q] for b in requests[k:] for q in b}
                            requests = requests[:k] + plan_requests({q: len(it["ids"]) for q, it in rest.items()}, budget)
                            unit = requests[k:k + session_passes]
                        continue
                    # outside the frozen rule: the row is left unread, and its cells are reported missing
                    logger.warning(f"row {r} ({n_now} tokens) cannot be read on this deployment; left unread")
                    _record_unread(owner(r) if owner else Path(label), rid(r), n_now)
                    requests = requests[:k] + requests[k + 1:]
                    means = None
                    break
                if attempt == retries:
                    raise
                time.sleep(30 * attempt)
        if means is None:
            continue
        for batch, m in zip(unit, means):
            m = m.detach().float().numpy()
            if not np.isfinite(m).all():
                raise RuntimeError(f"rows {batch[:3]}...: non-finite reads")
            save(batch, m)
        logger.info(f"{label}: passes {k + 1}-{k + len(unit)}/{len(requests)}, {sum(len(b) for b in unit)} rows, "
                    f"{time.time() - start:.0f} s")
        k += len(unit)
        # keep the size that worked, doubling after three successes in a row: near the memory limit every
        # failed session costs an adapter upload, and a jump straight back to full size fails again
        streak = streak + 1 if len(unit) == session_passes else 1
        if streak >= 3:
            session_passes, streak = min(full_passes, session_passes * 2), 0


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


FREEZE = S.ROOT / "logs" / "freeze.txt"


def check_freeze() -> str:
    """The SHA-256 of spec.py, which must equal the one recorded at the freeze; logged by every test
    read, so the logs show the protocol was fixed before the read."""
    import hashlib

    digest = hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest().upper()
    if not FREEZE.exists() or digest not in FREEZE.read_text(encoding="utf-8"):
        raise RuntimeError(f"spec.py ({digest[:12]}) is not the frozen protocol ({FREEZE})")
    logger.info(f"spec.py sha256 {digest} = the freeze ({FREEZE.read_text(encoding='utf-8').strip()})")
    return digest


def test(dataset: str, read: str = "base", layers=None, budget: int | None = None,
         session_passes: int = SESSION_PASSES, renders=None) -> None:
    """One test dataset, read by the base model (`read="base"`) or by its organism (`"organism"`:
    the merged copy where there is one). Adapter reads through the trace go through `organism.py`."""
    check_freeze()
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


def k_inside(dataset: str, budget: int | None = None, session_passes: int = SESSION_PASSES) -> None:
    """K's 240 rows read through an organism's merged copy at the organism grid, so K can be built
    inside it (in-trace adapters: `organism.py read --what k`)."""
    from experiments.external.aletheia import k as K

    d = S.BY_KEY[dataset]
    if not d.merged:
        raise ValueError(f"{dataset} has no merged copy")
    model_obj = load_model(d.model, d.merged)
    read_rows(model_obj, R.tokenizer(d.model), K.rendered(d.model), list(S.MODELS[d.model].organism_grid),
              folder_for(d.model, "merged-" + d.merged.split("/")[-1], "k", "no-reasoning"),
              budget or BUDGET[d.model], session_passes=session_passes)


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


def reread(dataset: str, read: str = "base", render: str = "no-reasoning", picks=(0, 1), min_cos: float = 0.9998) -> None:
    """Spot re-read: two stored rows of a test dataset read again by the same model at its middle
    block must match their stored reads (NDIF can redeploy a model mid-run). Appends to checks/rereads.jsonl."""
    d = S.BY_KEY[dataset]
    repo = d.merged if read == "organism" else None
    read_name = "base" if repo is None else "merged-" + repo.split("/")[-1]
    folder = folder_for(d.model, read_name, f"test__{dataset}", render)
    stored = load(folder)
    rows = [r for r in D.rows(dataset) if int(r["index"]) in stored][:max(picks) + 1]
    rows = [rows[i] for i in picks]
    from experiments.external.aletheia.score import _layer_pos

    block = S.MODELS[d.model].mid
    pos = _layer_pos(folder, block)
    items = [R.render(r, d.model, render) for r in rows]
    again = session_means(load_model(d.model, repo), [tensors(R.tokenizer(d.model), items)], [block])[0]
    cos = []
    for j, r in enumerate(rows):
        x, y = again[j, 0, 0].double().numpy(), stored[int(r["index"])][0, pos].astype(np.float64)
        cos.append(float(x @ y / np.linalg.norm(x) / np.linalg.norm(y)))
    result = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset": dataset, "read": read_name, "block": block,
              "rows": [int(r["index"]) for r in rows], "cosine": cos, "passed": min(cos) >= min_cos}
    path = S.RESULTS / "checks" / "rereads.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    logger.info(f"re-read {result}")
    if not result["passed"]:
        raise RuntimeError(f"a re-read differs from the stored read: {result}")


def parity_gemma(dataset: str = "instruct_dishonesty__plain", picks=(0, 1, 2, 3)) -> None:
    """This environment (transformers 5.8, this reader) against the replication's (4.55, `activations`):
    Liars' Bench rows tokenised as the replication tokenised them, read here, must reproduce the pooled
    vectors stored at block 31 (cos >= 0.9998, `forced_denial_bank.MIN_COSINE`)."""
    from deception_detection.repository import DatasetRepository
    from experiments.other_models import lb_models as LB
    from experiments.knowledge.forced_denial_bank import MIN_COSINE
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
    fire.Fire({"test": test, "dev": dev, "k_inside": k_inside, "smoke": smoke, "parity_gemma": parity_gemma,
               "reread": reread})
