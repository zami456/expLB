"""The competition's organisms read through their LoRA adapters inside the NDIF pass, and the checks
that must pass first.

The AISI organisms are read through their public merged copies (`read.py`), which NDIF deploys on the
pilot key. BotC, Echoblast and the two Hidden-Goal organisms have no merged copy, so their adapters
are applied in the pass (`read.session_means`), as the thesis read its fine-tunes. Before an adapter
read is used:

- `behaviour`: the adapter, applied in the pass at every block, must make the organism's own dev
  rollouts (instructed deception, written by the organism) more likely than the base model does, on
  at least `spec.BEHAVIOUR_MIN` of 20 rows (mean log-probability of the reply tokens). The adapters'
  configs and training files disagree on the scale for rsLoRA, so the scale is swept (alpha/r,
  alpha/sqrt(r), 1.0) and the one that makes the rollouts most likely is kept; that is the scale the
  organism was run with. The verdict and the read's name go to results/aletheia/checks/organism_<key>.json.
- `parity_merged`: for an AISI organism, the merged copy and its adapter applied in the pass must read
  alike at block 16, which also settles the AISI scale.

    python experiments/external/aletheia/organism.py behaviour --dataset BotC
    python experiments/external/aletheia/organism.py read --dataset BotC
"""

import json
import math
import time

import fire
import numpy as np
import torch

from deception_detection.log import logger
from experiments.external.aletheia import data as D
from experiments.external.aletheia import k as K
from experiments.external.aletheia import read as AR
from experiments.external.aletheia import render as R
from experiments.external.aletheia import spec as S

LOGIT_TOKENS = 1024  # positions per log-prob pass: the head writes 248k logits a position
INT8_ABOVE = 380 * 2**20  # bf16 factors above this go as int8 (395 MiB went through, 790 MiB was refused)


def _config(lora: str) -> dict:
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(lora, "adapter_config.json", token=R._token()), encoding="utf-8") as f:
        return json.load(f)


def candidate_scales(lora: str) -> dict[str, float]:
    c = _config(lora)
    r, alpha = float(c["r"]), float(c["lora_alpha"])
    return {"alpha/r": alpha / r, "alpha/sqrt(r)": alpha / math.sqrt(r), "1.0": 1.0}


def plan_for(lora: str, max_block: int, scale: float):
    """The adapter's call plan up to `max_block` and the read's name: exact bf16 factors when they fit
    one request, int8 with one scale per row otherwise."""
    from deception_detection.lora_intervention import call_plan, load_adapter
    from experiments.organisms.soft_trigger import quantised_plan

    adapter = load_adapter(lora, max_block=max_block, scale=scale)
    size = adapter.n_bytes()
    if size <= INT8_ABOVE:
        return call_plan(adapter), adapter, False, size
    return quantised_plan(adapter), adapter, True, size // 2


def _dev_rollouts(d: S.Dataset, n_per_class: int = 10) -> list[dict]:
    key = f"instructed_deception__{d.model}__{d.lora.split('/')[-1]}"
    rows = D.rows(key, "dev")
    labels = D.labels(key, "dev")
    rng = np.random.default_rng(S.SEED)
    out = []
    for cls in (True, False):
        ids = sorted(int(r["index"]) for r in rows if labels[int(r["index"])] == cls)
        keep = set(int(i) for i in rng.choice(ids, size=n_per_class, replace=False))
        out += [r for r in rows if int(r["index"]) in keep]
    return out


def reply_logprobs(model_obj, tok, rows: list[dict], model: str, plan, scales: list[float]) -> np.ndarray:
    """[scales, rows]: the mean log-probability of each row's reply tokens (the content span), with the
    adapter's update times each scale in the pass at every block the plan covers; scale 0 is the base
    model. One NDIF session, so the factors go up once for all the scales."""
    return item_logprobs(model_obj, tok, [R.render(r, model) for r in rows], plan, scales)


def item_logprobs(model_obj, tok, items: list[dict], plan, scales: list[float], remote: bool | str = True) -> np.ndarray:
    """`reply_logprobs` on rendered rows (`render.render`'s dicts)."""
    import nnsight

    from deception_detection.activations import decoder_blocks

    blocks = decoder_blocks(model_obj)
    head = model_obj.lm_head
    out = np.zeros((len(scales), len(items)))
    groups, cur, cur_len = [], [], 0
    for i, it in enumerate(items):
        if cur and (len(cur) + 1) * max(cur_len, len(it["ids"])) > LOGIT_TOKENS:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(i)
        cur_len = max(cur_len, len(it["ids"]))
    groups.append(cur)
    inputs = []
    for g in groups:
        tokens, mask, spans = AR.tensors(tok, [items[i] for i in g])
        targets = torch.zeros_like(tokens)
        targets[:, :-1] = tokens[:, 1:]
        tmask = torch.zeros(tokens.shape, dtype=torch.float32)
        tmask[:, :-1] = spans[:, 0, 1:]  # position t predicts token t+1: score the reply tokens
        inputs.append((tokens, mask, targets, tmask))
    if remote == "local":
        from nnsight.intervention.backends.local_simulation import LocalSimulationBackend

        session = model_obj.session(backend=LocalSimulationBackend(model_obj))
    else:
        session = model_obj.session(remote=remote)
    with session:
        results = nnsight.save(list())
        for tokens, mask, targets, tmask in inputs:
            for scale in scales:
                with model_obj.trace(tokens, attention_mask=mask):
                    for block in sorted(plan):
                        for entry in plan[block]:
                            module = getattr(getattr(blocks[block], entry[0]), entry[1])
                            x = module.input
                            if len(entry) == 4:
                                a = entry[2].to(x.device, x.dtype)
                                b = entry[3].to(x.device, x.dtype)
                            else:
                                a = entry[2].to(x.device, x.dtype) * entry[3].to(x.device, x.dtype)
                                b = entry[4].to(x.device, x.dtype) * entry[5].to(x.device, x.dtype)
                            module.output = module.output + ((x @ a.T) @ b.T) * scale
                    logits = head.output
                    lp = torch.log_softmax(logits.float(), dim=-1)
                    picked = lp.gather(-1, targets.to(lp.device).unsqueeze(-1)).squeeze(-1)
                    m = tmask.to(lp.device)
                    results.append(((picked * m).sum(-1) / m.sum(-1)).cpu())
    k = 0
    for g in groups:
        for j in range(len(scales)):
            out[j, g] = results[k].detach().float().numpy()
            k += 1
    return out

def behaviour(dataset: str, n_per_class: int = 10) -> None:
    d = S.BY_KEY[dataset]
    if d.lora is None or d.merged:
        raise ValueError(f"{dataset} is read without an in-trace adapter")
    AR.raise_timeouts()
    model_obj = AR.load_model(d.model)
    tok = R.tokenizer(d.model)
    rows = _dev_rollouts(d, n_per_class)
    full = S.MODELS[d.model].n_blocks - 1
    start = time.time()
    candidates = candidate_scales(d.lora)
    plan, adapter, int8, sent = plan_for(d.lora, full, 1.0)  # the scale is applied in the pass
    logger.info(f"{dataset}: adapter at every block, {'int8' if int8 else 'bf16'}, {sent / 2**20:.0f} MiB")
    lps = reply_logprobs(model_obj, tok, rows, d.model, plan, [0.0] + list(candidates.values()))
    base = lps[0]
    result = {"dataset": dataset, "adapter": d.lora, "rows": [int(r["index"]) for r in rows],
              "factors": f"{'int8' if int8 else 'bf16'}, {sent / 2**20:.0f} MiB", "base": base.round(4).tolist(),
              "scales": {}}
    for (name, scale), lp in zip(candidates.items(), lps[1:]):
        result["scales"][name] = {"scale": scale, "mean": float(lp.mean()), "beats base": int((lp > base).sum()),
                                  "logprob": lp.round(4).tolist()}
        logger.info(f"{dataset} scale {name} = {scale:.4g}: mean reply log-prob {lp.mean():.4f} "
                    f"(base {base.mean():.4f}), beats base on {(lp > base).sum()}/{len(rows)}")
    best = max(result["scales"], key=lambda n: result["scales"][n]["mean"])
    chosen = result["scales"][best]
    passed = chosen["beats base"] >= S.BEHAVIOUR_MIN * len(rows) // 20
    _, _, int8_read, _ = plan_for(d.lora, d.base.mid, chosen["scale"])
    result |= {"chosen scale": best, "scale": chosen["scale"], "passed": bool(passed),
               "read": f"lora-{d.lora.split('/')[-1]}@{chosen['scale']:.4g}" + ("-int8" if int8_read else ""),
               "read_int8": int8_read, "seconds": round(time.time() - start)}
    path = S.RESULTS / "checks" / f"organism_{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    logger.info(f"{dataset}: scale {best} ({chosen['scale']:.4g}) beats base on {chosen['beats base']}/{len(rows)}: "
                f"{'passed' if passed else 'FAILED'}")


def parity_merged(dataset: str = "GS-F", block: int = 16, n_rows: int = 4) -> None:
    """An AISI organism's merged copy against its adapter applied in the pass, at `block`, for each
    candidate scale (cosine per row of the content-span means)."""
    d = S.BY_KEY[dataset]
    AR.raise_timeouts()
    tok = R.tokenizer(d.model)
    rows = _dev_rollouts(d, n_rows // 2)
    items = [R.render(r, d.model) for r in rows]
    batch = [AR.tensors(tok, items)]
    merged = AR.session_means(AR.load_model(d.model, d.merged), batch, [block])[0][:, 0, 0].double()
    base_model = AR.load_model(d.model)
    base = AR.session_means(base_model, batch, [block])[0][:, 0, 0].double()
    result = {"dataset": dataset, "block": block,
              "cos merged . base": torch.nn.functional.cosine_similarity(merged, base, dim=1).tolist()}
    for name, scale in candidate_scales(d.lora).items():
        plan, _, int8, sent = plan_for(d.lora, block, scale)
        traced = AR.session_means(base_model, batch, [block], plan, scale)[0][:, 0, 0].double()
        result[f"cos merged . in-trace @ {name} ({scale:.4g})"] = \
            torch.nn.functional.cosine_similarity(merged, traced, dim=1).tolist()
        result[f"factors @ {name}"] = f"{'int8' if int8 else 'bf16'}, {sent / 2**20:.0f} MiB"
    path = S.RESULTS / "checks" / f"parity_merged_{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    logger.info(f"parity {result}")


def config_scale(lora: str) -> float:
    """The scale PEFT itself would apply: alpha / sqrt(r) for rsLoRA, else alpha / r."""
    c = _config(lora)
    r, alpha = float(c["r"]), float(c["lora_alpha"])
    return alpha / (math.sqrt(r) if c.get("use_rslora") else r)


def split_logprobs(model_obj, tok, items: list[dict], first: dict, second: dict, mid: int, n: int, scale: float,
                   remote: bool | str = True) -> tuple[np.ndarray, np.ndarray]:
    """The reply's mean token log-prob, base and adapted, with the adapted pass split after block `mid`
    into two requests (int8 plans `first` for blocks <= mid and `second` for the rest)."""
    import nnsight

    from deception_detection.activations import decoder_blocks

    def session():
        if remote == "local":
            from nnsight.intervention.backends.local_simulation import LocalSimulationBackend

            return model_obj.session(backend=LocalSimulationBackend(model_obj))
        return model_obj.session(remote=remote)

    blocks = decoder_blocks(model_obj)
    head = model_obj.lm_head
    groups, cur, cur_len = [], [], 0
    for i, it in enumerate(items):
        if cur and (len(cur) + 1) * max(cur_len, len(it["ids"])) > LOGIT_TOKENS:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(i)
        cur_len = max(cur_len, len(it["ids"]))
    groups.append(cur)
    inputs = []
    for g in groups:
        tokens, mask, spans = AR.tensors(tok, [items[i] for i in g])
        targets = torch.zeros_like(tokens)
        targets[:, :-1] = tokens[:, 1:]
        tmask = torch.zeros(tokens.shape, dtype=torch.float32)
        tmask[:, :-1] = spans[:, 0, 1:]
        inputs.append((tokens, mask, targets, tmask))
    with session():
        hs = nnsight.save(list())
        for tokens, mask, _, _ in inputs:
            with model_obj.trace(tokens, attention_mask=mask) as tracer:
                for block in range(mid + 1):
                    for entry in first.get(block, []):
                        module = getattr(getattr(blocks[block], entry[0]), entry[1])
                        x = module.input
                        a = entry[2].to(x.device, x.dtype) * entry[3].to(x.device, x.dtype)
                        b = entry[4].to(x.device, x.dtype) * entry[5].to(x.device, x.dtype)
                        module.output = module.output + ((x @ a.T) @ b.T) * scale
                h = blocks[mid].output
                h = h[0] if isinstance(h, tuple) else h
                hs.append(h.cpu())
                tracer.stop()
    hs = [h.detach() for h in hs]
    with session():
        res = nnsight.save(list())
        for (tokens, mask, targets, tmask), h in zip(inputs, hs):
            for adapted in (False, True):
                with model_obj.trace(tokens, attention_mask=mask):
                    if adapted:
                        x0 = blocks[mid + 1].input
                        blocks[mid + 1].input = h.to(x0.device, x0.dtype)
                        for block in range(mid + 1, n):
                            for entry in second.get(block, []):
                                module = getattr(getattr(blocks[block], entry[0]), entry[1])
                                x = module.input
                                a = entry[2].to(x.device, x.dtype) * entry[3].to(x.device, x.dtype)
                                b = entry[4].to(x.device, x.dtype) * entry[5].to(x.device, x.dtype)
                                module.output = module.output + ((x @ a.T) @ b.T) * scale
                    logits = head.output
                    lp = torch.log_softmax(logits.float(), dim=-1)
                    picked = lp.gather(-1, targets.to(lp.device).unsqueeze(-1)).squeeze(-1)
                    m = tmask.to(lp.device)
                    res.append(((picked * m).sum(-1) / m.sum(-1)).cpu())
    base, adapted = np.zeros(len(items)), np.zeros(len(items))
    k = 0
    for g in groups:
        base[g] = res[k].detach().float().numpy()
        adapted[g] = res[k + 1].detach().float().numpy()
        k += 2
    return base, adapted


def split_logprobs_scales(model_obj, tok, items: list[dict], first: dict, second: dict, mid: int, n: int,
                          scales: list[float], positions: int = 512) -> np.ndarray:
    """`split_logprobs` at several scales from one upload of the factors: [1 + len(scales), rows], the
    base model first. Each scale's residual stream after `mid` is fetched and sent back (small). Rows go
    in groups of at most `positions` positions, since the head writes 248k logits a position."""
    import nnsight

    from deception_detection.activations import decoder_blocks

    blocks = decoder_blocks(model_obj)
    head = model_obj.lm_head
    groups, cur, cur_len = [], [], 0
    for i, it in enumerate(items):
        if cur and (len(cur) + 1) * max(cur_len, len(it["ids"])) > positions:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(i)
        cur_len = max(cur_len, len(it["ids"]))
    groups.append(cur)
    inputs = []
    for g in groups:
        tokens, mask, spans = AR.tensors(tok, [items[i] for i in g])
        targets = torch.zeros_like(tokens)
        targets[:, :-1] = tokens[:, 1:]
        tmask = torch.zeros(tokens.shape, dtype=torch.float32)
        tmask[:, :-1] = spans[:, 0, 1:]
        inputs.append((tokens, mask, targets, tmask))
    with model_obj.session(remote=True):
        hs = nnsight.save(list())
        for tokens, mask, _, _ in inputs:
            for scale in scales:
                with model_obj.trace(tokens, attention_mask=mask) as tracer:
                    for block in range(mid + 1):
                        for entry in first.get(block, []):
                            module = getattr(getattr(blocks[block], entry[0]), entry[1])
                            x = module.input
                            a = entry[2].to(x.device, x.dtype) * entry[3].to(x.device, x.dtype)
                            b = entry[4].to(x.device, x.dtype) * entry[5].to(x.device, x.dtype)
                            module.output = module.output + ((x @ a.T) @ b.T) * scale
                    h = blocks[mid].output
                    h = h[0] if isinstance(h, tuple) else h
                    hs.append(h.cpu())
                    tracer.stop()
    hs = [h.detach() for h in hs]
    with model_obj.session(remote=True):
        res = nnsight.save(list())
        for gi, (tokens, mask, targets, tmask) in enumerate(inputs):
            for j in range(len(scales) + 1):
                with model_obj.trace(tokens, attention_mask=mask):
                    if j > 0:
                        scale = scales[j - 1]
                        x0 = blocks[mid + 1].input
                        blocks[mid + 1].input = hs[gi * len(scales) + j - 1].to(x0.device, x0.dtype)
                        for block in range(mid + 1, n):
                            for entry in second.get(block, []):
                                module = getattr(getattr(blocks[block], entry[0]), entry[1])
                                x = module.input
                                a = entry[2].to(x.device, x.dtype) * entry[3].to(x.device, x.dtype)
                                b = entry[4].to(x.device, x.dtype) * entry[5].to(x.device, x.dtype)
                                module.output = module.output + ((x @ a.T) @ b.T) * scale
                    logits = head.output
                    lp = torch.log_softmax(logits.float(), dim=-1)
                    picked = lp.gather(-1, targets.to(lp.device).unsqueeze(-1)).squeeze(-1)
                    m = tmask.to(lp.device)
                    res.append(((picked * m).sum(-1) / m.sum(-1)).cpu())
    out = np.zeros((len(scales) + 1, len(items)))
    k = 0
    for g in groups:
        for j in range(len(scales) + 1):
            out[j, g] = res[k].detach().float().numpy()
            k += 1
    return out

def behaviour_merged(dataset: str, n_per_class: int = 10) -> None:
    """The behaviour check for an AISI organism read through its merged copy: the merged model must
    make the organism's own dev rollouts more likely than the base model does, on at least
    `spec.BEHAVIOUR_MIN` of 20 rows. Written to checks/organism_merged_<key>.json."""
    d = S.BY_KEY[dataset]
    if not d.merged:
        raise ValueError(f"{dataset} has no merged copy")
    tok = R.tokenizer(d.model)
    rows = _dev_rollouts(d, n_per_class)
    items = [R.render(r, d.model) for r in rows]
    merged = item_logprobs(AR.load_model(d.model, d.merged), tok, items, {}, [0.0])[0]
    base = item_logprobs(AR.load_model(d.model), tok, items, {}, [0.0])[0]
    beats = int((merged > base).sum())
    result = {"dataset": dataset, "merged": d.merged, "rows": [int(r["index"]) for r in rows],
              "base": base.round(4).tolist(), "merged lp": merged.round(4).tolist(), "base mean": float(base.mean()),
              "merged mean": float(merged.mean()), "beats base": beats,
              "passed": bool(beats >= S.BEHAVIOUR_MIN * len(rows) // 20)}
    path = S.RESULTS / "checks" / f"organism_merged_{dataset}.json"
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    logger.info(f"{dataset} merged: mean reply log-prob {merged.mean():.4f} (base {base.mean():.4f}), beats base on "
                f"{beats}/{len(rows)}: {'passed' if result['passed'] else 'FAILED'}")


def scale_diagnostic(dataset: str = "Echo", n_rows: int = 20, source: str = "test") -> None:
    """A diagnostic, not the pre-registered check: the organism's own replies (the first `n_rows` test
    rows by index, no labels; or dev rollouts) scored by the base model and by the adapter at each
    candidate scale, the pass split after the middle block (`split_logprobs_scales`)."""
    from deception_detection.lora_intervention import load_adapter
    from experiments.organisms.soft_trigger import quantised_plan

    d = S.BY_KEY[dataset]
    AR.raise_timeouts()
    model_obj = AR.load_model(d.model)
    tok = R.tokenizer(d.model)
    if source == "test":
        rows = sorted(D.rows(dataset), key=lambda r: int(r["index"]))[:n_rows]
    else:
        rows = _dev_rollouts(d, n_rows // 2)
    items = [R.render(r, d.model) for r in rows]
    n, mid = S.MODELS[d.model].n_blocks, d.base.mid
    candidates = candidate_scales(d.lora)
    plan = quantised_plan(load_adapter(d.lora, max_block=n - 1, scale=1.0))
    first = {b: e for b, e in plan.items() if b <= mid}
    second = {b: e for b, e in plan.items() if b > mid}
    start = time.time()
    lps = split_logprobs_scales(model_obj, tok, items, first, second, mid, n, list(candidates.values()))
    result = {"dataset": dataset, "rows from": source, "rows": [int(r["index"]) for r in rows],
              "base mean": float(lps[0].mean()), "scales": {}}
    for (name, scale), lp in zip(candidates.items(), lps[1:]):
        result["scales"][name] = {"scale": scale, "mean": float(lp.mean()), "beats base": int((lp > lps[0]).sum())}
    result["seconds"] = round(time.time() - start)
    path = S.RESULTS / "checks" / f"scale_diagnostic_{dataset}_{source}.json"
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    logger.info(f"scale diagnostic {result}")


def accept_on_test_rows(dataset: str = "Echo", min_rows: int = S.BEHAVIOUR_MIN) -> None:
    """A recorded deviation (user's decision, 2026-10-15): accept an adapter whose pre-registered
    dev-row check failed because the dev rows tagged with it were evidently not written through it
    (the base model finds them more likely, and some match the base model's replies word for word),
    when the same adapter at its config scale makes the organism's own test replies more likely than
    the base model on at least `min_rows` of 20 (`scale_diagnostic`, text only, no labels). The failed
    check is kept as organism_<key>_dev_check.json."""
    d = S.BY_KEY[dataset]
    checks = S.RESULTS / "checks"
    dev = json.loads((checks / f"organism_{dataset}.json").read_text(encoding="utf-8"))
    if dev.get("deviation"):
        logger.info(f"{dataset}: already accepted on its test replies")
        return
    diag = json.loads((checks / f"scale_diagnostic_{dataset}_test.json").read_text(encoding="utf-8"))
    scale = config_scale(d.lora)
    chosen = next(v for v in diag["scales"].values() if abs(v["scale"] - scale) < 1e-9)
    if dev.get("passed") or chosen["beats base"] < min_rows:
        raise RuntimeError(f"{dataset}: nothing to accept (dev check passed {dev.get('passed')}, test rows "
                           f"{chosen['beats base']}/20)")
    (checks / f"organism_{dataset}_dev_check.json").write_text(json.dumps(dev, indent=1) + "\n", encoding="utf-8")
    _, _, int8_read, _ = plan_for(d.lora, d.base.mid, scale)
    accepted = {"dataset": dataset, "adapter": d.lora, "scale": scale, "passed": True,
                "read": f"lora-{d.lora.split('/')[-1]}@{scale:.4g}" + ("-int8" if int8_read else ""),
                "read_int8": int8_read,
                "deviation": "the pre-registered dev-row check failed (organism_" + dataset + "_dev_check.json): the "
                             "dev rows tagged with this adapter are more likely under the base model; accepted on the "
                             "organism's own test replies instead (text only, no labels), by the user's decision",
                "test-row check": {"rows": diag["rows"], "base mean": diag["base mean"], "adapted mean": chosen["mean"],
                                   "beats base": chosen["beats base"], "other scales": diag["scales"]}}
    (checks / f"organism_{dataset}.json").write_text(json.dumps(accepted, indent=1) + "\n", encoding="utf-8")
    logger.info(f"{dataset}: accepted on its test replies ({chosen['beats base']}/20 at scale {scale:g})")


def behaviour_split(dataset: str, n_per_class: int = 10, source: str = "dev") -> None:
    """`behaviour` in two requests, for an adapter whose factors at every block are too large for one
    request over this uplink (Echoblast: ~670 MiB in int8). Request 1 runs the pass up to the middle
    block with the first half of the factors and returns the residual stream after it; request 2 runs
    the base model (scale 0) and, from that residual stream injected at the next block's input, the
    second half (`blocks[mid + 1].input = h`, checked locally to reproduce the full pass). The scale is
    the one PEFT applies (`config_scale`), as BotC's sweep chose for its non-rsLoRA adapter."""
    from deception_detection.lora_intervention import load_adapter
    from experiments.organisms.soft_trigger import quantised_plan

    d = S.BY_KEY[dataset]
    AR.raise_timeouts()
    model_obj = AR.load_model(d.model)
    tok = R.tokenizer(d.model)
    if source == "dev":
        rows = _dev_rollouts(d, n_per_class)
    else:  # a diagnostic, not the pre-registered check: the first rows of the test set by index, no labels
        rows = sorted(D.rows(dataset), key=lambda r: int(r["index"]))[:2 * n_per_class]
    items = [R.render(r, d.model) for r in rows]
    n, mid = S.MODELS[d.model].n_blocks, d.base.mid
    scale = config_scale(d.lora)
    plan = quantised_plan(load_adapter(d.lora, max_block=n - 1, scale=scale))
    first = {b: e for b, e in plan.items() if b <= mid}
    second = {b: e for b, e in plan.items() if b > mid}
    mib = lambda p: sum(t.numel() * t.element_size() for es in p.values() for e in es for t in e[2:]) / 2**20  # noqa: E731
    logger.info(f"{dataset}: int8 factors {mib(first):.0f} MiB up to block {mid}, {mib(second):.0f} MiB after")
    start = time.time()
    base, adapted = split_logprobs(model_obj, tok, items, first, second, mid, n, scale)
    logger.info(f"{dataset}: two requests, {time.time() - start:.0f} s")
    beats = int((adapted > base).sum())
    passed = beats >= S.BEHAVIOUR_MIN * len(rows) // 20
    _, _, int8_read, _ = plan_for(d.lora, mid, scale)
    result = {"dataset": dataset, "adapter": d.lora, "method": "two requests split after the middle block (int8)",
              "rows": [int(r["index"]) for r in rows], "scale": scale, "base": base.round(4).tolist(),
              "adapted": adapted.round(4).tolist(), "base mean": float(base.mean()), "adapted mean": float(adapted.mean()),
              "beats base": beats, "passed": bool(passed),
              "read": f"lora-{d.lora.split('/')[-1]}@{scale:.4g}" + ("-int8" if int8_read else ""),
              "read_int8": int8_read, "seconds": round(time.time() - start)}
    result["rows from"] = source
    name = f"organism_{dataset}.json" if source == "dev" else f"organism_{dataset}_{source}_rows_diagnostic.json"
    path = S.RESULTS / "checks" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    logger.info(f"{dataset} ({source} rows): adapted mean reply log-prob {adapted.mean():.4f} (base {base.mean():.4f}), beats base on "
                f"{beats}/{len(rows)}: {'passed' if passed else 'FAILED'}")


def int8_parity(dataset: str, block: int = 8, n_rows: int = 8, min_cos: float = 0.9998) -> None:
    """The int8 factors against the exact ones, read at `block` on dev rollouts (cos per row)."""
    from deception_detection.lora_intervention import call_plan, load_adapter
    from experiments.organisms.soft_trigger import quantised_plan

    d = S.BY_KEY[dataset]
    AR.raise_timeouts()
    model_obj = AR.load_model(d.model)
    tok = R.tokenizer(d.model)
    items = [R.render(r, d.model) for r in _dev_rollouts(d, n_rows // 2)]
    scale = config_scale(d.lora)
    adapter = load_adapter(d.lora, max_block=block, scale=scale)
    batch = [AR.tensors(tok, items)]
    exact = AR.session_means(model_obj, batch, [block], call_plan(adapter), scale)[0][:, 0, 0].double()
    rounded = AR.session_means(model_obj, batch, [block], quantised_plan(adapter), scale)[0][:, 0, 0].double()
    base = AR.session_means(model_obj, batch, [block])[0][:, 0, 0].double()
    cos = torch.nn.functional.cosine_similarity(exact, rounded, dim=1).tolist()
    result = {"dataset": dataset, "block": block, "cos int8 . exact": cos,
              "cos base . exact": torch.nn.functional.cosine_similarity(exact, base, dim=1).tolist(),
              "passed": min(cos) >= min_cos}
    path = S.RESULTS / "checks" / f"int8_parity_{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    logger.info(f"int8 parity {result}")


def _reader(d: S.Dataset):
    check = json.loads((S.RESULTS / "checks" / f"organism_{d.key}.json").read_text(encoding="utf-8"))
    if not check["passed"]:
        raise RuntimeError(f"{d.key}'s adapter failed its behaviour check")
    plan, _, int8, sent = plan_for(d.lora, d.base.mid, check["scale"])
    logger.info(f"{d.key}: adapter to block {d.base.mid}, {'int8' if int8 else 'bf16'}, {sent / 2**20:.0f} MiB a session")
    return plan, check["scale"], check["read"]


def read(dataset: str, what: str = "k,dev,test", budget: int | None = None, session_passes: int = 60) -> None:
    """Through the adapter, up to the middle block, at the organism grid, in as few sessions as the
    rows allow (`read.read_sets`): K's 240 rows, the organism's dev rows (`spec.DEV_PER_CLASS` a class,
    both dev splits), the persona bank's subsample (`bank`) and its test rows in every render (`test`)."""
    d = S.BY_KEY[dataset]
    AR.raise_timeouts()
    plan, scale, read_name = _reader(d)
    model_obj = AR.load_model(d.model)
    tok = R.tokenizer(d.model)
    layers = list(S.MODELS[d.model].organism_grid)
    parts = what.split(",") if isinstance(what, str) else list(what)
    sets = {}
    if "k" in parts:
        sets[AR.folder_for(d.model, read_name, "k", "no-reasoning")] = K.rendered(d.model)
    if "dev" in parts:
        lora = d.lora.split("/")[-1]
        for key in [k for k in D.dev_keys() if k.split("__")[1] == d.model and k.split("__")[2] == lora]:
            rows = AR.dev_sample(key)
            sets[AR.folder_for(d.model, read_name, f"dev__{key}", "no-reasoning")] = AR._render_all(rows, d.model,
                                                                                                  "no-reasoning")
    if "bank" in parts:  # S4: the persona bank's subsample through the adapter
        from experiments.external.aletheia import bank as B

        sets[AR.folder_for(d.model, read_name, "bank_subsample", "no-reasoning")] = B.rendered(d.model)
    if "rule_k" in parts:  # post hoc: rule K's rows through the adapter (`posthoc.py`)
        from experiments.external.aletheia import posthoc as PH

        sets[AR.folder_for(d.model, read_name, PH.FOLDER, "no-reasoning")] = PH.rendered(d.model)
    if "test" in parts:
        AR.check_freeze()
        rows = D.rows(dataset)
        for render in AR.renders_for(rows):
            sets[AR.folder_for(d.model, read_name, f"test__{dataset}", render)] = AR._render_all(rows, d.model, render)
    AR.read_sets(model_obj, tok, sets, layers, budget or AR.BUDGET[d.model], plan, scale, session_passes)

if __name__ == "__main__":
    fire.Fire({"behaviour": behaviour, "behaviour_split": behaviour_split, "int8_parity": int8_parity,
               "scale_diagnostic": scale_diagnostic, "behaviour_merged": behaviour_merged,
               "accept_on_test_rows": accept_on_test_rows,
               "parity_merged": parity_merged, "read": read})
