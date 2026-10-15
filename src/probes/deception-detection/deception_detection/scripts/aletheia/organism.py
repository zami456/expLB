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

    python deception_detection/scripts/aletheia/organism.py behaviour --dataset BotC
    python deception_detection/scripts/aletheia/organism.py read --dataset BotC
"""

import json
import math
import time

import fire
import numpy as np
import torch

from deception_detection.log import logger
from deception_detection.scripts.aletheia import data as D
from deception_detection.scripts.aletheia import k as K
from deception_detection.scripts.aletheia import read as AR
from deception_detection.scripts.aletheia import render as R
from deception_detection.scripts.aletheia import spec as S

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
    from deception_detection.scripts.soft_trigger import quantised_plan

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


def _reader(d: S.Dataset):
    check = json.loads((S.RESULTS / "checks" / f"organism_{d.key}.json").read_text(encoding="utf-8"))
    if not check["passed"]:
        raise RuntimeError(f"{d.key}'s adapter failed its behaviour check")
    plan, _, int8, sent = plan_for(d.lora, d.base.mid, check["scale"])
    logger.info(f"{d.key}: adapter to block {d.base.mid}, {'int8' if int8 else 'bf16'}, {sent / 2**20:.0f} MiB a session")
    return plan, check["scale"], check["read"]


def read(dataset: str, what: str = "k,dev,test", budget: int | None = None, session_passes: int = 30) -> None:
    """Through the adapter, up to the middle block, at the organism grid: K's 240 rows, the organism's
    dev rows (`spec.DEV_PER_CLASS` a class, both dev splits) and its test rows in every render."""
    d = S.BY_KEY[dataset]
    AR.raise_timeouts()
    plan, scale, read_name = _reader(d)
    model_obj = AR.load_model(d.model)
    tok = R.tokenizer(d.model)
    layers = list(S.MODELS[d.model].organism_grid)
    budget = budget or AR.BUDGET[d.model]
    parts = what.split(",") if isinstance(what, str) else list(what)
    if "k" in parts:
        AR.read_rows(model_obj, tok, K.rendered(d.model), layers, AR.folder_for(d.model, read_name, "k", "no-reasoning"),
                     budget, plan, scale, session_passes)
    if "dev" in parts:
        lora = d.lora.split("/")[-1]
        for key in [k for k in D.dev_keys() if k.split("__")[1] == d.model and k.split("__")[2] == lora]:
            rows = AR.dev_sample(key)
            rendered = AR._render_all(rows, d.model, "no-reasoning")
            AR.read_rows(model_obj, tok, rendered, layers, AR.folder_for(d.model, read_name, f"dev__{key}", "no-reasoning"),
                         budget, plan, scale, session_passes)
    if "test" in parts:
        rows = D.rows(dataset)
        for render in AR.renders_for(rows):
            rendered = AR._render_all(rows, d.model, render)
            AR.read_rows(model_obj, tok, rendered, layers,
                         AR.folder_for(d.model, read_name, f"test__{dataset}", render), budget, plan, scale,
                         session_passes)


if __name__ == "__main__":
    fire.Fire({"behaviour": behaviour, "parity_merged": parity_merged, "read": read})
