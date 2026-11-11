"""D2, the thesis's base probe (the RepE probe), for a base model without Liars' Bench reads.

The recipe is `layer_sweep.probe`'s: RepE's "plain" set (`data/repe.py`: each true fact under "Pretend
you're an honest person ..." and under "... a dishonest person ..."), every token of the fact except its
last five words, the first 80% of the interleaved rows for training, a StandardScaler and a logistic
regression with C = 0.1 and no intercept (`detectors.LogisticRegressionDetector`, reg_coeff 10). It
scores a pooled read as `((z - mean) / scale) @ w`, as `persona_ood.run_probe` does. Only the fact's
first part is rendered: the model is causal, so the last five words never touch the tokens read.

    python experiments/external/aletheia/repe.py read --model qwen35
    python experiments/external/aletheia/repe.py train --model qwen35
"""

import time

import fire
import numpy as np
import torch

from deception_detection.log import logger
from experiments.external.aletheia import read as AR
from experiments.external.aletheia import render as R
from experiments.external.aletheia import spec as S

REG_COEFF = 10.0
TRAIN_SHARE = 0.8


def rows() -> list[dict]:
    from deception_detection.data.repe import RepeHonestDishonestFacts
    from deception_detection.types import Label

    data = RepeHonestDishonestFacts(variant="plain")
    out = []
    for i, (dialogue, label) in enumerate(zip(data.dialogues, data.labels)):
        user = next(m.content for m in dialogue if m.role == "user")
        start = next(m.content for m in dialogue if m.role == "assistant" and m.detect)
        if not start.strip():
            continue  # a fact of five words or fewer: nothing is detected, as in the thesis's probe
        out.append({"index": i, "label": int(label == Label.DECEPTIVE),
                    "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": start}]})
    return out



def token_path(model: str, block: int):
    return S.ACTS / model / "base" / f"repe_tokens_B{block}.npz"


def read(model: str = "qwen35", block: int | None = None, budget: int = 8000, retries: int = 4) -> None:
    """Every row's span tokens at `block` (the middle block by default), float16, one file."""
    from deception_detection.activations import decoder_blocks

    block = int(block or S.MODELS[model].mid)
    out = token_path(model, block)
    if out.exists():
        logger.info(f"{out} exists")
        return
    model_obj = AR.load_model(model)
    tok = R.tokenizer(model)
    items = {r["index"]: R.render(r, model) for r in rows()}
    requests = AR.plan_requests({i: len(it["ids"]) for i, it in items.items()}, budget)
    blocks = decoder_blocks(model_obj)
    feats, owners = [], []
    for k, batch in enumerate(requests):
        tokens, mask, spans = AR.tensors(tok, [items[i] for i in batch])
        for attempt in range(1, retries + 1):
            try:
                with model_obj.trace(tokens, attention_mask=mask, remote=True) as tracer:
                    h = blocks[block].output
                    h = h[0] if isinstance(h, tuple) else h
                    saved = h.to(torch.float16).cpu().save()
                    tracer.stop()
                break
            except Exception as error:  # noqa: BLE001
                if attempt == retries:
                    raise
                logger.warning(f"request {k + 1} failed ({error!r:.200}); retry {attempt}")
                time.sleep(30 * attempt)
        h = saved.float().numpy()
        for j, i in enumerate(batch):
            sel = spans[j, 0].bool().numpy()
            feats.append(h[j, sel].astype(np.float16))
            owners.append(np.full(int(sel.sum()), i))
        logger.info(f"RepE tokens: request {k + 1}/{len(requests)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, tokens=np.concatenate(feats), owner=np.concatenate(owners))
    logger.info(f"-> {out}")


def train(model: str = "qwen35", block: int | None = None) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    block = int(block or S.MODELS[model].mid)
    saved = np.load(token_path(model, block))
    x, owner = saved["tokens"].astype(np.float32), saved["owner"]
    meta = {r["index"]: r["label"] for r in rows()}
    order = sorted(meta)
    first = set(order[:int(TRAIN_SHARE * len(order))])
    keep = np.array([int(i) in first for i in owner])  # the first 80% of the interleaved rows
    y = np.array([meta[int(i)] for i in owner])
    scaler = StandardScaler().fit(x[keep])
    probe = LogisticRegression(C=1 / REG_COEFF, fit_intercept=False, max_iter=1000)
    probe.fit(scaler.transform(x[keep]), y[keep])
    path = S.RESULTS / "repe" / f"probe_{model}_B{block}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=scaler.mean_, scale=scaler.scale_, w=probe.coef_[0])
    held = ~keep
    acc = float((probe.predict(scaler.transform(x[held])) == y[held]).mean())
    logger.info(f"{model} RepE probe at block {block}: {int(keep.sum())} train tokens, held-out token accuracy "
                f"{acc:.3f} -> {path}")


def probe(model: str, block: int):
    path = S.RESULTS / "repe" / f"probe_{model}_B{block}.npz"
    if not path.exists():
        return None
    p = np.load(path)
    mean, scale, w = p["mean"], p["scale"], p["w"]
    return lambda z: ((z - mean) / scale) @ w


if __name__ == "__main__":
    fire.Fire({"read": read, "train": train})
