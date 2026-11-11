"""Replicating the knowledge direction k with an independent, frame-matched item set.

k was built from one hand-written set of 60 known + 60 unknown questions (`recall_items.py`), whose
two sides also differ in style (textbook trivia against private particulars). This builds k' the
same way from a second set (`recall_items_matched.py`): new questions, each known item paired with an
unknown one of the same form and answer type. Same template, same two forced denial sentences, same
no-role system prompt, same reads, at every sweep layer.

    python deception_detection/scripts/k_replication.py verify          # NDIF: the model answers the known items
    python deception_detection/scripts/k_replication.py items
    python deception_detection/scripts/k_replication.py read --pilot    # 20 + 20 items first
    python deception_detection/scripts/k_replication.py read            # NDIF, resumable
    python deception_detection/scripts/k_replication.py score
    python deception_detection/scripts/k_replication.py style           # text only: how separable are the sets?

Replication criteria, fixed before running: at layers 36, 40 and 44, cos(k', k) > 0.5; k' scores
HP-KR's identical-text slice above the random-direction 97.5th percentile and above the 97.5th
percentile of balanced splits of its own items; and k' scores Gender-Secret above the random 97.5th
percentile at layers 40 and 44.

Outputs go to `results/recall_denial/matched/`.
"""

import json
import time

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from deception_detection.scripts import k_robustness as kr
from deception_detection.scripts import recall_denial as rd
from deception_detection.scripts import recall_items_matched
from deception_detection.scripts.forced_denial_bank import DENY, NO_ROLE_SYSTEM
from deception_detection.scripts.layer_sweep import ALL_LAYERS, done_rows, save_request

OUT = rd.OUT / "matched"
PARTS = OUT / "parts_answer"
N_PILOT = 20
BUDGET = 16000
CRITERIA_LAYERS = (36, 40, 44)
N_BALANCED = 500


def build_rows(items: list[dict], system: str = NO_ROLE_SYSTEM) -> list[dict]:
    """Each item under the no-role system prompt with each forced denial sentence."""
    rows = []
    for it in items:
        for s, sentence in enumerate(DENY):
            rows.append({k: it[k] for k in ("item", "subject", "knows", "user")}
                        | {"arm": "none", "system": system, "kind": "deny", "string": s, "assistant": sentence})
    for r, row in enumerate(rows):
        row["row"] = r
    return rows


def pilot_rows(rows: list[dict], n: int = N_PILOT) -> list[dict]:
    """The rows of the first `n` known and first `n` unknown items."""
    keep = {f"gen-known-{i}" for i in range(n)} | {f"gen-unknown-{i}" for i in range(n)}
    return [r for r in rows if r["item"] in keep]


# Another model replicates its own K (`k_models.py`): the known items are checked by the letter with
# the highest logit (as `k_models.verify`), the rows carry the model's HP-KR system turn, and the reads
# stop after the deepest block. Its folder is `results/xmodel/<model>/k_matched/`.

def out_dir(model: str | None = None):
    if model is None or model == "llama":
        return OUT
    from deception_detection.scripts import lb_models as LB

    return LB.XMODEL_RESULTS / LB.spec(model).key / "k_matched"


def verify(max_new_tokens: int = 8, model: str | None = None) -> None:
    if model is None:
        rd.verify(max_new_tokens, source=recall_items_matched, out=OUT)
        return
    from deception_detection.scripts import k_models as km

    km.verify(model, source=recall_items_matched, out=out_dir(model))


def items(model: str | None = None) -> None:
    out = out_dir(model)
    verified = {v["question"]: v["correct"] for v in json.loads((out / "verify.json").read_text(encoding="utf-8"))["known"]}
    general = rd.general_items(verified, source=recall_items_matched)
    replaced = [q for q, *_ in recall_items_matched.KNOWN if not verified.get(q, True)]
    if model is None:
        rows = build_rows(general)
    else:
        from deception_detection.scripts import k_models as km

        rows = build_rows(general, km.system_turn(km._target(model)))
    (out / "items.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(f"{len(rows)} rows from {len(general)} items; known items replaced from the reserve: {replaced}")


def _rows(model: str | None = None) -> list[dict]:
    return json.loads((out_dir(model) / "items.json").read_text(encoding="utf-8"))


def read_model(model: str, layers=None) -> None:
    """Another model's answer-token means at every block of its sweep, the pass stopped after the
    deepest one (`k_models._read_batch`). Resumable."""
    from deception_detection.models import ModelName, get_model_and_tokenizer
    from deception_detection.scripts import forced_denial_bank as fdb
    from deception_detection.scripts import k_models as km
    from deception_detection.scripts import lb_models as LB
    from deception_detection.scripts.gender_secret import _environment

    target = km._target(model)
    layers = [int(x) for x in (layers or LB.spec(model).layers)]
    rows = _rows(model)
    _environment()
    remote, tokenizer = get_model_and_tokenizer(ModelName(target.model), template_date=km._template_date(target))
    fdb._check_identical_answers(tokenizer, rows)
    km._check_empty_system(tokenizer, rows)
    parts = out_dir(model) / "parts_answer"
    done = done_rows(parts)
    todo = [r for r in rows if r["row"] not in done]
    requests = fdb._requests(tokenizer, todo, km.TOKEN_BUDGET) if todo else []
    logger.info(f"{len(todo)} rows to read at {len(layers)} blocks in {len(requests)} requests")
    for k, batch in enumerate(requests):
        means = km._read_batch(remote, tokenizer, batch, layers)
        if not np.isfinite(means).all():
            raise RuntimeError(f"request {k + 1} returned non-finite reads")
        save_request(parts, [r["row"] for r in batch], layers, means)
        logger.info(f"request {k + 1}/{len(requests)}: {len(batch)} rows")


def read(pilot: bool = False, layers=ALL_LAYERS, retries: int = 3, model: str | None = None) -> None:
    """Answer-token means through the base model at every sweep layer. Resumable. `model`
    reads another model instead (`read_model`)."""
    if model is not None:
        read_model(model)
        return
    from deception_detection.models import get_model_and_tokenizer
    from deception_detection.scripts import forced_denial_bank as fdb
    from deception_detection.scripts.gender_secret import _environment

    layers = [int(x) for x in layers]
    rows = pilot_rows(_rows()) if pilot else _rows()
    _environment()
    model, tokenizer = get_model_and_tokenizer("llama-70b-3.3")
    fdb._check_spans(tokenizer, rows, out=OUT)
    fdb._check_identical_answers(tokenizer, rows)
    done = done_rows(PARTS)
    todo = [r for r in rows if r["row"] not in done]
    requests = fdb._requests(tokenizer, todo, BUDGET) if todo else []
    logger.info(f"{len(todo)} rows to read at {len(layers)} layers in {len(requests)} requests")
    for k, batch in enumerate(requests):
        start = time.time()
        for attempt in range(1, retries + 1):
            try:
                means = fdb._read_batch(model, tokenizer, batch, "answer", layers)
                break
            except Exception as error:  # NDIF drops a request now and then
                if attempt == retries:
                    raise
                logger.warning(f"request {k + 1} failed ({error!r}); retry {attempt}/{retries - 1}")
                time.sleep(30 * attempt)
        if not np.isfinite(means).all():
            raise RuntimeError(f"request {k + 1} returned non-finite reads")
        save_request(PARTS, [r["row"] for r in batch], layers, means)
        logger.info(f"request {k + 1}/{len(requests)}: {len(batch)} rows, {time.time() - start:.0f} s")


def item_means(layer: int, rows: list[dict] | None = None, model: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Each matched item's mean over its two forced sentences at `layer`, and whether it is known."""
    rows = {r["row"]: r for r in (rows or _rows(model))}
    vecs: dict[str, list[np.ndarray]] = {}
    for row, by_layer in rd._load(out_dir(model) / "parts_answer").items():
        if row in rows and layer in by_layer:
            vecs.setdefault(rows[row]["item"], []).append(by_layer[layer])
    names = sorted({r["item"] for r in rows.values()})
    if any(len(vecs.get(n, [])) != 2 for n in names):
        raise RuntimeError(f"layer {layer}: some items lack both reads")
    knows = {r["item"]: bool(r["knows"]) for r in rows.values()}
    return np.stack([np.mean(vecs[n], axis=0) for n in names]), np.array([knows[n] for n in names])


def score(layers=None, n_null: int = kr.N_NULL, n_balanced: int = N_BALANCED, seed: int = kr.SEED,
          model: str | None = None, bank: str | None = None) -> None:
    """k' at every layer: its cosine with k, and k's headline numbers recomputed with k', beside the
    random band and (at the criteria layers) balanced splits of the matched items. Another model's
    criteria layers are its 36, 40 and 44."""
    inputs = rd.Inputs(model, bank)
    out = out_dir(model)
    layers = layers or (ALL_LAYERS if inputs.llama else inputs.spec.layers)
    criteria = CRITERIA_LAYERS if inputs.llama else inputs.spec.k_layers
    rng = np.random.default_rng(seed)
    rows = []
    for layer in [int(x) for x in layers]:
        X, known = item_means(layer, model=model)
        c0 = kr.weights(known, np.flatnonzero(known), np.flatnonzero(~known))
        k_new = rd._unit(X.T @ c0)
        Xo, known_o, _ = inputs.item_means(layer)
        k_old = rd._unit(Xo.T @ kr.weights(known_o, np.flatnonzero(known_o), np.flatnonzero(~known_o)))
        sets = kr.Sets(layer, X, inputs)
        point = sets.metrics(c0)
        band = sets.random_band(n_null, rng)
        row = {"layer": layer, "cos(k', k)": float(k_new @ k_old)} | point | {f"random97.5 {m}": b for m, b in band.items()}
        if layer in criteria:
            balanced = [sets.metrics(kr.balanced_weights(known, rng)) for _ in range(n_balanced)]
            for m in ("k: HP-KR identical", "k: Gender-Secret"):
                if m in point:
                    v = np.array([d[m] for d in balanced])
                    row[f"balanced97.5 {m}"] = float(np.quantile(v, 0.975))
                    row[f"balanced share >= {m}"] = float(np.mean(v >= point[m]))
        rows.append(row)
        logger.info(f"layer {layer} scored")
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    # the D5 + k and D9 + k columns depend on the persona bank: another bank's run sits beside the primary bank's
    suffix = "" if inputs.bank == inputs.spec.primary_bank else f"_{inputs.bank}"
    table.round(4).to_csv(out / f"score{suffix}.csv", index=False)

    verdicts = []
    for layer in criteria:
        r = table.set_index("layer").loc[layer]
        verdicts.append({"layer": layer, "criterion": "cos(k', k) > 0.5", "value": r["cos(k', k)"], "pass": r["cos(k', k)"] > 0.5})
        if "k: HP-KR identical" in r:
            verdicts.append({"layer": layer, "criterion": "HP-KR identical text above random 97.5%", "value": r["k: HP-KR identical"],
                             "pass": r["k: HP-KR identical"] > r["random97.5 k: HP-KR identical"]})
            verdicts.append({"layer": layer, "criterion": "HP-KR identical text above balanced splits' 97.5%", "value": r["k: HP-KR identical"],
                             "pass": r["k: HP-KR identical"] > r["balanced97.5 k: HP-KR identical"]})
        if layer in criteria[1:] and "k: Gender-Secret" in r:
            verdicts.append({"layer": layer, "criterion": "Gender-Secret above random 97.5%", "value": r["k: Gender-Secret"],
                             "pass": r["k: Gender-Secret"] > r["random97.5 k: Gender-Secret"]})
    verdicts = pd.DataFrame(verdicts)
    verdicts.round(4).to_csv(out / f"verdicts{suffix}.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["layer", "cos(k', k)", "k: HP-KR identical", "random97.5 k: HP-KR identical", "k: HP-KR denials", "k: Gender-Secret",
            "random97.5 k: Gender-Secret", "k: HP-C", "k: mean of five", "D5 + k: mean of five", "D9 + k: mean of five"]
    logger.info(f"k' (matched items) across depth:\n{table[[c for c in show if c in table]].round(3).to_string(index=False)}")
    logger.info(f"replication criteria:\n{verdicts.round(3).to_string(index=False)}")


def separability(texts: list[str], labels: np.ndarray, folds: int = 10, seed: int = 0) -> float:
    """Cross-validated accuracy of a bag-of-words classifier telling known from unknown questions."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return float(cross_val_score(model, texts, labels, cv=cv).mean())


def style() -> None:
    """How much the two sides of each item set differ in wording alone, before any model is involved."""
    from deception_detection.scripts import recall_items

    out = {}
    for name, source in (("original", recall_items), ("matched", recall_items_matched)):
        known = [q for q, *_ in source.KNOWN[:rd.N_GENERAL]]
        unknown = [q for q, _ in source.UNKNOWN[:rd.N_GENERAL]]
        out[name] = separability(known + unknown, np.array([1] * len(known) + [0] * len(unknown)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "style.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    logger.info(f"10-fold accuracy telling known from unknown questions by their words alone: {out}")


if __name__ == "__main__":
    fire.Fire({"verify": verify, "items": items, "read": read, "score": score, "style": style})
