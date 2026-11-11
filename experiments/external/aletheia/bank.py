"""The persona bank's rollouts read by this code: a parity check against the bank, and D9 rebuilt
inside each fine-tune (spec S4).

- `parity`: one persona's kept rollouts, rendered by `render.py` and read by `read.py` through the base
  model, must average to the bank's own vector at the middle block (the bank was read by the
  collaborator's pipeline, right-padded, in other batches; TF32 noise leaves cos ~0.9999).
- `subsample` / `read_inside`: the thesis's rule for a fine-tune (`persona_finetuned.py`): each of the
  25 contrast personas contributes one kept rollout per question, at most 30, drawn at random, and
  the same rollouts are read through the organism (and the base model), so the persona contrasts can
  be rebuilt inside it.

    python experiments/external/aletheia/bank.py parity --model qwen35
"""

import json

import fire
import numpy as np

from deception_detection import persona_pcs as pcs
from deception_detection.log import logger
from experiments.external.aletheia import read as AR
from experiments.external.aletheia import render as R
from experiments.external.aletheia import spec as S

CONTRAST = tuple(sorted(set(pcs.DECEPTION_PERSONAS + pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS
                            + pcs.KNOWLEDGE_CANDOUR_PERSONAS)))
KEEP = {"vectors_s3": (3,), "vectors_s23": (2, 3)}


def persona_root(model: str):
    from experiments.external.aletheia.score import inputs

    return inputs(model).bank_dir.parent, inputs(model).bank_dir.name


def kept(model: str, persona: str) -> list[dict]:
    """The rollouts the primary bank averaged for `persona`, in file order (`default` keeps all)."""
    root, bank = persona_root(model)
    rows = [json.loads(line) for line in (root / "responses" / f"{persona}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if persona == "default":
        return rows
    scores = json.loads((root / "scores" / f"{persona}.json").read_text(encoding="utf-8"))
    return [r for r in rows if scores.get(f"pos_p{r['prompt_index']}_q{r['question_index']}") in KEEP[bank]]


def _row(r: dict, index: int) -> dict:
    return {"index": index, "messages": [{"role": m["role"], "content": m["content"]} for m in r["conversation"]]}


def parity(model: str = "qwen35", persona: str = "pc_coverup_operator", min_cos: float = 0.9995) -> None:
    import torch

    rows = [_row(r, i) for i, r in enumerate(kept(model, persona))]
    root, bank = persona_root(model)
    block = S.MODELS[model].mid
    rendered = {r["index"]: R.render(r, model) for r in rows}
    folder = AR.folder_for(model, "base", f"bank_parity__{persona}", "no-reasoning")
    AR.read_rows(AR.load_model(model), R.tokenizer(model), rendered, [block], folder, AR.BUDGET[model])
    reads = AR.load(folder)
    mean = np.mean([v[0, 0] for v in reads.values()], axis=0).astype(np.float64)
    payload = torch.load(root / bank / f"{persona}.pt", weights_only=True)
    layers = [int(x) for x in payload.get("layers", range(payload["vector"].shape[0]))]
    stored = payload["vector"][layers.index(block)].double().numpy()
    cos = float(mean @ stored / np.linalg.norm(mean) / np.linalg.norm(stored))
    result = {"model": model, "bank": bank, "persona": persona, "block": block, "rollouts": len(rows),
              "bank n": int(payload["n"]), "cosine": cos, "passed": cos >= min_cos and len(rows) == int(payload["n"])}
    path = S.RESULTS / "checks" / f"bank_parity_{model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    logger.info(f"bank parity {result}")
    if not result["passed"]:
        raise RuntimeError(f"this reader does not rebuild the bank's vector: {result}")


def subsample(model: str, seed: int = S.SEED) -> list[dict]:
    """One kept rollout per question for each of the 25 contrast personas, at random; frozen in a file."""
    path = S.RESULTS / "bank" / f"subsample_{model}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["rows"]
    rng = np.random.default_rng(seed)
    out = []
    for persona in CONTRAST:
        by_q: dict[int, list[dict]] = {}
        for r in kept(model, persona):
            by_q.setdefault(int(r["question_index"]), []).append(r)
        for q in sorted(by_q):
            pick = by_q[q][int(rng.integers(len(by_q[q])))]
            out.append(_row(pick, len(out)) | {"persona": persona, "question_index": q})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"bank": persona_root(model)[1], "rows": out}, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    logger.info(f"{model}: {len(out)} rollouts of {len(CONTRAST)} personas -> {path}")
    return out


def subsample_bank(model: str) -> str:
    """The bank (vectors_s3 or vectors_s23) whose kept rollouts the subsample was drawn from."""
    return json.loads((S.RESULTS / "bank" / f"subsample_{model}.json").read_text(encoding="utf-8"))["bank"]


def rendered(model: str) -> dict[int, dict]:
    return {int(r["index"]): R.render(r, model) for r in subsample(model)}


def contrasts_from_reads(model: str, reads: dict[int, np.ndarray], layer_pos: int) -> dict[str, np.ndarray]:
    """D5, D8 and D9 from the subsample's reads: each persona's mean over its rollouts, then the
    difference of the group means of personas (equal weight per persona, as the bank's axes)."""
    meta = {int(r["index"]): r["persona"] for r in subsample(model)}
    persona_mean = {}
    for p in CONTRAST:
        vs = [v[0, layer_pos] for i, v in reads.items() if meta[i] == p]
        persona_mean[p] = np.mean(vs, axis=0).astype(np.float64)

    def group(names):
        return np.mean([persona_mean[n] for n in names], axis=0)

    def unit(v):
        return v / np.linalg.norm(v)

    return {"D5": unit(group(pcs.DECEPTION_PERSONAS) - group(pcs.HONESTY_PERSONAS)),
            "D8": unit(group(pcs.KNOWLEDGE_DENIAL_PERSONAS) - group(pcs.KNOWLEDGE_CANDOUR_PERSONAS)),
            "D9": unit(group(pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS)
                       - group(pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS))}


def read_merged(dataset: str) -> None:
    """The subsample through an AISI organism's merged copy and through the base model."""
    d = S.BY_KEY[dataset]
    layers = list(S.MODELS[d.model].organism_grid)
    items = rendered(d.model)
    for name, repo in (("merged-" + d.merged.split("/")[-1], d.merged), ("base", None)):
        AR.read_rows(AR.load_model(d.model, repo), R.tokenizer(d.model), items, layers,
                     AR.folder_for(d.model, name, "bank_subsample", "no-reasoning"), AR.BUDGET[d.model])


if __name__ == "__main__":
    fire.Fire({"parity": parity, "subsample": subsample, "read_merged": read_merged})
