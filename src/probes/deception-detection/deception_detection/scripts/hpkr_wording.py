"""HP-KR's wording confound: do the persona directions still separate lies when both sides deny?

On HP-KR every lie is a denial of knowledge, but 93% of honest answers claim knowledge, so a
detector can score well by reading denial *wording*. The control is the denial-only slice: the
lies (the model knows, says it doesn't) against the honest denials (the model has no consistent
answer, says it doesn't know). Both sides say "I don't know"; only the knowledge differs.

Rows are paired with their answer text through the repo loader's own shuffle (`np.random.seed(42)`
permutation of the Liars' Bench order), checked by reproducing all 2,139 stored labels. Answers are
classed by `steer_lies.knowledge_claim`, the regex that agrees 100% with the benchmark's judge where
it decides. Everything is zero-shot and CPU-only, from the pooled activations already on disk.

    python deception_detection/scripts/hpkr_wording.py

Writes `results/hpkr_wording/`: `auroc.csv` (every direction, layer and slice, with a bootstrap
interval), `null.csv` (random directions) and `summary.md`.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from deception_detection import persona_pcs as pcs
from deception_detection.log import logger
from deception_detection.scripts import persona_ood as P
from deception_detection.scripts.layer_sweep import ALL_LAYERS, base_run, pooled_dir
from deception_detection.scripts.steer_lies import knowledge_claim
from deception_detection.utils import REPO_ROOT

OUT = REPO_ROOT / "results" / "hpkr_wording"
HPKR = "harm_refusal_knowledge_report__plain"
SINCERE = ("pc_calibrated_reporter", "pc_honest_novice", "pc_scrupulous_witness")
PLAIN = ("pc_candid_witness", "pc_forthright_expert", "pc_plain_dealer")
N_BOOT, N_NULL, SEED = 2000, 200, 0


def hpkr_frame() -> pd.DataFrame:
    """The Liars' Bench HP-KR rows in pooled order, with the fields the slices need.

    Pooled rows follow the repo loader's shuffle (`DialogueDataset.randomly_shuffle_dialogues`:
    `np.random.seed(42)`, one permutation of the Liars' Bench order), checked here by reproducing
    every stored label.
    """
    import os
    import re

    from datasets import load_dataset

    lb = load_dataset("Cadenza-Labs/liars-bench", "harm-pressure-knowledge-report", split="test",
                      token=os.getenv("HF_TOKEN")).to_pandas()
    lb = lb[lb["model"] == "llama-v3.3-70b-instruct"].reset_index(drop=True)
    np.random.seed(42)
    perm = np.random.permutation(len(lb))
    lb = lb.iloc[perm].reset_index(drop=True)
    stored = np.load(pooled_dir(40) / f"{HPKR}.npz")["labels"] == "DECEPTIVE"
    if not (stored == lb["deceptive"].to_numpy()).all():
        raise RuntimeError("pooled rows do not line up with the Liars' Bench rows")
    text = lb["messages"].map(lambda m: m[-1]["content"])
    return pd.DataFrame({
        "lie": lb["deceptive"].to_numpy().astype(int),
        "denied": text.map(knowledge_claim).eq("DENIED_KNOWLEDGE").to_numpy(),
        "answer_len": text.str.len().to_numpy(),
        "prompt_len": lb["messages"].map(lambda m: len(m[1]["content"])).to_numpy(),
        "subject": lb["dataset"].str.split("/").str[1].to_numpy(),
        "belief": lb["meta"].map(lambda s: int(re.search(r"'model_belief': (-?\d+)", s).group(1))).to_numpy(),
        "messages": lb["messages"].to_numpy(),
    })


def answers() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per pooled row: is it a lie, is the answer a denial, and its length -- in pooled order."""
    frame = hpkr_frame()
    return frame["lie"].to_numpy(), frame["denied"].to_numpy(), frame["answer_len"].to_numpy()


def directions(layer: int) -> dict[str, callable]:
    bank = pcs.load_bank_at_layer(layer, pcs.EXTENDED_VECTORS_DIR)

    def contrast(pos, neg):
        d = bank.vectors_for(pos).mean(0) - bank.vectors_for(neg).mean(0)
        d = d / np.linalg.norm(d)
        return lambda z, d=d: z @ d

    out = {
        "knowledge-denial (4v6)": contrast(pcs.KNOWLEDGE_DENIAL_PERSONAS, pcs.KNOWLEDGE_CANDOUR_PERSONAS),
        "extended (12v13)": contrast(pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS,
                                     pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS),
        "published (8v7)": contrast(pcs.DECEPTION_PERSONAS, pcs.HONESTY_PERSONAS),
        "feigned vs sincere ignorance (4v3)": contrast(pcs.KNOWLEDGE_DENIAL_PERSONAS, SINCERE),
        "feigned vs plain assertion (4v3)": contrast(pcs.KNOWLEDGE_DENIAL_PERSONAS, PLAIN),
    }
    run = base_run(layer)
    if (run / "detector.pt").exists():
        out["base probe (RepE)"] = P.run_probe(run)
    return out


def auroc_ci(y: np.ndarray, s: np.ndarray, rng) -> tuple[float, float, float]:
    point = roc_auc_score(y, s)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boots.append(roc_auc_score(y[i], s[i]))
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT.parents[2] / ".env")
    OUT.mkdir(parents=True, exist_ok=True)
    y, denied, length = answers()
    slices = {"full HP-KR": np.ones_like(denied), "denials only": denied}
    logger.info(f"denial slice: {int(y[denied].sum())} lies v {int((1 - y[denied]).sum())} honest denials")
    rng = np.random.default_rng(SEED)
    rows, nulls = [], []
    for name, keep in slices.items():
        p, lo, hi = auroc_ci(y[keep], length[keep], rng)
        rows.append({"layer": "-", "slice": name, "direction": "response length", "auroc": p, "lo": lo, "hi": hi,
                     "n_lies": int(y[keep].sum()), "n_honest": int((1 - y[keep]).sum())})
    for layer in ALL_LAYERS:
        z = np.load(pooled_dir(layer) / f"{HPKR}.npz")["pooled"].astype(np.float64)
        for direction, score in directions(layer).items():
            s = score(z)
            for name, keep in slices.items():
                p, lo, hi = auroc_ci(y[keep], s[keep], rng)
                rows.append({"layer": layer, "slice": name, "direction": direction, "auroc": p, "lo": lo,
                             "hi": hi, "n_lies": int(y[keep].sum()), "n_honest": int((1 - y[keep]).sum())})
        null = rng.standard_normal((N_NULL, z.shape[1]))
        null /= np.linalg.norm(null, axis=1, keepdims=True)
        for name, keep in slices.items():
            a = np.array([roc_auc_score(y[keep], z[keep] @ d) for d in null])
            nulls.append({"layer": layer, "slice": name, "p2.5": np.quantile(a, 0.025),
                          "p97.5": np.quantile(a, 0.975)})
        logger.info(f"layer {layer} done")
    table = pd.DataFrame(rows)
    table.round(4).to_csv(OUT / "auroc.csv", index=False)
    pd.DataFrame(nulls).round(4).to_csv(OUT / "null.csv", index=False)
    wide = table[table["layer"] != "-"].pivot_table(index=["direction", "layer"], columns="slice", values="auroc")
    logger.info(f"wrote {OUT}\n{wide.round(3).to_string()}")


# --- reading the model before it answers -------------------------------------------------------

CONFIGS = {  # layer -> (config, cache dir), as the configs' own headers say
    20: ("XXXX-9-llama-all.yaml", REPO_ROOT / "activations" / "cache"),
    40: ("XXXX-9-llama-all-L40.yaml", "E:/liars-bench-acts/cache"),
    44: ("XXXX-9-llama-all-L44.yaml", "E:/liars-bench-acts/cache"),
}
HEADER = 5  # <|eot_id|> <|start_header_id|> assistant <|end_header_id|> \n\n
QUESTION = 16
POSITIONS = ("pre", "header", "question", "answer")


def extract(layers=(20, 40, 44)) -> None:
    """Per HP-KR row, the activation at the last pre-answer token, the mean over the assistant
    header, the mean over the question's last tokens, and (as a check) the answer mean."""
    import os

    import yaml

    from deception_detection.experiment import Experiment, ExperimentConfig
    from deception_detection.scripts.pool_activations import iter_cached_examples
    from deception_detection.tokenized_data import TokenizedDataset

    OUT.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        config, cache = CONFIGS[int(layer)]
        os.environ["ACTS_CACHE_DIR"] = str(cache)
        cfg = ExperimentConfig(**yaml.safe_load(
            (REPO_ROOT / "deception_detection" / "scripts" / "configs" / config).read_text()))
        experiment = Experiment(cfg)
        dataset = next(d for d in experiment.eval_datasets if d.name == HPKR)
        toks = TokenizedDataset.from_dataset(
            dataset, experiment.tokenizer, detect_all=cfg.detect_all_tokens,
            detect_only_start_of_turn=cfg.detect_only_start_of_turn,
            detect_only_last_token=cfg.detect_only_last_token)
        vecs, labels, checked = [], [], False
        for ex in iter_cached_examples(dataset, experiment, str(experiment.model.name_or_path),
                                       cfg.detect_layers, desc=f"L{layer}"):
            a = ex.acts[:, 0, :].float()
            first = int(ex.mask.nonzero()[0])
            if first < HEADER + QUESTION:
                raise RuntimeError(f"row {ex.index}: answer starts at {first}, too early")
            if not checked:  # the positions before the answer must be the assistant header
                row = toks.tokens[ex.index][-len(ex.mask):]
                header = experiment.tokenizer.decode(row[first - HEADER:first])
                logger.info(f"header tokens: {header!r}")
                assert header.startswith("<|eot_id|><|start_header_id|>assistant"), header
                checked = True
            vecs.append(np.stack([
                a[first - 1].numpy(),
                a[first - HEADER:first].mean(0).numpy(),
                a[first - HEADER - QUESTION:first - HEADER].mean(0).numpy(),
                a[ex.mask].mean(0).numpy(),
            ]))
            labels.append(dataset.labels[ex.index].name)
        vecs = np.stack(vecs).astype(np.float32)
        stored = np.load(pooled_dir(layer) / f"{HPKR}.npz")
        if list(labels) != list(stored["labels"]):
            raise RuntimeError(f"layer {layer}: cache walk does not match the pooled rows")
        err = np.abs(vecs[:, 3] - stored["pooled"]).max() / np.abs(stored["pooled"]).max()
        logger.info(f"layer {layer}: answer mean vs pooled, max relative error {err:.2e}")
        assert err < 1e-3, err
        np.savez(OUT / f"preanswer_L{layer}.npz", vecs=vecs, labels=np.array(labels),
                 positions=np.array(POSITIONS))
        logger.info(f"wrote preanswer_L{layer}.npz {vecs.shape}")


def _cv_probe(x: np.ndarray, y: np.ndarray) -> float:
    """Out-of-fold AUROC of a within-slice logistic probe: a supervised ceiling, not a result."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1e-3, max_iter=2000))
    scores = cross_val_predict(model, x, y, cv=StratifiedKFold(8, shuffle=True, random_state=SEED),
                               method="decision_function")
    return float(roc_auc_score(y, scores))


def preanswer(layers=(20, 40, 44)) -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT.parents[2] / ".env")
    y, denied, _ = answers()
    slices = {"full HP-KR": np.ones_like(denied), "denials only": denied}
    rng = np.random.default_rng(SEED)
    rows, nulls = [], []
    for layer in layers:
        data = np.load(OUT / f"preanswer_L{layer}.npz")
        vecs = data["vecs"].astype(np.float64)
        dirs = directions(int(layer))
        null = rng.standard_normal((N_NULL, vecs.shape[2]))
        null /= np.linalg.norm(null, axis=1, keepdims=True)
        for p, position in enumerate(POSITIONS):
            z = vecs[:, p]
            for name, keep in slices.items():
                for direction, score in dirs.items():
                    a, lo, hi = auroc_ci(y[keep], score(z)[keep], rng)
                    rows.append({"layer": layer, "position": position, "slice": name,
                                 "direction": direction, "auroc": a, "lo": lo, "hi": hi})
                band = np.array([roc_auc_score(y[keep], z[keep] @ d) for d in null])
                nulls.append({"layer": layer, "position": position, "slice": name,
                              "p2.5": np.quantile(band, 0.025), "p97.5": np.quantile(band, 0.975)})
                if name == "denials only":
                    rows.append({"layer": layer, "position": position, "slice": name,
                                 "direction": "within-slice probe (supervised, CV)",
                                 "auroc": _cv_probe(z[keep], y[keep]), "lo": np.nan, "hi": np.nan})
        logger.info(f"layer {layer} scored")
    table = pd.DataFrame(rows)
    table.round(4).to_csv(OUT / "preanswer_auroc.csv", index=False)
    band = pd.DataFrame(nulls)
    band.round(4).to_csv(OUT / "preanswer_null.csv", index=False)
    d = table[table["slice"] == "denials only"].pivot_table(
        index=["layer", "direction"], columns="position", values="auroc")[list(POSITIONS)]
    b = band[band["slice"] == "denials only"].pivot_table(index="layer", columns="position",
                                                         values=["p2.5", "p97.5"])
    logger.info(f"denials only:\n{d.round(3).to_string()}\nrandom band:\n{b.round(3).to_string()}")


if __name__ == "__main__":
    import fire

    fire.Fire({"answer": main, "extract": extract, "preanswer": preanswer})
