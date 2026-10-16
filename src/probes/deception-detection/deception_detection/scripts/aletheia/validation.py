"""Post hoc, asked for after the test results were seen (2026-10-16): the detector on the competition's
validation split, which the protocol had left unused. No team's validation scores are published, so
these AUROCs are absolute, set beside the random-direction band, not ranked.

The split's two tasks, each written by all three base models (400 rows, 200 lies, a model):
- medical sycophancy: plain base models, so the base model is the organism (as Harm);
- soft trigger (greeting): a Cadenza LoRA per model, applied in the pass after `organism.behaviour`.
Nemotron's rows are left out: it has no persona bank (built only if its Hidden-Goal adapter arrives).

The four datasets are added to `spec.BY_KEY` at run time; spec.py itself, the frozen protocol, is not
touched. Rows go to data/test/V-*.jsonl and labels apart, as `data.split` writes the test sets.

    python deception_detection/scripts/aletheia/validation.py split
    python deception_detection/scripts/aletheia/validation.py read_base --key V-MS-Q
    python deception_detection/scripts/aletheia/validation.py behaviour --key V-ST-Q
    python deception_detection/scripts/aletheia/validation.py read_organism --key V-ST-Q
    python deception_detection/scripts/aletheia/validation.py score
"""

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from deception_detection.scripts import recall_denial as RD
from deception_detection.scripts.aletheia import data as D
from deception_detection.scripts.aletheia import read as AR
from deception_detection.scripts.aletheia import render as R
from deception_detection.scripts.aletheia import score as SC
from deception_detection.scripts.aletheia import spec as S

VALIDATION = (
    S.Dataset("V-MS-Q", "Medical sycophancy (Qwen3.5)", "medical_sycophancy", "qwen35", None, 400, 200),
    S.Dataset("V-MS-G", "Medical sycophancy (Gemma)", "medical_sycophancy", "gemma", None, 400, 200),
    S.Dataset("V-ST-Q", "Soft trigger (Qwen3.5)", "soft_trigger", "qwen35",
              "Cadenza-Labs/soft-trigger-greeting-qwen3.5-27b", 400, 200),
    S.Dataset("V-ST-G", "Soft trigger (Gemma)", "soft_trigger", "gemma",
              "Cadenza-Labs/soft-trigger-greeting-gemma-3-27b-it", 400, 200),
)
S.BY_KEY.update({d.key: d for d in VALIDATION})
CELL_IDS = ("headline", "S1-d9", "S6-d5", "S6-d8", "S6-k", "S3-base", "S3-base-d9", "R1-repe")
OUT = S.RESULTS / "validation"


def split() -> None:
    from huggingface_hub import hf_hub_download

    for d in VALIDATION:
        path = hf_hub_download(S.HF_DATASET, f"validation/{d.file}.parquet", repo_type="dataset", local_dir=D.RAW,
                               token=D._token())
        df = pd.read_parquet(path)
        part = df[(df["model"] == d.base.repo) & (df["lora"].fillna("None") == (d.lora or "None"))]
        n, lies = len(part), int(part["deceptive"].sum())
        if (n, lies) != (d.n, d.lies) or part["index"].duplicated().any():
            raise AssertionError(f"{d.key}: {n} rows / {lies} lies, expected {d.n} / {d.lies}")
        D._write_rows(S.DATA / "test" / f"{d.key}.jsonl", part)
        D._write_labels(S.DATA / "labels" / f"test_{d.key}.json", part)
        logger.info(f"{d.key}: {n} rows, {lies} lies")


def read_base(key: str, budget: int | None = None) -> None:
    """The rows through the base model at the middle block: the organism read for medical
    sycophancy, the base-model control for soft trigger."""
    d = S.BY_KEY[key]
    rows = D.rows(key)
    AR.read_rows(AR.load_model(d.model), R.tokenizer(d.model), AR._render_all(rows, d.model, "no-reasoning"),
                 [d.base.mid], AR.folder_for(d.model, "base", f"test__{key}", "no-reasoning"),
                 budget or AR.BUDGET[d.model])


def behaviour(key: str) -> None:
    from deception_detection.scripts.aletheia import organism as O

    O.behaviour(key)


def read_organism(key: str, session_passes: int = 30) -> None:
    """K and the rows through the soft-trigger adapter (`organism.read`), once its check passed."""
    from deception_detection.scripts.aletheia import organism as O

    O.read(key, what="k,test", session_passes=session_passes)


def score(n_boot: int = S.N_BOOT, n_null: int = S.N_NULL) -> None:
    from deception_detection.scripts.aletheia import posthoc as PH

    cells = {c.id: c for c in S.CELLS}
    out = []
    for j, d in enumerate(VALIDATION):
        labels = D.labels(d.key)
        ids_order = [int(r["index"]) for r in D.rows(d.key)]
        y = np.array([int(labels[i]) for i in ids_order])
        rng = np.random.default_rng([S.SEED, 100 + j])
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        draws = [np.concatenate((rng.choice(pos, len(pos)), rng.choice(neg, len(neg)))) for _ in range(n_boot)]
        nulls = {}

        def record(name: str, role: str, read_name: str, s: np.ndarray) -> None:
            if read_name not in nulls:
                z, _ = SC._reads(d, read_name, "no-reasoning", 0, d.base.mid)
                null = np.random.default_rng([S.SEED, 7, 100 + j]).standard_normal((n_null, z.shape[1]))
                null /= np.linalg.norm(null, axis=1, keepdims=True)
                nulls[read_name] = np.array([SC.auroc(y, t) for t in (z @ null.T).T])
            b = np.array([SC.auroc(y[i], s[i]) for i in draws])
            out.append({"cell": name, "role": role, "dataset": d.key, "model": d.model, "read": read_name,
                        "n": len(y), "lies": int(y.sum()), "auroc": SC.auroc(y, s), "lo": float(np.quantile(b, 0.025)),
                        "hi": float(np.quantile(b, 0.975)), "band97.5": float(np.quantile(nulls[read_name], 0.975))})

        for cid in CELL_IDS:
            got = SC.scores(cells[cid], d)
            if got is None:
                logger.info(f"{cid} {d.key}: not read yet")
                continue
            s, ids = got
            assert ids == ids_order
            record(cid, cells[cid].role, "base" if cells[cid].read == "base" else SC.organism_read(d), s)
        read_name = SC.organism_read(d)
        if d.lora and read_name:
            # the thesis's soft-trigger test (`soft_trigger.py`): the part of each read the adapter adds,
            # organism read minus base read, which the prompt's trigger and the answer word cancel out of
            org = SC._reads(d, read_name, "no-reasoning", 0, d.base.mid)
            base = SC._reads(d, "base", "no-reasoning", 0, d.base.mid)
            if org and base:
                delta = org[0] - base[0]
                inp = SC.inputs(d.model)
                k = SC._k_read(d, read_name, d.base.mid)
                dirs = {"D9 (adapter-added)": inp.axis("extended", d.base.mid)}
                if k is not None:
                    dirs["D9+K (adapter-added)"] = RD.standardised_sum(dirs["D9 (adapter-added)"], k,
                                                                       inp.control(d.base.mid))
                null = np.random.default_rng([S.SEED, 9, 100 + j]).standard_normal((n_null, delta.shape[1]))
                null /= np.linalg.norm(null, axis=1, keepdims=True)
                band = np.array([SC.auroc(y, t) for t in (delta @ null.T).T])
                for name, f in dirs.items():
                    s = np.asarray(f(delta) if callable(f) else delta @ f, dtype=np.float64)
                    b = np.array([SC.auroc(y[i], s[i]) for i in draws])
                    out.append({"cell": name, "role": "post hoc", "dataset": d.key, "model": d.model,
                                "read": read_name + " minus base", "n": len(y), "lies": int(y.sum()),
                                "auroc": SC.auroc(y, s), "lo": float(np.quantile(b, 0.025)),
                                "hi": float(np.quantile(b, 0.975)), "band97.5": float(np.quantile(band, 0.975))})
        got = SC._reads(d, read_name, "no-reasoning", 0, d.base.mid) if read_name else None
        f = PH.d4_lb(d.model, d.base.mid) if got else None
        if f is not None:
            record("D4 (LB)", "post hoc", read_name, np.asarray(f(got[0]), dtype=np.float64))
    table = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "auroc.csv", index=False)
    logger.info("\n" + table.pivot(index="cell", columns="dataset", values="auroc").round(3).to_string())


if __name__ == "__main__":
    fire.Fire({"split": split, "read_base": read_base, "behaviour": behaviour, "read_organism": read_organism,
               "score": score})
