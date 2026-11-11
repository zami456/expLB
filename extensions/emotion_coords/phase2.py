"""Phase 2 of the emotion-coordinate study: the full run's Liars' Bench rows and activations.

    python -m extensions.emotion_coords.phase2 subset       # the capped rows: <= 1,000 per class per set (500 for IT), seeded
    python -m extensions.emotion_coords.phase2 extract_bc   # (b) last token and (c) pre-response at layer 20, from the cache

Pooling (a) needs no extraction: the thesis's pooled files hold every row at layers 20, 28, 40 and 52 (Phase 0
matched them row by row). The D2/D3 prompts are written by `rows.py d2d3 --full` and read on NDIF by
`reader/full.jobs`.
"""

import json

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from extensions.emotion_coords import config as C
from extensions.emotion_coords import lbdata as LB


def subset() -> None:
    rows = LB.full_subset()
    (C.data_dir("index") / "full_subset.json").write_text(
        json.dumps({n: [int(r) for r in v] for n, v in rows.items()}), encoding="utf-8")
    counts = []
    for name, keep in rows.items():
        labels = LB.index(name).set_index("pooled_row").loc[keep, "label"]
        counts.append({"dataset": name, "short": LB.short(name), "rows": len(keep),
                       "deceptive": int((labels == "DECEPTIVE").sum()), "honest": int((labels == "HONEST").sum()),
                       "available": len(LB.index(name))})
    table = pd.DataFrame(counts)
    table.to_csv(C.results_dir("phase2") / "full_subset_counts.csv", index=False)
    logger.info(f"full subset:\n{table.to_string(index=False)}\ntotal {table['rows'].sum()}")


def extract_bc() -> None:
    LB.extract_bc("full")


if __name__ == "__main__":
    fire.Fire({"subset": subset, "extract_bc": extract_bc})
