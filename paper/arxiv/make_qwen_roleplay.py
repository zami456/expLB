"""How fully Qwen2.5-72B plays the persona roles under its two banks' prompt formats, from the judge's
score files on the data drive: the original bank (the role line as the system message) and the re-run
(the role line and a stay-in-character sentence in the user turn). Writes `data/qwen_roleplay.csv`, which
`make_arxiv_assets.py` reads, so that later builds do not need the drive. Run from
`src/probes/deception-detection` when the drive is mounted:

    python paper/arxiv/make_qwen_roleplay.py
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "paper" / "thesis"))
import make_xmodel_assets as X  # noqa: E402  its score-file reader, so the counts match the thesis's

from experiments.other_models import lb_models as LB  # noqa: E402

FOLDERS = {"system message": LB.XMODEL_ROOT / "banks" / "qwen-72b" / "scores",
           "user turn": LB.XMODEL_ROOT / "banks" / "qwen-72b" / "userstay" / "persona" / "scores"}
MIN_KEPT = 20  # a role enters the strict bank with at least this many rollouts scored 3 (the chain's rule)


def main() -> None:
    rows = []
    for fmt, folder in FOLDERS.items():
        if not folder.exists():
            raise SystemExit(f"the judge scores are not reachable: {folder}")
        per_role = pd.DataFrame(X._roleplay_rows(folder, f"Qwen2.5-72B, {fmt}"))
        rows.append({"format": fmt, "roles": len(per_role), "judged": int(per_role["judged"].sum()),
                     "score 3": int(per_role["score 3"].sum()),
                     "share fully in role": float(per_role["score 3"].sum() / per_role["judged"].sum()),
                     "min kept per role": int(per_role["score 3"].min()),
                     f"roles under {MIN_KEPT}": int((per_role["score 3"] < MIN_KEPT).sum())})
    out = pd.DataFrame(rows)
    (HERE / "data").mkdir(exist_ok=True)
    out.to_csv(HERE / "data" / "qwen_roleplay.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
