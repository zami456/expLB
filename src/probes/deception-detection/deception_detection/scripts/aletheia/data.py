"""The competition's rows, split into the leaderboard's ten datasets, with the labels kept apart.

`download` fetches the `dev` and `test` configs of the gated HF dataset (never `validation`) into
E:/aletheia/data/raw. `split` writes each test dataset's rows without their labels to
data/test/<key>.jsonl and the labels to data/labels/test_<key>.json, which only `score.py` opens; it
asserts the row and lie counts of `spec.DATASETS`. The dev rows are split the same way by (split,
base model, adapter), as dev/<split>__<model>__<adapter>.jsonl; their labels are free to use.

    python deception_detection/scripts/aletheia/data.py download
    python deception_detection/scripts/aletheia/data.py split
"""

import json
import os
from pathlib import Path

import fire
import pandas as pd

from deception_detection.log import logger
from deception_detection.scripts.aletheia import spec as S

RAW = S.DATA / "raw"
DEV_FILES = ("dev/instructed_deception.parquet", "dev/varied_deception.parquet")
TEST_FILES = tuple(sorted({f"test/{d.file}.parquet" for d in S.DATASETS}))
SHORT = {m.repo: k for k, m in S.MODELS.items()}


def _token() -> str | None:
    from dotenv import load_dotenv

    load_dotenv(S.DD.parents[2] / ".env")
    return os.environ.get("HF_TOKEN")


def download() -> None:
    from huggingface_hub import hf_hub_download

    for name in DEV_FILES + TEST_FILES:
        path = hf_hub_download(S.HF_DATASET, name, repo_type="dataset", local_dir=RAW, token=_token())
        logger.info(f"{name} -> {path}")


def _messages(m) -> list[dict]:
    return [{"role": t["role"], "content": t["content"] or "", "reasoning": t.get("reasoning") or ""} for t in m]


def _write_rows(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in frame.itertuples(index=False):
            row = {"index": int(r.index), "model": r.model, "lora": r.lora, "messages": _messages(r.messages)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_labels(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(int(i)): bool(d) for i, d in zip(frame["index"], frame["deceptive"])}),
                    encoding="utf-8")


def split() -> None:
    for d in S.DATASETS:
        df = pd.read_parquet(RAW / "test" / f"{d.file}.parquet")
        part = df[(df["model"] == d.base.repo) & (df["lora"].fillna("None") == (d.lora or "None"))]
        n, lies = len(part), int(part["deceptive"].sum())
        if (n, lies) != (d.n, d.lies):
            raise AssertionError(f"{d.key}: {n} rows / {lies} lies, expected {d.n} / {d.lies}")
        if part["index"].duplicated().any():
            raise AssertionError(f"{d.key}: duplicate indices")
        _write_rows(S.DATA / "test" / f"{d.key}.jsonl", part)
        _write_labels(S.DATA / "labels" / f"test_{d.key}.json", part)
        logger.info(f"test {d.key}: {n} rows, {lies} lies")
    covered = sum(d.n for d in S.DATASETS)
    total = sum(len(pd.read_parquet(RAW / f)) for f in TEST_FILES)
    if covered != total:
        raise AssertionError(f"the ten datasets cover {covered} of {total} test rows")
    for name in DEV_FILES:
        df = pd.read_parquet(RAW / name)
        split_name = Path(name).stem
        for (model, lora), part in df.groupby([df["model"], df["lora"].fillna("None")]):
            key = f"{split_name}__{SHORT[model]}__{lora.split('/')[-1]}"
            _write_rows(S.DATA / "dev" / f"{key}.jsonl", part)
            _write_labels(S.DATA / "labels" / f"dev_{key}.json", part)
            logger.info(f"dev {key}: {len(part)} rows, {int(part['deceptive'].sum())} lies")


def rows(key: str, split_name: str = "test") -> list[dict]:
    path = S.DATA / split_name / f"{key}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dev_keys() -> list[str]:
    return sorted(p.stem for p in (S.DATA / "dev").glob("*.jsonl"))


def labels(key: str, split_name: str = "test") -> dict[int, bool]:
    """Only `score.py` calls this for the test split."""
    raw = json.loads((S.DATA / "labels" / f"{split_name}_{key}.json").read_text(encoding="utf-8"))
    return {int(i): v for i, v in raw.items()}


if __name__ == "__main__":
    fire.Fire({"download": download, "split": split})
