"""Llama-3.3-70B's own lies from AISI's lie-detection rollouts.

EleutherAI's evaluation battery (github.com/EleutherAI/how-to-catch-an-ai-liar) samples its
`dyl_varied` and `dyl_sandbag_prompted` sets from AISI's `ai-safety-institute/lie-detection-rollouts`.
That dataset also holds Llama-3.3-70B-Instruct's own rollouts on the same two tasks. Llama wrote
these replies itself, so, unlike the battery's transcripts from other models, they are on-policy for
the model the probes read.

- `varied_deception`: trivia questions under system prompts that tell the model to lie, or to answer
  honestly.
- `multiple_choice_sandbagging`: multiple-choice questions with or without a prompt to sandbag.
- `varied_deception_validation`: the same task under 20 system prompts the training split never uses,
  sharing 2 of its questions.

AISI's labels are belief-checked: a lie is a wrong answer to a question the model gets right under
a benign prompt. Rows it marks `ambiguous` are dropped: questions the model gets wrong even
benignly, errors under an honest prompt, and non-answers.

The data is licensed "other", with no terms stated, so it is read from the Hub at a pinned commit
and never copied into this repo.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from deception_detection.data.base import DialogueDataset, DialogueDataType, Message
from deception_detection.types import Dialogue, Label
from deception_detection.utils import REPO_ROOT

REPO_ID = "ai-safety-institute/lie-detection-rollouts"
REVISION = "177a81666ab378ce9d055507bc30f480ab63e74b"
CONFIG = "meta-llama-llama-3.3-70b-instruct"
SPLITS = {
    "varied": "varied_deception",
    "sandbag": "multiple_choice_sandbagging",
    "validation": "varied_deception_validation",
}
# Replies this long are generations cut off at the 6,144-token cap. Two training rows are such
# repetition loops (24-25k characters; the next longest reply in any split is 7.3k). Under the probe's
# per-token training those two rows alone would carry 7.5% of the training tokens.
MAX_REPLY_CHARS = 20_000
# Rows whose question also appears in a Liars' Bench dataset we evaluate on. Written by
# `experiments/probes/aisi_probe.py audit`, which must run before anything trains on this.
EXCLUSIONS = REPO_ROOT / "data" / "aisi_llama" / "liars_bench_overlap.json"


# Another model's own rollouts of the same tasks, at the same pinned commit (`AISIGemmaRollouts`).
GEMMA_CONFIG = "google-gemma-3-27b-it"
GEMMA_EXCLUSIONS = REPO_ROOT / "data" / "aisi_gemma" / "liars_bench_overlap.json"


def split_path(split: str, config: str = CONFIG) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, f"{config}/{split}.parquet", repo_type="dataset",
                                revision=REVISION))


def load_rows(split: str, config: str = CONFIG) -> list[dict[str, Any]]:
    """One split's labelled rows, in file order.

    AISI's `ambiguous` rows are dropped, as are replies cut off at the generation cap
    (`MAX_REPLY_CHARS`).
    """
    frame = pd.read_parquet(split_path(split) if config == CONFIG else split_path(split, config))
    rows = []
    for i, (messages, verdict, reason) in enumerate(
        zip(frame["messages"], frame["is_lie"], frame["lie_reason"])
    ):
        if verdict == "ambiguous":
            continue
        if verdict not in ("lie", "honest"):
            raise ValueError(f"{split} row {i}: unexpected is_lie {verdict!r}")
        messages = json.loads(messages) if isinstance(messages, str) else list(messages)
        if len(messages[-1]["content"]) > MAX_REPLY_CHARS:
            continue
        rows.append({"split": split, "row": i, "messages": messages,
                     "deceptive": verdict == "lie", "lie_reason": str(reason)})
    return rows


def system_prompt(row: dict[str, Any]) -> str:
    return next((m["content"] for m in row["messages"] if m["role"] == "system"), "")


def question(row: dict[str, Any]) -> str:
    """The question itself: the last user turn up to its first blank line, which drops the
    multiple-choice options and answer-format instruction."""
    user = [m["content"] for m in row["messages"] if m["role"] == "user"][-1]
    return user.strip().split("\n\n")[0]


def instruction_matched(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows under a system prompt that both classes of their split were answered under.

    On these, the instruction cannot tell a lie from an honest answer. Both classes hold rows the
    model was told to lie on, and it either did or answered correctly anyway.
    """
    classes = defaultdict(set)
    for row in rows:
        classes[row["split"], system_prompt(row)].add(row["deceptive"])
    return [row for row in rows if len(classes[row["split"], system_prompt(row)]) == 2]


def load_exclusions(path: Path | None = None) -> set[tuple[str, int]]:
    path = EXCLUSIONS if path is None else path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python experiments/probes/aisi_probe.py audit` "
            "first: it checks these questions against the Liars' Bench datasets."
        )
    audit = json.loads(path.read_text(encoding="utf-8"))
    return {(split, int(i)) for split, rows in audit["excluded"].items() for i in rows}


class AISILlamaRollouts(DialogueDataset):
    """Llama-3.3-70B's own lies and honest answers from AISI's rollouts; see the module docstring.

    Variants: `pooled` (both training splits), `varied`, `sandbag`, `matched` (the pooled rows
    under an instruction both classes share) and `validation` (the held-out prompts).
    """

    base_name = "aisi_llama"
    all_variants: list[str] = ["pooled", "varied", "sandbag", "matched", "validation"]
    default_variant = "pooled"
    # detect exactly the reply's tokens, as the Liars' Bench datasets and RepE do
    padding = {model_type: {"left": 0, "right": 0} for model_type in ["llama", "mistral", "qwen", "gemma"]}
    config = CONFIG
    exclusions: Path | None = None  # None: the module's EXCLUSIONS, read at load time

    def _rows(self) -> list[dict[str, Any]]:
        if self.variant in ("varied", "sandbag", "validation"):
            rows = load_rows(SPLITS[self.variant], self.config)
        else:
            rows = load_rows(SPLITS["varied"], self.config) + load_rows(SPLITS["sandbag"], self.config)
        excluded = load_exclusions(self.exclusions)
        rows = [row for row in rows if (row["split"], row["row"]) not in excluded]
        return instruction_matched(rows) if self.variant == "matched" else rows

    def _get_dialogues(self) -> DialogueDataType:
        dialogues: list[Dialogue] = []
        labels: list[Label] = []
        metadata: dict[str, list[Any]] = {"split": [], "row": [], "lie_reason": []}
        for row in self._rows():
            dialogue = [Message(m["role"], m["content"], detect=False) for m in row["messages"]]
            if dialogue[-1].role != "assistant":
                raise ValueError(f"{row['split']} row {row['row']} does not end with the reply")
            dialogue[-1].detect = True
            dialogues.append(dialogue)
            labels.append(Label.DECEPTIVE if row["deceptive"] else Label.HONEST)
            for key in metadata:
                metadata[key].append(row[key])
        return dialogues, labels, metadata

    @staticmethod
    def get_grading_prompt(
        input_dialogue: Dialogue, response: Message, metadata: dict[str, Any] | None = None
    ) -> dict[str, str]:
        raise NotImplementedError("AISI's rollouts come labelled; there is nothing to grade")


class AISIGemmaRollouts(AISILlamaRollouts):
    """Gemma-3-27B-it's own lies and honest answers on the same AISI tasks, for the thesis's D3
    replicated on Gemma (the only other Liars' Bench model AISI covers). Its exclusion list is
    written by `aisi_probe.py audit --model gemma`."""

    base_name = "aisi_gemma"
    config = GEMMA_CONFIG
    exclusions = GEMMA_EXCLUSIONS

    def _get_dialogues(self) -> DialogueDataType:
        """Each message trimmed, as Gemma's chat template trims it before formatting (`| trim`):
        some replies end in whitespace, which the formatted text then lacks, and the detection
        mask is located by matching each message's text. The tokens are those the template gives
        either way."""
        dialogues, labels, metadata = super()._get_dialogues()
        for dialogue in dialogues:
            for message in dialogue:
                message.content = message.content.strip()
        return dialogues, labels, metadata
