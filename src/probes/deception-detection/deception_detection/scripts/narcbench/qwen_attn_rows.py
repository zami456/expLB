"""Post hoc (2026-10-24): rows and basis for the attention probe on persona-projected tokens (Qwen3-32B).

User (2026-10-24): an EleutherAI-style attention probe (https://blog.eleuther.ai/attention-probes/) on D4's persona
projection instead of a last-token or mean readout, trained on Core colluders (their recipe), attending over the
scored turn's own tokens, 8 heads with 1 and 2 as reference. Plan: `C:/Users/zamig/.claude/plans/
ethereal-inventing-flask.md`.

Each row is read whole by Qwen3-32B (`reader/seq_read.py`); only the tokens of its spans come back, projected on
NDIF onto `BASIS_FILE`: per layer 26-30, the 24 extended persona PCs (nested; the first 3 = the frozen D4 basis)
and 100 random orthonormal 3-dim null subspaces (`qwen_d4_pipeline.random_subspace`), 324 rows.

- core: their released Qwen3-32B Core metadata, public discussion rows (the only Core rows their probe and methods
  use), each matched (`rows.align`) to our row of the released Core transcripts and rendered as their extractor
  renders it (system, user, assistant = the reply; Qwen3-32B template). Their Qwen3 Core activations come from a
  sampling whose transcripts are not released (G1: 4 of 1,054 token counts match), so these are the same runs,
  agents, roles and labels with the released text, not their exact inputs. Span: the reply's tokens to the last
  token of the rendering (their read position).
- refl: `qwen_reflection` rows unchanged; span: the reflection turn's own tokens to their point.
- verdict: `qwen_verdict` rows unchanged; two spans: (0) the agent's last own turn before the verdict-phase prompt,
  to the newline after its <|im_end|> -- used for pre_decision and verdict_start, whose points sit in that prompt,
  before the agent's verdict turn; (1) the verdict turn's own tokens to the decision token.
"""

import json
from datetime import datetime
from pathlib import Path

import fire
import numpy as np

from deception_detection.scripts.narcbench import freeze as FZ
from deception_detection.scripts.narcbench import qwen_d4_ksweep as QK
from deception_detection.scripts.narcbench import qwen_d4_pipeline as QP
from deception_detection.scripts.narcbench import rows as R
from deception_detection.scripts.narcbench import spec as S
from deception_detection.scripts.narcbench import verdict_spec as V

NAMES = {"core": "qwen_attn_core", "refl": "qwen_attn_refl", "verdict": "qwen_attn_verdict"}
BASE = {"core": 110_000, "refl": 120_000, "verdict": 130_000}
SOURCE = {"refl": "qwen_reflection", "verdict": "qwen_verdict"}
SPANS = {"core": ("reply",), "refl": ("reflection",), "verdict": ("last_own_turn", "verdict_turn")}
BASIS_FILE = S.FROZEN / "qwen_attn_basis.npz"
LAYERS = QP.LAYERS
N_PCS = 24
N_NULL = 100
MARK = "== POST HOC ATTENTION PROBE ON PERSONA-PROJECTED TOKENS, ROWS AND BASIS"


class Turns:
    """Chat-turn positions in Qwen3 token ids."""

    def __init__(self, tok):
        self.tok = tok
        self.start, self.end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
        self.think, self.unthink = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
        self.assistant = tok("assistant", add_special_tokens=False)["input_ids"]

    def starts(self, ids: list[int], before: int) -> list[int]:
        return [j for j in range(before + 1) if ids[j] == self.start]

    def is_assistant(self, ids: list[int], s: int) -> bool:
        return ids[s + 1:s + 1 + len(self.assistant)] == self.assistant

    def content(self, ids: list[int], s: int) -> int:
        """The first token of an assistant turn's own text: after the role line and any (empty) think block."""
        c = s + 1 + len(self.assistant) + 1  # role, then "\n"
        if c < len(ids) and ids[c] == self.think:
            c = ids.index(self.unthink, c) + 1
            while c < len(ids) and ids[c] != self.end and not self.tok.decode([ids[c]]).strip():
                c += 1
        return c

    def span_to(self, ids: list[int], point: int) -> list[int]:
        """[a, b): the own tokens of the assistant turn holding `point`, to `point` inclusive."""
        s = max(self.starts(ids, point))
        assert self.is_assistant(ids, s), f"point {point} is not in an assistant turn"
        return [self.content(ids, s), point + 1]

    def last_own_turn(self, ids: list[int], before: int) -> list[int]:
        """[a, b): the last assistant turn starting before `before`, to the newline after its <|im_end|>."""
        s = max(j for j in self.starts(ids, before) if self.is_assistant(ids, j))
        e = ids.index(self.end, s)
        return [self.content(ids, s), e + 2]


def _tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(V.QWEN_MODEL)


def core_rows(tok, turns: Turns) -> tuple[list[dict], list[dict], dict]:
    meta = R.released_meta("core", "qwen3_32b")
    ours = [r for r in R.load_rows("core") if r["rose_row"]]
    pos = R.align(ours, meta)
    out, their_meta, bad = [], [], []
    for j, (p, m) in enumerate(zip(pos, meta)):
        if m["phase"] != "discussion" or m["channel"] != "public":
            continue
        assert p is not None, f"released Core row {j} has no transcript row"
        r = ours[p]
        assert (r["role"], r["mode"]) == (m["role"], m["mode"]), f"released Core row {j}: role or mode differs"
        text = tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]},
                                        {"role": "assistant", "content": r["reply"]}], tokenize=False)
        ids = [int(x) for x in tok(text)["input_ids"]]
        span = turns.span_to(ids, len(ids) - 1)
        own = tok.decode(ids[span[0]:ids.index(turns.end, span[0])]).strip()
        if own != r["reply"].strip():
            bad.append(j)
        out.append({"row_id": BASE["core"] + len(out), "ids": ids, "spans": [span]})
        their_meta.append(m | {"meta_index": j, "transcript_row": r["row_id"]})
    return out, their_meta, {"rows": len(out), "reply_decode_mismatch": bad}


def verdict_rows(kind: str, turns: Turns) -> tuple[list[dict], list[dict], dict]:
    out, meta = [], []
    stats = {"rows": 0, "no_own_turn_before_verdict": 0}
    with open(V.ROWS / f"{SOURCE[kind]}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids, pts = r["ids"], r["points"]
            if kind == "refl":
                spans = [turns.span_to(ids, pts[0])]
            else:
                vs, pre, dec = pts  # verdict_start, pre_decision, decision (verdict_rows.py)
                try:
                    prev = turns.last_own_turn(ids, vs)
                except ValueError:
                    prev, stats["no_own_turn_before_verdict"] = [0, 0], stats["no_own_turn_before_verdict"] + 1
                spans = [prev, turns.span_to(ids, dec)]
            out.append({"row_id": BASE[kind] + len(out), "ids": ids, "spans": spans})
            meta.append({k: r[k] for k in ("row_id", "rep", "episode_index", "agent", "run_id")} | {"points": pts})
    stats["rows"] = len(out)
    return out, meta, stats


def basis() -> dict[int, np.ndarray]:
    pb, _ = QK.persona_bases(N_PCS)
    frozen = QP.load_basis()
    out = {}
    for L in LAYERS:
        assert np.allclose(pb[L][:3], frozen[L], atol=1e-10), "the first 3 PCs are not the frozen basis"
        out[L] = np.concatenate([pb[L]] + [QP.random_subspace(i, L) for i in range(N_NULL)]).astype(np.float32)
    return out


def build() -> None:
    tok = _tok()
    turns = Turns(tok)
    summary = {}
    for kind in NAMES:
        rows, meta, stats = core_rows(tok, turns) if kind == "core" else verdict_rows(kind, turns)
        lens = np.array([[b - a for a, b in r["spans"]] for r in rows])
        stats |= {"spans": SPANS[kind], "span_tokens_median": np.median(lens, 0).tolist(),
                  "span_tokens_max": lens.max(0).tolist(), "span_tokens_total": int(lens.sum()),
                  "context_tokens_total": int(sum(len(r["ids"]) for r in rows)),
                  "store_gb_float32": round(float(lens.sum()) * len(LAYERS) * (N_PCS + 3 * N_NULL) * 4 / 1e9, 2)}
        with open(V.ROWS / f"{NAMES[kind]}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        (V.ROWS / f"{NAMES[kind]}_meta.json").write_text(json.dumps({"spans": SPANS[kind], "stats": stats, "rows": meta}),
                                                        encoding="utf-8")
        summary[kind] = stats
        r0 = rows[0]
        print(kind, json.dumps({k: v for k, v in stats.items() if k != "reply_decode_mismatch"}),
              "| mismatches", len(stats.get("reply_decode_mismatch", [])))
        for a, b in r0["spans"]:
            print("   span head", repr(tok.decode(r0["ids"][a:a + 12])), "| tail", repr(tok.decode(r0["ids"][b - 6:b])))
    B = basis()
    np.savez(BASIS_FILE, **{f"B|L{L}": B[L] for L in LAYERS})
    print("basis", BASIS_FILE, {L: B[L].shape for L in LAYERS})


def declare() -> None:
    files = [V.ROWS / f"{n}{s}" for n in NAMES.values() for s in (".jsonl", "_meta.json")]
    lines = [f"{MARK} (declared before any read or score)  {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"qwen_attn_rows.py sha256 {FZ.sha256(Path(__file__))}",
             *[f"{f.name} sha256 {FZ.sha256(f)}" for f in files],
             f"{BASIS_FILE.name} sha256 {FZ.sha256(BASIS_FILE)}",
             f"qwen3_d4_basis.npz sha256 {FZ.sha256(QP.BASIS)}",
             f"seq_read.py sha256 {FZ.sha256(Path('E:/narcbench/verdict/reader/seq_read.py'))} (deployed; "
             f"verdict_read.py unchanged, sha256 {FZ.sha256(Path('E:/narcbench/verdict/reader/verdict_read.py'))})",
             "design: as in the plan of 2026-10-24 (attention probe, Core colluders only, scored turn's own tokens, "
             "H = 8 with 1 and 2 as reference, k = 3 headline); reader reader/seq_read.py projects each span token "
             "on NDIF onto the 324 rows of the basis (24 persona PCs + 100 random 3-dim nulls), float16 upload; "
             "declared fallback if the smoke shows the upload refused or too slow: the first 30 null subspaces; Core "
             "text = the released transcripts (their Qwen3 Core inputs are unreleased)", ""]
    with open(S.LOGS / "freeze.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    fire.Fire({"build": build, "declare": declare})
