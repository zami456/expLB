"""Post hoc (2026-10-23): the Liars' Bench rows Qwen3-32B reads for the Liars' Bench-labelled D4 pipeline.

User (2026-10-23): "using liars bench label after projection" -- D4's persona projection inside their pipeline on
Qwen3-32B (`qwen_d4_pipeline.py`), with the probe direction trained on Liars' Bench labels instead of Core
colluders. Reader Qwen3-32B; rows = the thesis's five Liars' Bench sets (user's choice): exactly the Llama-3.3-written
replies of CG, HP-C, HP-KR, IT and ID that `layer_sweep.pool` kept for `activations/pooled_L40` (the thesis D4 (LB)
training rows), rebuilt from the runner's experiment config and checked row by row against those files' labels.

Each row is rendered as their Core extractor renders a row (`extract_core._build_gen_text`: the dialogue's messages,
same-role turns merged as the runner merges them, through the Qwen3-32B chat template without a generation prompt)
and read by `reader/verdict_read.py` (unchanged) at two points, layers 26-30:
  point 0, THEIRS: the last token of the rendering -- the position their Core probe is trained on;
  point 1, REPLY_LAST: the reply's last token, before its <|im_end|> (reference only).
The design of the analysis is in `DESIGN`, declared with the rows before any read.
"""

import json
from datetime import datetime
from pathlib import Path

import fire
import numpy as np

from deception_detection.scripts.narcbench import freeze as FZ
from deception_detection.scripts.narcbench import qwen_d4_pipeline as QP
from deception_detection.scripts.narcbench import spec as S
from deception_detection.scripts.narcbench import verdict_spec as V

NAME = "qwen_lb_five"
ROWS_FILE = V.ROWS / f"{NAME}.jsonl"
META = V.ROWS / f"{NAME}_meta.json"
BASE = 100_000
POINTS = ("THEIRS", "REPLY_LAST")
MARK = "== POST HOC QWEN3-32B D4 PIPELINE, LIARS' BENCH LABELS"
DESIGN = ("the direction per layer and seed = their train_deception_probe, unchanged, on the projected Liars' Bench rows "
          "(HONEST/DECEPTIVE only; meta makes every row a public discussion row of a collusion run, deceptive = "
          "colluder; rows in one seeded random order, so its balancing truncates at random, not by set), at point "
          "THEIRS; it replaces their Core-trained direction everywhere in their compute_all_results (their module's "
          "train_deception_probe patched; Core CV folds get the same direction); Asymmetry and Sorted Concat "
          "classifiers still fit on Core runs, Activation Alignment uses no direction; Rose's label (Table 1) and "
          "Shi's; 10 seeds, mean +- their SE; basis = qwen3_d4_basis.npz (declared for qwen_d4_pipeline); null = 100 "
          "random orthonormal 3-dim subspaces per layer (qwen_d4_pipeline.random_subspace), Liars' Bench rows "
          "projected the same way and the direction refitted in each, seed 0, Core CV skipped; dev check on Liars' "
          "Bench only: leave-one-set-out AUROC of the projected direction (seed 0, layer mean); REPLY_LAST reported "
          "as reference; descriptive, no Holm")


def rows() -> None:
    from transformers import AutoTokenizer

    from deception_detection.scripts import layer_sweep as LS
    from deception_detection.scripts.k_robustness import FIVE
    from deception_detection.utils import preprocess_dialogue

    cfg, experiment = LS._experiment()
    flags = dict(detect_all=cfg.detect_all_tokens, detect_only_start_of_turn=cfg.detect_only_start_of_turn,
                 detect_only_last_token=cfg.detect_only_last_token)
    by_name = {d.name: d for d in [experiment.train_dataset, *experiment.eval_datasets, *experiment.control_datasets]}
    tok = AutoTokenizer.from_pretrained(V.QWEN_MODEL)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    out, meta, summary, rid = [], {}, {}, BASE
    for name in FIVE:
        ds = by_name[name]
        _, _, masks, labels = LS.tokenize_for_pool(ds, experiment.tokenizer, flags, None, None)
        kept = [i for i in range(len(ds)) if all(bool(m[i].any()) for m in masks)]
        with np.load(LS.pooled_dir(QP.S.HEADLINE_LAYER) / f"{name}.npz") as z:
            thesis = [str(x) for x in z["labels"]]
        assert [labels[""][i] for i in kept] == thesis, f"{name}: not the rows of the thesis's pooled file"
        lens = []
        for i in kept:
            msgs = preprocess_dialogue(ds.dialogues[i])
            assert msgs[-1]["role"] == "assistant", f"{name} row {i}: the last turn is {msgs[-1]['role']}"
            ids = tok(tok.apply_chat_template(msgs, tokenize=False))["input_ids"]
            last_end = max(j for j, t in enumerate(ids) if t == im_end)
            out.append({"row_id": rid, "ids": ids, "groups": [], "points": [len(ids) - 1, last_end - 1]})
            meta[str(rid)] = {"set": name, "index": i, "label": labels[""][i]}
            lens.append(len(ids))
            rid += 1
        summary[name] = {"rows": len(kept), "deceptive": thesis.count("DECEPTIVE"), "honest": thesis.count("HONEST"),
                         "other": len(thesis) - thesis.count("DECEPTIVE") - thesis.count("HONEST"),
                         "tokens_median": float(np.median(lens)), "tokens_max": int(max(lens)),
                         "tokens_total": int(sum(lens))}
    with open(ROWS_FILE, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    META.write_text(json.dumps({"points": POINTS, "summary": summary, "rows": meta}), encoding="utf-8")
    first = out[0]
    print(json.dumps(summary, indent=1))
    print("tail of row", first["row_id"], repr(tok.decode(first["ids"][-6:])), "| point THEIRS",
          repr(tok.decode([first["ids"][first["points"][0]]])), "| point REPLY_LAST",
          repr(tok.decode([first["ids"][first["points"][1]]])))


def declare() -> None:
    lines = [f"{MARK} (declared before any Liars' Bench read or score)  {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"qwen_lb_rows.py sha256 {FZ.sha256(Path(__file__))}",
             f"{ROWS_FILE.name} sha256 {FZ.sha256(ROWS_FILE)}",
             f"{META.name} sha256 {FZ.sha256(META)}",
             f"qwen3_d4_basis.npz sha256 {FZ.sha256(QP.BASIS)}",
             "reader: reader/verdict_read.py unchanged (deployed copy, freeze V), --model qwen --batched, points "
             f"{', '.join(POINTS)} at layers {QP.LAYERS}",
             f"design: {DESIGN}", ""]
    with open(S.LOGS / "freeze.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    fire.Fire({"rows": rows, "declare": declare})
