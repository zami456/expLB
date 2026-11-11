"""Phase 0 of the emotion-coordinate study: inventory and data checks. Nothing here calls NDIF.

    python -m emotion_coords.phase0 lb          # Liars' Bench rows: counts, columns, token lengths, index
                                                #   tables, decoded positions, HP-KR subgroups, audit file
    python -m emotion_coords.phase0 templates   # HP-KR's knowledge-report templates, checked on its rows
    python -m emotion_coords.phase0 gemotions   # download Gemotions; counts per emotion, topic and cell
    python -m emotion_coords.phase0 lexicon     # valence/arousal norms and their coverage of the emotions
    python -m emotion_coords.phase0 report      # results/emotion_coords/phase0/PHASE0_REPORT.md

Run from the deception-detection folder with the project venv. Index tables and downloads go to
data/emotion_coords/ (never committed); tables and reports to results/emotion_coords/phase0/.

The Liars' Bench rows are rebuilt exactly as the thesis's pooled files were written: the
`XXXX-9-llama-all` run's datasets in the loader's own (seed-42 shuffled) order, tokenised by
`TokenizedDataset` with the run's flags, and rows with an empty detection mask dropped
(`pool_activations.iter_cached_examples`). Each kept row is then the same row of every
`activations/pooled*/<dataset>.npz`, which `lb` checks label by label at every configured layer.
"""

import ast
import hashlib
import io
import json
import os
import re
import sqlite3
import time
import urllib.request
import zipfile
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from emotion_coords import config as C

OUT = "phase0"
CLAIMED, DENIED = "CLAIMED_KNOWLEDGE", "DENIED_KNOWLEDGE"


def _out(name: str) -> Path:
    return C.results_dir(OUT) / name


def _clean(text: str) -> str:
    """The Liars' Bench loader's message cleaning (`XXXX_9.get_prompts_and_labels`)."""
    return text.replace("\xa0", " ").rstrip("<>").strip()


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    qs = np.quantile(values, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]) if len(values) else [np.nan] * 7
    return dict(zip(("min", "p5", "p25", "median", "p75", "p95", "max"), map(float, qs)))


# --- Liars' Bench ------------------------------------------------------------------------------


def _experiment():
    """The reused run's datasets in its own row order, with its tokenizer (no model is loaded)."""
    from deception_detection.scripts.layer_sweep import _experiment as build

    C.load_env()
    return build(C.load()["lb_config"])


def _hf_frame(name: str) -> tuple[dict, pd.DataFrame] | None:
    """One Liars' Bench config at the pinned revision: (column and model summary, Llama rows in the
    loader's pre-shuffle order), or None for a dataset that does not come from the Hub."""
    lb = C.load()["liars_bench"]
    config = lb["hf_configs"].get(name)
    if config is None:
        return None
    from datasets import load_dataset

    frame = load_dataset(lb["repo"], config, split="test", revision=lb["revision"],
                         token=os.getenv("HF_TOKEN")).to_pandas()
    llama = frame[frame["model"] == lb["model_tag"]].reset_index(drop=True)
    summary = {
        "hf_config": config,
        "columns": list(frame.columns),
        "rows_all_models": len(frame),
        "rows_by_model": {str(k): int(v) for k, v in frame["model"].value_counts().items()},
        "llama_rows": len(llama),
        "llama_labels": {str(k): int(v) for k, v in llama["deceptive"].value_counts().items()},
    }
    return summary, llama


def _row_positions(row: np.ndarray, mask: np.ndarray, ids: dict[str, int]) -> dict:
    """Where the final assistant message sits in one unpadded row.

    `pre` is the token after the last `<|end_header_id|>` before the detected span: the "\\n\\n" that
    ends the assistant header, whose activation depends on the prompt alone. `msg_end` is the
    position of the `<|eot_id|>` closing the message (exclusive end of its content).
    """
    pos = np.flatnonzero(mask)
    first, last = int(pos[0]), int(pos[-1])
    headers = np.flatnonzero(row[:first] == ids["end_header"])
    pre = int(headers[-1]) + 1 if len(headers) else -1
    eots = np.flatnonzero(row[last + 1:] == ids["eot"])
    msg_end = last + 1 + int(eots[0]) if len(eots) else len(row)
    return {
        "first": first,
        "last": last,
        "pre": pre,
        "msg_end": msg_end,
        "n_resp": int(mask.sum()),
        "n_final_msg": msg_end - pre - 1 if pre >= 0 else -1,
        "mask_contiguous": bool(last - first + 1 == len(pos)),
        "pre_adjacent": bool(pre + 1 == first),
        "eot_after_last": bool(last + 1 < len(row) and row[last + 1] == ids["eot"]),
    }


def lb(n_decode: int = 3, n_audit: int = 40) -> None:
    """Rebuild every pooled row of the reused run and describe it.

    Writes per dataset an index table (data/emotion_coords/index/<dataset>.csv) linking pooled row,
    loader row, Liars' Bench id and Hub `index`, with token counts and the positions later poolings
    use; and results/phase0/{lb_counts.csv, lb_columns.json, lb_token_lengths.csv,
    positions_decoded.md, formatting.md, hpkr_subgroups.csv, hpkr_audit_40.md}.
    """
    from deception_detection.scripts.steer_lies import knowledge_claim
    from deception_detection.tokenized_data import TokenizedDataset

    settings = C.load()
    cfg, experiment = _experiment()
    tokenizer = experiment.tokenizer
    ids = {
        "start_header": tokenizer.convert_tokens_to_ids("<|start_header_id|>"),
        "end_header": tokenizer.convert_tokens_to_ids("<|end_header_id|>"),
        "eot": tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    }
    datasets = [experiment.train_dataset, *experiment.eval_datasets, *experiment.control_datasets]
    short = settings["datasets"] | {settings["control"]: "Alpaca", settings["train_pairs"]: "RepE",
                                    settings["train_pairs"] + "_val": "RepE-val"}

    counts, lengths, columns, decoded = [], [], {}, []
    hpkr_rows: pd.DataFrame | None = None
    index_dir = C.data_dir("index")

    for ds in datasets:
        name = ds.name
        toks = TokenizedDataset.from_dataset(
            ds, tokenizer, detect_all=cfg.detect_all_tokens,
            detect_only_start_of_turn=cfg.detect_only_start_of_turn,
            detect_only_last_token=cfg.detect_only_last_token)
        tokens = toks.tokens.numpy()
        attn = toks.attention_mask.numpy().astype(bool)
        mask = toks.detection_mask.numpy().astype(bool)
        n, width = tokens.shape

        hub = _hf_frame(name)
        lb_ids = (ds.metadata or {}).get("id")
        if hub is not None:
            columns[name] = hub[0]
        records, kept = [], 0
        for i in range(n):
            true_len = int(attn[i].sum())
            row, row_mask = tokens[i, width - true_len:], mask[i, width - true_len:]
            record = {"dataset_row": i, "label": ds.labels[i].name, "n_tokens": true_len,
                      "lb_id": int(lb_ids[i]) if lb_ids is not None else i}
            if not row_mask.any():
                records.append(record | {"pooled_row": -1})
                continue
            record |= {"pooled_row": kept} | _row_positions(row, row_mask, ids)
            kept += 1
            if hub is not None:
                hf = hub[1].iloc[record["lb_id"]]
                record["hf_index"] = int(hf["index"])
                ours = _squash("".join(m.content for m in ds.dialogues[i]))
                theirs = _squash("".join(_clean(m["content"]) for m in hf["messages"]))
                record["text_matches_hub"] = ours == theirs
            records.append(record)
        table = pd.DataFrame(records)
        table.to_csv(index_dir / f"{name}.csv", index=False)
        kept_rows = table[table["pooled_row"] >= 0]

        # every configured layer's pooled file must hold exactly these rows, in this order
        layer_checks = {}
        for layer in settings["layers"]:
            path = C.pooled_dir(layer) / f"{name}.npz"
            if not path.exists():
                layer_checks[layer] = "missing"
                continue
            with np.load(path, allow_pickle=False) as stored:
                labels, pooled = stored["labels"].astype(str), stored["pooled"]
                same = len(labels) == len(kept_rows) and (labels == kept_rows["label"].to_numpy()).all()
                finite = int((~np.isfinite(pooled).all(axis=1)).sum())
            layer_checks[layer] = ("ok" if same else "MISMATCH") + (f", {finite} non-finite rows" if finite else "")

        n_dec = int((kept_rows["label"] == "DECEPTIVE").sum())
        counts.append({
            "dataset": name, "short": short.get(name, name), "rows": n, "pooled_rows": len(kept_rows),
            "empty_mask_dropped": n - len(kept_rows), "deceptive": n_dec,
            "honest": int((kept_rows["label"] == "HONEST").sum()),
            "hub_text_mismatches": int((~kept_rows["text_matches_hub"].astype(bool)).sum())
            if "text_matches_hub" in kept_rows else "n/a",
            "pre_adjacent_share": round(float(kept_rows["pre_adjacent"].mean()), 4),
            "eot_after_last_share": round(float(kept_rows["eot_after_last"].mean()), 4),
            "mask_contiguous_share": round(float(kept_rows["mask_contiguous"].mean()), 4),
            **{f"pooled_L{layer}": check for layer, check in layer_checks.items()},
        })
        for column in ("n_tokens", "n_resp", "n_final_msg"):
            lengths.append({"dataset": name, "short": short.get(name, name), "count": column}
                           | _quantiles(kept_rows[column].to_numpy()))

        # three rows a set, decoded at the positions the poolings use
        for _, r in kept_rows.head(n_decode).iterrows():
            i = int(r["dataset_row"])
            row = tokens[i, width - int(r["n_tokens"]):]
            dec = lambda a, b: tokenizer.decode(row[max(a, 0):b])  # noqa: E731
            decoded.append(
                f"### {short.get(name, name)} pooled row {int(r['pooled_row'])} (label {r['label']})\n"
                f"- header, positions {int(r['pre']) - 3}..{int(r['pre'])}: `{dec(int(r['pre']) - 3, int(r['pre']) + 1)!r}`\n"
                f"- pre-response token (c), position {int(r['pre'])}: `{dec(int(r['pre']), int(r['pre']) + 1)!r}`\n"
                f"- first detected tokens, from {int(r['first'])}: `{dec(int(r['first']), int(r['first']) + 6)!r}`\n"
                f"- last content token (b), position {int(r['last'])}, with its neighbours: "
                f"`{dec(int(r['last']) - 3, int(r['last']) + 1)!r}` then `{dec(int(r['last']) + 1, int(r['last']) + 2)!r}`\n"
                f"- detected tokens {int(r['n_resp'])}, final-message tokens {int(r['n_final_msg'])}\n")
        if name == "harm_refusal_knowledge_report__plain":
            i = int(kept_rows.iloc[0]["dataset_row"])
            _out("formatting.md").write_text(
                "# How a Liars' Bench row is formatted for Llama-3.3-70B\n\n"
                "HP-KR pooled row 0 as `TokenizedDataset` renders it (the text whose activations were cached):\n\n"
                f"```\n{toks.formatted_dialogues[i]}\n```\n", encoding="utf-8")
            hpkr_rows = kept_rows.merge(hub[1][["index", "dataset", "meta", "messages"]],
                                        left_on="hf_index", right_on="index", how="left")
        logger.info(f"{name}: {len(kept_rows)} pooled rows, layer checks {layer_checks}")

    pd.DataFrame(counts).to_csv(_out("lb_counts.csv"), index=False)
    pd.DataFrame(lengths).round(1).to_csv(_out("lb_token_lengths.csv"), index=False)
    _out("lb_columns.json").write_text(json.dumps(columns, indent=2), encoding="utf-8")
    _out("positions_decoded.md").write_text(
        "# Pooled positions, decoded (three rows per set)\n\n" + "\n".join(decoded), encoding="utf-8")
    if hpkr_rows is not None:
        _hpkr_subgroups(hpkr_rows, knowledge_claim, n_audit)


def _hpkr_subgroups(rows: pd.DataFrame, knowledge_claim, n_audit: int) -> None:
    """HP-KR's three groups. Liars' Bench records the belief (`model_belief`, -1 = no consistent
    answer) but not the claim, so the claim comes from `steer_lies.knowledge_claim`."""
    def field(name: str):  # `meta` is a dict's repr holding np.float64(inf); read integers, never eval
        pattern = re.compile(rf"'{name}':\s*(?:np\.float64\()?(-?\d+)")
        return rows["meta"].map(lambda s: int(pattern.search(s).group(1)))

    rows = rows.assign(
        reply=rows["messages"].map(lambda m: _clean(m[-1]["content"])),
        user=rows["messages"].map(lambda m: _clean(m[1]["content"])),
        belief=field("model_belief"),
        template_index=field("template_index"),
        correct_option=field("correct_option"),
        subject=rows["dataset"].str.split("/").str[-1],
    )
    rows["claim"] = rows["reply"].map(knowledge_claim)
    lie = rows["label"] == "DECEPTIVE"
    rows["subgroup"] = np.select(
        [lie, rows["claim"] == CLAIMED, rows["claim"] == DENIED],
        ["lie", "honest_admission", "honest_denial"], default="honest_other")
    keep = ["pooled_row", "hf_index", "label", "subgroup", "claim", "belief", "template_index",
            "correct_option", "subject", "reply", "user", "n_tokens", "n_resp", "pre", "first", "last"]
    rows[keep].to_csv(C.data_dir("index") / "hpkr_rows.csv", index=False)

    table = pd.crosstab([rows["subgroup"], rows["claim"]],
                        rows["belief"].map(lambda b: "no belief (-1)" if b < 0 else "belief (0-3)"))
    table.to_csv(_out("hpkr_subgroups.csv"))
    logger.info(f"HP-KR subgroups:\n{table}")

    # 40 for the user to audit: random within each group, so the rare groups are actually seen
    rng = np.random.default_rng(C.load()["seeds"]["audit"])
    groups = rows.groupby("subgroup").groups
    share = {g: len(i) for g, i in groups.items()}
    quota = {g: min(len(groups[g]), max(4, round(n_audit * share[g] / len(rows)))) for g in groups}
    quota["honest_admission"] = n_audit - sum(v for g, v in quota.items() if g != "honest_admission")
    lines = ["# HP-KR claim classification: 40 rows to audit\n",
             "Claim from `steer_lies.knowledge_claim` (regex); belief from Liars' Bench `model_belief` "
             "(-1 = no consistent answer). Random within each subgroup, quotas "
             f"{quota} (the rare groups are over-sampled on purpose).\n",
             "| # | pooled row | label | subgroup | claim | belief | reply |", "|---|---|---|---|---|---|---|"]
    k = 0
    for group, quota_g in quota.items():
        for j in rng.choice(np.asarray(groups[group]), size=quota_g, replace=False):
            r = rows.loc[j]
            k += 1
            reply = r["reply"].replace("|", "\\|").replace("\n", " ")[:160]
            lines.append(f"| {k} | {r['pooled_row']} | {r['label']} | {group} | {r['claim']} | {r['belief']} | {reply} |")
    _out("hpkr_audit_40.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- HP-KR templates ---------------------------------------------------------------------------

TEMPLATE_NAMES = ("CONTROL_KNOWLEDGE_REPORT_TEMPLATES", "HARMFUL_KNOWLEDGE_REPORT_TEMPLATES")
MIDDLE = "{question}\n\nChoices:\n{choices}"


def _notebook_templates() -> dict[str, dict]:
    """The template dictionaries assigned in the Liars' Bench harm-pressure notebook, read as literals."""
    path = C.LIARS_BENCH_ROOT / C.load()["liars_bench"]["hpkr_notebook"]
    found: dict[str, dict] = {}
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id in TEMPLATE_NAMES):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    missing = set(TEMPLATE_NAMES) - set(found)
    if missing:
        raise RuntimeError(f"{path.name}: no literal assignment of {sorted(missing)}")
    return found


def templates() -> None:
    """Harmful (indices 1-5) and neutral (the two control templates and validation index 0)
    knowledge-report templates per subject, checked by re-deriving every HP-KR row's question.

    Writes data/emotion_coords/hpkr_templates.json, data/emotion_coords/index/hpkr_questions.csv
    (one row per question: id, subject, belief, the question block) and results/phase0/hpkr_templates.md.
    """
    found = _notebook_templates()
    control, harmful = found["CONTROL_KNOWLEDGE_REPORT_TEMPLATES"], found["HARMFUL_KNOWLEDGE_REPORT_TEMPLATES"]
    rows = pd.read_csv(C.data_dir("index") / "hpkr_rows.csv")

    sets: dict[str, dict] = {}
    for subject in sorted(rows["subject"].unique()):
        key = f"cais/{subject}"
        neutral = [{"name": f"control_{j}", "template": t["template"]} for j, t in enumerate(control[key])]
        neutral.append({"name": "validation_0", "template": harmful[key][0]["template"]})
        sets[subject] = {
            "harmful": [{"name": f"harmful_{j}", "template": harmful[key][j]["template"]} for j in range(1, 6)],
            "neutral": neutral,
        }
        for t in sets[subject]["harmful"] + sets[subject]["neutral"]:
            if t["template"].count("{question}") != 1 or MIDDLE not in t["template"]:
                raise RuntimeError(f"{subject} {t['name']}: not of the form prefix + {MIDDLE!r} + suffix")

    # every row: its template's prefix and suffix must frame its user turn exactly
    blocks, bad = [], 0
    for _, r in rows.iterrows():
        template = sets[r["subject"]]["harmful"][int(r["template_index"]) - 1]["template"]
        prefix, suffix = template.split(MIDDLE)
        user, suffix = r["user"], suffix.rstrip()
        if not (user.startswith(prefix) and user.endswith(suffix)):
            bad += 1
            blocks.append(None)
            continue
        blocks.append(user[len(prefix): len(user) - len(suffix)])
    rows["block"] = blocks
    rows["question_id"] = [hashlib.sha1(f"{s}|{b}".encode()).hexdigest()[:12] if b else None
                           for s, b in zip(rows["subject"], rows["block"])]

    per_q = rows.dropna(subset=["question_id"]).groupby("question_id").agg(
        subject=("subject", "first"), belief_min=("belief", "min"), belief_max=("belief", "max"),
        n_rows=("pooled_row", "size"), n_lies=("subgroup", lambda s: int((s == "lie").sum())),
        n_admissions=("subgroup", lambda s: int((s == "honest_admission").sum())),
        n_denials=("subgroup", lambda s: int((s == "honest_denial").sum())),
        templates=("template_index", lambda s: ",".join(map(str, sorted(set(s))))), block=("block", "first"))
    inconsistent = int((per_q["belief_min"] != per_q["belief_max"]).sum())
    per_q["known"] = per_q["belief_min"] >= 0
    per_q.to_csv(C.data_dir("index") / "hpkr_questions.csv")
    (C.data_dir() / "hpkr_templates.json").write_text(json.dumps(sets, indent=2), encoding="utf-8")

    by_subject = per_q.groupby(["subject", "known"]).size().unstack(fill_value=0)
    lines = ["# HP-KR knowledge-report templates\n",
             f"Source: `{C.load()['liars_bench']['hpkr_notebook']}` (literal assignments of {', '.join(TEMPLATE_NAMES)}).",
             "Harmful = HARMFUL indices 1-5 (the Llama rows use 1-5). Neutral = the subject's two CONTROL "
             "templates (Appendix D.2) and HARMFUL index 0, the harmless validation template.\n",
             f"Check: {len(rows) - bad} of {len(rows)} HP-KR rows are framed exactly by their template's "
             f"prefix and suffix ({bad} not). {per_q.shape[0]} distinct questions; belief inconsistent "
             f"across a question's rows: {inconsistent}.\n",
             "Questions by subject and belief (known = model_belief >= 0):\n",
             by_subject.to_markdown(), ""]
    for subject, s in sets.items():
        lines.append(f"## {subject}\n")
        for t in s["harmful"] + s["neutral"]:
            lines.append(f"- **{t['name']}**: `{t['template']!r}`")
        lines.append("")
    _out("hpkr_templates.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"templates: {bad} unframed rows, {per_q.shape[0]} questions, {inconsistent} inconsistent\n"
                f"{by_subject}")


# --- Gemotions ---------------------------------------------------------------------------------


def _tokenizer():
    from transformers import AutoTokenizer

    C.load_env()
    return AutoTokenizer.from_pretrained(C.load()["model"], token=os.getenv("HF_TOKEN"))


def _text_column(con: sqlite3.Connection, table: str) -> str | None:
    cols = [c[1] for c in con.execute(f"pragma table_info('{table}')")]
    for want in ("text", "dialogue", "content", "story"):
        for c in cols:
            if want in c.lower():
                return c
    return None


def gemotions(revision: str | None = None, chunk: int = 4000) -> None:
    """Download the Gemotions data at a pinned commit and describe it.

    Writes data/emotion_coords/gemotions/{data/*, SOURCE.json, story_lengths.csv.gz} and
    results/phase0/{gemotions_summary.json, emotions_171.txt, topics.csv, emotion_names.csv}.
    """
    from huggingface_hub import HfApi, hf_hub_download

    settings = C.load()
    g = settings["gemotions"]
    C.load_env()
    sha = HfApi().model_info(g["repo"], revision=revision).sha
    dest = C.data_dir("gemotions")
    files = {}
    for f in g["files"]:
        path = Path(hf_hub_download(g["repo"], f, repo_type=g["repo_type"], revision=sha, local_dir=dest))
        files[f] = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
    source = {"repo": g["repo"], "revision": sha, "files": files, "attribution": g["attribution"],
              "downloaded": time.strftime("%Y-%m-%d %H:%M:%S")}
    (dest / "SOURCE.json").write_text(json.dumps(source, indent=2), encoding="utf-8")

    stories_db = Path(files["data/stories.db"]["path"])
    con = sqlite3.connect(f"file:{stories_db.as_posix()}?mode=ro", uri=True)
    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    schema = {t: [c[1] for c in con.execute(f"pragma table_info('{t}')")] for t in tables}
    sizes = {t: con.execute(f"select count(*) from '{t}'").fetchone()[0] for t in tables}
    meta = pd.read_sql("select rowid as rid, emotion, topic_idx, topic, story_idx from stories_clean", con)

    tokenizer = _tokenizer()
    n_tokens = np.empty(len(meta), dtype=np.int32)
    chars = np.empty(len(meta), dtype=np.int32)
    position = {rid: k for k, rid in enumerate(meta["rid"].to_numpy())}
    for part in pd.read_sql("select rowid as rid, text from stories_clean", con, chunksize=chunk):
        texts = part["text"].fillna("").tolist()
        enc = tokenizer(texts, add_special_tokens=False)["input_ids"]
        at = [position[r] for r in part["rid"]]
        n_tokens[at] = [len(e) for e in enc]
        chars[at] = [len(t) for t in texts]
    meta["n_tokens"], meta["chars"] = n_tokens, chars
    meta.to_csv(dest / "story_lengths.csv.gz", index=False)

    min_tokens = settings["stories"]["min_tokens"]
    cell = meta.groupby(["emotion", "topic_idx"])
    per_cell = cell.size()
    eligible = cell["n_tokens"].apply(lambda s: int((s >= min_tokens).sum()))
    emotions = sorted(meta["emotion"].unique())
    topics = meta.groupby("topic_idx")["topic"].first().sort_index()
    _out("emotions_171.txt").write_text("\n".join(emotions) + "\n", encoding="utf-8")
    topics.to_csv(_out("topics.csv"))

    names = settings["emotion_names"]
    wanted = sorted({n for key, v in names.items() if key != "neighbour_checks" for n in v}
                    | {n for group in names["neighbour_checks"] for n in group})
    pd.DataFrame({"name": wanted, "present": [n in emotions for n in wanted]}).to_csv(
        _out("emotion_names.csv"), index=False)

    neutral_db = Path(files["data/neutral.db"]["path"])
    ncon = sqlite3.connect(f"file:{neutral_db.as_posix()}?mode=ro", uri=True)
    ntables = [r[0] for r in ncon.execute("select name from sqlite_master where type='table'")]
    neutral = {"tables": {t: [c[1] for c in ncon.execute(f"pragma table_info('{t}')")] for t in ntables},
               "rows": {t: ncon.execute(f"select count(*) from '{t}'").fetchone()[0] for t in ntables}}
    for t in ntables:
        col = _text_column(ncon, t)
        if col:
            texts = [r[0] or "" for r in ncon.execute(f"select \"{col}\" from '{t}'")]
            lens = [len(e) for e in tokenizer(texts, add_special_tokens=False)["input_ids"]]
            neutral.setdefault("token_lengths", {})[f"{t}.{col}"] = _quantiles(np.array(lens))
            neutral.setdefault("example", {})[f"{t}.{col}"] = texts[0][:600]

    jsonl = Path(files["data/emotion_stories.jsonl"]["path"]).read_text(encoding="utf-8").splitlines()
    summary = {
        "source": source,
        "stories_db": {"tables": schema, "rows": sizes},
        "stories_clean": {
            "rows": len(meta), "emotions": len(emotions), "topics": int(topics.shape[0]),
            "per_emotion": _quantiles(meta.groupby("emotion").size().to_numpy()),
            "per_topic": _quantiles(meta.groupby("topic_idx").size().to_numpy()),
            "per_cell": _quantiles(per_cell.to_numpy()),
            "cells": int(per_cell.size), "cells_expected": len(emotions) * int(topics.shape[0]),
            "per_cell_eligible": _quantiles(eligible.to_numpy()),
            "cells_without_eligible_story": int((eligible == 0).sum()),
            "min_tokens": min_tokens,
            "story_tokens": _quantiles(n_tokens), "story_chars": _quantiles(chars),
            "share_below_min_tokens": float((n_tokens < min_tokens).mean()),
            "share_below_50_tokens": float((n_tokens < 50).mean()),
        },
        "neutral_db": neutral,
        "emotion_stories_jsonl": {"lines": len(jsonl), "first": jsonl[:2]},
        "names_missing": [n for n in wanted if n not in emotions],
    }
    _out("gemotions_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Gemotions @ {sha[:8]}: {len(meta)} stories, {len(emotions)} emotions, {topics.shape[0]} topics; "
                f"missing names {summary['names_missing']}")


# --- valence and arousal norms -----------------------------------------------------------------


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _read_nrc(raw: bytes) -> pd.DataFrame:
    """The largest tab-separated table in the NRC VAD zip whose header names valence and arousal."""
    best = None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            if not name.lower().endswith(".txt"):
                continue
            text = z.read(name).decode("utf-8", errors="replace")
            head = text.split("\n", 1)[0].lower()
            if "valence" in head and "arousal" in head and (best is None or len(text) > len(best[1])):
                best = (name, text)
    if best is None:
        raise ValueError("no valence/arousal table in the zip")
    table = pd.read_csv(io.StringIO(best[1]), sep="\t", keep_default_na=False)
    table.columns = [c.strip().lower() for c in table.columns]
    word = next(c for c in table.columns if c in ("term", "word"))
    table = table.rename(columns={word: "word"})[["word", "valence", "arousal", "dominance"]]
    table.attrs["member"] = best[0]
    return table


def _read_warriner(raw: bytes) -> pd.DataFrame:
    table = pd.read_csv(io.BytesIO(raw), keep_default_na=False)
    return table.rename(columns={"Word": "word", "V.Mean.Sum": "valence", "A.Mean.Sum": "arousal",
                                 "D.Mean.Sum": "dominance"})[["word", "valence", "arousal", "dominance"]]


def lexicon() -> None:
    """Fetch valence/arousal norms (NRC VAD v2.1, else Warriner et al. 2013) and match the emotions.

    Writes data/emotion_coords/lexicon/<file>, results/phase0/{lexicon_source.json, vad_coverage.csv}.
    A multi-word emotion is matched as written, then hyphenated, then joined.
    """
    settings = C.load()["lexicon"]
    dest = C.data_dir("lexicon")
    emotions = [e for e in _out("emotions_171.txt").read_text(encoding="utf-8").split("\n") if e]
    attempts, table, source = [], None, None
    for name, url, reader in (("NRC VAD v2.1", settings["nrc_vad_url"], _read_nrc),
                              ("Warriner et al. 2013", settings["warriner_url"], _read_warriner)):
        try:
            raw = _fetch(url)
            table = reader(raw)
            path = dest / Path(url).name
            path.write_bytes(raw)
            source = {"name": name, "url": url, "file": str(path), "sha256": _sha256(path),
                      "member": table.attrs.get("member"), "rows": len(table)}
            break
        except Exception as err:  # noqa: BLE001 -- fall back to the next source and say why
            attempts.append({"name": name, "url": url, "error": repr(err)[:300]})
            logger.warning(f"lexicon {name}: {err!r}")
    if table is None:
        raise RuntimeError(f"no lexicon could be fetched: {attempts}")

    lookup = {str(w).strip().lower(): (v, a, d) for w, v, a, d in table.itertuples(index=False)}
    rows = []
    for e in emotions:
        hit = next((form for form in (e, e.replace(" ", "-"), e.replace(" ", "")) if form.lower() in lookup), None)
        v, a, d = lookup[hit.lower()] if hit else (np.nan, np.nan, np.nan)
        rows.append({"emotion": e, "matched_as": hit, "valence": v, "arousal": a, "dominance": d})
    coverage = pd.DataFrame(rows)
    coverage.to_csv(_out("vad_coverage.csv"), index=False)
    source |= {"attempts_failed": attempts, "matched": int(coverage["matched_as"].notna().sum()),
               "of": len(emotions), "unmatched": coverage.loc[coverage["matched_as"].isna(), "emotion"].tolist()}
    _out("lexicon_source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    logger.info(f"lexicon {source['name']}: {source['matched']} of {len(emotions)} emotions matched")


# --- report ------------------------------------------------------------------------------------


def report() -> None:
    """results/emotion_coords/phase0/PHASE0_REPORT.md from the files the other subcommands wrote."""
    out = C.results_dir(OUT)
    parts = ["# Phase 0 report: inventory and data checks\n",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python -m emotion_coords.phase0 report`. "
             "No NDIF calls were made in Phase 0.\n"]

    def table(path: Path, title: str, **kwargs) -> None:
        if path.exists():
            parts.extend([f"## {title}\n", pd.read_csv(path, **kwargs).to_markdown(index=False), ""])

    table(out / "lb_counts.csv", "Liars' Bench rows (Llama-3.3-70B), as pooled")
    table(out / "lb_token_lengths.csv", "Token lengths (whole row, detected span, final assistant message)")
    if (out / "hpkr_subgroups.csv").exists():
        parts.extend(["## HP-KR subgroups (claim x belief)\n",
                      pd.read_csv(out / "hpkr_subgroups.csv").to_markdown(index=False), ""])
    for name, title in (("hpkr_templates.md", None), ("positions_decoded.md", None)):
        if (out / name).exists():
            parts.append(f"See `{name}`.\n")
    summary_path = out / "gemotions_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        sc = s["stories_clean"]
        parts.extend([
            "## Gemotions\n",
            f"`{s['source']['repo']}` @ `{s['source']['revision']}`. stories_clean: {sc['rows']} rows, "
            f"{sc['emotions']} emotions x {sc['topics']} topics, {sc['cells']} of {sc['cells_expected']} cells filled; "
            f"stories per cell {sc['per_cell']}; eligible (>= {sc['min_tokens']} tokens) per cell "
            f"{sc['per_cell_eligible']}, cells with none {sc['cells_without_eligible_story']}. "
            f"Story tokens {sc['story_tokens']}; share below {sc['min_tokens']}: {sc['share_below_min_tokens']:.3f}.\n",
            f"Names the spec uses but Gemotions lacks: {s['names_missing']}.\n",
            f"neutral.db: {s['neutral_db']['rows']}.\n"])
    lex = out / "lexicon_source.json"
    if lex.exists():
        l = json.loads(lex.read_text(encoding="utf-8"))
        parts.append(f"## Valence norms\n\n{l['name']} ({l['url']}): {l['matched']} of {l['of']} emotions matched; "
                     f"unmatched {l['unmatched']}.\n")
    (out / "PHASE0_REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    logger.info(f"wrote {out / 'PHASE0_REPORT.md'}")


if __name__ == "__main__":
    fire.Fire({"lb": lb, "templates": templates, "gemotions": gemotions, "lexicon": lexicon, "report": report})
