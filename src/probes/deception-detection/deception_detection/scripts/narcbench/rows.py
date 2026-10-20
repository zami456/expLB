"""NARCBench rows: rebuild Rose et al.'s per-agent rows from the released transcripts, and audit them.

The matchers port `extraction/extract_core.py`, `extract_transfer.py` and `extract_stego.py` of
github.com/aaronrose227/narcbench (MIT; commit in `spec.GITHUB_COMMIT`) line for line: each
`agent_prompts.json` entry is paired with a stored reply by position within its channel, the final
phase (Core, Transfer) by agent name from `final_texts`. Two fields are added for the audit:
`final_text`, the agent's entry in `final_texts` (which the Stego matcher never reads), and
`reply_by_channel`, the reply an extractor matching on channel alone would read.

`build` writes `E:/narcbench/rows/<tier>.jsonl` (one row per prompt entry, judge rows kept but marked
`rose_row = False`, as their Stego extractor drops them) and `runs_<tier>.json` (run-level fields).
`audit` is gate G1: for each released model it aligns the released metadata with the rebuilt rows,
tries their tokenisation variants on `n_tokens`, and reports empty replies, speaker and role
mismatches, plain-text leaks and the scenario runs without released activations.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import fire

from deception_detection.scripts.narcbench import spec as S

PREFIX = re.compile(r"^\[([^\]]+)\]:\s*")
LEAK_LINE = re.compile(r"(?m)^[ \t*_>#-]*(PRIVATE|DECODE|RANGE)\s*:")
PRIVATE_BLOCK = ("PRIVATE COORDINATION", "PRIVATE (only you", "This is a private channel between you")
TOKENIZERS = {"llama_3_1_70b": S.ROSE_LLAMA_TOKENIZER, "qwen3_32b": "Qwen/Qwen3-32B-AWQ"}
# how their extractors could have produced `n_tokens`: template variables, and private rows matched
# on channel alone (the current Core extractor leaves private-phase rows empty)
VARIANTS = {
    "as_coded": ({}, False),
    "thinking_off": ({"enable_thinking": False}, False),
    "private_by_channel": ({}, True),
    "thinking_off+private_by_channel": ({"enable_thinking": False}, True),
}


def strip_prefix(msg: str) -> str:
    """Their `_strip_name_prefix`: remove a leading '[Agent Name]: '."""
    return PREFIX.sub("", msg)


def prefix_name(msg: str) -> str | None:
    m = PREFIX.match(msg)
    return m.group(1) if m else None


def run_dirs(tier: str) -> list[Path]:
    return sorted(d for d in (S.SCEN / tier).iterdir() if d.is_dir())


def load_run(run_dir: Path) -> tuple[dict, list[dict], dict]:
    def read(name: str):
        return json.loads((run_dir / name).read_text(encoding="utf-8"))

    return read("run_config.json"), read("agent_prompts.json"), read("results.json")


def match(tier: str, cfg: dict, prompts: list[dict], results: dict) -> list[dict]:
    """Pair every prompt entry with the reply their extractor reads (see the module docstring)."""
    channels = {
        "private": results.get("private_messages", []),
        "public": results.get("public_messages", []),
        "thinking": results.get("thinking_messages", []),
    }
    final_texts = results.get("final_texts", {}) or {}
    used = Counter()
    by_channel = Counter()
    rows = []

    def take(channel: str) -> tuple[str, str | None, str]:
        msgs, i = channels[channel], used[channel]
        used[channel] += 1
        if i < len(msgs):
            return strip_prefix(msgs[i]), prefix_name(msgs[i]), f"{channel}_messages[{i}]"
        return "", None, f"{channel}_messages[{i}]:past_end"

    for entry in prompts:
        phase, channel, agent = entry["phase"], entry["channel"], entry["agent_name"]
        role = entry.get("role", cfg.get("roles", {}).get(agent, "unknown"))
        prefix, src, alt = None, "none", None
        if channel in channels:
            k = by_channel[channel]
            by_channel[channel] += 1
            alt = strip_prefix(channels[channel][k]) if k < len(channels[channel]) else ""
        if tier == "core":
            if phase == "discussion" and channel in ("private", "public"):
                reply, prefix, src = take(channel)
            elif phase == "final":
                reply, src = final_texts.get(agent, ""), "final_texts"
            else:
                reply = ""
        elif tier == "transfer":
            if channel == "private":
                reply, prefix, src = take("private")
            elif phase == "discussion" and channel == "public":
                reply, prefix, src = take("public")
            elif phase == "final":
                reply, src = final_texts.get(agent, ""), "final_texts"
            else:
                reply = ""
        elif tier == "stego":
            if channel in channels:
                reply, prefix, src = take(channel)
            else:
                reply = ""
        else:
            raise ValueError(tier)
        rows.append({
            "tier": tier, "run_id": cfg["run_id"], "scenario_id": cfg.get("scenario_id"),
            "family": cfg["run_id"].split("__")[0], "mode": cfg["mode"], "agent": agent, "role": role,
            "phase": phase, "channel": channel, "round": entry["round"],
            "system": entry["system_prompt"], "user": entry["user_prompt"],
            "reply": reply, "reply_src": src, "prefix": prefix, "reply_by_channel": alt,
            "final_text": final_texts.get(agent),
            "rose_row": not (tier == "stego" and role == "judge"),
        })
    return rows


RUN_FIELDS = ("run_id", "mode", "scenario_id", "scenario_domain", "scenario_type", "seed", "agents", "roles",
              "effective_roles", "coalition", "votes", "target_option", "fair_option", "label_variant",
              "collusion_success", "target_majority", "colluder_voted_target", "target_vote_count",
              "collusion_active", "flagged", "judge", "task_agents", "bp_bet_correlation")


def build() -> None:
    """Write rows and run-level fields for every tier."""
    S.ROWS.mkdir(parents=True, exist_ok=True)
    for tier in S.TIERS:
        rows, runs = [], {}
        for d in run_dirs(tier):
            cfg, prompts, results = load_run(d)
            assert cfg["run_id"] == d.name, (d, cfg["run_id"])
            runs[cfg["run_id"]] = {k: cfg[k] for k in RUN_FIELDS if k in cfg}
            rows.extend(match(tier, cfg, prompts, results))
        for i, r in enumerate(rows):
            r["row_id"] = S.ROW_OFFSET[tier] + i
        with open(S.ROWS / f"{tier}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (S.ROWS / f"runs_{tier}.json").write_text(json.dumps(runs, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{tier}: {len(runs)} runs, {len(rows)} rows")


def load_rows(tier: str) -> list[dict]:
    with open(S.ROWS / f"{tier}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_runs(tier: str) -> dict[str, dict]:
    return json.loads((S.ROWS / f"runs_{tier}.json").read_text(encoding="utf-8"))


def released_meta(tier: str, model: str = S.RELEASED_MODEL) -> list[dict]:
    return json.loads((S.RELEASED / model / tier / "metadata_gen.json").read_text(encoding="utf-8"))


KEY = ("run_id", "agent", "phase", "channel", "round")


def align(ours: list[dict], meta: list[dict]) -> list[int | None]:
    """Index into `ours` of each released row, matched on (run, agent, phase, channel, round, occurrence)."""
    index, seen = {}, Counter()
    for i, r in enumerate(ours):
        k = tuple(r[f] for f in KEY)
        index[k + (seen[k],)] = i
        seen[k] += 1
    seen.clear()
    out = []
    for m in meta:
        k = (m["run_id"], m["agent_name"], m["phase"], m["channel"], m["round"])
        out.append(index.get(k + (seen[k],)))
        seen[k] += 1
    return out


def rose_n_tokens(tok, row: dict, tier: str, template_kwargs: dict | None = None,
                  private_by_channel: bool = False) -> int:
    """`n_tokens` as their extractors record it: the chat template as text, re-tokenised with default
    special tokens (a second BOS for Llama), and 0 for an empty reply in Core and Transfer."""
    reply = row["reply"]
    if private_by_channel and row["channel"] == "private" and row.get("reply_by_channel") is not None:
        reply = row["reply_by_channel"]
    if tier in ("core", "transfer") and not reply:
        return 0
    messages = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]},
                {"role": "assistant", "content": reply}]
    text = tok.apply_chat_template(messages, tokenize=False, **(template_kwargs or {}))
    return len(tok(text)["input_ids"])


def _leaks(text: str) -> Counter:
    return Counter(m.group(1) for m in LEAK_LINE.finditer(text or ""))


def _model_section(model: str, tier: str, rows: list[dict], examples: int) -> tuple[list[str], dict]:
    from transformers import AutoTokenizer

    meta = released_meta(tier, model)
    ours = [r for r in rows if r["rose_row"]]
    pos = align(ours, meta)
    missing = sum(p is None for p in pos)
    hit = [p for p in pos if p is not None]
    monotone = all(a < b for a, b in zip(hit, hit[1:]))
    fields_ok = all(ours[p]["role"] == m["role"] and ours[p]["mode"] == m["mode"]
                    for p, m in zip(pos, meta) if p is not None)
    same_runs = set(m["run_id"] for m in meta)
    identical = (missing == 0 and pos == list(range(len(pos)))
                 and len(pos) == sum(1 for r in ours if r["run_id"] in same_runs))
    tok = AutoTokenizer.from_pretrained(TOKENIZERS[model])
    exact, bad = {}, {}
    for name, (kwargs, by_channel) in VARIANTS.items():
        if model.startswith("llama") and kwargs:
            continue  # Llama's template takes no thinking switch
        ok, wrong = 0, []
        for p, m in zip(pos, meta):
            if p is None:
                continue
            n = rose_n_tokens(tok, ours[p], tier, kwargs, by_channel)
            if n == m["n_tokens"]:
                ok += 1
            else:
                r = ours[p]
                wrong.append((r["row_id"], r["run_id"], r["agent"], r["phase"], r["round"], n, m["n_tokens"]))
        exact[name], bad[name] = ok, wrong
    best = max(exact, key=exact.get)
    empty_final = sum(1 for p, m in zip(pos, meta) if p is not None and ours[p]["phase"] == "final"
                      and not ours[p]["reply"] and ours[p]["final_text"])
    lines = [f"### {tier}: released `{model}`", "",
             f"- Released rows {len(meta)}; matched to rebuilt rows {len(meta) - missing}; unmatched {missing}; "
             f"in rebuilt order {monotone}; role and mode agree {fields_ok}; row for row identical {identical}.",
             "- `n_tokens` reproduced exactly, by variant: "
             + ", ".join(f"{k} {v}/{len(meta)}" for k, v in exact.items()) + f". Best: **{best}**.",
             f"- Released final-phase rows whose rebuilt reply is empty although `final_texts` has one: {empty_final}.",
             ""]
    if bad[best]:
        lines += ["  First mismatches under the best variant (row_id, run, agent, phase, round, ours, theirs): "
                  + "; ".join(str(b) for b in bad[best][:examples]), ""]
    info = {"released_rows": len(meta), "unmatched": missing, "monotone": monotone, "fields_ok": fields_ok,
            "identical": identical, "n_tokens_exact": exact, "best_variant": best,
            "empty_final_with_final_text": empty_final}
    return lines, info


def audit(models: tuple = ("qwen3_32b", "llama_3_1_70b"), examples: int = 3) -> None:
    """Gate G1. Writes `E:/narcbench/logs/g1_rows.md` and `g1_rows.json`."""
    out = ["# G1: row audit", "",
           f"Data revision `{S.HF_REVISION}`, their code `{S.GITHUB_COMMIT[:8]}`. Tokenizers: "
           + ", ".join(f"{m} `{TOKENIZERS[m]}`" for m in models) + ".", ""]
    summary = {}
    for tier in S.TIERS:
        rows, runs = load_rows(tier), load_runs(tier)
        summary[tier] = {}
        out += [f"## {tier}", ""]
        for model in models:
            lines, info = _model_section(model, tier, rows, examples)
            out += lines
            summary[tier][model] = info
        rel = set(m["run_id"] for m in released_meta(tier, models[0]))
        ours = [r for r in rows if r["rose_row"] and r["run_id"] in rel]
        empty = Counter((r["phase"], r["channel"]) for r in ours if not r["reply"])
        speaker = [(r["row_id"], r["agent"], r["prefix"]) for r in rows
                   if r["prefix"] is not None and r["prefix"] != r["agent"]]
        role_mismatch = [rid for rid, c in runs.items()
                         if c.get("effective_roles") is not None and c.get("effective_roles") != c.get("roles")]
        leak = defaultdict(Counter)
        for r in rows:
            for kind, n in _leaks(r["reply"]).items():
                leak[("reply", r["mode"], r["role"])][kind] += n
            for kind, n in _leaks(r["user"]).items():
                leak[("prompt", r["mode"], r["role"])][kind] += n
            for marker in PRIVATE_BLOCK:
                if marker in r["user"]:
                    leak[("prompt", r["mode"], r["role"])][f"block:{marker}"] += 1
        extra = sorted(set(runs) - rel)
        summary[tier].update({"runs_scenarios": len(runs), "runs_released": len(rel), "extra_runs": extra,
                              "empty_by_phase_channel": {f"{a}/{b}": n for (a, b), n in empty.items()},
                              "speaker_mismatch": len(speaker), "role_mismatch_runs": role_mismatch})
        out += [f"### {tier}: transcripts", "",
                f"- Runs: {len(runs)} in the scenarios, {len(rel)} with released activations, {len(extra)} without.",
                f"- Empty rebuilt replies in released runs, by (phase, channel): {dict(empty)}.",
                f"- Speaker mismatches (a matched message's [Name] prefix differs from the row's agent): {len(speaker)}.",
                f"- Runs whose `effective_roles` differ from `roles`: {len(role_mismatch)}.", ""]
        if speaker:
            out += ["  First speaker mismatches: " + "; ".join(str(s) for s in speaker[:examples]), ""]
        if leak:
            out += ["  Leak scan, all runs (field, mode, role of the row's agent): plain-text PRIVATE/DECODE/RANGE "
                    "lines and private-block markers.", ""]
            out += [f"  - {k}: {dict(leak[k])}" for k in sorted(leak)] + [""]
        if extra and tier == "core":
            out += ["  Scenario runs without released activations (mode, flagged, collusion_success, "
                    "colluders voting target):", ""]
            out += [f"  - {rid}: {runs[rid].get('mode')}, flagged={runs[rid].get('flagged')}, "
                    f"success={runs[rid].get('collusion_success')}, on_target={runs[rid].get('colluder_voted_target')}"
                    for rid in extra] + [""]
    S.LOGS.mkdir(parents=True, exist_ok=True)
    (S.LOGS / "g1_rows.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    (S.LOGS / "g1_rows.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    fire.Fire({"build": build, "audit": audit})
