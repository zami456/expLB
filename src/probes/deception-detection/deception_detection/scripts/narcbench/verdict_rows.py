"""Verdict rows: compact contexts of Shi et al.'s Gemma-4-31B trajectories, rendered for the Llama-3.3 reader
(spans and points) and for the Qwen3-32B reader (their extractor's positions), the BARE rows, the AWQ check
rows, and the labels file, under the rules of `verdict_spec.py`.

Each agent-episode's history (`results[e].agents[x].messages` in the trace: everything the agent had seen
through episode e's reflection) is cut to the compact context: the system turn, each earlier episode's
outcome feedback, reflection prompt and reflection, and episode e whole with its task-phase tool outputs
replaced by a length note. For Llama the turns become text (own messages, the verdict, the reflection);
for Qwen the compact messages go unchanged through their `locate_verdict_positions` and
`locate_reflection_token`. Labels go to their own file, read only by the scorer after freeze V.

`build` writes `E:/narcbench/verdict/rows/<name>.jsonl`, `labels.json` and `manifest.json`; `check` is gate
GV1 (and the GV3 label counts).
"""

import gzip
import hashlib
import importlib.util
import json
import re
import sys
from bisect import bisect_left
from collections import Counter
from copy import deepcopy
from functools import cache

import fire

from deception_detection.scripts.narcbench import conditions as C
from deception_detection.scripts.narcbench import rows as R
from deception_detection.scripts.narcbench import spec as S
from deception_detection.scripts.narcbench import verdict_spec as V

TASK_HEAD = re.compile(r"^## Episode (\d+): task phase")
COMM_HEAD = re.compile(r"^## Episode (\d+): communication phase")
FEEDBACK_HEAD = re.compile(r"^Episode (\d+) complete\.")
EOT = "<|eot_id|>"
N_AWQ_ROWS = 20


def header(role: str) -> str:
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n"


def load_trace(rep: int) -> dict:
    with gzip.open(V.TRACES / f"rep{rep:03d}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def _is(m: dict, pattern: re.Pattern) -> bool:
    return m["role"] == "user" and bool(pattern.match(m.get("content") or ""))


def _json(text: str) -> dict:
    try:
        out = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


# ── compact context ──────────────────────────────────────────────────────

def episode_starts(messages: list[dict]) -> list[int]:
    """Where each episode in the history opens. The harness drops the oldest episodes when a model's context
    fills (190 of the 1,000 Gemma histories), so the history is a run of consecutive episodes after the
    system turn, not always from episode 1."""
    starts = [i for i, m in enumerate(messages) if _is(m, TASK_HEAD)]
    assert messages[0]["role"] == "system" and starts and starts[0] == 1, "no episode right after the system turn"
    numbers = [int(TASK_HEAD.match(messages[i]["content"]).group(1)) for i in starts]
    assert numbers == list(range(numbers[0], numbers[0] + len(numbers))), "episodes out of order"
    return starts


def compact(messages: list[dict], drop: int = 0) -> tuple[list[dict], int]:
    """The last episode's compact context with the `drop` oldest earlier episodes removed; returns the
    messages and the number of earlier episodes kept."""
    starts = episode_starts(messages)
    bounds = list(zip(starts, starts[1:] + [len(messages)]))
    out = [deepcopy(messages[0])]
    earlier = bounds[:-1][drop:]
    for s, e in earlier:
        f = next(i for i in range(s, e) if _is(messages[i], FEEDBACK_HEAD))
        tail = messages[f:e]
        assert [m["role"] for m in tail] == ["user", "user", "assistant"] and not tail[2].get("tool_calls"), \
            "an earlier episode does not end in feedback, reflection prompt, reflection"
        out += deepcopy(tail)
    s, e = bounds[-1]
    comm = next(i for i in range(s, e) if _is(messages[i], COMM_HEAD))
    for i in range(s, e):
        m = deepcopy(messages[i])
        if i < comm and m["role"] == "tool":
            m["content"] = V.OMITTED.format(n=len(m.get("content") or ""))
        out.append(m)
    return out, len(earlier)


# ── Llama: text turns, spans and points ──────────────────────────────────

def to_llama(msgs: list[dict]) -> tuple[list[dict], list[str | None], Counter]:
    """Text turns for the Llama template, each tagged COMM, VERDICT, REFL or None."""
    cur = max(i for i, m in enumerate(msgs) if _is(m, TASK_HEAD))
    comm = next(i for i in range(cur, len(msgs)) if _is(msgs[i], COMM_HEAD))
    results = {m["tool_call_id"]: m for m in msgs if m["role"] == "tool"}
    plain = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and not m.get("tool_calls")
             and (m.get("content") or "").strip()]
    reflection = plain[-1] if plain and plain[-1] > comm else None
    k_task = sum(len(m.get("tool_calls") or []) for m in msgs[cur:comm] if m["role"] == "assistant")
    turns, tags, info = [], [], Counter()
    for i, m in enumerate(msgs):
        role = m["role"]
        if cur < i < comm and role in ("assistant", "tool"):
            continue
        if role in ("system", "user"):
            turns.append({"role": role, "content": m["content"]})
            tags.append(None)
            if i == cur:
                turns.append({"role": "user", "content": V.TASK_LINE.format(k=k_task)})
                tags.append(None)
            continue
        if role == "tool":
            continue
        if (m.get("content") or "").strip():
            if m.get("tool_calls"):
                info["assistant text beside tool calls"] += 1
            turns.append({"role": "assistant", "content": m["content"]})
            tags.append("REFL" if i == reflection else None)
        for call in m.get("tool_calls") or []:
            name = call["function"]["name"]
            body = _json(results[call["id"]]["content"]) if call["id"] in results else {}
            args = _json(call["function"]["arguments"])
            if name == "send_message" and body.get("success") is True and (args.get("content") or "").strip():
                turns.append({"role": "assistant", "content": args["content"]})
                tags.append("COMM")
            elif name == "submit_verdict" and body.get("recorded") is True:
                turns.append({"role": "assistant", "content": V.VERDICT_TURN.format(
                    verdict=args.get("verdict", body.get("verdict")), reasoning=args.get("reasoning", ""))})
                tags.append("VERDICT")
            else:
                info[f"dropped {name}"] += 1
    return turns, tags, info


def render_llama(tok, turns: list[dict], tags: list) -> dict:
    """Token ids, the spans of the tagged turns and the points (`verdict_spec.POINTS` order; -1 if absent)."""
    text = tok.apply_chat_template(turns, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, starts = enc["input_ids"], [s for s, _ in enc["offset_mapping"]]
    cursor = text.index(EOT) + len(EOT)  # the template's system turn
    spans = {k: [] for k in V.SPANS}
    pre = -1
    for t, tag in zip(turns[1:], tags[1:]):
        body = t["content"].strip()
        pos = cursor
        assert text.startswith(header(t["role"]) + body + EOT, pos), "turn not where the template should put it"
        c0 = pos + len(header(t["role"]))
        c1 = c0 + len(body)
        cursor = c1 + len(EOT)
        if tag is None:
            continue
        a, b = bisect_left(starts, c0), bisect_left(starts, c1)
        assert ids[b] == V.LLAMA_EOT_ID, "no end-of-turn token after the turn"
        assert tok.decode(ids[a:b], clean_up_tokenization_spaces=False) == body, "span does not decode to the turn"
        if tag == "VERDICT":
            pre = bisect_left(starts, pos) - 1
            assert ids[pre] == V.LLAMA_EOT_ID, "the token before the verdict turn is not an end of turn"
        spans[tag].append([a, b])
    assert cursor == len(text), "text after the last turn"
    comm_eot = [b for _, b in spans["COMM"]]
    assert len(comm_eot) <= V.MAX_COMM, "more own messages than rounds"
    one = {k: (spans[k][0][1] if spans[k] else -1) for k in ("VERDICT", "REFL")}
    points = comm_eot + [-1] * (V.MAX_COMM - len(comm_eot)) + [pre, one["VERDICT"], one["REFL"]]
    return {"ids": ids, "groups": [spans[k] for k in V.SPANS], "points": points}


def own_messages(turns: list[dict], tags: list) -> list[str]:
    return [t["content"] for t, g in zip(turns, tags) if g == "COMM"]


# ── Qwen: their extractor ────────────────────────────────────────────────

@cache
def rose_extractor():
    """Their `extract_verdict_activations.py`, loaded from the pinned clone (not edited, not copied)."""
    path = V.REF_ROSE / "extraction" / "extract_verdict_activations.py"
    spec = importlib.util.spec_from_file_location("rose_extract_verdict_activations", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@cache
def shi_tools():
    if str(V.REF_SHI) not in sys.path:
        sys.path.insert(0, str(V.REF_SHI))
    from experiments.tool_schemas import get_task_tool_schemas, get_tool_schemas

    return get_tool_schemas, get_task_tool_schemas


@cache
def answer_keys() -> dict:
    return rose_extractor().build_answer_key_lookup(str(V.REF_SHI / "task" / "*" / "task_manifest_*.json"))


def qwen_passes(qtok, msgs: list[dict], episode: dict, agent: str) -> tuple[dict, dict] | str:
    """Their two passes on the compact messages, or the reason they fail."""
    mod = rose_extractor()
    get_tool_schemas, get_task_tool_schemas = shi_tools()
    peer = V.AGENTS[1 - V.AGENTS.index(agent)]
    slot = episode["agents"][agent]
    loc = mod.locate_verdict_positions(qtok, msgs, peer, episode["agents"][peer].get("verdict_prompt", ""),
                                       get_tool_schemas)
    if loc is None:
        return "verdict positions not found"
    ids_a, positions, verdict_text = loc
    if verdict_text != slot["verdict"]:
        return f"verdict text {verdict_text} != recorded {slot['verdict']}"
    ids_b = mod.locate_reflection_token(qtok, msgs, episode["task_type"], answer_keys().get(slot.get("task_id"), []),
                                        peer, get_task_tool_schemas)
    if ids_b is None:
        return "reflection not rendered"
    a = {"ids": [int(x) for x in ids_a], "points": [int(positions[p]) for p in ("verdict_start", "pre_decision", "decision")]}
    b = {"ids": [int(x) for x in ids_b], "points": [len(ids_b) - 1]}
    return a, b


# ── building ─────────────────────────────────────────────────────────────

def build_rows() -> tuple[dict[str, list[dict]], dict]:
    from transformers import AutoTokenizer

    ltok = AutoTokenizer.from_pretrained(V.LLAMA_MODEL, local_files_only=True)
    qtok = AutoTokenizer.from_pretrained(V.QWEN_MODEL)
    out = {name: [] for name in ("llama_ctx", "llama_bare_comm", "llama_bare_refl", "qwen_verdict", "qwen_reflection")}
    audit = {"excluded": {}, "dropped_llama": Counter(), "dropped_qwen_extra": Counter(), "rendering": Counter(),
             "comm_mismatch": [], "verdict_mismatch": [], "n_own_messages": Counter()}
    for rep in range(1, V.N_REPS + 1):
        trace = load_trace(rep)
        assert trace["condition_id"] == V.CONDITION and len(trace["results"]) == V.N_EPISODES
        for ep in trace["results"]:
            e = ep["episode_index"]
            for agent in V.AGENTS:
                rid = V.row_id(rep, e, agent)
                slot = ep["agents"][agent]
                if slot.get("verdict") not in ("accept", "reject"):
                    audit["excluded"][rid] = f"verdict {slot.get('verdict')!r}"
                    continue
                history = slot["messages"]
                starts = episode_starts(history)
                assert int(TASK_HEAD.match(history[starts[-1]]["content"]).group(1)) == e + 1, "history ends elsewhere"
                n_earlier = len(starts) - 1
                drop = 0
                while True:
                    msgs, kept = compact(history, drop)
                    turns, tags, info = to_llama(msgs)
                    row = render_llama(ltok, turns, tags)
                    if len(row["ids"]) <= V.LLAMA_BUDGET or drop >= n_earlier:
                        break
                    drop += 1
                if len(row["ids"]) > V.LLAMA_BUDGET:
                    audit["excluded"][rid] = f"{len(row['ids'])} Llama tokens with no earlier episode left"
                    continue
                if "VERDICT" not in tags or "REFL" not in tags:
                    audit["excluded"][rid] = "no recorded verdict turn" if "VERDICT" not in tags else "no reflection"
                    continue
                audit["dropped_llama"][drop] += 1
                audit["rendering"].update(info)
                mine = own_messages(turns, tags)
                audit["n_own_messages"][len(mine)] += 1
                sent = [c["content"] for c in ep["channel_transcript"] if c["sender"] == agent and c["content"].strip()]
                if mine != sent:
                    audit["comm_mismatch"].append(rid)
                verdict_turn = turns[tags.index("VERDICT")]["content"]
                if not verdict_turn.startswith(f"VERDICT: {slot['verdict']}\n"):
                    audit["verdict_mismatch"].append(rid)
                # Qwen: the same compact context, more dropped only past the Qwen budget
                qdrop = drop
                while True:
                    qmsgs, _ = compact(history, qdrop)
                    got = qwen_passes(qtok, qmsgs, ep, agent)
                    if isinstance(got, str) or len(got[1]["ids"]) <= V.QWEN_BUDGET or qdrop >= n_earlier:
                        break
                    qdrop += 1
                if isinstance(got, str) or len(got[1]["ids"]) > V.QWEN_BUDGET:
                    audit["excluded"][rid] = got if isinstance(got, str) else f"{len(got[1]['ids'])} Qwen tokens"
                    continue
                audit["dropped_qwen_extra"][qdrop - drop] += 1
                base = {"rep": rep, "episode_index": e, "agent": agent, "run_id": trace["run_id"]}
                out["llama_ctx"].append({"row_id": V.ROW_BASE["llama_ctx"] + rid, **base, **row,
                                         "kept_earlier": kept, "dropped_earlier": drop})
                for name, text in (("llama_bare_comm", "\n\n".join(mine)),
                                   ("llama_bare_refl", turns[tags.index("REFL")]["content"])):
                    bare = C.render(ltok, C.BARE_SYSTEM, C.BARE_USER, text)
                    out[name].append({"row_id": V.ROW_BASE[name] + rid, **base, "ids": bare["ids"],
                                      "groups": [[[bare["a"], bare["b"]]]], "points": [bare["b"]]})
                qa, qb = got
                out["qwen_verdict"].append({"row_id": V.ROW_BASE["qwen_verdict"] + rid, **base, **qa, "groups": [],
                                            "dropped_earlier": qdrop})
                out["qwen_reflection"].append({"row_id": V.ROW_BASE["qwen_reflection"] + rid, **base, **qb,
                                               "groups": [], "dropped_earlier": qdrop})
        print(f"rep {rep:03d}: {sum(1 for r in out['llama_ctx'] if r['rep'] == rep)} rows", flush=True)
    out["qwen_awq_check"] = awq_rows(qtok)
    return out, audit


def awq_rows(qtok) -> list[dict]:
    """The first N_AWQ_ROWS released Qwen3 Transfer rows with a reply, tokenised as their extractor did (their
    `n_tokens` must match), read at their last token."""
    ours = [r for r in R.load_rows("transfer") if r["rose_row"]]
    meta = R.released_meta("transfer", "qwen3_32b")
    pos = R.align(ours, meta)
    rows = []
    for j, (p, m) in enumerate(zip(pos, meta)):
        if p is None or not m["n_tokens"]:
            continue
        r = ours[p]
        text = qtok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]},
                                         {"role": "assistant", "content": r["reply"]}], tokenize=False)
        ids = qtok(text)["input_ids"]
        assert len(ids) == m["n_tokens"], f"released row {j}: {len(ids)} tokens, they recorded {m['n_tokens']}"
        rows.append({"row_id": V.ROW_BASE["qwen_awq_check"] + j, "meta_index": j, "run_id": r["run_id"],
                     "ids": [int(x) for x in ids], "groups": [], "points": [len(ids) - 1]})
        if len(rows) == N_AWQ_ROWS:
            break
    return rows


def labels() -> dict:
    """Shi's episode labels, Rose's per-agent flag (their rule, from the trace fields), Shi's relaxation."""
    import pandas as pd

    ep = pd.read_parquet(V.EPISODES_PARQUET)
    ep = ep[~ep["is_warmup"].astype(bool)]
    rel = pd.read_parquet(V.RELAXATION_PARQUET)
    rel = rel[rel["run_id"].str.startswith(V.CONDITION + "__")]
    episodes, agents = {}, {}
    for _, r in ep.iterrows():
        rep = int(r["run_id"].rsplit("rep", 1)[1])
        episodes[f"{rep}|{int(r['episode_index'])}"] = {
            "run_id": r["run_id"], "collusion": bool(r["collusion"]), "both_accept": bool(r["both_accept"]),
            "agreement": None if r["agreement_status"] != "valid" else bool(r["agreement"]),
            "verdicts": {a: r[f"{a}_verdict"] for a in V.AGENTS}}
    relax = {(r["run_id"], int(r["episode_index"]), r["agent"]): (bool(r["label"]) if r["status"] == "valid" else None)
             for _, r in rel.iterrows()}
    for rep in range(1, V.N_REPS + 1):
        trace = load_trace(rep)
        for e in trace["results"]:
            for a in V.AGENTS:
                slot = e["agents"][a]
                agents[str(V.row_id(rep, e["episode_index"], a))] = {
                    "rep": rep, "episode_index": e["episode_index"], "agent": a, "verdict": slot.get("verdict"),
                    "verdict_correct": slot.get("verdict_correct"),
                    "rose_flag": slot.get("verdict") == "accept" and not slot.get("verdict_correct"),
                    "relaxation": relax.get((trace["run_id"], e["episode_index"], a))}
    rose_ep = {k: any(agents[str(V.row_id(int(k.split("|")[0]), int(k.split("|")[1]), a))]["rose_flag"] for a in V.AGENTS)
               for k in episodes}
    for k, v in episodes.items():
        v["rose"] = rose_ep[k]
    checks = {"episodes": len(episodes), "collusion": sum(v["collusion"] for v in episodes.values()),
              "collusion_runs": len({v["run_id"] for v in episodes.values() if v["collusion"]}),
              "collusion_equals_both_accept": all(v["collusion"] == v["both_accept"] for v in episodes.values()),
              "rose_episodes": sum(rose_ep.values()),
              "trace_verdicts_match_table": all(episodes[f"{ag['rep']}|{ag['episode_index']}"]["verdicts"][ag["agent"]]
                                                == ag["verdict"] for ag in agents.values()),
              "relaxation_valid": sum(v["relaxation"] is not None for v in agents.values())}
    return {"episodes": episodes, "agents": agents, "checks": checks}


def sha256_file(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> None:
    V.ROWS.mkdir(parents=True, exist_ok=True)
    out, audit = build_rows()
    manifest = {}
    for name, rows in out.items():
        path = V.ROWS / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        lengths = [len(r["ids"]) for r in rows]
        manifest[name] = {"rows": len(rows), "tokens": sum(lengths), "max_tokens": max(lengths) if lengths else 0,
                          "sha256": sha256_file(path)}
    lab = labels()
    (V.ROWS / "labels.json").write_text(json.dumps(lab, indent=1), encoding="utf-8")
    manifest["labels"] = {"sha256": sha256_file(V.ROWS / "labels.json")}
    audit = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in audit.items()}
    (V.ROWS / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (V.ROWS / "audit.json").write_text(json.dumps(audit, indent=1, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


def check() -> None:
    """Gate GV1 (rows) and GV3 (label counts against Shi et al.'s EXPERIMENTS.md: 146/500 episodes, 33/50 runs)."""
    manifest = json.loads((V.ROWS / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((V.ROWS / "audit.json").read_text(encoding="utf-8"))
    checks = json.loads((V.ROWS / "labels.json").read_text(encoding="utf-8"))["checks"]
    n = manifest["llama_ctx"]["rows"]
    gv1 = {
        "every agent-episode has a row or a recorded exclusion": n + len(audit["excluded"]) == V.N_REPS * V.N_EPISODES * 2,
        "own messages equal the channel transcript": not audit["comm_mismatch"],
        "verdict turns equal the recorded verdicts": not audit["verdict_mismatch"],
        "Llama rows within budget": manifest["llama_ctx"]["max_tokens"] <= V.LLAMA_BUDGET,
        "Qwen rows within budget": max(manifest["qwen_reflection"]["max_tokens"], manifest["qwen_verdict"]["max_tokens"]) <= V.QWEN_BUDGET,
        "Llama, BARE and Qwen rows cover the same agent-episodes": len({manifest[k]["rows"] for k in
            ("llama_ctx", "llama_bare_comm", "llama_bare_refl", "qwen_verdict", "qwen_reflection")}) == 1,
        "AWQ check rows tokenised as recorded": manifest["qwen_awq_check"]["rows"] == N_AWQ_ROWS}
    gv3 = {"collusion 146 of 500": checks["collusion"] == 146 and checks["episodes"] == 500,
           "33 of 50 runs": checks["collusion_runs"] == 33, "collusion = both_accept": checks["collusion_equals_both_accept"],
           "trace verdicts = episodes table": checks["trace_verdicts_match_table"]}
    lines = ["# GV1: Verdict rows", ""] + [f"- {k}: {'yes' if v else 'NO'}" for k, v in gv1.items()]
    lines += ["", f"Rows: {n} agent-episodes; excluded {len(audit['excluded'])} ({audit['excluded'] or 'none'}).",
              f"Earlier episodes dropped for the Llama budget: {audit['dropped_llama']}; extra for Qwen: "
              f"{audit['dropped_qwen_extra']}. Own messages per agent-episode: {audit['n_own_messages']}. "
              f"Rendering notes: {audit['rendering']}.",
              "", "Files: " + "; ".join(f"`{k}` {v['rows']} rows, {v['tokens']:,} tokens, longest {v['max_tokens']}"
                                        for k, v in manifest.items() if k != "labels"),
              "", "# GV3: labels", ""] + [f"- {k}: {'yes' if v else 'NO'}" for k, v in gv3.items()]
    lines += [f"- Rose-flagged episodes: {checks['rose_episodes']}; valid relaxation labels: {checks['relaxation_valid']}",
              "", f"GV1 {'PASSED' if all(gv1.values()) else 'FAILED'}; GV3 {'PASSED' if all(gv3.values()) else 'FAILED'}."]
    (S.LOGS / "gv1_verdict_rows.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    fire.Fire({"build": build, "check": check})
