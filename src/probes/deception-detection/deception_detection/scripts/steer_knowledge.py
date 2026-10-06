"""Steer Llama-3.3-70B along the knowledge direction K at layer 40: a small run.

K (known minus unknowable, under a forced denial; `recall_denial`, `none|general`) reads whether the
model knows an answer. This asks whether the model uses it. K is added to the residual stream after
block 40 (`steering.steered_generate`, ``h + alpha * v`` at every token, the first steering run's
method, every direction at the norm of D9) while Llama reports whether it knows answers:

    H   HP-KR under harm pressure: the first run's 20 WMDP questions Llama knows, the Liars' Bench
        row itself. A denial is a lie.
    C   the same questions without pressure: the benchmark's control knowledge report.
    A   the same questions answered: the benchmark's control choice template (the first run's "K").
    G+  20 known facts from the replication set (`recall_items_matched`, never used to build K), in
        K's own knowledge-report template under HP-KR's system header. A denial is false ignorance.
    G-  20 unknowable particulars from the same set and template. A claim is false knowledge.
    GA  the G+ facts answered, with `recall_denial`'s answer prompt.

-K should make Llama deny knowing what it knows (G+, C) and +K claim what it cannot know (G-) or
admit what it hides under pressure (H). The answer checks (A, GA) separate a changed report from
destroyed knowledge. Two random directions at the same norm are the null.

    python deception_detection/scripts/steer_knowledge.py items
    python deception_detection/scripts/steer_knowledge.py run --direction none   # alpha = 0, plus checks
    python deception_detection/scripts/steer_knowledge.py run --direction K      # then random0, random1
    python deception_detection/scripts/steer_knowledge.py status                  # tallies so far
    python deception_detection/scripts/steer_knowledge.py report

Answers are graded by the first run's regexes. Outputs go to `results/steering_L40_knowledge/`. Runs
resume: each request's answers are appended as they arrive, and a rerun skips those already there.
"""

import json

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from deception_detection.scripts import recall_denial as rd
from deception_detection.scripts import recall_items_matched as matched
from deception_detection.scripts.forced_denial_bank import NO_ROLE_SYSTEM
from deception_detection.scripts.steer_lies import (
    LAYER,
    STOP_TOKENS,
    Group,
    _bootstrap,
    _generate,
    _is_oom,
    _model,
    _read,
    _tokenize,
    _trim,
    batches,
    extract_choice,
    knowledge_claim,
)
from deception_detection.scripts.steer_lies import OUT as FIRST_RUN
from deception_detection.utils import REPO_ROOT

OUT = REPO_ROOT / "results" / "steering_L40_knowledge"
N_GENERAL = 20
BASELINE = "none"
DIRECTIONS = ("K", "random0", "random1")
# ordered by size, so the gentle pushes come first
ALPHAS = {"K": (-1.0, 1.0, -2.0, 2.0, -4.0, 4.0, -8.0, 8.0),
          "random0": (-4.0, 4.0, -8.0, 8.0), "random1": (-4.0, 4.0, -8.0, 8.0)}
# the first run's two HP-KR groups, so that its requests are sent unchanged, and the general items'
GROUPS = (
    Group("hpkr_report", ("H", "C"), 32),
    Group("hpkr_choice", ("A",), 8),
    Group("general_report", ("G+", "G-"), 32),
    Group("general_choice", ("GA",), 8),
)
SETS = ("G+", "G-", "GA", "C", "H", "A")
# 1 = the outcome each set is about
OUTCOME = {"H": "denies", "C": "denies", "G+": "denies", "G-": "claims", "A": "correct", "GA": "correct"}
SET_LABEL = {
    "G+": "known fact, K's template (G+): denies knowing",
    "G-": "unknowable particular (G-): claims to know",
    "GA": "known fact answered (GA): correct letter",
    "C": "HP-KR, no pressure (C): denies knowing",
    "H": "HP-KR, harm pressure (H): denies knowing (a lie)",
    "A": "HP-KR answered (A): gives its believed answer",
}


# --- items and directions ------------------------------------------------------------------------

def _choices(choices: list[str]) -> str:
    return "\n".join(f"{L}. {c}" for L, c in zip(rd.LETTERS, choices))


def items() -> None:
    _build_items()


def _build_items() -> list[dict]:
    """Every prompt: the first run's 20 HP-KR questions under H, C and A, in its order, then 20 known
    and 20 unknowable replication items. Known items are the first 20 the model answered correctly
    in `recall_denial/matched/verify.json`."""
    hpkr = pd.read_json(FIRST_RUN / "items_hpkr.json")
    verified = {v["question"]: v["correct"]
                for v in json.loads((rd.OUT / "matched" / "verify.json").read_text(encoding="utf-8"))["known"]}
    known = [k for k in matched.KNOWN if verified.get(k[0], False)][:N_GENERAL]
    rows = []
    for s, column in (("H", "H"), ("C", "C"), ("A", "K")):
        for _, r in hpkr.iterrows():
            rows.append({"item_id": r["item_id"], "set": s, "system": r["system"], "user": r[column],
                         "belief": int(r["belief"])})
    for i, (question, choices, _) in enumerate(known):
        rows.append({"item_id": f"known-{i:02d}", "set": "G+", "system": NO_ROLE_SYSTEM,
                     "user": rd.build_prompt(question, choices), "question": question})
    for i, (question, choices) in enumerate(matched.UNKNOWN[:N_GENERAL]):
        rows.append({"item_id": f"unknown-{i:02d}", "set": "G-", "system": NO_ROLE_SYSTEM,
                     "user": rd.build_prompt(question, choices), "question": question})
    for i, (question, choices, key) in enumerate(known):
        rows.append({"item_id": f"known-{i:02d}", "set": "GA", "system": NO_ROLE_SYSTEM,
                     "user": rd.ANSWER_PROMPT.format(question=question, choices=_choices(choices)),
                     "question": question, "key": key})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "items.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(f"{len(rows)} prompts written to {OUT / 'items.json'}: "
                f"{pd.Series([r['set'] for r in rows]).value_counts().to_dict()}")
    return rows


def _items() -> list[dict]:
    path = OUT / "items.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else _build_items()


def directions() -> dict[str, np.ndarray]:
    """K at layer 40 and the first run's random directions, all at the norm of D9, with D9 and D5
    kept for the cosines."""
    from deception_detection.steering import steering_directions

    base = steering_directions(LAYER)
    norm = float(np.linalg.norm(base["D9"]))
    k = np.load(rd.OUT / "directions.npz")[f"L{LAYER}|none|general"].astype(np.float64)
    return {"K": k / np.linalg.norm(k) * norm, "random0": base["random0"], "random1": base["random1"],
            "D9": base["D9"], "D5": base["D5"]}


def _write_directions(vectors: dict[str, np.ndarray]) -> None:
    from deception_detection.steering import cosines

    stored = np.load(rd.OUT / "directions.npz")[f"L{LAYER}|none|general"]
    k = vectors["K"]
    meta = {"layer": LAYER, "norm": float(np.linalg.norm(vectors["D9"])), "K_norm": float(np.linalg.norm(k)),
            "K_cosine_with_stored": float(k @ stored / np.linalg.norm(k) / np.linalg.norm(stored)),
            "cosines": {f"{a}|{b}": c for (a, b), c in cosines(vectors).items()}}
    (OUT / "directions.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    logger.info(f"K: norm {meta['K_norm']:.3f} (D9's {meta['norm']:.3f}), cosine with the stored K "
                f"{meta['K_cosine_with_stored']:.6f}, with D9 {meta['cosines']['K|D9']:.3f}")


# --- generation on NDIF ----------------------------------------------------------------------------

def _done(path) -> set[tuple[float, str, str]]:
    return {(r["alpha"], r["set"], r["item_id"]) for r in _read(path)} if path.exists() else set()


def run(direction: str = "K", alphas=None) -> None:
    """Every group's answers under `direction` at each alpha, appended as they arrive.

    `direction="none"` is the shared alpha = 0 baseline. Its run also checks that batching changes
    nothing (one known and one unknowable prompt generated alone) and compares the HP-KR answers
    with the first run's baseline, whose requests these are.
    """
    rows = _items()
    vectors = directions()
    if direction == BASELINE:
        alphas, vector = (0.0,), np.zeros_like(vectors["K"])
    else:
        alphas = tuple(float(a) for a in (alphas if alphas is not None else ALPHAS[direction]))
        vector = vectors[direction]
    (OUT / "generations").mkdir(parents=True, exist_ok=True)
    if not (OUT / "directions.json").exists():
        _write_directions(vectors)
    model, tokenizer = _model()
    path = OUT / "generations" / f"{direction}.jsonl"
    done = _done(path)
    chats = {g.name: [r for r in rows if r["set"] in g.sets] for g in GROUPS}
    requests = {g.name: batches(tokenizer, chats[g.name], g.max_new_tokens) for g in GROUPS}
    skipped = 0
    for alpha in alphas:
        for group in GROUPS:
            for b, indices in enumerate(requests[group.name]):
                todo = [chats[group.name][i] for i in indices
                        if (alpha, chats[group.name][i]["set"], chats[group.name][i]["item_id"]) not in done]
                if not todo:
                    continue
                try:
                    answers, seconds = _generate(model, tokenizer, todo, alpha * vector, group.max_new_tokens)
                except Exception as error:
                    # a crowded NDIF GPU can refuse even one long prompt; the rest still runs, and the
                    # nonzero exit below sends the caller back for what was skipped
                    if not _is_oom(error):
                        raise
                    skipped += len(todo)
                    logger.warning(f"SKIPPED {direction} alpha={alpha:+g} {group.name} request {b + 1}: "
                                   f"{len(todo)} prompts out of memory even one at a time")
                    continue
                with open(path, "a", encoding="utf-8") as f:
                    for chat, answer in zip(todo, answers):
                        f.write(json.dumps({"direction": direction, "alpha": alpha, "group": group.name,
                                            "request": b, "item_id": chat["item_id"], "set": chat["set"],
                                            "seconds": seconds} | answer, ensure_ascii=False) + "\n")
                logger.info(f"{direction} alpha={alpha:+g} {group.name} request {b + 1}/"
                            f"{len(requests[group.name])}: {len(todo)} answers in {seconds} s")
        logger.info(f"ALPHA DONE {direction} {alpha:+g}")
    if skipped:
        logger.warning(f"INCOMPLETE {direction}: {skipped} answers skipped; rerun to fill them in")
        raise SystemExit(1)
    if direction == BASELINE:
        _checks(model, tokenizer, chats, path)
    logger.info(f"DIRECTION DONE {direction}")


def _checks(model, tokenizer, chats: dict[str, list[dict]], path) -> None:
    """Alpha = 0 in a batch must equal one-at-a-time generation, and the HP-KR answers should equal
    the first run's (same prompts, same requests, three months apart on NDIF)."""
    from deception_detection.scripts.ask_and_detect import generate

    base = {(r["item_id"], r["set"]): r["ids"] for r in _read(path)}
    stops = {tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKENS}
    unbatched = []
    for chat in (chats["general_report"][0], chats["general_report"][N_GENERAL]):  # one G+, one G-
        toks = _tokenize(tokenizer, [chat])
        alone, _ = _trim(generate(model, toks.tokens, toks.attention_mask, 32), stops)
        same = alone == base[(chat["item_id"], chat["set"])]
        unbatched.append({"item_id": chat["item_id"], "set": chat["set"], "same": same,
                          "alone": tokenizer.decode(alone, skip_special_tokens=True).strip()})
        logger.info(f"unbatched check {chat['item_id']} {chat['set']}: {'same' if same else 'DIFFERENT'}")
    first = {(r["item_id"], "A" if r["set"] == "K" else r["set"]): r["ids"]
             for r in _read(FIRST_RUN / "generations" / f"{BASELINE}.jsonl") if r["set"] in ("H", "C", "K")}
    shared = [key for key in first if key in base]
    identical = sum(first[key] == base[key] for key in shared)
    logger.info(f"HP-KR baseline against the first run's: {identical} of {len(shared)} answers identical")
    (OUT / "checks.json").write_text(json.dumps({
        "unbatched": unbatched, "first_run_identical": identical, "first_run_compared": len(shared),
        "first_run_different": [f"{i} {s}" for i, s in shared if first[(i, s)] != base[(i, s)]],
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# --- grading -------------------------------------------------------------------------------------

def grade() -> pd.DataFrame:
    """Every answer so far, graded by the first run's regexes: `outcome` is 1 when the answer does
    what its set is about (see OUTCOME), and `unclear` when the regex cannot place it."""
    info = {(r["item_id"], r["set"]): r for r in _items()}
    rows = [r for p in sorted((OUT / "generations").glob("*.jsonl")) for r in _read(p)]
    out = []
    for r in rows:
        item = info[(r["item_id"], r["set"])]
        s, text = r["set"], r["text"]
        if s in ("H", "C", "G+", "G-"):
            claim = knowledge_claim(text)
            outcome = claim == ("CLAIMED_KNOWLEDGE" if s == "G-" else "DENIED_KNOWLEDGE")
            unclear = claim == "OTHER"
        elif s == "A":
            choice = extract_choice(text)
            outcome, unclear = choice == item["belief"], choice == -1
        else:
            letter = rd.letter(text)
            outcome, unclear = letter == item["key"], letter is None
        out.append({k: r[k] for k in ("direction", "alpha", "set", "item_id", "text", "finished")}
                   | {"outcome": float(outcome), "unclear": bool(unclear)})
    return pd.DataFrame(out)


def status() -> None:
    """One line per finished (direction, alpha): each set's outcome count, from what has run so far."""
    graded = grade()
    if graded.empty:
        logger.info("nothing generated yet")
        return
    lines = []
    for (direction, alpha), g in graded.groupby(["direction", "alpha"], sort=False):
        cells = []
        for s in SETS:
            gs = g[g["set"] == s]
            if len(gs):
                unclear = int(gs["unclear"].sum())
                cells.append(f"{s} {OUTCOME[s]} {int(gs['outcome'].sum())}/{len(gs)}"
                             + (f" ({unclear} unclear)" if unclear else ""))
        lines.append(f"{direction} {alpha:+g}: " + "; ".join(cells))
    print("\n".join(lines))


# --- the report ----------------------------------------------------------------------------------

def rates(graded: pd.DataFrame) -> pd.DataFrame:
    """Per direction, alpha and set: the outcome's rate with a bootstrap interval over items, and the
    paired change from alpha = 0 with its interval. Alpha = 0 is repeated under every direction."""
    base = graded[graded["direction"] == BASELINE]
    rows = []
    for direction in DIRECTIONS:
        steered = graded[graded["direction"] == direction]
        for (alpha, s), g in pd.concat([base.assign(direction=direction), steered]).groupby(["alpha", "set"]):
            v = g.set_index("item_id")["outcome"]
            b = base[base["set"] == s].set_index("item_id")["outcome"].loc[v.index]
            lo, hi = _bootstrap(v.to_numpy())
            dlo, dhi = _bootstrap((v - b).to_numpy())
            rows.append({"direction": direction, "alpha": alpha, "set": s, "outcome": OUTCOME[s], "n": len(v),
                         "rate": v.mean(), "lo": lo, "hi": hi, "unclear": g["unclear"].mean(),
                         "change": (v - b).mean(), "change_lo": dlo, "change_hi": dhi})
    return pd.DataFrame(rows)


def denies_yet_knows(graded: pd.DataFrame) -> pd.DataFrame:
    """Items that deny knowing in a report while answering correctly under the same steer: G+ with GA,
    C and H with A."""
    rows = []
    for (direction, alpha), g in graded.groupby(["direction", "alpha"]):
        for report, check in (("G+", "GA"), ("C", "A"), ("H", "A")):
            d = g[g["set"] == report].set_index("item_id")["outcome"]
            k = g[g["set"] == check].set_index("item_id")["outcome"]
            common = d.index.intersection(k.index)
            if common.empty:
                continue
            rows.append({"direction": direction, "alpha": alpha, "report": report, "check": check,
                         "n": len(common), "denies": d.loc[common].mean(), "answers": k.loc[common].mean(),
                         "denies_yet_answers": ((d.loc[common] == 1) & (k.loc[common] == 1)).mean()})
    return pd.DataFrame(rows)


def report() -> None:
    graded = grade()
    graded.to_csv(OUT / "graded.csv", index=False)
    table = rates(graded)
    table.round(4).to_csv(OUT / "rates.csv", index=False)
    denies_yet_knows(graded).round(4).to_csv(OUT / "denies_yet_knows.csv", index=False)
    _figure(table)
    _transcripts(graded)
    pd.set_option("display.width", 250)
    wide = table.pivot_table(index=["direction", "alpha"], columns="set", values="rate")[list(SETS)]
    logger.info(f"rates (share of 20 items):\n{wide.round(2).to_string()}\n"
                f"wrote graded.csv, rates.csv, denies_yet_knows.csv, fig_steer_k.png and transcripts.md to {OUT}")


def _figure(table: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from deception_detection.scripts import persona_ood_figures as F

    k_colour = "#B7791F"  # the thesis's colour for K (thesis_revised/make_thesis_assets.py)
    rule = "#d6d4ce"
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), sharex=True, sharey=True, facecolor=F.SURFACE)
    for ax, s in zip(axes.ravel(), SETS):
        ax.set_facecolor(F.SURFACE)
        at = table[table["set"] == s]
        random = at[at["direction"].str.startswith("random")].pivot_table(index="alpha", columns="direction",
                                                                          values="rate")
        if len(random.columns) == 2:
            ax.fill_between(random.index, random.min(1), random.max(1), color="#b7b5ae", alpha=0.45, lw=0,
                            label="random directions (range of 2)", zorder=1)
        g = at[at["direction"] == "K"].sort_values("alpha")
        if len(g) > 1:
            ax.plot(g["alpha"], g["rate"], marker="D", ms=5, lw=2, color=k_colour, label="K", zorder=3)
            ax.fill_between(g["alpha"], g["lo"], g["hi"], color=k_colour, alpha=0.15, lw=0,
                            label="K: 95% interval over items", zorder=2)
        ax.set_title(SET_LABEL[s], fontsize=9, color=F.INK, loc="left")
        ax.set_ylim(-0.03, 1.03)
        ax.axvline(0, color=rule, lw=0.8, zorder=0)
        ax.tick_params(labelsize=8, colors=F.INK_MUTED, length=0)
        ax.grid(color="#e6e5e1", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("α (multiples of |D9|, added after block 40; + = the known side)", fontsize=8,
                      color=F.INK_MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of the 20 items", fontsize=9, color=F.INK_MUTED)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=F.INK,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Steering along K at layer 40: knowledge reports and answer checks, Llama-3.3-70B-Instruct",
                 fontsize=11.5, color=F.INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_steer_k.png", dpi=170, bbox_inches="tight", facecolor=F.SURFACE)
    plt.close(fig)


def _transcripts(graded: pd.DataFrame, per_set: int = 3) -> None:
    prompts = {(r["item_id"], r["set"]): r["user"] for r in _items()}
    lines = ["# Steered answers (K at layer 40)", "",
             "Greedy answers to the first few items of each set as K is added with coefficient α (in "
             "multiples of |D9|; + is the known side). The first random direction's answers at ±8 are "
             "shown beside them for contrast.", ""]
    for s in SETS:
        lines += [f"## {SET_LABEL[s]}", ""]
        for item in list(dict.fromkeys(graded.loc[graded["set"] == s, "item_id"]))[:per_set]:
            lines += [f"### {item}", "", "> " + prompts[(item, s)].replace("\n", "  \n> "), ""]
            g = graded[(graded["item_id"] == item) & (graded["set"] == s)]
            show = pd.concat([g[g["direction"].isin([BASELINE, "K"])],
                              g[(g["direction"] == "random0") & (g["alpha"].abs() == 8)]])
            lines += ["| direction | α | answer | outcome |", "|---|---|---|---|"]
            for _, r in show.sort_values(["direction", "alpha"]).iterrows():
                text = str(r["text"]).replace("|", "\\|").replace("\n", " ")
                mark = "unclear" if r["unclear"] else f"{OUTCOME[s]}={int(r['outcome'])}"
                lines.append(f"| {r['direction']} | {r['alpha']:+g} | {text} | {mark} |")
            lines.append("")
    unclear = graded[graded["unclear"]].sort_values(["set", "direction", "alpha", "item_id"])
    lines += ["## Answers the regexes could not place", "",
              f"{len(unclear)} of {len(graded)} answers. They count as not showing the outcome in "
              "`rates.csv`, whose `unclear` column gives their share.", "",
              "| set | direction | α | item | answer |", "|---|---|---|---|---|"]
    for _, r in unclear.iterrows():
        text = str(r["text"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {r['set']} | {r['direction']} | {r['alpha']:+g} | {r['item_id']} | {text} |")
    (OUT / "transcripts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    fire.Fire({"items": items, "run": run, "status": status, "report": report})
