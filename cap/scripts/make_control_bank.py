#!/usr/bin/env python3
"""Assemble the format-matched control bank, reusing the capitulation half already extracted.

THE TEST (Tier 0.4). F-66/F-68 found the six position-holding personas are load-bearing: remove
them from the PCA basis and the advantage vanishes (p = 0.56); contrast against them rather than
bare `default` and you gain +0.045. F-55 found those same personas produce text nearly identical
to no system prompt at all. A reviewer's deflationary reading: they may contribute a
representation of THE INSTRUCTION, not of a disposition. Two papers make that live -- Kumar finds
probes detect the deceptive prompt regime rather than whether deception happened, and Kirch et
al. (App. G) find leaving the instruction in context makes a probe partly learn the prompt.

Three conditions, identical in every way except what the negative-side prompt SAYS:

    HOLDING      the original six position-holding personas        (already run)
    NEUTRAL      six mundane conventions, same five sentence frames, same length
    PARAPHRASE   the same six dispositions, reworded

If the advantage survives with NEUTRAL, the signal is the instruction's presence and F-66's
reading is wrong. If it survives with PARAPHRASE but not NEUTRAL, the signal is the
position-holding CONTENT and F-66 stands.

WHY THIS IS CHEAP. The six capitulation personas and `default` are common to all three
conditions and were extracted in the original cap run. Only the twelve control personas are new,
so this is 1,200 rollouts rather than 2,600.
"""
from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "personas" / "cap_instructions"
OUT = ROOT / "personas" / "cap_instructions_controls"

CAP = ["pc_caver", "pc_deferrer", "pc_appeaser", "pc_second_guesser",
       "pc_consensus_follower", "pc_conflict_avoider"]
HOLD = ["pc_position_holder", "pc_evidence_bound", "pc_steadfast_reviewer",
        "pc_reasoned_defender", "pc_calibrated_holder", "pc_disagreement_tolerant"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    # the twelve NEW personas, which are all that need generating
    for tag in ("neutral", "paraphrase"):
        d = ROOT / "personas" / f"cap_instructions_{tag}"
        for f in sorted(d.glob("*.json")):
            shutil.copy2(f, OUT / f.name)
            n += 1
    # plus capitulation and default, so the generator can be pointed at one directory and the
    # judge can find a rubric for every role it scores
    for r in CAP + ["default"]:
        shutil.copy2(SRC / f"{r}.json", OUT / f"{r}.json")

    neutral = sorted(p.stem for p in (ROOT / "personas" / "cap_instructions_neutral").glob("*.json"))
    para = sorted(p.stem for p in (ROOT / "personas" / "cap_instructions_paraphrase").glob("*.json"))

    groups = {
        "name": "cap-format-controls",
        "description": ("the capitulation half held fixed against three different negative "
                        "sides: the original holding personas, format-matched neutral roles, "
                        "and paraphrases of the holding personas"),
        "subject_model": "meta-llama/Llama-3.3-70B-Instruct",
        "source_role_list": "cap/personas/cap_instructions_controls",
        "question_set": "cap/questions/cap_questions_v3.jsonl (20 items, teacher-forced turn 1)",
        "groups": {
            "capitulation": {"roles": CAP, "note": "unchanged from the original cap bank"},
            "neutral": {"roles": neutral,
                        "note": "format-matched, no stance on keeping or revising an answer"},
            "paraphrase": {"roles": para,
                           "note": "the holding dispositions reworded"},
            "assistant_anchor": {"roles": ["default"], "note": "no role prompt"},
        },
        "contrasts": [
            {"name": "cap_vs_neutral", "pos": ["capitulation"], "neg": ["neutral"],
             "basis": ["capitulation", "neutral", "assistant_anchor"],
             "note": ("THE TEST. If this matches cap_vs_holding, the negative side's CONTENT is "
                      "irrelevant and F-66's signal is the instruction's presence.")},
            {"name": "cap_vs_para", "pos": ["capitulation"], "neg": ["paraphrase"],
             "basis": ["capitulation", "paraphrase", "assistant_anchor"],
             "note": ("THE CONTROL FOR THE TEST. Same content as the original holding personas, "
                      "different words. Should match cap_vs_holding if content is what matters.")},
            {"name": "cap_only_ctrl", "pos": ["capitulation"], "neg": ["assistant_anchor"],
             "note": "the 7-role basis again, for comparability with F-66's cap_only"},
        ],
    }
    p = ROOT / "personas" / "cap_control_groups.json"
    io.open(p, "w", encoding="utf-8").write(json.dumps(groups, indent=2) + "\n")

    print(f"{OUT.name}: {len(list(OUT.glob('*.json')))} persona files "
          f"({n} new controls + {len(CAP)} capitulation + default)")
    print(f"  neutral    ({len(neutral)}): {', '.join(neutral)}")
    print(f"  paraphrase ({len(para)}): {', '.join(para)}")
    print(f"wrote {p.relative_to(ROOT.parent)}")
    print(f"\nTO GENERATE: only the {n} control personas are new; capitulation and default "
          f"already have rollouts in run-cap/responses/")


if __name__ == "__main__":
    main()
