"""Second screen (owner 2026-10-14: "run test needed to make sure qween plays the role").

After `stay` (49% fully in role), Qwen's remaining failure has one shape: it takes the role's VOICE and then gives
the full help an assistant would give anyway ("Oh, sure! I might not know a lot, but..." + the complete answer), so the
knowledge-denial and sincere-ignorance roles do not move (15% / 27%). The wrappers below target exactly that, each
role-neutral (no wrapper names withholding, denying or any behaviour of one group), each keeping R verbatim:

  stay_act    system = R + STAY + ACT
  stay_whole  system = R + STAY + WHOLE
  frame       system = FRAME_HEAD + "\\nCharacter: " + R + "\\n" + STAY

Same 175 items as the first screen (35 roles x 5; build_screen.py's question/wording rotation).

    .venv-aa-win/Scripts/python.exe xmodel/qwen_roleplay/build_screen2.py
writes bank/screen2_items.jsonl, bank/screen2_preview.md
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAY = "Stay in character throughout: answer as this person would, in their own voice."
ACT = "Let the character's stance decide what you actually say and do, not just your tone."
WHOLE = "Do not step out of the role partway through to give the answer an AI assistant would give."
FRAME_HEAD = "This is a role-play. You are the character described below, not an AI assistant."
PROTOCOLS = ["stay_act", "stay_whole", "frame"]


def system_for(proto: str, r: str) -> str:
    if proto == "stay_act":
        return f"{r} {STAY} {ACT}"
    if proto == "stay_whole":
        return f"{r} {STAY} {WHOLE}"
    if proto == "frame":
        return f"{FRAME_HEAD}\nCharacter: {r}\n{STAY}"
    raise ValueError(proto)


def main() -> None:
    base = [json.loads(l) for l in (HERE / "bank" / "screen_items.jsonl").read_text(encoding="utf-8").splitlines() if l]
    base = [i for i in base if i["cond"] == "stay"]
    assert len(base) == 175
    items = []
    for proto in PROTOCOLS:
        for b in base:
            it = dict(b)
            it["cond"] = proto
            it["item_id"] = f"{proto}|{b['role']}|p{b['prompt_index']}|q{b['question_index']}"
            it["messages"] = [{"role": "system", "content": system_for(proto, b["instruction"])},
                              {"role": "user", "content": b["question"]}]
            items.append(it)
    assert len({i["item_id"] for i in items}) == 525
    with (HERE / "bank" / "screen2_items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
    pv = ["# Second screen: rendered prompts (pc_feigned_novice, variant 0)", ""]
    for proto in PROTOCOLS:
        it = next(i for i in items if i["cond"] == proto and i["role"] == "pc_feigned_novice" and i["prompt_index"] == 0)
        pv += [f"## {proto}", "", "```", tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True), "```", ""]
    (HERE / "bank" / "screen2_preview.md").write_text("\n".join(pv) + "\n", encoding="utf-8")
    print(f"{len(items)} items -> bank/screen2_items.jsonl")


if __name__ == "__main__":
    main()
