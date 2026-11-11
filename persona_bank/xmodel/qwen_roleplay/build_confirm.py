"""Confirm `user_stay` on Qwen's weakest roles (owner 2026-10-14: "run test needed to make sure qween plays the role").

The screen gave each role only 5 items. A role is usable for the s3 bank if at least 20 of its 150 replies are
fully in role (13.3%). Here every role with <= 3 of 5 under user_stay gets 15 NEW items -- each wording 3 times,
questions (j + 6v + {2, 12, 22}) mod 30, disjoint from the screen's (j + 6v) mod 30 -- so each has 20 judged
items in all. Reference: Llama-3.3's original rate on the same role over all its 150 replies (ref_llama33_kd).

    .venv-aa-win/Scripts/python.exe xmodel/qwen_roleplay/build_confirm.py
writes bank/confirm_items.jsonl
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs_screen" / "user_stay" / "scores"


def main() -> None:
    screen = [json.loads(l) for l in (HERE / "bank" / "screen_items.jsonl").read_text(encoding="utf-8").splitlines() if l]
    us = [i for i in screen if i["cond"] == "user_stay"]
    roles = list(dict.fromkeys(i["role"] for i in us))  # build_screen.py's order: role j
    weak = []
    for role in roles:
        sc = json.loads((RUNS / f"{role}.json").read_text(encoding="utf-8"))
        if sum(v == 3 for v in sc.values()) <= 3:
            weak.append(role)
    used = {(i["role"], i["prompt_index"], i["question_index"]) for i in us}
    items = []
    for role in weak:
        j = roles.index(role)
        tmpl = next(i for i in us if i["role"] == role)
        instr = json.loads((HERE / "roles_kd" / f"{role}.json").read_text(encoding="utf-8"))["instruction"]
        qs = {i["question_index"]: i["question"] for i in screen}
        all_q = [json.loads(l)["question"] for l in
                 (HERE.parents[1] / "xmodel/code/kd/questions/kd_questions_shared.jsonl").read_text(encoding="utf-8").splitlines() if l]
        for v in range(5):
            for off in (2, 12, 22):
                qi = (j + 6 * v + off) % 30
                assert (role, v, qi) not in used
                r = instr[v]["pos"].strip()
                q = all_q[qi]
                items.append({**{k: tmpl[k] for k in ("bank", "group")},
                              "item_id": f"user_stay|{role}|p{v}|q{qi}", "cond": "user_stay", "role": role,
                              "prompt_index": v, "question_index": qi, "question": q, "instruction": r,
                              "messages": [{"role": "system", "content": ""},
                                           {"role": "user", "content": tmpl["messages"][1]["content"].split("\n\n")[0]
                                            .replace(tmpl["instruction"], r) + "\n\n" + q}],
                              "prefill": "", "smoke": False})
    assert len({i["item_id"] for i in items}) == len(items) == 15 * len(weak)
    # the user message must be exactly R + " " + STAY + "\n\n" + question, as in the screen
    stay = us[0]["messages"][1]["content"].split("\n\n")[0][len(us[0]["instruction"]) + 1:]
    for it in items:
        assert it["messages"][1]["content"] == f"{it['instruction']} {stay}\n\n{it['question']}"
    with (HERE / "bank" / "confirm_items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(weak)} weak roles: {weak}\n{len(items)} items -> bank/confirm_items.jsonl\nSTAY = {stay!r}")


if __name__ == "__main__":
    main()
