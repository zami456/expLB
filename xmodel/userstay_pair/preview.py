"""Render the matched pair's prompts exactly as the deployed generator builds them, before any NDIF run.

For each model's deployed run folder (%USERPROFILE%/xmodel-runs-userstay/<Model>/code): import its 1_generate_ndif.py,
set --prompt_format user_stay, build every one of the 36 x 5 x 30 = 5,400 prompts and
  1. check that every role prompt the fix screen also sent is the SAME message list (bank/screen_items.jsonl, user_stay),
  2. render the default role's 5 variants and one role's 5 variants (question 0) with the model's chat template,
  3. for Llama, check that an explicit empty system message renders exactly as no system message.
Writes preview.md next to this file. Tokenizer only -- no NDIF.

    .venv-aa-win/Scripts/python.exe xmodel/userstay_pair/preview.py
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = Path(os.environ["USERPROFILE"]) / "xmodel-runs-userstay"
MODELS = [("Qwen/Qwen2.5-72B-Instruct", "Qwen2.5-72B-Instruct", "Qwen"),
          ("meta-llama/Llama-3.3-70B-Instruct", "Llama-3.3-70B-Instruct", "Llama")]
SHOW_ROLE = "pc_gatekeeper"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    screen = [json.loads(l) for l in (REPO / "xmodel/qwen_roleplay/bank/screen_items.jsonl").read_text(encoding="utf-8").splitlines() if l]
    screen = {(i["role"], i["prompt_index"], i["question_index"]): i["messages"] for i in screen if i["cond"] == "user_stay"}
    out = ["# user_stay matched pair: prompts as the deployed generator builds them", "",
           "Built by `preview.py` from each run folder's deployed `1_generate_ndif.py` (`--prompt_format user_stay`).", ""]
    for repo_id, folder, short in MODELS:
        code = BASE / folder / "code"
        os.environ["XMODEL_SHORT_NAME"] = short
        gen = load(code / "src/assistant_axis_ndif/1_generate_ndif.py", f"gen_{short}")
        nc = load(code / "src/assistant_axis_ndif/ndif_common.py", f"nc_{short}")
        gen.PROMPT_FORMAT = "user_stay"
        tok = nc.load_tokenizer(repo_id)
        instr_dir = code / "kd/personas/kd_instructions"
        questions = gen.load_questions(code / "kd/questions/kd_questions_shared.jsonl", 30)
        n = same = 0
        for f in sorted(instr_dir.glob("*.json")):
            variants = [d.get("pos", "") for d in json.loads(f.read_text(encoding="utf-8"))["instruction"]]
            for pi, instr in enumerate(variants):
                for qi, q in enumerate(questions):
                    msgs = gen.build_messages(instr, q)
                    assert msgs[0] == {"role": "system", "content": ""} and len(msgs) == 2
                    n += 1
                    ref = screen.get((f.stem, pi, qi))
                    if ref is not None:
                        assert msgs == ref, (f.stem, pi, qi, msgs, ref)
                        same += 1
        out += [f"## {repo_id}", "", f"- {n} prompts built (36 roles x 5 variants x 30 questions); "
                f"{same} of them were also sent in the fix screen and are identical message lists.", ""]
        if short == "Llama":
            q = questions[0]
            a = tok.apply_chat_template([{"role": "system", "content": ""}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
            b = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
            out += [f"- Llama: explicit empty system renders identically to no system message: **{a == b}**", ""]
        for role in ("default", SHOW_ROLE):
            variants = [d.get("pos", "") for d in json.loads((instr_dir / f"{role}.json").read_text(encoding="utf-8"))["instruction"]]
            for pi, instr in enumerate(variants):
                text = tok.apply_chat_template(gen.build_messages(instr, questions[0]), tokenize=False, add_generation_prompt=True)
                out += [f"### {role}, variant {pi} ({instr!r})", "", "```", text, "```", ""]
        print(f"{repo_id}: {n} prompts, {same} identical to the screen")
    (HERE / "preview.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"-> {HERE / 'preview.md'}")


if __name__ == "__main__":
    sys.exit(main())
