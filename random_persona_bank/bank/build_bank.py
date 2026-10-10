"""Build the random persona bank: `default` + 35 roles drawn at random from the Assistant Axis library.

The control for the kd bank (36 = `default` + 35 purpose-written honest / deceptive / sycophantic /
anti-sycophantic personas). Here the 35 are drawn uniformly, with a fixed seed, from the Assistant Axis
role list (`assistant_axis/data/roles/role_list.json`, 275 roles) minus `assistant` itself, which would
duplicate the `default` anchor. No stance filter (owner 2026-10-10, option 1a): a drawn role that happens
to be deception-flavoured stays in. The draw is made ONCE; it is never re-drawn after looking at it.

Everything except the roles is the kd bank's, so the role set is the only thing that changes:
  - questions: kd/questions/kd_questions_shared.jsonl, byte-identical (the generator reads only this file;
    each role file's own "questions" list is set to the same 30 for consistency with the kd files);
  - `default.json`: kd/personas/kd_instructions/default.json, byte-identical;
  - system prompts: each AA role's own 5 instruction variants, verbatim;
  - judge prompt: each AA role's own `eval_prompt`, verbatim. AA wrote it for Gemma ("an AI assistant,
    Gemma, an LLM"); the deploy step swaps in the judged model's name, as xmodel does for the kd prompts.
    Its 0-3 scale words 0 and 1 differ slightly from the kd prompts'; 2 and 3 -- the only scores the
    s23 and s3 banks keep -- mean the same in both.

Random contrasts R5-R9 (the null for D5-D9). The kd bank's 35 roles fall into six DISJOINT groups --
deception 8, honesty 7, sycophancy 5, anti-sycophancy 5, knowledge denial 4, knowledge candour 6 -- and
D5-D9 are built from them. Each split here deals the 35 random roles into six groups of exactly those
sizes (A 8, B 7, C 5, D 5, E 4, F 6) and builds the same contrasts:
    R5 = A v B (8 v 7)   R6 = C v D (5 v 5)   R7 = A+C v B+D (13 v 12, pooled)
    R8 = E v F (4 v 6)   R9 = A+E v B+F (12 v 13, pooled)
100 splits are fixed here, before any data exists; split 0 is the primary one (built into files like
D5-D9), the rest form the null distribution.

Files are written ASCII-only (json ensure_ascii): the pipeline reads role files with Path.read_text()
and no encoding, i.e. cp1252 on Windows.

    python random_persona_bank/bank/build_bank.py            # writes the bank; refuses to overwrite
    python random_persona_bank/bank/build_bank.py --check    # rebuilds in memory, compares with disk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
AA = REPO / "assistant_axis" / "data" / "roles"
KD = REPO / "kd"
LABELS = REPO / "src" / "assistant_axis_ndif"

SEED = 20261010                 # the draw; the date the owner asked for the bank
N_DRAW = 35                     # + default = 36, the kd bank's size
SPLIT_SEED_BASE = 1_000_000     # split i uses random.Random(SPLIT_SEED_BASE + i)
N_SPLITS = 100
GROUP_SIZES = {"A": 8, "B": 7, "C": 5, "D": 5, "E": 4, "F": 6}   # = the kd groups, in this order
MIRRORS = {"A": "deception", "B": "honesty", "C": "sycophancy", "D": "anti_sycophancy",
           "E": "knowledge_denial", "F": "knowledge candour (sincere_ignorance + plain_assertion)"}
CONTRASTS = {
    "R5": (("A",), ("B",), "D5", "8 v 7"),
    "R6": (("C",), ("D",), "D6", "5 v 5"),
    "R7": (("A", "C"), ("B", "D"), "D7", "13 v 12, pooled"),
    "R8": (("E",), ("F",), "D8", "4 v 6"),
    "R9": (("A", "E"), ("B", "F"), "D9", "12 v 13, pooled"),
}
GEMMA = "an AI assistant, Gemma, an LLM"


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def md5(p: Path) -> str:
    return md5_bytes(p.read_bytes())


def dumps(obj) -> bytes:
    return (json.dumps(obj, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def labels_of(role: str) -> dict:
    out = {}
    for rule in ("loose", "strict"):
        doc = json.loads((LABELS / f"hf275_role_groups_{rule}.json").read_text(encoding="utf-8"))
        out[rule] = sorted(g for g, spec in doc["groups"].items() if role in spec["roles"])
    return out


def build() -> dict[str, bytes]:
    """-> {relative path: file bytes}. Pure: reads inputs, writes nothing."""
    role_list = json.loads((AA / "role_list.json").read_text(encoding="utf-8"))
    pool = sorted(r for r in role_list if r != "assistant")
    assert len(role_list) == 275 and len(pool) == 274, (len(role_list), len(pool))
    drawn = sorted(random.Random(SEED).sample(pool, N_DRAW))

    questions_path = KD / "questions" / "kd_questions_shared.jsonl"
    questions = [json.loads(l)["question"] for l in questions_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(questions) == 30

    files: dict[str, str] = {}
    sources = {}
    for r in drawn:
        src = AA / "instructions" / f"{r}.json"
        d = json.loads(src.read_text(encoding="utf-8"))
        instr = d["instruction"]
        assert len(instr) == 5 and all(set(i) == {"pos"} and i["pos"].strip() for i in instr), r
        assert not any("{" in i["pos"] or "}" in i["pos"] for i in instr), f"{r}: brace in a system prompt"
        ep = d["eval_prompt"]
        assert ep.count(GEMMA) == 3, f"{r}: {ep.count(GEMMA)} x '{GEMMA}'"
        filled = ep.format(question="<Q>", answer="<A>")          # the judge's own call; any stray brace fails here
        assert "<Q>" in filled and "<A>" in filled
        nonascii = sorted({c for c in json.dumps(d, ensure_ascii=False) if ord(c) > 127})
        out = {"instruction": instr, "questions": questions, "eval_prompt": ep,
               "_source": {"bank": "random persona bank (random_persona_bank/)",
                           "aa_role": r, "aa_description": role_list[r],
                           "aa_file": f"assistant_axis/data/roles/instructions/{r}.json", "aa_file_md5": md5(src),
                           "instructions": "AA's 5 variants, verbatim",
                           "eval_prompt": "AA's, verbatim; the model it names is renamed to the judged model at deploy",
                           "questions": "the kd bank's 30 shared questions (kd/questions/kd_questions_shared.jsonl)"}}
        files[f"rpb_instructions/{r}.json"] = dumps(out)
        sources[r] = {"description": role_list[r], "labels_hf275": labels_of(r), "aa_file_md5": md5(src),
                      "non_ascii_chars_in_source": nonascii}

    # default and questions: byte copies of the kd bank's (bytes, so CRLF and all survive)
    files["rpb_instructions/default.json"] = (KD / "personas" / "kd_instructions" / "default.json").read_bytes()
    files["questions/kd_questions_shared.jsonl"] = questions_path.read_bytes()

    groups = {
        "name": "random-persona-bank",
        "description": ("Control bank for the kd bank: the same 30 questions, the same default anchor, and 35 roles "
                        "drawn uniformly at random (seed %d) from the 274 Assistant Axis roles other than 'assistant', "
                        "with no honesty / deception / sycophancy filter." % SEED),
        "source_role_list": "random_persona_bank/bank/rpb_instructions",
        "question_set": "kd/questions/kd_questions_shared.jsonl (byte-identical copy in bank/questions/)",
        "contrasts": [],
        "groups": {
            "assistant_anchor": {"contrast_with": None, "note": "the kd bank's default.json, byte-identical",
                                 "roles": ["default"]},
            "random": {"contrast_with": None,
                       "note": "35 AA roles, random draw; no behaviour groups -- R5-R9 come from bank/splits.json",
                       "roles": drawn},
        },
    }
    files["rpb_role_groups.json"] = dumps(groups)

    splits = []
    for i in range(N_SPLITS):
        perm = random.Random(SPLIT_SEED_BASE + i).sample(drawn, N_DRAW)
        g, at = {}, 0
        for name, size in GROUP_SIZES.items():
            g[name] = sorted(perm[at:at + size])
            at += size
        assert at == N_DRAW and len({r for v in g.values() for r in v}) == N_DRAW
        splits.append({"split": i, "seed": SPLIT_SEED_BASE + i, "groups": g})
    files["splits.json"] = dumps({
        "about": ("Pre-registered random partitions of the 35 drawn roles into six disjoint groups with the kd "
                  "groups' sizes; R5-R9 are built from them exactly as D5-D9 are built from the kd groups. "
                  "Split 0 is the primary (materialised like D5-D9); splits 1-99 are the null distribution. "
                  "Fixed 2026-10-10, before any rollout existed."),
        "group_sizes": GROUP_SIZES, "mirrors": MIRRORS,
        "contrasts": {k: {"positive": list(p), "negative": list(n), "mirrors": m, "sizes": s}
                      for k, (p, n, m, s) in CONTRASTS.items()},
        "seed_rule": f"split i: random.Random({SPLIT_SEED_BASE} + i).sample(sorted(roles), 35), dealt A..F in order",
        "splits": splits})

    files["draw.json"] = dumps({
        "rule": f"random.Random({SEED}).sample(sorted(role_list minus 'assistant'), {N_DRAW}); sorted for storage",
        "seed": SEED, "pool_size": len(pool), "drawn": drawn,
        "role_list": "assistant_axis/data/roles/role_list.json", "role_list_md5": md5(AA / "role_list.json"),
        "kd_default_md5": md5(KD / "personas" / "kd_instructions" / "default.json"),
        "kd_questions_md5": md5(questions_path),
        "labels_note": ("For information only -- the draw ignores labels. Our hf275 labels "
                        "(src/assistant_axis_ndif/hf275_role_groups_{loose,strict}.json, Claude-labelled 2026-09-18)."),
        "roles": sources})
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="rebuild in memory and compare with the bank on disk")
    a = ap.parse_args()
    files = build()
    for rel, data in files.items():
        if not data.isascii():
            sys.exit(f"{rel} is not ASCII")
    if a.check:
        bad = [rel for rel, data in files.items() if (HERE / rel).read_bytes() != data]
        extra = sorted({str(p.relative_to(HERE)).replace("\\", "/") for p in (HERE / "rpb_instructions").glob("*.json")}
                       - set(files))
        print(f"{len(files)} files rebuilt; {len(bad)} differ from disk; {len(extra)} unexpected role files")
        for rel in bad + extra:
            print("  ", rel)
        sys.exit(1 if bad or extra else 0)
    clash = [rel for rel in files if (HERE / rel).exists()]
    if clash:
        sys.exit(f"{len(clash)} bank files exist (e.g. {clash[0]}); refusing to overwrite -- use --check")
    for rel, data in files.items():
        p = HERE / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    for rel, src in (("rpb_instructions/default.json", KD / "personas" / "kd_instructions" / "default.json"),
                     ("questions/kd_questions_shared.jsonl", KD / "questions" / "kd_questions_shared.jsonl")):
        assert md5(HERE / rel) == md5(src), f"{rel} is not a byte copy of {src}"
    manifest = "".join(f"{md5(HERE / rel)}  {rel}\n" for rel in sorted(files))
    (HERE / "MANIFEST.md5").write_text(manifest, encoding="ascii")
    print(f"wrote {len(files)} files + MANIFEST.md5 under {HERE}")


if __name__ == "__main__":
    main()
