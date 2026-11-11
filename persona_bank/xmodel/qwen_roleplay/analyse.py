"""Read the Qwen role-play test: judge scores plus two judge-free measures, per condition and role group.

  s3      share of replies the judge (Llama-3.1-70B, the roles' own Qwen judge prompts) scores 3 = fully in role
  named   share of replies that name the role (its head noun) -- for D_prefill on the continuation only, so the
          prefill "As a pilot," does not count
  dsim    TF-IDF cosine of a reply (D: its continuation) to Qwen's own NO-ROLE replies to the same question
          (both banks' stored default.jsonl), divided by no-role-vs-no-role cosine for that question;
          1.0 = indistinguishable from Qwen answering with no role
"stored" = the original runs' replies to exactly these items (same roles, wordings and questions; condition A as it
was run 2026-10-07 / 10-10), so A_system against stored is the reproducibility check.

    .venv-analysis/Scripts/python.exe -I xmodel/qwen_roleplay/analyse.py --runs xmodel/qwen_roleplay/runs_smoke
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STORED = {"kd": Path.home() / "xmodel-runs" / "Qwen2.5-72B-Instruct" / "persona",
          "random": REPO / "random_persona_bank" / "results" / "Qwen2.5-72B-Instruct" / "persona"}
CONDS = ["stored", "A_system", "B_user", "C_directive", "D_prefill"]


def load_bank():
    items = [json.loads(l) for l in (HERE / "bank" / "items.jsonl").read_text(encoding="utf-8").splitlines() if l]
    roles = {}
    for i in items:
        roles[i["role"]] = (i["bank"], i["group"], i["prefill"] or roles.get(i["role"], ("", "", ""))[2])
    return items, roles


def noun(prefill: str) -> str:
    return prefill.rstrip(",").split()[-1]  # "As a coverup operator," -> "operator"


def load_rows(runs: Path, items, roles):
    want = {(i["role"], i["prompt_index"], i["question_index"]) for i in items}
    rows = []  # cond, role, group, pi, qi, text_for_measures, score
    for cond in CONDS[1:]:
        for f in sorted((runs / cond / "responses").glob("*.jsonl")) if (runs / cond).exists() else []:
            role = f.stem
            sc_f = runs / cond / "scores" / f"{role}.json"
            sc = json.loads(sc_f.read_text(encoding="utf-8")) if sc_f.exists() else {}
            for line in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                rows.append(dict(cond=cond, role=role, group=roles[role][1], pi=r["prompt_index"],
                                 qi=r["question_index"], text=r["continuation"],
                                 score=sc.get(f"pos_p{r['prompt_index']}_q{r['question_index']}")))
    present = {(r["role"], r["pi"], r["qi"]) for r in rows}
    for role, (bank, group, _) in roles.items():
        sc = json.loads((STORED[bank] / "scores" / f"{role}.json").read_text(encoding="utf-8"))
        for line in (STORED[bank] / "responses" / f"{role}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            k = (role, r["prompt_index"], r["question_index"])
            if k in want and k in present:  # stored rows only for items this run has, so the comparison is paired
                rows.append(dict(cond="stored", role=role, group=group, pi=k[1], qi=k[2],
                                 text=r["conversation"][-1]["content"], score=sc.get(f"pos_p{k[1]}_q{k[2]}")))
    return rows


def default_replies():
    out = defaultdict(list)
    for d in STORED.values():
        for line in (d / "responses" / "default.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["question_index"]].append(r["conversation"][-1]["content"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    a = ap.parse_args()
    items, roles = load_bank()
    rows = load_rows(a.runs, items, roles)
    defaults = default_replies()

    qis = sorted({r["qi"] for r in rows})
    dtexts = [(q, t) for q in qis for t in defaults[q]]
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
    X = vec.fit_transform([t for _, t in dtexts] + [r["text"] for r in rows])
    X = X.multiply(1 / np.sqrt(X.multiply(X).sum(1) + 1e-12)).tocsr()
    nd = len(dtexts)
    dix = defaultdict(list)
    for k, (q, _) in enumerate(dtexts):
        dix[q].append(k)
    base = {}
    for q, ix in dix.items():
        S = (X[ix] @ X[ix].T).toarray(); n = len(ix)
        base[q] = (S.sum() - np.trace(S)) / (n * (n - 1))
    for k, r in enumerate(rows):
        r["dsim"] = float((X[nd + k] @ X[dix[r["qi"]]].T).toarray().mean()) / base[r["qi"]]
        r["named"] = bool(re.search(rf"\b{re.escape(noun(roles[r['role']][2]))}s?\b", r["text"], re.I))

    def cell(sel):
        sc = [r["score"] for r in sel if r["score"] is not None]
        return dict(n=len(sel), judged=len(sc), s3=float(np.mean([s == 3 for s in sc])) if sc else None,
                    named=float(np.mean([r["named"] for r in sel])), dsim=float(np.mean([r["dsim"] for r in sel])))

    out = {}
    fmt = lambda v: "  -  " if v is None else f"{v:5.2f}"
    for level, keyf in (("group", lambda r: r["group"]), ("role", lambda r: r["role"])):
        keys = list(dict.fromkeys(keyf(r) for r in rows if r["cond"] == "stored")) + ["ALL"]
        print(f"\n=== by {level}: s3 / named / dsim  (n judged) ===")
        print(f"{'':22}" + "".join(f"{c:>26}" for c in CONDS))
        for key in keys:
            line = f"{key:22}"
            for cond in CONDS:
                sel = [r for r in rows if r["cond"] == cond and (key == "ALL" or keyf(r) == key)]
                if not sel:
                    line += f"{'':>26}"; continue
                c = cell(sel)
                out[f"{level}|{key}|{cond}"] = c
                line += f"   {fmt(c['s3'])}/{fmt(c['named'])}/{fmt(c['dsim'])} ({c['judged']:3d})"
            print(line)
    (a.runs / "analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> {a.runs / 'analysis.json'}")


if __name__ == "__main__":
    main()
