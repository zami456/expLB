"""Read the fix screen (build_screen.py): which wrapper makes Qwen play the 35 kd roles, and at what cost.

Per protocol (orig = Qwen's stored replies to the same 175 items; Llama-3.3 / Llama-405B / Mistral-3.1 / Gemma-3 stored
replies to the same items as references):
  s3        share judged 3 (fully in role); also per kd group and per role (k of 5)
  roles>=1  roles with at least 1 of 5 replies in role (a 1-in-5 rate projects to ~30 of 150, above the s3 bank's 20)
  zero      share judged 0 (no trace of the role; refusals land here)
  refuse    share of replies opening with a refusal ("I can't", "I'm sorry", "As an AI")
  dsim      distance-from-no-role: TF-IDF cosine to Qwen's own no-role replies to the same question / no-role
            baseline (1.0 = as if no role); for the reference models, to THAT model's own no-role replies
  sep       1 - cosine(deception-group centroid, honesty-group centroid) in TF-IDF space: how different the two
            sides of the deception contrast read (higher = more separated)
  len       mean reply length, characters

    .venv-analysis/Scripts/python.exe -I xmodel/qwen_roleplay/analyse_screen.py --runs xmodel/qwen_roleplay/runs_screen
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
RUNS_ROOT = Path.home() / "xmodel-runs"
REFS = {"orig": RUNS_ROOT / "Qwen2.5-72B-Instruct" / "persona",
        "Llama-3.3": HERE / "ref_llama33_kd",  # lab PC run-kd, copied md5-checked (ref_llama33_kd/SOURCES.md)
        "Llama-405B": RUNS_ROOT / "Llama-3.1-405B-Instruct" / "persona",
        "Mistral-3.1": RUNS_ROOT / "Mistral-Small-3.1-24B-Instruct-2503" / "persona",
        "Gemma-3": RUNS_ROOT / "gemma-3-27b-it" / "persona"}
PROTOS = ["stay", "stay_plus", "user_stay"]
REFUSE = re.compile(r"^\s*(i can'?t|i cannot|i'?m sorry|sorry|as an ai|i am an ai|i'?m an ai|i won'?t)", re.I)


def load_items():
    items = [json.loads(l) for l in (HERE / "bank" / "screen_items.jsonl").read_text(encoding="utf-8").splitlines() if l]
    return [i for i in items if i["cond"] == "stay"]  # the 175 (role, variant, question) keys; same in every protocol


def stored(d: Path, keys, group_of):
    rows = []
    for role in sorted({k[0] for k in keys}):
        sc = json.loads((d / "scores" / f"{role}.json").read_text(encoding="utf-8"))
        for line in (d / "responses" / f"{role}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            k = (role, r["prompt_index"], r["question_index"])
            if k in keys:
                rows.append(dict(role=role, group=group_of[role], pi=k[1], qi=k[2],
                                 text=r["conversation"][-1]["content"], score=sc.get(f"pos_p{k[1]}_q{k[2]}")))
    defaults = defaultdict(list)
    for line in (d / "responses" / "default.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        defaults[r["question_index"]].append(r["conversation"][-1]["content"])
    return rows, defaults


def generated(runs: Path, proto: str, group_of):
    rows = []
    for f in sorted((runs / proto / "responses").glob("*.jsonl")):
        sc_f = runs / proto / "scores" / f"{f.stem}.json"
        sc = json.loads(sc_f.read_text(encoding="utf-8")) if sc_f.exists() else {}
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rows.append(dict(role=f.stem, group=group_of[f.stem], pi=r["prompt_index"], qi=r["question_index"],
                             text=r["conversation"][-1]["content"],
                             score=sc.get(f"pos_p{r['prompt_index']}_q{r['question_index']}")))
    return rows


def add_text_measures(rows, defaults):
    """dsim per row (vs this model's own no-role replies) + group-separation, in one TF-IDF space per call."""
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
    cen = {}
    for g in ("deception", "honesty"):
        ix = [nd + k for k, r in enumerate(rows) if r["group"] == g]
        c = np.asarray(X[ix].mean(0)).ravel()
        cen[g] = c / (np.linalg.norm(c) + 1e-12)
    return 1 - float(cen["deception"] @ cen["honesty"])


def summarise(name, rows, sep):
    sc = [r["score"] for r in rows if r["score"] is not None]
    per_role = defaultdict(list)
    for r in rows:
        if r["score"] is not None:
            per_role[r["role"]].append(r["score"] == 3)
    return dict(name=name, n=len(rows), judged=len(sc), s3=float(np.mean([s == 3 for s in sc])) if sc else float("nan"),
                zero=float(np.mean([s == 0 for s in sc])) if sc else float("nan"),
                roles_ge1=sum(any(v) for v in per_role.values()), n_roles=len(per_role),
                refuse=float(np.mean([bool(REFUSE.search(r["text"])) for r in rows])),
                dsim=float(np.mean([r["dsim"] for r in rows])), sep=sep,
                len=float(np.mean([len(r["text"]) for r in rows])),
                groups={g: float(np.mean([r["score"] == 3 for r in rows if r["group"] == g and r["score"] is not None]))
                        for g in sorted({r["group"] for r in rows})},
                roles={k: int(sum(v)) for k, v in per_role.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--llama_runs", type=Path, default=None, help="the same screen generated on Llama-3.3")
    a = ap.parse_args()
    items = load_items()
    group_of = {i["role"]: i["group"] for i in items}
    keys = {(i["role"], i["prompt_index"], i["question_index"]) for i in items}

    res = []
    q_rows, q_defaults = stored(REFS["orig"], keys, group_of)
    res.append(summarise("Qwen orig", q_rows, add_text_measures(q_rows, q_defaults)))
    for p in PROTOS:
        rows = generated(a.runs, p, group_of)
        if rows:
            res.append(summarise(f"Qwen {p}", rows, add_text_measures(rows, q_defaults)))
    for name in ("Llama-3.3", "Llama-405B", "Mistral-3.1", "Gemma-3"):
        rows, defaults = stored(REFS[name], keys, group_of)
        res.append(summarise(f"{name} orig", rows, add_text_measures(rows, defaults)))
        if name == "Llama-3.3" and a.llama_runs:  # the same wrappers on Llama-3.3: would the fix change the thesis model?
            for p in PROTOS:
                rows = generated(a.llama_runs, p, group_of)
                if rows:
                    res.append(summarise(f"Llama-3.3 {p}", rows, add_text_measures(rows, defaults)))

    groups = list(res[0]["groups"])
    print(f"{'':20}{'judged':>7}{'s3':>7}{'roles>=1':>9}{'zero':>7}{'refuse':>7}{'dsim':>7}{'sep':>7}{'len':>7}  "
          + "  ".join(f"{g[:10]:>10}" for g in groups))
    for r in res:
        print(f"{r['name']:20}{r['judged']:7d}{r['s3']:7.2f}{r['roles_ge1']:5d}/{r['n_roles']:<3d}{r['zero']:7.2f}"
              f"{r['refuse']:7.2f}{r['dsim']:7.2f}{r['sep']:7.3f}{r['len']:7.0f}  "
              + "  ".join(f"{r['groups'].get(g, float('nan')):10.2f}" for g in groups))
    print("\nper role, replies in role of 5:")
    roles = sorted(res[0]["roles"], key=lambda k: (group_of[k], k))
    print(f"{'role':28}{'group':18}" + "".join(f"{r['name'].replace(' orig', '').replace('Llama-', 'L')[:11]:>12}" for r in res))
    for k in roles:
        print(f"{k:28}{group_of[k]:18}" + "".join(f"{r['roles'].get(k, -1):12d}" for r in res))
    (a.runs / "analysis_screen.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n-> {a.runs / 'analysis_screen.json'}")


if __name__ == "__main__":
    main()
