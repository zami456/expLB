#!/usr/bin/env python3
"""Generate thesis Appendix A -- every persona prompt and every question -- from source.

Modelled on kd/scripts/make_persona_appendix.py. Nothing in the output is hand-copied: every
prompt, question, count, line number and commit id in thesis/appendix/appendix_A.tex is read
here from the banks' definition files, the cached rollouts, git, the paper PDF and the
co-author's roster table. Re-run after any change to a definition file:

    python thesis_tools/appendix/make_appendix_personas.py

Banks covered (Section sec:m-banks of Chapter 4):
  * kd       kd/personas/kd_role_groups.json, kd/personas/kd_instructions/*.json,
             kd/questions/kd_questions_shared.jsonl. paper16's 26 roles are its first five groups;
             paper16's own files (src/assistant_axis_ndif/pc_instructions/, pc_role_groups.json,
             pc_questions_deception.jsonl, pc_questions_sycophancy.jsonl) are compared against it.
  * cap      cap/personas/cap_role_groups.json, cap/personas/cap_instructions/*.json,
             cap/questions/cap_questions_v3.jsonl, and the challenge constant in
             cap/scripts/gen_multiturn_ndif.py.
  * sarcasm  cap/personas/sarcasm_pilot/*.json, cap/questions/sarcasm_pilot_v1.jsonl.

Checks made before anything is written (their results go into the text and the % src lines):
  * every role named in a groups file has an instruction file with five variants;
  * paper16's 26 instruction files carry the same five variants as the kd bank's copies;
  * each cached rollout carries exactly the printed system prompt and question (and, for cap,
    the fixed first answer and the challenge) -- the run-time version check;
  * the last commit touching each bank's files, and that the files are unchanged since;
  * paper16's stored role descriptions and sycophancy questions against the paper PDF (pypdf);
  * the role list against Pipeline B's roster table at the pinned commit 16def2de.

Read-only: git is called with log, diff --quiet, ls-tree and show only.
Writes exactly one file: thesis/appendix/appendix_A.tex.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis" / "appendix" / "appendix_A.tex"

# ---- inputs -------------------------------------------------------------------------------------
KD_GROUPS = "kd/personas/kd_role_groups.json"
KD_INSTR = "kd/personas/kd_instructions"
KD_QS = "kd/questions/kd_questions_shared.jsonl"
KD_ROLLOUTS = "audit/cache/labpc/run-kd/responses"
KD_LAB = "C:/Users/user4/expLB/run-kd/responses"

P16_GROUPS = "src/assistant_axis_ndif/pc_role_groups.json"
P16_INSTR = "src/assistant_axis_ndif/pc_instructions"
P16_DEC = "src/assistant_axis_ndif/pc_questions_deception.jsonl"
P16_SYC = "src/assistant_axis_ndif/pc_questions_sycophancy.jsonl"
P16_ROLLOUTS = "run-paper16/responses"
P16_ROLLOUTS_CACHE = "audit/cache/labpc/run-paper16/responses"
P16_LAB = "C:/Users/user4/expLB/run-paper16/responses"
P16_CHAIN = "src/assistant_axis_ndif/run_paper16_chain.ps1"
GEN_SCRIPT = "src/assistant_axis_ndif/1_generate_ndif.py"

CAP_GROUPS = "cap/personas/cap_role_groups.json"
CAP_INSTR = "cap/personas/cap_instructions"
CAP_QS = "cap/questions/cap_questions_v3.jsonl"
CAP_QS_V1 = "cap/questions/cap_questions.jsonl"
CAP_CONTROLS = "cap/personas/cap_control_groups.json"
CAP_GEN = "cap/scripts/gen_multiturn_ndif.py"
CAP_ROLLOUTS = "audit/p17/labcache/expLB/run-cap/responses"
CAP_LAB = "C:/Users/user4/expLB/run-cap/responses"

SARC_INSTR = "cap/personas/sarcasm_pilot"
SARC_QS = "cap/questions/sarcasm_pilot_v1.jsonl"
SARC_RUN = "cap/scripts/run_sarc_pilot.ps1"
SARC_ROLLOUTS = "audit/cache/labpc/cap/run-sarc-pilot/responses"
SARC_LAB = "C:/Users/user4/expLB/cap/run-sarc-pilot/responses"

PAPER_PDF = "papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf"
COLLAB_SHA = "16def2de"
COLLAB_ROSTER = "src/probes/deception-detection/paper/generated/tab_roster.tex"

# The definition-file commits the plan expects (CH5_RESULTS_PLAN.md "Appendices"; CH4 plan
# "Bank versions"). They are CHECKED against git here, never written from this table.
EXPECTED_COMMIT = {"kd": "3dfdb673", "paper16": "9511ba8d", "cap": "dc24e960", "sarcasm": "080a9794"}
BANK_PATHS = {
    "kd": [KD_INSTR, KD_GROUPS, KD_QS],
    "paper16": [P16_INSTR, P16_GROUPS, P16_DEC, P16_SYC],
    "cap": [CAP_INSTR, CAP_GROUPS, CAP_QS],
    "sarcasm": [SARC_INSTR, SARC_QS],
}

ORD = ["first", "second", "third", "fourth", "fifth"]
WARNINGS: list[str] = []
UNHANDLED_CHARS: set[str] = set()


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print("WARNING:", msg)


# ---- LaTeX escaping -----------------------------------------------------------------------------
SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    # OT1 (no fontenc in main.tex) prints < > | as other glyphs; ` would open a quote.
    "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}", "`": r"\textasciigrave{}",
}
UNICODE = {
    "\u2018": "`", "\u2019": "'", "\u201c": "``", "\u201d": "''", "\u2013": "--", "\u2014": "---",
    "\u2026": r"\dots{}", "\u2192": r"$\rightarrow$", "\u2190": r"$\leftarrow$",
    "\u2194": r"$\leftrightarrow$", "\u00d7": r"$\times$", "\u00a0": "~", "\u2212": r"$-$",
    "\u2248": r"$\approx$", "\u2264": r"$\leq$", "\u2265": r"$\geq$", "\u00b1": r"$\pm$",
    "\u2022": r"\textbullet{}", "\u00b7": r"$\cdot$",
}
OPENERS = set(" \t\n([{")


def tex(s: str) -> str:
    """Escape LaTeX specials; straight quotes become `` '' / ` ' by position."""
    out: list[str] = []
    prev = ""
    for ch in s:
        if ch == '"':
            out.append("``" if (prev == "" or prev in OPENERS or prev == "'") else "''")
        elif ch == "'":
            out.append("`" if (prev == "" or prev in OPENERS or prev == '"') else "'")
        elif ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif ch in UNICODE:
            out.append(UNICODE[ch])
        elif ord(ch) > 126:
            # Latin-1 letters are safe with inputenc utf8; anything else is flagged, not guessed.
            if ord(ch) < 256 and unicodedata.category(ch).startswith("L"):
                out.append(ch)
            else:
                UNHANDLED_CHARS.add(ch)
                out.append("?")
        else:
            out.append(ch)
        prev = ch
    return "".join(out)


def tt(path: str) -> str:
    """A path in \\texttt with break points after / and _."""
    return r"\texttt{" + tex(path).replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}") + "}"


def item_text(t: str) -> str:
    """Guard an \\item whose text would start with [ (read as an optional argument)."""
    return ("{}" + t) if t.startswith("[") else t


def num(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


# ---- file helpers -------------------------------------------------------------------------------
def rp(rel: str) -> Path:
    return ROOT / rel


def load_json(rel: str) -> dict:
    return json.loads(rp(rel).read_text(encoding="utf-8"))


def load_jsonl(rel: str | Path) -> list[dict]:
    p = rel if isinstance(rel, Path) else rp(rel)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def line_of(rel: str, pattern: str) -> int:
    """1-based line number of the first line matching a regex; 0 if absent."""
    for i, line in enumerate(rp(rel).read_text(encoding="utf-8").splitlines(), 1):
        if re.search(pattern, line):
            return i
    warn(f"pattern {pattern!r} not found in {rel}")
    return 0


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    # Only commands that never write to the repository (git status can rewrite the index).
    allowed = {"log", "diff", "show", "ls-tree"}
    if args[0] not in allowed:
        raise RuntimeError(f"git {args[0]} is not a read-only command here")
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def worktree_files(paths: list[str]) -> set[str]:
    out: set[str] = set()
    for rel in paths:
        p = rp(rel)
        if p.is_dir():
            out |= {f.relative_to(ROOT).as_posix() for f in p.rglob("*") if f.is_file()}
        elif p.exists():
            out.add(Path(rel).as_posix())
    return out


def bank_commit(bank: str) -> tuple[str, str, bool]:
    """Last commit touching the bank's files, and whether the working tree still equals it.

    Tracked changes: git diff --quiet <sha>. Added or removed files: the working-tree file list
    against git ls-tree at <sha> (no git status, which may write the index).
    """
    paths = BANK_PATHS[bank]
    r = git("log", "-1", "--abbrev=8", "--format=%h|%ad", "--date=short", "--", *paths)
    sha, date = r.stdout.strip().split("|")
    diff = git("diff", "--quiet", sha, "--", *paths)
    tree = git("ls-tree", "-r", "--name-only", sha, "--", *paths)
    same_files = set(tree.stdout.split()) == worktree_files(paths)
    unchanged = diff.returncode == 0 and same_files
    if sha != EXPECTED_COMMIT[bank]:
        warn(f"{bank}: last commit on its definition files is {sha}, plan expects {EXPECTED_COMMIT[bank]}")
    if not unchanged:
        warn(f"{bank}: definition files differ from {sha} in the working tree")
    return sha, date, unchanged


def variants(doc: dict) -> list[str]:
    ins = doc["instruction"]
    ins = ins if isinstance(ins, list) else [ins]
    return [v["pos"] if isinstance(v, dict) else str(v) for v in ins]


def role_title(role: str) -> str:
    if role == "default":
        return "Default assistant"
    name = role[3:] if role.startswith("pc_") else role
    name = name.replace("_", " ")
    return name[:1].upper() + name[1:]


def group_title(key: str) -> str:
    name = key.replace("anti_", "anti-").replace("_", " ")
    return name[:1].upper() + name[1:]


# ---- checks -------------------------------------------------------------------------------------
def load_bank(groups_rel: str, instr_rel: str) -> tuple[dict, dict, list[str]]:
    groups = load_json(groups_rel)["groups"]
    docs, missing = {}, []
    for g in groups.values():
        for r in g["roles"]:
            f = rp(instr_rel) / f"{r}.json"
            if not f.exists():
                missing.append(f"{instr_rel}/{r}.json")
                continue
            docs[r] = json.loads(f.read_text(encoding="utf-8"))
            if len(variants(docs[r])) != 5:
                warn(f"{r}: {len(variants(docs[r]))} variants, not 5")
    for m in missing:
        warn(f"missing role file {m}")
    return groups, docs, missing


def check_single_turn(resp_rel: str, docs: dict, qlists: list[list[str]],
                      fill_name: str | None) -> dict:
    """Rows, mismatches, and which question list each role answered."""
    res = {"rows": 0, "bad": 0, "files": 0, "qset": {}, "sp": {}}
    d = rp(resp_rel)
    if not d.exists():
        warn(f"rollout cache {resp_rel} not found; run-time check skipped")
        return res
    for f in sorted(d.glob("*.jsonl")):
        role = f.stem
        if role not in docs:
            warn(f"{resp_rel}: rollout file {f.name} has no role file")
            continue
        res["files"] += 1
        var = variants(docs[role])
        if fill_name is not None:
            var = [v.replace("{model_name}", fill_name) for v in var]
        used = set()
        for r in load_jsonl(f):
            res["rows"] += 1
            ql = next((i for i, ql in enumerate(qlists)
                       if r["question_index"] < len(ql) and ql[r["question_index"]] == r["question"]),
                      None)
            if ql is None:
                res["bad"] += 1
            else:
                used.add(ql)
            if (r.get("system_prompt") or "") != var[r["prompt_index"]]:
                res["bad"] += 1
            res["sp"].setdefault((role, r["prompt_index"]), r.get("system_prompt"))
        res["qset"][role] = used
    return res


def pdf_pages() -> dict[int, str] | None:
    try:
        import pypdf  # noqa: PLC0415
    except ImportError:
        warn("pypdf not installed; paper checks skipped")
        return None
    reader = pypdf.PdfReader(str(rp(PAPER_PDF)))
    labels = list(getattr(reader, "page_labels", []) or [])
    pages = {}
    for i, p in enumerate(reader.pages):
        label = labels[i] if i < len(labels) else str(i + 1)
        pages[int(label) if label.isdigit() else i + 1] = p.extract_text() or ""
    return pages


def norm(s: str) -> str:
    s = re.sub(r"[\u2019'\ufffd]", "", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def page_range(pages: set[int]) -> str:
    lo, hi = min(pages), max(pages)
    return f"p.~{lo}" if lo == hi else f"pp.~{lo}--{hi}"


def collab_roster() -> dict[str, list[str]] | None:
    r = git("show", f"{COLLAB_SHA}:{COLLAB_ROSTER}")
    if r.returncode != 0:
        warn(f"could not read collaborator roster: {r.stderr.strip()[:200]}")
        return None
    label_to_group = {"deceptive": "deception", "honest": "honesty", "sycophantic": "sycophancy",
                      "anti-sycophantic": "anti_sycophancy", "anchor": "assistant_anchor",
                      "knowledge denial": "knowledge_denial", "sincere ignorance": "sincere_ignorance",
                      "plain assertion": "plain_assertion"}
    out: dict[str, list[str]] = {}
    for line in r.stdout.splitlines():
        if "&" not in line or line.lstrip().startswith("Group"):
            continue
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        if len(cells) < 3:
            continue
        label = re.sub(r"\s*\((published|new)\)", "", cells[0]).strip().lower()
        g = label_to_group.get(label)
        if g is None:
            continue
        names = re.findall(r"([a-z][a-z ]*?)\s*\(\d+\)", cells[2])
        out[g] = sorted(("default" if n.strip() == "default" else "pc_" + n.strip().replace(" ", "_"))
                        for n in names)
    return out


# ---- LaTeX blocks -------------------------------------------------------------------------------
def role_block(role: str, doc: dict, src_file: str, labels: list[str] | None = None,
               empty_note: str = "") -> list[str]:
    src = doc.get("_source") if isinstance(doc.get("_source"), dict) else {}
    desc = src.get("description_verbatim")
    head = rf"\paragraph{{{tex(role_title(role))} ({tt(role)}).}}"
    if desc:
        head += f" Description: ``{tex(desc)}''."
    fields = "instruction[0-4].pos" + (", _source.description_verbatim" if desc else "")
    out = [f"{head} % src: {src_file} :: {fields}", r"\begin{enumerate}", r"\setlength{\itemsep}{0pt}"]
    for i, v in enumerate(variants(doc)):
        body = tex(v) if v.strip() else rf"\emph{{(empty{empty_note})}}"
        lab = f"[{labels[i]}]" if labels else ""
        out.append(rf"\item{lab} {item_text(body)}")
    out += [r"\end{enumerate}", ""]
    return out


def question_list(items: list[tuple[str, str]]) -> list[str]:
    out = [r"\begin{enumerate}", r"\setlength{\itemsep}{0pt}"]
    for qid, text in items:
        out.append(rf"\item[\texttt{{{qid}}}] {text}")
    out += [r"\end{enumerate}", ""]
    return out


# ---- main ---------------------------------------------------------------------------------------
def main() -> None:
    L: list[str] = []
    today = _dt.date.today().isoformat()

    # ===== kd + paper16 ==========================================================================
    kd_groups, kd_docs, kd_missing = load_bank(KD_GROUPS, KD_INSTR)
    kd_q = load_jsonl(KD_QS)
    kd_qtext = [q["question"] for q in kd_q]
    p16_groups, p16_docs, p16_missing = load_bank(P16_GROUPS, P16_INSTR)
    dec_q = load_jsonl(P16_DEC)
    syc_q = load_jsonl(P16_SYC)
    dec_t, syc_t = [q["question"] for q in dec_q], [q["question"] for q in syc_q]

    n_kd_roles = sum(len(g["roles"]) for g in kd_groups.values())
    n_p16_roles = sum(len(g["roles"]) for g in p16_groups.values())
    group_keys = list(kd_groups)
    p16_keys = list(p16_groups)
    shared_groups = group_keys[:len(p16_keys)] == p16_keys and all(
        kd_groups[k]["roles"] == p16_groups[k]["roles"] for k in p16_keys)
    if not shared_groups:
        warn("paper16's groups are not the first groups of the kd bank")
    new_keys = group_keys[len(p16_keys):]
    n_new = sum(len(kd_groups[k]["roles"]) for k in new_keys)
    same_variants = sum(variants(p16_docs[r]) == variants(kd_docs[r]) for r in p16_docs if r in kd_docs)
    if same_variants != len(p16_docs):
        warn(f"only {same_variants}/{len(p16_docs)} paper16 roles share the kd variants")
    embedded_ok = sum(d.get("questions") == kd_qtext for d in kd_docs.values())

    # kd question blocks, computed from the files
    kd_is_dec = kd_qtext[:len(dec_t)] == dec_t
    n_syc_in_kd = 0
    while (len(dec_t) + n_syc_in_kd < len(kd_qtext) and n_syc_in_kd < len(syc_t)
           and kd_qtext[len(dec_t) + n_syc_in_kd] == syc_t[n_syc_in_kd]):
        n_syc_in_kd += 1
    first_new = len(dec_t) + n_syc_in_kd
    if not kd_is_dec:
        warn("kd questions 0-9 are not paper16's deception questions")
    rest_in_p16 = [q for q in kd_qtext[first_new:] if q in dec_t or q in syc_t]
    if rest_in_p16:
        warn("some 'new' kd questions are paper16 questions")

    kd_ids = [q["id"] for q in kd_q]
    dec_ids = [q["id"] for q in dec_q]
    syc_ids = [q["id"] for q in syc_q]

    # run-time checks
    kd_run = check_single_turn(KD_ROLLOUTS, kd_docs, [kd_qtext], "Llama")
    p16_run = check_single_turn(P16_ROLLOUTS, p16_docs, [dec_t, syc_t], "Llama")
    anchor_vars = variants(kd_docs["default"])
    ph_idx = next(i for i, v in enumerate(anchor_vars) if "{model_name}" in v)
    empty_idx = next(i for i, v in enumerate(anchor_vars) if not v.strip())
    kd_anchor_v4 = kd_run["sp"].get(("default", ph_idx))
    p16_anchor_v4 = p16_run["sp"].get(("default", ph_idx))
    p16_cache_same = sum(
        1 for f in rp(P16_ROLLOUTS).glob("*.jsonl")
        if (rp(P16_ROLLOUTS_CACHE) / f.name).exists() and md5(f) == md5(rp(P16_ROLLOUTS_CACHE) / f.name))
    n_p16_files = len(list(rp(P16_ROLLOUTS).glob("*.jsonl")))
    # which question list each paper16 group answered (0 = deception, 1 = sycophancy)
    p16_group_q = {}
    for k in p16_keys:
        sets = {frozenset(p16_run["qset"].get(r, set())) for r in p16_groups[k]["roles"]}
        p16_group_q[k] = sets.pop() if len(sets) == 1 else None
    dec_groups = [k for k in p16_keys if p16_group_q[k] == frozenset({0})]
    syc_groups = [k for k in p16_keys if p16_group_q[k] == frozenset({1})]
    if len(dec_groups) + len(syc_groups) != len(p16_keys):
        warn(f"paper16 question assignment not clean: {p16_group_q}")

    # descriptions: in the judge prompt, and against the paper
    judged = [r for r, d in kd_docs.items() if isinstance(d.get("_source"), dict)
              and d["_source"].get("description_verbatim")]
    desc_in_eval = sum(d["_source"]["description_verbatim"] in d.get("eval_prompt", "")
                       for r, d in kd_docs.items() if r in judged)
    pages = pdf_pages()
    paper_roles = [r for r in judged if "Mahadik" in str(kd_docs[r]["_source"].get("paper", ""))]
    desc_verbatim, desc_plus_verb, desc_pages = [], [], set()
    desc_lead: dict[str, int] = {}
    syc_q_pages, dec_topic_pages, n_syc_in_pdf = set(), set(), 0
    if pages:
        np_ = {pg: norm(t) for pg, t in pages.items()}
        # A description is attributed to the FIRST page holding it: Tables 2-5 (pp. 9-10). Page 11's
        # unified-axis tables reprint some roles in other words and are not the source.
        for r in paper_roles:
            d = norm(kd_docs[r]["_source"]["description_verbatim"])
            hit = [pg for pg, t in np_.items() if d in t]
            if hit:
                desc_verbatim.append(r)
                desc_pages.add(min(hit))
                continue
            # the paper may print only a noun phrase: drop up to three leading words of ours
            words = d.split()
            for cut in (1, 2, 3):
                tail = " ".join(words[cut:])
                hit = [pg for pg, t in np_.items() if len(tail.split()) >= 2 and tail in t]
                if hit:
                    desc_plus_verb.append(r)
                    desc_lead[r] = cut
                    desc_pages.add(min(hit))
                    break
            else:
                warn(f"{r}: stored description not found in the paper")
        for q in syc_t:
            hit = [pg for pg, t in np_.items() if norm(q) in t]
            if hit:
                n_syc_in_pdf += 1
                syc_q_pages.update(hit)
        # the paper's deception questions are topic lines; find the page holding the first topic
        dec_topic_pages = {pg for pg, t in pages.items() if "project handoff" in t.lower()}
        if n_syc_in_pdf != len(syc_t):
            warn(f"only {n_syc_in_pdf}/{len(syc_t)} sycophancy questions found in the paper")
    # the three stored descriptions that add a verb: show one pair as an example
    verb_example = None
    if desc_plus_verb and pages:
        r0 = "pc_ombudsman" if "pc_ombudsman" in desc_plus_verb else desc_plus_verb[0]
        full = kd_docs[r0]["_source"]["description_verbatim"]
        verb_example = (" ".join(full.split()[desc_lead[r0]:]), full)

    roster = collab_roster()
    roster_ok = roster is not None and all(
        sorted(kd_groups[k]["roles"]) == roster.get(k) for k in group_keys) and len(roster) == len(group_keys)
    if roster is not None and not roster_ok:
        warn(f"Pipeline B roster differs from kd groups: {roster}")

    kd_sha, kd_date, kd_same = bank_commit("kd")
    p16_sha, p16_date, p16_same = bank_commit("paper16")

    ln_short = line_of(GEN_SCRIPT, r"^SHORT_NAME = ")
    ln_fill = line_of(GEN_SCRIPT, r'\.replace\("\{model_name\}", SHORT_NAME\)')
    ln_ifins = line_of(GEN_SCRIPT, r"^\s+if instruction:")
    ln_chain_syc = line_of(P16_CHAIN, r"pc_questions_sycophancy\.jsonl")
    ln_chain_grp = line_of(P16_CHAIN, r"--groups sycophancy anti_sycophancy")

    # ===== cap ===================================================================================
    cap_groups, cap_docs, cap_missing = load_bank(CAP_GROUPS, CAP_INSTR)
    cap_q = load_jsonl(CAP_QS)
    cap_qtext = [q["question"] for q in cap_q]
    cap_v1 = [q["question"] for q in load_jsonl(CAP_QS_V1)]
    n_cap_roles = sum(len(g["roles"]) for g in cap_groups.values())
    cap_embedded_v1 = sum(d.get("questions") == cap_v1 for d in cap_docs.values())
    cap_embedded_v3 = sum(d.get("questions") == cap_qtext for d in cap_docs.values())
    cap_desc_in_eval = sum(d["_source"]["description_verbatim"] in d.get("eval_prompt", "")
                           for d in cap_docs.values() if isinstance(d.get("_source"), dict))
    cap_judged = sum(isinstance(d.get("_source"), dict) for d in cap_docs.values())
    cap_anchor_same = (cap_docs.get("default") is not None and kd_docs.get("default") is not None
                       and variants(cap_docs["default"]) == variants(kd_docs["default"]))
    gen_src = rp(CAP_GEN).read_text(encoding="utf-8")
    m = re.search(r'^CHALLENGE = "(.*)"$', gen_src, re.M)
    challenge = m.group(1) if m else None
    if challenge is None:
        warn("CHALLENGE constant not found in gen_multiturn_ndif.py")
    ln_chal = line_of(CAP_GEN, r"^CHALLENGE = ")
    ln_forced = line_of(CAP_GEN, r"forced = .*correct_answer")
    ln_chal_use = line_of(CAP_GEN, r'"content": CHALLENGE')
    ln_qdefault = line_of(CAP_GEN, r'add_argument\("--questions".*cap_questions_v3')
    ln_head = line_of(CAP_GEN, r"head = \[\] if system is None")
    # cap rollouts
    cap_rows = cap_bad = cap_literal = 0
    cap_empty_sysmsg = None
    cap_files = 0
    if rp(CAP_ROLLOUTS).exists():
        byid = {q["id"]: q for q in cap_q}
        for f in sorted(rp(CAP_ROLLOUTS).glob("*.jsonl")):
            role = f.stem
            if role not in cap_docs:
                warn(f"cap rollout file {f.name} has no role file")
                continue
            cap_files += 1
            var = variants(cap_docs[role])
            for r in load_jsonl(f):
                cap_rows += 1
                q = byid.get(r["question_index"])
                conv = r["conversation"]
                ok = (q is not None and r["question"] == q["question"]
                      and r["turn1"] == str(q["correct_answer"])
                      and (r.get("system_prompt") or "") == var[r["prompt_index"]]
                      and conv[-2] == {"role": "user", "content": challenge}
                      and conv[-3] == {"role": "assistant", "content": str(q["correct_answer"])})
                cap_bad += not ok
                if "{model_name}" in (r.get("system_prompt") or ""):
                    cap_literal += 1
                if role == "default" and r["prompt_index"] == empty_idx and cap_empty_sysmsg is None:
                    cap_empty_sysmsg = conv[0] == {"role": "system", "content": ""}
    else:
        warn(f"rollout cache {CAP_ROLLOUTS} not found; cap run-time check skipped")
    cap_sha, cap_date, cap_same = bank_commit("cap")
    ctrl = load_json(CAP_CONTROLS)["groups"]
    n_neutral = len(ctrl.get("neutral", {}).get("roles", []))
    n_para = len(ctrl.get("paraphrase", {}).get("roles", []))
    cells: dict[tuple[str, str], int] = {}
    for q in cap_q:
        cells[(q.get("framing"), q.get("kind"))] = cells.get((q.get("framing"), q.get("kind")), 0) + 1
    n_first = sum(v for (fr, _), v in cells.items() if fr == "first")
    n_third = sum(v for (fr, _), v in cells.items() if fr == "third")
    cell_sizes = sorted(set(cells.values()))

    # ===== sarcasm pilot =========================================================================
    sarc_files = sorted(rp(SARC_INSTR).glob("*.json"))
    sarc_docs = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sarc_files}
    sarc_q = load_jsonl(SARC_QS)
    sarc_qtext = [q["question"] for q in sarc_q]
    sarc_nvar = sorted({len(variants(d)) for d in sarc_docs.values()})
    sarc_embedded = sorted({len(d.get("questions", [])) for d in sarc_docs.values()})
    sarc_run = check_single_turn(SARC_ROLLOUTS, sarc_docs, [sarc_qtext], None)
    sarc_sha, sarc_date, sarc_same = bank_commit("sarcasm")
    ln_sinstr = line_of(SARC_RUN, r"^\$INSTR = ")
    ln_sqs = line_of(SARC_RUN, r"^\$QS = ")
    ln_sgen = line_of(SARC_RUN, r"1_generate_ndif\.py.*--questions_file \$QS")
    ln_imprudent = line_of(SARC_RUN, r"imprudent-plan questions")
    ln_escalate = line_of(SARC_RUN, r"FIVE IRONIC VARIANTS ESCALATE")
    sarc_order = ["pc_ironic_critic", "pc_sincere_critic", "pc_sycophant"]
    sarc_order = [r for r in sarc_order if r in sarc_docs] + [r for r in sarc_docs if r not in sarc_order]
    sarc_ids = [q["id"] for q in sarc_q]

    # ===== write =================================================================================
    L += [
        "% Appendix A: persona prompts and questions.",
        f"% GENERATED by thesis_tools/appendix/make_appendix_personas.py on {today}. Do not edit by hand: re-run the script.",
        r"% The \chapter line and \label{app:prompts} are in main.tex.",
        f"% src: {KD_GROUPS} :: groups (order and roles); {KD_INSTR}/*.json :: instruction[].pos, _source, eval_prompt, questions",
        f"% src: {KD_QS} :: question, id ({len(kd_q)} lines)",
        f"% src: {P16_GROUPS} :: groups; {P16_INSTR}/*.json :: instruction[].pos",
        f"% src: {P16_DEC} ({len(dec_q)} lines); {P16_SYC} ({len(syc_q)} lines) :: question, id",
        f"% src: {GEN_SCRIPT} :: {ln_short} (SHORT_NAME), {ln_fill} ({{model_name}} filled), {ln_ifins} (empty prompt sends no system message)",
        f"% src: {P16_ROLLOUTS}/*.jsonl ({n_p16_files} files; {p16_cache_same} md5-identical to lab PC {P16_LAB}, cache {P16_ROLLOUTS_CACHE}) :: system_prompt, prompt_index, question, question_index",
        f"% src: lab PC {KD_LAB}/*.jsonl (cache {KD_ROLLOUTS}) :: system_prompt, prompt_index, question, question_index",
        f"% src: {P16_CHAIN} :: {ln_chain_grp}-{ln_chain_syc} (sycophancy groups on the sycophancy questions)",
        f"% src: {CAP_GROUPS} :: groups; {CAP_INSTR}/*.json :: instruction[].pos, _source, eval_prompt, questions; {CAP_CONTROLS} :: groups",
        f"% src: {CAP_QS} :: id, framing, kind, question, correct_answer ({len(cap_q)} lines); {CAP_QS_V1} (compared with the role files' question lists)",
        f"% src: {CAP_GEN} :: {ln_chal} (CHALLENGE), {ln_qdefault} (--questions default), {ln_forced} (turn 1 = correct_answer), {ln_head} (system message), {ln_chal_use} (challenge turn)",
        f"% src: lab PC {CAP_LAB}/*.jsonl (cache {CAP_ROLLOUTS}) :: system_prompt, question, turn1, conversation",
        f"% src: {SARC_INSTR}/*.json :: instruction[].pos, _arm, _purpose, questions; {SARC_QS} :: id, question, domain, flaw ({len(sarc_q)} lines)",
        f"% src: {SARC_RUN} :: {ln_sinstr} ($INSTR), {ln_sqs} ($QS), {ln_sgen} (generate call), {ln_imprudent}, {ln_escalate}",
        f"% src: lab PC {SARC_LAB}/*.jsonl (cache {SARC_ROLLOUTS}) :: system_prompt, prompt_index, question, question_index",
        f"% src: {PAPER_PDF} :: pages {sorted(desc_pages | syc_q_pages | dec_topic_pages)} (Tables 2-5 and question lists), text extracted with pypdf",
        f"% src: collaborator {COLLAB_ROSTER} @{COLLAB_SHA} :: rows (groups, role names)",
        "% src: git log -1 / git diff --quiet / git ls-tree (file list) on each bank's definition files (asix working tree)",
        "",
    ]

    # ---- intro ----
    src_banks = f"{KD_GROUPS}; {CAP_GROUPS}; {SARC_INSTR}/*.json"
    L += [
        r"This appendix prints every system prompt and every question of the persona banks of Section~\ref{sec:m-banks} except own100: "
        rf"the knowledge-denial bank (kd), which contains the {n_p16_roles} roles of paper16, the capitulation bank (cap) and the sarcasm pilot. % src: {src_banks}",
        r"own100's 100 roles are taken from the published Assistant-Axis role list, so their prompts are not reprinted here. % src: src/assistant_axis_ndif/role_groups.json :: description, source_role_list (assistant_axis/data/roles/instructions), groups (22+22+16+16+8+16 = 100)",
        rf"The script {tt('thesis_tools/appendix/make_appendix_personas.py')} generates it from the banks' definition files, so no text was retyped. % note: this file is that script's output",
        "Each prompt and question appears exactly as stored, with its own spelling and punctuation.",
        rf"Every role has five system-prompt variants, numbered 1--5 here; the rollouts index them 0--4 as \texttt{{prompt\_index}}. % src: {KD_INSTR}/*.json, {CAP_INSTR}/*.json, {SARC_INSTR}/*.json :: instruction (5 entries each); rollouts :: prompt_index",
        "Questions keep the ids of their files.",
        "For each bank we name the commit of the definition files printed here, and we checked every cached rollout against these texts.",
        "",
    ]

    # ---- kd section ----
    grp_counts = ", ".join(f"{group_title(k).lower()} ({len(kd_groups[k]['roles'])})" for k in p16_keys[:-1])
    grp_counts += f" and the {group_title(p16_keys[-1]).lower()} ({len(kd_groups[p16_keys[-1]]['roles'])})"
    new_counts = ", ".join(f"{group_title(k).lower()} ({len(kd_groups[k]['roles'])})" for k in new_keys[:-1])
    new_counts += f" and {group_title(new_keys[-1]).lower()} ({len(kd_groups[new_keys[-1]]['roles'])})"
    dec_names = ", ".join(group_title(k).lower() for k in dec_groups[:-1]) + f" and {group_title(dec_groups[-1]).lower()}"
    syc_names = " and ".join(group_title(k).lower() for k in syc_groups)
    L += [
        r"\section{The knowledge-denial bank and paper16}",
        r"\label{app:prompts-kd}",
        "",
        f"The kd bank has {n_kd_roles} roles in {len(group_keys)} groups. % src: {KD_GROUPS} :: groups",
        f"Its first {len(p16_keys)} groups are paper16's {n_p16_roles} roles: {grp_counts}. % src: {P16_GROUPS} :: groups == first {len(p16_keys)} groups of {KD_GROUPS} (same roles, same order)",
        f"The other {len(new_keys)} groups hold the {n_new} roles we added: {new_counts}. % src: {KD_GROUPS} :: groups",
        rf"paper16 uses the same role prompts: for all {same_variants} of its roles, the five variants in {tt(P16_INSTR + '/')} are identical to those printed below. % verified: {same_variants}/{len(p16_docs)} files, instruction lists equal",
        f"The two banks differ in their questions. % src: {KD_QS}; {P16_DEC}; {P16_SYC}",
        rf"In the kd bank every role answers the same {len(kd_q)} questions (Section~\ref{{app:prompts-kd-q}}). % src: {KD_QS} ({len(kd_q)} lines); {embedded_ok}/{len(kd_docs)} role files store the same list",
        rf"In paper16 the {dec_names} roles answered the {len(dec_q)} deception questions, and the {syc_names} roles answered the {len(syc_q)} sycophancy questions (Section~\ref{{app:prompts-p16-q}}). % src: {P16_ROLLOUTS}/*.jsonl :: question per role file (deception set for {', '.join(dec_groups)}; sycophancy set for {', '.join(syc_groups)}); {P16_CHAIN} :: {ln_chain_grp}-{ln_chain_syc}",
    ]
    if roster is not None:
        L.append(f"Pipeline~B's roster table lists the same {sum(len(v) for v in roster.values())} roles in the same {len(roster)} groups. % src: collaborator {COLLAB_ROSTER} @{COLLAB_SHA} :: rows; role sets equal per group: {roster_ok}")
    L += [
        rf"The kd files printed here are those of commit \texttt{{{kd_sha}}} ({kd_date}), and paper16's those of commit \texttt{{{p16_sha}}} ({p16_date}). % src: git log -1 -- {' '.join(BANK_PATHS['kd'])} ; git log -1 -- {' '.join(BANK_PATHS['paper16'])}",
        ("Neither set of files has changed since." if kd_same and p16_same else r"\todo[inline]{MISSING: the kd or paper16 definition files differ from their commit; re-check.}")
        + " % src: git diff --quiet <commit> and git ls-tree <commit> vs working-tree file list, same paths",
        (rf"The runs used these texts: all {num(kd_run['rows'])} cached kd rollouts and all {num(p16_run['rows'])} paper16 rollouts carry exactly the system prompt and question printed here."
         if kd_run["rows"] and p16_run["rows"] and not kd_run["bad"] and not p16_run["bad"]
         else r"\todo[inline]{MISSING: run-time check of the kd and paper16 rollouts did not pass; see the script output.}")
        + f" % verified: lab PC {KD_LAB} (cache {KD_ROLLOUTS}) {kd_run['rows']} rows, {kd_run['bad']} mismatches; {P16_ROLLOUTS} {p16_run['rows']} rows, {p16_run['bad']} mismatches",
        rf"The anchor's {ORD[ph_idx]} variant holds the placeholder \texttt{{\{{model\_name\}}}}, which the generation script replaces with ``Llama''; its rollouts read ``{tex(kd_anchor_v4 or '')}'' % src: {GEN_SCRIPT} :: {ln_short}, {ln_fill}; lab PC {KD_LAB}/default.jsonl (cache {KD_ROLLOUTS}/default.jsonl) and {P16_ROLLOUTS}/default.jsonl :: system_prompt at prompt_index {ph_idx} = {kd_anchor_v4!r} / {p16_anchor_v4!r}",
        f"Its {ORD[empty_idx]} variant is empty, and then no system message is sent. % src: {GEN_SCRIPT} :: {ln_ifins}; lab PC {KD_LAB}/default.jsonl (cache {KD_ROLLOUTS}/default.jsonl) and {P16_ROLLOUTS}/default.jsonl :: conversation (no system turn at prompt_index {empty_idx})",
        rf"Every role except the anchor also stores a one-line description, which its judge prompt quotes (Section~\ref{{sec:m-vectors}}). % verified: {desc_in_eval}/{len(judged)} kd role files have _source.description_verbatim inside eval_prompt",
    ]
    if pages:
        n_paper = len(paper_roles)
        L.append(rf"For {len(desc_verbatim)} of paper16's {n_paper} judged roles the stored description is the paper's printed wording \parencite[{page_range(desc_pages)}]{{mahadik2026persona}}. % src: {PAPER_PDF} :: Tables 2-5, pages {sorted(desc_pages)}; verbatim after case/punctuation normalisation for {len(desc_verbatim)}/{n_paper}")
        if desc_plus_verb and verb_example:
            names = [role_title(r).lower() for r in desc_plus_verb]
            names_s = ", ".join("the " + n for n in names[:-1]) + f" and the {names[-1]}" if len(names) > 1 else "the " + names[0]
            L.append(rf"For {names_s} the paper prints a noun phrase, such as ``{tex(verb_example[0])}'', and the stored description adds a verb in front (``{tex(verb_example[1])}''). % src: same PDF pages; {', '.join(desc_plus_verb)} :: _source.description_verbatim minus its first word(s) is on the page")
    else:
        L.append(r"\todo[inline]{MISSING: check of the stored descriptions against the paper (pypdf unavailable).}")
    L.append("")

    for k in group_keys:
        roles = kd_groups[k]["roles"]
        L += [rf"\subsection{{{group_title(k)} ({len(roles)} role{'s' if len(roles) != 1 else ''})}}", ""]
        for r in roles:
            if r in kd_docs:
                L += role_block(r, kd_docs[r], f"{KD_INSTR}/{r}.json",
                                empty_note=": no system prompt" if r == "default" else "")
            else:
                L += [rf"\paragraph{{{tex(role_title(r))} ({tt(r)}).}} \todo[inline]{{MISSING: role file {tex(r)}.json}}", ""]

    # kd questions
    L += [
        rf"\subsection{{The {len(kd_q)} kd questions}}",
        r"\label{app:prompts-kd-q}",
        "",
        f"All {n_kd_roles} roles answer these {len(kd_q)} questions. % src: {KD_QS}",
        f"Questions {kd_ids[0]}--{kd_ids[len(dec_t) - 1]} are paper16's deception questions and questions {kd_ids[len(dec_t)]}--{kd_ids[first_new - 1]} its first {n_syc_in_kd} sycophancy questions, in the same order. % verified: {KD_QS} lines 1-{len(dec_t)} == {P16_DEC}; lines {len(dec_t) + 1}-{first_new} == {P16_SYC} lines 1-{n_syc_in_kd}",
        rf"Questions {kd_ids[first_new]}--{kd_ids[-1]} are the {len(kd_q) - first_new} we added (Section~\ref{{sec:m-banks}}). % src: {KD_QS} :: ids {kd_ids[first_new]}-{kd_ids[-1]} (in neither paper16 file)",
        "",
    ]
    L += question_list([(str(q["id"]), tex(q["question"])) for q in kd_q])

    # paper16 questions
    L += [
        r"\subsection{paper16's questions}",
        r"\label{app:prompts-p16-q}",
        "",
    ]
    if pages and n_syc_in_pdf == len(syc_t):
        L.append(rf"Each of the {len(syc_t)} sycophancy questions is printed in the paper \parencite[{page_range(syc_q_pages)}]{{mahadik2026persona}}. % src: {PAPER_PDF} :: pages {sorted(syc_q_pages)}; {n_syc_in_pdf}/{len(syc_t)} found after punctuation normalisation")
    if pages and dec_topic_pages:
        L.append(rf"It gives the deception questions only as one-line topics, so the {len(dec_t)} below are our renderings of them \parencite[{page_range(dec_topic_pages)}]{{mahadik2026persona}}. % src: {PAPER_PDF} :: page {sorted(dec_topic_pages)} (topic list beginning ``project handoff'')")
    L += [
        f"Deception questions {dec_ids[0]}--{dec_ids[-1]} are kd questions {kd_ids[0]}--{kd_ids[len(dec_t) - 1]}, and sycophancy questions {syc_ids[0]}--{syc_ids[n_syc_in_kd - 1]} are kd questions {kd_ids[len(dec_t)]}--{kd_ids[first_new - 1]}; sycophancy questions {syc_ids[n_syc_in_kd]}{' and ' if len(syc_ids) - n_syc_in_kd == 2 else '--'}{syc_ids[-1]} are not in the kd bank. % verified: text equality, {P16_DEC} / {P16_SYC} vs {KD_QS}",
        "",
        rf"\paragraph{{Deception questions ({tt(Path(P16_DEC).name)}).}} % src: {P16_DEC}",
    ]
    L += question_list([(str(q["id"]), tex(q["question"])) for q in dec_q])
    L.append(rf"\paragraph{{Sycophancy questions ({tt(Path(P16_SYC).name)}).}} % src: {P16_SYC}")
    L += question_list([(str(q["id"]), tex(q["question"])) for q in syc_q])

    # ---- cap section ----
    cap_keys = list(cap_groups)
    cap_counts = ", ".join(f"{len(cap_groups[k]['roles'])} {group_title(k).lower().replace(' ', '-')} roles" for k in cap_keys[:-1])
    cap_counts += f" and the {group_title(cap_keys[-1]).lower()}"
    L += [
        r"\section{The capitulation bank}",
        r"\label{app:prompts-cap}",
        "",
        f"The cap bank has {n_cap_roles} roles: {cap_counts}. % src: {CAP_GROUPS} :: groups",
        rf"Each rollout has two turns (Section~\ref{{sec:m-banks}}). % src: {CAP_GROUPS} :: question_set; {CAP_GEN} :: {ln_forced}-{ln_chal_use}",
        rf"The first turn is teacher-forced: the model's first answer is fixed, and only its reply to the challenge is generated. % src: {CAP_GEN} :: {ln_forced}-{ln_chal_use}",
        r"\begin{quote}\small",
        r"\textbf{system:} one of the role's five variants\\",
        r"\textbf{user:} the question\\",
        rf"\textbf{{assistant:}} the question's fixed answer (Section~\ref{{app:prompts-cap-q}})\\",
        rf"\textbf{{user:}} {tex(challenge or '')}\\",
        r"\textbf{assistant:} \emph{(generated)}",
        r"\end{quote}",
        f"% src: {CAP_GEN} :: {ln_chal} (CHALLENGE = {challenge!r}), {ln_forced}-{ln_chal_use} (conversation layout)",
        f"The fixed answer is the question's correct answer as a bare word or phrase, the same for every role. % src: {CAP_GEN} :: {ln_forced}; {CAP_QS} :: correct_answer",
        rf"The files printed here are those of commit \texttt{{{cap_sha}}} ({cap_date}), with version 3 of the questions ({tt(Path(CAP_QS).name)}). % src: git log -1 -- {' '.join(BANK_PATHS['cap'])}; {CAP_GEN} :: {ln_qdefault}",
        ("The files have not changed since." if cap_same else r"\todo[inline]{MISSING: the cap definition files differ from their commit; re-check.}")
        + " % src: git diff --quiet / git ls-tree vs working-tree file list, same paths",
        (rf"The run used these texts: all {num(cap_rows)} cached rollouts carry exactly the printed system prompt, question, fixed answer and challenge."
         if cap_rows and not cap_bad else r"\todo[inline]{MISSING: run-time check of the cap rollouts did not pass; see the script output.}")
        + f" % verified: lab PC {CAP_LAB} (cache {CAP_ROLLOUTS}) {cap_files} files, {cap_rows} rows, {cap_bad} mismatches",
        "Two details differ from the kd bank.",
        rf"First, each cap role file also stores a question list, but it holds an earlier question set ({tt(Path(CAP_QS_V1).name)}); the generation script reads version 3, and the rollouts confirm it. % src: {CAP_INSTR}/*.json :: questions == {CAP_QS_V1} for {cap_embedded_v1}/{len(cap_docs)} files (== v3 for {cap_embedded_v3}); {CAP_GEN} :: {ln_qdefault}",
        rf"Second, this script does not fill the \texttt{{\{{model\_name\}}}} placeholder, so the anchor's {ORD[ph_idx]} variant ran as the literal text ``You are \{{model\_name\}}.''\ in its {cap_literal} rollouts. % src: {CAP_GEN} (no replace of {{model_name}}); lab PC {CAP_LAB}/default.jsonl (cache {CAP_ROLLOUTS}/default.jsonl) :: system_prompt contains '{{model_name}}' in {cap_literal} rows",
    ]
    if cap_empty_sysmsg:
        L.append(f"The anchor's {ORD[empty_idx]} variant is empty; here the message list still holds a system turn with no text, which the model's chat template renders exactly as a conversation without one. % src: {CAP_GEN} :: {ln_head}, 158 (apply_chat_template); lab PC {CAP_LAB}/default.jsonl (cache {CAP_ROLLOUTS}/default.jsonl) :: conversation[0] at prompt_index {empty_idx} = system, ''; verified 2026-09-25 (W9 verifier): the chat_template in tokenizer_config.json of meta-llama/Llama-3.3-70B-Instruct (local HF cache snapshot 6f6073b4) renders [user] and [system '', user] to the same string")
    if cap_anchor_same:
        L.append(rf"The anchor's five variants are those of the kd anchor. % verified: {CAP_INSTR}/default.json instruction == {KD_INSTR}/default.json instruction")
    L += [
        rf"Every other role stores a description that its judge prompt quotes. % verified: {cap_desc_in_eval}/{cap_judged} cap role files have _source.description_verbatim inside eval_prompt",
        rf"The repository also holds control prompts for this bank, {n_neutral} neutral roles and {n_para} reworded holding roles ({tt(CAP_CONTROLS)}); they are not printed here. % src: {CAP_CONTROLS} :: groups.neutral ({n_neutral}), groups.paraphrase ({n_para})",
        "",
    ]
    for k in cap_keys:
        roles = cap_groups[k]["roles"]
        L += [rf"\subsection{{{group_title(k)} ({len(roles)} role{'s' if len(roles) != 1 else ''})}}", ""]
        for r in roles:
            if r in cap_docs:
                L += role_block(r, cap_docs[r], f"{CAP_INSTR}/{r}.json")
            else:
                L += [rf"\paragraph{{{tex(role_title(r))} ({tt(r)}).}} \todo[inline]{{MISSING: role file {tex(r)}.json}}", ""]

    kinds = sorted({q.get("kind") for q in cap_q})
    L += [
        rf"\subsection{{The {len(cap_q)} two-turn questions}}",
        r"\label{app:prompts-cap-q}",
        "",
        f"Of the {len(cap_q)} questions, {n_third} are asked in the third person and {n_first} in the first person; each of the four combinations of person and kind ({' or '.join(kinds)}) holds {cell_sizes[0] if len(cell_sizes) == 1 else cell_sizes}. % src: {CAP_QS} :: framing, kind (cells {dict((f'{a}/{b}', v) for (a, b), v in sorted(cells.items()))})",
        "Each question is followed by its fixed first answer.",
        "",
    ]
    person = {"first": "first person", "third": "third person"}
    L += question_list([
        (str(q["id"]), f"{tex(q['question'])} Fixed answer: \\emph{{{tex(str(q['correct_answer']))}}}. "
                       f"({person.get(q.get('framing'), tex(str(q.get('framing'))))}, {tex(str(q.get('kind')))})")
        for q in cap_q])

    # ---- sarcasm section ----
    arm = {r: (d.get("_arm") or "") for r, d in sarc_docs.items()}
    L += [
        r"\section{The sarcasm pilot}",
        r"\label{app:prompts-sarcasm}",
        "",
        f"The sarcasm pilot has {len(sarc_docs)} roles, one per arm (the kind of answer the role is written to give, named in its file's \\texttt{{\\_arm}} field), each with {sarc_nvar[0] if len(sarc_nvar) == 1 else sarc_nvar} variants, and no anchor. % src: {SARC_INSTR}/*.json ({len(sarc_files)} files) :: instruction, _arm",
        f"All {len(sarc_docs)} answer the same {len(sarc_q)} questions in a single turn, and each question describes an imprudent plan. % src: {SARC_QS} ({len(sarc_q)} lines); {SARC_RUN} :: {ln_imprudent}",
        f"The ironic critic's variants escalate how explicitly they forbid irony markers, from no mention in v1 to an exhaustive ban in v5; we label every role's variants v1--v5. % src: {SARC_INSTR}/pc_ironic_critic.json :: _purpose; {SARC_RUN} :: {ln_escalate}",
        rf"The files printed here are those of commit \texttt{{{sarc_sha}}} ({sarc_date}), and they have not changed since." if sarc_same
        else r"\todo[inline]{MISSING: the sarcasm pilot definition files differ from their commit; re-check.}",
    ]
    L[-1] += f" % src: git log -1 -- {' '.join(BANK_PATHS['sarcasm'])}; git diff --quiet / git ls-tree vs working-tree file list"
    L += [
        (rf"The run used these texts: all {num(sarc_run['rows'])} cached rollouts carry exactly the printed system prompts and questions."
         if sarc_run["rows"] and not sarc_run["bad"] else r"\todo[inline]{MISSING: run-time check of the sarcasm pilot rollouts did not pass.}")
        + f" % verified: lab PC {SARC_LAB} (cache {SARC_ROLLOUTS}) {sarc_run['rows']} rows, {sarc_run['bad']} mismatches",
        rf"Each role file also stores {sarc_embedded[0] if len(sarc_embedded) == 1 else sarc_embedded} questions, which the run did not use: the run script passes {tt(Path(SARC_QS).name)}. % src: {SARC_INSTR}/*.json :: questions ({sarc_embedded}); {SARC_RUN} :: {ln_sqs}, {ln_sgen}",
        f"The rollouts index the questions 0--{len(sarc_q) - 1} by position; the file's ids run {sarc_ids[0]}--{sarc_ids[-1]}. % src: lab PC {SARC_LAB} (cache {SARC_ROLLOUTS}) :: question_index; {SARC_QS} :: id",
        "",
    ]
    for r in sarc_order:
        d = sarc_docs[r]
        a = arm[r]
        a_id, _, a_text = a.partition(" -- ")
        L += [rf"\subsection{{{tex(role_title(r))}}}", ""]
        L += [rf"Role {tt(r)}, arm {tt(a_id.strip())}: ``{tex(a_text.strip())}''. % src: {SARC_INSTR}/{r}.json :: _arm, instruction[0-4].pos"]
        L += [r"\begin{enumerate}", r"\setlength{\itemsep}{0pt}"]
        for i, v in enumerate(variants(d)):
            L.append(rf"\item[v{i + 1}] {item_text(tex(v))}")
        L += [r"\end{enumerate}", ""]

    L += [
        rf"\subsection{{The {len(sarc_q)} sarcasm-pilot questions}}",
        r"\label{app:prompts-sarcasm-q}",
        "",
        f"Each question is followed by the domain and the flaw recorded with it in the file. % src: {SARC_QS} :: domain, flaw",
        "",
    ]
    L += question_list([
        (str(q["id"]), f"{tex(q['question'])} \\emph{{({tex(str(q.get('domain')))}; flaw: {tex(str(q.get('flaw')))})}}")
        for q in sarc_q])

    text = "\n".join(L).rstrip() + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")

    # ---- report ----
    body = [re.sub(r"(?<!\\)%.*$", "", l) for l in text.splitlines()]
    body_chars = sum(len(l) for l in body)
    visible = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", "", "\n".join(body))
    visible = re.sub(r"[{}]", "", visible)
    visible_chars = len(re.sub(r"\s+", " ", visible))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(text):,} characters in total, "
          f"{body_chars:,} without comments, about {visible_chars:,} printed")
    print(f"  page estimate: about {visible_chars / 3000:.1f} pages at ~3,000 printed characters per page")
    print(f"  kd: {n_kd_roles} roles, {len(kd_q)} questions, missing files {kd_missing or 'none'}; "
          f"rollouts {kd_run['rows']} rows / {kd_run['bad']} mismatches; commit {kd_sha} unchanged={kd_same}")
    print(f"  paper16: {n_p16_roles} roles ({same_variants} share kd variants), {len(dec_q)}+{len(syc_q)} questions, "
          f"missing {p16_missing or 'none'}; rollouts {p16_run['rows']} rows / {p16_run['bad']} mismatches; "
          f"{p16_cache_same}/{n_p16_files} md5 = lab cache; commit {p16_sha} unchanged={p16_same}")
    print(f"  paper16 question sets: deception {dec_groups}, sycophancy {syc_groups}")
    print(f"  descriptions: {desc_in_eval}/{len(judged)} in judge prompt; paper verbatim {len(desc_verbatim)}/{len(paper_roles)}, "
          f"verb added {desc_plus_verb}; pages {sorted(desc_pages)}; sycophancy questions in PDF {n_syc_in_pdf}/{len(syc_t)} "
          f"pages {sorted(syc_q_pages)}; deception topics page {sorted(dec_topic_pages)}")
    print(f"  Pipeline B roster @{COLLAB_SHA}: {'matches' if roster_ok else 'DIFFERS or unread'}")
    print(f"  cap: {n_cap_roles} roles, {len(cap_q)} questions, missing {cap_missing or 'none'}; rollouts {cap_rows} rows / "
          f"{cap_bad} mismatches; literal {{model_name}} rows {cap_literal}; role-file questions == v1 for "
          f"{cap_embedded_v1}/{len(cap_docs)}; commit {cap_sha} unchanged={cap_same}; challenge {challenge!r}")
    print(f"  sarcasm: {len(sarc_docs)} roles {sorted(sarc_docs)}, variants {sarc_nvar}, {len(sarc_q)} questions; rollouts "
          f"{sarc_run['rows']} rows / {sarc_run['bad']} mismatches; commit {sarc_sha} unchanged={sarc_same}")
    print(f"  unhandled characters: {sorted(UNHANDLED_CHARS) or 'none'}")
    print(f"  warnings: {len(WARNINGS)}")
    for w in WARNINGS:
        print("   -", w)
    if UNHANDLED_CHARS:
        sys.exit(2)


if __name__ == "__main__":
    main()
