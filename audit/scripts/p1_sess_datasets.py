"""Phase 1 (extra task): dataset construction / change events from the session transcripts.

v2 - selective. Three sources only:
  run   a dataset/persona/question BUILDER was actually executed
        (python .../make_view.py, build_*.py, gen_*.py, *_personas.py, 10_/13_ prep,
        make_control_bank, make_format_controls, build_label_banks, hpkr top-up, ...)
  file  a persona-instruction / question / role-group / dataset file was written or
        edited (Write/Edit/MultiEdit on kd|cap/personas, kd|cap/questions,
        assistant_axis/data/roles, pc_instructions, *questions*.jsonl, role_groups*.json)
  text  the session states a dataset decision in words (11th dataset, matched view,
        top-up, regenerate/discard the bank, new question set, vN questions)

Output: audit/phase1/sessions/dataset_events.csv
Usage:  PYTHONIOENCODING=utf-8 python audit/scripts/p1_sess_datasets.py
"""
import csv
import io
import json
import os
import re
import sys

SRC = r"C:\Users\SURAIYA\.claude\projects\C--Users-SURAIYA-Documents-STUDY-thesis-expLB"
OUT = os.path.join("audit", "phase1", "sessions", "dataset_events.csv")
SKIP = {"4f0b290f-ea7f-4f72-a408-fb581d5c6df4", "1a88b065-7482-40e0-8990-d494b5ad7108"}

BUILDERS = (r"make_view(?:_ac)?|build_knowledge_eval|build_factor_probes|build_sarcasm_natural|"
            r"build_share_L20|build_deceptive_L20|gen_hpkr_denial|gen_multiturn_ndif|"
            r"make_control_bank|make_format_controls|make_labels|kd_personas|cap_personas|"
            r"pc_personas|16_regroup_hf|10_sycophancy_ndif|13_sycophancy_multiturn|"
            r"make_cap_review|make_review_packet|make_review_page|make_cap_page|"
            r"make_persona_appendix|audit_bank|verify_s23|score_summary|eval_datasets|"
            r"1_generate_ndif|build_dump")
RUN_BUILDER = re.compile(
    r"(?:^|[\s;|&(\"'])(?:uv\s+run\s+)?(?:[\w:.\\/~-]*[\\/])?(?:python3?|py)(?:\.exe)?\s+"
    r"(?:-u\s+)?([\w./\\-]*(?:" + BUILDERS + r")\.py)([^\n;|&]{0,200})")
RUN_PS1_DS = re.compile(r"([\w./\\-]*(?:run_hpkr_topup|build_label_banks|run_sarc_pilot|"
                        r"rebuild_banks|run_extra_extractions)\.ps1)([^\n;|&\"']{0,120})", re.I)
FILE_RE = re.compile(
    r"^(?:.*/)?(?:kd|cap)/(?:personas|questions)/|"
    r"assistant_axis/data/roles/|"
    r"src/assistant_axis_ndif/(?:pc_instructions|pc_questions|pc_role_groups|kd_|role_groups)|"
    r"[\w./-]*(?:questions|personas|instructions)[\w./-]*\.(?:jsonl|json|md)$|"
    r"[\w./-]*role_groups[\w./-]*\.json$", re.I)
TEXT_RE = re.compile(
    r"[^\n]{0,120}\b(?:11th dataset|matched view|matched slice|kr-?matched|top-?up(?:ped)?|"
    r"regenerat\w+|re-?generate\w*|discard\w+|new question set|question bank|persona bank|"
    r"rebuil[dt] the bank|v[234] questions|questions v[234]|drop(?:ped)? the .{0,30}dataset|"
    r"re-?score\w* the|filtered out)\b[^\n]{0,160}", re.I)
OPS_RE = re.compile(
    r"\bshard\d?|in-?flight|resum\w+|restart\w*|stage[ -]?1|supervisor|NDIF job|queue|"
    r"crash\w*|the run would|wasting|Colab\b", re.I)
DATA_WORD = re.compile(
    r"\bquestion|persona|bank|corpus|dataset|rows?\b|items?\b|prompt set|instruction|"
    r"variant|label|slice|view\b|top-?up", re.I)
DATASET_NAME = re.compile(
    r"\b(harm-pressure-knowledge-report|HP-?KRm?|sycophancy_mt|sycophancy|mask|ai_liar|ai_audit|"
    r"roleplaying|convincing-game|instructed-deception|sandbagging\w*|alpaca|soft-trigger|"
    r"gender-secret|kd11|cap_questions\w*|sarcasm\w*|wmdp|knowledge_matched|text_matched|"
    r"kd_questions\w*|paper16|own100|hf275|kd bank|cap bank)\b", re.I)


def blocks(msg):
    if not isinstance(msg, dict):
        return []
    c = msg.get("content")
    if isinstance(c, str):
        return [{"type": "text", "text": c}]
    if isinstance(c, list):
        return c
    return []


def one(s, n):
    return re.sub(r"\s+", " ", (s or "").replace("\x00", "")).strip()[:n]


def action_of(txt, default="build"):
    t = txt.lower()
    if re.search(r"discard|abandon|scrap|threw away", t):
        return "discard"
    if re.search(r"regenerat|re-?generate|re-?ran the|rebuil|re-?extract|re-?score", t):
        return "regenerate"
    if re.search(r"top-?up|topped up", t):
        return "top-up"
    if re.search(r"filter|drop|exclude|keep_scores|dedup|--write.*filter", t):
        return "filter"
    if "--write" in t or re.search(r"\bbuild|wrote|create|--out", t):
        return "build"
    return default


def main():
    rows = []
    for fn in sorted(os.listdir(SRC)):
        if not fn.endswith(".jsonl") or fn[:-6] in SKIP:
            continue
        sid = fn[:-6][:8]
        with open(os.path.join(SRC, fn), encoding="utf-8", errors="replace") as src:
            for ln, line in enumerate(src, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("type") not in ("user", "assistant"):
                    continue
                ts = (d.get("timestamp") or "")[:19]
                role = d.get("type")
                for b in blocks(d.get("message")):
                    bt = b.get("type")
                    if bt == "tool_use":
                        name = b.get("name") or ""
                        inp = b.get("input") or {}
                        desc = one(inp.get("description") or "", 130)
                        if name in ("Bash", "PowerShell"):
                            cmd = inp.get("command") or ""
                            hits = [(m.group(1), m.group(0)) for m in RUN_BUILDER.finditer(cmd)]
                            hits += [(m.group(1), m.group(0)) for m in RUN_PS1_DS.finditer(cmd)]
                            for script, frag in hits:
                                ds = DATASET_NAME.search(frag) or DATASET_NAME.search(cmd[:400])
                                rows.append(dict(
                                    when=ts, session=sid, line=ln, source="run",
                                    dataset=(ds.group(0) if ds else ""),
                                    action=action_of(frag),
                                    detail=desc or one(frag, 130),
                                    script=script.replace("\\", "/").split("/")[-1],
                                    evidence_quote=one(frag, 200)))
                        elif name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
                            fp = str(inp.get("file_path") or inp.get("notebook_path") or "").replace("\\", "/")
                            tail = fp.split("expLB/")[-1]
                            if not FILE_RE.search(tail):
                                continue
                            ds = DATASET_NAME.search(tail)
                            body = one(json.dumps(inp.get("content") or inp.get("new_string") or "")[:300], 200)
                            rows.append(dict(
                                when=ts, session=sid, line=ln, source="file",
                                dataset=(ds.group(0) if ds else tail.split("/")[-1]),
                                action=("build" if name == "Write" else "edit"),
                                detail=(desc or "") + " " + tail, script="tool:" + name,
                                evidence_quote=body))
                    elif bt == "text":
                        txt = b.get("text") or ""
                        for m in list(TEXT_RE.finditer(txt))[:2]:
                            frag = m.group(0)
                            # keep only fragments that are about DATA, not about a job restart
                            if OPS_RE.search(frag):
                                continue
                            if not (DATASET_NAME.search(frag) or DATA_WORD.search(frag)):
                                continue
                            ds = DATASET_NAME.search(frag)
                            rows.append(dict(
                                when=ts, session=sid, line=ln, source="text:" + role,
                                dataset=(ds.group(0) if ds else ""),
                                action=action_of(frag, "edit"), detail="", script="",
                                evidence_quote=one(frag, 200)))
        print("scanned", fn, file=sys.stderr)

    # de-duplicate: same session + script/file + first 70 chars of quote
    seen = set()
    ded = []
    for r in sorted(rows, key=lambda r: (r["when"], r["session"], r["line"])):
        k = (r["session"], r["script"], r["dataset"], r["evidence_quote"][:70])
        if k in seen:
            continue
        seen.add(k)
        ded.append(r)
    cols = ["when", "session", "line", "dataset", "action", "detail", "script", "source", "evidence_quote"]
    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in ded:
            w.writerow({c: r[c] for c in cols})
    print("wrote", OUT, len(ded), "rows (from", len(rows), "hits)")


if __name__ == "__main__":
    main()
