"""Phase 1 (extra task): pull candidate EXPERIMENT/TEST events out of the session transcripts.

Streams each .jsonl line by line. For every tool_use that looks like it RAN something
(python/uv/nbconvert/.ps1/ssh/scheduled task) or CREATED a script, it writes one raw row:
  session, line, ts, tool, kind, cmd (trimmed), and the first numeric-looking snippet of
  the matching tool_result (by tool_use_id) plus the next assistant text mentioning a metric.

Raw output (working file, reviewed by hand into session_experiments.csv):
  audit/phase1/sessions/run_events_raw.tsv

Usage:  PYTHONIOENCODING=utf-8 python audit/scripts/p1_sess_events.py
"""
import json
import os
import re
import sys

SRC = r"C:\Users\SURAIYA\.claude\projects\C--Users-SURAIYA-Documents-STUDY-thesis-expLB"
OUT = os.path.join("audit", "phase1", "sessions")
SKIP = {"4f0b290f-ea7f-4f72-a408-fb581d5c6df4", "1a88b065-7482-40e0-8990-d494b5ad7108"}

RUN_RE = re.compile(
    r"(?:^|[;&|\n]\s*|\(\s*)(?:uv\s+run\s+)?(?:python3?|py)\s+[\w./\\-]+\.py"
    r"|nbconvert"
    r"|[\w./\\-]+\.ps1"
    r"|Register-ScheduledTask|Start-ScheduledTask|schtasks\s+/(?:run|create)"
    r"|labpc_run\.py|ssh\s+[^\"']*user4",
    re.I)
METRIC_RE = re.compile(r"\b(?:auroc|auc|accuracy|acc|q\s*=|p\s*=|cos|delta|diff|mean|gain)\b[^\n]{0,80}?\d\.\d{2,4}", re.I)
NUM_RE = re.compile(r"0\.\d{3,4}")


def blocks(msg):
    if not isinstance(msg, dict):
        return []
    c = msg.get("content")
    if isinstance(c, str):
        return [{"type": "text", "text": c}]
    if isinstance(c, list):
        return c
    return []


def flat(x, limit=4000):
    if isinstance(x, str):
        return x[:limit]
    if isinstance(x, list):
        out = []
        for b in x:
            if isinstance(b, dict):
                out.append(b.get("text") or "")
            else:
                out.append(str(b))
        return " ".join(out)[:limit]
    return str(x)[:limit]


def one(s, n):
    return re.sub(r"\s+", " ", s or "").strip()[:n]


def main():
    os.makedirs(OUT, exist_ok=True)
    fh_out = open(os.path.join(OUT, "run_events_raw.tsv"), "w", encoding="utf-8", newline="")
    fh_out.write("session\tline\tts\ttool\tkind\tcmd\tresult_snip\tnext_text\n")
    total = 0
    for fn in sorted(os.listdir(SRC)):
        if not fn.endswith(".jsonl") or fn[:-6] in SKIP:
            continue
        sid = fn[:-6][:8]
        pending = {}       # tool_use_id -> row index in rows
        rows = []
        with open(os.path.join(SRC, fn), encoding="utf-8", errors="replace") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get("type")
                if t not in ("user", "assistant"):
                    continue
                ts = (d.get("timestamp") or "")[:19]
                for b in blocks(d.get("message")):
                    bt = b.get("type")
                    if bt == "tool_use":
                        name = b.get("name") or "?"
                        inp = b.get("input") or {}
                        if name in ("Bash", "PowerShell"):
                            cmd = inp.get("command") or ""
                            if not RUN_RE.search(cmd):
                                continue
                            kind = "run"
                        elif name == "Write":
                            fp = str(inp.get("file_path") or "")
                            if not fp.lower().endswith((".py", ".ps1", ".ipynb")):
                                continue
                            cmd = "WRITE " + fp
                            kind = "write_script"
                        elif name in ("NotebookEdit",):
                            cmd = "NB " + str(inp.get("notebook_path") or "")
                            kind = "notebook"
                        else:
                            continue
                        rows.append([sid, str(ln), ts, name, kind, one(cmd, 420), "", ""])
                        tid = b.get("id")
                        if tid:
                            pending[tid] = len(rows) - 1
                    elif bt == "tool_result":
                        tid = b.get("tool_use_id")
                        i = pending.pop(tid, None)
                        if i is None:
                            continue
                        txt = flat(b.get("content"))
                        m = METRIC_RE.search(txt)
                        snip = m.group(0) if m else ""
                        if not snip:
                            tail = txt.strip().split("\n")
                            snip = " | ".join(x.strip() for x in tail[-3:] if x.strip())
                        rows[i][6] = one(snip, 260)
                    elif bt == "text" and t == "assistant":
                        txt = b.get("text") or ""
                        if not txt:
                            continue
                        m = METRIC_RE.search(txt) or NUM_RE.search(txt)
                        if not m:
                            continue
                        # attach to the most recent row without next_text, within 60 rows
                        for i in range(len(rows) - 1, max(-1, len(rows) - 60), -1):
                            if not rows[i][7]:
                                s = max(0, m.start() - 60)
                                rows[i][7] = one(txt[s:m.end() + 120], 240)
                                break
        for r in rows:
            fh_out.write("\t".join(x.replace("\t", " ") for x in r) + "\n")
        total += len(rows)
        print("done", fn, len(rows), "events", file=sys.stderr)
    fh_out.close()
    print("wrote", os.path.join(OUT, "run_events_raw.tsv"), total, "rows")


if __name__ == "__main__":
    main()
