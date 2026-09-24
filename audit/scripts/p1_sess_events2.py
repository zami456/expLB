"""Phase 1 (extra task), v2: candidate EXPERIMENT/TEST events from the session transcripts.

v1 (p1_sess_events.py) missed inline tests (`python -c`, `python - <<'PY'`), which are
exactly the ones that never reached a file. v2 captures:
  run_script   python/uv run <file>.py [args]
  run_ps1      a .ps1 driver (local or via ssh)
  run_inline   python -c / python - <<HEREDOC   (the body is kept, trimmed)
  sched_task   Register-ScheduledTask / Start-ScheduledTask / schtasks /run
  notebook     nbconvert --execute
  write_script Write tool creating a .py/.ps1
and drops pure ssh/status polls (ssh lines with no python/ps1/task verb).

Each row keeps the tool_result's metric-bearing snippet (by tool_use_id) and the next
assistant text that states a number.

Output: audit/phase1/sessions/run_events2_raw.tsv  (working file; hand-curated afterwards)
Usage:  PYTHONIOENCODING=utf-8 python audit/scripts/p1_sess_events2.py
"""
import json
import os
import re
import sys

SRC = r"C:\Users\SURAIYA\.claude\projects\C--Users-SURAIYA-Documents-STUDY-thesis-expLB"
OUT = os.path.join("audit", "phase1", "sessions")
SKIP = {"4f0b290f-ea7f-4f72-a408-fb581d5c6df4", "1a88b065-7482-40e0-8990-d494b5ad7108"}

RE_SCRIPT = re.compile(
    r"(?:^|[\s;|&(\"'])(?:uv\s+run\s+)?(?:[\w:.\\/~-]*[\\/])?(?:python3?|py)(?:\.exe)?\s+"
    r"(?:-u\s+)?([\w./\\-]+\.py)([^\n;|&]{0,160})")
RE_PS1 = re.compile(r"([\w./\\-]+\.ps1)([^\n;|&\"']{0,120})")
RE_INLINE = re.compile(r"python3?\s+(?:-c\s|-\s*<<|<<)", re.I)
RE_TASK = re.compile(r"(Register-ScheduledTask|Start-ScheduledTask|schtasks\s+/(?:run|create))[^\n]{0,140}", re.I)
RE_NB = re.compile(r"nbconvert[^\n]{0,140}")
METRIC_RE = re.compile(r"[^\n]{0,70}\b(?:AUROC|auroc|AUC|acc(?:uracy)?|q=|p=|cos|delta|gain|mean|median|n=)\b[^\n]{0,60}?\d\.\d{2,4}[^\n]{0,40}")
NUM_RE = re.compile(r"\d\.\d{3,4}")


def blocks(msg):
    if not isinstance(msg, dict):
        return []
    c = msg.get("content")
    if isinstance(c, str):
        return [{"type": "text", "text": c}]
    if isinstance(c, list):
        return c
    return []


def flat(x, limit=8000):
    if isinstance(x, str):
        return x[:limit]
    if isinstance(x, list):
        out = []
        for b in x:
            out.append(b.get("text") or "" if isinstance(b, dict) else str(b))
        return " ".join(out)[:limit]
    return str(x)[:limit]


def one(s, n):
    return re.sub(r"\s+", " ", (s or "").replace("\x00", "")).strip()[:n]


RE_MEAS = re.compile(
    r"roc_auc|AUROC|auroc|bootstrap|LogisticRegression|np\.dot|cosine|corrcoef|ttest|"
    r"permutation|percentile|\.mean\(|\bauc\b|significan|q_bh|pairs_won|proj|axis|probe",
    re.I)
RE_WRAPPER = re.compile(r"(?:\./)?(?:psrun|pyrun)\.sh|ssh\s[^\n]{0,120}powershell", re.I)
POLLER = re.compile(r"stat|status|progress|alive|watch|sync|drive|check|inventory|space|list_|peek|tail|health|cov", re.I)


def classify(cmd):
    """Return list of (kind, signature, detail) for one shell command."""
    out = []
    for m in RE_SCRIPT.finditer(cmd):
        f = m.group(1).replace("\\", "/").split("/")[-1]
        out.append(("run_script", f, m.group(0)))
    for m in RE_PS1.finditer(cmd):
        f = m.group(1).replace("\\", "/").split("/")[-1]
        kind = "poll_ps1" if POLLER.search(f) else "run_ps1"
        out.append((kind, f, m.group(0)))
    if RE_INLINE.search(cmd):
        kind = "run_inline" if RE_MEAS.search(cmd) else "inline_inspect"
        out.append((kind, "INLINE", cmd))
    # lab-PC runs sent through the session's psrun.sh / pyrun.sh wrappers (a PowerShell
    # here-string over ssh): the python call is inside the here-string, so the patterns
    # above never see it.
    if RE_WRAPPER.search(cmd) and not out:
        pys = [p for p in re.findall(r"([\w./\\-]+\.py)\b", cmd)
               if not p.endswith(("psrun.py", "pyrun.py"))]
        if pys:
            f = pys[0].replace("\\", "/").split("/")[-1]
            out.append(("run_lab", f, cmd))
        elif RE_MEAS.search(cmd):
            out.append(("run_inline", "INLINE-LAB", cmd))
    for m in RE_TASK.finditer(cmd):
        out.append(("sched_task", "TASK", m.group(0)))
    for m in RE_NB.finditer(cmd):
        out.append(("notebook", "NBCONVERT", m.group(0)))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    fh_out = open(os.path.join(OUT, "run_events2_raw.tsv"), "w", encoding="utf-8", newline="")
    fh_out.write("session\tline\tts\tkind\tsig\tdesc\tdetail\tresult_snip\tnext_text\n")
    total = 0
    for fn in sorted(os.listdir(SRC)):
        if not fn.endswith(".jsonl") or fn[:-6] in SKIP:
            continue
        sid = fn[:-6][:8]
        pending = {}
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
                        made = []
                        desc = one(inp.get("description") or "", 120)
                        if name in ("Bash", "PowerShell"):
                            cmd = inp.get("command") or ""
                            for kind, sig, det in classify(cmd):
                                made.append((kind, sig, det))
                        elif name == "Write":
                            fp = str(inp.get("file_path") or "").replace("\\", "/")
                            if fp.lower().endswith((".py", ".ps1")):
                                made.append(("write_script", fp.split("/")[-1], fp))
                        if not made:
                            continue
                        idxs = []
                        for kind, sig, det in made:
                            rows.append([sid, str(ln), ts, kind, sig, desc, one(det, 1400), "", ""])
                            idxs.append(len(rows) - 1)
                        tid = b.get("id")
                        if tid:
                            pending[tid] = idxs
                    elif bt == "tool_result":
                        idxs = pending.pop(b.get("tool_use_id"), None)
                        if not idxs:
                            continue
                        txt = flat(b.get("content"))
                        ms = METRIC_RE.findall(txt)
                        if ms:
                            snip = " ;; ".join(ms[:3])
                        else:
                            tail = [x.strip() for x in txt.strip().split("\n") if x.strip()]
                            snip = " | ".join(tail[-3:])
                        for i in idxs:
                            rows[i][7] = one(snip, 300)
                    elif bt == "text" and t == "assistant":
                        txt = b.get("text") or ""
                        m = METRIC_RE.search(txt) or NUM_RE.search(txt)
                        if not m:
                            continue
                        for i in range(len(rows) - 1, max(-1, len(rows) - 40), -1):
                            if not rows[i][8]:
                                s = max(0, m.start() - 80)
                                rows[i][8] = one(txt[s:m.end() + 140], 280)
                                break
        for r in rows:
            fh_out.write("\t".join(x.replace("\t", " ") for x in r) + "\n")
        total += len(rows)
        print("done", fn, len(rows), file=sys.stderr)
    fh_out.close()
    print("wrote run_events2_raw.tsv", total)


if __name__ == "__main__":
    main()
