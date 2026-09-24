"""Phase 1 (extra task): index the owner's Claude session transcripts.

Streams every .jsonl in the Claude projects dir line by line (json.loads per line,
never the whole file in memory) and writes one row per session:
  file, first/last timestamp, records, user msgs, assistant msgs, tool calls,
  top tools, top paths/scripts mentioned, lab-PC indicator, first user message.

Usage (from repo root):
  PYTHONIOENCODING=utf-8 python audit/scripts/p1_sess_index.py

Output: audit/phase1/sessions/sessions_index.tsv
Read-only on the transcripts; writes only under audit/phase1/sessions/.
"""
import json
import os
import re
import sys
from collections import Counter

SRC = r"C:\Users\SURAIYA\.claude\projects\C--Users-SURAIYA-Documents-STUDY-thesis-expLB"
OUT = os.path.join("audit", "phase1", "sessions")
SKIP = {"4f0b290f-ea7f-4f72-a408-fb581d5c6df4", "1a88b065-7482-40e0-8990-d494b5ad7108"}

PATH_RE = re.compile(r"[A-Za-z0-9_\-./\\]*[A-Za-z0-9_\-]+\.(?:py|ps1|ipynb|jsonl|json|csv|pt|npz|md|log|parquet)\b")
RUN_RE = re.compile(r"\brun-[a-z0-9_\-]+(?:/[A-Za-z0-9_\-]+)?")
LAB_RE = re.compile(r"100\.85\.217\.68|user4|ssh -|Register-ScheduledTask|Start-ScheduledTask|schtasks", re.I)
NDIF_RE = re.compile(r"\bndif\b|nnsight|remote=True", re.I)

META_PREFIXES = ("<local-command", "<command-name>", "<command-message>", "<command-args>",
                 "Caveat:", "<system-reminder>", "[Request interrupted")


def blocks(msg):
    if not isinstance(msg, dict):
        return []
    c = msg.get("content")
    if isinstance(c, str):
        return [{"type": "text", "text": c}]
    if isinstance(c, list):
        return c
    return []


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for fn in sorted(os.listdir(SRC)):
        if not fn.endswith(".jsonl"):
            continue
        sid = fn[:-6]
        if sid in SKIP:
            continue
        path = os.path.join(SRC, fn)
        n = nuser = nasst = ntool = 0
        uuids = set()
        first_ts = last_ts = None
        tools = Counter()
        paths = Counter()
        runs = Counter()
        lab = 0
        ndif = 0
        first_user = ""
        user_texts = []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ts = d.get("timestamp")
                if ts:
                    if first_ts is None or ts < first_ts:
                        first_ts = ts
                    if last_ts is None or ts > last_ts:
                        last_ts = ts
                t = d.get("type")
                if t not in ("user", "assistant"):
                    continue
                if ts and d.get("uuid"):
                    uuids.add((ts, d["uuid"]))
                for b in blocks(d.get("message")):
                    bt = b.get("type")
                    if bt == "text":
                        txt = b.get("text") or ""
                        if t == "user":
                            s = txt.strip()
                            if s and not s.startswith(META_PREFIXES) and not d.get("isMeta"):
                                nuser += 1
                                user_texts.append(s)
                                if not first_user:
                                    first_user = s[:400]
                        else:
                            nasst += 1
                        if LAB_RE.search(txt):
                            lab += 1
                        if NDIF_RE.search(txt):
                            ndif += 1
                    elif bt == "tool_use":
                        ntool += 1
                        tools[b.get("name") or "?"] += 1
                        blob = json.dumps(b.get("input"))[:6000]
                        if LAB_RE.search(blob):
                            lab += 1
                        if NDIF_RE.search(blob):
                            ndif += 1
                        for m in PATH_RE.findall(blob):
                            pass
                        for m in PATH_RE.finditer(blob):
                            p = m.group(0).replace("\\\\", "/").replace("\\", "/").lstrip("./")
                            if len(p) > 3 and not p.startswith(("audit/",)):
                                paths[p] += 1
                        for m in RUN_RE.finditer(blob):
                            runs[m.group(0)] += 1
        top_paths = "; ".join("%s(%d)" % (p, c) for p, c in paths.most_common(12))
        top_runs = "; ".join("%s(%d)" % (p, c) for p, c in runs.most_common(8))
        top_tools = "; ".join("%s(%d)" % (p, c) for p, c in tools.most_common(8))
        rows.append(dict(
            uuids=uuids,
            file=fn, session=sid[:8], first_ts=first_ts or "", last_ts=last_ts or "",
            records=n, user_msgs=nuser, asst_msgs=nasst, tool_calls=ntool,
            top_tools=top_tools, top_paths=top_paths, top_runs=top_runs,
            lab_hits=lab, ndif_hits=ndif,
            first_user=first_user.replace("\t", " ").replace("\n", " | ")[:400],
            bytes=os.path.getsize(path)))
        print("done", fn, n, "records", file=sys.stderr)

    # a session whose records are a subset of another's is a fork/duplicate transcript
    for a in rows:
        a["dup_of"] = ""
        if not a["uuids"]:
            a["dup_of"] = "STUB (no user/assistant records)"
            continue
        for b in rows:
            if a is b or not b["uuids"]:
                continue
            if a["uuids"] <= b["uuids"] and len(a["uuids"]) < len(b["uuids"]):
                a["dup_of"] = "subset of " + b["session"]
                break
    for r in rows:
        del r["uuids"]

    rows.sort(key=lambda r: r["first_ts"] or "")
    cols = ["file", "session", "first_ts", "last_ts", "bytes", "records", "user_msgs",
            "asst_msgs", "tool_calls", "lab_hits", "ndif_hits", "dup_of", "top_tools",
            "top_runs", "top_paths", "first_user"]
    with open(os.path.join(OUT, "sessions_index.tsv"), "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]).replace("\t", " ") for c in cols) + "\n")
    print("wrote", os.path.join(OUT, "sessions_index.tsv"), len(rows), "rows")


if __name__ == "__main__":
    main()
