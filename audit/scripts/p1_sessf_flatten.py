"""Phase 1 / transcript mining: flatten Claude session JSONL into a grep-able TSV.

Streams each .jsonl line by line (never loads a file into memory) and writes one
TSV row per text chunk:

    session8 \t lineno \t timestamp \t kind \t chunkidx \t text

`text` has tab/newline/backslash escaped (\\t, \\n, \\\\), so every chunk is one
physical line and plain `grep` works.  Chunks are CHUNK characters long.

kind is one of: user, asst, think, sys, tool_use:<ToolName>, tool_res, meta.

Usage:
    python audit/scripts/p1_sessf_flatten.py <srcdir> <outdir> [skip_prefix ...]

Read-only on the transcripts.  Output goes wherever <outdir> points (the audit
uses the session scratchpad, not the repo).
"""
import json
import os
import sys

CHUNK = 1200


def esc(s):
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\r", "").replace("\n", "\\n")


def emit(out, sess, lineno, ts, kind, text):
    if not text:
        return
    text = text.strip()
    if not text:
        return
    n = 0
    for i in range(0, len(text), CHUNK):
        out.write("%s\t%d\t%s\t%s\t%d\t%s\n" % (sess, lineno, ts, kind, n, esc(text[i:i + CHUNK])))
        n += 1


def block_text(b):
    """Return (kind_suffix, text) for one content block."""
    t = b.get("type")
    if t == "text":
        return "asst", b.get("text", "")
    if t == "thinking":
        return "think", b.get("thinking", "")
    if t == "tool_use":
        name = b.get("name", "?")
        inp = b.get("input", {})
        if isinstance(inp, dict):
            parts = []
            for k, v in inp.items():
                if isinstance(v, str):
                    parts.append("%s=%s" % (k, v))
                else:
                    parts.append("%s=%s" % (k, json.dumps(v)[:4000]))
            s = " | ".join(parts)
        else:
            s = str(inp)
        return "tool_use:" + name, s
    if t == "tool_result":
        c = b.get("content")
        if isinstance(c, str):
            return "tool_res", c
        if isinstance(c, list):
            return "tool_res", "\n".join(
                x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
        return "tool_res", json.dumps(c)[:20000] if c is not None else ""
    return None, None


def main():
    srcdir, outdir = sys.argv[1], sys.argv[2]
    skips = tuple(sys.argv[3:])
    os.makedirs(outdir, exist_ok=True)
    idx = open(os.path.join(outdir, "sessions_index.tsv"), "w", encoding="utf-8")
    idx.write("session8\tfile\tbytes\tlines\tfirst_ts\tlast_ts\trec_types\n")
    for fn in sorted(os.listdir(srcdir)):
        if not fn.endswith(".jsonl"):
            continue
        if any(fn.startswith(s) for s in skips):
            print("SKIP", fn)
            continue
        sess = fn[:8]
        path = os.path.join(srcdir, fn)
        outp = os.path.join(outdir, "flat_%s.tsv" % sess)
        first_ts = last_ts = ""
        types = {}
        n = 0
        with open(path, encoding="utf-8", errors="replace") as f, \
                open(outp, "w", encoding="utf-8") as out:
            for lineno, line in enumerate(f, 1):
                n = lineno
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                rt = d.get("type", "?")
                types[rt] = types.get(rt, 0) + 1
                ts = d.get("timestamp") or ""
                if ts:
                    if not first_ts:
                        first_ts = ts
                    last_ts = ts
                if rt in ("user", "assistant"):
                    m = d.get("message")
                    if not isinstance(m, dict):
                        continue
                    role = m.get("role", rt)
                    c = m.get("content")
                    if isinstance(c, str):
                        emit(out, sess, lineno, ts, "user" if role == "user" else "asst", c)
                    elif isinstance(c, list):
                        for b in c:
                            if not isinstance(b, dict):
                                continue
                            k, t = block_text(b)
                            if k:
                                if k == "asst" and role == "user":
                                    k = "user"
                                emit(out, sess, lineno, ts, k, t)
                elif rt == "system":
                    emit(out, sess, lineno, ts, "sys", str(d.get("content", ""))[:20000])
        idx.write("%s\t%s\t%d\t%d\t%s\t%s\t%s\n" % (
            sess, fn, os.path.getsize(path), n, first_ts, last_ts,
            ";".join("%s:%d" % kv for kv in sorted(types.items()))))
        print("done", fn, n, "lines ->", os.path.getsize(outp), "B")
    idx.close()


if __name__ == "__main__":
    main()
