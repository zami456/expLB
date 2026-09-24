"""Phase 1, runs ON the lab PC. READ-ONLY on owner files; writes only audit_scratch/out/log_index.tsv.
One row per file in C:/Users/user4/expLB/logs (recursive): size, mtime, first/last timestamp found in the text,
number of lines, counts of failure/success markers, the script names mentioned, and the first and last non-empty lines.
Timestamps inside logs are better run-time evidence than file mtimes (copies reset mtimes)."""
import collections, datetime, os, re

ROOT = "C:/Users/user4/expLB/logs"
OUT = "C:/Users/user4/audit_scratch/out/log_index.tsv"
TS = [re.compile(r"(20\d\d[-/]\d\d[-/]\d\d[ T]\d\d:\d\d:\d\d)"), re.compile(r"\b(\d\d-\d\d \d\d:\d\d:\d\d)\b")]
SCRIPT = re.compile(r"([A-Za-z0-9_\-]+\.(?:py|ps1))")
FAIL = re.compile(r"Traceback|Error|ERROR|FAILED|OOM|out of memory|Killed|exit [1-9]|EXIT [1-9]", re.I)
DONE = re.compile(r"\bDone\b|\bDONE\b|complete|\.done|EXIT 0|exit 0")

def clean(s):
    return s.replace("\t", " ").replace("\r", "").strip()[:220]

rows = []
for dp, dn, fn in os.walk(ROOT):
    for f in sorted(fn):
        p = os.path.join(dp, f).replace("\\", "/")
        st = os.stat(p)
        first_ts = last_ts = first = last = ""
        enc = "utf-8"
        n = fails = dones = 0
        scripts = collections.Counter()
        try:
            head = open(p, "rb").read(4096)
            enc = "utf-16" if head[:2] in (bytes([255, 254]), bytes([254, 255])) else ("utf-16-le" if head.count(0) > len(head) // 4 else "utf-8")
            with open(p, encoding=enc, errors="replace") as fh:
                for line in fh:
                    n += 1
                    s = line.strip()
                    if s and not first:
                        first = s
                    if s:
                        last = s
                    for rx in TS:
                        m = rx.search(line)
                        if m:
                            if not first_ts:
                                first_ts = m.group(1)
                            last_ts = m.group(1)
                            break
                    if FAIL.search(line):
                        fails += 1
                    if DONE.search(line):
                        dones += 1
                    for m in SCRIPT.findall(line):
                        scripts[m] += 1
        except OSError:
            pass
        rows.append([p[len(ROOT) + 1:] + ("" if enc == "utf-8" else " [utf16]"), st.st_size, f"{datetime.datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M:%S}",
                     first_ts, last_ts, n, fails, dones, ",".join(k for k, _ in scripts.most_common(4)), clean(first), clean(last)])
with open(OUT, "w", encoding="utf-8", newline="\n") as out:
    out.write("log\tbytes\tmtime\tfirst_ts\tlast_ts\tlines\tfail_lines\tdone_lines\tscripts_mentioned\tfirst_line\tlast_line\n")
    for r in rows:
        out.write("\t".join(str(x) for x in r) + "\n")
print(len(rows), "logs indexed ->", OUT)
