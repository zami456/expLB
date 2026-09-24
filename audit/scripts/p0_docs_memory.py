"""Phase 0: inventory of the Claude auto-memory dir (READ-ONLY).
Usage: PYTHONIOENCODING=utf-8 python audit/scripts/p0_docs_memory.py > audit/phase0/memory_inventory.tsv
For each file: name, description (frontmatter), frontmatter 'modified' if any, filesystem mtime, bytes, lines,
F-ids cited, Q-ids cited, other repo docs cited."""
import os, re, datetime
MEM = os.path.expanduser(r"~/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB/memory")
FID = re.compile(r"\bF-(\d{1,3})\b")
QID = re.compile(r"\bQ(\d{1,2}[a-z]?)\b")
DOC = re.compile(r"[\w./-]*(?:FINDINGS|REPORT|DESIGN|TODO|OPEN_DECISIONS|ONBOARDING|BRIEFING|NEXT|REVIEW_PACKET\w*|ROADMAP\w*|PREREGISTRATION|NOVELTY\w*|persona-coordinates-70b-results|sarcasm-pilot-negative-result)\.md")
def fm(txt):
    m = re.match(r"^---\n(.*?)\n---\n", txt, re.S)
    d = {}
    if m:
        for k in ("name", "description", "modified", "type"):
            mm = re.search(rf"^\s*{k}:\s*(.*)$", m.group(1), re.M)
            if mm: d[k] = mm.group(1).strip().strip('"')
    return d
print("\t".join(["file", "bytes", "lines", "fs_mtime", "fm_modified", "fm_type", "F_ids", "Q_ids", "docs_cited", "name", "description"]))
for fn in sorted(os.listdir(MEM)):
    p = os.path.join(MEM, fn)
    if not os.path.isfile(p): continue
    raw = open(p, "rb").read()
    txt = raw.decode("utf-8", "replace").replace("\r\n", "\n")
    d = fm(txt)
    f = sorted({int(x) for x in FID.findall(txt)})
    q = sorted(set(QID.findall(txt)), key=lambda s: (int(re.sub(r"\D", "", s)), s))
    docs = sorted(set(DOC.findall(txt)))
    mt = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")
    print("\t".join([fn, str(len(raw)), str(txt.count("\n")), mt, d.get("modified", ""), d.get("type", ""),
                     ",".join(f"F-{x:02d}" for x in f), ",".join("Q" + x for x in q), ",".join(docs),
                     d.get("name", ""), d.get("description", "")[:220]]))
