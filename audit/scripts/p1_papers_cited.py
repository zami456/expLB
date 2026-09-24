"""Phase 1 papers: every arXiv id cited in asix docs/code (tracked, excl. vendored/library/data), his branch's
added files, the draft PDF text and the recovered abstract; which have a PDF in papers/ and which a notes file."""
import collections, re, subprocess
RX = re.compile(r"(?:arXiv[:\s/]*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})", re.I)
present = {"2511.16035", "2502.03407", "2605.09391", "2309.15840", "2605.27958", "2609.02893", "2601.10387",
           "2511.17408", "2503.03750", "2606.15821", "2602.20273"}
cites = collections.defaultdict(list)
def scan(ref, paths):
    out = subprocess.run(["git", "grep", "-n", "-I", "-i", "-E", r"arxiv[: /]*[0-9]{4}\.[0-9]{4,5}|arxiv\.org/(abs|pdf)/[0-9]{4}\.[0-9]{4,5}", ref, "--"] + paths,
                         capture_output=True).stdout.decode("utf-8", "replace")
    for line in out.splitlines():
        loc, _, text = line.partition(":")[2].partition(":")[2], None, None
        parts = line.split(":", 3)
        where = f"{parts[0]}:{parts[1]}:{parts[2]}"
        for aid in RX.findall(parts[3] if len(parts) > 3 else ""):
            cites[aid].append((where, (parts[3] if len(parts) > 3 else "").strip()[:160]))
scan("HEAD", [".", ":!src/probes/deception-detection", ":!docs/concepts", ":!docs/usage", ":!docs/models", ":!docs/remote",
              ":!docs/patterns", ":!docs/errors", ":!docs/gotchas", ":!docs/reference", ":!docs/developing", ":!data", ":!results", ":!NNsight.md", ":!papers"])
scan("origin/feat/persona-pc-ood", ["src/probes/deception-detection/results", "src/probes/deception-detection/deception_detection/scripts",
                                    "src/probes/deception-detection/deception_detection/persona_pcs.py", "src/probes/deception-detection/deceptive"])
for label, path in (("draft PDF", "audit/phase0/draft_text.txt"), ("abstract 09-21", "audit/phase0/collab_abstract_2026-09-21.md")):
    for i, line in enumerate(open(path, encoding="utf-8"), 1):
        for aid in RX.findall(line):
            cites[aid].append((f"{label}:{i}", line.strip()[:160]))
with open("audit/phase1/papers_cited_raw.tsv", "w", encoding="utf-8") as f:
    f.write("arxiv_id\tpdf_present\tn_citations\tfirst_location\tcontext\n")
    for aid in sorted(cites, key=lambda a: -len(cites[a])):
        w, ctx = cites[aid][0]
        f.write(f"{aid}\t{'y' if aid in present else 'n'}\t{len(cites[aid])}\t{w}\t{ctx}\n")
print(len(cites), "distinct arXiv ids;", sum(a in present for a in cites), "with PDF;", sum(a not in present for a in cites), "without")
for aid in sorted(cites, key=lambda a: -len(cites[a])):
    if aid not in present:
        print(f"  MISSING {aid} x{len(cites[aid])}  {cites[aid][0][0][:60]} | {cites[aid][0][1][:110]}")
