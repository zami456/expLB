"""Phase 0: history of the findings ledger kd/docs/FINDINGS.md, one row per F-id.
Read-only: uses only `git log` / `git show`.

Usage (from the repo root):
  PYTHONIOENCODING=utf-8 python audit/scripts/p0_docs_fhistory.py            # writes audit/phase0/fnumber_history.csv, prints summary
  PYTHONIOENCODING=utf-8 python audit/scripts/p0_docs_fhistory.py --commits  # also prints per-commit section add/edit table

What it does
  1. Lists every commit that touched the ledger (git log --follow, oldest first).
  2. At each commit, reads the file with `git show <commit>:<path>` and splits it into sections at
     headings `### F-NN · date · title` (a section runs to the next heading).
  3. Per F-id: first commit it appeared in, later commits that changed its text, whether any of
     those changes removed/replaced existing lines (a non-append edit), title changes, deletions.
  4. At HEAD: parses the section's own `supersedes:` / `amends:` fields, CORRECTION-to titles and
     `SUPERSEDED BY` lines (these relations are DOC-ONLY: what the ledger says about itself).
"""
import csv, difflib, os, re, subprocess, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
PATH = "kd/docs/FINDINGS.md"
OUT = "audit/phase0/fnumber_history.csv"
HEAD_RX = re.compile(r"^#{1,4}\s+\**\s*(F-(\d+))\b(.*)$")
FID = re.compile(r"\bF-(\d{1,3})\b")


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, encoding="utf-8",
                          errors="replace").stdout


def sections(txt):
    """-> (preamble_lines, OrderedDict fid -> dict(heading, title, date, start, end, lines))"""
    lines = txt.replace("\r\n", "\n").split("\n")
    out = collections.OrderedDict()
    pre, cur = [], None
    for i, line in enumerate(lines, 1):
        m = HEAD_RX.match(line)
        if m:
            fid = f"F-{int(m.group(2)):02d}"
            rest = m.group(3).strip().lstrip("·").strip()
            parts = [p.strip() for p in rest.split("·")]
            date = parts[0] if parts else ""
            title = " · ".join(parts[1:]) if len(parts) > 1 else rest
            if fid in out:                       # duplicate heading in one version
                fid = fid + f"#dup@{i}"
            cur = dict(heading=line, title=title.strip("* ").strip(), date=date, start=i, lines=[])
            out[fid] = cur
        elif cur is None:
            pre.append(line)
        else:
            cur["lines"].append(line)
    for fid, s in out.items():
        # trim trailing blank lines / '---' separators for stable comparison
        body = s["lines"]
        while body and body[-1].strip() in ("", "---"):
            body = body[:-1]
        s["lines"] = body
        s["end"] = s["start"] + len(body)
    return pre, out


def edit_kind(old, new):
    """'append' if every old line survives in order (only insertions), else 'modify'."""
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    ops = {tag for tag, *_ in sm.get_opcodes() if tag != "equal"}
    if not ops:
        return None
    return "append" if ops <= {"insert"} else "modify"


def relations(sec):
    """F-ids this section says it corrects / supersedes / amends (DOC-ONLY), and F-ids it says supersede it."""
    txt_lines = sec["lines"]
    corrects, by = set(), set()
    # (a) title
    t = sec["title"]
    for m in re.finditer(r"(?i)correct(?:ion|ed)?\s+(?:to\s+)?(F-\d+)", t):
        corrects.add(m.group(1))
    # (b) fields "- **supersedes:**" / "- **amends:**" (+ continuation lines, until next "- **field")
    field, buf = None, []
    fields = []
    for line in txt_lines + ["- **end:**"]:
        m = re.match(r"^\s*-\s+\*\*([a-z ()]+):\*\*(.*)$", line)
        if m:
            if field:
                fields.append((field, " ".join(buf)))
            field, buf = m.group(1).strip(), [m.group(2)]
        elif field and line.startswith("  "):
            buf.append(line.strip())
        elif field:
            fields.append((field, " ".join(buf))); field, buf = None, []
    inline = []
    for f, text in fields:
        if f.startswith("supersedes") or f.startswith("amends"):
            inline.append(text)
        if f.startswith("status"):
            for m in re.finditer(r"(?i)superseded[^.;]*?\bby\s+(F-\d+)", text):
                by.add(m.group(1))
            inline.append(text)  # "status: measured; supersedes F-89's ..." pattern
    # inline "**amends:** F-xx" inside other fields (e.g. "- **supports:** F-97. **amends:** F-98")
    for f, text in fields:
        for m in re.finditer(r"\*\*amends:\*\*([^*]*)", text):
            inline.append(m.group(1))
    for text in inline:
        # keep only sentences that actually supersede/amend; drop "Confirms/supports/bounds ..." sentences
        for sent in re.split(r"(?<=[.;])\s+|\*\*(?=Confirms|Bounds|Corrects in advance|contradicts)", text):
            s = sent.strip()
            if not s or re.match(r"(?i)^(nothing|confirms|bounds|supports|corrects in advance|contradicts)", s):
                continue
            if re.search(r"(?i)^(measured|verified|provisional|inherited)\b", s) and "supersed" not in s.lower() and "amend" not in s.lower():
                continue
            if re.search(r"(?i)superseded[^.;]*?\bby\s+F-\d+", s):
                continue  # handled as 'by'
            for m in FID.finditer(s):
                corrects.add(f"F-{int(m.group(1)):02d}")
    # (c) body markers
    for line in txt_lines:
        for m in re.finditer(r"SUPERSEDED BY (F-\d+)", line):
            by.add(m.group(1))
        for m in re.finditer(r"(?i)\bCORRECTION TO (F-\d+)", line):
            corrects.add(m.group(1))
    norm = lambda s: {f"F-{int(x[2:]):02d}" for x in s}
    return norm(corrects), norm(by)


# Manual reading (audit, 2026-09-22) of each section's `supersedes:` / `amends:` / `status:` fields,
# CORRECTION titles and SUPERSEDED BY lines at HEAD 43ee6d3. The automatic parse above over-collects
# (e.g. "supersedes: nothing in F-13's numbers", "F-58 remains superseded", "Quote F-94, not this entry"),
# so for F-01..F-109 these curated lists are used; any F-id added later falls back to the automatic parse,
# and the script prints every section where automatic and curated disagree so the curation can be re-checked.
# CORR = what the section itself says it corrects / supersedes / amends.
CORR = {
    "F-30": "F-27", "F-39": "F-11", "F-40": "F-35", "F-42": "F-41", "F-43": "F-41 F-42", "F-45": "F-35 F-40",
    "F-46": "F-36", "F-49": "F-45", "F-50": "F-49", "F-51": "F-28", "F-58": "F-39", "F-60": "F-58",
    "F-63": "F-58 F-60", "F-66": "F-65", "F-67": "F-66", "F-68": "F-65", "F-69": "F-20", "F-71": "F-20 F-69",
    "F-72": "F-03", "F-75": "F-72", "F-76": "F-75", "F-81": "F-80", "F-83": "F-79 F-81", "F-90": "F-79",
    "F-94": "F-89", "F-96": "F-63", "F-97": "F-85", "F-99": "F-98", "F-100": "F-96 F-63", "F-101": "F-100",
    "F-102": "F-63 F-100", "F-105": "F-93", "F-108": "F-79 F-80", "F-109": "F-85",
}
# OTHER = non-correcting relations the fields state (confirms / supports / bounds / contradicts / contrasts).
OTHER = {
    "F-29": "supplies-argument-for:F-13 (says 'supersedes: nothing in F-13's numbers')",
    "F-65": "confirms:F-13,F-62; contradicts:F-55(representational half)",
    "F-68": "contradicts:F-55(contrast half)",
    "F-77": "bounds:F-66; contrasts:F-76",
    "F-80": "confirms:F-79",
    "F-96": "notes F-58 remains superseded",
    "F-98": "supports:F-97,F-79,F-80",
    "F-99": "supports:F-97,F-98",
    "F-103": "supports:F-98,F-99; bounds:F-102",
    "F-109": "supports:F-91",
}


def field_text(sec):
    out, field = [], None
    for line in sec["lines"]:
        m = re.match(r"^\s*-\s+\*\*(supersedes|amends|supports|status)[^:]*:\*\*(.*)$", line)
        if m and (m.group(1) != "status" or re.search(r"(?i)supersed|amend", m.group(2))):
            field = m.group(1); out.append(f"{field}: {m.group(2).strip()}"); continue
        if m:
            field = None; continue
        if field and line.startswith("  "):
            out[-1] += " " + line.strip(); continue
        field = None
        if re.search(r"SUPERSEDED BY|CORRECTION TO F-", line, re.I):
            out.append("body: " + line.strip().lstrip("> "))
    return " || ".join(out)


def main():
    log = git("log", "--follow", "--reverse", "--format=%h|%ad|%s", "--date=format:%Y-%m-%d %H:%M", "--", PATH)
    commits = [l.split("|", 2) for l in log.splitlines() if l.strip()]
    hist = collections.OrderedDict()   # fid -> info
    per_commit = []
    prev = {}
    prev_pre = None
    file_events = []
    for h, d, s in commits:
        txt = git("show", f"{h}:{PATH}")
        pre, secs = sections(txt)
        if prev_pre is not None and pre != prev_pre:
            file_events.append(f"{h} {d}: preamble changed")
        prev_pre = pre
        added, changed, modified = [], [], []
        order = list(secs)
        dups = [k for k in order if "#dup" in k]
        if dups:
            file_events.append(f"{h} {d}: DUPLICATE headings {dups}")
        # were new ids appended at the end?
        old_ids = [k for k in order if k in prev]
        new_ids = [k for k in order if k not in prev]
        if prev and new_ids and old_ids and min(order.index(k) for k in new_ids) < max(order.index(k) for k in old_ids):
            file_events.append(f"{h} {d}: new section(s) {new_ids} inserted BEFORE existing ones")
        if [k for k in order if k in prev] != [k for k in prev if k in secs]:
            file_events.append(f"{h} {d}: existing sections REORDERED")
        for k in prev:
            if k not in secs:
                file_events.append(f"{h} {d}: section {k} DELETED (title was: {prev[k]['title'][:80]})")
        for k, sec in secs.items():
            if k not in hist:
                hist[k] = dict(first=h, first_date=d, first_title=sec["title"], first_date_in_heading=sec["date"],
                               first_lines=sec["lines"], changes=[], modify=[], titles=[sec["title"]], headings=[sec["heading"]])
                added.append(k)
            else:
                p = prev.get(k)
                if p is None:
                    file_events.append(f"{h} {d}: section {k} RE-APPEARED after deletion")
                    continue
                if sec["title"] != hist[k]["titles"][-1]:
                    hist[k]["titles"].append(sec["title"])
                if sec["heading"] != hist[k]["headings"][-1]:
                    hist[k]["headings"].append(sec["heading"])
                kind = edit_kind(p["lines"], sec["lines"])
                if kind or sec["heading"] != p["heading"]:
                    hist[k]["changes"].append(h)
                    changed.append(k)
                    if kind == "modify" or sec["heading"] != p["heading"]:
                        hist[k]["modify"].append(h)
                        modified.append(k)
        per_commit.append((h, d, s, added, changed, modified, len(secs)))
        prev = secs
    head_pre, head = sections(git("show", f"HEAD:{PATH}"))
    # relations at HEAD
    rel, auto_diff = {}, []
    for k, sec in head.items():
        c, by = relations(sec)
        c.discard(k)
        if int(k[2:]) <= 109:
            cur = set(CORR.get(k, "").split())
            if cur != c:
                auto_diff.append(f"{k}: auto={sorted(c)} curated={sorted(cur)}")
            c = cur
        rel[k] = (c, by)
    inverse = collections.defaultdict(set)
    for k, (c, by) in rel.items():
        for t in c:
            inverse[t].add(k)
        for t in by:
            inverse[k].add(t)
    rows = []
    for k, sec in head.items():
        hi = hist.get(k, {})
        status = ""
        for line in sec["lines"]:
            m = re.match(r"^\s*-\s+\*\*status(?: \(revised\))?:\*\*\s*(.*)$", line)
            if m:
                status = m.group(1).strip()
        amend_markers = sum(1 for line in sec["lines"] if re.search(r"\bAMEND(ED|MENT)\b|SECOND AMENDMENT", line))
        sim = difflib.SequenceMatcher(a=hi.get("first_lines", []), b=sec["lines"], autojunk=False).ratio()
        rows.append({
            "f_id": k,
            "heading_title_at_HEAD": sec["title"],
            "date_in_heading": sec["date"],
            "first_commit": hi.get("first", ""),
            "first_commit_date": hi.get("first_date", ""),
            "n_later_commits_changing_section": len(hi.get("changes", [])),
            "n_later_commits_nonappend": len(hi.get("modify", [])),
            "later_commits": " ".join(hi.get("changes", [])),
            "title_changed": ("y | old: " + " || ".join(hi["titles"][:-1])) if len(hi.get("titles", [])) > 1 else "n",
            "similarity_first_vs_HEAD": f"{sim:.3f}",
            "corrects_supersedes_amends_DOCONLY": " ".join(sorted(rel[k][0], key=lambda x: int(x[2:]))),
            "later_said_to_correct_this_DOCONLY": " ".join(sorted(inverse.get(k, set()), key=lambda x: int(x[2:]))),
            "other_relations_DOCONLY": OTHER.get(k, ""),
            "relation_field_text_DOCONLY": field_text(sec)[:300],
            "in_section_amendment_markers": amend_markers,
            "status_line_at_HEAD_DOCONLY": status[:160],
            "line_range_at_HEAD": f"{sec['start']}-{sec['end']}",
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    # ---- summary ----
    ids = list(head)
    nums = [int(k[2:]) for k in ids if "#dup" not in k]
    print(f"commits touching {PATH}: {len(commits)}  (first {commits[0][0]} {commits[0][1]}, last {commits[-1][0]} {commits[-1][1]})")
    print(f"HEAD: {len(ids)} sections, F-{min(nums):02d}..F-{max(nums):02d}, missing={[n for n in range(min(nums), max(nums)+1) if n not in nums]}, "
          f"dups={[k for k in ids if '#dup' in k]}, out_of_order={[ids[i] for i in range(1, len(nums)) if nums[i] < nums[i-1]]}")
    print("ids ever seen in any version:", len(hist), " ids never at HEAD:", [k for k in hist if k not in head])
    print("file-level events (deletions / reorders / insertions / preamble / dups):")
    for e in file_events or ["  none"]:
        print("  ", e)
    print("sections changed after first appearance:", sum(1 for r in rows if r["n_later_commits_changing_section"]),
          "| of which with non-append edits:", sum(1 for r in rows if r["n_later_commits_nonappend"]),
          "| title changed:", sum(1 for r in rows if r["title_changed"] != "n"))
    print("heading date vs first-commit date: sections whose heading date is earlier than first commit day:",
          sum(1 for r in rows if re.match(r"\d{4}-\d\d-\d\d", r["date_in_heading"]) and r["date_in_heading"] < r["first_commit_date"][:10]))
    print("low similarity first-vs-HEAD (<0.5, possible reuse):",
          [(r["f_id"], r["similarity_first_vs_HEAD"]) for r in rows if float(r["similarity_first_vs_HEAD"]) < 0.5])
    print("relations: sections that say they correct/supersede/amend another F-id:",
          sum(1 for r in rows if r["corrects_supersedes_amends_DOCONLY"]),
          "| sections said (by others or by a SUPERSEDED BY line) to be corrected:",
          sum(1 for r in rows if r["later_said_to_correct_this_DOCONLY"]))
    print("automatic-vs-curated relation differences (curated used):")
    for d_ in auto_diff:
        print("   ", d_)
    if "--commits" in sys.argv:
        print("\nper commit: hash date | #sections | added | changed (append-only) | changed with removed/replaced lines")
        for h, d, s, a, c, m, n in per_commit:
            ap = [x for x in c if x not in m]
            print(f"{h} {d} | {n:3d} | +{','.join(a) if len(a) < 8 else a[0]+'..'+a[-1]+f' ({len(a)})'} | "
                  f"app:{','.join(ap)} | MOD:{','.join(m)} | {s[:70]}")
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
