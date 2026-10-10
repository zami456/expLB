"""Print every \\label in the thesis as a markdown table, with its kind and exact title or caption.

    python thesis_tools/label_index.py            # all files main.tex inputs
    python thesis_tools/label_index.py --terms    # also the terms Chapter 5 defines with \\emph{}

For each label it finds the heading it follows (\\chapter, \\section, \\subsection, \\subsubsection,
\\paragraph), or the \\caption of the float it sits in, or the equation it tags. Captions and titles
are brace-matched, so nested \\texttt{...} is kept. Chapter labels sit on the first line of a chapter
file (the \\chapter command itself is in main.tex). Used to keep GLOSSARY.md's label list exact.
"""
import re
import sys
from pathlib import Path

THESIS = Path(__file__).resolve().parents[1] / "thesis"


def strip_comment(line: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", line)


def braced(text: str, start: int) -> str:
    """Contents of the {...} group that opens at text[start] == '{'."""
    depth, i = 0, start
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
        i += 1
    return text[start + 1:]


def files_in_order():
    main = THESIS / "main.tex"
    out = [main]
    for m in re.finditer(r"\\input\{([^}]+)\}", "\n".join(strip_comment(l) for l in main.read_text(encoding="utf-8").splitlines())):
        p = THESIS / m.group(1)
        if p.suffix != ".tex":
            p = p.with_suffix(".tex")
        if p.exists():
            out.append(p)
    return out


HEAD = re.compile(r"\\(chapter|section|subsection|subsubsection|paragraph)\*?\{")
KIND = {"chapter": "Chapter", "section": "Section", "subsection": "Subsection",
        "subsubsection": "Subsubsection", "paragraph": "Paragraph"}


def labels_in(path: Path, main_chapter_titles: dict):
    text = "\n".join(strip_comment(l) for l in path.read_text(encoding="utf-8").splitlines())
    rows = []
    for m in re.finditer(r"\\label\{([^}]+)\}", text):
        lab, pos = m.group(1), m.start()
        before = text[:pos]
        # innermost open float or equation
        env = None
        for e in ("figure", "table", "equation"):
            b = before.rfind(r"\begin{" + e)
            if b >= 0 and before.rfind(r"\end{" + e) < b:
                env = (e, b)
        if env and env[0] in ("figure", "table"):
            seg_end = text.find(r"\end{" + env[0], pos)
            seg = text[env[1]:seg_end]
            c = seg.find(r"\caption{")
            title = braced(seg, c + len(r"\caption")) if c >= 0 else ""
            rows.append((lab, env[0].capitalize(), " ".join(title.split())))
            continue
        if env and env[0] == "equation":
            seg_end = text.find(r"\end{equation}", pos)
            body = text[env[1] + len(r"\begin{equation}"):seg_end]
            body = re.sub(r"\\label\{[^}]*\}", "", body)
            rows.append((lab, "Equation", "$" + " ".join(body.split()) + "$"))
            continue
        heads = list(HEAD.finditer(before))
        if heads:
            h = heads[-1]
            title = braced(before, h.end() - 1)
            rows.append((lab, KIND[h.group(1)], " ".join(title.split())))
        else:
            rows.append((lab, "Chapter", main_chapter_titles.get(path.name, "")))
    return rows


def main():
    files = files_in_order()
    main_text = "\n".join(strip_comment(l) for l in files[0].read_text(encoding="utf-8").splitlines())
    # chapter title that precedes each \input in main.tex
    ch_titles = {}
    for m in re.finditer(r"\\chapter\{([^}]*)\}\s*\n?\s*(?:\\label\{[^}]*\}\s*)?\\input\{([^}]+)\}", main_text):
        ch_titles[Path(m.group(2)).with_suffix(".tex").name] = " ".join(m.group(1).split())
    print("| Label | Kind | Title / caption (exact) | File |")
    print("|---|---|---|---|")
    for f in files:
        for lab, kind, title in labels_in(f, ch_titles):
            title = title.replace("|", "\\|")
            print(f"| `{lab}` | {kind} | {title} | `{f.relative_to(THESIS).as_posix()}` |")
    if "--terms" in sys.argv:
        ch6 = THESIS / "chapters" / "chapter_6.tex"
        print("\n| Term (as emphasised) | Defining sentence (exact, comments removed) |")
        print("|---|---|")
        seen = set()
        for line in ch6.read_text(encoding="utf-8").splitlines():
            clean = strip_comment(line).strip()
            for m in re.finditer(r"\\emph\{", clean):
                term = braced(clean, m.end() - 1)
                if term in seen or term.endswith(".") or len(term) > 60:
                    continue
                seen.add(term)
                print(f"| {term} | {clean.replace('|', chr(92) + '|')} |")


if __name__ == "__main__":
    main()
