"""Static checks for the thesis LaTeX (no TeX install needed).

    python thesis_tools/check_latex.py [chapter.tex ...]

With no arguments, checks every file main.tex \\input's. Reports:
  1. unbalanced braces (ignoring \\{ \\} and comments), per file
  2. \\begin/\\end mismatches, in order (a stack, so nesting errors are caught)
  3. \\ref / \\eqref / \\cref targets with no \\label anywhere in the thesis
  4. \\cite-family keys missing from bibliography/references.bib
  5. \\includegraphics files missing from images/ (exact capitalisation; \\graphicspath{{images/}})
Exit code 1 if any check fails.
"""
import os
import re
import sys
from pathlib import Path

THESIS = Path(__file__).resolve().parents[1] / "thesis"
BIB = THESIS / "bibliography" / "references.bib"
IMAGES = THESIS / "images"


def strip_comments(text: str) -> str:
    # remove from an unescaped % to end of line
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def inputs_of_main() -> list[Path]:
    main = strip_comments((THESIS / "main.tex").read_text(encoding="utf-8"))
    files = [(THESIS / "main.tex").resolve()]
    for m in re.finditer(r"\\(?:input|include)\{([^}]+)\}", main):
        p = THESIS / m.group(1)
        if p.suffix != ".tex":
            p = p.with_suffix(".tex")
        files.append(p.resolve())
    return files


def check_braces(name: str, text: str, problems: list[str]) -> None:
    depth, line = 0, 1
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\n":
            line += 1
        elif c == "\\":
            i += 2          # skip the escaped character (\{ \} \\ \%)
            continue
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                problems.append(f"{name}:{line}: unmatched '}}'")
                depth = 0
        i += 1
    if depth != 0:
        problems.append(f"{name}: {depth} unclosed '{{' at end of file")


def check_envs(name: str, text: str, problems: list[str]) -> None:
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        line = text.count("\n", 0, m.start()) + 1
        kind, env = m.groups()
        if kind == "begin":
            stack.append((env, line))
        elif not stack:
            problems.append(f"{name}:{line}: \\end{{{env}}} with no open environment")
        elif stack[-1][0] != env:
            problems.append(f"{name}:{line}: \\end{{{env}}} closes \\begin{{{stack[-1][0]}}} opened at line {stack[-1][1]}")
            stack.pop()
        else:
            stack.pop()
    for env, line in stack:
        problems.append(f"{name}:{line}: \\begin{{{env}}} never closed")


def main(argv: list[str]) -> int:
    files = [Path(a).resolve() for a in argv] if argv else inputs_of_main()
    all_files = inputs_of_main()                     # labels may live in any input
    problems: list[str] = []
    texts = {}
    for f in set(files) | set(all_files):
        if f.exists():
            texts[f] = strip_comments(f.read_text(encoding="utf-8"))
        elif f in files:
            problems.append(f"{f}: file does not exist")

    labels = set()
    for t in texts.values():
        labels |= set(re.findall(r"\\label\{([^}]+)\}", t))
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", BIB.read_text(encoding="utf-8")))
    on_disk = {p.name for p in IMAGES.iterdir()} if IMAGES.exists() else set()

    for f in files:
        if f not in texts:
            continue
        t = texts[f]
        name = f.relative_to(THESIS).as_posix() if THESIS.resolve() in f.parents else f.name
        check_braces(name, t, problems)
        check_envs(name, t, problems)
        for m in re.finditer(r"\\(?:ref|eqref|cref|Cref|autoref|pageref)\{([^}]+)\}", t):
            for lab in m.group(1).split(","):
                if lab.strip() not in labels:
                    problems.append(f"{name}:{t.count(chr(10), 0, m.start()) + 1}: \\ref{{{lab.strip()}}} has no \\label")
        for m in re.finditer(r"\\(?:[a-zA-Z]*cite[a-zA-Z]*)\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}", t):
            for key in m.group(1).split(","):
                if key.strip() not in bib_keys:
                    problems.append(f"{name}:{t.count(chr(10), 0, m.start()) + 1}: cite key '{key.strip()}' not in references.bib")
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t):
            target = m.group(1).strip()
            base = Path(target).name
            candidates = [base] if Path(base).suffix else [base + ext for ext in (".pdf", ".png", ".jpg", ".jpeg", ".eps")]
            if not any(c in on_disk for c in candidates):
                near = [n for n in on_disk if n.lower() in {c.lower() for c in candidates}]
                hint = f" (case mismatch: {near})" if near else ""
                problems.append(f"{name}:{t.count(chr(10), 0, m.start()) + 1}: image '{target}' not found in images/{hint}")

    print(f"checked {len(files)} file(s); {len(labels)} labels; {len(bib_keys)} bib keys; {len(on_disk)} image files")
    for p in problems:
        print("FAIL", p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
