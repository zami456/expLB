"""Pack the arXiv source bundle: main.tex, the ICLR style files, the sections, the generated tables and
numbers, the figures, the bibliography and the compiled .bbl, and nothing else. Checks that every file
main.tex and the sections input or include exists in the bundle, that no path is absolute or carries a
space, and that no hidden file is packed. Run from arxiv/:

    python make_bundle.py            # writes arxiv_bundle.zip
    python make_bundle.py --check    # only lists what would be packed
"""

import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "arxiv_bundle.zip"

TOP = ("main.tex", "main.bbl", "refs.bib", "iclr2027_conference.sty", "iclr2027_conference.bst", "fancyhdr.sty",
       "natbib.sty")
FOLDERS = ("sections", "generated", "images")
KEEP = {".tex", ".bbl", ".bib", ".sty", ".bst", ".pdf", ".png"}


def inputs_of(text: str) -> list[str]:
    return re.findall(r"\\(?:input|include|includegraphics(?:\[[^\]]*\])?|bibliography)\{([^}]*)\}", text)


def main() -> int:
    files: list[Path] = []
    for name in TOP:
        path = HERE / name
        if not path.exists():
            print(f"missing: {name}")
            return 1
        files.append(path)
    for folder in FOLDERS:
        for path in sorted((HERE / folder).rglob("*")):
            if path.is_file() and path.suffix in KEEP and not path.name.startswith("."):
                files.append(path)
    rel = {p.relative_to(HERE).as_posix() for p in files}
    # every referenced file is in the bundle (figures resolve through \graphicspath{{images/}})
    texts = [p for p in files if p.suffix == ".tex"]
    problems = []
    for tex in texts:
        for ref in inputs_of(tex.read_text(encoding="utf-8")):
            if ref in ("refs", "iclr2027_conference"):
                continue
            candidates = [ref, f"{ref}.tex", f"images/{ref}", f"{ref}.pdf"]
            if not any(c in rel for c in candidates):
                problems.append(f"{tex.name} references {ref}, not in the bundle")
    for r in rel:
        if " " in r or r.startswith("/") or ":" in r:
            problems.append(f"bad path: {r}")
    if problems:
        print("\n".join(problems))
        return 1
    size = sum(p.stat().st_size for p in files)
    print(f"{len(files)} files, {size / 1e6:.1f} MB")
    if "--check" in sys.argv:
        print("\n".join(sorted(rel)))
        return 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(HERE).as_posix())
    print(f"wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
