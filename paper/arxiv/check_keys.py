"""List every \\V{key} the paper's sections use that no generated numbers file defines, and the
design labels, em dashes and American spellings the text must not carry. Run from arxiv/:

    python check_keys.py            # keys only
    python check_keys.py --style    # keys and the style greps
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"
SECTIONS = HERE / "sections"

SETV = re.compile(r"\\setV\{([^}]*)\}")
USE = re.compile(r"\\V\{([^}]*)\}")
LABEL = re.compile(r"(?<![A-Za-z\\])D[1-9](?![0-9A-Za-z])|(?<![A-Za-z\\\-])K(?![A-Za-z'$])")
AMERICAN = re.compile(r"\b(generaliz\w*|optimiz\w*|minimiz\w*|maximiz\w*|standardiz\w*|summariz\w*|"
                      r"characteriz\w*|analyz\w*|behavior\w*|color\w*|labeled|modeling|center\b|centered|"
                      r"favorable|unfavorable|fulfill\w*|catalog\b|defense|license[sd]?\b|practice[sd]\b)", re.I)


def keys() -> set[str]:
    out = set()
    for f in sorted(GENERATED.glob("numbers*.tex")):
        out |= set(SETV.findall(f.read_text(encoding="utf-8")))
    return out


def main() -> int:
    defined = keys()
    used: dict[str, set[str]] = {}
    texts = {f.name: f.read_text(encoding="utf-8") for f in sorted(SECTIONS.glob("*.tex"))}
    for name, text in texts.items():
        for k in USE.findall(text):
            used.setdefault(k, set()).add(name)
    missing = {k: v for k, v in used.items() if k not in defined}
    print(f"{len(used)} keys used, {len(missing)} missing")
    for k, v in sorted(missing.items()):
        print(f"  MISSING {k}  ({', '.join(sorted(v))})")
    if "--style" in sys.argv:
        for name, text in texts.items():
            body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("%"))
            for i, line in enumerate(body.splitlines(), 1):
                plain = re.sub(r"\$[^$]*\$", "", line)  # labels inside maths are allowed
                if LABEL.search(plain):
                    print(f"  LABEL   {name}:{i}: {line.strip()[:110]}")
                if "---" in line or "\u2014" in line:
                    print(f"  DASH    {name}:{i}: {line.strip()[:110]}")
                if AMERICAN.search(plain):
                    print(f"  SPELL   {name}:{i}: {AMERICAN.search(plain).group(0)}")
                if "\\textbf" in line:
                    print(f"  BOLD    {name}:{i}: {line.strip()[:110]}")
    if "--find" in sys.argv:
        for pat in sys.argv[sys.argv.index("--find") + 1:]:
            hits = sorted(k for k in defined if pat in k)
            print(f"{pat} -> {hits[:60]}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
