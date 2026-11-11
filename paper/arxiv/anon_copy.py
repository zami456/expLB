"""Write main_anon.tex, the double-blind variant of main.tex (the \\anonymoustrue switch), so that it can be
compiled beside the preprint without touching main.tex. Run from arxiv/:

    python anon_copy.py && tectonic main_anon.tex
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE / "main.tex").read_text(encoding="utf-8")
switch = "\\anonymousfalse\n"
if source.count(switch) != 1:
    raise SystemExit(f"main.tex does not carry exactly one {switch.strip()!r} line")
(HERE / "main_anon.tex").write_text(source.replace(switch, "\\anonymoustrue\n"), encoding="utf-8")
print("wrote main_anon.tex")
