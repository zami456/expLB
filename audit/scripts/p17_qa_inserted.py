"""P17 QA: dump every line inserted since the pre_wave3 snapshots (file:line: text)."""
import pathlib, difflib, sys
import itertools
ROOT = pathlib.Path(__file__).resolve().parents[2]
def resolve(name):
    parts = name.split("_")
    for mask in itertools.product(["/", "_"], repeat=len(parts)-1):
        q = parts[0] + "".join(m + x for m, x in zip(mask, parts[1:]))
        if (ROOT / q).is_file(): yield q
SNAP = ROOT / "audit/p17/pre_wave3"
out = []
for f in sorted(SNAP.iterdir()):
    p = next(resolve(f.name))
    a = f.read_text(encoding="utf-8").splitlines(); b = (ROOT / p).read_text(encoding="utf-8").splitlines()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "equal":
            for j in range(j1, j2): out.append(f"{p}:{j+1}: {b[j]}")
pathlib.Path(sys.argv[1]).write_text("\n".join(out), encoding="utf-8")
print(len(out), "inserted lines")
