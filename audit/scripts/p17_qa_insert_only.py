"""P17 QA: independent insert-only check. For each pre_wave3 snapshot, verify every snapshot line
appears in the current file, in order (subsequence, exact bytes incl. line endings).
Also self-test: must FAIL on an altered temp copy."""
import pathlib, sys, hashlib, tempfile, os
ROOT = pathlib.Path(__file__).resolve().parents[2]
SNAP = ROOT / "audit/p17/pre_wave3"

def resolve(name):
    parts = name.split("_")
    # try all ways to join parts with '/' or '_'
    import itertools
    n = len(parts)
    for mask in itertools.product(["/", "_"], repeat=n-1):
        p = parts[0] + "".join(m + q for m, q in zip(mask, parts[1:]))
        if (ROOT / p).is_file():
            yield p

def subseq(snap_lines, cur_lines):
    j = 0; missing = []
    for i, l in enumerate(snap_lines):
        while j < len(cur_lines) and cur_lines[j] != l:
            j += 1
        if j == len(cur_lines):
            return False, i
        j += 1
    return True, None

def check(snapb, curb):
    s = snapb.splitlines(keepends=True); c = curb.splitlines(keepends=True)
    ok, at = subseq(s, c)
    return ok, at, len(s), len(c)

res = []
for f in sorted(SNAP.iterdir()):
    cands = list(resolve(f.name))
    if len(cands) != 1:
        res.append((f.name, "AMBIGUOUS/MISSING", cands)); continue
    p = cands[0]
    sb = f.read_bytes(); cb = (ROOT / p).read_bytes()
    ok, at, ns, nc = check(sb, cb)
    same = hashlib.md5(sb).hexdigest() == hashlib.md5(cb).hexdigest()
    res.append((p, "OK" if ok else f"FAIL at snapshot line {at+1}", f"{ns}->{nc} (+{nc-ns}){' unchanged' if same else ''}"))
for r in res: print(*r)
# self-test
f = SNAP / "audit_PROGRESS.md"; sb = f.read_bytes(); cb = (ROOT / "audit/PROGRESS.md").read_bytes()
lines = cb.splitlines(keepends=True)
k = len(lines)//2; lines[k] = lines[k].replace(b"a", b"b", 1) if b"a" in lines[k] else b"X" + lines[k]
ok, *_ = check(sb, b"".join(lines)); print("SELFTEST altered-line ->", "FAIL (good)" if not ok else "PASS (BAD CHECKER)")
lines = cb.splitlines(keepends=True); del lines[100]
ok, *_ = check(sb, b"".join(lines)); print("SELFTEST deleted-line ->", "FAIL (good)" if not ok else "PASS (BAD CHECKER)")
lines = cb.splitlines(keepends=True); lines[10], lines[11] = lines[11], lines[10]
ok, *_ = check(sb, b"".join(lines)); print("SELFTEST swapped-lines ->", "FAIL (good)" if not ok else "PASS (maybe identical lines)")
