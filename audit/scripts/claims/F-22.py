"""F-22 - `1_generate_ndif.py` never got the exc_tail fix (source line: 1_generate_ndif.py:145).

Checks, purely from git blobs + the working tree (no lab PC needed for the code claim):
  1. In each committed version of src/assistant_axis_ndif/1_generate_ndif.py, where does
     `str(e)[:200]` appear and where does `exc_tail` appear?
  2. Did 8_coords_ndif.py / ndif_common.py have exc_tail at the time F-22 was written (2026-09-20)?
  3. Does HEAD still contain a str(e)[:200] handler in 1_generate_ndif.py (the sibling)?
"""
import subprocess, re, sys

def blob(rev, path):
    r = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True)
    if r.returncode: return None
    return r.stdout.decode("utf-8", "replace")

def hits(txt, pat):
    return [(i+1, l.strip()) for i, l in enumerate(txt.splitlines()) if re.search(pat, l)]

revs = ["8078702", "5d7b856", "56891ec", "107879d", "106d09f", "3dfdb67", "HEAD"]
dates = {}
for r in revs:
    d = subprocess.run(["git", "show", "-s", "--format=%cs %h", r], capture_output=True).stdout.decode().strip()
    dates[r] = d

print("=== src/assistant_axis_ndif/1_generate_ndif.py ===")
for r in revs:
    t = blob(r, "src/assistant_axis_ndif/1_generate_ndif.py")
    if t is None:
        print(f"{r} ({dates[r]}): ABSENT"); continue
    s200 = hits(t, r"str\(e\)\[:200\]")
    et   = hits(t, r"exc_tail")
    print(f"\n-- {r} ({dates[r]})  {len(t.splitlines())} lines")
    for ln, l in s200: print(f"   str(e)[:200]  L{ln}: {l[:110]}")
    for ln, l in et:   print(f"   exc_tail      L{ln}: {l[:110]}")

print("\n=== siblings at the two commits around F-22's date ===")
for path in ["src/assistant_axis_ndif/8_coords_ndif.py", "src/assistant_axis_ndif/ndif_common.py",
             "src/assistant_axis_ndif/2_activations_ndif.py"]:
    for r in ["107879d", "106d09f", "3dfdb67", "HEAD"]:
        t = blob(r, path)
        if t is None:
            print(f"{path} @{r}: ABSENT"); continue
        print(f"{path} @{r} ({dates[r]}): exc_tail={len(hits(t,'exc_tail'))} str(e)[:200]={len(hits(t,r'str.e..:200.'))}")

print("\n=== context of lines 138-152 at 106d09f (the pre-fix version F-22 describes) ===")
t = blob("106d09f", "src/assistant_axis_ndif/1_generate_ndif.py")
for i, l in enumerate(t.splitlines()[137:152], start=138):
    print(f"{i:4d}  {l}")

print("\n=== context of lines 140-152 and 180-190 at HEAD ===")
t = blob("HEAD", "src/assistant_axis_ndif/1_generate_ndif.py")
L = t.splitlines()
for i, l in enumerate(L[139:152], start=140): print(f"{i:4d}  {l}")
print("   ...")
for i, l in enumerate(L[179:192], start=180): print(f"{i:4d}  {l}")
