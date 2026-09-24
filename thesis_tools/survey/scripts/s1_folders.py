"""S1 part 4: per branch, file count + top extensions per top-level folder (git ls-tree -r -l). Also README/DESIGN docs found per folder.
Writes s1_data/folders.json."""
import os, subprocess, collections, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
def g(*a): return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
br = json.load(open(os.path.join(ROOT, "thesis_tools/survey/s1_data/branches.json")))["branches"]
res = {}
for b in br:
    d = collections.defaultdict(lambda: dict(files=0, bytes=0, ext=collections.Counter(), docs=[], gitlinks=0))
    for l in g("ls-tree", "-r", "-l", b).splitlines():
        meta, path = l.split("\t", 1); mode, typ, sha, size = meta.split()
        t = path.split("/")[0] if "/" in path else "(root files)"
        if typ == "commit": d[t]["gitlinks"] += 1; continue
        d[t]["files"] += 1; d[t]["bytes"] += int(size) if size != "-" else 0
        d[t]["ext"][os.path.splitext(path)[1].lower() or "(none)"] += 1
        base = os.path.basename(path).upper()
        if path.count("/") <= 2 and (base.startswith("README") or base in ("DESIGN.MD", "INDEX.MD")): d[t]["docs"].append(path)
    res[b] = {k: dict(v, ext=dict(v["ext"].most_common(5))) for k, v in d.items()}
json.dump(res, open(os.path.join(ROOT, "thesis_tools/survey/s1_data/folders.json"), "w"), indent=1)
folders = sorted(set(k for v in res.values() for k in v))
short = {b: b.replace("origin/", "o/").replace("claude/", "c/")[:14] for b in br}
print("folder".ljust(28) + " ".join(short[b].rjust(14) for b in br))
for f in folders: print(f[:28].ljust(28) + " ".join(str(res[b].get(f, {}).get("files", "-")).rjust(14) for b in br))
