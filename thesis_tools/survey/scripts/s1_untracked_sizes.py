"""S1 part 3: for every '!!' / '??' entry of `git status --ignored --short`, count files and bytes on disk
(os.walk, symlinks/junctions not followed). Writes s1_data/untracked_sizes.tsv."""
import os, subprocess, collections
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
out = subprocess.run(["git", "status", "--ignored", "--short"], cwd=ROOT, capture_output=True, text=True).stdout
rows = []
for l in out.splitlines():
    st, p = l[:2], l[3:].strip().strip('"')
    if st not in ("!!", "??"): continue
    full = os.path.join(ROOT, p.rstrip("/"))
    n = b = 0; ext = collections.Counter(); links = 0; newest = 0
    if os.path.isfile(full):
        n, b = 1, os.path.getsize(full); ext[os.path.splitext(p)[1] or "(none)"] += 1; newest = os.path.getmtime(full)
    else:
        for dp, dns, fns in os.walk(full, followlinks=False):
            for d in list(dns):
                if os.path.islink(os.path.join(dp, d)) or (hasattr(os.path, "isjunction") and os.path.isjunction(os.path.join(dp, d))):
                    links += 1; dns.remove(d)
            for f in fns:
                fp = os.path.join(dp, f)
                try: s = os.stat(fp, follow_symlinks=False)
                except OSError: continue
                n += 1; b += s.st_size; ext[os.path.splitext(f)[1].lower() or "(none)"] += 1; newest = max(newest, s.st_mtime)
    import time
    rows.append((st, p, n, b, links, time.strftime("%Y-%m-%d", time.localtime(newest)) if newest else "-", ",".join(f"{e}:{c}" for e, c in ext.most_common(5))))
with open(os.path.join(ROOT, "thesis_tools/survey/s1_data/untracked_sizes.tsv"), "w", encoding="utf-8") as fh:
    fh.write("status\tpath\tfiles\tbytes\tskipped_links\tnewest_mtime\ttop_ext\n")
    for r in rows: fh.write("\t".join(map(str, r)) + "\n")
tot = collections.Counter()
for st, p, n, b, *_ in rows: tot[st] += b
for r in rows: print(f"{r[0]} {r[1]:<45} files={r[2]:>7} size={r[3]/2**20:>10.1f}MiB links_skipped={r[4]} newest={r[5]} {r[6]}")
print({k: f"{v/2**30:.2f} GiB" for k, v in tot.items()})
