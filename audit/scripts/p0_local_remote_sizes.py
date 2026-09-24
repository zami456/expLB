"""Phase-0: which local folder (if any) does the root file remote_sizes.txt describe?

remote_sizes.txt (tracked, added in 3dfdb67) is a 'relative\\path|bytes' listing of some
remote results folder. For every local folder that holds a summary.json, compare the
listing against the local files (name + size). Read-only; prints a table.

Re-run:  PYTHONIOENCODING=utf-8 python audit/scripts/p0_local_remote_sizes.py
"""
import os

ROOT = r"C:\Users\SURAIYA\Documents\STUDY\thesis\expLB"
listing = {}
with open(os.path.join(ROOT, 'remote_sizes.txt'), encoding='utf-8') as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        p, s = line.rsplit('|', 1)
        listing[p.replace('\\', '/')] = int(s)
print('entries in remote_sizes.txt:', len(listing), ' total bytes:', sum(listing.values()))
subdirs = sorted({p.split('/')[0] for p in listing if '/' in p})
print('sub-folders named:', subdirs, ' layers/poolings in matrices:',
      sorted(p for p in listing if p.startswith('matrices/')))

cands = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in ('.git', '.venv', '.venv-aa', '.venv-aa-win', 'audit')]
    if 'summary.json' in fn:
        cands.append(dp)
for c in cands:
    same = diff = missing = 0
    diffs = []
    for p, s in listing.items():
        f = os.path.join(c, *p.split('/'))
        if not os.path.exists(f):
            missing += 1
            continue
        ls = os.path.getsize(f)
        # tracked text files may be CRLF-expanded on this checkout; compare LF size too
        if ls == s:
            same += 1
        else:
            with open(f, 'rb') as fh:
                lf = len(fh.read().replace(b'\r\n', b'\n'))
            if lf == s:
                same += 1
            else:
                diff += 1
                if len(diffs) < 3:
                    diffs.append(f'{p}: remote {s} local {ls} (LF {lf})')
    print(f'{os.path.relpath(c, ROOT):45s} match={same:3d} size_differs={diff:3d} absent={missing:3d} {diffs}')
