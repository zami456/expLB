"""Phase-0 hint: byte-identical result files inside the persona-coordinates result areas.

Hashes (md5) every .json/.jsonl/.csv under run-crossgen, run-coords, run-paper16, run-cap,
kd, cap, figures/crossgen (never .pt/.npy/.npz). Line endings are normalised (CRLF->LF) before
hashing because this checkout converts text files. Prints duplicate groups summarised by
folder pair. Read-only; a HINT for Phase 1's duplicate analysis, not a verdict.

Re-run:  PYTHONIOENCODING=utf-8 python audit/scripts/p0_local_dupes.py
"""
import hashlib
import os
from collections import Counter, defaultdict

ROOT = r"C:\Users\SURAIYA\Documents\STUDY\thesis\expLB"
AREAS = ['run-crossgen', 'run-coords', 'run-paper16', 'run-cap', 'kd', 'cap', 'figures/crossgen']
by = defaultdict(list)
n = 0
for a in AREAS:
    for dp, dn, fn in os.walk(os.path.join(ROOT, *a.split('/'))):
        for f in fn:
            if not f.endswith(('.json', '.jsonl', '.csv')):
                continue
            p = os.path.join(dp, f)
            with open(p, 'rb') as fh:
                h = hashlib.md5(fh.read().replace(b'\r\n', b'\n')).hexdigest()
            by[h].append(os.path.relpath(p, ROOT).replace(os.sep, '/'))
            n += 1
groups = {h: ps for h, ps in by.items() if len(ps) > 1}
print('files hashed:', n, ' duplicate groups:', len(groups), ' files in groups:', sum(len(v) for v in groups.values()))
pairs = Counter()
same_name = 0
for ps in groups.values():
    dirs = sorted({p.rsplit('/', 1)[0] for p in ps})
    pairs[' == '.join(dirs[:4]) + (' ...' if len(dirs) > 4 else '')] += 1
    if len({p.rsplit('/', 1)[1] for p in ps}) == 1:
        same_name += 1
print('groups whose members share the file name:', same_name)
for k, v in pairs.most_common(25):
    print(f'{v:5d}  {k}')
