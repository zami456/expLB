# -*- coding: utf-8 -*-
"""P9 batch, lab part 4. READ-ONLY.
F-25 : run-crossgen/kd11/passC/significance.json (1,152 tests) + the .stale-576 file
       + logs/kd_passAC.log lines 1337-1348 (the PermissionError traceback)
       + when the write_json retry loop was added.
Also  : the framing/kind split of the cap question sets as they sit on the lab PC.
"""
import datetime
import io
import json
import os
import re

R = r'C:\Users\user4\expLB'


def P(*a):
    return os.path.join(R, *a)


def read_text(p):
    raw = open(p, 'rb').read()
    for enc in ('utf-16', 'utf-8-sig', 'utf-8'):
        try:
            t = raw.decode(enc)
        except Exception:
            continue
        if t.count('\x00') < len(t) * 0.01 and t.count('\n') > 10:
            return t, enc
    return raw.decode('utf-8', 'replace'), 'utf-8/replace'


print('#' * 78)
print('# F-25 : passC root significance files')
print('#' * 78)
for d in (P('run-crossgen', 'kd11', 'passC'), P('run-crossgen', 'passC')):
    print('== dir:', d, 'exists =', os.path.isdir(d))
    if not os.path.isdir(d):
        continue
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if os.path.isdir(p):
            print('   [dir] %-40s %d entries' % (fn, len(os.listdir(p))))
            continue
        sz = os.path.getsize(p)
        mt = datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec='seconds')
        extra = ''
        if fn.startswith('significance'):
            try:
                o = json.load(io.open(p, encoding='utf-8'))
                if isinstance(o, dict):
                    ks = list(o.keys())
                    n = None
                    for k in ('tests', 'results', 'comparisons'):
                        if isinstance(o.get(k), list):
                            n = len(o[k])
                            extra = '  %s=%d' % (k, n)
                    extra += '  keys=%s' % ks[:8]
                    if 'note' in o:
                        extra += '  note=%r' % str(o['note'])[:140]
                elif isinstance(o, list):
                    extra = '  top-level list len=%d' % len(o)
            except Exception as e:
                extra = '  JSON ERROR %s' % e
        print('   %-44s %10d  %s%s' % (fn, sz, mt, extra))
print()

print('#' * 78)
print('# F-25 : logs/kd_passAC.log around the C-root failure')
print('#' * 78)
lp = P('logs', 'kd_passAC.log')
T, enc = read_text(lp)
L = T.splitlines()
print('lines:', len(L), 'encoding:', enc, 'mtime:',
      datetime.datetime.fromtimestamp(os.path.getmtime(lp)).isoformat(timespec='seconds'))
print('-- lines 1330-1352 (1-based) --')
for i in range(1329, min(1352, len(L))):
    print('  %5d| %s' % (i + 1, L[i][:220]))
print()
print('-- every line mentioning PermissionError / WinError 5 / significance.json.tmp / stale --')
for i, l in enumerate(L, 1):
    if re.search(r'PermissionError|WinError 5|significance\.json\.tmp|stale-|exit 1', l):
        print('  %5d| %s' % (i, l[:220]))
print()
print('-- C-root step markers --')
for i, l in enumerate(L, 1):
    if re.search(r'C-root|stage_summarize|write_json', l):
        print('  %5d| %s' % (i, l[:220]))
print()

print('#' * 78)
print('# F-25 : does write_json now carry a retry loop?  (source, lab copy)')
print('#' * 78)
for cand in (P('src', 'assistant_axis_ndif', '27_crossgen.py'),
             P('src', 'assistant_axis_ndif', 'ndif_common.py')):
    if not os.path.exists(cand):
        print('  MISSING', cand)
        continue
    S = io.open(cand, encoding='utf-8', errors='replace').read()
    m = re.search(r'def write_json\b.*?(?=\ndef |\nclass |\Z)', S, re.S)
    print('== %s  mtime=%s' % (cand, datetime.datetime.fromtimestamp(os.path.getmtime(cand)).isoformat(timespec='seconds')))
    if m:
        print(m.group(0)[:1600])
    else:
        print('   no write_json in this file')
print()

print('#' * 78)
print('# cap question sets on the lab PC (framing/kind)')
print('#' * 78)
import collections
qd = P('cap', 'questions')
for fn in sorted(os.listdir(qd)):
    p = os.path.join(qd, fn)
    rows = [json.loads(l) for l in io.open(p, encoding='utf-8') if l.strip()]
    print('  %-34s n=%2d  kind=%s  framing=%s  with_answer=%d' % (
        fn, len(rows),
        dict(collections.Counter(r.get('kind') for r in rows)),
        dict(collections.Counter(r.get('framing') for r in rows)),
        sum(1 for r in rows if r.get('correct_answer'))))
