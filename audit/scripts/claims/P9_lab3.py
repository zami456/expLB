# -*- coding: utf-8 -*-
"""P9 batch, lab part 3. READ-ONLY.
What the PRODUCTION cap run (the one behind F-65/F-66) actually used:
which persona variant text, which question set, whether turn 1 was teacher-forced.
Also: the cap persona files on the lab PC vs the repo copies (md5), and the
pre-rewrite variant text recoverable from cap_smoke.log / cap_smoke_v2.log."""
import collections
import hashlib
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
print('# lab cap/personas/cap_instructions  vs  the repo copies')
print('#' * 78)
d = P('cap', 'personas', 'cap_instructions')
if os.path.isdir(d):
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        b = open(p, 'rb').read()
        import datetime
        print('  %-30s %7d  md5=%s  mtime=%s' % (
            fn, len(b), hashlib.md5(b).hexdigest()[:12],
            datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec='seconds')))
else:
    print('  MISSING', d)
print()
qd = P('cap', 'questions')
if os.path.isdir(qd):
    import datetime
    for fn in sorted(os.listdir(qd)):
        p = os.path.join(qd, fn)
        b = open(p, 'rb').read()
        print('  %-34s %7d  md5=%s  mtime=%s' % (
            fn, len(b), hashlib.md5(b).hexdigest()[:12],
            datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec='seconds')))
print()

print('#' * 78)
print('# run-cap/responses : what the production rollouts record')
print('#' * 78)
rd = P('run-cap', 'responses')
files = sorted(os.listdir(rd))
print('files:', len(files), files[:20])
first = os.path.join(rd, files[0])
recs = []
with io.open(first, encoding='utf-8', errors='replace') as f:
    for line in f:
        if line.strip():
            recs.append(json.loads(line))
print('%s : %d rows' % (files[0], len(recs)))
print('keys:', list(recs[0].keys()))
r0 = recs[0]
for k, v in r0.items():
    if k in ('conversation', 'messages'):
        print('  %s:' % k)
        for m in v:
            print('     [%s] %s' % (m.get('role'), (m.get('content') or '')[:300].replace('\n', ' ')))
    else:
        print('  %s: %s' % (k, str(v)[:260]))
print()
# turn-1 identical across personas?
t1 = collections.Counter()
sysp = collections.Counter()
nq = collections.Counter()
total = 0
for fn in files:
    if not fn.endswith('.jsonl'):
        continue
    with io.open(os.path.join(rd, fn), encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            conv = rec.get('conversation') or rec.get('messages') or []
            asst = [m for m in conv if m.get('role') == 'assistant']
            if asst:
                t1[(rec.get('question_index'), asst[0].get('content'))] += 1
            sysp[rec.get('system_prompt')] += 1
            nq[rec.get('question_index')] += 1
print('total production rollouts:', total)
print('distinct system prompts  :', len(sysp))
print('distinct question indices:', len(nq), sorted(k for k in nq if k is not None)[:25])
print('distinct (question_index, turn-1 assistant text) pairs:', len(t1))
print('  -> if that equals the number of questions, turn 1 is IDENTICAL for every persona')
print('-- the turn-1 texts (first 8) --')
for (qi, txt), c in sorted(t1.items(), key=lambda x: (x[0][0] if x[0][0] is not None else -1))[:8]:
    print('   q=%s  n=%d  %r' % (qi, c, (txt or '')[:90]))
print()
print('-- a few system prompts --')
for s, c in list(sysp.items())[:6]:
    print('   n=%d  %r' % (c, (s or '')[:120]))
print()

print('#' * 78)
print('# pre-rewrite variant text recoverable from the smoke logs')
print('#' * 78)
for fn in ('cap_smoke.log', 'cap_smoke_v2.log', 'cap_tf_smoke.log', 'cap_v4_cells.log'):
    p = P('logs', fn)
    if not os.path.exists(p):
        print('  MISSING', fn)
        continue
    T, e = read_text(p)
    import datetime
    print('== %s  (%s, %d chars, mtime %s)' % (fn, e, len(T),
          datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec='seconds')))
    roles = re.findall(r'ROLE\s+(\S+)', T)
    print('   ROLE headers:', roles)
    for l in T.splitlines():
        s = l.strip()
        if s.startswith('SYSTEM'):
            print('   ', s[:200])
