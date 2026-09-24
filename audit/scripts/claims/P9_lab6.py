# -*- coding: utf-8 -*-
"""P9 / F-26: dump the four smoke responses from logs/cap_smoke.log (UTF-16), READ-ONLY."""
import os
import re

R = r'C:\Users\user4\expLB'
p = os.path.join(R, 'logs', 'cap_smoke.log')
raw = open(p, 'rb').read()
T = raw.decode('utf-16')
L = [l.rstrip() for l in T.splitlines()]

keep = []
emit = False
for i, l in enumerate(L):
    s = l.strip()
    if s.startswith('=====') or s.startswith('ROLE ') or s.startswith('SYSTEM') or \
       s.startswith('--- Q') or s.startswith('[TURN') or s.startswith('[USER'):
        emit = True
        keep.append((i + 1, s))
        continue
    if emit and s and not s.startswith('20') and 'HTTP Request' not in s and \
       'INFO -' not in s and 'nnsight' not in s and not s.startswith('+') and \
       not s.startswith('At line') and 'CategoryInfo' not in s and 'FullyQualified' not in s:
        keep.append((i + 1, s))
    if emit and not s:
        emit = False

print('kept lines:', len(keep))
for ln, s in keep:
    print('%6d| %s' % (ln, s[:400]))
print()
print('=' * 70)
print('token/char counts of the TURN blocks')
# reconstruct blocks
blocks = []
cur = None
for l in L:
    s = l.strip()
    m = re.match(r'^ROLE\s+(\S+)', s)
    if m:
        role = m.group(1)
    if s.startswith('--- Q'):
        qid = s
    if s.startswith('[TURN 1'):
        cur = ['t1', role, qid, []]
        blocks.append(cur)
        continue
    if s.startswith('[TURN 2'):
        cur = ['t2', role, qid, []]
        blocks.append(cur)
        continue
    if s.startswith('[USER'):
        cur = None
        continue
    if cur is not None:
        if s and 'HTTP Request' not in s and ' - INFO - ' not in s and not s.startswith('====='):
            cur[3].append(s)
        if s.startswith('====='):
            cur = None
for b in blocks:
    txt = ' '.join(b[3])
    print('%-3s %-20s %-70s chars=%d' % (b[0], b[1], b[2][:70], len(txt)))
