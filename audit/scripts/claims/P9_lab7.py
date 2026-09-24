# -*- coding: utf-8 -*-
"""P9 / F-26 (2): exact block dump of logs/cap_smoke.log. READ-ONLY.
Prints every line between a [TURN n] marker and the next marker, and locates the
four strings F-26 quotes."""
import os
import re

R = r'C:\Users\user4\expLB'
p = os.path.join(R, 'logs', 'cap_smoke.log')
T = open(p, 'rb').read().decode('utf-16')
L = [l.rstrip() for l in T.splitlines()]

MARK = re.compile(r'^\s*(ROLE\s|SYSTEM|---\s*Q\d|\[TURN|\[USER|=====)')
NOISE = re.compile(r'HTTP Request|- INFO -|- WARNING -|nnsight|^\s*\+|^At line|CategoryInfo|FullyQualified|python\.exe :')

idx = [i for i, l in enumerate(L) if MARK.match(l)]
for a, i in enumerate(idx):
    j = idx[a + 1] if a + 1 < len(idx) else len(L)
    head = L[i].strip()
    body = [l.strip() for l in L[i + 1:j] if l.strip() and not NOISE.search(l)]
    txt = ' '.join(body)
    if head.startswith('[TURN') or head.startswith('[USER'):
        print('-' * 70)
        print('line %d  %s   body_chars=%d' % (i + 1, head, len(txt)))
        print(txt[:1400])
    elif head.startswith('ROLE') or head.startswith('SYSTEM') or head.startswith('--- Q'):
        print()
        print('### line %d  %s' % (i + 1, head[:200]))

print()
print('=' * 70)
print('locations of the strings F-26 quotes')
for q in [u'Wait, what do you think',
          u'not so clear-cut after all',
          u'No, wait, I think you',
          u'maybe I was wrong to begin with',
          u'In my judgment',
          u'In my view',
          u'I still believe',
          u'open to revising',
          u'Firstly', u'Secondly', u'Thirdly']:
    for i, l in enumerate(L, 1):
        if q in l:
            print('  %-34s line %5d : %s' % (repr(q), i, l.strip()[:220]))
