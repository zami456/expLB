# -*- coding: utf-8 -*-
"""P9 batch, lab-PC half, part 2. READ-ONLY.
F-26 (cap_smoke.log, encoding-aware), F-15 (lexical proxy for the mislabelled positives),
F-33 (pre-restart attempt count)."""
import collections
import io
import json
import os
import re

R = r'C:\Users\user4\expLB'


def P(*a):
    return os.path.join(R, *a)


def read_text(p):
    raw = open(p, 'rb').read()
    for enc in ('utf-16', 'utf-16-le', 'utf-8-sig', 'utf-8'):
        try:
            t = raw.decode(enc)
        except Exception:
            continue
        # heuristic: a real log has many newlines and few NULs
        if t.count('\x00') < len(t) * 0.01 and t.count('\n') > 10:
            return t, enc, len(raw)
    return raw.decode('utf-8', 'replace'), 'utf-8/replace', len(raw)


print('#' * 78)
print('# F-26 : logs/cap_smoke.log  (encoding-aware)')
print('#' * 78)
cs = P('logs', 'cap_smoke.log')
C, enc, nb = read_text(cs)
print('bytes: %d   decoded as: %s   chars: %d   lines: %d' % (nb, enc, len(C), C.count('\n') + 1))
print('mtime:', __import__('datetime').datetime.fromtimestamp(os.path.getmtime(cs)).isoformat(timespec='seconds'))
pats = [u'ROLE ', u'SYSTEM', u'TURN 1', u'TURN 2', u'[USER]',
        u'pc_caver', u'pc_position_holder',
        u'No, wait, I think you', u'maybe I was wrong to begin with',
        u'Wait, what do you think', u"not so clear-cut after all",
        u'caver', u'cave', u'spelunk', u'Spelunk',
        u'In my judgment', u'In my view', u'I still believe',
        u'open to revising', u'Firstly', u'Secondly', u'Thirdly',
        u'abandons your position as soon as someone pushes back',
        u'flaky test suite', u'documentation before the code']
for p in pats:
    print('  %-56s count=%d' % (repr(p), C.count(p)))
print()
print('-- first 25 non-blank lines --')
n = 0
for l in C.splitlines():
    if l.strip():
        print('   ', l.strip()[:200])
        n += 1
        if n >= 25:
            break
print()
print('-- lines mentioning pc_caver or pc_position_holder (first 40) --')
n = 0
for l in C.splitlines():
    if 'pc_caver' in l or 'pc_position_holder' in l:
        print('   ', l.strip()[:250])
        n += 1
        if n >= 40:
            break
print('   (total such lines: %d)' % sum(1 for l in C.splitlines() if 'pc_caver' in l or 'pc_position_holder' in l))
print()
print('-- any SYSTEM/system_prompt json-ish lines (first 10) --')
n = 0
for l in C.splitlines():
    if 'system' in l.lower() and n < 10:
        print('   ', l.strip()[:250])
        n += 1
print()

print('#' * 78)
print('# other cap smoke-ish logs on the lab PC')
print('#' * 78)
ld = P('logs')
for fn in sorted(os.listdir(ld)):
    if 'cap' in fn.lower() or 'smoke' in fn.lower() or 'sarc' in fn.lower():
        p = os.path.join(ld, fn)
        print('   %10d  %s  %s' % (os.path.getsize(p),
                                   __import__('datetime').datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec='seconds'), fn))
print()
print('-- cap/run smoke dirs --')
for d in ('run', 'run-cap'):
    base = P('cap', d) if d == 'run' else P(d)
    if os.path.isdir(base):
        for sub in sorted(os.listdir(base)):
            print('   %s/%s' % (base.replace(R + os.sep, ''), sub))
print()

print('#' * 78)
print('# F-15 : lexical proxy for "a third of the 33 labels were wrong"')
print('#' * 78)


def jl(p):
    with io.open(p, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    pass


rows = list(jl(P('run-sycophancy', 'rows.jsonl')))
pos = [r for r in rows if r.get('label') == 1]
CORRECT = re.compile(
    r'\bis not\b|\bwas not\b|\bare not\b|\bdoes not\b|\bdid not\b|, not |\bnot the\b|\bactually\b'
    r'|\bmyth\b|\bmisconception\b|debunk|\bincorrect\b|\bI\'m afraid\b|\bno scientific basis\b'
    r'|\bis a common\b|\bin fact\b', re.I)
hits = [r for r in pos if CORRECT.search(r.get('response_str') or '')]
print('positives: %d   lexical "correction/refutation" hits: %d  (%.1f%% = %.2f of the set)' % (
    len(pos), len(hits), 100.0 * len(hits) / len(pos), len(hits) / float(len(pos))))
print('NOTE: this is a PROXY rule invented by the audit, not the original judgement.')
print('-- hits --')
for r in hits:
    print('   %-30s %s' % (r.get('row_id'), (r.get('response_str') or '').replace('\n', ' ')[:110]))
print('-- non-hits --')
for r in pos:
    if r not in hits:
        print('   %-30s %s' % (r.get('row_id'), (r.get('response_str') or '').replace('\n', ' ')[:110]))
print()
print('-- prompts of the 33 positives (arm split) --')
print('   arm ->', dict(collections.Counter(r.get('arm') for r in pos)))
print()

print('#' * 78)
print('# F-33 : the pre-restart 3B attempt sequence')
print('#' * 78)
lg = P('logs', 'scale_3b_vectors_all.log')
T, e2, b2 = read_text(lg)
L = T.splitlines()
print('decoded as', e2, 'lines', len(L))
seq = []
for l in L:
    m = re.search(r'Role (\S+?):? (no vector produced|saved \()', l)
    if m:
        seq.append((l[:19], m.group(1), 'FAIL' if 'no vector' in m.group(2) else 'OK'))
fails = [s for s in seq if s[2] == 'FAIL']
oks = [s for s in seq if s[2] == 'OK']
print('FAIL events: %d over %d distinct roles' % (len(fails), len(set(s[1] for s in fails))))
print('OK   events: %d over %d distinct roles' % (len(oks), len(set(s[1] for s in oks))))
print('first FAIL', fails[0] if fails else None, ' last FAIL', fails[-1] if fails else None)
print('first OK  ', oks[0] if oks else None, ' last OK  ', oks[-1] if oks else None)
print('-- FAIL roles in order --')
print('   ', [s[1] for s in fails])
print('-- roles that appear ONLY in FAIL (never saved) --')
print('   ', sorted(set(s[1] for s in fails) - set(s[1] for s in oks)))
# how many roles were "attempted": count 'Role X' dispatch lines
att = sorted(set(m.group(1) for l in L for m in [re.search(r'Role (\S+?)[:\s]', l)] if m))
print('distinct roles named anywhere in the log: %d' % len(att))
