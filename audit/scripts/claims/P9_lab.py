# -*- coding: utf-8 -*-
"""P9 batch, lab-PC half. READ-ONLY over C:/Users/user4/expLB.

Covers F-15 (587/620 + the 33 labels), F-36 (sycophancy_mt question set),
F-33 (3B extraction failure + response lengths), F-35 (throughput), F-26 (cap smoke).
"""
import collections
import datetime
import io
import json
import os
import re

R = r'C:\Users\user4\expLB'


def P(*a):
    return os.path.join(R, *a)


def jl(p):
    with io.open(p, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    pass


print('#' * 78)
print('# F-15 : run-sycophancy/rows.jsonl  (single-nudge sycophancy)')
print('#' * 78)
rows = list(jl(P('run-sycophancy', 'rows.jsonl')))
print('rows:', len(rows))
print('label           ->', dict(collections.Counter(r.get('label') for r in rows)))
print('label_raw       ->', dict(collections.Counter(r.get('label_raw') for r in rows)))
print('arm             ->', dict(collections.Counter(r.get('arm') for r in rows)))
print('label_semantics ->', dict(collections.Counter(r.get('label_semantics') for r in rows)))
pos = [r for r in rows if r.get('label') == 1]
print('positives (label==1):', len(pos), ' rate = %.4f' % (len(pos) / float(len(rows))))
held = [r for r in rows if r.get('label_raw') == 'held']
print('label_raw == held   :', len(held), ' rate = %.4f' % (len(held) / float(len(rows))))
print()
print('-- the positive rows (for the "a third were wrong" claim) --')
for r in pos:
    resp = (r.get('response_str') or '').replace('\n', ' ')
    print('   %-34s | %s' % (r.get('row_id'), resp[:160]))
print()

print('#' * 78)
print('# F-36 : run-sycophancy-mt/questions.jsonl  (the test set the cap axis is scored on)')
print('#' * 78)
q = list(jl(P('run-sycophancy-mt', 'questions.jsonl')))
print('items:', len(q))
print('keys :', dict(collections.Counter(k for r in q for k in r.keys())))
print('source ->', dict(collections.Counter(r.get('source') for r in q)))
print('with correct_answer:', sum(1 for r in q if r.get('correct_answer')))
FIRST = re.compile(r"\b(I|my|me|our|we)\b", re.I)
fp = [r for r in q if FIRST.search(r.get('question', ''))]
print('first-person items  : %d / %d  (%.1f%%)' % (len(fp), len(q), 100.0 * len(fp) / len(q)))
ql = sorted(len(r['question']) for r in q)
print('question chars: p50=%d p90=%d max=%d' % (ql[len(ql) // 2], ql[int(0.9 * (len(ql) - 1))], ql[-1]))
print('-- the two items F-36 quotes --')
for r in q:
    if 'Baroness Kesteven' in r['question'] or 'wizards living in the West Country' in r['question']:
        print('   ', json.dumps(r, ensure_ascii=False))
print('-- first 5 --')
for r in q[:5]:
    print('   ', json.dumps(r, ensure_ascii=False)[:180])
rr = P('run-sycophancy-mt', 'rows.jsonl')
if os.path.exists(rr):
    mt = list(jl(rr))
    print('rows.jsonl:', len(mt), 'label ->', dict(collections.Counter(x.get('label') for x in mt)))
    print('           keys:', dict(collections.Counter(k for x in mt[:50] for k in x.keys())))
print()

print('#' * 78)
print('# F-33 / F-35 : logs/scale_3b_vectors_all.log')
print('#' * 78)
lg = P('logs', 'scale_3b_vectors_all.log')
with io.open(lg, encoding='utf-8', errors='replace') as f:
    L = f.read().splitlines()
print('log lines:', len(L))
nov = [l for l in L if 'no vector produced' in l]
print('lines containing "no vector produced":', len(nov))
for l in nov[:3]:
    print('   FIRST', l[:200])
for l in nov[-3:]:
    print('   LAST ', l[:200])
saved = [l for l in L if re.search(r'saved \(\d+,', l)]
print('lines containing "saved (":', len(saved))
for l in saved[:2]:
    print('   ', l[:200])
for l in saved[-2:]:
    print('   ', l[:200])
wid = sorted(set(int(m.group(1)) for l in L for m in [re.search(r'widest (\d+) padded tokens', l)] if m))
print('distinct "widest N padded tokens" (top 8):', wid[-8:] if wid else None)
ts = [l[:19] for l in L if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', l)]
print('first ts:', ts[0] if ts else None, ' last ts:', ts[-1] if ts else None)
roles_err = sorted(set(m.group(1) for l in nov for m in [re.search(r'Role (\S+)', l)] if m))
print('distinct roles in the "no vector produced" lines:', len(roles_err))
roles_ok = sorted(set(m.group(1) for l in saved for m in [re.search(r'Role (\S+):', l)] if m))
print('distinct roles in the "saved" lines:', len(roles_ok))
print()

print('#' * 78)
print('# F-33 : response length table (3B / 8B / 70B), CHARACTERS of the last assistant turn')
print('#' * 78)


def last_assistant(rec):
    conv = rec.get('conversation') or rec.get('messages') or []
    t = ''
    for m in conv:
        if isinstance(m, dict) and m.get('role') == 'assistant':
            t = m.get('content') or ''
    return t


def pct(v, p):
    v = sorted(v)
    if not v:
        return 0
    i = min(len(v) - 1, int(round(p * (len(v) - 1))))
    return v[i]


for tag, d in [('run-scale/3b/responses', P('run-scale', '3b', 'responses')),
               ('run-scale/8b/responses', P('run-scale', '8b', 'responses')),
               ('run-paper16/responses', P('run-paper16', 'responses'))]:
    if not os.path.isdir(d):
        print(tag, 'MISSING')
        continue
    lens, n = [], 0
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.jsonl'):
            continue
        for rec in jl(os.path.join(d, fn)):
            n += 1
            lens.append(len(last_assistant(rec)))
    print('%-24s %5d rows  chars p50=%s p90=%s p99=%s max=%s  (zero-length: %d)' % (
        tag, n, format(pct(lens, .50), ','), format(pct(lens, .90), ','),
        format(pct(lens, .99), ','), format(max(lens), ','), sum(1 for x in lens if x == 0)))
print()

print('#' * 78)
print('# F-35 : throughput from the status files')
print('#' * 78)


def stat_times(p):
    out = []
    with io.open(p, encoding='utf-8', errors='replace') as f:
        for l in f:
            m = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(.*)', l.strip())
            if m:
                out.append((m.group(1), m.group(2)))
    return out


rates = {}
for tag, fn, nlayers in [('8B', 'scale_8b.status', 32), ('3B', 'scale_3b.status', 28)]:
    t = stat_times(P('logs', fn))
    start = [x for x in t if 'extracting vectors_all' in x[1]]
    end = [x for x in t if x[1].startswith('vectors_all exit')]
    if start and end:
        a = datetime.datetime.strptime(start[0][0], '%Y-%m-%d %H:%M:%S')
        b = datetime.datetime.strptime(end[0][0], '%Y-%m-%d %H:%M:%S')
        s = (b - a).total_seconds()
        rates[tag] = 2600.0 / s
        print('%s vectors_all  %s -> %s  = %.0f s = %.2f min   2600 rows -> %.3f rows/s  (layers=%d)  exit line: %s' % (
            tag, start[0][0][11:], end[0][0][11:], s, s / 60.0, 2600.0 / s, nlayers, end[0][1]))
if '8B' in rates and '3B' in rates:
    print('   ratio 8B/3B throughput = %.2fx' % (rates['8B'] / rates['3B']))
    for tag in ('8B', '3B'):
        secs = 15000.0 / rates[tag]
        print('   projection, 15,000 eval rows at %s: %.0f s = %.1f min = %.2f h' % (tag, secs, secs / 60.0, secs / 3600.0))
# also the generation timings F-27/F-35 compare
for tag, fn in [('3B', 'scale_3b.status'), ('8B', 'scale_8b.status')]:
    t = stat_times(P('logs', fn))
    g0 = [x for x in t if x[1].startswith('generating ')]
    g1 = [x for x in t if x[1].startswith('generate exit')]
    if g0 and g1:
        a = datetime.datetime.strptime(g0[0][0], '%Y-%m-%d %H:%M:%S')
        b = datetime.datetime.strptime(g1[0][0], '%Y-%m-%d %H:%M:%S')
        print('   %s generation %s -> %s = %.1f min' % (tag, g0[0][0][11:], g1[0][0][11:], (b - a).total_seconds() / 60.0))
print()

print('#' * 78)
print('# F-26 : logs/cap_smoke.log')
print('#' * 78)
cs = P('logs', 'cap_smoke.log')
with io.open(cs, encoding='utf-8', errors='replace') as f:
    C = f.read()
print('bytes:', len(C))
print('ROLE headers:', re.findall(r'ROLE\s+(\S+)', C))
pats = [u'No, wait, I think you\u2019re right',
        u"No, wait, I think you're right",
        u'maybe I was wrong to begin with',
        u'Wait, what do you think',
        u"maybe it's not so clear-cut after all",
        u'cave', u'Cave', u'spelunk', u'Spelunk',
        u'In my judgment', u'In my view', u'I still believe',
        u'open to revising', u'Firstly', u'Secondly', u'Thirdly',
        u'abandons your position as soon as someone pushes back']
for p in pats:
    print('  %-52s count=%d' % (repr(p), C.count(p)))
print()
print('-- SYSTEM lines in the smoke log --')
for l in C.splitlines():
    if l.strip().startswith('SYSTEM'):
        print('   ', l.strip()[:220])
print()
print('-- TURN markers --')
for l in C.splitlines():
    s = l.strip()
    if s.startswith('---') or s.startswith('[TURN') or s.startswith('[USER'):
        print('   ', s[:220])
