"""P9: recompute the F-28 and F-32 lexical confound counts on the CURRENT cap bank.
Regexes recovered verbatim from the producing commands in session b035ccb3
(lines 4097, 4102 -> F-28; line 4679 -> F-32). Read-only."""
import json, re, os, glob, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D = os.path.join(ROOT, 'cap', 'personas', 'cap_instructions')

PAIRS = [('pc_caver','pc_position_holder'),('pc_deferrer','pc_evidence_bound'),
         ('pc_appeaser','pc_steadfast_reviewer'),('pc_second_guesser','pc_reasoned_defender'),
         ('pc_consensus_follower','pc_calibrated_holder'),('pc_conflict_avoider','pc_disagreement_tolerant')]
CAP  = [a for a,_ in PAIRS]
HOLD = [b for _,b in PAIRS]

def V(r):
    with open(os.path.join(D, r + '.json'), encoding='utf-8') as f:
        return [v['pos'] for v in json.load(f)['instruction']]

# --- F-28's regexes (session line 4097 / 4102) ---
EXPLAIN_28 = re.compile(r'explain|showing the argument|lay out why|reasoning|grounds|argument', re.I)
AFFECT_28  = re.compile(r'unhappy|displeasure|dissatisf|uncomfort|discomfort|comfortable|feel', re.I)
PRAISE_28  = re.compile(r'\b(reasoned|calibrated|evidence bound|steadfast|principled|rigorous)\b', re.I)
LEAK_28    = re.compile(r'without evidence|disagreement does not|not because it was doubted|Objection alone|until something new', re.I)

# --- F-32's acceptance-test regexes (session line 4679) ---
EXPLAIN_32 = re.compile(r'explain|showing the argument|lay out why|reasons aloud|argues why|grounds|argument', re.I)
AFFECT_32  = re.compile(r'unhappy|displeasure|dissatisf|comfort', re.I)
COMMIT_32  = re.compile(r'answers plainly|answer plainly|states a view|gives a view|gives a clear answer|commits to|takes a position|states a verdict|gives an answer|gives an assessment|stating a view', re.I)
PRAISE_32  = re.compile(r'\breasoned|calibrated|bound reasoner\b', re.I)
LEAK_32    = re.compile(r'without evidence|disagreement does not|because it was doubted|Objection alone|until something new', re.I)

def cnt(rx, vs): return sum(bool(rx.search(x)) for x in vs)

print('### file inventory (current working tree)')
for p in sorted(glob.glob(os.path.join(D, '*.json'))):
    b = open(p,'rb').read()
    d = json.loads(b.decode('utf-8'))
    print('  %-28s variants=%d  questions=%d  md5=%s  mtime=%s' % (
        os.path.basename(p), len(d['instruction']), len(d.get('questions',[])),
        hashlib.md5(b).hexdigest()[:12], __import__('datetime').datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec='seconds')))

print()
print('### A. F-32 ACCEPTANCE TABLE re-run on current files (F-32 regexes)')
print('%-24s %-26s  explain   affect   turn1-commit' % ('capitulation','holding'))
for a,b in PAIRS:
    va, vb = V(a), V(b)
    print('%-24s %-26s  %d/5 %d/5   %d/5 %d/5    %d/5 %d/5' % (
        a, b, cnt(EXPLAIN_32,va), cnt(EXPLAIN_32,vb), cnt(AFFECT_32,va), cnt(AFFECT_32,vb),
        cnt(COMMIT_32,va), cnt(COMMIT_32,vb)))
allv = [x for r in CAP for x in V(r)] + [x for r in HOLD for x in V(r)]
print()
print('praise words remaining (F-32 PRAISE on VARIANT TEXT):', cnt(PRAISE_32, allv))
print('leak phrases remaining (F-32 LEAK on VARIANT TEXT)  :', cnt(LEAK_32, allv))
print('max variant length:', max(len(x) for x in allv), 'chars')
print('total variants counted:', len(allv))

print()
print('### B. F-28 regexes re-run on the CURRENT (post-rewrite) files')
print('-- CLAIM 1: explanatory labour (F-28 EXPLAIN, includes "reasoning") --')
for a,b in PAIRS:
    print('  %-24s %d/5   vs  %-26s %d/5' % (a, cnt(EXPLAIN_28,V(a)), b, cnt(EXPLAIN_28,V(b))))
print('-- CLAIM 2: affective vocabulary (F-28 AFFECT) --')
for a,b in PAIRS:
    print('  %-24s %d/5   vs  %-26s %d/5' % (a, cnt(AFFECT_28,V(a)), b, cnt(AFFECT_28,V(b))))
print('-- CLAIM 3: approving modifier in the ROLE NAME (F-28 PRAISE on role keys) --')
print('   capitulation:', [r for r in CAP if PRAISE_28.search(r.replace('_',' '))] or 'NONE')
print('   holding:     ', [r for r in HOLD if PRAISE_28.search(r.replace('_',' '))] or 'NONE')
print('-- CLAIM 3b: same PRAISE regex applied to VARIANT TEXT --')
for r in CAP+HOLD:
    c = cnt(PRAISE_28, V(r))
    if c: print('   %-26s %d/5' % (r, c))
print('-- CLAIM 4: leak phrasing (F-28 LEAK), holding side --')
n=0
for r in HOLD:
    for i,v in enumerate(V(r)):
        if LEAK_28.search(v): print('   %-26s v%d  %s' % (r,i,v)); n+=1
print('   holding matches:', n)
n2=0
for r in CAP:
    for i,v in enumerate(V(r)):
        if LEAK_28.search(v): print('   %-26s v%d  %s' % (r,i,v)); n2+=1
print('   capitulation matches:', n2)

print()
print('### C. regex DELTA between the F-28 count and the F-32 "same counts re-run"')
for name, r28, r32 in [('EXPLAIN', EXPLAIN_28, EXPLAIN_32), ('AFFECT', AFFECT_28, AFFECT_32),
                       ('PRAISE', PRAISE_28, PRAISE_32), ('LEAK', LEAK_28, LEAK_32)]:
    same = r28.pattern == r32.pattern
    print('  %-8s identical=%s' % (name, same))
    if not same:
        print('      F-28: %s' % r28.pattern)
        print('      F-32: %s' % r32.pattern)
print('  PRAISE target: F-28 = role KEY string; F-32 = variant TEXT  -> different quantity')
print('  COMMIT  : new in F-32, no F-28 counterpart')
