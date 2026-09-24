"""P9 / F-36: characterise the three cap question sets and the sycophancy_mt test set.
Read-only, local files."""
import json, os, collections, hashlib, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
QD = os.path.join(ROOT, 'cap', 'questions')

def load(fn):
    p = os.path.join(QD, fn)
    b = open(p, 'rb').read()
    rows = [json.loads(l) for l in b.decode('utf-8').splitlines() if l.strip()]
    return p, rows, hashlib.md5(b).hexdigest()[:12], datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec='seconds')

FIRST = ('my ', 'my_', "i ", "i'", ' me ', ' my ', 'our ', ' we ')
def is_first_person(q):
    t = ' ' + q.lower() + ' '
    return any(k in t for k in (' my ', ' i ', " i'", ' me ', ' our ', ' we '))

for fn in ['cap_questions.jsonl', 'cap_questions_v2_draft.jsonl', 'cap_questions_v3.jsonl']:
    p, rows, md5, mt = load(fn)
    print('==== %s  n=%d  md5=%s  mtime=%s' % (fn, len(rows), md5, mt))
    keys = collections.Counter()
    for r in rows: keys.update(r.keys())
    print('   keys:', dict(keys))
    for k in ('kind', 'source', 'framing', 'type', 'category'):
        if any(k in r for r in rows):
            print('   %-8s ->' % k, dict(collections.Counter(r.get(k) for r in rows)))
    has_ans = sum(1 for r in rows if r.get('correct_answer'))
    print('   with correct_answer:', has_ans, '/', len(rows))
    fp = [r for r in rows if is_first_person(r.get('question',''))]
    print('   first-person items:', len(fp), '->', [r.get('id') for r in fp][:12])
    for r in rows[:3]:
        print('   e.g.', json.dumps(r, ensure_ascii=False)[:220])
    print()
