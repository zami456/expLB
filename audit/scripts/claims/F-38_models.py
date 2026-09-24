"""P9 / F-38: the `model` column of the three subsets F-38 tabulates. Read-only, no truncation."""
import csv, os, collections, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
for name in ['gender-secret', 'soft-trigger', 'instructed-deception']:
    p = os.path.join(ROOT, 'results', 'subsets', name + '.csv')
    c = collections.Counter(); n = 0
    with open(p, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        for row in r:
            n += 1; c[row.get('model')] += 1
    print('== %s  rows=%d  file=%s' % (name, n, p))
    print('   columns:', cols)
    print('   distinct models: %d' % len(c))
    for m, k in c.most_common():
        print('      %-46s %6d' % (m, k))
    print()
