"""P9: F-26 and F-33 both cite 'length-only off-diagonal 0.553 vs raw 0.665'.
Recompute the off-diagonal means of the `length` and `raw` features locally."""
import json, glob, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
def off(A, n): return sum(A[i][j] for i in range(n) for j in range(n) if i != j) / (n*(n-1))
for p in sorted(glob.glob(os.path.join(ROOT,'run-crossgen','kd11','matrices','mean_L40_s42.json'))
              + glob.glob(os.path.join(ROOT,'run-crossgen','matrices','mean_L40_s42.json'))):
    d = json.load(open(p, encoding='utf-8')); n = len(d['datasets'])
    print('== %s  n_datasets=%d probe=paper' % (os.path.relpath(p, ROOT), n))
    for r in d['results']:
        if r['probe'] != 'paper': continue
        if r['feature'] in ('length','raw','datapca','random','axes'):
            print('   %-12s off-diag %.4f   (%d cells)' % (r['feature'], off(r['auroc'], n), n*(n-1)))
