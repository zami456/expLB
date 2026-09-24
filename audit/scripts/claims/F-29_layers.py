"""P9 / F-29 robustness: is 'inverted beats random as TARGET but not as SOURCE' layer-robust?"""
import json, glob, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
G = sorted(glob.glob(os.path.join(ROOT, 'run-crossgen', 'kd11', 'matrices', '*_L*_s42.json')))
def means(d, feat, probe='paper'):
    ds = d['datasets']; j = ds.index('sycophancy_mt')
    for r in d['results']:
        if r['feature'] == feat and r['probe'] == probe:
            A = r['auroc']
            col = [A[i][j] for i in range(len(ds)) if i != j]
            row = [A[j][i] for i in range(len(ds)) if i != j]
            return sum(col)/len(col), sum(row)/len(row)
    return None, None
print('%-6s %-5s %7s %8s %8s %8s %8s %8s' % ('pool','L','pc1','axT_inv','rndT','dT','axS_inv','rndS'))
tw = ts = n = 0
for p in G:
    d = json.load(open(p, encoding='utf-8'))
    ds = d['datasets']
    if 'sycophancy_mt' not in ds: continue
    j = ds.index('sycophancy_mt')
    pc1 = [z for z in d['zeroshot'] if z['direction'] != 'contrast'][0]['auroc'][j]
    aT, aS = means(d, 'axes'); rT, rS = means(d, 'random')
    if aT is None: continue
    n += 1
    tw += ((1-aT) > rT); ts += ((1-aS) > rS)
    print('%-6s %-5d %7.4f %8.4f %8.4f %+8.4f %8.4f %8.4f' % (
        d['pooling'], d['layer'], pc1, 1-aT, rT, (1-aT)-rT, 1-aS, rS))
print()
print('inverted axes beats random AS TARGET  : %d of %d matrices' % (tw, n))
print('inverted axes beats random AS SOURCE  : %d of %d matrices' % (ts, n))
