# -*- coding: utf-8 -*-
"""P9 / F-25 follow-up. READ-ONLY.
The recovered kd11/passC/significance.json is byte-identical (md5 122adfbf...) to
run-crossgen/passC/significance.json written two days earlier. Work out what that means:
what grid each file covers, and what the kd11 passC matrices actually contain."""
import collections
import io
import json
import os

R = r'C:\Users\user4\expLB'


def P(*a):
    return os.path.join(R, *a)


for tag, d in (('kd11/passC', P('run-crossgen', 'kd11', 'passC')),
               ('root/passC', P('run-crossgen', 'passC'))):
    print('=' * 70)
    print(tag)
    md = os.path.join(d, 'matrices')
    fs = sorted(os.listdir(md))
    print('  matrices: %d ->' % len(fs), fs[:80])
    sig = os.path.join(d, 'significance.json')
    o = json.load(io.open(sig, encoding='utf-8'))
    print('  design       :', str(o.get('design'))[:150])
    print('  boot_datasets:', o.get('boot_datasets'))
    print('  note         :', o.get('note'))
    t = o['tests']
    print('  tests        :', len(t))
    print('  test keys    :', list(t[0].keys()))
    for k in ('block', 'pooling', 'layer', 'probe', 'comparison'):
        if k in t[0]:
            c = collections.Counter(x.get(k) for x in t)
            print('    %-11s -> %s' % (k, dict(sorted(c.items(), key=lambda y: str(y[0])))))
    # sample one matrix for the dataset list
    m = json.load(io.open(os.path.join(md, fs[0]), encoding='utf-8'))
    print('  matrix[0] %s: layer=%s pooling=%s seed=%s bank=%s bank_name=%s k=%s' % (
        fs[0], m.get('layer'), m.get('pooling'), m.get('seed'), m.get('bank'), m.get('bank_name'), m.get('k')))
    print('    datasets (%d): %s' % (len(m.get('datasets', [])), m.get('datasets')))
    if 'block9' in m:
        print('    block9   (%d): %s' % (len(m['block9']), m['block9']))
    print('  summary.json size:', os.path.getsize(os.path.join(d, 'summary.json')))
    s = json.load(io.open(os.path.join(d, 'summary.json'), encoding='utf-8'))
    if isinstance(s, dict):
        print('  summary keys:', list(s.keys())[:10])
        for k in ('records', 'rows', 'results'):
            if isinstance(s.get(k), list):
                print('   summary %s: %d' % (k, len(s[k])))
                rec = s[k][0]
                print('    rec keys:', list(rec.keys())[:14])
                print('    layers in summary:', sorted(set(x.get('layer') for x in s[k])))
                print('    blocks in summary:', sorted(set(str(x.get('block')) for x in s[k])))
