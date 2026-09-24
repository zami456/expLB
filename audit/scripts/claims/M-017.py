# M-017: pc:unified@2 - datapca@2 = -0.003 (matched control), run-crossgen/significance.json
import json
P='run-crossgen/significance.json'
s=json.load(open(P,encoding='utf-8'))
print('top-level keys:', list(s.keys()))
tests=s['tests'] if isinstance(s,dict) and 'tests' in s else s
print('n tests in this (git) copy =', len(tests), '  <- BH family size')
sel=[t for t in tests if t.get('block')=='10' and t.get('layer')==40 and t.get('pooling')=='mean'
     and t.get('probe')=='paper']
print('\nL40 / mean / paper / block "10" tests:')
for t in sel:
    print(f'  {t["comparison"]:34s} diff={t.get("diff"):+.4f} pairs_won={t.get("pairs_won")} '
          f'ci95={t.get("dataset_ci95")} q_bh={t.get("dataset_q_bh")} p={t.get("p")}')
