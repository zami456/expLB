"""P3-C: fingerprint the 5 stores whose bank BASENAME is ambiguous (provenance.csv).

Logic: the same bank on the same activation store must give IDENTICAL persona-feature
off-diagonal means. 27_crossgen.py records only Path(bank).name (line 612), so a store
whose basename exists in >1 folder cannot be resolved from its own file. Compare its
'axes' / 'pc:deception' values against kd11 (known paper16 fp32) on matched keys.
Read-only. Input: audit/phase2/layer_sensitivity.csv
"""
import csv, collections, sys

ROWS = list(csv.DictReader(open('audit/phase2/layer_sensitivity.csv', encoding='utf-8-sig')))
KEY = ('pooling', 'layer', 'seed', 'dataset_set', 'probe', 'feature')

def index(store):
    d = {}
    for r in ROWS:
        if r['store'] == store:
            d[tuple(r[k] for k in KEY)] = float(r['offdiag_mean'])
    return d

REF = {'kd11': index('kd11'), '<root>': index('<root>')}
targets = ['kd11/banks/kd', 'kd11/banks/kd_s23', 'kd11/banks/cap_s23', 'kd11/banks/cap_ctrl',
           'kd11/banks/cap_all', 'kd11/banks/collab', 'banks/paper16_shared', 'banks/own100',
           'banks/paper16_famcentred', 'banks/hf100', 'scale_70b', 'scale_70b_paper16fp32']

for t in targets:
    d = index(t)
    for refname, ref in REF.items():
        common = [k for k in d if k in ref and (k[-1] == 'axes' or k[-1].startswith('pc:'))]
        if not common:
            continue
        diffs = [abs(d[k] - ref[k]) for k in common]
        exact = sum(1 for x in diffs if x == 0.0)
        print(f"{t:28s} vs {refname:8s} n={len(common):5d} exact={exact:5d} "
              f"max|d|={max(diffs):.6f} mean|d|={sum(diffs)/len(diffs):.6f}")
    # also raw/datapca (baseline columns, expected identical by design)
    common = [k for k in d if k in REF['kd11'] and k[-1] in ('raw', 'datapca')]
    if common:
        diffs = [abs(d[k] - REF['kd11'][k]) for k in common]
        print(f"{'':28s}    baseline(raw/datapca) n={len(common):5d} "
              f"exact={sum(1 for x in diffs if x==0.0):5d} max|d|={max(diffs):.6f}")
    print()
