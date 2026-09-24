"""M-004: memory note crossgen-80-20-retrain.md:23
"ID-trained row, L40, mean over the 9 other datasets (100%): mms mean pooling raw 0.817 /
pc:deception 0.649 / axes 0.727 (w/o sycophancy 0.846 / 0.693 / 0.776); last pooling raw 0.778 /
pc 0.654 / axes 0.724; lr_reg10 mean raw 0.727. Controls are high on HP-KR (random 0.767, length 0.776)."

ID = instructed-deception (row 0 of the 10x10 crossgen matrix).
Reads run-crossgen/matrices/{mean,last}_L40_s42.json (path given as argv[1] root).
Read-only.
"""
import json, sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "run-crossgen/matrices"

def mat(path):
    d = json.load(open(path))
    return d

def offdiag_row(d, feature, probe, row_ds, exclude=()):
    ds = d['datasets']
    i = ds.index(row_ds)
    for r in d['results']:
        if r['feature'] == feature and r['probe'] == probe:
            vals = [(ds[j], r['auroc'][i][j]) for j in range(len(ds))
                    if j != i and ds[j] not in exclude]
            m = sum(v for _, v in vals) / len(vals)
            return m, len(vals), vals
    return None, 0, []

for pool in ['mean', 'last']:
    p = os.path.join(root, f'{pool}_L40_s42.json')
    d = mat(p)
    print(f'### {p}  bank={d["bank"]} k={d["k"]} layer={d["layer"]} seed={d["seed"]} '
          f'pooling={d["pooling"]} n_datasets={len(d["datasets"])}')
    for probe in ['mms', 'lr_reg10', 'paper']:
        for feat in ['raw', 'pc:deception', 'axes']:
            m, n, _ = offdiag_row(d, feat, probe, 'instructed-deception')
            m2, n2, _ = offdiag_row(d, feat, probe, 'instructed-deception',
                                    exclude=('sycophancy_mt',))
            print(f'  {pool:5s} {probe:8s} {feat:13s} ID-row mean over {n} others = {m:.4f}'
                  f'   | w/o sycophancy_mt (n={n2}) = {m2:.4f}')
    # controls on HP-KR: which direction? report both row-mean and column-mean
    ds = d['datasets']
    k = ds.index('harm-pressure-knowledge-report')
    for feat in ['random', 'length']:
        for probe in ['mms', 'paper', 'lr_reg10']:
            for r in d['results']:
                if r['feature'] == feat and r['probe'] == probe:
                    a = r['auroc']
                    diag = a[k][k]
                    rowm = sum(a[k][j] for j in range(len(ds)) if j != k) / (len(ds) - 1)
                    colm = sum(a[i][k] for i in range(len(ds)) if i != k) / (len(ds) - 1)
                    print(f'  {pool:5s} {probe:8s} {feat:8s} HP-KR diag={diag:.4f} '
                          f'row(off)={rowm:.4f} col(off)={colm:.4f} ID->HPKR={a[0][k]:.4f}')
    print()
