"""M-032: memory note persona-axes-encode-register.md:19-22 (claim row points at line 20)

    | harm-pressure-knowledge-report | ... | 0.427 target / 0.300 source |
    | sycophancy_mt                  | ... | 0.355 target / 0.379 source |
    "On both, a RANDOM direction beats the persona axes (HP-KR 0.698/0.659; SYC 0.541/0.633)."

Note header: "Measured 2026-09-20 on the 11-dataset grid (run-crossgen/kd11, L40, mean, paper probe)."

"target" is read as the column mean (that dataset as the TEST set, trained elsewhere) and
"source" as the row mean (trained on that dataset, tested elsewhere); both off-diagonal.
Also prints the complementary reading so the note cannot be scored on one guess.
Read-only.
"""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'run-crossgen/kd11/matrices/mean_L40_s42.json'
probe = sys.argv[2] if len(sys.argv) > 2 else 'paper'

d = json.load(open(path))
ds = d['datasets']
n = len(ds)
print(f'### {path}')
print(f'    pooling={d["pooling"]} layer={d["layer"]} seed={d["seed"]} bank={d["bank"]} k={d["k"]}')
print(f'    datasets ({n}): {ds}')

targets = ['harm-pressure-knowledge-report', 'sycophancy_mt']
feats = ['axes', 'random', 'pc:deception', 'pc:unified', 'raw', 'datapca']

for feat in feats:
    m = None
    for r in d['results']:
        if r['feature'] == feat and r['probe'] == probe:
            m = r['auroc']
    if m is None:
        continue
    print(f'\n  feature={feat} probe={probe}')
    for t in targets:
        k = ds.index(t)
        colm = sum(m[i][k] for i in range(n) if i != k) / (n - 1)   # dataset as TEST target
        rowm = sum(m[k][j] for j in range(n) if j != k) / (n - 1)   # dataset as TRAIN source
        # the same, restricted to the 10 datasets excluding the other special dataset
        print(f'    {t:32s} column(off-diag, as test target) = {colm:.4f}   '
              f'row(off-diag, as train source) = {rowm:.4f}   diag = {m[k][k]:.4f}')
