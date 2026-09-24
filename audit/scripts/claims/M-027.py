# M-027: F-96 vs F-63 -- "3B margin +0.202 at L16 'matches the paper's +0.22';
#          L16 is the max of 3B's 7 layers; layer-mean +0.120"
# Repo side: audit/phase2/layer_sensitivity.csv (W1, built from the lab scale stores).
# Paper side: the per-cell Delta matrices printed in the PDF (Appendix Figure 9, 3B deception, L14, k=1).
import csv, statistics, pypdf, re

SRC = 'audit/phase2/layer_sensitivity.csv'
rows = [r for r in csv.DictReader(open(SRC, encoding='utf-8'))
        if r['pooling'] == 'mean' and r['probe'] == 'paper' and r['seed'] == '42'
        and r['dataset_set'] == 'paper5' and r['k'] == '3']
print(f'{SRC}: {len(rows)} rows with pooling=mean probe=paper seed=42 dataset_set=paper5 k=3')

def by(store):
    d = {}
    for r in rows:
        if r['store'] != store:
            continue
        d.setdefault(int(r['layer']), {})[r['feature']] = float(r['offdiag_mean'])
    return d

for store, cell in [('scale_3b', 16), ('scale_8b', 16), ('scale_70b', 40)]:
    d = by(store)
    if not d:
        print(f'{store}: NO ROWS'); continue
    margins = {}
    for L, f in sorted(d.items()):
        if 'datapca' not in f:
            continue
        best = max(v for k, v in f.items() if k in ('axes', 'pc:deception'))
        margins[L] = best - f['datapca']
    order = sorted(margins.items(), key=lambda t: -t[1])
    rank = [L for L, _ in order].index(cell) + 1 if cell in margins else None
    print(f'\n{store}: {len(margins)} layers with both a persona feature and datapca')
    print('   per layer best(axes,pc:deception) - datapca:',
          {L: round(v, 4) for L, v in sorted(margins.items())})
    print(f'   L{cell} = {margins[cell]:+.4f}   layer-mean = {statistics.fmean(margins.values()):+.4f}'
          f'   range [{min(margins.values()):+.4f}, {max(margins.values()):+.4f}]   rank of L{cell} = {rank} of {len(margins)}')

print('\n--- the paper comparator "+0.22" (Mahadik & Skapars, Appendix Fig 9, 3B deception, layer 14, k=1)')
r = pypdf.PdfReader('papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf')
t = r.pages[11].extract_text()
blocks = re.split(r'(Random Subspace |Dataset PCA |Axis PC Projection )', t)
def grab(after):
    i = t.index(after)
    pre = t[:i].strip().split('\n')
    nums = []
    for line in reversed(pre):
        vals = re.findall(r'-?\d\.\d\d', line)
        if len(vals) == 5:
            nums.insert(0, [float(v) for v in vals])
        elif nums:
            break
        if len(nums) == 5:
            break
    return nums
for label in ['Random Subspace Δ', 'Dataset PCA Δ', 'Axis PC Projection Δ']:
    M = grab(label)
    off = [M[i][j] for i in range(5) for j in range(5) if i != j]
    print(f'   {label:26s} 5x5 read from the PDF, off-diagonal mean over {len(off)} cells = {statistics.fmean(off):+.4f}')
    globals()[label.split()[0]] = statistics.fmean(off)
print(f'   Axis PC Projection minus Dataset PCA = {globals()["Axis"] - globals()["Dataset"]:+.4f}  <- the "+0.22"')
