# M-002: at L20 the paper's 8v7 axis (axis1) reads 0.6511 TRAINED off-diagonal vs 0.7909 ZERO-SHOT
# Source: collaborator branch, read with git show (no checkout).
import subprocess, io, csv, statistics
BR='origin/feat/persona-pc-ood'
base='src/probes/deception-detection/results'
def show(p):
    return subprocess.run(['git','show',f'{BR}:{p}'],capture_output=True,text=True,encoding='utf-8').stdout
for L in ['20','40','44']:
    d=f'{base}/persona_ood_ext_L{L}'
    s=show(f'{d}/summary.csv')
    if not s: print(f'==== L{L}: NO summary.csv on branch'); continue
    print(f'==== L{L}  ({d})')
    rd=list(csv.DictReader(io.StringIO(s)))
    for r in rd:
        if r.get('features') in ('axis1','ext_axis1','baseline','pc1','ext_pc1'):
            print(f'   summary.csv  {r["features"]:10s} dim={r.get("dim")}  off_diagonal_mean={r.get("off_diagonal_mean")}')
    z=show(f'{d}/zero_shot.csv')
    zr=[r for r in csv.DictReader(io.StringIO(z)) if r.get('auroc_axis','').strip()]
    print('   zero_shot.csv scored datasets:', [r['dataset'] for r in zr])
    for col in ['auroc_axis','auroc_ext_axis','auroc_pc1','auroc_ext_pc1']:
        v=[float(r[col]) for r in zr if r.get(col,'').strip()]
        if v: print(f'   mean({col}) over {len(v)} datasets = {statistics.fmean(v):.10f}')
    m=show(f'{d}/matrix_axis1.csv')
    mr=[r for r in csv.reader(io.StringIO(m)) if r]
    hdr=mr[0][1:]; rows=mr[1:]
    vals=[]
    for row in rows:
        for j,c in enumerate(row[1:]):
            if row[0]!=hdr[j] and c.strip(): vals.append(float(c))
    print(f'   recomputed matrix_axis1 off-diag mean over {len(vals)} cells = {statistics.fmean(vals):.10f}')
