# M-019 / M-020 / M-021: construct BLOCKS, run-crossgen/summary.json, L40 mean paper probe, 10 datasets
import json
d=json.load(open('run-crossgen/summary.json',encoding='utf-8'))
print('design:',d['design']); print('datasets:',d['datasets']); print('main_seed',d['main_seed'],'n_records',len(d['records']))
recs=[r for r in d['records'] if r.get('layer')==40 and r.get('pooling')=='mean' and r.get('probe')=='paper'
      and r.get('seed')==42 and 'blocks' in r]
print('matching records:',len(recs),'features:',[r['feature'] for r in recs])
blocks=list(recs[0]['blocks'].keys())
print('\nblocks:',blocks)
feats=[r['feature'] for r in recs]
hdr='block'.ljust(26)+''.join(f'{f:>14s}' for f in feats)
print('\n'+hdr)
for b in blocks:
    n=recs[0]['blocks'][b]['n_cells']
    line=f'{b+" (n="+str(n)+")":<26s}'
    for r in recs:
        v=r['blocks'].get(b)
        line+=f'{v["mean_auroc"]:>14.3f}' if v else f'{"-":>14s}'
    print(line)
