"""P3-C: (a) how favourable are the reported layers 20/40/44 vs the other stored layers,
(b) how large is the seed spread where >1 seed exists. Read-only."""
import csv, collections, statistics as st
ROWS=list(csv.DictReader(open('audit/phase2/layer_sensitivity.csv',encoding='utf-8-sig')))
PERSONA={'axes','axes_resid','pc:deception','pc:unified','pc:unified@2','pc:knowledge_denial',
         'pc:kd_assertion','pc:capitulation','pc:capitulation@2','pc:cap_only','pc:cap_only@2',
         'pc:cap_vs_default','pc:cap_vs_default@2','pc:cap_vs_neutral','pc:cap_vs_neutral@2',
         'pc:cap_vs_para','pc:cap_vs_para@2','pc:cap_only_ctrl','pc:cap_only_ctrl@2'}
CTRL={'random','random@2','datapca','datapca@2','length'}

print("=== (a) rank of the reported layer, gap = feature - raw, seed 42, mean, paper probe ===")
by=collections.defaultdict(dict)
for r in ROWS:
    if r['seed']=='42' and r['pooling']=='mean' and r['probe']=='paper':
        by[(r['store'],r['dataset_set'],r['feature'])][int(r['layer'])]=float(r['offdiag_mean'])
tally=collections.defaultdict(list)
for (s,ds,f),d in by.items():
    raw=by.get((s,ds,'raw'),{})
    common=sorted(set(d)&set(raw))
    if len(common)<7 or f in ('raw',): continue
    gap={L:d[L]-raw[L] for L in common}
    order=sorted(common,key=lambda L:-gap[L])
    grp='persona' if f in PERSONA else ('control' if f in CTRL else 'other')
    for rep in (20,40,44):
        if rep in gap:
            tally[(grp,rep)].append((order.index(rep)+1)/len(common))
for (grp,rep),v in sorted(tally.items()):
    print(f"  {grp:8s} L{rep:<3d} n={len(v):3d}  mean percentile (1.0=worst layer) = {st.mean(v):.3f}  "
          f"median {st.median(v):.3f}  top-third share {sum(1 for x in v if x<=1/3)/len(v):.2f}")

print("\n=== (b) seed spread: same store/ds/feature/probe/pooling/layer, seeds present ===")
sd=collections.defaultdict(dict)
for r in ROWS:
    sd[(r['store'],r['dataset_set'],r['feature'],r['probe'],r['pooling'],r['layer'])][r['seed']]=float(r['offdiag_mean'])
spreads=collections.defaultdict(list)
for k,v in sd.items():
    if len(v)>=5:
        spreads[k[2]].append(max(v.values())-min(v.values()))
for f,v in sorted(spreads.items(), key=lambda x:-st.mean(x[1]))[:14]:
    print(f"  {f:20s} cells={len(v):4d}  mean range across seeds = {st.mean(v):.4f}  "
          f"max {max(v):.4f}  p90 {sorted(v)[int(.9*len(v))]:.4f}")
