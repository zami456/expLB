"""W4b CONFLICT 2 detail: exact feature/probe/layer sets per (store,seed), and which
(store, feature, probe, layer, pooling, k, dataset_set) cells actually have >1 seed."""
import csv, collections, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
rows = list(csv.DictReader(open(ROOT/'audit/phase2/layer_sensitivity.csv', newline='', encoding='utf-8-sig')))

print("=== A. exact feature sets per (store, seed-group) ===")
feat = collections.defaultdict(set); lays = collections.defaultdict(set)
for r in rows:
    feat[(r['store'], r['seed'])].add(r['feature']); lays[(r['store'], r['seed'])].add(int(r['layer']))
for s in ['<root>','passA','passC','banks/hf100','banks/own100','banks/paper16_famcentred',
          'banks/paper16_shared','kd11/banks/cap_s23']:
    seeds = sorted({sd for (st,sd) in feat if st==s}, key=lambda x:int(x))
    for sd in seeds:
        f = sorted(feat[(s,sd)]); L = sorted(lays[(s,sd)])
        print(f"{s:<26} seed={sd:<3} nfeat={len(f):<3} layers({len(L)})={L if len(L)<=8 else str(L[:4])+'...'+str(L[-2:])}")
        print(f"{'':<26}   features={f}")
    print()

print("=== B. multi-seed cells: how many seeds per (store,k,pooling,layer,dataset_set,feature,probe) ===")
cell = collections.defaultdict(set)
for r in rows:
    cell[(r['store'], r['k'], r['pooling'], r['layer'], r['dataset_set'], r['feature'], r['probe'])].add(r['seed'])
c = collections.Counter(len(v) for v in cell.values())
print("n_seeds -> n_cells:", dict(sorted(c.items())))
print()
per_store = collections.defaultdict(collections.Counter)
for k, v in cell.items():
    per_store[k[0]][len(v)] += 1
for s in sorted(per_store):
    print(f"{s:<26} {dict(sorted(per_store[s].items()))}")

print("\n=== C. which (store,feature,probe) combos EVER have 6 seeds ===")
six = collections.defaultdict(set)
for k, v in cell.items():
    if len(v) >= 6:
        six[k[0]].add((k[5], k[6]))
for s in sorted(six):
    print(f"{s:<26} {sorted(six[s])}")
print("\n=== D. cap_s23 multi-seed layers ===")
cs = collections.defaultdict(set)
for r in rows:
    if r['store'] == 'kd11/banks/cap_s23':
        cs[r['seed']].add(r['layer'])
for sd in sorted(cs, key=lambda x:int(x)):
    print(f"  seed={sd}: layers={sorted(cs[sd], key=int)}")
