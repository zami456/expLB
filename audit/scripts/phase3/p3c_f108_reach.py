"""P3-C: how far does the HP-KR / kr-matched row-level leak (F-108) reach into the
REPORTED pooled-training numbers? Recompute the 11-dataset LODO headline with the two
contaminated target cells dropped, and compare against F-108's leave-one-CLUSTER-out fix.
Read-only; inputs are the cached lab JSONs under audit/cache/labpc/run-premise/."""
import json, glob, os, statistics as st
LEAK = {'harm-pressure-knowledge-report', 'harm-pressure-kr-matched'}
print(f"{'layer':5s} {'n':3s} {'single':>8s} {'LODO(11)':>9s} {'gain11':>8s} "
      f"{'LODO(9 clean)':>13s} {'gain9':>8s} {'delta':>8s}")
for f in sorted(glob.glob('audit/cache/labpc/run-premise/premise_L*_mean_s42.json'),
                key=lambda p: int(p.split('_L')[1].split('_')[0])):
    d = json.load(open(f, encoding='utf-8'))
    pd = d['per_dataset']
    if len(pd) != 11:
        continue
    L = d['layer']
    s_all = st.mean(v['single_offdiag'] for v in pd.values())
    p_all = st.mean(v['pooled_lodo'] for v in pd.values())
    keep = {k: v for k, v in pd.items() if k not in LEAK}
    s_cl = st.mean(v['single_offdiag'] for v in keep.values())
    p_cl = st.mean(v['pooled_lodo'] for v in keep.values())
    print(f"{L:5d} {len(pd):3d} {s_all:8.4f} {p_all:9.4f} {p_all-s_all:+8.4f} "
          f"{p_cl:13.4f} {p_cl-s_cl:+8.4f} {(p_cl-s_cl)-(p_all-s_all):+8.4f}")
print("\n--- the two contaminated cells at L40 ---")
d = json.load(open('audit/cache/labpc/run-premise/premise_L40_mean_s42.json', encoding='utf-8'))
for k in sorted(d['per_dataset']):
    v = d['per_dataset'][k]
    tag = '  <-- LEAKED PAIR' if k in LEAK else ''
    print(f"  {k:36s} single {v['single_offdiag']:.4f}  LODO {v['pooled_lodo']:.4f} "
          f"gain {v['pooled_lodo']-v['single_offdiag']:+.4f}{tag}")
print("\n--- F-108's leave-one-CLUSTER-out (the leak-free version), from the clustered store ---")
for f in sorted(glob.glob('audit/cache/labpc/run-premise/clustered/clustered_L*_mean_s42.json'),
                key=lambda p: int(p.split('_L')[1].split('_')[0])):
    c = json.load(open(f, encoding='utf-8'))
    s = c['summary']
    print(f"  L{c['layer']:<3d} " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in s.items() if not isinstance(v, dict)))
