# M-026: "F-95 | pooled probe peaks at L40 | peak is L36 (0.8916 vs 0.8872)"
# Source: the 63 falsify_L*_s*.json files (21 layers x 3 seeds), key 'real'.
import json, glob, os, re, statistics, hashlib
D = 'audit/cache/labpc/run-premise/falsify'
vals = {}
for p in sorted(glob.glob(os.path.join(D, 'falsify_L*_s*.json'))):
    base = os.path.basename(p)
    m = re.search(r'falsify_L(\d+)_s(\d+)\.json$', base)
    L, s = int(m.group(1)), int(m.group(2))
    d = json.load(open(p, encoding='utf-8'))
    v = d.get('real')
    if isinstance(v, dict):
        v = v.get('auroc') or v.get('mean')
    vals.setdefault(L, {})[s] = float(v)
print(f'{len(vals)} layers x {len(next(iter(vals.values())))} seeds from {D}')
print(f'{"layer":>6}{"s42":>10}{"s7":>10}{"s13":>10}{"seed-mean":>12}')
rows = []
for L in sorted(vals):
    r = vals[L]
    mm = statistics.fmean(r.values())
    rows.append((L, mm))
    print(f'{L:>6}{r.get(42, float("nan")):>10.4f}{r.get(7, float("nan")):>10.4f}'
          f'{r.get(13, float("nan")):>10.4f}{mm:>12.4f}')
top = sorted(rows, key=lambda t: -t[1])[:5]
print('\ntop 5 layers by seed-mean:', [(L, round(v, 4)) for L, v in top])
d = {L: v for L, v in rows}
print(f'L36 = {d[36]:.4f}   L40 = {d[40]:.4f}   L36-L40 = {d[36]-d[40]:+.4f}')
print('\nmd5 / size (RF: falsify_L40_s13.json was rewritten 2026-09-23 by the audit incident):')
for f in ['falsify_L40_s13.json', 'falsify_L40_s42.json', 'falsify_L36_s42.json']:
    p = os.path.join(D, f)
    print(' ', f, hashlib.md5(open(p, 'rb').read()).hexdigest(), os.path.getsize(p), 'B',
          __import__('datetime').datetime.fromtimestamp(os.path.getmtime(p)))
