"""P3-C: of the dataset-level tests that clear BH q<0.05 anywhere in the crossgen
significance files, how many are a PERSONA FEATURE BEATING a baseline, and how many are a
control losing? Read-only; inputs audit/cache/labpc/run-crossgen/**/significance.json."""
import json, glob, os, collections

PERSONA = ('axes', 'pc:')
BASE = ('raw', 'datapca', 'random', 'length')
rows = []
for f in sorted(glob.glob('audit/cache/labpc/run-crossgen/**/significance.json', recursive=True)):
    d = json.load(open(f, encoding='utf-8'))
    tests = d['tests'] if isinstance(d, dict) and 'tests' in d else d
    for t in tests:
        q = t.get('dataset_q_bh')
        if q is None or q >= 0.05:
            continue
        rows.append((f.split('run-crossgen')[1].replace(os.sep, '/'), t))

print(f"{len(rows)} tests clear BH q<0.05 across all cached significance files\n")
cat = collections.Counter()
detail = collections.Counter()
for f, t in rows:
    comp = t.get('comparison', '?')
    diff = t.get('diff', 0)
    if ' - ' not in comp:
        cat['unparsed'] += 1
        continue
    a, b = [s.strip() for s in comp.split(' - ', 1)]
    a_p = a.startswith(PERSONA)
    b_p = b.startswith(PERSONA)
    if a_p and not b_p:
        cat['persona WINS over baseline' if diff > 0 else 'persona LOSES to baseline'] += 1
        if diff > 0:
            detail[f"{f} | {comp} | L{t.get('layer')} {t.get('pooling')} {t.get('probe')} "
                   f"blk{t.get('block')} diff={diff:+.4f} q={t.get('dataset_q_bh'):.4f}"] += 1
    elif b_p and not a_p:
        cat['baseline beats persona' if diff < 0 else 'persona(rhs) loses'] += 1
    elif a_p and b_p:
        cat['persona vs persona'] += 1
    else:
        cat[f'control vs control ({comp})'] += 1
for k, v in cat.most_common():
    print(f"  {v:4d}  {k}")
print("\n--- every clearing test in which a PERSONA FEATURE BEATS a baseline ---")
for k in sorted(detail):
    print(' ', k)
