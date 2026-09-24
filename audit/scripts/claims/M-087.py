"""M-087: memory note significance-and-random-nulls.md:17-19
"Row bootstrap (resample test rows, paired): nearly every Q2/Q6 difference is significant.
 axes vs raw same layer 40 mean +0.052 [+0.040,+0.062]; L44 mean +0.068; L44 last +0.046.
 Best-vs-best: mean +0.014 [-0.000,+0.028] p=0.056; last +0.026 p=0.002."
Source named in the note: 23_significance.py -> run-coords/significance.json.
Read-only.
"""
import json, sys

p = sys.argv[1] if len(sys.argv) > 1 else 'run-coords/significance.json'
d = json.load(open(p))
print(f'### {p}')
print(f'    boot_rows={d["boot_rows"]} boot_datasets={d["boot_datasets"]} margin={d["margin"]}')
print(f'    datasets ({len(d["datasets"])}): {d["datasets"]}')
print(f'    reproduction_worst_diff = {d["reproduction_worst_diff"]}')
print(f'    comparisons: {len(d["comparisons"])}, per_method: {len(d["per_method"])}\n')
for c in d['comparisons']:
    name = c.get('name') or c.get('label') or c.get('comparison') or str(list(c)[:3])
    print('-', json.dumps({k: v for k, v in c.items()
                           if not isinstance(v, (list, dict))}, default=str))
    for k, v in c.items():
        if isinstance(v, (list, dict)) and k not in ('per_dataset',):
            print(f'    {k} = {json.dumps(v)[:300]}')
    print()
