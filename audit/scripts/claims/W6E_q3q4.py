# W6E-memory-q : Q3 (M-058) and Q4 (M-059..M-063) claims
# Source notes: memory/q3-advantage-grows-with-n.md, memory/q4-signal-survives-assistant-erasure.md
import json, os
ROOT = r'C:/Users/SURAIYA/Documents/STUDY/thesis/expLB'
p = os.path.join(ROOT, 'run-coords/q3_q4_bootstrap.json')
d = json.load(open(p))
print('FILE', p)
print('q3_reproduction_worst =', d['q3_reproduction_worst'])
print('boot_rows =', d['boot_rows'], ' boot_datasets =', d['boot_datasets'])
print()
print('=== Q3 curves (auroc by train n) ===')
for k, v in d['q3_curves'].items():
    print(f'-- {k}')
    for rec in v:
        print(f'   n={rec["n"]:>5}  auroc={rec["auroc"]:.4f}  row_ci95=[{rec["row_ci95"][0]:.4f},{rec["row_ci95"][1]:.4f}]  dataset_ci95=[{rec["dataset_ci95"][0]:.4f},{rec["dataset_ci95"][1]:.4f}]')
    a = {r['n']: r['auroc'] for r in v}
    if 1024 in a and 16 in a:
        print(f'   n1024 - n16 = {a[1024]-a[16]:+.4f}')
print()
print('=== Q4 table ===')
for r in d['q4_table']:
    print('ROW:', json.dumps({k: v for k, v in r.items()}))
    print(f'      erased_row_ci95=[{r["erased_row_ci95"][0]:.4f},{r["erased_row_ci95"][1]:.4f}] '
          f'erased_dataset_ci95=[{r["erased_dataset_ci95"][0]:.4f},{r["erased_dataset_ci95"][1]:.4f}] '
          f'erased_above_chance_datasets={r["erased_above_chance_datasets"]}')
print()
print('=== comparisons ===')
for c in d['comparisons']:
    print(f'[{c["question"]}] {c["name"]}: diff={c["diff"]:+.4f} row_ci95=[{c["row_ci95"][0]:+.4f},{c["row_ci95"][1]:+.4f}] '
          f'dataset_ci95=[{c["dataset_ci95"][0]:+.4f},{c["dataset_ci95"][1]:+.4f}] row_p={c["row_p"]:.4f} dataset_p={c["dataset_p"]:.4f} '
          f'q_bh={c["dataset_q_bh"]:.4f} sig={c["significant"]}')
