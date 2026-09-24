# M-001: "they mean-pool (`pool_activations.py:84`, delta=0.0000 reproductions)"
# Checks the cited line at the commit the memory note names (f386a72) AND at branch head,
# and finds the "Delta = 0.0000" statements in the collaborator's own summaries.
# git show only; no checkout.
import subprocess

BR = 'origin/feat/persona-pc-ood'
F = 'src/probes/deception-detection/deception_detection/scripts/pool_activations.py'


def show(rev, path):
    return subprocess.run(['git', 'show', f'{rev}:{path}'], capture_output=True,
                          text=True, encoding='utf-8').stdout


print('commit named by the memory note: f386a72')
print(subprocess.run(['git', 'log', '-1', '--format=%H %ad %s', 'f386a72'],
                     capture_output=True, text=True, encoding='utf-8').stdout.strip())
print('branch head:')
print(subprocess.run(['git', 'log', '-1', '--format=%H %ad %s', BR],
                     capture_output=True, text=True, encoding='utf-8').stdout.strip())
anc = subprocess.run(['git', 'merge-base', '--is-ancestor', 'f386a72', BR]).returncode == 0
print('f386a72 is an ancestor of the branch head:', anc)

for rev in ['f386a72', BR]:
    lines = show(rev, F).split('\n')
    print(f'\n--- {F} @ {rev}: {len(lines)} lines')
    print(f'    line 84  : {lines[83].strip()!r}')
    pool = [i + 1 for i, l in enumerate(lines) if '.float().mean(0).squeeze(0).numpy()' in l]
    print(f'    mean-pool line(s): {pool} -> {lines[pool[0]-1].strip()!r}')

print('\n--- "Delta = 0.0000" statements in his own result summaries')
for d in ['persona_ood', 'persona_ood_L40', 'persona_ood_ext_L20', 'persona_ood_ext_L40',
          'persona_ood_ext_L44']:
    p = f'src/probes/deception-detection/results/{d}/summary.md'
    t = show(BR, p)
    hits = [l.strip() for l in t.split('\n') if '0.0000' in l or 'Pooling verified' in l]
    print(f'  {d}: ' + ('; '.join(hits) if hits else 'NO such statement'))
