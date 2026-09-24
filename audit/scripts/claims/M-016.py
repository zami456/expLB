# M-016: "I expected 7b, 'persona vectors are judge-dependent', from the ~7-9% own100-vs-hf275
#         gap. That comparison confounded judge with generation run."
# (a) where the ~7-9% comes from, and what it is;  (b) that own100 and hf275 are different rollouts.
import re, subprocess, torch, glob, os, json

PLAN = os.path.expanduser('~/.claude/plans/harmonic-cuddling-kay.md')
print('--- (a) provenance of "~7-9%"')
for i, l in enumerate(open(PLAN, encoding='utf-8').read().split('\n'), 1):
    if '7–9' in l or '7-9' in l:
        print(f'  {PLAN}:{i}: {l.strip()[:190]}')

print('\n  The plan says "already partly measured"; no result file is named.')
print('  The only own100-vs-hf275 quantity in the repo is the Q1 contamination table')
print('  (docs/thesis/persona-coordinates-70b-results.md:523-529). Relative gap:')
q1 = {20: (0.729, 0.686), 22: (0.710, 0.662), 40: (0.660, 0.610), 60: (0.573, 0.516)}
for L, (o, h) in q1.items():
    print(f'    L{L}: |own100| {o:.3f} vs |hf275| {h:.3f}  abs {o-h:+.3f}  '
          f'rel-to-own100 {100*(o-h)/o:.1f}%  rel-to-hf275 {100*(o-h)/h:.1f}%')

print('\n--- (b) are own100 and hf275 different rollouts?')
print('  hf275 vectors are DOWNLOADED, not generated here:')
src = open('src/assistant_axis_ndif/31_fetch_hf_vectors.py', encoding='utf-8').read()
for l in src.split('\n'):
    if 'REPO =' in l or 'snapshot_download' in l or 'MODEL_DIR' in l:
        print('    31_fetch_hf_vectors.py: ' + l.strip())
print('  local HF cache present:',
      bool(glob.glob(os.path.expanduser(
          '~/.cache/huggingface/hub/datasets--lu-christina--assistant-axis-vectors/snapshots/*/llama-3.3-70b/assistant_axis.pt'))))
doc = open('docs/thesis/persona-coordinates-70b-results.md', encoding='utf-8').read().split('\n')
for i, l in enumerate(doc, 1):
    if l.startswith('| `own100`') or l.startswith('| `hf275`'):
        print(f'    persona-coordinates-70b-results.md:{i}: {l.strip()[:200]}')
