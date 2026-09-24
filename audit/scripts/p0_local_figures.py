"""Phase-0: where do the figures in figures/thesis, figures/persona-coords and figures/crossgen
come from, and is the data behind each one present in this working tree? (read-only)

Writes audit/phase0/local_figures_provenance.csv, one row per figure (PNG/PDF pair collapsed to
one row keyed by the stem), with: producing script + function (from reading the scripts),
declared input files, whether each exists locally, git last-commit dates of figure and inputs,
and notes (e.g. numbers hard-coded in the script).

Mapping rules (from reading the scripts during this audit):
  figures/thesis/*        kd/scripts/make_thesis_figures.py
                          fig1/fig3/fig4: numbers HARD-CODED in the script (no input file)
                          fig2: --data kd/results/fig_data.json (itself exported by
                                kd/scripts/export_fig_data.py from the 'source' it records)
                          fig5: --cap run-cap/export/cap_fig_data.json (exported by
                                cap/scripts/export_cap_fig.py)
  figures/persona-coords  src/assistant_axis_ndif/19_figures.py (via plot_style.save); inputs
                          are the run-coords/<name> JSONs passed to load() in each function
  figures/crossgen/...    src/assistant_axis_ndif/28_crossgen_figures.py; run root is
                          run-crossgen (+ /kd11, + /kd11/passA, INFERRED from the figure path);
                          figure6_style & bank_comparison read matrices/<pool>_L<L>_s42.json
                          (bank files merge baselines from the root file); matched reads
                          significance.json; blocks/per_source/per_target read summary.json.

Re-run:  PYTHONIOENCODING=utf-8 python audit/scripts/p0_local_figures.py
"""
import ast
import csv
import glob
import json
import os
import re
import subprocess
from collections import defaultdict

ROOT = r"C:\Users\SURAIYA\Documents\STUDY\thesis\expLB"
OUT = os.path.join(ROOT, 'audit', 'phase0', 'local_figures_provenance.csv')
BANKS = sorted(['kd', 'kd_s23', 'hf100', 'own100', 'paper16_famcentred', 'paper16_shared', 'hf275',
                'paper16', 'cap_ctrl', 'cap_s23', 'cap_all'], key=len, reverse=True)


def git(*a):
    return subprocess.run(['git', '-C', ROOT, '-c', 'core.quotepath=off'] + list(a), capture_output=True,
                          check=True).stdout.decode('utf-8', 'replace')


last = {}
cur = None
for line in git('log', '--no-renames', '--format=@@%cI', '--name-only', 'HEAD').splitlines():
    if line.startswith('@@'):
        cur = line[2:12]
    elif line.strip():
        last.setdefault(line.strip(), cur)


def ex(rel):
    return os.path.exists(os.path.join(ROOT, *rel.split('/')))


def glast(rel):
    if rel in last:
        return last[rel]
    # a directory: newest commit of anything beneath it
    ds = [d for p, d in last.items() if p.startswith(rel.rstrip('/') + '/')]
    return max(ds) if ds else ''


rows = []


def add(fig_stem, exts, script, func, inputs, note=''):
    ins = []
    for i in inputs:
        ins.append(f"{i} [{'Y' if ex(i) else 'N'}; git {glast(i) or '-'}]")
    figfile = fig_stem + '.' + ('png' if 'png' in exts else exts[0])
    rows.append([fig_stem, '+'.join(sorted(exts)), glast(figfile), script, func, ' | '.join(ins),
                 'Y' if inputs and all(ex(i) for i in inputs) else ('n/a' if not inputs else 'N'), note])


def stems(folder):
    out = defaultdict(list)
    for p in glob.glob(os.path.join(ROOT, *folder.split('/'), '**', '*'), recursive=True):
        if p.lower().endswith(('.png', '.pdf')):
            rel = os.path.relpath(p, ROOT).replace(os.sep, '/')
            s, e = rel.rsplit('.', 1)
            out[s].append(e)
    return out


# ---------------------------------------------------------------- figures/thesis
fd = 'kd/results/fig_data.json'
fd_note = ''
if ex(fd):
    d = json.load(open(os.path.join(ROOT, *fd.split('/')), encoding='utf-8'))
    src = d.get('source', '')
    lay = sorted({r['layer'] for v in d['profile'].values() for r in v})
    fd_note = (f"fig_data.json records source={src}, {len(lay)} layers, census n_tests={d['census']['n_tests']}; ")
    if src and ex(src):
        s = json.load(open(os.path.join(ROOT, *src.split('/')), encoding='utf-8'))
        sl = sorted({t['layer'] for t in s['tests']})
        fd_note += f"LOCAL {src} has {len(s['tests'])} tests over {len(sl)} layers {sl}"
        if len(s['tests']) != d['census']['n_tests']:
            fd_note += ' -> CONFLICT: local source file is not the version fig_data.json was exported from'
    elif src:
        fd_note += 'source NOT present locally'
thesis_map = {
    'fig1_register': ([], 'fig1_register', 'HARD-CODED values in script (axis/random/raw AUROCs for HP-KR & sycophancy_mt); docstring cites F-13'),
    'fig2_layer_profile': ([fd, 'kd/scripts/export_fig_data.py'], 'fig2_layer_profile', fd_note),
    'fig3_baseline_artifact': ([], 'fig3_baseline', 'HARD-CODED values in script (3B/8B/70B raw & deltas); docstring cites F-63, supersedes F-58'),
    'fig4_geometry_by_scale': ([], 'fig4_geometry', 'HARD-CODED values in script (cos PC1 vs assistant/deception/sycophancy by scale); docstring cites F-37'),
    'fig5_capitulation': (['run-cap/export/cap_fig_data.json', 'cap/scripts/export_cap_fig.py'], 'fig5_capitulation',
                          'cap_fig_data.json is exported on the lab PC from run-crossgen/kd11/banks/cap_*/matrices + run-cap/axes_s23.pt (banks cap_* NOT in local run-crossgen/kd11/banks)'),
}
for stem, exts in sorted(stems('figures/thesis').items()):
    name = stem.rsplit('/', 1)[-1]
    ins, fn, note = thesis_map.get(name, ([], '?', 'no mapping found'))
    add(stem, exts, 'kd/scripts/make_thesis_figures.py', fn, ins, note)

# ---------------------------------------------------------------- figures/persona-coords
src19 = open(os.path.join(ROOT, 'src', 'assistant_axis_ndif', '19_figures.py'), encoding='utf-8').read()
tree = ast.parse(src19)
fig2inputs = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        loads, saves = [], []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fname = getattr(sub.func, 'id', None) or getattr(sub.func, 'attr', None)
                if fname == 'load' and sub.args:
                    a = sub.args[0]
                    if isinstance(a, ast.Constant):
                        loads.append(a.value)
                    elif isinstance(a, ast.JoinedStr):
                        loads.append(''.join(v.value if isinstance(v, ast.Constant) else '*' for v in a.values))
                if fname == 'save' and len(sub.args) >= 2 and isinstance(sub.args[1], ast.Constant):
                    saves.append(sub.args[1].value)
                if fname == 'glob' and sub.args and isinstance(sub.args[0], ast.Constant):
                    loads.append(sub.args[0].value)
        for s in saves:
            fig2inputs[s.rsplit('.', 1)[0]] = (node.name, loads)
for stem, exts in sorted(stems('figures/persona-coords').items()):
    name = stem.rsplit('/', 1)[-1]
    fn, loads = fig2inputs.get(name, ('?', []))
    ins = []
    for l in loads:
        if '*' in l:
            hits = sorted(os.path.relpath(p, ROOT).replace(os.sep, '/') for p in glob.glob(os.path.join(ROOT, 'run-coords', *l.split('/'))))
            ins.extend(hits or ['run-coords/' + l])
        else:
            ins.append('run-coords/' + l)
    add(stem, exts, 'src/assistant_axis_ndif/19_figures.py (+plot_style.py)', fn, ins,
        '' if loads else 'no load() call found in the saving function')

# ---------------------------------------------------------------- figures/crossgen
for stem, exts in sorted(stems('figures/crossgen').items()):
    rel = stem[len('figures/crossgen/'):]
    parts = rel.split('/')
    run = 'run-crossgen'
    if parts[0] == 'kd11':
        run += '/kd11'
        parts = parts[1:]
        if parts and parts[0] == 'passA':
            run += '/passA'
            parts = parts[1:]
    kind = parts[0]
    name = parts[-1]
    bank = None
    if kind == 'figure6_style' and len(parts) >= 4 and parts[1] == 'banks':
        bank = parts[2]
    elif kind in ('matched', 'blocks', 'per_source', 'per_target'):
        for b in BANKS:
            if name.startswith(b + '_'):
                bank = b
                name = name[len(b) + 1:]
                break
    m = re.search(r'L(\d+)_(mean|last)_', name) or re.search(r'L(\d+)_(mean|last)', name)
    L, pool = (m.group(1), m.group(2)) if m else (None, None)
    bdir = run if bank in (None, 'paper16') else f'{run}/banks/{bank}'
    ins, fn = [], ''
    if kind == 'figure6_style':
        fn = 'fig6'
        ins = [f'{bdir}/matrices/{pool}_L{L}_s42.json']
        if bdir != run:
            ins.append(f'{run}/matrices/{pool}_L{L}_s42.json')
    elif kind == 'bank_comparison':
        fn = 'bank_comparison'
        ins = [f'{run}/matrices/{pool}_L{L}_s42.json']
        ins += sorted(f'{run}/banks/{b}/matrices/{pool}_L{L}_s42.json'
                      for b in os.listdir(os.path.join(ROOT, *run.split('/'), 'banks'))) \
            if os.path.isdir(os.path.join(ROOT, *run.split('/'), 'banks')) else []
    elif kind == 'matched':
        fn = 'fig_matched'
        ins = [f'{bdir}/significance.json']
    elif kind in ('blocks', 'per_source', 'per_target'):
        fn = {'blocks': 'fig_blocks', 'per_source': 'fig_per_dataset', 'per_target': 'fig_per_dataset'}[kind]
        ins = [f'{bdir}/summary.json'] + ([f'{run}/summary.json'] if bdir != run else [])
    add(stem, exts, 'src/assistant_axis_ndif/28_crossgen_figures.py', fn, ins,
        f'run root INFERRED from figure path: {run}; bank={bank or "(root)"}')

with open(OUT, 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['figure_stem', 'formats', 'figure_git_last_commit', 'producing_script', 'function',
                'inputs [exists_locally; git last commit]', 'all_inputs_local', 'notes'])
    w.writerows(rows)

from collections import Counter
print('figures (stems):', len(rows))
for area in ('figures/thesis', 'figures/persona-coords', 'figures/crossgen'):
    sub = [r for r in rows if r[0].startswith(area)]
    print(area, len(sub), 'all_inputs_local:', Counter(r[6] for r in sub))
for r in rows:
    if r[6] != 'Y' or 'CONFLICT' in r[7]:
        print(' ', r[0], '|', r[4], '|', r[5][:200], '|', r[7][:260])
