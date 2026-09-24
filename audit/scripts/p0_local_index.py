"""Phase-0 local results index (read-only).

Walks the working tree (stat only; never opens .pt/.npy/.npz), groups files into
result-bearing folders, and writes audit/phase0/local_results_index.csv.

Per folder: git status of its files (tracked / ignored / untracked), counts by type,
bytes, mtimes, first/last git commit date of tracked content, manifest-like files,
Git-LFS pointer files, files of exactly 204800 bytes (scp-truncation hypothesis),
0-byte files, and cheap structural truncation hints for text results:
  - .json that fails json.loads (files <= 60 MB only)
  - .jsonl whose last non-empty line fails json.loads
  - .csv that does not end in a newline
All of these are STRUCTURAL facts, not verdicts on results.

Also writes audit/phase0/local_files_flagged.csv listing each individual flagged file.

Re-run:  PYTHONIOENCODING=utf-8 python audit/scripts/p0_local_index.py
"""
import csv
import datetime as dt
import json
import os
import subprocess
from collections import defaultdict

ROOT = r"C:\Users\SURAIYA\Documents\STUDY\thesis\expLB"
OUT_CSV = os.path.join(ROOT, "audit", "phase0", "local_results_index.csv")
OUT_FLAG = os.path.join(ROOT, "audit", "phase0", "local_files_flagged.csv")

# top-level dirs excluded entirely
EXC_TOP = {'.git', '.venv', '.venv-aa', '.venv-aa-win', 'mask', 'pacchiardi', 'soft-trigger',
           'assistant_axis', 'behavioral-self-awareness', 'audit', '.claude', '.pytest_cache',
           'dishonesty_bench.egg-info', '.dist'}
# vendored nnsight library docs (not project content)
NNSIGHT_DOCS = {'docs/concepts', 'docs/developing', 'docs/errors', 'docs/gotchas', 'docs/models',
                'docs/patterns', 'docs/reference', 'docs/remote', 'docs/usage'}
NNSIGHT_ROOT_FILES = {'NNsight.md', 'docs/NNSIGHT_UPSTREAM.md'}

RESULT_EXT = {'.json', '.jsonl', '.csv', '.tsv', '.npy', '.npz', '.pt', '.pth', '.safetensors',
              '.log', '.parquet', '.png', '.pdf', '.pkl', '.xlsx', '.txt', '.html', '.svg', '.yaml', '.yml'}
TYPE_COLS = ['json', 'jsonl', 'csv', 'npy', 'npz', 'pt', 'log', 'parquet', 'png', 'pdf']


def max_depth_for(rel):
    """Granularity of index rows (repo-relative depth, 1 = top-level dir)."""
    if rel.startswith('run-crossgen') or rel.startswith('figures/crossgen'):
        return 4
    if rel.startswith('src/probes/deception-detection'):
        return 4
    return 3


def role_for(rel):
    top = rel.split('/')[0]
    if top == 'figures' or rel.startswith('kd/figures'):
        return 'figures'
    if top in ('results', 'logs', 'RESULTAXIS', 'aa-smoke', 'aa-smoke-win') or top.startswith('run-') \
            or rel.startswith('kd/run') or rel.startswith('kd/results') or rel.startswith('cap/run'):
        return 'results'
    if top == 'data' or rel.startswith(('kd/personas', 'kd/questions', 'cap/personas', 'cap/questions',
                                        'src/assistant_axis_ndif/pc_instructions', 'src/gender_secret')):
        return 'inputs'
    if top in ('docs', 'papers', 'full paper draft for this experiments') or rel.endswith('/docs'):
        return 'docs'
    if rel.startswith('src/probes/deception-detection/example_results'):
        return 'results(upstream-example)'
    if rel.startswith('src/probes/deception-detection/data'):
        return 'inputs'
    return 'code/mixed'


MANIFEST_HINTS = ('readme', 'manifest', 'index.md', 'config', 'cfg', 'splits.json', 'summary.json',
                  'significance.json', 'notes', 'md5', 'sha256', 'provenance', 'meta')


def is_manifest(name):
    n = name.lower()
    return any(h in n for h in MANIFEST_HINTS) or n.endswith(('.yaml', '.yml', '.toml', '.md'))


def git(*args):
    return subprocess.run(['git', '-C', ROOT, '-c', 'core.quotepath=off'] + list(args),
                          capture_output=True, check=True).stdout


def iso(ts):
    return dt.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')


def main():
    tracked = set(p for p in git('ls-files', '-z').decode('utf-8').split('\0') if p)

    # per-file git history on HEAD (no rename following: first date = first appearance at this path)
    first, last = {}, {}
    ncommits = defaultdict(int)
    authors = defaultdict(set)
    log = git('log', '--no-renames', '--format=@@%H|%cI|%an', '--name-only', 'HEAD').decode('utf-8', 'replace')
    cur = None
    for line in log.splitlines():
        if line.startswith('@@'):
            h, d, a = line[2:].split('|', 2)
            cur = (h, d[:10], a)
            continue
        if not line.strip() or cur is None:
            continue
        p = line.strip()
        d = cur[1]
        last.setdefault(p, d)          # log is newest-first
        first[p] = d                   # overwritten until oldest
        ncommits[p] += 1
        authors[p].add(cur[2])

    # walk
    files = []   # (rel, size, mtime)
    empty_dirs = []
    for dp, dn, fn in os.walk(ROOT):
        rel_dir = os.path.relpath(dp, ROOT).replace(os.sep, '/')
        if rel_dir == '.':
            dn[:] = [d for d in dn if d not in EXC_TOP]
            rel_dir = ''
        else:
            dn[:] = [d for d in dn if (rel_dir + '/' + d) not in NNSIGHT_DOCS and d != '__pycache__']
            if not dn and not fn:
                empty_dirs.append(rel_dir)
        for f in fn:
            rel = (rel_dir + '/' + f) if rel_dir else f
            if rel in NNSIGHT_ROOT_FILES:
                continue
            st = os.lstat(os.path.join(dp, f))
            files.append((rel, st.st_size, st.st_mtime))

    untracked = [r for r, _, _ in files if r not in tracked]
    ignored = set()
    if untracked:
        out = subprocess.run(['git', '-C', ROOT, 'check-ignore', '-z', '--stdin'],
                             input=('\0'.join(untracked) + '\0').encode('utf-8'),
                             capture_output=True).stdout
        ignored = set(p for p in out.decode('utf-8').split('\0') if p)

    # per-file structural flags
    flagged = []
    finfo = {}
    for rel, size, mtime in files:
        ext = os.path.splitext(rel)[1].lower()
        full = os.path.join(ROOT, rel)
        fl = []
        if size == 204800:
            fl.append('size_204800')
        if size == 0:
            fl.append('empty')
        if size < 200:
            with open(full, 'rb') as fh:
                if fh.read(40).startswith(b'version https://git-lfs'):
                    fl.append('lfs_pointer')
        if ext == '.json' and 0 < size <= 60_000_000:
            try:
                with open(full, 'rb') as fh:
                    json.loads(fh.read().decode('utf-8'))
            except Exception as e:
                fl.append('json_unparseable:' + type(e).__name__)
        elif ext == '.jsonl' and size > 0:
            with open(full, 'rb') as fh:
                fh.seek(max(0, size - 2_000_000))
                tail = fh.read().decode('utf-8', 'replace').rstrip('\n').split('\n')[-1]
            try:
                json.loads(tail)
            except Exception as e:
                fl.append('jsonl_lastline_unparseable:' + type(e).__name__)
        elif ext == '.csv' and size > 0:
            with open(full, 'rb') as fh:
                fh.seek(size - 1)
                if fh.read(1) != b'\n':
                    fl.append('csv_no_trailing_newline')
        st = 'tracked' if rel in tracked else ('ignored' if rel in ignored else 'untracked')
        finfo[rel] = (size, mtime, ext, st, fl)
        if fl:
            flagged.append((rel, size, iso(mtime), st, ';'.join(fl)))

    # choose index rows: every dir up to its area's max depth containing >=1 result-type file (recursive)
    agg = {}

    def unit_dirs(rel):
        parts = rel.split('/')[:-1]
        out = []
        for i in range(1, len(parts) + 1):
            d = '/'.join(parts[:i])
            if i > max_depth_for(d):
                break
            out.append(d)
        return out or ['(repo root files)']

    for rel, (size, mtime, ext, st, fl) in finfo.items():
        for d in unit_dirs(rel):
            a = agg.setdefault(d, dict(n=0, bytes=0, types=defaultdict(int), other=defaultdict(int),
                                       st=defaultdict(int), mt=[], manifests=[], lfs=0, s204800=0,
                                       empty=0, json_bad=0, jsonl_bad=0, csv_nonl=0, gfirst=[], glast=[],
                                       commits=0, authors=set(), result_files=0))
            a['n'] += 1
            a['bytes'] += size
            e = ext.lstrip('.')
            if e in TYPE_COLS:
                a['types'][e] += 1
            else:
                a['other'][e or '(none)'] += 1
            if ext in RESULT_EXT:
                a['result_files'] += 1
            a['st'][st] += 1
            a['mt'].append(mtime)
            base = rel.rsplit('/', 1)[-1]
            if is_manifest(base) and rel.count('/') - d.count('/') <= 2:
                a['manifests'].append(rel[len(d) + 1:] if d != '(repo root files)' else rel)
            for f in fl:
                if f == 'lfs_pointer':
                    a['lfs'] += 1
                elif f == 'size_204800':
                    a['s204800'] += 1
                elif f == 'empty':
                    a['empty'] += 1
                elif f.startswith('json_'):
                    a['json_bad'] += 1
                elif f.startswith('jsonl_'):
                    a['jsonl_bad'] += 1
                elif f.startswith('csv_'):
                    a['csv_nonl'] += 1
            if rel in first:
                a['gfirst'].append(first[rel])
                a['glast'].append(last[rel])
                a['authors'] |= authors[rel]

    # distinct commits per folder
    commit_sets = defaultdict(set)
    cur = None
    for line in log.splitlines():
        if line.startswith('@@'):
            cur = line[2:].split('|', 1)[0]
            continue
        p = line.strip()
        if not p or p not in finfo:
            continue
        for d in unit_dirs(p):
            commit_sets[d].add(cur)

    listed = sorted(d for d, a in agg.items() if a['result_files'] > 0)
    for d in empty_dirs:
        if d.count('/') + 1 <= max_depth_for(d):
            listed.append(d)
    listed = sorted(set(listed))

    cols = ['folder', 'depth', 'n_child_rows', 'role_INFERRED', 'git_status', 'n_tracked', 'n_ignored',
            'n_untracked', 'n_files', 'bytes', 'MB'] + ['n_' + t for t in TYPE_COLS] + \
           ['n_other', 'other_exts', 'mtime_oldest', 'mtime_newest', 'git_first_commit', 'git_last_commit',
            'git_n_commits', 'git_authors', 'n_lfs_pointers', 'n_size_204800', 'n_empty_files',
            'n_json_unparseable', 'n_jsonl_lastline_unparseable', 'n_csv_no_trailing_newline',
            'manifest_like_files']
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for d in listed:
            a = agg.get(d)
            nchild = sum(1 for x in listed if x != d and x.startswith(d + '/') and x.count('/') == d.count('/') + 1)
            if a is None:  # empty dir
                w.writerow([d, d.count('/') + 1, 0, role_for(d), 'EMPTY-DIR (git cannot track empty dirs)', 0, 0, 0, 0, 0, 0]
                           + [0] * len(TYPE_COLS) + [0, '', '', '', '', '', 0, '', 0, 0, 0, 0, 0, 0, ''])
                continue
            stc = a['st']
            if len([k for k in stc if stc[k]]) == 1:
                gs = next(k for k in stc if stc[k])
            else:
                gs = 'mixed'
            ms = a['manifests']
            man = ';'.join(ms[:12]) + (f';...(+{len(ms) - 12})' if len(ms) > 12 else '')
            other = ';'.join(f'{k}:{v}' for k, v in sorted(a['other'].items(), key=lambda kv: -kv[1]))
            w.writerow([d, 0 if d == '(repo root files)' else d.count('/') + 1, nchild, role_for(d), gs,
                        stc['tracked'], stc['ignored'], stc['untracked'], a['n'], a['bytes'],
                        round(a['bytes'] / 1e6, 2)] + [a['types'][t] for t in TYPE_COLS] +
                       [sum(a['other'].values()), other, iso(min(a['mt'])), iso(max(a['mt'])),
                        min(a['gfirst']) if a['gfirst'] else '', max(a['glast']) if a['glast'] else '',
                        len(commit_sets.get(d, ())), ';'.join(sorted(a['authors'])), a['lfs'], a['s204800'],
                        a['empty'], a['json_bad'], a['jsonl_bad'], a['csv_nonl'], man])

    with open(OUT_FLAG, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['path', 'bytes', 'mtime', 'git_status', 'flags'])
        for r in sorted(flagged):
            w.writerow(r)

    # console summary
    tot = len(files)
    totb = sum(s for _, s, _ in files)
    print(f'files walked: {tot}  bytes: {totb} ({totb / 1e9:.3f} GB)')
    from collections import Counter
    print('git status:', Counter(v[3] for v in finfo.values()))
    print('by ext (top 25):', Counter(v[2] for v in finfo.values()).most_common(25))
    fc = Counter()
    for _, _, _, _, f in flagged:
        for x in f.split(';'):
            fc[x.split(':')[0]] += 1
    print('flags:', dict(fc))
    print('rows written:', len(listed), '->', OUT_CSV)
    print('empty dirs (any depth):', empty_dirs)


if __name__ == '__main__':
    main()
