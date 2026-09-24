"""Phase-0 rough map: script -> outputs it writes / inputs it reads (read-only, HEURISTIC).

Scans every script in src/assistant_axis_ndif/ (incl. tests/), kd/scripts/, cap/scripts/,
scale/scripts/, scripts/, src/visualization/, top-level *.py/*.ps1/*.sh, and the ignored
aa-smoke-win/*.py, and writes audit/phase0/scripts_to_outputs.csv.

Everything in the path columns is INFERRED by pattern matching:
  * Python path constants are resolved when built from Path(__file__).parent(s) / "a" / "b",
    or from other resolved constants (one assignment per line, top-level-ish).
  * PowerShell: $PSScriptRoot, Split-Path ... -Parent, Join-Path $var "lit", and quoted
    repo-relative literals (drivers Set-Location to the repo root).
  * A path is an OUTPUT if it (or the constant holding it) appears on a line that writes
    (open(...,'w'/'a'), json.dump, to_csv, torch.save, np.save*, savefig, write_text, Out-File,
    Set-Content, Add-Content, Tee-Object, >>, --out*/--output*/--figdir argument), an INPUT if on
    a line that reads (open(...) without w, json.load, read_csv, torch.load, np.load, read_text,
    Get-Content, Import-Csv, --run_dir/--run/--data_dir/--in*), else it lands in 'other'.
  * exists_locally is checked against this working tree only.
Git columns come from `git log --no-renames` on HEAD (history at THIS path only).

Re-run:  PYTHONIOENCODING=utf-8 python audit/scripts/p0_local_scripts.py
"""
import ast
import csv
import os
import posixpath
import re
import subprocess
from collections import defaultdict, Counter

ROOT = r"C:\Users\SURAIYA\Documents\STUDY\thesis\expLB"
OUT_CSV = os.path.join(ROOT, 'audit', 'phase0', 'scripts_to_outputs.csv')

SCRIPT_DIRS = ['src/assistant_axis_ndif', 'src/assistant_axis_ndif/tests', 'kd/scripts', 'cap/scripts',
               'scale/scripts', 'scripts', 'src/visualization', 'aa-smoke-win', '']
EXTS = ('.py', '.ps1', '.sh')
KNOWN_ROOTS = ('run-', 'kd', 'cap', 'scale', 'figures', 'results', 'logs', 'data', 'data_syco', 'docs',
               'aa-smoke', 'src', 'RESULTAXIS', 'thinking', 'assistant_axis', 'papers', 'scripts')

W_PY = re.compile(r"""open\([^)]*,\s*['"][wa]b?\+?['"]|\.write_text\(|\.write_bytes\(|json\.dump\(|\.to_csv\(|"""
                  r"""\.to_json\(|\.to_parquet\(|torch\.save\(|np\.save|savefig\(|\.to_excel\(|\.mkdir\(|"""
                  r"""shutil\.copy|os\.replace\(|\.rename\(|pickle\.dump\(""")
R_PY = re.compile(r"""open\((?![^)]*['"][wa]b?\+?['"])|json\.load|read_csv\(|torch\.load\(|np\.load\(|"""
                  r"""\.read_text\(|\.read_bytes\(|pd\.read_|load_dataset\(|\.glob\(|\.iterdir\(|\.exists\(\)""")
W_PS = re.compile(r"Out-File|Set-Content|Add-Content|Tee-Object|>>|\s>\s|Export-Csv|New-Item\s+-ItemType\s+Directory|"
                  r"Copy-Item|Move-Item|rclone\s+(copy|sync)", re.I)
R_PS = re.compile(r"Get-Content|Import-Csv|Test-Path|Get-ChildItem", re.I)
OUT_ARG = re.compile(r"--(out\w*|output\w*|figdir|fig\w*|dest\w*|save\w*|log\w*)\b", re.I)
IN_ARG = re.compile(r"--(run_dir|run|data_dir|in\w*|store\w*|vectors?\w*|axes\w*|acts\w*|src\w*|source\w*|bank\w*)\b", re.I)


def git(*args):
    return subprocess.run(['git', '-C', ROOT, '-c', 'core.quotepath=off'] + list(args),
                          capture_output=True, check=True).stdout.decode('utf-8', 'replace')


def norm(p):
    p = p.replace('\\', '/')
    p = re.sub(r'/+', '/', p)
    p = posixpath.normpath(p) if p else p
    return p


def exists_local(rel):
    if rel is None:
        return ''
    rel = rel.split('{')[0].split('*')[0].split('$')[0].rstrip('/')
    if not rel or rel == '.':
        return 'Y'
    return 'Y' if os.path.exists(os.path.join(ROOT, *rel.split('/'))) else 'N'


def looks_rootrel(s):
    s2 = s.replace('\\', '/').lstrip('./')
    first = s2.split('/')[0]
    return any(first == r or (r.endswith('-') and first.startswith(r)) or first.startswith(r + '-') and r in ('aa-smoke',)
               for r in KNOWN_ROOTS) or first.startswith('run-')


# ---------------------------------------------------------------- python
PY_ASSIGN = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*(#.*)?$')
STRLIT = re.compile(r'''(?<![A-Za-z0-9_])[rbfRBF]?(["'])((?:(?!\1).){1,200})\1''')


def resolve_py(expr, env, scriptdir):
    """Resolve a Path expression to a repo-relative path (or None)."""
    e = expr.strip()
    e = re.sub(r'^str\((.*)\)$', r'\1', e)
    e = e.replace('.resolve()', '').replace('.absolute()', '')
    m = re.match(r'^Path\(__file__\)((?:\.parents\[\d+\]|\.parent(?!s))*)(.*)$', e)
    base = None
    rest = ''
    if m:
        base = scriptdir + '/__file__'
        chain, rest = m.group(1), m.group(2)
        for part in re.findall(r'\.parent(?!s)|\.parents\[(\d+)\]', chain):
            # findall returns '' for .parent, digit for parents[k]
            k = int(part) + 1 if part != '' else 1
            for _ in range(k):
                base = posixpath.dirname(base)
    else:
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)((?:\.parents\[\d+\]|\.parent(?!s))*)(.*)$', e)
        if m and m.group(1) in env and env[m.group(1)] is not None:
            base = env[m.group(1)]
            for part in re.findall(r'\.parent(?!s)|\.parents\[(\d+)\]', m.group(2)):
                k = int(part) + 1 if part != '' else 1
                for _ in range(k):
                    base = posixpath.dirname(base)
            rest = m.group(3)
        else:
            m = re.match(r'^Path\(\s*[rRbBuU]?(["\'])([^"\']+)\1\s*\)(.*)$', e)
            if m and looks_rootrel(m.group(2)):
                lit = m.group(2).replace('\\', '/')
                base = norm(scriptdir + '/' + lit) if lit.startswith('../') else norm(lit)
                rest = m.group(3)
            else:
                m = re.match(r'^[rR]?(["\'])([^"\']+)\1\s*$', e)
                if m and looks_rootrel(m.group(2)) and ('/' in m.group(2) or '\\' in m.group(2) or m.group(2).startswith('run-')):
                    lit = m.group(2).replace('\\', '/')
                    if lit.startswith('../'):   # cwd-relative; assume cwd = script dir (INFERRED)
                        return norm(scriptdir + '/' + lit)
                    return norm(lit)
                return None
    comps = re.findall(r'/\s*(?:[rRfF]?(["\'])([^"\']*)\1|([A-Za-z_][A-Za-z0-9_]*))', rest)
    for q, lit, ident in comps:
        pre = (base + '/') if base else ''
        if lit:
            base = pre + lit
        elif ident and ident in env and env[ident] and not env[ident].startswith('/'):
            # identifier holding a filename literal
            base = pre + posixpath.basename(env[ident])
        else:
            base = pre + '{' + (ident or '?') + '}'
    return norm(base) if base is not None else None


def py_path_exprs(line):
    """Find candidate Path expressions in a line: NAME / "x" chains, Path(__file__) chains, literals."""
    out = []
    for m in re.finditer(r'(Path\(__file__\)(?:\.resolve\(\))?(?:\.parent(?!s)|\.parents\[\d+\])*(?:\s*/\s*(?:[rRfF]?["\'][^"\']*["\']|[A-Za-z_]\w*))*)', line):
        out.append(m.group(1))
    for m in re.finditer(r'(?<![\w.])([A-Za-z_]\w*(?:\.parent(?!s)|\.parents\[\d+\])*(?:\s*/\s*(?:[rRfF]?["\'][^"\']*["\']|[A-Za-z_]\w*))+)', line):
        out.append(m.group(1))
    for m in STRLIT.finditer(line):
        s = m.group(2)
        if ' ' not in s and looks_rootrel(s) and ('/' in s or '\\' in s or s.startswith('run-')):
            out.append(m.group(0).lstrip('rbfRBF'))
    return out


def py_docline(src):
    try:
        d = ast.get_docstring(ast.parse(src))
    except Exception:
        d = None
        m = re.match(r'\s*(?:#![^\n]*\n)?(?:\s*#[^\n]*\n)*\s*[rR]?("""|\'\'\')(.*?)\1', src, re.S)
        if m:
            d = m.group(2)
    if d:
        for ln in d.strip().splitlines():
            if ln.strip():
                return ln.strip()
    for ln in src.splitlines()[:15]:
        if ln.startswith('#') and not ln.startswith('#!') and ln.strip('# ').strip():
            return ln.strip('# ').strip()
    return ''


def scan_py(rel, src):
    scriptdir = posixpath.dirname(rel)
    env = {}
    lines = src.splitlines()
    for ln in lines:
        m = PY_ASSIGN.match(ln)
        if m and not ln.startswith((' ' * 8)):
            name, expr = m.group(1), m.group(2)
            r = resolve_py(expr, env, scriptdir)
            if r is not None:
                env[name] = r
            else:
                mm = re.match(r'^[rR]?(["\'])([\w\-.]+\.(?:json|jsonl|csv|pt|npy|npz|png|pdf|log|txt|md|html|parquet|pkl))\1$', expr.strip())
                if mm and name not in env:
                    env[name] = mm.group(2)  # bare filename constant (relative)
    outs, ins, other = [], [], []
    wops = Counter()
    out_args, in_args = [], []
    calls = set()
    for ln in lines:
        s = ln.strip()
        if s.startswith('#'):
            continue
        for k in ('json.dump(', '.to_csv(', 'torch.save(', 'np.save', 'savefig(', '.write_text(', 'pickle.dump(', '.to_parquet('):
            if k in s:
                wops[k.strip('.(')] += 1
        if re.search(r"""open\([^)]*,\s*['"][wa]""", s):
            wops['open-w'] += 1
        for m in re.finditer(r'''["']([\w\-./]+\.(?:py|ps1|sh))["']''', s):
            calls.add(posixpath.basename(m.group(1)))
        im = re.match(r'^(?:from\s+(\w+)\s+import|import\s+(\w+))', s)
        if im:
            mod = im.group(1) or im.group(2)
            if os.path.exists(os.path.join(ROOT, *scriptdir.split('/'), mod + '.py')):
                calls.add(mod + '.py')
        am = re.search(r'add_argument\(\s*["\'](--[\w\-]+)["\'](.*)', s)
        if am:
            flag, restarg = am.group(1), am.group(2)
            dm = re.search(r'default\s*=\s*(.+?)(?:,\s*(?:help|type|choices|nargs|action|required)\s*=|\)\s*$|,\s*$)', restarg)
            if dm:
                dv = dm.group(1).strip()
                r = resolve_py(dv, env, scriptdir)
                txt = f"{flag}={r if r is not None else dv[:60]}"
                if OUT_ARG.match(flag):
                    out_args.append(txt + ('' if r is None else f' [{exists_local(r)}]'))
                    if r:
                        outs.append(r)
                else:
                    if r:
                        in_args.append(txt + f' [{exists_local(r)}]')
                        ins.append(r)
            elif 'required=True' in restarg.replace(' ', '') and (OUT_ARG.match(flag) or IN_ARG.match(flag)):
                (out_args if OUT_ARG.match(flag) else in_args).append(flag + '=REQUIRED')
            continue
        is_w = bool(W_PY.search(s))
        is_r = bool(R_PY.search(s)) and not is_w
        found = []
        for ex in py_path_exprs(s):
            r = resolve_py(ex, env, scriptdir)
            if r is not None:
                found.append(r)
        # constants used on the line
        for name, val in env.items():
            if val and '/' in val and re.search(r'(?<![\w.])' + re.escape(name) + r'(?![\w])', s):
                if not any(f.startswith(val) for f in found):
                    found.append(val)
        m = PY_ASSIGN.match(ln)
        if m and m.group(1) in env and not is_w and not is_r:
            # definition line: classify by the constant's name
            nm = m.group(1).upper()
            v = env[m.group(1)]
            if '/' in (v or ''):
                if re.search(r'OUT|FIG|DEST|SAVE|REPORT|EXPORT|DOC$|PAGE', nm):
                    outs.append(v)
                elif not re.search(r'ROOT|^REPO|PIPELINE|^HERE|^BASE|_DIR$|^D$', nm):
                    other.append(v)
            continue
        for f in found:
            (outs if is_w else ins if is_r else other).append(f)
    return env, outs, ins, other, wops, out_args, in_args, calls


# ---------------------------------------------------------------- powershell / sh
def ps_docline(src):
    m = re.search(r'<#(.*?)#>', src, re.S)
    if m:
        for ln in m.group(1).splitlines():
            t = ln.strip().lstrip('.').strip()
            if t and not t.upper().startswith(('SYNOPSIS', 'DESCRIPTION')):
                return t
    for ln in src.splitlines()[:20]:
        t = ln.strip()
        if t.startswith('#') and not t.startswith('#!') and t.strip('# ').strip():
            return t.strip('# ').strip()
    return ''


def scan_ps(rel, src, is_sh=False):
    scriptdir = posixpath.dirname(rel)
    env = {}
    outs, ins, other = [], [], []
    wops = Counter()
    out_args, in_args = [], []
    calls = set()
    task = ''
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith('#') or not s:
            continue
        # variable assignments
        m = re.match(r'^\$(\w+)\s*=\s*(.+)$', s) if not is_sh else re.match(r'^(?:export\s+)?(\w+)=(.+)$', s)
        if m:
            name, expr = m.group(1), m.group(2).strip()
            val = None
            if '$PSScriptRoot' in expr and 'Split-Path' not in expr:
                val = scriptdir
            elif 'Split-Path' in expr:
                bm = re.search(r'\$(\w+|PSScriptRoot)', expr)
                b = scriptdir if (bm and bm.group(1) == 'PSScriptRoot') else env.get(bm.group(1)) if bm else None
                if b is not None:
                    for _ in range(expr.count('-Parent')):
                        b = posixpath.dirname(b)
                    val = b
            elif is_sh and ('dirname' in expr and '$0' in expr or 'BASH_SOURCE' in expr):
                val = scriptdir
            else:
                jm = re.search(r'Join-Path\s+\$(\w+)\s+["\']([^"\']+)["\']', expr)
                if jm and env.get(jm.group(1)) is not None:
                    val = norm(env[jm.group(1)] + '/' + jm.group(2)) if env[jm.group(1)] else norm(jm.group(2))
                else:
                    lm = re.match(r'^["\']([^"\'$]+)["\']$', expr)
                    if lm and re.match(r'(?i)^[A-Z]:[\\/].*[\\/]expLB[\\/]?$', lm.group(1)):
                        val = ''  # hard-coded absolute repo root, e.g. the lab PC's C:\Users\user4\expLB
                    elif lm and looks_rootrel(lm.group(1)):
                        val = norm(lm.group(1))
            if name.lower() in ('name', 'taskname') and not task:
                tm = re.match(r'^["\']([^"\']+)["\']', expr)
                if tm:
                    task = tm.group(1)
            if val is not None:
                env[name] = val
        tm = re.search(r'-TaskName\s+["\']([^"\']+)["\']', s)
        if tm and not task:
            task = tm.group(1)
        for m2 in re.finditer(r'([\w\-.\\/]+\.(?:py|ps1|sh))\b', s):
            b = posixpath.basename(m2.group(1).replace('\\', '/'))
            if b != posixpath.basename(rel) and not b.startswith('.') and b != 'astral.sh':
                calls.add(b)
        is_w = bool(W_PS.search(s))
        is_r = bool(R_PS.search(s)) and not is_w
        found = []
        for jm in re.finditer(r'Join-Path\s+\$(\w+)\s+["\']([^"\']+)["\']', s):
            b = env.get(jm.group(1))
            if b is not None:
                found.append(norm((b + '/' if b else '') + jm.group(2)))
        # quoted literals, with --flag context
        toks = re.findall(r'["\']([^"\']*)["\']|(--[\w\-]+)', s)
        prev_flag = None
        for lit, flag in toks:
            if flag:
                prev_flag = flag
                continue
            if lit.startswith('--'):
                prev_flag = lit
                continue
            if lit and looks_rootrel(lit) and not lit.endswith(('.py', '.ps1', '.sh')):
                p = norm(lit)
                if prev_flag and OUT_ARG.match(prev_flag):
                    outs.append(p)
                    out_args.append(f'{prev_flag}={p} [{exists_local(p)}]')
                elif prev_flag and IN_ARG.match(prev_flag):
                    ins.append(p)
                    in_args.append(f'{prev_flag}={p} [{exists_local(p)}]')
                else:
                    found.append(p)
            prev_flag = None
        for name, val in env.items():
            if val and re.search(r'\$' + re.escape(name) + r'(?![\w])', s) and not m:
                found.append(val)
        if is_w:
            for k in ('Out-File', 'Set-Content', 'Add-Content', 'Tee-Object', 'Copy-Item', 'rclone'):
                if k.lower() in s.lower():
                    wops[k] += 1
        for f in found:
            if f == scriptdir or f == '' or f == '.':
                continue
            (outs if is_w else ins if is_r else other).append(f)
    return env, outs, ins, other, wops, out_args, in_args, calls, task


def main():
    # git history per path
    log = git('log', '--no-renames', '--format=@@%cI|%an', '--name-only', 'HEAD')
    first, last, n, auth = {}, {}, Counter(), defaultdict(set)
    cur = None
    for line in log.splitlines():
        if line.startswith('@@'):
            cur = line[2:].split('|', 1)
            continue
        p = line.strip()
        if p and cur:
            last.setdefault(p, cur[0][:10])
            first[p] = cur[0][:10]
            n[p] += 1
            auth[p].add(cur[1])
    tracked = set(git('ls-files').splitlines())
    all_files = set()
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in ('.git', '.venv', '.venv-aa', '.venv-aa-win', '__pycache__')]
        for f in fn:
            all_files.add(f)

    scripts = []
    for d in SCRIPT_DIRS:
        full = os.path.join(ROOT, *d.split('/')) if d else ROOT
        for f in sorted(os.listdir(full)):
            if f.endswith(EXTS) and os.path.isfile(os.path.join(full, f)):
                scripts.append((d + '/' + f) if d else f)

    cols = ['script', 'kind', 'git_status', 'lines', 'first_commit', 'last_commit', 'n_commits', 'authors',
            'doc_first_line', 'scheduled_task_name', 'calls_scripts', 'calls_missing_locally', 'write_ops',
            'out_arg_defaults', 'out_paths_INFERRED', 'out_paths_missing_locally',
            'in_arg_defaults', 'in_paths_INFERRED', 'in_paths_missing_locally', 'other_paths_INFERRED']
    rows = []
    for rel in scripts:
        with open(os.path.join(ROOT, *rel.split('/')), encoding='utf-8', errors='replace') as fh:
            src = fh.read()
        kind = rel.rsplit('.', 1)[-1]
        task = ''
        if kind == 'py':
            env, outs, ins, other, wops, oa, ia, calls = scan_py(rel, src)
            doc = py_docline(src)
        else:
            env, outs, ins, other, wops, oa, ia, calls, task = scan_ps(rel, src, is_sh=(kind == 'sh'))
            doc = ps_docline(src) if kind == 'ps1' else ps_docline(src.replace('<#', ''))

        def uniq(xs):
            seen, out = set(), []
            for x in xs:
                x = x.rstrip('/')
                if x and x not in seen and x not in ('.', 'src', 'kd', 'cap'):
                    seen.add(x)
                    out.append(x)
            return out
        outs, ins = uniq(outs), uniq(ins)
        other = [x for x in uniq(other) if x not in outs and x not in ins]
        miss_calls = sorted(c for c in calls if c not in all_files)
        rows.append([
            rel, kind, 'tracked' if rel in tracked else 'untracked/ignored', src.count('\n') + (0 if src.endswith('\n') else 1),
            first.get(rel, ''), last.get(rel, ''), n.get(rel, 0), ';'.join(sorted(auth.get(rel, ()))),
            doc[:200], task, ';'.join(sorted(calls)), ';'.join(miss_calls),
            ';'.join(f'{k}x{v}' for k, v in wops.most_common()),
            ' | '.join(oa), ' | '.join(f'{p}' for p in outs), ' | '.join(p for p in outs if exists_local(p) == 'N'),
            ' | '.join(ia), ' | '.join(ins), ' | '.join(p for p in ins if exists_local(p) == 'N'),
            ' | '.join(other[:25]) + (f' | ...(+{len(other) - 25})' if len(other) > 25 else ''),
        ])
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)
    print('scripts:', len(rows), Counter(r[1] for r in rows))
    print('by dir:', Counter(posixpath.dirname(r[0]) or '(root)' for r in rows))
    print('never committed at this path:', [r[0] for r in rows if not r[4]])
    mc = Counter()
    for r in rows:
        for c in r[11].split(';'):
            if c:
                mc[c] += 1
    print('called-but-missing scripts:', dict(mc))
    om = Counter()
    for r in rows:
        for p in r[15].split(' | ') + r[18].split(' | '):
            if p:
                om[p.split('/')[0] + ('/' + p.split('/')[1] if '/' in p else '')] += 1
    print('referenced paths missing locally (by first 2 components):', om.most_common(60))


if __name__ == '__main__':
    main()
