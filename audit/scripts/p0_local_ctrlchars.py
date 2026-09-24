"""Phase-0: find stray control characters (e.g. a backspace 0x08 left by an unescaped '\\b'
in a generated path) in tracked text sources: *.py, *.ps1, *.sh, *.md, *.json, *.jsonl, *.yaml.

Prints path, line number, the control bytes, and the line with controls shown as <0xNN>.
Read-only.  Re-run:  PYTHONIOENCODING=utf-8 python audit/scripts/p0_local_ctrlchars.py
"""
import os
import re
import subprocess

ROOT = r"C:\Users\SURAIYA\Documents\STUDY\thesis\expLB"
EXT = ('.py', '.ps1', '.sh', '.md', '.json', '.yaml', '.yml', '.toml', '.txt', '.tex')
CTRL = re.compile(rb'[\x00-\x08\x0b\x0c\x0e-\x1f]')

files = subprocess.run(['git', '-C', ROOT, '-c', 'core.quotepath=off', 'ls-files'], capture_output=True,
                       check=True).stdout.decode('utf-8').splitlines()
hits = 0
for rel in files:
    if not rel.endswith(EXT) or rel.startswith(('docs/concepts', 'docs/usage', 'docs/models', 'docs/patterns',
                                                  'docs/developing', 'docs/errors', 'docs/gotchas',
                                                  'docs/reference', 'docs/remote')):
        continue
    p = os.path.join(ROOT, *rel.split('/'))
    if not os.path.isfile(p) or os.path.getsize(p) > 20_000_000:
        continue
    with open(p, 'rb') as fh:
        data = fh.read()
    if not CTRL.search(data):
        continue
    for i, line in enumerate(data.split(b'\n'), 1):
        if CTRL.search(line):
            hits += 1
            shown = CTRL.sub(lambda m: f'<0x{m.group(0)[0]:02x}>'.encode(), line).decode('utf-8', 'replace').strip()
            print(f'{rel}:{i}: {shown[:220]}')
print('lines with control chars:', hits)
