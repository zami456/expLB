# READ-ONLY. Same as p1_labpc_runpy.ps1 + p1_peek.py (already in audit_scratch/scripts), but forces UTF-8
# stdout so files with a BOM / non-cp1252 characters print. Sent via -EncodedCommand; writes nothing on the lab PC.
# Usage: python audit/scripts/labpc_run.py audit/scripts/p1_fmap_F082-F109_peek.ps1 "ARGS=<path> <mode> [arg]"
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$py = 'C:\Users\user4\expLB\.venv-analysis\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) { $py = 'C:\Users\user4\expLB\.venv-aa\Scripts\python.exe' }
& $py 'C:\Users\user4\audit_scratch\scripts\p1_peek.py' __ARGS__
"python exit $LASTEXITCODE"
