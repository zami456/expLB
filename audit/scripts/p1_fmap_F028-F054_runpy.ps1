# Same as p1_labpc_runpy.ps1 (read-only runner for p1_peek.py already in audit_scratch\scripts),
# but forces UTF-8 stdout so files with a BOM / non-cp1252 characters can be printed. No writes on the lab PC.
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location -LiteralPath 'C:\Users\user4\audit_scratch'
$py = 'C:\Users\user4\expLB\.venv-analysis\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) { $py = 'C:\Users\user4\expLB\.venv-aa\Scripts\python.exe' }
& $py ('C:\Users\user4\audit_scratch\scripts\' + '__FILE__') __ARGS__
"python exit $LASTEXITCODE"
