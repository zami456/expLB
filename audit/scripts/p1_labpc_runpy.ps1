# Run an audit Python script that was uploaded to audit_scratch\scripts, with the lab's analysis venv.
# Working directory = audit_scratch; no bytecode writes; the script must only READ owner files.
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location 'C:\Users\user4\audit_scratch'
$py = 'C:\Users\user4\expLB\.venv-analysis\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) { $py = 'C:\Users\user4\expLB\.venv-aa\Scripts\python.exe' }
& $py ('C:\Users\user4\audit_scratch\scripts\' + '__FILE__') __ARGS__
"python exit $LASTEXITCODE"
