# Run an audit Python script on the lab PC with the .venv-aa environment (has transformers).
# Working directory = audit_scratch; no bytecode writes; the script must only READ owner files.
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location 'C:\Users\user4\audit_scratch'
$py = 'C:\Users\user4\expLB\.venv-aa\Scripts\python.exe'
& $py ('C:\Users\user4\audit_scratch\scripts\' + '__FILE__') __ARGS__
"python exit $LASTEXITCODE"
