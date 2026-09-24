# READ-ONLY. Runs a short READ-ONLY python snippet on the lab PC by piping it to the lab python's stdin.
# Nothing is uploaded or written on the lab PC (sent via -EncodedCommand; keep the snippet short so the
# encoded command stays < 7000 chars, otherwise labpc_run.py would upload it into audit_scratch).
# Usage: python audit/scripts/labpc_run.py audit/scripts/p1_fmap_F001-F027_pyc.ps1 "CODE=$(cat snippet.py)"
# The snippet must use only ASCII and must not open files for writing.
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::ASCII
Set-Location 'C:\Users\user4\expLB'
$py = 'C:\Users\user4\expLB\.venv-analysis\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) { $py = 'C:\Users\user4\expLB\.venv-aa\Scripts\python.exe' }
$code = @'
__CODE__
'@
$code | & $py -
"python exit $LASTEXITCODE"
