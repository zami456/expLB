$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONIOENCODING='utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location 'C:\Users\user4\audit_scratch'
& 'C:\Users\user4\expLB\.venv-analysis\Scripts\python.exe' 'C:\Users\user4\audit_scratch\scripts\w4_null_lab.py'
"python exit $LASTEXITCODE"
