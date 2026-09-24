# READ-ONLY: show the audit's own census log + output size (scratch folder only).
Get-ChildItem -LiteralPath 'C:\Users\user4\audit_scratch' -Recurse -File | ForEach-Object { $_.FullName + ' ' + $_.Length + ' ' + $_.LastWriteTime.ToString('HH:mm:ss') }
if (Test-Path 'C:\Users\user4\audit_scratch\census\labpc_census.log') { Get-Content 'C:\Users\user4\audit_scratch\census\labpc_census.log' -Tail 5 }
