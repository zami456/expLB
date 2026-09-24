$ErrorActionPreference='Continue'
$R='C:\Users\user4\expLB'
Write-Output "=== run-sycophancy\rows.jsonl first line ==="
Get-Content (Join-Path $R 'run-sycophancy\rows.jsonl') -TotalCount 1
Write-Output "=== run-sycophancy\rows.jsonl line count ==="
(Get-Content (Join-Path $R 'run-sycophancy\rows.jsonl') | Measure-Object -Line).Lines
Write-Output "=== run-sycophancy-mt\questions.jsonl first 3 ==="
Get-Content (Join-Path $R 'run-sycophancy-mt\questions.jsonl') -TotalCount 3
Write-Output "=== questions.jsonl line count ==="
(Get-Content (Join-Path $R 'run-sycophancy-mt\questions.jsonl') | Measure-Object -Line).Lines
Write-Output "=== scale_3b.status ==="
Get-Content (Join-Path $R 'logs\scale_3b.status')
Write-Output "=== scale_8b.status ==="
Get-Content (Join-Path $R 'logs\scale_8b.status')
Write-Output "=== run-scale\3b\responses first file, first line (trunc 600) ==="
$f=(Get-ChildItem (Join-Path $R 'run-scale\3b\responses') -File)[0]
Write-Output $f.Name
(Get-Content $f.FullName -TotalCount 1).Substring(0,[Math]::Min(600,(Get-Content $f.FullName -TotalCount 1).Length))
