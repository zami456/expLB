# Phase 1: create the approved audit scratch folder on the lab PC (the ONLY place the audit writes there).
$s = 'C:\Users\user4\audit_scratch'
if (-not (Test-Path -LiteralPath $s)) { New-Item -ItemType Directory -Path $s | Out-Null; "CREATED $s" } else { "EXISTS $s" }
New-Item -ItemType Directory -Force -Path "$s\census","$s\scripts","$s\out" | Out-Null
Get-ChildItem -LiteralPath $s | ForEach-Object { $_.FullName }
