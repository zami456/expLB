# Owner-authorised live test of the RF-14 fix, on the cheapest chain (Falsify: every cell exists, so it only
# re-evaluates gates; the 09-22 run took 5 s). The status file is copied to audit_scratch first.
$repo = 'C:\Users\user4\expLB'
$st = "$repo\logs\falsify.status"
Copy-Item -LiteralPath $st -Destination 'C:\Users\user4\audit_scratch\out\falsify.status.before_guardtest' -Force
"--- before: last 2 lines"; Get-Content $st -Tail 2
"=== TEST 1: normal run (must SKIP because the chain is DONE)"
$t0 = Get-Date
& powershell -NoProfile -ExecutionPolicy Bypass -File "$repo\src\assistant_axis_ndif\run_falsify.ps1" | Out-Null
"    exit=$LASTEXITCODE  elapsed=$([math]::Round(((Get-Date)-$t0).TotalSeconds,1))s"
"    status tail:"; Get-Content $st -Tail 2
"=== TEST 2: forced run (must PROCEED past the guard)"
$t1 = Get-Date
& powershell -NoProfile -ExecutionPolicy Bypass -File "$repo\src\assistant_axis_ndif\run_falsify.ps1" -Force | Out-Null
"    exit=$LASTEXITCODE  elapsed=$([math]::Round(((Get-Date)-$t1).TotalSeconds,1))s"
"    status tail:"; Get-Content $st -Tail 4
