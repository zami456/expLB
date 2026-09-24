# The missing half of the RF-14 proof: with a DONE line present, a NORMAL run must skip in ~1 s.
# (The -Force path is proven by the 02:19-03:40 run, which ran all 63 cells despite that DONE line.)
$repo = 'C:\Users\user4\expLB'
$st = "$repo\logs\falsify.status"
Copy-Item -LiteralPath $st -Destination 'C:\Users\user4\audit_scratch\out\falsify.status.before_skiptest' -Force
"before, last line: " + (Get-Content $st -Tail 1)
$t0 = Get-Date
& powershell -NoProfile -ExecutionPolicy Bypass -File "$repo\src\assistant_axis_ndif\run_falsify.ps1" | Out-Null
"normal run: exit=$LASTEXITCODE elapsed=$([math]::Round(((Get-Date)-$t0).TotalSeconds,1))s"
"after, last 2 lines:"; Get-Content $st -Tail 2
