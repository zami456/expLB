$r = "C:/Users/user4/expLB/run-crossgen/banks/paper16_shared/matrices"
if (Test-Path $r) { Get-ChildItem $r | ForEach-Object { $_.Name } } else { "MISSING $r" }
"--- local vs lab: does the lab hold more layers? ---"
$r2 = "C:/Users/user4/expLB/run-crossgen/banks"
Get-ChildItem $r2 -Directory | ForEach-Object { "{0}: {1} matrices" -f $_.Name, (Get-ChildItem "$($_.FullName)/matrices" -ErrorAction SilentlyContinue).Count }
