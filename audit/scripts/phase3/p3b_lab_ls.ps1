$r = "C:/Users/user4/expLB/run-crossgen/kd11/banks"
Get-ChildItem -Path $r -Directory | ForEach-Object { $_.Name }
"--- collab ---"
if (Test-Path "$r/collab/matrices") { Get-ChildItem "$r/collab/matrices" | ForEach-Object { "{0} {1}" -f $_.Name, $_.Length } } else { "NO collab/matrices" }
