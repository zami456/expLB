# Wait for one stage to finish, then run the next (stops if the first did not finish cleanly).
param([string]$After = "stage1", [string]$Next = "stage2")
$st = "E:\narcbench\logs\$After.status"
while ($true) {
  $s = (Get-Content $st -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($s -eq "DONE") { break }
  if ($s -and $s.StartsWith("FAILED")) { exit 1 }
  Start-Sleep -Seconds 60
}
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "E:\narcbench\reader\run_stage.ps1" -Stage $Next
