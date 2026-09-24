# READ-ONLY: wait (max 12 min) until no run_falsify.ps1 is running, then report the status tail.
$deadline = (Get-Date).AddMinutes(12)
while ((Get-Date) -lt $deadline) {
  $p = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_falsify' })
  if ($p.Count -eq 0) { break }
  Start-Sleep -Seconds 15
}
$p = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_falsify' })
"still running: " + $p.Count
"status tail:"; Get-Content 'C:\Users\user4\expLB\logs\falsify.status' -Tail 3
