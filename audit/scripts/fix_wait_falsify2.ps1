# READ-ONLY: wait (max 55 min) for run_falsify.ps1 to finish, then report.
$deadline = (Get-Date).AddMinutes(55)
while ((Get-Date) -lt $deadline) {
  if (@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_falsify' }).Count -eq 0) { break }
  Start-Sleep -Seconds 30
}
"still running: " + @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_falsify' }).Count
Get-Content 'C:\Users\user4\expLB\logs\falsify.status' -Tail 3
