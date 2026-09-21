# Register HPKR-Topup: full harm-pressure-knowledge-report into passA and passC.
#
# Scheduled Task, never Start-Process or a long SSH command. AtStartup + 12-minute delay so an
# unprompted reboot resumes it; 8_coords_ndif.py is resumable per chunk by row_id, so re-entry
# costs nothing and cannot duplicate a row.
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "HPKR-Topup"
$runner = Join-Path $D "run_hpkr_topup.ps1"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument ('-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $runner + '"') `
    -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 10) -StartWhenAvailable
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT12M"

Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger $trigger -Description "HP-KR full extraction into passA (80 layers) and passC (per-token)" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 20
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
Get-Content (Join-Path $repo "logs\hpkr_topup.status") -Tail 6 -ErrorAction SilentlyContinue
