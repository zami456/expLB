# The s23 arm: waits for KD-Chain, extracts persona vectors from scores 2 AND 3, builds axes,
# scores the 11-dataset grid. Fixes the 1.31x sample imbalance that score-3 filtering imposes
# on the knowledge_denial vs sincere_ignorance contrast. See run_kd_s23.ps1 for the numbers.
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "KD-S23"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\run_kd_s23.ps1"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 16) -StartWhenAvailable
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT25M"
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger $trigger -Description "kd s23 arm: vectors from scores 2+3, axes, 11-dataset crossgen" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 12
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
Get-Content (Join-Path $repo "logs\kd_s23.status") -Tail 2 -ErrorAction SilentlyContinue
