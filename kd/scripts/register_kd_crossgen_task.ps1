# Scheduled Task for the 11-dataset crossgen (layers 40+44, then 20, then the rest).
# CPU only -- 27_crossgen.py has no nnsight. Scheduled Task rather than Start-Process over
# SSH, which OpenSSH kills with the session.
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "KD-Crossgen"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\run_kd_crossgen.ps1"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 24) -StartWhenAvailable
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT15M"
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger $trigger -Description "11-dataset crossgen on passB: 40/44, then 20, then rest" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 20
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
Get-Content (Join-Path $repo "logs\kd_crossgen.status") -Tail 3 -ErrorAction SilentlyContinue
