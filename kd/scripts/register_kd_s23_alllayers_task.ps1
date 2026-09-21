# CPU-only: extend the kd_s23 crossgen grid from 3 layers to all 21 stored in passB.
# No GPU, no NDIF -- 27_crossgen.py reads stored activations and never imports nnsight, so
# this runs beside the scale rungs without competing for the API.
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "KD-S23-AllLayers"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\run_kd_s23_alllayers.ps1"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 20) -StartWhenAvailable
# AtStartup so an unprompted lab-PC reboot resumes the sweep rather than silently dropping it;
# crossgen skips finished matrices, so a restart costs nothing and repeats nothing.
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT40M"
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger $trigger -Description "kd_s23 crossgen across all 21 passB layers (NEXT.md item 3)" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 20
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
Get-Content (Join-Path $repo "logs\kd_s23_alllayers.status") -Tail 4 -ErrorAction SilentlyContinue
