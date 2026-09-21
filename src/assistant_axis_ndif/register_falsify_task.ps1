# CPU-only: the F-86 falsification battery across 4 layers x 3 seeds.
# Reads stored passB activations, never imports nnsight, no GPU and no NDIF -- safe to run
# beside anything else on the lab PC.
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "Falsify-MeanDir"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\run_falsify.ps1"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 12) -StartWhenAvailable
# AtStartup so a reboot resumes the sweep; finished cells are skipped by their output file,
# so a restart repeats nothing.
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT30M"
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger $trigger -Description "F-86 falsification: 3 nulls x 4 layers x 3 seeds (review q4 + w4)" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 20
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
Get-Content (Join-Path $repo "logs\falsify.status") -Tail 5 -ErrorAction SilentlyContinue
