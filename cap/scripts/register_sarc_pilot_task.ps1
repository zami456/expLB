# The deadpan-elicitation pilot: 300 NDIF rollouts + judge + marker scoring.
# A Scheduled Task, not Start-Process over SSH -- OpenSSH takes the process tree down with the
# session and this is remote generation that must survive a dropped connection.
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "Sarc-Pilot"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\run_sarc_pilot.ps1"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 8) -StartWhenAvailable
# AtStartup so a reboot resumes; 1_generate_ndif.py skips (prompt_index, question_index) pairs
# already on disk, so a restart repeats nothing.
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT20M"
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger $trigger -Description "Sarcasm design step 2: can the model produce UNMARKED sarcasm?" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 25
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
Get-Content (Join-Path $repo "logs\sarc_pilot.status") -Tail 5 -ErrorAction SilentlyContinue
