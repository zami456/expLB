# Scheduled Task for Phase 3: the 36-persona kd bank (5,400 rollouts -> axes).
# AtStartup trigger so an unprompted reboot resumes it; every stage is resumable and the
# chain exits immediately if its status file already says DONE.
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "KD-Chain"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\run_kd_chain.ps1"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 24) -StartWhenAvailable `
    -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 10)
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT6M"
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger $trigger -Description "kd Phase 3: 36 personas x 5 variants x 30 questions" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 30
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
Get-Content (Join-Path $repo "logs\kd_chain.status") -Tail 5 -ErrorAction SilentlyContinue
