# The capitulation run, end to end. NDIF-bound and ~3h, so a Scheduled Task rather than
# Start-Process over SSH -- OpenSSH takes the process tree down with the session.
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "Cap-Chain"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\run_cap_chain.ps1"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 20) -StartWhenAvailable
# AtStartup so an unprompted lab-PC reboot resumes it. Generation skips roles whose .jsonl
# already exists, so a restart costs only the step that was interrupted.
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT30M"
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger $trigger -Description "cap: generate, judge, vectors x2, axes x3, crossgen" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 25
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
Get-Content (Join-Path $repo "logs\cap_chain.status") -Tail 5 -ErrorAction SilentlyContinue
