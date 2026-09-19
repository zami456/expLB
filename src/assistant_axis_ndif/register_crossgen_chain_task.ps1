$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "Crossgen-Chain"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\run_crossgen_chain.ps1"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 48) `
    -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 10) -StartWhenAvailable

# The lab PC reboots unprompted (2026-09-14, 2026-09-19) and a task that was Running simply
# does not come back -- it goes Ready with LastTaskResult 267014 (SCHED_S_TASK_TERMINATED)
# and waits for a human. On 2026-09-19 that stranded the chain for ten hours, 97% of the way
# through an upload. An AtStartup trigger re-enters the chain after a reboot; every step is
# resumable and skips finished files, and run_crossgen_chain.ps1 exits immediately if its
# status file already says DONE, so firing on an unrelated boot costs seconds.
# The 5-minute delay lets networking and Tailscale settle before rclone starts.
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT5M"

Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger $trigger `
    -Description "remaining cross-generalization arms: B new banks, A, B_fine, P (resumes at startup)" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 5
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
