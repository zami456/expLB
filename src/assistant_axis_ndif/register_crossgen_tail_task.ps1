# One-shot task for the tail of the crossgen chain: the two steps that never ran.
#
# B_fine and P died on 2026-09-19 for two DIFFERENT reasons, an hour apart:
#   1. `-Seeds ""` / `-Banks ""` are dropped by `powershell.exe -File`, so both steps exited 1
#      in the second they started, writing no log. Fixed in run_crossgen_chain.ps1 (pass " ").
#   2. The relaunch was started with Start-Process over SSH, and OpenSSH took the process tree
#      down with the session 24 seconds later -- the failure ONBOARDING.md documents and that
#      this repo has now hit four times.
#
# Hence this file: a Scheduled Task is the only supported way to run long work here.
# -Force because the status file is mid-run rather than DONE; -Only to skip the two steps
# that already completed (B_newbanks 7 min, A 24 min) instead of re-verifying them.
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "Crossgen-Tail"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D +
       '\run_crossgen_chain.ps1" -Force -Only "B_fine P"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 12) -StartWhenAvailable

Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Description "crossgen chain tail: B_fine and P only" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 8
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
