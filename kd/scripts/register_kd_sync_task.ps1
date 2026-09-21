# Periodic Drive sync for kd/, every 15 minutes, indefinitely.
#
# Registered 2026-09-19 after noticing kd/ was covered by NOTHING: night_chain syncs
# run-coords and the sycophancy runs, sync_crossgen syncs crossgen, and kd/ fell between
# them. The WMDP knowledge scores existed on one machine, and that machine had already
# rebooted unprompted once that day.
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "KD-Sync-Periodic"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D +
       '\sync_kd.ps1" -OnBusy Skip -Tag periodic'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2) -StartWhenAvailable

# repeat forever; also fire after a reboot so an unattended restart resumes syncing
$t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes 15)
$t2 = New-ScheduledTaskTrigger -AtStartup
$t2.Delay = "PT8M"

Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger @($t1, $t2) -Description "push kd/ to tamara:expLB/kd every 15 minutes" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 20
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
Get-Content (Join-Path $repo "logs\kd_sync.status") -Tail 3 -ErrorAction SilentlyContinue
