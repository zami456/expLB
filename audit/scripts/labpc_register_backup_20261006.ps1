# Registers the 2026-10-06 lab-PC backup (labpc_backup_20261006.ps1) as a scheduled task.
# Same pattern as kd/scripts/register_kd_sync_task.ps1, which is known to fire on this box:
# user4 / S4U, a 1-minute start plus a 30-minute repeat for 3 days, and an at-startup trigger,
# so an unprompted reboot resumes the backup. IgnoreNew keeps it to one instance; the script
# itself exits at once after "ALL DONE".
$ErrorActionPreference = "Stop"
$name = "Backup-LabPC-20261006"
$script = "C:\Users\user4\audit_scratch\backup\scripts\labpc_backup_20261006.ps1"
if (-not (Test-Path $script)) { throw "missing $script" }
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $script + '"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory "C:\Users\user4\audit_scratch\backup"
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Days 2) -StartWhenAvailable
$t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Days 3)
$t2 = New-ScheduledTaskTrigger -AtStartup
$t2.Delay = "PT8M"
if (Get-ScheduledTask -TaskName $name -EA SilentlyContinue) { throw "task $name already exists - not touching it" }
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger @($t1, $t2) -Description "one-off backup of the thesis data to tamara:expLB/labpc-backup-2026-10-06" | Out-Null
$i = Get-ScheduledTaskInfo -TaskName $name
"{0}: state {1}, last run {2}, next run {3}" -f $name, (Get-ScheduledTask -TaskName $name).State, $i.LastRunTime, $i.NextRunTime
