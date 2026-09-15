<#
  Register the coords supervisor as a Scheduled Task.

  Not Start-Process over SSH. Windows OpenSSH can take the process tree down with the
  session, and this run has to outlive every SSH connection from the laptop. The same
  S4U task pattern carried the 100-role Assistant-Axis run for two days.

      powershell -NoProfile -ExecutionPolicy Bypass -File src\assistant_axis_ndif\register_coords_task.ps1
#>
$ErrorActionPreference = 'Stop'
$D    = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = 'Coords-Supervisor'

# A supervisor killed mid-run leaves its single-instance lock behind, and the next one
# reads the dead pid and exits immediately. Clear it here rather than debugging it later.
$lock = "$repo\logs\coords_supervisor.lock"
if (Test-Path $lock) { Remove-Item $lock -Force; Write-Output 'cleared stale coords_supervisor.lock' }

if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false
    Write-Output 'unregistered pre-existing task'
}

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument ('-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\coords_supervisor.ps1"') `
    -WorkingDirectory $repo

# every boot, 2 min in so networking is up
$boot = New-ScheduledTaskTrigger -AtStartup
$boot.Delay = 'PT2M'
# Every 15 min forever, to revive a supervisor that died without a reboot.
#
# Parallel, NOT IgnoreNew. With IgnoreNew a terminated-but-not-reaped instance made the
# scheduler believe one was still running, so the 14:59 and 15:14 triggers were dropped
# silently and a dead supervisor stayed dead for 28 minutes. The script's own lock is
# what prevents two supervisors; the scheduler does not need to help, and when it tries
# it removes the only recovery path.
$rep = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddSeconds(30)) -RepetitionInterval (New-TimeSpan -Minutes 15)

# S4U: runs whether or not user4 is logged on, with no stored password, inside user4's
# profile so the NDIF and HF credentials the pipeline needs resolve normally.
$principal = New-ScheduledTaskPrincipal -UserId 'user4' -LogonType S4U -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -MultipleInstances Parallel -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)

Register-ScheduledTask -TaskName $name -Action $action -Trigger @($boot, $rep) `
    -Principal $principal -Settings $settings `
    -Description 'Keeps the persona-coordinates activation extraction (Pass B) alive on NDIF.' | Out-Null

Write-Output ('registered ' + $name)
Start-ScheduledTask -TaskName $name
Get-ScheduledTask -TaskName $name | Select-Object TaskName, State | Format-Table -AutoSize
