<#
  One-shot out-of-band push of the whole run-crossgen tree to Drive, for when finished
  banks are sitting on the lab PC and you do not want to wait for the running step to
  reach its final sync. A Scheduled Task, not Start-Process: OpenSSH's job object kills
  the process tree when the session drops, and this upload runs for an hour or more.

  Safe to fire while a crossgen step is running -- sync_crossgen.ps1 takes an exclusive
  lock, and -DupCheck reports (never deletes) any name collisions afterwards.
#>
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "Crossgen-Sync-Now"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\sync_crossgen.ps1"' +
       ' -Tag manual -OnBusy Wait -WaitMinutes 240 -DupCheck'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 12) `
    -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 10)
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Description "out-of-band rclone push of run-crossgen to tamara:expLB/crossgen" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 5
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
