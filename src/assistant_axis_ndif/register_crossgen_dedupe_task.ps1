<#
  Register and start the crossgen dedupe. It waits for every upload to finish by itself,
  so it is safe to fire while the writers are still running -- and it must be a Scheduled
  Task, because that wait outlives the SSH session that starts it.

  Pass -DryRun to register the listing-only variant instead.
#>
param([switch]$DryRun)
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "Crossgen-Dedupe"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\dedupe_crossgen.ps1"'
if ($DryRun) { $arg = $arg + ' -DryRun' }
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 10)
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Description "wait for crossgen uploads, then name-based dedupe of tamara:expLB/crossgen" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 5
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
