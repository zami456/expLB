<#
  Register and start the post-upload verification. It waits for the crossgen writers to
  finish, then reports completeness and duplicates into logs\crossgen_verify.txt.
  A Scheduled Task so the wait outlives the SSH session that started it.
#>
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "Crossgen-Verify"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\verify_crossgen_sync.ps1"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 8)
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Description "wait for crossgen uploads, then verify completeness and list duplicates" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 5
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
