$ErrorActionPreference="Stop"
$repo="C:\Users\user4\expLB"; $name="Cap-Controls"
$runner=Join-Path $repo "cap\scripts\run_cap_controls.ps1"
$action=New-ScheduledTaskAction -Execute "powershell.exe" -Argument ('-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "'+$runner+'"') -WorkingDirectory $repo
$principal=New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 8) -StartWhenAvailable
$trigger=New-ScheduledTaskTrigger -AtStartup; $trigger.Delay="PT30M"
Stop-ScheduledTask -TaskName $name -EA SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -EA SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings -Trigger $trigger -Description "Tier 0.4 format-matched persona control" | Out-Null
Start-ScheduledTask -TaskName $name; Start-Sleep -Seconds 30
Write-Output ("{0}: {1}" -f $name,(Get-ScheduledTask -TaskName $name).State)
Get-Content (Join-Path $repo "logs\cap_controls.status") -Tail 4 -EA SilentlyContinue
