$ErrorActionPreference="Stop"
$repo="C:\Users\user4\expLB"; $name="Cap-Seeds"
$runner=Join-Path $repo "cap\scripts\run_cap_seeds.ps1"
$action=New-ScheduledTaskAction -Execute "powershell.exe" -Argument ('-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "'+$runner+'"') -WorkingDirectory $repo
$principal=New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 4) -StartWhenAvailable
$trigger=New-ScheduledTaskTrigger -AtStartup; $trigger.Delay="PT25M"
Stop-ScheduledTask -TaskName $name -EA SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -EA SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings -Trigger $trigger -Description "cap result at seeds 1/7/13/42" | Out-Null
Start-ScheduledTask -TaskName $name; Start-Sleep -Seconds 20
Write-Output ("{0}: {1}" -f $name,(Get-ScheduledTask -TaskName $name).State)
Get-Content (Join-Path $repo "logs\cap_seeds.status") -Tail 3 -EA SilentlyContinue
