$ErrorActionPreference="Stop"
$repo="C:\Users\user4\expLB"; $name="Premise-Layers"
$runner=Join-Path $repo "src\assistant_axis_ndif\run_premise_layers.ps1"
$action=New-ScheduledTaskAction -Execute "powershell.exe" -Argument ('-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "'+$runner+'"') -WorkingDirectory $repo
$principal=New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 6) -StartWhenAvailable
$trigger=New-ScheduledTaskTrigger -AtStartup; $trigger.Delay="PT18M"
Stop-ScheduledTask -TaskName $name -EA SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -EA SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings -Trigger $trigger -Description "pooled-training premise across all 21 stored layers" | Out-Null
Start-ScheduledTask -TaskName $name; Start-Sleep -Seconds 15
Write-Output ("{0}: {1}" -f $name,(Get-ScheduledTask -TaskName $name).State)
Get-Content (Join-Path $repo "logs\premise_layers.status") -Tail 3 -EA SilentlyContinue
