# Register PassAC-11: the passA/passC grids re-run with the eleventh dataset.
# CPU only, no NDIF. Scheduled Task, never Start-Process or a long SSH command.
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "PassAC-11"
$runner = Join-Path $D "run_passAC_11.ps1"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument ('-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $runner + '"') `
    -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 6) -StartWhenAvailable
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT12M"

Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger $trigger -Description "passA/passC 11-dataset grid: layer profile for harm-pressure-kr-matched" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 30
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
Get-Content (Join-Path $repo "logs\passAC11.status") -Tail 8 -ErrorAction SilentlyContinue
