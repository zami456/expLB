# Registers the scale-ladder rungs. Each waits for NDIF to free up, then runs the full
# persona pipeline at that model size. AtStartup trigger so a reboot resumes rather than
# strands; every stage is resumable and skips finished work.
param([ValidateSet("3b","8b")][string]$Tag = "3b")
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "Scale-$Tag"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D +
       '\run_scale_chain.ps1" -Tag ' + $Tag
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 20) -StartWhenAvailable
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT30M"
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger $trigger -Description "scale ladder rung: paper16 personas on $Tag" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 12
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
Get-Content (Join-Path $repo "logs\scale_$Tag.status") -Tail 3 -ErrorAction SilentlyContinue
