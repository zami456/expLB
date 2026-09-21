# Scheduled Task for the WMDP knowledge-scoring pass (kd Phase 2b).
#
# A Scheduled Task, not Start-Process over SSH: OpenSSH takes the process tree down with the
# session, which killed a crossgen step 24 seconds in on 2026-09-19. The script is resumable
# at item granularity (it skips item_ids already in scores.jsonl), so re-entry is free.
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "KD-Knowledge"
$py = Join-Path $repo ".venv-aa\Scripts\python.exe"
$script = Join-Path $D "score_knowledge_ndif.py"
$log = Join-Path $repo "logs\kd_knowledge.log"

$cmd = '-NoProfile -ExecutionPolicy Bypass -Command ' +
       '"$env:PYTHONIOENCODING=''utf8''; & ''' + $py + ''' ''' + $script +
       ''' 2>&1 | Out-File -Append -Encoding utf8 ''' + $log + '''"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $cmd -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 6) -StartWhenAvailable
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT10M"

Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger $trigger -Description "score WMDP items for what Llama-3.3-70B actually knows" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 10
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
