# Scheduled Task for the Qwen K chain (k_chain.ps1), laptop, current user, Interactive logon.
# A 30-minute repeat for 3 days is the watchdog (IgnoreNew; the chain is resumable and exits at once
# after DONE), plus AtLogOn. Registering STARTS it; k_run.py's gate then waits for the persona chain
# to free Qwen. After registering, CHECK that LastRunTime changed.
param([string]$Model = "qwen-72b", [string]$Folder = "Qwen2.5-72B-Instruct", [string]$Name = "XModel-qwen-K",
      [int]$TokenBudget = 2048)
# 2026-10-07: one K task per model; every output (and its live logs) inside that model's run folder.
$ErrorActionPreference = "Stop"
$name = $Name
$script = Join-Path $PSScriptRoot "k_chain.ps1"
$runRoot = Join-Path (Join-Path $env:USERPROFILE "xmodel-runs") $Folder
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $script + '"'
if ($Model -ne "qwen-72b") {
    $arg += ' -Model ' + $Model + ' -OutRoot "' + (Join-Path $runRoot "K") + '" -LogDir "' + (Join-Path $runRoot "K-logs") +
            '" -Gate "' + (Join-Path $runRoot "logs\chain.status") + '"'
    if ($TokenBudget -ne 2048) { $arg += ' -TokenBudget ' + $TokenBudget }
}
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $PSScriptRoot
# S4U, not Interactive: see xmodel/register_task.ps1 (0xC000013A at ~02:14 on 2026-10-07)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Days 3) -StartWhenAvailable
$t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Days 3)
$t2 = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$t2.Delay = "PT4M"
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
try {
    Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
        -Trigger @($t1, $t2) -Description "$Model K (zami's knowledge direction) beside the xmodel persona chain" | Out-Null
} catch {
    # S4U needs admin rights on this laptop ("Access is denied", 2026-10-07 02:2x): fall back to
    # Interactive; its 30-minute watchdog re-enters the resumable chain if the process is killed.
    if ($principal.LogonType -ne 'S4U') { throw }
    Write-Output "S4U refused ($($_.Exception.Message.Trim())); registering Interactive instead"
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
        -Trigger @($t1, $t2) -Description "$Model K (zami's knowledge direction) beside the xmodel persona chain" | Out-Null
}
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 20
$i = Get-ScheduledTaskInfo -TaskName $name
Write-Output ("{0}: state={1} lastRun={2} lastResult={3}" -f $name, (Get-ScheduledTask -TaskName $name).State, $i.LastRunTime, $i.LastTaskResult)
