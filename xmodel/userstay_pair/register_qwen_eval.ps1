# Registers (and so STARTS) Scheduled Task XModel-qwen-evallb for qwen_eval.ps1 (owner 2026-10-16: "Run Qwen detection").
# Same settings as register_eval_queue.ps1: S4U if allowed (needs admin here), else Interactive; a 30-minute watchdog for
# 2 days plus AtLogOn re-enters the resumable read after a crash, sleep or reboot. CHECK that LastRunTime changed.
#
#   powershell -ExecutionPolicy Bypass -File xmodel\userstay_pair\register_qwen_eval.ps1
$ErrorActionPreference = "Stop"
$name = "XModel-qwen-evallb"
$script = Join-Path $PSScriptRoot "qwen_eval.ps1"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $script + '"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $PSScriptRoot
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Days 2) -StartWhenAvailable
$t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Days 2)
$t2 = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$t2.Delay = "PT4M"
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
$desc = "Qwen2.5-72B Liars' Bench read (ID, HP-KR, HP-C; every 4th layer + 40) for the user_stay detection comparison"
try {
    # -ErrorAction Stop explicitly: the ScheduledTasks (CDXML) cmdlets do not always see the script's preference
    Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
        -Trigger @($t1, $t2) -Description $desc -ErrorAction Stop | Out-Null
} catch {
    if ($principal.LogonType -ne 'S4U') { throw }
    Write-Output "S4U refused ($($_.Exception.Message.Trim())); registering Interactive instead"
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
        -Trigger @($t1, $t2) -Description $desc | Out-Null
}
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 20
$i = Get-ScheduledTaskInfo -TaskName $name
Write-Output ("{0}: state={1} lastRun={2} lastResult={3}" -f $name, (Get-ScheduledTask -TaskName $name).State, $i.LastRunTime, $i.LastTaskResult)
