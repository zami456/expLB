# Scheduled Task XModel-eval-queue for eval_queue.ps1 (Liars' Bench eval reads for Mistral-3.1, Qwen, Gemma-3; owner
# 2026-10-09). Laptop, current user; S4U is tried and falls back to Interactive (S4U needs admin here). A 30-minute repeat
# for 3 days is the watchdog (IgnoreNew; the queue is resumable and exits at once after ALL DONE). Registering STARTS it;
# it then waits for -WaitFor (Mistral-3.1's Alpaca chain) to end in DONE. CHECK that LastRunTime changed.
$ErrorActionPreference = "Stop"
$name = "XModel-eval-queue"
$script = Join-Path $PSScriptRoot "eval_queue.ps1"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $script + '"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $PSScriptRoot
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Days 3) -StartWhenAvailable
$t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Days 3)
$t2 = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$t2.Delay = "PT4M"
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
$desc = "Liars' Bench eval reads (every 4th layer + half depth) for Mistral-3.1, Qwen, Gemma-3"
try {
    Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
        -Trigger @($t1, $t2) -Description $desc | Out-Null
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
