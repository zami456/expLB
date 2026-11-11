# Registers (and so STARTS) the Scheduled Task for the user_stay matched pair: pair_chain.ps1 = Qwen, then Llama-3.3.
# The two run folders must already be deployed:
#   register_task.ps1 -Tag qwen    -Machine laptop -RunBase %USERPROFILE%\xmodel-runs-userstay -DeployOnly
#   register_task.ps1 -Tag llama33 -Machine laptop -RunBase %USERPROFILE%\xmodel-runs-userstay -DeployOnly
# Same task settings as register_task.ps1's laptop branch: S4U if allowed (needs admin here), else Interactive; a
# 30-minute watchdog for 4 days plus AtLogOn re-enters the resumable chain after a crash, sleep or reboot.
# After this runs, CHECK that LastRunTime changed: "registered" is not "ran".
#
#   powershell -ExecutionPolicy Bypass -File xmodel\userstay_pair\register_pair.ps1
$ErrorActionPreference = "Stop"
$name = "XModel-userstay-pair"
$Base = Join-Path $env:USERPROFILE "xmodel-runs-userstay"
foreach ($f in "Qwen2.5-72B-Instruct", "Llama-3.3-70B-Instruct") {
    if (-not (Test-Path (Join-Path $Base "$f\code\run_chain.ps1"))) { throw "not deployed: $Base\$f\code" }
}
# the pair script runs from the repo copy, so a fix there reaches the run
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + (Join-Path $PSScriptRoot "pair_chain.ps1") + '"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $Base
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Days 4) -StartWhenAvailable `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 10)
$t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Days 4)
$t2 = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$t2.Delay = "PT3M"
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
$desc = "xmodel user_stay matched pair (F-130): kd persona step, Qwen2.5-72B then Llama-3.3-70B"
try {
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
        -Trigger @($t1, $t2) -Description $desc | Out-Null
} catch {
    Write-Output "S4U refused ($($_.Exception.Message.Trim())); registering Interactive instead"
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
        -Trigger @($t1, $t2) -Description $desc | Out-Null
}
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 20
$i = Get-ScheduledTaskInfo -TaskName $name
Write-Output ("{0}: state={1} lastRun={2} lastResult={3} nextRun={4}" -f $name, (Get-ScheduledTask -TaskName $name).State, $i.LastRunTime, $i.LastTaskResult, $i.NextRunTime)
