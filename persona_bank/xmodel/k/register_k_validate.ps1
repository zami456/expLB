# Scheduled Task for the Llama-3.3 validation of the K setup (k_validate.ps1). Tries S4U and falls back to
# Interactive (S4U needs admin on this laptop). A 30-minute repeat for 1 day is the watchdog (IgnoreNew; the
# script is resumable and exits at once after DONE). Registering STARTS it; it then waits for the Qwen K chain.
$ErrorActionPreference = "Stop"
$name = "XModel-K-validate"
$script = Join-Path $PSScriptRoot "k_validate.ps1"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $script + '"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $PSScriptRoot
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Days 1) -StartWhenAvailable
$t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Days 1)
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
try {
    Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
        -Trigger $t1 -Description "Llama-3.3 validation of the Qwen K setup against zami's results" | Out-Null
} catch {
    if ($principal.LogonType -ne 'S4U') { throw }
    Write-Output "S4U refused ($($_.Exception.Message.Trim())); registering Interactive instead"
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
        -Trigger $t1 -Description "Llama-3.3 validation of the Qwen K setup against zami's results" | Out-Null
}
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 15
$i = Get-ScheduledTaskInfo -TaskName $name
Write-Output ("{0}: state={1} lastRun={2} lastResult={3}" -f $name, (Get-ScheduledTask -TaskName $name).State, $i.LastRunTime, $i.LastTaskResult)
