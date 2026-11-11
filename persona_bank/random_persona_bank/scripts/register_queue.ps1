<#
  Deploy every model's run folder and register the RPB-queue Scheduled Task. REGISTERING STARTS THE RUN
  (first trigger one minute later): run this only when the run should begin.
  After it runs, CHECK that LastRunTime changed and queue.status has a QUEUE START line: "registered" is not "ran".

  Laptop pattern from xmodel/register_task.ps1: the current user; S4U if allowed (needs admin here), otherwise
  Interactive (runs while signed in); a 30-minute repeat for 5 days as a watchdog plus AtLogOn, so a crash, sleep
  or reboot re-enters the queue, which re-enters the resumable chain.

      powershell -ExecutionPolicy Bypass -File random_persona_bank\scripts\register_queue.ps1
#>
param([string[]]$Tags = @("llama33", "qwen", "mistral2501", "mistral31", "gemma3"), [int]$StallMinutes = 25)
$ErrorActionPreference = "Stop"
$Base    = Split-Path $PSScriptRoot -Parent
$Repo    = Split-Path $Base -Parent
$Results = Join-Path $Base "results"
$Py      = Join-Path $Repo ".venv-aa-win\Scripts\python.exe"
$Drive   = "tamara:expLB/random_persona_bank_2026-10-10"
$rc      = (Get-Command rclone -ErrorAction Stop).Source
if (-not (Test-Path $Py)) { throw "missing $Py" }

& (Join-Path $PSScriptRoot "deploy.ps1") -Tags $Tags
$Q = Join-Path $Results "_queue"
New-Item -ItemType Directory -Force -Path $Q | Out-Null
Copy-Item (Join-Path $PSScriptRoot "queue.ps1") $Q -Force

$name = "RPB-queue"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + (Join-Path $Q "queue.ps1") + '" -Tags ' + ($Tags -join ",") +
       ' -RunBase "' + $Results + '" -Py "' + $Py + '" -DriveBase "' + $Drive + '" -Rclone "' + $rc + '" -StallMinutes ' + $StallMinutes
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
$t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Days 5)
$t2 = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$t2.Delay = "PT3M"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $Q
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Days 5) -StartWhenAvailable `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 10)
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
$desc = "random persona bank: persona chain for $($Tags -join ', '), in order"
try {
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
Write-Output ("{0}: state={1} lastRun={2} lastResult={3} nextRun={4}" -f $name, (Get-ScheduledTask -TaskName $name).State,
              $i.LastRunTime, $i.LastTaskResult, $i.NextRunTime)
