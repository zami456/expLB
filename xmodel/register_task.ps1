# Scheduled Task for one xmodel run. The chain is resumable and exits at once if already DONE.
# After this runs, CHECK that LastRunTime changed: "registered" is not "ran".
#
#   -Machine lab     (default, unchanged from 2026-09-29) S4U as user4 -- the pattern that fires
#                    without a stored password (the /RU user4 route silently never runs) --
#                    AtStartup so an unprompted reboot resumes it, StartWhenAvailable + restarts.
#   -Machine laptop  (2026-10-06, lab PC going down for maintenance) the current user, Interactive
#                    logon (no stored password; runs while signed in), a 30-minute repeat for 3 days
#                    as a watchdog plus AtLogOn, so a crash, sleep or reboot re-enters the chain.
#                    Deploys xmodel\code + the two .ps1 into <RunBase>\<Folder>\code if absent.
#                    Persona vectors only (-SkipEval), owner 2026-10-06.
param([ValidateSet("qwen", "mistral2501", "mistral31", "gemma3", "llama405", "llama33")][string]$Tag = "qwen", [string]$Folder = "",
      [ValidateSet("lab", "laptop")][string]$Machine = "lab",
      [string]$VecSuffix = "",
      # laptop: deploy the run folder's code (and rename the judge prompts) without registering, so the
      # deployed copy can be smoke-tested before the run starts (2026-10-07)
      [switch]$DeployOnly,
      # laptop: where the run folder goes (2026-10-14: the user_stay matched pair deploys to its own base,
      # %USERPROFILE%\xmodel-runs-userstay, so the original runs are never touched); empty = xmodel-runs
      [string]$RunBase = "")
# one run folder per model (owner 2026-10-07); the judge prompt names the model being judged
$known = @{ "qwen" = @("Qwen2.5-72B-Instruct", "Qwen"); "mistral2501" = @("Mistral-Small-24B-Instruct-2501", "Mistral")
            "mistral31" = @("Mistral-Small-3.1-24B-Instruct-2503", "Mistral"); "gemma3" = @("gemma-3-27b-it", "Gemma")
            "llama405" = @("Llama-3.1-405B-Instruct", "Llama"); "llama33" = @("Llama-3.3-70B-Instruct", "Llama") }
if (-not $Folder) { $Folder = $known[$Tag][0] }
$Short = $known[$Tag][1]
# Registering STARTS the run (the laptop's first trigger fires one minute later): register only
# when the run should begin.
$ErrorActionPreference = "Stop"
$name = "XModel-$Tag"

if ($Machine -eq "lab") {
    $Code = "D:\user4fina\$Folder\code"
    $arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $Code + '\run_chain.ps1" -Tag ' + $Tag
    $principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $trigger.Delay = "PT6M"
    $triggers = @($trigger)
} else {
    $repo     = Split-Path $PSScriptRoot -Parent
    if (-not $RunBase) { $RunBase = Join-Path $env:USERPROFILE "xmodel-runs" }
    $Root     = Join-Path $RunBase $Folder
    $Code     = Join-Path $Root "code"
    $Py       = Join-Path $repo ".venv-aa-win\Scripts\python.exe"
    $Drive    = "tamara:expLB/xmodel_2026-10-06"
    $rc       = (Get-Command rclone -ErrorAction Stop).Source
    if (-not (Test-Path $Py)) { throw "missing $Py" }
    if (-not (Test-Path $Code)) {
        New-Item -ItemType Directory -Force -Path $Code | Out-Null
        robocopy (Join-Path $PSScriptRoot "code") $Code /E /XD __pycache__ /NFL /NDL /NJH /NJS | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }
        Write-Output "deployed code -> $Code"
        # xmodel/code's kd copy is Qwen's: its 35 judge prompts say "(an AI assistant, Qwen, an LLM, etc.)".
        # Name the model actually being judged, and prove all 35 changed and nothing else did.
        if ($Short -ne "Qwen") {
            $n = 0
            foreach ($f in Get-ChildItem (Join-Path $Code "kd\personas\kd_instructions") -Filter *.json) {
                $t = [IO.File]::ReadAllText($f.FullName)
                if ($t.Contains("an AI assistant, Qwen, an LLM")) {
                    [IO.File]::WriteAllText($f.FullName, $t.Replace("an AI assistant, Qwen, an LLM", "an AI assistant, $Short, an LLM"))
                    $n++
                }
            }
            $left = @(Select-String -Path (Join-Path $Code "kd\personas\kd_instructions\*.json") -Pattern "Qwen").Count
            if ($n -ne 35 -or $left -ne 0) { throw "judge prompt rename: $n files changed (want 35), $left Qwen mentions left" }
            Write-Output "judge prompts: 35 files now name $Short"
        }
    }
    # the chain and sync scripts always come from the repo copy, so a fix there reaches the run
    Copy-Item (Join-Path $PSScriptRoot "run_chain.ps1"), (Join-Path $PSScriptRoot "sync_drive.ps1") $Code -Force
    if ($DeployOnly) { Write-Output "deploy only: $Code (no task registered)"; return }
    $arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $Code + '\run_chain.ps1" -Tag ' + $Tag +
           ' -RunBase "' + $RunBase + '" -Py "' + $Py + '" -DriveBase "' + $Drive + '" -Rclone "' + $rc + '" -SkipEval -MinDownloadKbps 40 -DownloadGrace 20'
    if ($VecSuffix) { $arg += ' -VecSuffix ' + $VecSuffix }
    # S4U, not Interactive (2026-10-07): an Interactive task died with 0xC000013A (console close) at
    # ~02:14 when the owner's VS Code / Claude session closed; S4U runs outside the desktop session.
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
    $t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
            -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Days 3)
    $t2 = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    $t2.Delay = "PT3M"
    $triggers = @($t1, $t2)
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $Code
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Days 3) -StartWhenAvailable `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 10)
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
try {
    Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
        -Trigger $triggers -Description "xmodel $Tag ($Machine): kd persona bank, all layers" | Out-Null
} catch {
    # S4U needs admin rights on this laptop ("Access is denied", 2026-10-07 02:2x): fall back to
    # Interactive; its 30-minute watchdog re-enters the resumable chain if the process is killed.
    if ($principal.LogonType -ne 'S4U') { throw }
    Write-Output "S4U refused ($($_.Exception.Message.Trim())); registering Interactive instead"
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
        -Trigger $triggers -Description "xmodel $Tag ($Machine): kd persona bank, all layers" | Out-Null
}
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 20
$i = Get-ScheduledTaskInfo -TaskName $name
Write-Output ("{0}: state={1} lastRun={2} lastResult={3} nextRun={4}" -f $name, (Get-ScheduledTask -TaskName $name).State, $i.LastRunTime, $i.LastTaskResult, $i.NextRunTime)
