# Scheduled Task for Qwen's Alpaca eval read (alpaca_chain.ps1), laptop, current user. Tries S4U and falls back
# to Interactive (S4U needs admin on this laptop). A 30-minute repeat for 2 days is the watchdog (IgnoreNew; the
# chain is resumable and exits at once after DONE). Registering STARTS it. After registering, CHECK that
# LastRunTime changed.
#   -Gen (2026-10-08): instead, XModel-mistral2501-alpaca running alpaca_gen_chain.ps1 -- generate Mistral-2501's own
#   Alpaca replies (Liars' Bench has none for it), read them, build D5 + K and D9 + K. Its defaults are 2501's.
#   -Read gemma3 (2026-10-08): XModel-gemma3-alpaca running alpaca_chain.ps1 on Gemma-3's own Liars' Bench Alpaca rows
#   (code\results\subsets\alpaca.csv in its run folder); D5 + K / D9 + K are built by hand afterwards (d9k.py).
#   -Read mistral31 (2026-10-09): XModel-mistral31-alpaca running alpaca_gen_chain.ps1 for Mistral-3.1 -- waits for its K
#   chain's DONE, skips generation (Liars' Bench has its Alpaca rows), span check, read, D5-D9, D5 + K / D9 + K, Drive.
#   -Read llama405 (2026-10-09): XModel-llama405-alpaca, alpaca_gen_chain.ps1 for Llama-3.1-405B -- waits for its K chain,
#   GENERATES its own Alpaca replies (no Liars' Bench rows), reads every 4th layer + L63 in float32, D5-D9, D5 + K / D9 + K.
param([switch]$Gen, [ValidateSet("", "gemma3", "mistral31", "llama405")][string]$Read = "")
$ErrorActionPreference = "Stop"
$name = if ($Gen) { "XModel-mistral2501-alpaca" } elseif ($Read) { "XModel-$Read-alpaca" } else { "XModel-qwen-alpaca" }
$script = Join-Path $PSScriptRoot $(if ($Gen -or $Read -in @("mistral31", "llama405")) { "alpaca_gen_chain.ps1" } else { "alpaca_chain.ps1" })
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $script + '"'
if ($Read -eq "mistral31") {
    $runRoot = Join-Path $env:USERPROFILE "xmodel-runs"
    $m31 = Join-Path $runRoot "Mistral-Small-3.1-24B-Instruct-2503"
    $arg += ' -Run "' + $m31 + '" -Model mistralai/Mistral-Small-3.1-24B-Instruct-2503 -CsvModel mistral-small-3.1-24b-instruct' +
            ' -KKey mistral-small-3.1-24b -LogDir "' + (Join-Path $runRoot "alpaca-logs-mistral31") + '"' +
            ' -WaitFor "' + (Join-Path $m31 "K\k_chain.status") + '"'
}
if ($Read -eq "llama405") {
    $runRoot = Join-Path $env:USERPROFILE "xmodel-runs"
    $l405 = Join-Path $runRoot "Llama-3.1-405B-Instruct"
    $layers = ((0..31 | ForEach-Object { 4 * $_ }) + 63 | Sort-Object) -join ' '
    $arg += ' -Run "' + $l405 + '" -Model meta-llama/Llama-3.1-405B-Instruct -CsvModel llama-3.1-405b-instruct' +
            ' -KKey llama-405b-3.1 -LogDir "' + (Join-Path $runRoot "alpaca-logs-llama405") + '"' +
            ' -WaitFor "' + (Join-Path $l405 "K\k_chain.status") + '" -Layers "' + $layers + '" -StoreDtype float32'
}
if ($Read -eq "gemma3") {
    $runRoot = Join-Path $env:USERPROFILE "xmodel-runs"
    $arg += ' -Run "' + (Join-Path $runRoot "gemma-3-27b-it") + '" -Model google/gemma-3-27b-it -CsvModel gemma-3-27b-it' +
            ' -LogDir "' + (Join-Path $runRoot "alpaca-logs-gemma3") + '"' + ' -EvalDir eval_fp32 -StoreDtype float32 -StatusName alpaca_fp32.status' 
}
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
$desc = if ($Gen) { "Mistral-2501 own Alpaca replies -> all-layer read -> D5 + K, D9 + K" } `
        elseif ($Read) { "$Read Liars' Bench Alpaca rows, mean-pooled, all layers (control set for D9 + K)" } `
        else { "Qwen Liars' Bench Alpaca rows, mean-pooled, all layers (control set for D9 + K)" }
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
