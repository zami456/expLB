# Registers Crossgen-Final-Report: waits for Crossgen-Chain to stop, then writes
# logs\crossgen_final_report.txt (per-pass status, layer coverage, local vs Drive parity).
#
# This script was reconstructed 2026-09-19 from the live task definition. The task had been
# registered ad hoc, so losing it meant losing the only copy -- which matters, because it is
# the thing that tells you a 20-hour chain finished.
#
# The AtStartup trigger mirrors the one on Crossgen-Chain. The lab PC reboots unprompted
# (2026-09-14, 2026-09-19) and a Running task does not come back on its own; on 2026-09-19
# that killed both the chain and this reporter, so no report was ever written. The delay is
# PT15M against the chain's PT5M so the chain is already running again by the time this
# starts polling -- otherwise it would see "not running" at boot and report prematurely.
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "Crossgen-Final-Report"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\report_crossgen_done.ps1"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 26) -StartWhenAvailable
$trigger = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT15M"

# Stop before unregistering: Unregister-ScheduledTask does NOT stop a running instance, so
# re-registering while it runs leaves the old PowerShell alive and you get two of them.
Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger $trigger `
    -Description "wait for the crossgen chain, then report passes, layers, Drive parity" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 5
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
