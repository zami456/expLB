<#
  Register and start the one-shot Crossgen-Retrain scheduled task (S4U, runs whether or not
  user4 is logged on, survives the SSH session that registered it).

      powershell -NoProfile -ExecutionPolicy Bypass -File src\assistant_axis_ndif\register_crossgen_task.ps1 -Smoke
      powershell -NoProfile -ExecutionPolicy Bypass -File src\assistant_axis_ndif\register_crossgen_task.ps1 -ExtraSeeds "0 1 2 3 4"
#>
param(
    [switch]$Smoke,
    [string]$ExtraSeeds = "",
    [string]$ExtraLayers = ""
)
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = if ($Smoke) { "Crossgen-Smoke" } else { "Crossgen-Retrain" }

$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\run_crossgen.ps1"'
if ($Smoke) { $arg += " -Smoke" }
if ($ExtraSeeds.Trim()) { $arg += ' -ExtraSeeds "' + $ExtraSeeds.Trim() + '"' }
if ($ExtraLayers.Trim()) { $arg += ' -ExtraLayers "' + $ExtraLayers.Trim() + '"' }

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 48)

Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Description "10x10 cross-generalization retrain -> run-crossgen/ (never run-coords/)" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 5
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
