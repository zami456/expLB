<#
  Register and start the one-shot Crossgen-Banks scheduled task.

      powershell -NoProfile -ExecutionPolicy Bypass -File src\assistant_axis_ndif\register_crossgen_banks_task.ps1 -Layers "40"
#>
param(
    [string]$Layers = "40",
    [string]$Banks = "paper16_famcentred own100 hf100"
)
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "Crossgen-Banks"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\run_crossgen_banks.ps1"' +
       ' -Layers "' + $Layers + '" -Banks "' + $Banks + '"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 48)
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Description "10x10 cross-generalization for other persona banks -> run-crossgen/banks/" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 5
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
