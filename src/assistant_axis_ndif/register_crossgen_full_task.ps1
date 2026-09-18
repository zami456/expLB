<#
  Register and start a Crossgen-Full-<Pass> scheduled task (S4U, survives SSH drops).

      ... -File src\assistant_axis_ndif\register_crossgen_full_task.ps1 -Pass B
      ... -Pass A -Layers all
      ... -Pass P -Banks ""          # prompt-side control, root only
#>
param(
    [string]$Pass = "B",
    [string]$Layers = "all",
    [string]$Banks = "paper16_famcentred paper16_shared own100 hf100",
    [string]$Seeds = "0 1 2 3 4"
)
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "Crossgen-Full-$Pass"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\run_crossgen_full.ps1"' +
       ' -Pass "' + $Pass + '" -Layers "' + $Layers + '" -Banks "' + $Banks + '" -Seeds "' + $Seeds + '"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 48) `
    -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 10)
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Description "10x10 cross-generalization, pass $Pass, all layers and splits" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 5
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
