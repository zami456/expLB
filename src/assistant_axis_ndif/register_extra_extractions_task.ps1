<#
  Register and start the one-shot Extra-Extractions scheduled task (S4U, survives SSH drops).

      powershell -NoProfile -ExecutionPolicy Bypass -File src\assistant_axis_ndif\register_extra_extractions_task.ps1
      ... -Jobs "P0b"     # just the prompt-side pass
#>
param([string]$Jobs = "P0a P0b P0c")
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "Extra-Extractions"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\run_extra_extractions.ps1" -Jobs "' + $Jobs + '"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 24) `
    -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 10)
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Description "NDIF: sycophancy into passA/passC, prompt-side passP, passB_fine layers 37-47" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 5
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
