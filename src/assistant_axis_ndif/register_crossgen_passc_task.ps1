<#
  The pass-C cross-generalization arm: the per-token store's 6 layers (20,22,28,40,44,48),
  root + all seven persona banks, 5 extra 80/20 splits.

  Why it is a separate task rather than a step in run_crossgen_chain.ps1: the chain instance
  now running parsed its step list at launch, so appending a step there would not reach it.
  The chain script is left alone; this runs the same run_crossgen_full.ps1 with -Pass C.

  Safe to run alongside the chain's pass A. They are separate python processes on a 32-thread
  box, and their Drive syncs serialise through sync_crossgen.ps1's lock. Pass C reads ~32 GB
  from disk because each chunk carries its per-token block, so expect it to be I/O bound.
#>
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "Crossgen-PassC"
$banks = "paper16_famcentred paper16_shared own100 hf100 hf275_loose hf275_strict own100_strict"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\run_crossgen_full.ps1"' +
       ' -Pass C -Layers all -Banks "' + $banks + '" -Seeds "0 1 2 3 4"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 24) `
    -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 10)
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Description "cross-generalization on run-coords\passC (6 layers), root + 7 banks" | Out-Null
Start-ScheduledTask -TaskName $name
Start-Sleep -Seconds 5
Write-Output ("{0}: {1}" -f $name, (Get-ScheduledTask -TaskName $name).State)
