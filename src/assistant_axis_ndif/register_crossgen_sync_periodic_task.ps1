<#
  A whole-tree crossgen sync every 30 minutes, for as long as it is registered.

  Why this is needed on top of the per-bank syncs: those use -OnBusy Skip so an upload never
  delays the NDIF/probe work, and each pass's final sync (-OnBusy Wait) is what guarantees
  completeness. But when two passes run at once -- pass A finishing a bank every ~25 min and
  pass C every ~2.5 min, each sync taking ~20 min -- nearly every intermediate sync skips, and
  a long lock holder makes it worse: the 2026-09-18 dedupe held the lock 1 h 45 m and pass A's
  root and paper16_famcentred subtrees were still absent from Drive hours after they finished.
  Correctness was never at risk; freshness was. This bounds the lag at ~30 minutes.

  -OnBusy Skip so these never queue up behind each other, and whole-tree so it picks up
  whatever the per-bank calls missed, in any pass, without needing to know which.
#>
$ErrorActionPreference = "Stop"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
$name = "Crossgen-Sync-Periodic"
$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $D + '\sync_crossgen.ps1"' +
       ' -Tag periodic -OnBusy Skip'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId "user4" -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 6)
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 30)
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings `
    -Trigger $trigger -Description "whole-tree crossgen -> Drive every 30 min; skips if a sync is running" | Out-Null
Write-Output ("{0}: registered, first run in ~1 min, repeating every 30 min" -f $name)
