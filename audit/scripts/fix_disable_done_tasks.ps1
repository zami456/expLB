# Owner-authorised (2026-09-23): disable the project tasks whose work is COMPLETE, so a reboot cannot re-run them.
# Evidence per task in audit/fix/task_completion.txt. Pure Drive-sync tasks are deliberately LEFT ENABLED.
# Re-enable any of them with:  Enable-ScheduledTask -TaskName '<name>'
$targets = 'PassAC-11', 'PassC-PerToken', 'Night-Chain', 'Coords-Supervisor'
foreach ($n in $targets) {
  $t = Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue
  if (-not $t) { "MISSING $n"; continue }
  $before = $t.State
  if ($before -eq 'Running') { Stop-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue; Start-Sleep -Seconds 2 }
  Disable-ScheduledTask -TaskName $n | Out-Null
  $after = (Get-ScheduledTask -TaskName $n).State
  "{0,-20} {1} -> {2}" -f $n, $before, $after
}
"--- still enabled, project tasks (kept on purpose = Drive sync only):"
Get-ScheduledTask | Where-Object { $_.TaskPath -notlike '\Microsoft\*' -and $_.State -ne 'Disabled' } | ForEach-Object {
  $a = ($_.Actions | ForEach-Object { $_.Execute + ' ' + $_.Arguments }) -join ' '
  if ($a -match 'user4') { "    " + $_.TaskName + "  [" + $_.State + "]" }
}
