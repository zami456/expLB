# Phase 0, READ-ONLY: actions + triggers of the lab PC's project scheduled tasks that are still enabled.
$ErrorActionPreference = 'SilentlyContinue'
$names = 'Coords-Supervisor','Crossgen-Sync-Periodic','KD-Sync-Periodic','Night-Chain','PassC-PerToken','Crossgen-Sync-Now','Overseer'
foreach ($n in $names) {
  $t = Get-ScheduledTask -TaskName $n
  if (-not $t) { "=== $n : NOT FOUND"; continue }
  $i = $t | Get-ScheduledTaskInfo
  "=== $n | state=$($t.State) | last=$($i.LastRunTime) | next=$($i.NextRunTime) | result=$($i.LastTaskResult)"
  foreach ($a in $t.Actions) { "  action: " + $a.Execute + ' ' + $a.Arguments + '  (cwd=' + $a.WorkingDirectory + ')' }
  foreach ($tr in $t.Triggers) { "  trigger: " + $tr.CimClass.CimClassName + ' start=' + $tr.StartBoundary + ' every=' + $tr.Repetition.Interval + ' for=' + $tr.Repetition.Duration + ' enabled=' + $tr.Enabled }
}
"=== END"
