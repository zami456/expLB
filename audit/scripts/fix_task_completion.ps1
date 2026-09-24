# READ-ONLY: for each still-armed project task, is its work finished? (.done markers, status tails, what it does)
$checks = @(
  @{n='PassAC-11';      script='kd\scripts\run_passAC_11.ps1';                 status='passAC11.status'},
  @{n='PassC-PerToken'; script='..\run_passC.ps1';                              status='passC.log'},
  @{n='Night-Chain';    script='..\night_chain.ps1';                            status='night_chain.log'},
  @{n='Coords-Supervisor'; script='src\assistant_axis_ndif\coords_supervisor.ps1'; status='coords_supervisor.log'},
  @{n='KD-Sync-Periodic';  script='kd\scripts\sync_kd.ps1';                     status='kd_sync.status'},
  @{n='Crossgen-Sync-Periodic'; script='src\assistant_axis_ndif\sync_crossgen.ps1'; status='crossgen_sync.status'}
)
foreach ($c in $checks) {
  $t = Get-ScheduledTask -TaskName $c.n -ErrorAction SilentlyContinue
  $i = if ($t) { $t | Get-ScheduledTaskInfo }
  "=== {0}  state={1}  last={2} rc={3}  next={4}" -f $c.n, $t.State, $i.LastRunTime, $i.LastTaskResult, $i.NextRunTime
  $p = "C:\Users\user4\expLB\logs\" + $c.status
  if (Test-Path -LiteralPath $p) { "    last log line: " + ((Get-Content -LiteralPath $p -Tail 1) -replace '\s+', ' ') }
}
"--- .done markers"
foreach ($d in 'run-coords\passA\.done', 'run-coords\passB\.done', 'run-coords\passC\.done', 'run-crossgen\kd11\passA11\.done') {
  $p = "C:\Users\user4\expLB\$d"
  if (Test-Path -LiteralPath $p) { "    PRESENT  $d  (" + (Get-Item $p).LastWriteTime.ToString('yyyy-MM-dd HH:mm') + ")" } else { "    absent   $d" }
}
"--- is any experiment python running right now?"
@(Get-CimInstance Win32_Process -Filter "Name='python.exe'").Count
