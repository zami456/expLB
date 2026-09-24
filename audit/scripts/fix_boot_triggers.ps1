# READ-ONLY: last boot time, and which project tasks fire on boot (these re-ran the chains on 2026-09-22).
"now: " + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz')
"last boot: " + (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')
"--- project tasks with a boot trigger"
Get-ScheduledTask | Where-Object { $_.TaskPath -notlike '\Microsoft\*' } | ForEach-Object {
  $t = $_
  $boot = $t.Triggers | Where-Object { $_.CimClass.CimClassName -match 'BootTrigger' }
  if ($boot) {
    $i = $t | Get-ScheduledTaskInfo
    $delay = ($boot | ForEach-Object { $_.Delay }) -join ','
    "{0,-24} state={1,-9} delay={2,-8} last={3} result={4}" -f $t.TaskName, $t.State, $delay, $i.LastRunTime, $i.LastTaskResult
  }
}
