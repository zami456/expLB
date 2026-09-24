# READ-ONLY: every non-Microsoft scheduled task whose action runs something under C:\Users\user4 (the team's
# experiment drivers), as TSV: name, state, last run, last result, next run, executable + arguments, working dir.
$ErrorActionPreference = 'SilentlyContinue'
"name`tstate`tlast_run`tlast_result`tnext_run`taction`tcwd"
Get-ScheduledTask | Where-Object { $_.TaskPath -notlike '\Microsoft\*' } | ForEach-Object {
  $t = $_
  $acts = ($t.Actions | ForEach-Object { ($_.Execute + ' ' + $_.Arguments).Trim() }) -join ' ;; '
  if ($acts -match 'user4') {
    $i = $t | Get-ScheduledTaskInfo
    $cwd = ($t.Actions | Select-Object -First 1).WorkingDirectory
    "$($t.TaskName)`t$($t.State)`t$($i.LastRunTime.ToString('yyyy-MM-dd HH:mm'))`t$($i.LastTaskResult)`t$(if ($i.NextRunTime) { $i.NextRunTime.ToString('yyyy-MM-dd HH:mm') })`t$acts`t$cwd"
  }
}
