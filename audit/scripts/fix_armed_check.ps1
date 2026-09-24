Get-ScheduledTask | Where-Object { $_.TaskPath -notlike '\Microsoft\*' -and $_.State -ne 'Disabled' } | ForEach-Object {
  $t = $_; $a = ($t.Actions | ForEach-Object { $_.Execute + ' ' + $_.Arguments }) -join ' '
  if ($a -match 'user4' -and $t.Triggers.Count -gt 0) {
    $kinds = ($t.Triggers | ForEach-Object { $_.CimClass.CimClassName -replace 'MSFT_Task','' }) -join ','
    $i = $t | Get-ScheduledTaskInfo
    "{0,-24} triggers={1,-30} next={2}" -f $t.TaskName, $kinds, $i.NextRunTime
  }
}
