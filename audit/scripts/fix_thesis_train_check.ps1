# READ-ONLY: what would Thesis_Train do, and can it actually fire?
$t = Get-ScheduledTask -TaskName 'Thesis_Train'
$i = $t | Get-ScheduledTaskInfo
"state=$($t.State)  last=$($i.LastRunTime)  result=$($i.LastTaskResult)  next=$($i.NextRunTime)  missed=$($i.NumberOfMissedRuns)"
"principal: user=$($t.Principal.UserId) logonType=$($t.Principal.LogonType) runLevel=$($t.Principal.RunLevel)"
foreach ($tr in $t.Triggers) { "trigger: " + $tr.CimClass.CimClassName + " enabled=" + $tr.Enabled + " delay=" + $tr.Delay }
$s = $t.Settings
"settings: StartWhenAvailable=$($s.StartWhenAvailable) RunOnlyIfNetworkAvailable=$($s.RunOnlyIfNetworkAvailable) RunOnlyIfIdle=$($s.RunOnlyIfIdle) DisallowStartIfOnBatteries=$($s.DisallowStartIfOnBatteries) ExecutionTimeLimit=$($s.ExecutionTimeLimit) Enabled=$($s.Enabled)"
foreach ($a in $t.Actions) { "action: " + $a.Execute + " " + $a.Arguments + "  cwd=" + $a.WorkingDirectory }
"--- the script it runs"
$p = 'D:\user4fina\code\run_train.ps1'
if (Test-Path $p) { "path exists, " + (Get-Item $p).Length + " B, mtime " + (Get-Item $p).LastWriteTime; Get-Content $p } else { "NOT FOUND: $p" }
