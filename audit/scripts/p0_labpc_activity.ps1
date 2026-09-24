# Phase 0, READ-ONLY: is anything still running / writing on the lab PC?
$ErrorActionPreference = 'SilentlyContinue'
"now: " + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz')
"=== newest 12 files under C:\Users\user4\expLB\logs"
Get-ChildItem -LiteralPath 'C:\Users\user4\expLB\logs' -File -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 12 | ForEach-Object { $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss') + '  ' + $_.Length + '  ' + $_.FullName.Replace('C:\Users\user4\expLB\','') }
"=== python / powershell processes (name, id, start, cmdline head)"
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|pwsh|powershell|rclone' } | ForEach-Object { $_.Name + ' ' + $_.ProcessId + ' ' + $_.CreationDate.ToString('MM-dd HH:mm') + ' ' + ($_.CommandLine -replace '\s+',' ').Substring(0, [Math]::Min(180, ($_.CommandLine -replace '\s+',' ').Length)) }
"=== scheduled tasks under user4 that are Running or Ready (non-Microsoft)"
Get-ScheduledTask | Where-Object { $_.TaskPath -notlike '\Microsoft\*' } | ForEach-Object { $i = $_ | Get-ScheduledTaskInfo; $_.TaskName + ' | ' + $_.State + ' | last=' + $i.LastRunTime + ' | result=' + $i.LastTaskResult + ' | next=' + $i.NextRunTime }
"=== END"
