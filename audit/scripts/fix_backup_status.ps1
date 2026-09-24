# READ-ONLY: is the backup rclone still running, and what does Drive show?
Get-CimInstance Win32_Process -Filter "Name='rclone.exe'" | ForEach-Object { "PID " + $_.ProcessId + "  " + $_.CreationDate.ToString('HH:mm:ss') + "  " + ($_.CommandLine -replace '\s+',' ').Substring(0, [Math]::Min(230, ($_.CommandLine -replace '\s+',' ').Length)) }
"--- log sizes"
Get-ChildItem 'C:\Users\user4\audit_scratch\out' -Filter 'backup_*' | ForEach-Object { $_.Name + ' ' + $_.Length + ' ' + $_.LastWriteTime.ToString('HH:mm:ss') }
"--- tail of the copy log"
if (Test-Path 'C:\Users\user4\audit_scratch\out\backup_copy_expLB.log') { Get-Content 'C:\Users\user4\audit_scratch\out\backup_copy_expLB.log' -Tail 6 }
