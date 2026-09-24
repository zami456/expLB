# READ-ONLY: after the interrupted/duplicated falsify runs — is anything still running, and did any result file change?
"now: " + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
"--- python/powershell processes"
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|powershell' } | ForEach-Object {
  "  " + $_.Name + " PID " + $_.ProcessId + " started " + $_.CreationDate.ToString('HH:mm:ss') + " :: " + (($_.CommandLine -replace '\s+',' ')).Substring(0, [Math]::Min(150, ($_.CommandLine -replace '\s+',' ').Length))
}
"--- run-premise/falsify: file count and md5s"
Get-ChildItem 'C:\Users\user4\expLB\run-premise\falsify' -File | Sort-Object Name | ForEach-Object {
  (Get-FileHash -Algorithm MD5 -LiteralPath $_.FullName).Hash.ToLower() + '  ' + $_.Length + '  ' + $_.LastWriteTime.ToString('MM-dd HH:mm:ss') + '  ' + $_.Name
}
"--- falsify.status tail"
Get-Content 'C:\Users\user4\expLB\logs\falsify.status' -Tail 3
