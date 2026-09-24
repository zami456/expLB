Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
  "PID " + $_.ProcessId + "  started " + $_.CreationDate.ToString('yyyy-MM-dd HH:mm:ss') + "  " + (($_.CommandLine -replace '\s+',' ').Substring(0, [Math]::Min(200, ($_.CommandLine -replace '\s+',' ').Length)))
}
