# READ-ONLY: current md5 of every run-premise/falsify file, to compare with the 20:56 census.
Get-ChildItem 'C:\Users\user4\expLB\run-premise\falsify' -File | Sort-Object Name | ForEach-Object {
  (Get-FileHash -Algorithm MD5 -LiteralPath $_.FullName).Hash.ToLower() + '  ' + $_.Name
}
"run_falsify still running: " + @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_falsify' }).Count
