"run_falsify processes: " + @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_falsify' }).Count
Get-Content 'C:\Users\user4\expLB\logs\falsify.status' -Tail 2
