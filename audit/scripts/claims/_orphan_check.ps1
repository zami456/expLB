Get-CimInstance Win32_Process -Filter "Name like '%python%' or Name like '%powershell%'" |
  Select-Object ProcessId, Name, CreationDate, @{n='cmd';e={$_.CommandLine}} |
  ForEach-Object { "$($_.ProcessId) | $($_.Name) | $($_.CreationDate) | $($_.cmd)" }
