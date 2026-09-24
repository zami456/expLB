# READ-ONLY: why did the lab PC reboot on 2026-09-22 ~15:34? System log shutdown/startup events.
$ids = 1074, 6008, 6005, 6006, 41, 1076
Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=[datetime]'2026-09-22 14:00'; EndTime=[datetime]'2026-09-22 16:30'; Id=$ids} -ErrorAction SilentlyContinue |
  Sort-Object TimeCreated | ForEach-Object {
    $m = ($_.Message -replace '\s+', ' ')
    "{0}  id={1}  {2}  {3}" -f $_.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss'), $_.Id, $_.ProviderName, $m.Substring(0, [Math]::Min(230, $m.Length))
  }
"--- (1074 = initiated by a user/process, 6008 = unexpected shutdown, 41 = power loss/crash, 6005/6006 = log start/stop)"
