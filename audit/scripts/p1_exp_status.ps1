# READ-ONLY: print every small *.status file (and a few markers) in the lab logs folder. No writes.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$d = 'C:\Users\user4\expLB\logs'
Get-ChildItem -LiteralPath $d -File | Where-Object { $_.Name -like '*.status*' -and $_.Name -notlike 'kd_sync*' -and $_.Name -notlike 'crossgen_sync*' } | Sort-Object Name | ForEach-Object {
  "===== " + $_.Name + "  (" + $_.Length + " B, " + $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss') + ")"
  Get-Content -LiteralPath $_.FullName
}
"===== run/.launched"; Get-Content -LiteralPath 'C:\Users\user4\expLB\run\.launched'
"===== run/DELIVERABLE_READY"; Get-Content -LiteralPath 'C:\Users\user4\expLB\run\DELIVERABLE_READY'
"===== D:/expLB-acts/logs/venv_build.status"; Get-Content -LiteralPath 'D:\expLB-acts\logs\venv_build.status'
