# Phase 0, READ-ONLY: what do the still-enabled periodic tasks do? rclone verbs used + last log lines.
$ErrorActionPreference = 'SilentlyContinue'
$scripts = 'C:\Users\user4\expLB\src\assistant_axis_ndif\coords_supervisor.ps1','C:\Users\user4\night_chain.ps1','C:\Users\user4\run_passC.ps1','C:\Users\user4\expLB\src\assistant_axis_ndif\sync_crossgen.ps1','C:\Users\user4\expLB\kd\scripts\sync_kd.ps1'
foreach ($s in $scripts) {
  "=== $s"
  if (-not (Test-Path -LiteralPath $s)) { "NOT FOUND"; continue }
  $c = Get-Content -LiteralPath $s
  "lines=" + $c.Count + "  md5=" + (Get-FileHash -Algorithm MD5 -LiteralPath $s).Hash + "  mtime=" + (Get-Item -LiteralPath $s).LastWriteTime.ToString('yyyy-MM-dd HH:mm')
  "-- rclone / python / done-marker lines:"
  $c | Select-String -Pattern 'rclone|python|\.py|done|DONE|exit|Disable-ScheduledTask|Unregister' | Select-Object -First 25 | ForEach-Object { '  ' + $_.LineNumber + ': ' + $_.Line.Trim() }
}
foreach ($l in 'coords_supervisor.log','night_chain.log','crossgen_sync.status','kd_sync.status') {
  "=== tail logs\$l"
  Get-Content -LiteralPath ('C:\Users\user4\expLB\logs\' + $l) -Tail 8
}
$pc = Get-ChildItem -LiteralPath 'C:\Users\user4\expLB\logs' -File | Where-Object { $_.Name -match 'passC|passc' } | Sort-Object LastWriteTime -Descending | Select-Object -First 3
foreach ($f in $pc) { "=== tail " + $f.Name + " (" + $f.LastWriteTime + ")"; Get-Content -LiteralPath $f.FullName -Tail 5 }
"=== END"
