$ErrorActionPreference='Continue'
$R='C:\Users\user4\expLB'
Write-Output "=== run-sycophancy* dirs ==="
Get-ChildItem $R -Directory | Where-Object { $_.Name -like 'run-sycophancy*' -or $_.Name -like 'run-scale*' } | ForEach-Object { Write-Output ("DIR  " + $_.FullName) }
foreach ($d in @('run-sycophancy','run-sycophancy-mt')) {
  $p = Join-Path $R $d
  if (Test-Path $p) {
    Write-Output ("--- " + $d + " ---")
    Get-ChildItem $p -Recurse -File | Select-Object -First 40 | ForEach-Object { Write-Output ("  {0,12}  {1}  {2}" -f $_.Length, $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm'), $_.FullName.Substring($R.Length+1)) }
  } else { Write-Output ("MISSING " + $d) }
}
Write-Output "=== logs of interest ==="
foreach ($n in @('cap_smoke.log','scale_3b_vectors_all.log','scale_3b.status','scale_8b.status','scale_3b.log','scale_8b.log','scale_3b_axes.log','scale_8b_axes.log','paper16_shared_axes.log','kd_passAC.status','kd_passAC.log')) {
  $p = Join-Path $R ("logs\" + $n)
  if (Test-Path $p) { $f=Get-Item $p; Write-Output ("  {0,12}  {1}  logs\{2}" -f $f.Length, $f.LastWriteTime.ToString('yyyy-MM-dd HH:mm'), $n) } else { Write-Output ("  MISSING logs\" + $n) }
}
Write-Output "=== responses dirs (row counts) ==="
foreach ($d in @('run-scale\3b\responses','run-scale\8b\responses','run-paper16\responses','run\responses')) {
  $p = Join-Path $R $d
  if (Test-Path $p) { $c=(Get-ChildItem $p -File).Count; Write-Output ("  {0,5} files  {1}" -f $c, $d) } else { Write-Output ("  MISSING " + $d) }
}
