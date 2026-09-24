$R='C:\Users\user4\expLB'
foreach ($p in @('run-crossgen\kd11\passC\significance.json','run-crossgen\passC\significance.json','run-crossgen\kd11\passC\significance.json.stale-576','run-crossgen\passC\summary.json','run-crossgen\kd11\passC\summary.json')) {
  $f = Join-Path $R $p
  if (Test-Path $f) { $h=(Get-FileHash $f -Algorithm MD5).Hash; $s=(Get-Item $f).Length; Write-Output ("{0}  {1,10}  {2}" -f $h.ToLower(), $s, $p) } else { Write-Output ("MISSING  " + $p) }
}
