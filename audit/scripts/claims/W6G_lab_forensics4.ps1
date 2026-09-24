# W6G forensics part 4: the 2026-09-22 16:00-18:00 sync recovery window. READ-ONLY.
$ErrorActionPreference = 'Continue'
$L = 'C:\Users\user4\expLB\logs'
foreach ($lg in @('kd_sync.status','crossgen_sync.status','scale_sync.status')) {
  $p = Join-Path $L $lg
  Write-Output ("#### " + $lg)
  Get-Content $p | Where-Object { $_ -match '2026-09-22 1[5-9]:' } |
    ForEach-Object { Write-Output ("   " + $_) }
}
Write-Output "#### kd_rclone.log : last CRITICAL and first clean transfer after 16:00 on 09-22"
$c = Get-Content (Join-Path $L 'kd_rclone.log') | Where-Object { $_ -match '2026/09/22 1[5-9]:' }
$crit = $c | Where-Object { $_ -match 'CRITICAL' }
Write-Output ("   CRITICAL lines 15:00-19:59 : " + $crit.Count)
if ($crit.Count) { Write-Output ("   LAST CRITICAL : " + $crit[-1].Substring(0,[Math]::Min(120,$crit[-1].Length))) }
$c | Where-Object { $_ -match 'Transferred|nothing to transfer|Copied' } | Select-Object -First 8 |
  ForEach-Object { Write-Output ("   " + $_.Substring(0,[Math]::Min(150,$_.Length))) }
Write-Output "#### coords_rclone.log : first and last CRITICAL on 09-22"
$c2 = Get-Content (Join-Path $L 'coords_rclone.log') | Where-Object { $_ -match '2026/09/22' -and $_ -match 'CRITICAL' }
Write-Output ("   CRITICAL count on 09-22 : " + $c2.Count)
if ($c2.Count) {
  Write-Output ("   FIRST : " + $c2[0].Substring(0,[Math]::Min(120,$c2[0].Length)))
  Write-Output ("   LAST  : " + $c2[-1].Substring(0,[Math]::Min(120,$c2[-1].Length)))
}
