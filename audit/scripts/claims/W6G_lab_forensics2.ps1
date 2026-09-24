# W6G forensics part 2: S-19 (rclone token failure window), S-20 (conf backup diff, no secrets),
# S-21 (the 09-22 00:39-00:45 sweep and its resume loop). READ-ONLY.
$ErrorActionPreference = 'Continue'
$L = 'C:\Users\user4\expLB\logs'

Write-Output "#### S-20: rclone.conf vs its backup - structure only, NO secret values printed"
$a = Get-Content 'C:\Users\user4\AppData\Roaming\rclone\rclone.conf'
$b = Get-Content 'C:\Users\user4\AppData\Roaming\rclone\rclone.conf.bak-2026-09-22'
Write-Output ("  current lines=" + $a.Count + "  backup lines=" + $b.Count)
for ($i = 0; $i -lt [Math]::Max($a.Count, $b.Count); $i++) {
  $x = if ($i -lt $a.Count) { $a[$i] } else { '<missing>' }
  $y = if ($i -lt $b.Count) { $b[$i] } else { '<missing>' }
  $kx = ($x -split '=')[0]
  $same = if ($x -eq $y) { 'SAME' } else { 'DIFF' }
  Write-Output ("   line " + ($i+1) + " key='" + $kx + "' " + $same + " (len cur=" + $x.Length + " bak=" + $y.Length + ")")
}

Write-Output ""
Write-Output "#### S-19: rclone / sync logs, lines mentioning exit codes on 09-22"
foreach ($lg in @('kd_rclone.log','coords_rclone.log','night_rclone.log','scale_rclone.log','drive_sync.log','crossgen_rclone.log')) {
  $p = Join-Path $L $lg
  if (-not (Test-Path $p)) { Write-Output ("  " + $lg + " MISSING"); continue }
  Write-Output ("  --- " + $lg)
  Get-Content $p | Select-String -Pattern '09-22|9/22|exit|token|expired|invalid_grant|Failed|CRITICAL' |
    Select-Object -Last 25 | ForEach-Object { Write-Output ("      " + $_.Line.Substring(0, [Math]::Min(200, $_.Line.Length))) }
}

Write-Output ""
Write-Output "#### S-19b: sync .status files"
foreach ($lg in @('kd_sync.status','crossgen_sync.status','scale_sync.status','drive_check.status')) {
  $p = Join-Path $L $lg
  if (-not (Test-Path $p)) { Write-Output ("  " + $lg + " MISSING"); continue }
  Write-Output ("  --- " + $lg + "  mtime " + (Get-Item $p).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
  Get-Content $p | Select-Object -First 40 | ForEach-Object { Write-Output ("      " + $_) }
}

Write-Output ""
Write-Output "#### S-21: logs written 2026-09-22 00:20-02:00 (the sweep window)"
Get-ChildItem $L -Recurse -File | Where-Object {
  $_.LastWriteTime -ge [datetime]'2026-09-22 00:00:00' -and $_.LastWriteTime -le [datetime]'2026-09-22 03:00:00'
} | Sort-Object LastWriteTime | ForEach-Object {
  Write-Output ("  " + $_.Name.PadRight(34) + " " + $_.Length.ToString().PadLeft(10) + "  " + $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
}
