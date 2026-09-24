# W6G forensics part 3: S-19 (the 09-22 12:18-17:28 sync-failure window) and
# S-21 (the 70B layer sweep of 09-22 00:39-01:33 and the shared box). READ-ONLY.
$ErrorActionPreference = 'Continue'
$L = 'C:\Users\user4\expLB\logs'

Write-Output "#### S-19: every sync .status line on 2026-09-22"
foreach ($lg in @('kd_sync.status','crossgen_sync.status','scale_sync.status','coords_supervisor.log','kd_rclone.log','coords_rclone.log')) {
  $p = Join-Path $L $lg
  if (-not (Test-Path $p)) { Write-Output ("  " + $lg + " MISSING"); continue }
  Write-Output ("  --- " + $lg)
  Get-Content $p | Where-Object { $_ -match '2026-09-22|2026/09/22' } |
    Where-Object { $_ -match 'exit|EXIT|START|DONE|CRITICAL|Failed|token|expired|invalid' } |
    Where-Object { $_ -match '1[0-9]:[0-5][0-9]' } |
    Select-Object -First 60 | ForEach-Object { Write-Output ("      " + $_.Substring(0,[Math]::Min(170,$_.Length))) }
}

Write-Output ""
Write-Output "#### S-21: run-crossgen\scale_70b\matrices - file list with mtimes"
$d = 'C:\Users\user4\expLB\run-crossgen\scale_70b\matrices'
if (Test-Path $d) {
  Get-ChildItem $d -File | Sort-Object LastWriteTime |
    ForEach-Object { Write-Output ("  " + $_.Name.PadRight(28) + " " + $_.Length.ToString().PadLeft(9) + "  " + $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')) }
  Write-Output ("  count = " + (Get-ChildItem $d -File).Count)
} else { Write-Output "  MISSING" }

Write-Output ""
Write-Output "#### S-21b: loose logs / scripts touched 2026-09-21 23:00 - 2026-09-22 02:00"
Get-ChildItem 'C:\Users\user4\expLB' -File -Recurse -ErrorAction SilentlyContinue |
  Where-Object { $_.LastWriteTime -ge [datetime]'2026-09-21 23:00:00' -and $_.LastWriteTime -le [datetime]'2026-09-22 02:00:00' } |
  Sort-Object LastWriteTime | Select-Object -First 60 |
  ForEach-Object { Write-Output ("  " + $_.FullName.Replace('C:\Users\user4\expLB\','') + "  " + $_.Length + "  " + $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')) }

Write-Output ""
Write-Output "#### S-21c: is the box shared? (profile folder NAMES only, nothing read inside)"
Get-ChildItem 'C:\Users' -Directory -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ("  " + $_.Name) }
Write-Output "  physical memory:"
$os = Get-CimInstance Win32_OperatingSystem
Write-Output ("     total " + [math]::Round($os.TotalVisibleMemorySize/1MB,2) + " GB   free now " + [math]::Round($os.FreePhysicalMemory/1MB,2) + " GB")
