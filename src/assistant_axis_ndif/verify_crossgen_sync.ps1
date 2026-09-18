<#
  Wait for every crossgen upload to finish, then prove the Drive copy is complete and
  free of duplicates. Writes logs\crossgen_verify.txt and exits.

  Why this runs on the lab PC rather than from the laptop: the laptop has ~0.4 GB of
  7.7 GB free and the OS reclaims long-lived watcher processes there, so a wait loop
  started from the laptop dies before the upload it is waiting on. The lab PC has the
  headroom, is where the files and rclone already live, and keeps running when the SSH
  link drops. Read the report file afterwards instead of holding a session open.

  Read-only: rclone size, and dedupe in LIST mode. It never deletes.
#>
param(
    [int]$MaxWaitMinutes = 360,
    [int]$PollSeconds = 60
)
$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$out = Join-Path $repo "logs\crossgen_verify.txt"
# Out-File -Encoding utf8, never Tee-Object: Tee writes UTF-16 in PowerShell 5.1 and the
# report ends up half UTF-8, half UTF-16, unreadable without naming the encoding.
function Say($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $out }

$rc = Join-Path $repo "tools\rclone.exe"
if (-not (Test-Path $rc)) { $rc = "C:\Users\user4\tools\rclone.exe" }
if (-not (Test-Path $rc)) { Say "ABORT no rclone"; exit 1 }

"" | Out-File -Append -Encoding utf8 $out
Say "=== verification started ==="

# 1. wait for the writers. copy/sync/move only -- size and dedupe cannot collide.
$deadline = (Get-Date).AddMinutes($MaxWaitMinutes)
while ((Get-Date) -lt $deadline) {
    $w = @(Get-CimInstance Win32_Process -Filter "Name='rclone.exe'" -ErrorAction SilentlyContinue |
           Where-Object { $_.CommandLine -match 'expLB/crossgen' -and $_.CommandLine -match '\s(copy|sync|move)\s' })
    if ($w.Count -eq 0) { break }
    Say ("waiting on " + $w.Count + " writer(s): " + (($w | ForEach-Object { $_.ProcessId }) -join ', '))
    Start-Sleep -Seconds $PollSeconds
}
$w = @(Get-CimInstance Win32_Process -Filter "Name='rclone.exe'" -ErrorAction SilentlyContinue |
       Where-Object { $_.CommandLine -match 'expLB/crossgen' -and $_.CommandLine -match '\s(copy|sync|move)\s' })
if ($w.Count) { Say "GAVE UP waiting -- writers still running"; exit 2 }
Say "no writers left"

# 2. local truth
$f = Get-ChildItem (Join-Path $repo "run-crossgen") -Recurse -File -ErrorAction SilentlyContinue
Say ("LOCAL  run-crossgen: " + $f.Count + " files, " +
     [math]::Round((($f | Measure-Object Length -Sum).Sum)/1MB, 1) + " MB")

# 3. what Drive actually holds
Say "REMOTE tamara:expLB/crossgen:"
& $rc size "tamara:expLB/crossgen" 2>&1 | ForEach-Object { Say ("  " + $_) }

# 4. anything uploaded twice by the 2026-09-18 unlocked/orphaned writers. LIST ONLY.
Say "DUPLICATES (list only, nothing is deleted):"
$dup = & $rc dedupe --dedupe-mode list "tamara:expLB/crossgen" 2>&1
if ($dup) { $dup | ForEach-Object { Say ("  " + $_) } } else { Say "  none" }

Say "=== verification done ==="
