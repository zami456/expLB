<#
  Remove the duplicate objects the 2026-09-18 two-writer race created in
  tamara:expLB/crossgen, once every upload has finished.

  WHY NAME-BASED AND NOT --by-hash:
    The duplicates share a NAME -- two objects called banks/hf275_strict/csv/x.csv, and
    even two directories called csv, because both rclone processes created them at the
    same moment. Default dedupe groups by name within a directory and, with -dedupe-mode
    newest, keeps one and deletes the rest; it also merges the split directories.
    --by-hash would instead group by CONTENT across different names, and these result
    tables collide byte-for-byte quite naturally (the same numbers written under two
    filenames), so it could delete a file that is not a duplicate at all. Do not use it
    here.

  Nothing unique is lost: every duplicate is a second copy of a file that still exists in
  run-crossgen on this machine, which stays authoritative.

  Ordering: waits for the writers, THEN takes the same lock sync_crossgen.ps1 uses, so a
  per-bank sync from step A cannot write into the tree while it is being deduped.
#>
param(
    [int]$MaxWaitMinutes = 360,
    [int]$PollSeconds = 60,
    [switch]$DryRun
)
$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$logDir = Join-Path $repo "logs"
$out = Join-Path $logDir "crossgen_dedupe.log"
$lockPath = Join-Path $logDir "crossgen_rclone.lock"
function Say($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $out }

$rc = Join-Path $repo "tools\rclone.exe"
if (-not (Test-Path $rc)) { $rc = "C:\Users\user4\tools\rclone.exe" }
if (-not (Test-Path $rc)) { Say "ABORT no rclone"; exit 1 }

# Every caller wraps this in @(). A function returning a one-element array unrolls it to a
# scalar, and a zero-element one returns $null -- and $null.Count is $null, not 0, so the
# wait loop below would never see "no writers left" and would sit here until its deadline.
function Writers {
    @(Get-CimInstance Win32_Process -Filter "Name='rclone.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -match 'expLB/crossgen' -and $_.CommandLine -match '\s(copy|sync|move)\s' })
}

Say "=== dedupe run started (DryRun=$DryRun) ==="
$deadline = (Get-Date).AddMinutes($MaxWaitMinutes)
while ((Get-Date) -lt $deadline) {
    $w = @(Writers)
    if ($w.Count -eq 0) { break }
    Say ("waiting on " + $w.Count + " writer(s): " + (($w | ForEach-Object { $_.ProcessId }) -join ', '))
    Start-Sleep -Seconds $PollSeconds
}
if (@(Writers).Count) { Say "ABORT writers still running at deadline"; exit 2 }
Say "no writers left"

# Hold the lock so step A's per-bank syncs stand down while the tree is being rewritten.
$fs = $null
$lockDeadline = (Get-Date).AddMinutes(60)
while (-not $fs -and (Get-Date) -lt $lockDeadline) {
    try {
        $fs = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    } catch { Start-Sleep -Seconds 20 }
}
if (-not $fs) { Say "ABORT could not take the lock in 60m"; exit 3 }

try {
    if (@(Writers).Count) { Say "ABORT a writer appeared after the lock"; exit 4 }

    $before = & $rc size "tamara:expLB/crossgen" 2>&1 | Out-String
    Say ("BEFORE remote: " + ($before -replace "\s+", " ").Trim())
    $lf = Get-ChildItem (Join-Path $repo "run-crossgen") -Recurse -File -ErrorAction SilentlyContinue
    Say ("LOCAL: " + $lf.Count + " files")

    if ($DryRun) {
        Say "DRY RUN -- listing only"
        & $rc dedupe --dedupe-mode list "tamara:expLB/crossgen" 2>&1 |
            ForEach-Object { Say ("  " + $_) }
    } else {
        Say "dedupe --dedupe-mode newest (name-based; merges duplicate directories)"
        & $rc dedupe --dedupe-mode newest "tamara:expLB/crossgen" `
            --log-file (Join-Path $logDir "crossgen_dedupe_rclone.log") --log-level INFO
        Say ("dedupe exit " + $LASTEXITCODE)

        $after = & $rc size "tamara:expLB/crossgen" 2>&1 | Out-String
        Say ("AFTER remote: " + ($after -replace "\s+", " ").Trim())

        Say "re-listing to confirm nothing is left duplicated:"
        $left = & $rc dedupe --dedupe-mode list "tamara:expLB/crossgen" 2>&1 |
                Where-Object { $_ -match 'duplicate' }
        if ($left) { $left | Select-Object -First 20 | ForEach-Object { Say ("  " + $_) } }
        else { Say "  none -- clean" }
    }
    Say "=== dedupe run done ==="
} finally {
    $fs.Dispose()
}
