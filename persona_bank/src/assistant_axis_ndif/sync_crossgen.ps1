<#
.SYNOPSIS
  Single-writer push of run-crossgen, or one subtree of it, to Drive.

.DESCRIPTION
  Why this is its own script rather than a line inside run_crossgen_full.ps1:

    * That script called rclone exactly ONCE, at the very end. A step covering three
      banks therefore uploaded nothing for hours, and a crash before the end uploaded
      nothing at all -- on 2026-09-18 the finished hf275_loose bank sat on the lab PC
      only, invisible from the laptop. Calling this after every bank closes the window.
    * Two rclone processes copying the same tree into the same Drive folder can each
      decide a file is missing and upload it, and Drive keeps BOTH copies under the one
      name. So every writer takes an exclusive lock first. The lock is an open file
      HANDLE, not a flag file: if the holder dies, Windows drops the handle and the next
      writer walks in, so there is no stale lock to clear by hand.

  Drive throttles per object, not per byte: the 2026-09-18 whole-tree sync moved 14,597
  files in 2 h 30 m (~97 files/min) for 165 MB. Pass -SubPath whenever you can.

.PARAMETER SubPath
  Path relative to run-crossgen, e.g. "banks\hf275_loose" or "passA\banks\own100".
  Empty means the whole tree.

.PARAMETER OnBusy
  Skip returns immediately when another sync holds the lock; Wait blocks for up to
  -WaitMinutes. Per-bank calls Skip, so a sync never delays the NDIF work -- the next
  bank, or the final whole-tree sync, carries those files instead. The final sync waits.

.PARAMETER DupCheck
  After the copy, list (never delete) any Drive objects that share a name in the synced
  subtree. Use it after a sync that may have raced another writer.

.EXAMPLE
  .\sync_crossgen.ps1 -SubPath "banks\hf275_loose" -Tag hf275_loose -OnBusy Skip
  .\sync_crossgen.ps1 -Tag manual -DupCheck
#>
param(
    [string]$SubPath = "",
    [string]$Tag = "manual",
    [ValidateSet("Wait", "Skip")][string]$OnBusy = "Wait",
    [int]$WaitMinutes = 240,
    [switch]$DupCheck
)
$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$logDir = Join-Path $repo "logs"
$status = Join-Path $logDir "crossgen_sync.status"
$lockPath = Join-Path $logDir "crossgen_rclone.lock"
function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

$rc = Join-Path $repo "tools\rclone.exe"
if (-not (Test-Path $rc)) { $rc = "~\tools\rclone.exe" }
if (-not (Test-Path $rc)) { Note "$Tag ABORT (no rclone)"; exit 1 }

$sub = $SubPath.Trim().Trim([char[]]@('\', '/'))
$src = Join-Path $repo "run-crossgen"
$dst = "tamara:expLB/crossgen"
if ($sub) {
    $src = Join-Path $src $sub
    $dst = $dst + "/" + ($sub -replace '\\', '/')
}
if (-not (Test-Path $src)) { Note "$Tag SKIP (no $src)"; exit 0 }

# An exclusive handle the OS releases if this process dies -- see the note above.
$deadline = (Get-Date).AddMinutes($WaitMinutes)
$fs = $null
while (-not $fs) {
    try {
        $fs = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    } catch {
        if ($OnBusy -eq "Skip") { Note "$Tag SKIP (another sync holds the lock)"; exit 0 }
        if ((Get-Date) -ge $deadline) { Note "$Tag SKIP (lock busy ${WaitMinutes}m)"; exit 0 }
        Start-Sleep -Seconds 20
    }
}
try {
    # Holding the lock is not by itself proof that we are the only writer. Stopping this
    # script's Scheduled Task kills the PowerShell but ORPHANS its rclone child, which then
    # keeps uploading while holding nothing -- observed 2026-09-18. A pre-0918 copy of
    # run_crossgen_full.ps1 still in flight calls rclone directly for the same reason. So
    # look for a second writer on the wire, not just for the lock.
    # Only WRITERS count. rclone size/check/dedupe name the same remote but cannot collide,
    # and the verification task runs those -- matching them would skip syncs for no reason.
    $other = @(Get-CimInstance Win32_Process -Filter "Name='rclone.exe'" -ErrorAction SilentlyContinue |
               Where-Object { $_.CommandLine -match 'expLB/crossgen' -and
                              $_.CommandLine -match '\s(copy|sync|move)\s' -and
                              $_.ProcessId -ne $PID })
    if ($other.Count) {
        Note ("$Tag SKIP (rclone already writing crossgen: pid " + (($other | ForEach-Object { $_.ProcessId }) -join ', ') + ")")
        exit 0
    }
    Note "$Tag START $(if ($sub) { $sub } else { '<whole tree>' })"
    & $rc copy $src $dst --checksum --transfers 4 --checkers 8 `
        --log-file (Join-Path $logDir "crossgen_full_rclone.log") --log-level INFO
    $code = $LASTEXITCODE
    Note "$Tag EXIT $code"
    if ($DupCheck) {
        # list only: duplicates are reported, never removed. Experiment files are not
        # deleted without the owner saying so.
        $dupLog = Join-Path $logDir "crossgen_dupes.log"
        "=== $Tag $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss')) $dst" | Out-File -Append -Encoding utf8 $dupLog
        & $rc dedupe --dedupe-mode list $dst 2>&1 | Out-File -Append -Encoding utf8 $dupLog
        Note "$Tag DUPCHECK exit $LASTEXITCODE (see logs\crossgen_dupes.log)"
    }
    exit $code
} finally {
    $fs.Dispose()
}
