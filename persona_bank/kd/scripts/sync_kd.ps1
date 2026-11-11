<#
  Push kd/ to Drive (tamara:expLB/kd).

  Why this exists: nothing was syncing kd/ at all. night_chain covers run-coords and the
  sycophancy runs, sync_crossgen covers crossgen, and kd/ fell between them -- so the WMDP
  knowledge scores existed in exactly one place on one machine that reboots unprompted.

  Takes an exclusive file-handle lock, the same discipline as sync_crossgen.ps1: two rclone
  processes writing the same Drive path leave BOTH copies under one name, because Drive
  permits duplicate names. A dead holder's handle is released by the OS, so there is no
  stale lock to clear.

      -OnBusy Skip   return immediately if another sync holds the lock (for periodic runs)
      -OnBusy Wait   block until it is free (for a final, must-land sync)
#>
param(
    [ValidateSet("Skip", "Wait")][string]$OnBusy = "Skip",
    [string]$Tag = "kd",
    [int]$WaitSeconds = 1800
)
$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo

$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir "kd_sync.status"
$log = Join-Path $logDir "kd_rclone.log"
$lockPath = Join-Path $logDir "kd_rclone.lock"
function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

$rc = Join-Path $repo "tools\rclone.exe"
if (-not (Test-Path $rc)) { $rc = "~\tools\rclone.exe" }
if (-not (Test-Path $rc)) { Note "$Tag NO RCLONE"; exit 1 }

# Holding the lock does not prove sole ownership of the wire: Stop-ScheduledTask kills the
# PowerShell but orphans its rclone child, which keeps uploading with the handle released.
# So also refuse when any other rclone is already writing this Drive path.
$other = @(Get-CimInstance Win32_Process -Filter "Name='rclone.exe'" -EA SilentlyContinue |
           Where-Object { $_.CommandLine -like '*expLB/kd*' -and $_.ProcessId -ne $PID })
if ($other.Count) { Note "$Tag SKIP (another rclone already writing expLB/kd)"; exit 0 }

$deadline = (Get-Date).AddSeconds($WaitSeconds)
$fs = $null
while ($true) {
    try { $fs = [System.IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None'); break }
    catch {
        if ($OnBusy -eq "Skip") { Note "$Tag SKIP (lock held)"; exit 0 }
        if ((Get-Date) -gt $deadline) { Note "$Tag GAVE UP waiting for lock"; exit 1 }
        Start-Sleep -Seconds 10
    }
}
try {
    Note "$Tag START"
    & $rc copy (Join-Path $repo "kd") "tamara:expLB/kd" --checksum --transfers 4 `
        --log-file $log --log-level INFO
    $code = $LASTEXITCODE
    $n = @(Get-ChildItem (Join-Path $repo "kd") -Recurse -File -EA SilentlyContinue).Count

    # The 11-dataset crossgen cannot live under kd/: 27_crossgen.py's guard() requires
    # "run-crossgen" as an exact path component. So it writes to run-crossgen\kd11 and is
    # pushed from here, because sync_crossgen.ps1 is scoped to the 10-dataset results and
    # nothing else would carry it.
    # run-kd holds Phase 3: rollouts, judge scores, persona vectors, axes. Like kd11 it
    # cannot live under kd/ (1_generate_ndif.py and friends take an --output_dir and the
    # chain writes there), so it is pushed from here or nothing carries it.
    $rk = Join-Path $repo "run-kd"
    if (Test-Path $rk) {
        & $rc copy $rk "tamara:expLB/kd/run-kd" --checksum --transfers 4 `
            --log-file $log --log-level INFO
        $nk = @(Get-ChildItem $rk -Recurse -File -EA SilentlyContinue).Count
        Note "$Tag run-kd exit $LASTEXITCODE ($nk files)"
    }

    $k11 = Join-Path $repo "run-crossgen\kd11"
    if (Test-Path $k11) {
        & $rc copy $k11 "tamara:expLB/kd/crossgen11" --checksum --transfers 4 `
            --log-file $log --log-level INFO
        $n2 = @(Get-ChildItem $k11 -Recurse -File -EA SilentlyContinue).Count
        Note "$Tag crossgen11 exit $LASTEXITCODE ($n2 files)"
    }
    Note "$Tag DONE exit $code ($n local files)"
} finally {
    if ($fs) { $fs.Close(); $fs.Dispose() }
}
