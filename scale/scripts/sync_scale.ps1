<#
  Push run-scale/ to Drive (tamara:expLB/scale). Same lock discipline as sync_kd.ps1: an
  exclusive file handle PLUS a check for an orphaned rclone already writing the same path,
  because Stop-ScheduledTask kills the PowerShell but leaves its rclone child uploading.
#>
param([ValidateSet("Skip","Wait")][string]$OnBusy="Skip", [string]$Tag="scale", [int]$WaitSeconds=1800)
$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$logDir = Join-Path $repo "logs"; New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir "scale_sync.status"
$log = Join-Path $logDir "scale_rclone.log"
$lockPath = Join-Path $logDir "scale_rclone.lock"
function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }
$rc = Join-Path $repo "tools\rclone.exe"
if (-not (Test-Path $rc)) { $rc = "C:\Users\user4\tools\rclone.exe" }
if (-not (Test-Path $rc)) { Note "$Tag NO RCLONE"; exit 1 }
$other = @(Get-CimInstance Win32_Process -Filter "Name='rclone.exe'" -EA SilentlyContinue |
           Where-Object { $_.CommandLine -like '*expLB/scale*' -and $_.ProcessId -ne $PID })
if ($other.Count) { Note "$Tag SKIP (another rclone already writing expLB/scale)"; exit 0 }
$deadline = (Get-Date).AddSeconds($WaitSeconds); $fs = $null
while ($true) {
    try { $fs = [System.IO.File]::Open($lockPath,'OpenOrCreate','ReadWrite','None'); break }
    catch {
        if ($OnBusy -eq "Skip") { Note "$Tag SKIP (lock held)"; exit 0 }
        if ((Get-Date) -gt $deadline) { Note "$Tag GAVE UP waiting for lock"; exit 1 }
        Start-Sleep -Seconds 10
    }
}
try {
    Note "$Tag START"
    $src = Join-Path $repo "run-scale"
    if (Test-Path $src) {
        & $rc copy $src "tamara:expLB/scale" --checksum --transfers 4 --log-file $log --log-level INFO
        $n = @(Get-ChildItem $src -Recurse -File -EA SilentlyContinue).Count
        Note "$Tag DONE exit $LASTEXITCODE ($n files)"
    } else { Note "$Tag nothing to sync yet" }
} finally { if ($fs) { $fs.Close(); $fs.Dispose() } }
